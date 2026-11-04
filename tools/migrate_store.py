#!/usr/bin/env python3
"""Bring an existing ASMC store up to this version of the library, and optionally repair it.

Always:        responseWords computed for entries that lack them, and the
               long-term content-word index extended with them.
--reset-links  clear every long-term link list and re-stitch each memory once
               under the nearest-neighbour rule (ASMC_STITCH_NEIGHBOURS).
--drain        promote window entries beyond the maximum, skipping any that
               long-term memory already holds (what an unclean death leaves).
--dedupe       remove surplus copies of exactly duplicated long-term records
               (same text and creation time), keeping the lowest id; cleans the
               FAISS index and the inverted indexes too.
--drop-noise   remove entries whose prompt is harness text, not conversation:
               compaction requests and summaries, system notifications, tool
               notices, context blocks, task-complete nags. Harness decoration on
               an otherwise real prompt (datetime tag, tagged-files block) is
               stripped, not dropped.
--dry-run      report what would change and change nothing.

Run with the owning process STOPPED (one writer). Safe to run twice.
usage: python tools/migrate_store.py --store DIR [--reset-links] [--drain] [--dedupe] [--drop-noise] [--dry-run]
"""
import argparse, collections, contextlib, io, json, os, re, sys

NOISE_PREFIXES = (
    # compaction, in both harnesses
    "please create a detailed summary of the conversation so far",
    "some of the conversation history has been summarized",
    "this session is being continued from a previous conversation",
    # the Copilot harness talking to the agent, not the person
    "<system_notification>",
    "<tools_changed_notice>",
    "<context>",
    "you have not yet marked the task as complete",
    "<reminderinstructions>",
)
DATETIME_TAG = re.compile(r"^\s*<current_datetime>.*?</current_datetime>\s*", re.S)
# blocks appended to a real prompt by the harness: strip, keep the prompt
TRAILING_BLOCKS = re.compile(r"\s*<(tagged_files|context|tools_changed_notice|reminderInstructions|editorContext)>.*?</\1>\s*", re.S)
UNWRAP = re.compile(r"<userRequest>(.*?)</userRequest>", re.S)  # the person's words, wrapped: unwrap, keep

ap = argparse.ArgumentParser()
ap.add_argument("--store", required=True)
ap.add_argument("--reset-links", action="store_true")
ap.add_argument("--drain", action="store_true")
ap.add_argument("--dedupe", action="store_true")
ap.add_argument("--drop-noise", action="store_true")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
os.environ["ASMC_STORE_DIR"] = a.store
P = os.environ.get("ASMC_LIB") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # this repository, or ASMC_LIB
sys.path.insert(0, P); sys.path.insert(0, P + "/V2")
import numpy as np
import faiss
from ASMC_API import create_memory

def quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)

def dump(m):
    return json.dumps(m, default=lambda o: list(o) if isinstance(o, tuple) else o).encode()

def is_memory(m):
    return isinstance(m, dict) and "inputText" in m and (m.get("metaDataTag") or {}).get("type") != "scm_anchor"

def clean_prompt(text):
    """The person's words, without the harness decoration around them."""
    t = DATETIME_TAG.sub("", text or "", count=1)
    t = UNWRAP.sub(r"\1", t)
    return TRAILING_BLOCKS.sub(" ", t).strip()

def is_noise(text):
    t = clean_prompt(text).lower()
    return not t or any(t.startswith(p) for p in NOISE_PREFIXES)

mem = quiet(create_memory)
mm = mem._mm; ltm = mm._ltm; stm = mm._stm; svc = mm._spatial._svc
ltm.wait_for_cw_index()

def remove_ltm_ids(ids):
    """Delete long-term records and every reference to them: LMDB, inverted indexes, FAISS."""
    gone = set(ids)
    if not gone:
        return
    with ltm._env.begin(write=True) as txn:
        for mid in gone:
            txn.delete(str(mid).encode())
        for k, v in list(txn.cursor()):
            if b"\x00" not in k:
                continue
            try:
                lst = json.loads(v)
            except ValueError:
                continue
            if isinstance(lst, list) and any(i in gone for i in lst):
                kept = [i for i in lst if i not in gone]
                if kept:
                    txn.put(k, json.dumps(kept).encode())
                else:
                    txn.delete(k)
    ltm._index.remove_ids(np.array(sorted(gone), dtype=np.int64))
    faiss.write_index(ltm._index, ltm.faiss_path)

# ---- plan -------------------------------------------------------------------
records = {}
with ltm._env.begin() as txn:
    for k, v in txn.cursor():
        if k.isdigit():
            m = json.loads(v)
            if is_memory(m):
                records[int(k)] = m
groups = collections.defaultdict(list)
for mid, m in records.items():
    groups[(m["inputText"], m.get("timeDate") or "")].append(mid)
surplus = sorted(mid for ids in groups.values() if len(ids) > 1 for mid in sorted(ids)[1:])
need_words = [mid for mid, m in records.items() if "responseWords" not in m]
over = max(stm.count() - stm.STM_MAX, 0)
oldest = stm.get_recent(stm.STM_MAX + over)[:over] if over else []
drain_skip = sum((e["inputText"], e.get("timeDate") or "") in groups for e in oldest)
noise_ltm = sorted(mid for mid, m in records.items() if is_noise(m["inputText"]))
noise_stm = [i for i, e in enumerate(stm.get_recent(stm.STM_MAX + over)) if is_noise(e["inputText"])]
tagged_ltm = sorted(mid for mid, m in records.items() if clean_prompt(m["inputText"]) != (m["inputText"] or "").strip() and mid not in noise_ltm)
tagged_stm = [i for i, e in enumerate(stm.get_recent(stm.STM_MAX + over)) if clean_prompt(e["inputText"]) != (e["inputText"] or "").strip() and i not in noise_stm]
plan = {"store": a.store, "ltm_records": len(records), "ltm_distinct": len(groups),
        "responseWords_to_compute": len(need_words),
        "reset_links": {"links_to_clear": sum(len(m.get("linkedMemories") or []) for m in records.values())} if a.reset_links else "off",
        "dedupe": {"surplus_copies": len(surplus), "ids": surplus[:40]} if a.dedupe else "off",
        "drain": {"window": stm.count(), "max": stm.STM_MAX, "over": over, "already_in_ltm": drain_skip} if a.drain else "off",
        "drop_noise": {"ltm_to_drop": len(noise_ltm), "window_to_drop": len(noise_stm),
                       "decorations_to_strip": len(tagged_ltm) + len(tagged_stm),
                       "samples": [records[i]["inputText"][:60].replace("\n", " ") for i in noise_ltm[:4]]} if a.drop_noise else "off"}
print(json.dumps({"plan": plan}, indent=1))
if a.dry_run:
    sys.exit(0)

# ---- 1. answer words ----------------------------------------------------------
words_added = 0
with ltm._env.begin(write=True) as txn:
    for mid in need_words:
        m = records[mid]
        m["responseWords"] = quiet(svc.extractContentWords, m.get("responseText") or "")
        new_words = [w for w in m["responseWords"] if w not in (m.get("contentWords") or [])]
        if new_words:
            ltm._index_content_words(mid, new_words, txn=txn); words_added += len(new_words)
        txn.put(str(mid).encode(), dump(m))

# ---- 2. dedupe, drop noise ----------------------------------------------------------
removed = dropped = stripped = 0
to_remove = set(surplus) if a.dedupe else set()
if a.drop_noise:
    to_remove |= set(noise_ltm)
    with ltm._env.begin(write=True) as txn:
        for mid in tagged_ltm:
            if mid in to_remove:
                continue
            m = records[mid]; m["inputText"] = clean_prompt(m["inputText"])
            txn.put(str(mid).encode(), dump(m)); stripped += 1
if to_remove:
    remove_ltm_ids(to_remove)
    removed = len(to_remove & set(surplus)); dropped = len(to_remove - set(surplus))
    for mid in to_remove:
        records.pop(mid, None)
    groups = collections.defaultdict(list)
    for mid, m in records.items():
        groups[(m["inputText"], m.get("timeDate") or "")].append(mid)

# ---- 3. reset + re-stitch -----------------------------------------------------
links_cleared = restitched = 0
if a.reset_links:
    with ltm._env.begin(write=True) as txn:
        for mid, m in records.items():
            if m.get("linkedMemories"):
                links_cleared += len(m["linkedMemories"]); m["linkedMemories"] = []
                txn.put(str(mid).encode(), dump(m))
    for m in records.values():
        m.setdefault("linkedMemories", [])
        quiet(ltm._stitch, [m]); restitched += 1

# ---- 4. the window: drop noise, strip tags, drain -------------------------------
window_dropped = promoted = skipped = 0
with stm._lock:
    if a.drop_noise:
        before = len(stm._entries)
        stm._entries = [e for e in stm._entries if not is_noise(e["inputText"])]
        window_dropped = before - len(stm._entries)
        for e in stm._entries:
            if clean_prompt(e["inputText"]) != (e["inputText"] or "").strip():
                e["inputText"] = clean_prompt(e["inputText"]); stripped += 1
if a.drain:
    while stm.count() > stm.STM_MAX:
        e = stm.pop_oldest()
        if e is None:
            break
        if (e["inputText"], e.get("timeDate") or "") in groups:
            skipped += 1
        else:
            quiet(ltm.addMemory, e); promoted += 1

# ---- 5. window answer words + checkpoint ----------------------------------------
stm_touched = 0
with stm._lock:
    for e in stm._entries:
        if "responseWords" not in e:
            e["responseWords"] = quiet(svc.extractContentWords, e.get("responseText") or ""); stm_touched += 1
    stm._checkpoint()

print(json.dumps({"done": {"answer_words_indexed": words_added, "duplicates_removed": removed,
                           "noise_dropped_ltm": dropped, "noise_dropped_window": window_dropped, "datetime_tags_stripped": stripped,
                           "links_cleared": links_cleared, "restitched": restitched,
                           "window_promoted": promoted, "window_skipped_already_in_ltm": skipped,
                           "window_entries_updated": stm_touched, "window_now": stm.count()}}, indent=1))
