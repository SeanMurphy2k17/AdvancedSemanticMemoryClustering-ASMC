"""
LTM Migration Utility
Backfills contentWords and linkedMemories for any LTM entries missing them.
Also upgrades word_cache from 7d (56 bytes) to 13d (104 bytes) format.
Run once against any existing database to bring it up to date.
"""
import json
import os
import struct
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "V2"))

from nltk.corpus import wordnet as wn
from longTermMemory import longTermMemory
from spatialValenceCompute import spatialValenceCompute, DECAY

LINK_K  = 3    # spatial neighbors to link per memory
BATCH   = 2000 # word cache entries per write transaction


def rebuild_word_cache(svc):
    PACK_OLD = struct.calcsize("7d")
    PACK_NEW = struct.calcsize("13d")

    old_keys = []
    with svc._lmdb.begin() as txn:
        for key, val in txn.cursor():
            if len(val) == PACK_OLD:
                old_keys.append(bytes(key))

    total = len(old_keys)
    print(f"Word cache: {total:,} old entries to upgrade (7d → 13d)...")
    if total == 0:
        print("  Already up to date.")
        return

    updated = errors = 0
    for batch_start in range(0, total, BATCH):
        batch = old_keys[batch_start:batch_start + BATCH]
        with svc._lmdb.begin(write=True) as txn:
            for db_key in batch:
                try:
                    raw = txn.get(db_key)
                    if not raw or len(raw) != PACK_OLD:
                        continue
                    wx, wy, wz, wa, wb, wc, ww = struct.unpack("7d", raw)
                    key_str = db_key.decode()
                    lemma, wn_pos = key_str.split('\x00', 1)
                    synsets = wn.synsets(lemma, pos=wn_pos) or wn.synsets(lemma, pos='n')
                    if not synsets:
                        continue
                    chain = svc._hypernym_chain(synsets[0])
                    wj = wk = wl = wm = wn_ = wo = 0.0
                    for ancestor, hop in chain:
                        w   = DECAY ** hop
                        wj += svc._living_d(ancestor)  * w
                        wk += svc._spatial_e(ancestor) * w
                        wl += svc._comms_f(ancestor)   * w
                        wm += svc._agent_g(ancestor)   * w
                        wn_ += svc._cognit_h(ancestor) * w
                        wo += svc._relate_i(ancestor)  * w
                    txn.put(db_key, struct.pack("13d", wx, wy, wz, wa, wb, wc,
                                                       wj, wk, wl, wm, wn_, wo, ww))
                    updated += 1
                except Exception:
                    errors += 1
        print(f"  {min(batch_start + BATCH, total):,} / {total:,}")

    print(f"Done. upgraded={updated:,}  errors={errors:,}")


def run():
    ltm = longTermMemory()
    svc = spatialValenceCompute()

    rebuild_word_cache(svc)

    print("\nReading LTM database...")
    all_entries = {}
    with ltm._env.begin() as txn:
        for key, value in txn.cursor():
            ltm_id = int(key.decode())
            all_entries[ltm_id] = json.loads(value)

    total        = len(all_entries)
    words_added  = 0
    links_added  = 0
    world_added  = 0

    print(f"Processing {total} LTM entries...")
    for ltm_id, mem in all_entries.items():
        changed = False

        if not mem.get("contentWords"):
            mem["contentWords"] = svc.extractContentWords(mem.get("inputText", ""))
            words_added += 1
            changed = True

        if not mem.get("worldPos"):
            mem["worldPos"] = svc.computeWorldValence(mem.get("inputText", ""))
            world_added += 1
            changed = True

        if not mem.get("linkedMemories") and ltm._index.ntotal > 1:
            pos = mem.get("responsePos")
            if pos:
                k   = min(LINK_K + 1, ltm._index.ntotal)
                _, I = ltm._index.search(ltm._encode(pos), k)
                links = []
                for nid in I[0]:
                    if nid == -1 or nid == ltm_id:
                        continue
                    neighbour = all_entries.get(nid)
                    if neighbour and neighbour.get("responsePos"):
                        links.append(neighbour["responsePos"])
                if links:
                    mem["linkedMemories"] = links
                    links_added += 1
                    changed = True

        if changed:
            ltm.updatePayload(ltm_id, mem)

    print(f"Done.")
    print(f"  contentWords added  : {words_added} / {total}")
    print(f"  worldPos added      : {world_added} / {total}")
    print(f"  linkedMemories added: {links_added} / {total}")


if __name__ == "__main__":
    run()
