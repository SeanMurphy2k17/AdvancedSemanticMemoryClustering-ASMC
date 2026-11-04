#!/usr/bin/env python3
"""Measure how well a store can find its own memories.

For every long-term memory, build the query "what did we say about" plus the
memory's three rarest content words, run it through the real retrieval
pipeline (get_context), and check whether that memory comes back in the top k.
Two scores are reported:

  id-level    the exact record is in the top k
  same-text   a record with the same prompt text is in the top k (the fair
              score for a store where one message was stored several times)

Run it on a COPY of a store, never one another process has open: queries
stitch links, which writes. The library used is the one this file sits in,
or the one ASMC_LIB points at.

usage: bench_self_query.py --store DIR [--k 6] [--limit N] [--out results.json]
"""
import argparse, collections, contextlib, io, itertools, json, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--store", required=True, help="a copy of the store to measure")
ap.add_argument("--k", type=int, default=6)
ap.add_argument("--limit", type=int, default=0, help="measure at most N memories (0 = all)")
ap.add_argument("--out", default="", help="write the full report here as JSON")
a = ap.parse_args()
os.environ["ASMC_STORE_DIR"] = os.path.abspath(a.store)
LIB = os.environ.get("ASMC_LIB") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LIB); sys.path.insert(0, os.path.join(LIB, "V2"))
from ASMC_API import create_memory


def quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


K = a.k
mem = quiet(create_memory); mm = mem._mm; ltm = mm._ltm
ltm.wait_for_index(); ltm.wait_for_cw_index()
mems = {}
with ltm._env.begin() as txn:
    for k, v in txn.cursor():
        if not k.isdigit():
            continue
        try:
            m = json.loads(v)
        except Exception:
            continue
        if isinstance(m, dict) and "inputText" in m and (m.get("metaDataTag") or {}).get("type") != "scm_anchor":
            mems[int(k)] = m
df = collections.Counter()
for m in mems.values():
    for w in set(m.get("contentWords") or []):
        df[w] += 1

rows = []
for mid, m in sorted(mems.items()):
    if a.limit and len(rows) >= a.limit:
        break
    words = sorted({w for w in (m.get("contentWords") or []) if len(w) > 3}, key=lambda w: (df[w], w))[:3]
    if len(words) < 2:
        continue
    q = "what did we say about " + " ".join(words)
    t0 = time.time()
    ctx = quiet(mem.get_context, q, layer1_count=1, layer2_count=K)
    ids = [(e.get("metaDataTag") or {}).get("ltm_id") for e in ctx["layer2"]]
    rows.append({"id": mid, "query": q, "hit": mid in ids,
                 "hit_text": any(i in mems and mems[i]["inputText"] == m["inputText"] for i in ids),
                 "returned": ids, "secs": round(time.time() - t0, 2)})
n = len(rows)
report = {"store": a.store, "library": LIB, "k": K, "ltm_memories": len(mems), "queries": n,
          "recall_at_k": {"id_level": round(sum(r["hit"] for r in rows) / n, 3) if n else None,
                          "same_text": round(sum(r["hit_text"] for r in rows) / n, 3) if n else None},
          "mean_secs": round(sum(r["secs"] for r in rows) / n, 2) if n else None, "rows": rows}
if a.out:
    json.dump(report, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
