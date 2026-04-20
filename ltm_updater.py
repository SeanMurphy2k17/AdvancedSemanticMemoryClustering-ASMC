"""
LTM Migration Utility
Backfills contentWords and linkedMemories for any LTM entries missing them.
Run once against any existing database to bring it up to date.
"""
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "V2"))

from longTermMemory import longTermMemory
from spatialValenceCompute import spatialValenceCompute

LINK_K = 3  # spatial neighbors to link per memory


def run():
    ltm = longTermMemory()
    svc = spatialValenceCompute()

    print("Reading LTM database...")
    all_entries = {}
    with ltm._env.begin() as txn:
        for key, value in txn.cursor():
            ltm_id = int(key.decode())
            all_entries[ltm_id] = json.loads(value)

    total        = len(all_entries)
    words_added  = 0
    links_added  = 0

    print(f"Processing {total} entries...")
    for ltm_id, mem in all_entries.items():
        changed = False

        if not mem.get("contentWords"):
            mem["contentWords"] = svc.extractContentWords(mem.get("inputText", ""))
            words_added += 1
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
    print(f"  contentWords added : {words_added} / {total}")
    print(f"  linkedMemories added: {links_added} / {total}")


if __name__ == "__main__":
    run()
