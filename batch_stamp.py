"""
Batch-stamp existing LTM memories with ltm_id.
Run this once after the spider implementation to enable stitching on existing data.
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'V2'))

def batch_stamp():
    from longTermMemory import longTermMemory
    ltm = longTermMemory()

    stamped = 0
    missing = 0

    with ltm._env.begin(write=True) as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            mem_id = int(key)
            payload = json.loads(value)

            if "metaDataTag" not in payload:
                payload["metaDataTag"] = {}

            if "ltm_id" in payload["metaDataTag"]:
                missing += 1
                continue

            payload["metaDataTag"]["ltm_id"] = mem_id
            txn.put(key, json.dumps(payload, default=lambda o: list(o) if isinstance(o, tuple) else o).encode())
            stamped += 1

    print(f"Stamped {stamped} memories with ltm_id")
    print(f"Already had ltm_id: {missing}")
    print(f"Total processed: {stamped + missing}")

if __name__ == "__main__":
    batch_stamp()
