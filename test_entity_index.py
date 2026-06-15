#!/usr/bin/env python3
"""Test the entity reverse index."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "V2"))

from longTermMemory import longTermMemory

ltm = longTermMemory()

# Step 1: Build the index
print("\n" + "="*60)
print("STEP 1: Build entity reverse index")
print("="*60)
count = ltm.buildEntityIndex()
print(f"Total entities indexed: {count}")

# Step 2: Show index stats
print("\n" + "="*60)
print("STEP 2: Index statistics")
print("="*60)
total_keys = 0
total_refs = 0
with ltm._env.begin() as txn:
    cursor = txn.cursor()
    for key, val in cursor:
        if b"\x00" in key:
            total_keys += 1
            ids = json.loads(val)
            total_refs += len(ids)
print(f"  Unique entity keys:  {total_keys}")
print(f"  Total entity->id refs: {total_refs}")

# Step 3: Resolve entities from queries
print("\n" + "="*60)
print("STEP 3: Test resolveEntities()")
print("="*60)
test_queries = [
    "Arthur",
    "Guinevere",
    "Camelot",
    "the king sat on his throne",
    "i am hungry and want food",
    "three years ago",
]

for q in test_queries:
    results = ltm.resolveEntities(q)
    print(f"\n  Query: '{q}'")
    if not results:
        print("    (no matches)")
    for entity, mem_ids in sorted(results.items()):
        print(f"    {entity} → {len(mem_ids)} memories: {mem_ids[:5]}{'...' if len(mem_ids) > 5 else ''}")

# Step 4: Fetch a couple of found memories to verify
print("\n" + "="*60)
print("STEP 4: Sample memory fetches")
print("="*60)
results = ltm.resolveEntities("Arthur")
for entity, mem_ids in results.items():
    if mem_ids:
        first_id = mem_ids[0]
        mem = ltm.fetchById(first_id)
        if mem:
            print(f"\n  First match for '{entity}':")
            print(f"    ltm_id: {mem.get('metaDataTag', {}).get('ltm_id')}")
            print(f"    input:  {mem.get('inputText', '')[:80]}")
            print(f"    response: {mem.get('responseText', '')[:80]}")
            tags = mem.get("factualTags", {})
            if tags.get("entities"):
                print(f"    entities: {tags['entities'][:5]}")
        break

print("\n" + "="*60)
print("DONE")
print("="*60)
