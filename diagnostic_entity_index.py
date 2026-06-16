"""
Diagnostic: Entity index vs query entity extraction
Checks if entities extracted from queries match what's stored in the entity index.
"""
import os
import sys
import json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "V2"))

from ASMC_API import create_memory

QUERIES = [
    "Arthur and Guinevere in the court of the king",
    "Tennyson published Idylls of the King in 1859",
    "Alfred Lord Tennyson poet laureate of England",
    "The Lady of Shalott by Alfred Tennyson",
    "the sea and the waves crashing on shore",
]

def main():
    api = create_memory(verbose=False)
    ltm = api._mm._ltm
    svc = api._mm._spatial._svc

    print("=" * 70)
    print("ENTITY INDEX DIAGNOSTIC")
    print("=" * 70)

    # 1. Count entity keys in the index
    entity_key_count = 0
    all_entity_keys = []
    with ltm._env.begin() as txn:
        cursor = txn.cursor()
        for key, raw in cursor:
            if b"\x00" in key:
                entity_key_count += 1
                all_entity_keys.append((key, json.loads(raw)))

    print(f"\n[1] Entity index stats:")
    print(f"    Total entity keys: {entity_key_count}")
    print(f"    Index ready: {ltm._index_ready}")

    # 2. Show top 30 entity keys by number of linked memories
    entity_keys_by_popularity = sorted(all_entity_keys, key=lambda x: len(x[1]), reverse=True)
    print(f"\n[2] Top 30 most-linked entity keys (type\x00text -> [mem_ids]):")
    for key, mem_ids in entity_keys_by_popularity[:30]:
        key_str = key.decode('utf-8', errors='replace')
        print(f"    '{key_str}' -> {mem_ids[:10]}{'...' if len(mem_ids) > 10 else ''}")

    # 3. Extract entities from queries and check if they exist in the index
    print(f"\n[3] Query entity extraction vs index lookup:")
    for query in QUERIES:
        tags = svc.extractFactualTags(query)
        all_query_tags = (tags.get("entities", []) +
                          tags.get("dates", []) +
                          tags.get("quantities", []) +
                          tags.get("technical_terms", []))
        
        print(f"\n    Query: '{query}'")
        print(f"    Extracted tags: {all_query_tags}")
        
        found_keys = []
        missing_keys = []
        for etype, etext in all_query_tags:
            lookup_key = ltm._entity_key(etype, etext)
            found_key = ltm._entity_key(etype, etext)  # lookup in txn below
            with ltm._env.begin() as txn:
                raw = txn.get(lookup_key)
            if raw:
                mem_ids = json.loads(raw)
                found_keys.append((etype, etext, mem_ids))
            else:
                missing_keys.append((etype, etext))
        
        if found_keys:
            print(f"    ✅ FOUND in index:")
            for etype, etext, mem_ids in found_keys:
                print(f"       ({etype}, {etext}) -> {mem_ids[:5]}")
        if missing_keys:
            print(f"    ❌ MISSING from index:")
            for etype, etext in missing_keys:
                print(f"       ({etype}, {etext})")

    # 4. Check case sensitivity: do uppercase versions exist?
    print(f"\n[4] Case sensitivity check:")
    print(f"    Checking if uppercase versions of missing entities exist...")
    for query in QUERIES[:3]:
        tags = svc.extractFactualTags(query)
        all_query_tags = (tags.get("entities", []) +
                          tags.get("dates", []) +
                          tags.get("quantities", []) +
                          tags.get("technical_terms", []))
        for etype, etext in all_query_tags:
            # Check uppercase version
            lookup_key_lower = ltm._entity_key(etype, etext)
            lookup_key_upper = ltm._entity_key(etype, etext.capitalize())
            with ltm._env.begin() as txn:
                raw_lower = txn.get(lookup_key_lower)
                raw_upper = txn.get(lookup_key_upper)
            if not raw_lower and raw_upper:
                print(f"       ({etype}, '{etext}') -> NOT FOUND")
                print(f"       ({etype}, '{etext.capitalize()}') -> FOUND: {json.loads(raw_upper)[:5]}")
            elif raw_lower:
                pass  # already found, no issue
            else:
                print(f"       ({etype}, '{etext}') -> NOT FOUND (uppercase also missing)")

    # 5. Check what entity types exist in the index
    print(f"\n[5] Entity types in index:")
    type_counts = {}
    for key, _ in all_entity_keys:
        etype = key.split(b"\x00")[0].decode('utf-8')
        type_counts[etype] = type_counts.get(etype, 0) + 1
    for etype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {etype}: {count} keys")

    # 6. Sample stored memories - check their factualTags
    print(f"\n[6] Sample stored memories (first 5):")
    count = 0
    with ltm._env.begin() as txn:
        cursor = txn.cursor()
        for key, raw in cursor:
            if b"\x00" in key or key.startswith(b"__"):
                continue
            mem = json.loads(raw)
            meta = mem.get("metaDataTag", {})
            if meta.get("type") == "scm_anchor":
                continue
            count += 1
            factual = mem.get("factualTags", {})
            ents = factual.get("entities", [])
            tech = factual.get("technical_terms", [])
            print(f"    [{count}] id={meta.get('ltm_id')} entities={ents[:3]} technical={tech[:3]}")
            if count >= 5:
                break

    print(f"\n{'='*70}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
