"""
Fast Benchmark: Query-only test against existing LTM data.
Skips ingestion, just runs queries to verify all 3 retrieval channels.
"""
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "V2"))

from ASMC_API import create_memory

QUERIES = [
    # Generic queries (no named entities)
    "the sea and the waves crashing on shore",
    "death and grief and loss of a loved one",
    "love and longing for someone far away",
    "war and battle and knights in armor",
    "nature trees flowers and the forest",
    "god and faith and prayer to the divine",
    "darkness and shadow and the night sky",
    "the king ruled with wisdom and power",
    "a soldier returned home from the front",
    "the queen wore a golden crown",
    # Named-entity queries
    "Arthur and Guinevere in the court of the king",
    "Tennyson's Idylls of the King published in 1859",
    "Alfred Lord Tennyson poet laureate of England",
    "The Lady of Shalott by Alfred Tennyson",
    "Knights of the Round Table at Camelot",
    "Queen Mary and Henry VIII of England",
    "The Charge of the Light Brigade at Balaclava",
    "In Memoriam A.H.H. dedicated to Arthur Henry Hallam",
    "Tennyson wrote Maud and The Princess in the 1840s",
]


def main():
    print("=" * 70)
    print("ASMC QUERY BENCHMARK (query-only, no ingestion)")
    print("=" * 70)

    api = create_memory(verbose=False)
    stats = api.get_statistics()
    print(f"\nLTM memories: {stats['ltm_memories']}")
    print(f"STM entries: {stats['stm_entries']}")

    print(f"\n{'='*70}")
    print("RUNNING QUERIES")
    print(f"{'='*70}")

    results = []
    for i, query in enumerate(QUERIES, 1):
        t0 = time.perf_counter()
        ctx = api.get_context(query, layer2_count=5)
        elapsed = time.perf_counter() - t0

        result = {
            "query": query,
            "time": elapsed,
            "layer2": len(ctx.get("layer2", [])),
            "chain": len(ctx.get("layer2_chain", [])),
            "semantic": len(ctx.get("semantic_chain", [])),
        }
        results.append(result)

        # Determine query type
        has_entities = any(c.isupper() for c in query if c.isalpha())
        query_type = "named-entity" if has_entities else "generic"

        print(f"\n  [{i:2d}] ({query_type:14s}) '{query[:60]}'")
        print(f"       time: {elapsed:.3f}s")
        print(f"       layer2: {result['layer2']}  chain: {result['chain']}  semantic: {result['semantic']}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    times = [r["time"] for r in results]
    chains = [r["chain"] for r in results]
    layer2s = [r["layer2"] for r in results]
    sematics = [r["semantic"] for r in results]
    
    print(f"  Total queries:     {len(results)}")
    print(f"  Avg query time:    {sum(times)/len(times):.3f}s")
    print(f"  Avg chain length:  {sum(chains)/len(chains):.1f}")
    print(f"  Avg layer2:        {sum(layer2s)/len(layer2s):.1f}")
    print(f"  Avg semantic:      {sum(sematics)/len(sematics):.1f}")
    print(f"  Min time:          {min(times):.3f}s")
    print(f"  Max time:          {max(times):.3f}s")
    
    # Separate generic vs named-entity
    generic_results = [r for r in results if not any(c.isupper() for c in r["query"] if c.isalpha())]
    entity_results = [r for r in results if any(c.isupper() for c in r["query"] if c.isalpha())]
    
    if generic_results:
        g_times = [r["time"] for r in generic_results]
        g_chains = [r["chain"] for r in generic_results]
        print(f"\n  Generic queries ({len(generic_results)}):")
        print(f"    Avg time: {sum(g_times)/len(g_times):.3f}s")
        print(f"    Avg chain: {sum(g_chains)/len(g_chains):.1f}")
    
    if entity_results:
        e_times = [r["time"] for r in entity_results]
        e_chains = [r["chain"] for r in entity_results]
        print(f"\n  Named-entity queries ({len(entity_results)}):")
        print(f"    Avg time: {sum(e_times)/len(e_times):.3f}s")
        print(f"    Avg chain: {sum(e_chains)/len(e_chains):.1f}")

    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
