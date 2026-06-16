"""
Benchmark: Entity-tagged spider stitching on Tennyson poetry fragments.
Measures chain traversal improvement with factual entity overlap links.
"""
import os
import re
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "V2"))

import PyPDF2
from ASMC_API import create_memory

PDF_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "The_Poetry_of_Tennyson.pdf")
CHUNK_SIZE = 300

QUERIES = [
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


def extract_pdf_text(pdf_path):
    text = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text.append(page_text)
    raw = " ".join(text)
    return re.sub(r'\s+', ' ', raw).strip()


def ingest(api, pdf_path, chunk_size):
    text   = extract_pdf_text(pdf_path)
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    chunks = [c.strip() for c in chunks if len(c.strip()) > 40]
    print(f"Ingesting {len(chunks)} chunks from {os.path.basename(pdf_path)}...")
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue
        api.add_experience(situation=chunk, response=chunk)
        if (i + 1) % 100 == 0:
            stats = api.get_statistics()
            print(f"  chunk {i+1}/{len(chunks)}  STM={stats['stm_entries']}  LTM={stats['ltm_memories']}")
    stats = api.get_statistics()
    print(f"Done. STM={stats['stm_entries']}  LTM={stats['ltm_memories']}")


def count_entity_tags(api):
    """Count how many memories have factual tags populated."""
    import json
    # Use existing LTM instance from memoryManager (LMDB only allows one open per process)
    ltm = api._mm._ltm
    total = has_entities = has_dates = has_quantities = has_unknown = 0
    with ltm._env.begin() as txn:
        for key, raw in txn.cursor():
            # Skip entity index keys (contain \x00) and sentinel keys
            if b"\x00" in key or key.startswith(b"__"):
                continue
            entry = json.loads(raw)
            meta = entry.get("metaDataTag", {})
            if meta.get("type") == "scm_anchor":
                continue
            total += 1
            tags = entry.get("factualTags", {})
            if not tags:
                has_unknown += 1
            else:
                ents = tags.get("entities", [])
                if ents:
                    has_entities += 1
                dates = tags.get("dates", [])
                if dates:
                    has_dates += 1
                qty = tags.get("quantities", [])
                if qty:
                    has_quantities += 1
    return total, has_entities, has_dates, has_quantities, has_unknown


def run_query(api, query, query_num):
    """Run a single query and measure results."""
    t0 = time.perf_counter()
    ctx = api.get_context(query, layer2_count=5)
    elapsed = time.perf_counter() - t0

    print(f"\n{'='*70}")
    print(f"  QUERY {query_num}: '{query}'")
    print(f"  Time: {elapsed:.3f}s")
    print(f"{'='*70}")

    # Layer 2 (direct LTM)
    print(f"\n  layer2 ({len(ctx['layer2'])}):")
    for m in ctx["layer2"]:
        tags = m.get("factualTags", {})
        ent_str = ""
        if tags:
            ents = tags.get("entities", [])
            if ents:
                ent_str = f" [entities: {', '.join(e[1] for e in ents[:3])}]"
            elif any(v for v in tags.values()):
                ent_str = f" [tags populated]"
            else:
                ent_str = f" [tags: unknown]"
        print(f"    - {m['inputText'][:90]}{ent_str}")

    # Chain (linked traversal)
    print(f"\n  layer2_chain ({len(ctx['layer2_chain'])}):")
    for m in ctx["layer2_chain"]:
        tags = m.get("factualTags", {})
        ent_str = ""
        if tags:
            ents = tags.get("entities", [])
            if ents:
                ent_str = f" [entities: {', '.join(e[1] for e in ents[:3])}]"
            elif any(v for v in tags.values()):
                ent_str = f" [tags populated]"
            else:
                ent_str = f" [tags: unknown]"
        print(f"    ~ {m['inputText'][:90]}{ent_str}")

    # Semantic chain
    print(f"\n  semantic_chain ({len(ctx['semantic_chain'])}):")
    for m in ctx["semantic_chain"]:
        tags = m.get("factualTags", {})
        ent_str = ""
        if tags:
            ents = tags.get("entities", [])
            if ents:
                ent_str = f" [entities: {', '.join(e[1] for e in ents[:3])}]"
        print(f"    * {m['inputText'][:90]}{ent_str}")

    return {
        "query": query,
        "time": elapsed,
        "layer2": len(ctx["layer2"]),
        "chain": len(ctx["layer2_chain"]),
        "semantic": len(ctx["semantic_chain"]),
    }


def main():
    print("="*70)
    print("ASMC ENTITY-TAGGED SPIDER BENCHMARK")
    print("="*70)

    # Create fresh memory
    api = create_memory(verbose=True)

    # Wait for entity index rebuild to complete (if running)
    print(f"\nWaiting for entity index rebuild to complete...")
    ltm = api._mm._ltm
    ltm.wait_for_index(timeout=600)  # 10 min max

    # Ingest PDF
    ingest(api, PDF_PATH, CHUNK_SIZE)

    # Check entity tag coverage
    print(f"\n{'='*70}")
    print("ENTITY TAG COVERAGE")
    print(f"{'='*70}")
    total, has_ents, has_dates, has_qty, has_unknown = count_entity_tags(api)
    print(f"  Total memories:       {total}")
    print(f"  With entities:        {has_ents}")
    print(f"  With dates:           {has_dates}")
    print(f"  With quantities:      {has_qty}")
    print(f"  Tags unknown (empty): {has_unknown}")

    # Run queries
    print(f"\n{'='*70}")
    print("QUERY BENCHMARK")
    print(f"{'='*70}")
    results = []
    for i, q in enumerate(QUERIES, 1):
        r = run_query(api, q, i)
        results.append(r)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    avg_time = sum(r["time"] for r in results) / len(results)
    avg_chain = sum(r["chain"] for r in results) / len(results)
    print(f"  Avg query time:     {avg_time:.3f}s")
    print(f"  Avg chain reach:    {avg_chain:.1f} memories")
    for r in results:
        print(f"  '{r['query'][:50]:50s}'  chain={r['chain']:3d}  time={r['time']:.3f}s")


if __name__ == "__main__":
    main()
