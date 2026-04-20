import os
import re
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "V2"))

import PyPDF2
from ASMC_API import create_memory

PDF_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "The_Poetry_of_Tennyson.pdf")
CHUNK_SIZE = 300

QUERIES = [
    "the sea and the waves",
    "death and grief and loss",
    "love and longing",
    "war and battle and knights",
    "nature trees flowers",
    "god and faith and prayer",
    "darkness and shadow night",
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


def query_all(api):
    print(f"\n{'='*70}")
    print("QUERY TEST")
    print(f"{'='*70}")
    for q in QUERIES:
        ctx = api.get_context(q, layer2_count=5)
        print(f"\n  QUERY: '{q}'")
        print(f"  layer2 ({len(ctx['layer2'])}):")
        for m in ctx["layer2"]:
            print(f"    - {m['inputText'][:80]}")
        print(f"  layer2_chain ({len(ctx['layer2_chain'])}):")
        for m in ctx["layer2_chain"]:
            print(f"    ~ {m['inputText'][:80]}")
        print(f"  semantic_chain ({len(ctx['semantic_chain'])}):")
        for m in ctx["semantic_chain"]:
            print(f"    * {m['inputText'][:80]}")


if __name__ == "__main__":
    api = create_memory(verbose=True)
    api.clear_memory(confirm=True)
    ingest(api, PDF_PATH, CHUNK_SIZE)
    query_all(api)
