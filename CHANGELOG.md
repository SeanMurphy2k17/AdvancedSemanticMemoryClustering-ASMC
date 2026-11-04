# Changelog

## 2026-11 memory-fixes branch

Measured on two real stores with `tools/bench_self_query.py` (for each
long-term memory, a query made of its three rarest content words; a hit is
that memory, or one with the same prompt text, in the top six). Numbers are
from those two stores, not a benchmark suite; your store will differ.

| | store A (148 memories) | store B (204 memories) |
|---|---|---|
| recall@6, before this branch | 0.14 | 0.09 |
| recall@6, this branch | 0.91 | 0.79 (same-text; 0.40 exact id, since one message was stored many times) |
| seconds per query, before | 0.3 to 0.5 idle; 5 to 20 under load | 13 to 21 under load |
| seconds per query, this branch | 0.02 | 0.02 |

### Retrieval
- Overlap hits (entity and content-word channels) are interleaved with the
  FAISS hits in rank order. They used to be appended only into the slots
  FAISS left free, which was none once the index held more than k memories,
  so the channel that works best was discarded.
- Content-word scores are weighted by inverse document frequency.
- Answer words are indexed beside prompt words (`responseWords`).
- The short-term window is scanned for exact-word overlap; the inverted
  index covers long-term memory only.
- The chain traversal skips a linked memory whose text was already placed.
- Entities are extracted by merging consecutive proper nouns instead of
  running the NLTK chunker: sub-millisecond instead of 0.3 s per call, and
  no more lowercase chat words tagged as places and people. Multi-word
  names are entities, single names are technical terms; the reverse index
  takes both. The tokenizer passes are cached per line.
- An entity whose posting list exceeds 500 memories is treated as templated
  text and skipped at query time.

### Integrity
- The window checkpoint is written after the promoted entry is popped and
  written to long-term memory, never before, and always on a promotion. An
  unclean exit used to promote that entry again on the next start and leave
  the window one over its maximum for good.
- One re-entrant lock around `queryMemory` and `addMemory`. Threaded callers
  were racing a FAISS write against a search on an unlocked index.
- A window over its maximum is drained on open, skipping entries long-term
  memory already holds.
- `counter.json` is written atomically and recovered from the FAISS count if
  found corrupt.

### Web and speed
- The stitcher links each memory to its nearest few (`ASMC_STITCH_NEIGHBOURS`,
  default 5) within `ASMC_STITCH_MAX_DIST`, instead of everything inside a
  fixed 0.40 radius, which in a tight store was everything.
- FAISS is pinned to one thread at index load (`ASMC_FAISS_THREADS`); the
  per-link searches no longer wait on OpenMP under load.
- Chain traversal resolves all linked positions in one batched search.
- A memory with `ltm_id` 0 is no longer skipped by the stitcher.

### Store location
- The word cache follows `ASMC_STORE_DIR` like the window and the long-term
  store, so two stores never share a cache.
- `create_memory(db_path=..., max_entries=...)` now takes effect; both
  arguments were accepted and ignored.

### Repository
- Runtime store files and bytecode are no longer tracked.
- `tools/migrate_store.py`, `tools/bench_self_query.py`,
  `tests/test_window_checkpoint.py`.

Existing stores: run `python tools/migrate_store.py --store <dir> --dry-run`
with the owning process stopped, then without `--dry-run`.
