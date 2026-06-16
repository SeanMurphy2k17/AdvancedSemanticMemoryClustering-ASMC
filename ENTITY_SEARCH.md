# Entity-indexed retrieval

LTM queries now run three retrieval channels per query and merge the results,
instead of relying on FAISS semantic distance alone.

## The three channels

1. **FAISS semantic search** (existing) — encodes the query into the 6D
   spatial-valence coordinate and does a nearest-neighbor search over
   `IndexFlatL2`. Good for "feels similar" matches, weak on exact facts
   (names, dates, numbers) that don't move the coordinate much.

2. **Entity overlap** (new) — `spatialValenceCompute.extractFactualTags()`
   pulls people/orgs/locations, dates, quantities, and proper-noun
   "technical terms" out of text using NLTK (`ne_chunk` + `pos_tag`, no ML
   model). Every memory's tags get written into a reverse index in LMDB,
   keyed `entity_type\x00entity_text -> [mem_ids]`
   (`longTermMemory._index_entities`). At query time,
   `longTermMemory.resolveEntities(text)` extracts the same tags from the
   query and looks up exact (then lowercase-fallback) matches — giving
   O(1) lookups for "find every memory that mentions X" instead of a
   coordinate-distance proxy for it.

3. **Content word overlap** (new) — `longTermMemory.resolveContentWords()`
   scores memories by how many significant words they share with the
   query (`contentWords` field on each memory). A cheaper, fuzzier
   complement to exact entity matches — full LMDB cursor scan, fine at
   current scale, but the one part of this that doesn't have an index
   behind it yet if memory count grows a lot.

`memoryManager.queryMemory()` runs all three, unions the entity + content
IDs into a single `entity_ids` set, and passes that into
`longTermMemory.queryMemory(..., entity_ids=...)`, which merges
entity-matched memories into the FAISS results (capped at `k` total,
entity hits not already in the FAISS result set get appended).

## Where the tags get attached

`memorySpatial.composeRecord()` now calls `extractFactualTags()` when a
memory is first created (STM), so `factualTags` rides along with the
memory from the start instead of being computed lazily. `longTermMemory`
also backfills `factualTags` on any older memory that's missing them, the
first time it's touched by `_stitch()`.

## Index build / rebuild

The reverse index is built once per LTM database, in a background thread,
gated by an LMDB sentinel key (`__entity_index_ready__`):

- New DB / sentinel missing → `longTermMemory._init_entity_index()` kicks
  off `buildEntityIndex()` in a background thread on startup. Until it
  finishes, `resolveEntities()` returns `{}` (no entity channel, FAISS +
  content-word still work).
- Sentinel present → index is already built; nothing to do.
- New memories written after the index exists get indexed incrementally
  in `addMemory()`, no rebuild needed.
- Call `longTermMemory.buildEntityIndex()` directly to force a full
  rebuild (e.g. after changing `extractFactualTags()`'s extraction logic —
  it deletes all `entity_type\x00...` keys first, then re-extracts and
  re-indexes every memory in the DB from scratch).
- `wait_for_index(timeout=None)` blocks until a background rebuild
  finishes, if you need to guarantee the entity channel is live before
  querying (e.g. in tests/benchmarks).

## Spider stitching

Separately from retrieval, `longTermMemory._stitch()` runs on the top 3
direct hits of every query and proactively links memories together in
`linkedMemories` — bidirectionally, so if A links to B, B also links back
to A. A link is created when either:

- FAISS coordinate distance is within `0.40`, or
- the two memories share an entity or date tag (linked regardless of
  coordinate distance — this is what lets two memories about the same
  named thing connect even if they're semantically distant in the 6D
  space).

This is what builds the "semantic chain" that traversal
(`semanticTraverse`) walks across — the more queries run, the denser the
link graph gets between memories that share real-world facts, not just
similar valence.

## Benchmarking / debugging tools

- `diagnostic_entity_index.py` — inspect index state / sentinel / counts.
- `benchmark_entity_tags.py` — extraction throughput on sample text.
- `fast_query_benchmark.py` — end-to-end query latency across the three
  channels.
- `test_entity_index.py` — correctness checks for the reverse index.
