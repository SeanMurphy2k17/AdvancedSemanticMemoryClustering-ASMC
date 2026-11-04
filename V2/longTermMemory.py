import json
import math
from itertools import zip_longest
import os
import threading

import faiss
import lmdb
import numpy as np

from spatialValenceCompute import spatialValenceCompute

FUZZINESS    = 0.4
DIM          = 6
LTM_MAP_SIZE = 10 * 1024 * 1024 * 1024
# Entities whose reverse-index posting list exceeds this are treated as
# stopword-like false positives (e.g. templated text repeating "Level"/"Room")
# rather than genuine named entities, and are dropped at query time.
GENERIC_ENTITY_THRESHOLD = 500


# Proximity links per memory: the nearest few, not everything inside a radius.
# A denser web gives the traversal more to wander along (texture); a sparser one
# keeps links meaningful in a store whose memories all sit close together.
# Set ASMC_STITCH_NEIGHBOURS per store; 5 suits a tight space, 10-12 a wide one.
STITCH_NEIGHBOURS = int(os.environ.get("ASMC_STITCH_NEIGHBOURS", "5"))
STITCH_MAX_DIST   = float(os.environ.get("ASMC_STITCH_MAX_DIST", "0.40"))  # never link by proximity beyond this

class longTermMemory:
    def __init__(self):
        base             = os.path.dirname(os.path.abspath(__file__))
        self.ltm_dir     = os.path.join(base, "MemoryStructures", "LTM")
        self.faiss_path  = os.path.join(self.ltm_dir, "asmc.faiss")
        self.counter_path  = os.path.join(self.ltm_dir, "counter.json")
        self.cursors_path  = os.path.join(self.ltm_dir, "platform_cursors.json")
        os.makedirs(self.ltm_dir, exist_ok=True)
        self._cursors = json.load(open(self.cursors_path)) if os.path.exists(self.cursors_path) else {}

        self._env = lmdb.open(self.ltm_dir, map_size=LTM_MAP_SIZE)

        # One thread for a flat index of a few hundred 6-dim vectors: the search is
        # microseconds, and FAISS's default OpenMP fan-out made each of the hundreds
        # of per-link searches a query performs wait behind whatever else loads the
        # machine (measured: 725 searches, 23 ms each, 16.7 s of an 18 s query).
        faiss.omp_set_num_threads(int(os.environ.get("ASMC_FAISS_THREADS", "1")))
        if os.path.exists(self.faiss_path):
            self._index = faiss.read_index(self.faiss_path)
        else:
            self._index = faiss.IndexIDMap(faiss.IndexFlatL2(DIM))

        self._next_id = self._load_counter()
        self._index_event = threading.Event()  # set when entity rebuild completes
        self._cw_index_event = threading.Event()  # set when content-word rebuild completes
        self._init_entity_index()
        self._init_content_index()
        print(f"initialized {self.__class__.__name__}")

    def _init_entity_index(self):
        """Check if entity index exists; start async rebuild if missing (old DB)."""
        with self._env.begin() as txn:
            self._index_ready = txn.get(b"__entity_index_ready__") is not None

        if not self._index_ready:
            print("[ENTITY INDEX] ⚠ entity index not found — starting background rebuild...", flush=True)
            self._index_ready = False
            # Start rebuild in background thread so startup is not blocked
            def _rebuild_bg():
                try:
                    total = self.buildEntityIndex()
                    print(f"[ENTITY INDEX] ✅ background rebuild complete: {total} memories indexed", flush=True)
                except Exception as e:
                    print(f"[ENTITY INDEX] ⚠ background rebuild failed: {e}", flush=True)
                finally:
                    self._index_event.set()  # signal completion
            t = threading.Thread(target=_rebuild_bg, daemon=True)
            t.start()
        else:
            self._index_event.set()  # already ready, no wait needed
            print("[ENTITY INDEX] ✅ entity index present", flush=True)

    def wait_for_index(self, timeout=None):
        """Block until entity index rebuild completes. timeout in seconds (None=infinite)."""
        ready = self._index_event.wait(timeout=timeout)
        if not ready:
            print("[ENTITY INDEX] ⚠ wait timed out — index may still be rebuilding", flush=True)
            return False
        print("[ENTITY INDEX] ✅ rebuild finished (or already ready)", flush=True)
        return True

    def _init_content_index(self):
        """Check if content-word index exists; start async rebuild if missing (old DB)."""
        with self._env.begin() as txn:
            self._cw_index_ready = txn.get(b"__content_index_ready__") is not None

        if not self._cw_index_ready:
            print("[CONTENT INDEX] ⚠ content-word index not found — starting background rebuild...", flush=True)
            self._cw_index_ready = False
            def _rebuild_bg():
                try:
                    total = self.buildContentIndex()
                    print(f"[CONTENT INDEX] ✅ background rebuild complete: {total} memories indexed", flush=True)
                except Exception as e:
                    print(f"[CONTENT INDEX] ⚠ background rebuild failed: {e}", flush=True)
                finally:
                    self._cw_index_event.set()
            t = threading.Thread(target=_rebuild_bg, daemon=True)
            t.start()
        else:
            self._cw_index_event.set()
            print("[CONTENT INDEX] ✅ content-word index present", flush=True)

    def wait_for_cw_index(self, timeout=None):
        """Block until content-word index rebuild completes. timeout in seconds."""
        ready = self._cw_index_event.wait(timeout=timeout)
        if not ready:
            print("[CONTENT INDEX] ⚠ wait timed out — index may still be rebuilding", flush=True)
            return False
        print("[CONTENT INDEX] ✅ rebuild finished (or already ready)", flush=True)
        return True

    def _cw_key(self, word: str) -> bytes:
        """Encode a content word into a LMDB key: 'cw\x00word'"""
        return f"cw\x00{word}".encode()

    def _index_content_words(self, mem_id: int, content_words: list, txn=None):
        """Incrementally add content words from a single memory into the inverted index.
        Called automatically during addMemory(). Pass txn to reuse a parent write txn."""
        if not content_words:
            return
        own_txn = txn is None
        if own_txn:
            txn = self._env.begin(write=True)
        try:
            for word in content_words:
                key = self._cw_key(word)
                existing = txn.get(key)
                if existing:
                    ids = json.loads(existing)
                else:
                    ids = []
                if mem_id not in ids:
                    ids.append(mem_id)
                    ids.sort()
                    txn.put(key, json.dumps(ids).encode())
        finally:
            if own_txn:
                txn.commit()

    def _entity_key(self, entity_type: str, entity_text: str) -> bytes:
        """Encode an entity into a LMDB key: 'type\x00text'"""
        return f"{entity_type}\x00{entity_text}".encode()

    def _index_entities(self, mem_id: int, factual_tags: dict, txn=None):
        """Incrementally add entities from a single memory into the reverse index.
        Called automatically during addMemory(). Pass txn to reuse a parent write txn."""
        if not factual_tags:
            return
        tags = [
            (etype, etext) for etype, etext in factual_tags.get("entities", [])
        ] + [
            (etype, etext) for etype, etext in factual_tags.get("dates", [])
        ] + [
            (etype, etext) for etype, etext in factual_tags.get("quantities", [])
        ] + [
            (etype, etext) for etype, etext in factual_tags.get("technical_terms", [])
        ]
        if not tags:
            return
        # Open our own write txn if no parent was passed in
        own_txn = txn is None
        if own_txn:
            txn = self._env.begin(write=True)
        try:
            for etype, etext in tags:
                key = self._entity_key(etype, etext)
                existing = txn.get(key)
                if existing:
                    ids = json.loads(existing)
                else:
                    ids = []
                if mem_id not in ids:
                    ids.append(mem_id)
                    ids.sort()
                    txn.put(key, json.dumps(ids).encode())
        finally:
            if own_txn:
                txn.commit()

    def buildEntityIndex(self) -> int:
        """Bulk-build (or rebuild) the entity reverse index over all LTM memories.
        Returns the number of unique entities indexed."""
        print("[ENTITY INDEX] building reverse index...")
        indexed = 0
        # Step 1: collect all entity keys to delete (can't delete while iterating)
        entity_keys = []
        with self._env.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                if b"\x00" in key:
                    entity_keys.append(key)
        # Step 2: delete them in a write transaction
        if entity_keys:
            with self._env.begin(write=True) as txn:
                for key in entity_keys:
                    txn.delete(key)
                txn.delete(b"__entity_index_ready__")
        # Step 3: rebuild from all memories
        with self._env.begin(write=True) as txn:
            cursor = txn.cursor()
            for key, raw in cursor:
                if key == b"__entity_index_ready__" or b"\x00" in key:
                    continue
                mem = json.loads(raw)
                meta = mem.get("metaDataTag", {})
                if meta.get("type") == "scm_anchor":
                    continue
                mem_id = meta.get("ltm_id")
                if not mem_id:
                    continue
                # Always re-extract during rebuild — old tags may have
                # stale formats (e.g. tuple strings from NLTK tree leaves)
                factual_tags = spatialValenceCompute().extractFactualTags(
                    mem.get("inputText", "")
                )
                mem["factualTags"] = factual_tags
                txn.put(key, json.dumps(mem, default=lambda o: list(o)
                                      if isinstance(o, tuple) else o).encode())
                self._index_entities(mem_id, factual_tags, txn=txn)
                indexed += 1
        # Step 4: mark index as ready
        with self._env.begin(write=True) as txn:
            txn.put(b"__entity_index_ready__", b"1")
        self._index_ready = True
        print(f"[ENTITY INDEX] built: {indexed} memories indexed")
        return indexed

    def buildContentIndex(self) -> int:
        """Bulk-build (or rebuild) the content-word inverted index over all LTM memories.
        Returns the number of unique content words indexed."""
        print("[CONTENT INDEX] building inverted index...")
        indexed = 0
        cw_count = 0
        # Step 1: collect all cw keys to delete
        cw_keys = []
        with self._env.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                if key.startswith(b"cw\x00"):
                    cw_keys.append(key)
        # Step 2: delete them
        if cw_keys:
            with self._env.begin(write=True) as txn:
                for key in cw_keys:
                    txn.delete(key)
                txn.delete(b"__content_index_ready__")
        # Step 3: rebuild from all memories
        with self._env.begin(write=True) as txn:
            cursor = txn.cursor()
            for key, raw in cursor:
                if key == b"__content_index_ready__" or key.startswith(b"cw\x00") or b"\x00" in key:
                    continue
                mem = json.loads(raw)
                if not isinstance(mem, dict):
                    continue
                meta = mem.get("metaDataTag", {})
                if meta.get("type") == "scm_anchor":
                    continue
                mem_id = meta.get("ltm_id")
                if mem_id is None:
                    continue
                cw = list(dict.fromkeys((mem.get("contentWords") or []) + (mem.get("responseWords") or [])))
                if cw:
                    self._index_content_words(mem_id, cw, txn=txn)
                    cw_count += len(set(cw))
                indexed += 1
        # Step 4: mark index as ready
        with self._env.begin(write=True) as txn:
            txn.put(b"__content_index_ready__", b"1")
        self._cw_index_ready = True
        print(f"[CONTENT INDEX] built: {indexed} memories indexed, {cw_count} unique content words")
        return indexed

    def _load_counter(self) -> int:
        if os.path.exists(self.counter_path):
            try:
                with open(self.counter_path, "r") as f:
                    return json.load(f).get("next_id", 0)
            except (json.JSONDecodeError, OSError) as e:
                # counter.json can be left empty/corrupt if the process was
                # killed mid-write (the old _save_counter truncated in place
                # before writing). Recover by trusting FAISS's ntotal, which
                # is only ever written after a full addMemory() completes.
                recovered = int(self._index.ntotal)
                print(f"[LTM] ⚠ counter.json unreadable ({e}) — recovering "
                      f"next_id={recovered} from FAISS index", flush=True)
                return recovered
        return 0

    def _save_counter(self):
        # Atomic write: a plain open(path, "w") truncates the file before
        # writing, so a crash mid-write leaves an empty/corrupt counter.json.
        # Write to a temp file in the same dir and rename over it instead.
        tmp_path = self.counter_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump({"next_id": self._next_id}, f)
        os.replace(tmp_path, self.counter_path)

    def _encode(self, coord) -> np.ndarray:
        v = np.array(coord, dtype=np.float32)
        v[3:] *= FUZZINESS
        return v.reshape(1, -1)

    def addMemory(self, memory: dict):
        mem_id = self._next_id
        self._next_id += 1

        memory["metaDataTag"]["ltm_id"] = mem_id

        vec = self._encode(memory["responsePos"])
        self._index.add_with_ids(vec, np.array([mem_id], dtype=np.int64))

        payload = json.dumps(memory, default=lambda o: list(o) if isinstance(o, tuple) else o).encode()
        with self._env.begin(write=True) as txn:
            txn.put(str(mem_id).encode(), payload)

        # Incremental: index this memory's entities and content words
        factual_tags = memory.get("factualTags")
        if factual_tags:
            self._index_entities(mem_id, factual_tags)
        # Index words from the prompt and the answer: what was decided is usually in the answer.
        content_words = list(dict.fromkeys((memory.get("contentWords") or []) + (memory.get("responseWords") or [])))
        if content_words:
            self._index_content_words(mem_id, content_words)

        faiss.write_index(self._index, self.faiss_path)
        self._save_counter()
        platform = memory.get("metaDataTag", {}).get("platform")
        if platform:
            self._cursors[platform] = mem_id
            with open(self.cursors_path, "w") as f: json.dump(self._cursors, f)
        return mem_id

    def fetchById(self, ltm_id: int) -> dict:
        with self._env.begin() as txn:
            raw = txn.get(str(ltm_id).encode())
            return json.loads(raw) if raw else None

    def fetch_platform_cursor(self, platform: str) -> dict:
        ltm_id = self._cursors.get(platform)
        return self.fetchById(ltm_id) if ltm_id is not None else None

    def updatePayload(self, ltm_id: int, data: dict):
        """Overwrite an existing LMDB record without touching the FAISS index."""
        payload = json.dumps(data, default=lambda o: list(o) if isinstance(o, tuple) else o).encode()
        with self._env.begin(write=True) as txn:
            txn.put(str(ltm_id).encode(), payload)

    def _syn_rerank(self, candidates: list, syn_coord: tuple,
                    boost: float = 0.4) -> list:
        q = np.array(syn_coord, dtype=np.float32)
        scored = []
        for sem_dist, mem in candidates:
            syn_pos = mem.get("worldPos")
            if syn_pos:
                diff     = np.array(syn_pos, dtype=np.float32) - q
                syn_dist = float(np.dot(diff, diff))
            else:
                syn_dist = 0.0
            scored.append((sem_dist + boost * syn_dist, mem))
        scored.sort(key=lambda x: x[0])
        return scored

    def queryMemory(self, inputCoord, k: int = 10, syn_coord=None, entity_ids=None) -> dict:
        import time as _t
        import threading as _th
        _tid = _th.current_thread().name
        _q0 = _t.perf_counter()
        q        = self._encode(inputCoord)
        search_k = min(k * 4, max(self._index.ntotal, 1))
        D, I     = self._index.search(q, search_k)

        candidates = []
        with self._env.begin() as txn:
            for dist, mem_id in zip(D[0], I[0]):
                if mem_id == -1:
                    continue
                raw = txn.get(str(mem_id).encode())
                if raw:
                    candidates.append((float(dist), json.loads(raw)))

        if syn_coord and candidates:
            candidates = self._syn_rerank(candidates, syn_coord)

        direct   = [m for _, m in candidates[:k]]
        seen_ids = {int(m["metaDataTag"]["ltm_id"]) for m in direct if "ltm_id" in m.get("metaDataTag", {})}
        print(f"[LTM.queryMemory:{_tid}] faiss+decode direct: {_t.perf_counter()-_q0:.3f}s", flush=True)

        # Overlap channel (entity / content-word hits), kept in the caller's rank order.
        _q1 = _t.perf_counter()
        entity_direct = []
        if entity_ids:
            with self._env.begin() as txn:
                for eid in entity_ids:
                    eid = int(eid)
                    if eid in seen_ids:
                        continue
                    raw = txn.get(str(eid).encode())
                    if raw:
                        entity_direct.append(json.loads(raw))
                        seen_ids.add(eid)

        # Interleave, overlap first: an exact word or entity match is stronger
        # evidence than valence proximity. Appending overlap hits only into the
        # slots FAISS left free dropped them entirely once the index held more
        # than k memories (measured: self-query recall@6 0.14 -> 0.92 on one
        # store, 0.09 -> 0.47 on another).
        merged = []
        for pair in zip_longest(entity_direct, direct):
            for m in pair:
                if m is not None:
                    merged.append(m)
        direct = merged[:k]
        seen_ids = {int(m["metaDataTag"]["ltm_id"]) for m in direct if "ltm_id" in m.get("metaDataTag", {})}
        print(f"[LTM.queryMemory:{_tid}] entity merge: {_t.perf_counter()-_q1:.3f}s (entity_ids={len(entity_ids) if entity_ids else 0})", flush=True)

        # Spider: stitch links before chain traversal so traversal follows more edges
        _q2 = _t.perf_counter()
        self._stitch(direct[:3])
        print(f"[LTM.queryMemory:{_tid}] _stitch: {_t.perf_counter()-_q2:.3f}s", flush=True)

        _q3 = _t.perf_counter()
        # Collect every linked/prev position across all direct results, then
        # resolve them in ONE batched FAISS search instead of one search()
        # call per position — with dense link graphs (hundreds of edges) the
        # per-call Python/FAISS overhead, not the actual distance math, was
        # what made this loop take seconds.
        all_positions = []
        for mem in direct:
            all_positions.extend(mem.get("linkedMemories", []))
            prev_pos = mem.get("prevPos")
            if prev_pos:
                all_positions.append(prev_pos)
        total_links = len(all_positions)

        chain = []
        seen_texts = {m.get("inputText", "") for m in direct}
        if all_positions:
            batch = np.vstack([self._encode(p) for p in all_positions])
            _, I_batch = self._index.search(batch, 1)
            with self._env.begin() as txn:
                for row in I_batch:
                    linked_id = int(row[0])
                    if linked_id == -1 or linked_id in seen_ids:
                        continue
                    raw = txn.get(str(linked_id).encode())
                    if not raw:
                        continue
                    linked = json.loads(raw)
                    seen_ids.add(linked_id)
                    # Same text stored more than once (a harness re-sending a message
                    # across restarts) would otherwise fill consecutive chain slots.
                    if linked.get("inputText", "") in seen_texts:
                        continue
                    seen_texts.add(linked.get("inputText", ""))
                    chain.append(linked)
        print(f"[LTM.queryMemory:{_tid}] chain traversal: {_t.perf_counter()-_q3:.3f}s (total_links={total_links}, chain={len(chain)})", flush=True)

        return {"direct": direct, "chain": chain}

    def _stitch(self, direct_results, k=12):
        """
        For each direct result, find nearby LTM memories and add links.
        Called BEFORE chain traversal so the traversal follows more edges.
        Backfills factualTags on old memories that lack them.
        BIDIRECTIONAL: when A links to B, B also links back to A.
        Returns total number of new links created.
        """
        new_links = 0
        with self._env.begin(write=True) as txn:
            for mem in direct_results:
                meta = mem.get("metaDataTag", {})
                if meta.get("type") == "scm_anchor":
                    continue
                mem_id = meta.get("ltm_id")
                if mem_id is None:  # 0 is a valid id
                    continue
                mem_pos = mem.get("responsePos")
                if not mem_pos:
                    continue

                # Backfill factualTags on old memories that lack them
                if "factualTags" not in mem:
                    mem["factualTags"] = spatialValenceCompute().extractFactualTags(
                        mem.get("inputText", "")
                    )
                    txn.put(str(mem_id).encode(),
                        json.dumps(mem, default=lambda o: list(o)
                                   if isinstance(o, tuple) else o).encode())

                vec = self._encode(mem_pos)
                search_k = min(k, max(self._index.ntotal, 1))
                D, I = self._index.search(vec, search_k)

                existing = set()
                for coord in mem.get("linkedMemories", []):
                    existing.add(tuple(coord))

                # Collect query entities for overlap linking
                mem_entities = set()
                for etype, etext in mem.get("factualTags", {}).get("entities", []):
                    mem_entities.add((etype, etext))
                for etype, etext in mem.get("factualTags", {}).get("dates", []):
                    mem_entities.add((etype, etext))

                # Track candidates that need reverse-link updates
                cand_updates = {}  # cand_id -> updated cand dict
                prox_links = len(existing)  # proximity links this memory already has

                for dist, cand_id in zip(D[0], I[0]):
                    if cand_id == -1 or cand_id == mem_id:
                        continue
                    if cand_id in existing:
                        continue

                    raw = txn.get(str(cand_id).encode())
                    if not raw:
                        continue
                    cand = json.loads(raw)
                    cand_pos = cand.get("responsePos") or cand.get("inputPos")
                    if not cand_pos:
                        continue

                    # Backfill factualTags on candidate if missing
                    if "factualTags" not in cand:
                        cand["factualTags"] = spatialValenceCompute().extractFactualTags(
                            cand.get("inputText", "")
                        )
                        txn.put(str(cand_id).encode(),
                            json.dumps(cand, default=lambda o: list(o)
                                       if isinstance(o, tuple) else o).encode())

                    cand_entities = set()
                    for etype, etext in cand.get("factualTags", {}).get("entities", []):
                        cand_entities.add((etype, etext))
                    for etype, etext in cand.get("factualTags", {}).get("dates", []):
                        cand_entities.add((etype, etext))

                    should_link = False

                    # Entity overlap check - link even if FAISS distance > 0.40
                    overlap = mem_entities & cand_entities
                    if overlap and tuple(cand_pos) not in existing:
                        should_link = True

                    if not should_link:
                        # Proximity: the nearest few, not everything inside a fixed
                        # radius. In a store whose memories all sit within ~0.25 of
                        # each other, a 0.40 radius linked every memory to every
                        # other and the web carried no information.
                        if prox_links >= STITCH_NEIGHBOURS:
                            continue
                        raw_dist = math.sqrt(sum((a - b) ** 2
                                                 for a, b in zip(mem_pos, cand_pos)))
                        if raw_dist > STITCH_MAX_DIST:
                            continue
                        if tuple(cand_pos) in existing:
                            continue
                        should_link = True
                        prox_links += 1

                    if should_link:
                        existing.add(tuple(cand_pos))
                        my_anchor = list(cand_pos)
                        mem["linkedMemories"].append(my_anchor)
                        new_links += 1

                        # Bidirectional: candidate also remembers us
                        cand_my_anchor = list(mem_pos)
                        cand_links = cand.setdefault("linkedMemories", [])
                        if cand_my_anchor not in cand_links:
                            cand_links.append(cand_my_anchor)
                            cand_updates[cand_id] = cand

                if mem["linkedMemories"]:
                    txn.put(str(mem_id).encode(),
                        json.dumps(mem, default=lambda o: list(o)
                                   if isinstance(o, tuple) else o).encode())

                # Write all bidirectional updates back
                for cid, updated_cand in cand_updates.items():
                    txn.put(str(cid).encode(),
                        json.dumps(updated_cand, default=lambda o: list(o)
                                   if isinstance(o, tuple) else o).encode())

        return new_links

    def resolveEntities(self, text: str) -> dict:
        """Look up entities in the reverse index. Returns {entity_key: [mem_ids]}.
        Uses the same extractFactualTags to tokenize the input text.
        Case-insensitive: tries exact match first, then lowercase fallback."""
        if not self._index_ready:
            return {}
        tags = spatialValenceCompute().extractFactualTags(text)
        results = {}
        with self._env.begin() as txn:
            for etype, etext in (tags.get("entities", []) +
                                  tags.get("dates", []) +
                                  tags.get("quantities", []) +
                                  tags.get("technical_terms", [])):
                key = self._entity_key(etype, etext)
                raw = txn.get(key)
                if not raw:
                    # Fallback: try lowercase entity text (handles case mismatches)
                    key_lower = self._entity_key(etype, etext.lower())
                    raw = txn.get(key_lower)
                if raw:
                    ids = json.loads(raw)
                    if len(ids) > GENERIC_ENTITY_THRESHOLD:
                        # Posting list this large means the "entity" is actually
                        # a generic/templated word (e.g. repeated mission text),
                        # not a useful discriminator — skip it rather than pay
                        # to decode thousands of memories for it downstream.
                        continue
                    results[f"{etype}:{etext}"] = ids
        return results

    def resolveContentWords(self, content_words: list, top_n: int = 20) -> dict:
        """Look up memories by content word overlap using the inverted index.
        Returns {mem_id: overlap_count} for top_n memories ranked by overlap."""
        if not content_words:
            return {}
        query_set = set(content_words)
        total = max(self._index.ntotal, 1)
        # Aggregate mem_id -> score across query words; a word that appears in few
        # memories counts for more than one that appears in most ("do" vs "faiss").
        mem_scores = {}
        with self._env.begin() as txn:
            for word in query_set:
                raw = txn.get(self._cw_key(word))
                if not raw:
                    continue
                mem_ids = json.loads(raw)
                weight = math.log(1.0 + total / max(len(mem_ids), 1))
                for mid in mem_ids:
                    mem_scores[mid] = mem_scores.get(mid, 0.0) + weight
        sorted_results = sorted(mem_scores.items(), key=lambda x: (-x[1], x[0]))
        return dict(sorted_results[:top_n])

    def semanticTraverse(self, start_memories: list, query_words: list,
                         extract_fn, max_results: int = 8, max_nodes: int = 20) -> list:
        bucket      = []
        bucket_keys = set()
        visited     = set()
        queue       = list(start_memories)
        query_set   = set(query_words)
        nodes_seen  = 0
        while queue and len(bucket) < max_results and nodes_seen < max_nodes:
            mem = queue.pop(0)
            nodes_seen += 1
            cw  = mem.get("contentWords") or extract_fn(mem.get("inputText", ""))
            key = mem.get("inputText", "") + mem.get("timeDate", "")
            if (query_set & set(cw)) and key not in bucket_keys:
                bucket.append(mem)
                bucket_keys.add(key)
            # Batch this node's linked/prev positions into one FAISS search
            # instead of one search() call per position — with dense link
            # graphs a node can carry dozens of links, and per-call overhead
            # (not the distance math) is what made this loop slow.
            positions = list(mem.get("linkedMemories", []))
            prev_pos = mem.get("prevPos")
            if prev_pos:
                positions.append(prev_pos)
            if positions:
                batch = np.vstack([self._encode(p) for p in positions])
                _, I_batch = self._index.search(batch, 1)
                for row in I_batch:
                    nid = int(row[0])
                    if nid == -1 or nid in visited:
                        continue
                    visited.add(nid)
                    n = self.fetchById(nid)
                    if n:
                        queue.append(n)
        return bucket

    def _clear(self):
        with self._env.begin(write=True) as txn:
            txn.drop(self._env.open_db(), delete=False)
        self._index = faiss.IndexIDMap(faiss.IndexFlatL2(DIM))
        self._next_id = 0
        if os.path.exists(self.faiss_path):
            os.remove(self.faiss_path)
        self._save_counter()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from memorySpatial import memorySpatial

    PHRASES = [
        "my back is absolutely killing me",         "i have a terrible headache",
        "my feet are sore and aching",              "my muscles throb from the workout",
        "i am so incredibly happy today",           "this is the best day of my life",
        "everything feels wonderful and bright",    "my heart is overflowing with gratitude",
        "i feel completely broken and hollow",      "i lost someone i truly loved",
        "the sadness is slowly consuming me",       "i cannot stop crying",
        "i am terrified and i cannot move",         "something is very wrong and i am scared",
        "my heart is pounding and i am shaking",    "i feel paralysed by dread",
        "i am starving and i want pizza",           "the meal was absolutely delicious",
        "she cooked an incredible dinner",          "i could eat an entire cake right now",
        "the server crashed and lost all data",     "the algorithm runs in polynomial time",
        "the api keeps returning timeout errors",   "we deployed the build to production",
        "the sunset was breathtaking and vivid",    "the mountains looked ancient and vast",
        "the ocean stretched on forever",           "the forest was silent and still",
        "i cannot understand this concept",         "the problem is beyond my comprehension",
        "nothing makes sense to me anymore",        "i am completely lost and confused",
        "she laughed until her sides ached",        "we danced all night long",
        "the children played in the garden",        "he smiled and everything was okay",
        "the diagnosis was worse than expected",    "the news hit like a freight train",
        "everything i built has fallen apart",      "i have nothing left to hold on to",
        "the engine overheated and stalled",        "the brakes failed on the highway",
        "the battery died in the middle of nowhere","the tire blew out at speed",
        "i wrote a poem about the rain",            "the painting took three weeks to finish",
        "the music filled the whole room",          "she sang and the crowd fell silent",
        "the experiment failed again",              "the hypothesis was proven wrong",
        "we need to rethink our entire approach",   "the data does not support the conclusion",
        "the dog barked at every shadow",           "the cat knocked everything off the shelf",
        "the bird built its nest in three days",    "the whale surfaced close to the boat",
        "i ran five miles before breakfast",        "the weights felt heavier than usual",
        "my knees are shot from the climb",         "the swim left me completely exhausted",
        "the coffee was bitter and cold",           "the bread was stale and hard",
        "the soup was too salty to finish",         "the fruit was ripe and sweet",
        "i got lost in a city i do not know",       "the map was useless and wrong",
        "we missed the last train home",            "the flight was delayed by six hours",
        "i miss the way things used to be",         "everything has changed so fast",
        "i wish i could go back",                   "those days are gone forever",
        "justice was not served that day",          "the verdict shocked the entire court",
        "freedom means something different to everyone", "democracy requires constant effort",
        "the baby laughed for the first time",      "she took her first steps today",
        "he graduated after seven hard years",      "she finished the race she started",
        "the storm knocked out power for days",     "the flood destroyed everything downstream",
        "the fire spread faster than expected",     "the earthquake was shallow and violent",
        "i cannot sleep and it is three in the morning", "the nightmares keep coming back",
        "exhaustion has settled deep into my bones","i have not rested in weeks",
        "the meeting went completely off the rails","nobody agreed on anything",
        "the project is weeks behind schedule",     "the budget has been cut in half",
        "she forgave him after years of silence",   "they reconciled over a long dinner",
        "the apology came too late to matter",      "some wounds do not heal easily",
        "the idea clicked all at once",             "suddenly everything made perfect sense",
        "the solution was simpler than expected",   "we had been overthinking it entirely",
    ]

    _ms  = memorySpatial()
    _ltm = longTermMemory()
    _ltm._clear()

    print(f"\nInserting {len(PHRASES)} memories...")
    objects = []
    for phrase in PHRASES:
        obj = _ms.buildMemoryObject(
            inputText    = phrase,
            responseText = f"response: {phrase}",
            metaDataTag  = {"test": True},
        )
        _ltm.addMemory(obj)
        objects.append(obj)
    print(f"Inserted {len(objects)} memories.\n")

    print("Searching for each memory by responsePos (k=1)...")
    hits = 0
    misses = []
    for obj in objects:
        result = _ltm.queryMemory(obj["responsePos"], k=1)
        if result["direct"]:
            returned = result["direct"][0]
            if returned["responseText"] == obj["responseText"]:
                hits += 1
            else:
                misses.append((obj["inputText"], returned["inputText"]))
        else:
            misses.append((obj["inputText"], "NO RESULT"))

    print(f"  Hits:   {hits} / {len(objects)}")
    print(f"  Misses: {len(misses)}")
    for expected, got in misses:
        print(f"    expected: '{expected[:50]}'")
        print(f"    got:      '{got[:50]}'")

    # --- radial neighbourhood test ---
    QUERIES = [
        "my body is in agony",
        "i feel pure joy and happiness",
        "grief has swallowed me whole",
        "i am filled with terror",
        "the system is broken and crashing",
    ]

    print(f"\n{'='*70}")
    print("RADIAL SEARCH  k=10  — what lives near each query?")
    print(f"{'='*70}")
    for q in QUERIES:
        coord = _ms._svc.computeSpatialValence(q)
        result = _ltm.queryMemory(coord, k=10)
        print(f"\n  QUERY: '{q}'")
        print(f"  coord: {coord}")
        print(f"  direct ({len(result['direct'])}):")
        for m in result["direct"]:
            print(f"    - {m['inputText']}")

