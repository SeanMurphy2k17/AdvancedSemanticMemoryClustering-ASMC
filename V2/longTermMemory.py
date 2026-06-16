import json
import math
import os
import threading

import faiss
import lmdb
import numpy as np

from spatialValenceCompute import spatialValenceCompute

FUZZINESS    = 0.4
DIM          = 6
LTM_MAP_SIZE = 10 * 1024 * 1024 * 1024


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
                if not mem_id:
                    continue
                cw = mem.get("contentWords", [])
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
            with open(self.counter_path, "r") as f:
                return json.load(f).get("next_id", 0)
        return 0

    def _save_counter(self):
        with open(self.counter_path, "w") as f:
            json.dump({"next_id": self._next_id}, f)

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
        content_words = memory.get("contentWords", [])
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
        seen_ids = set(int(i) for i in I[0] if i != -1)

        # Merge entity-retrieved memories (overlap channel)
        entity_direct = []
        if entity_ids:
            entity_set = set(entity_ids)
            with self._env.begin() as txn:
                for eid in entity_set:
                    if eid in seen_ids:
                        continue
                    raw = txn.get(str(eid).encode())
                    if raw:
                        mem = json.loads(raw)
                        entity_direct.append(mem)
                        seen_ids.add(eid)

        direct = direct + entity_direct[:k - len(direct)]  # cap total at k

        # Spider: stitch links before chain traversal so traversal follows more edges
        self._stitch(direct[:3])

        chain = []
        with self._env.begin() as txn:
            for mem in direct:
                for linked_pos in mem.get("linkedMemories", []):
                    lq = self._encode(linked_pos)
                    lD, lI = self._index.search(lq, 1)
                    linked_id = int(lI[0][0])
                    if linked_id == -1 or linked_id in seen_ids:
                        continue
                    raw = txn.get(str(linked_id).encode())
                    if raw:
                        chain.append(json.loads(raw))
                        seen_ids.add(linked_id)
                prev_pos = mem.get("prevPos")
                if prev_pos:
                    pq = self._encode(prev_pos)
                    _, pI = self._index.search(pq, 1)
                    prev_id = int(pI[0][0])
                    if prev_id != -1 and prev_id not in seen_ids:
                        raw = txn.get(str(prev_id).encode())
                        if raw:
                            chain.append(json.loads(raw))
                            seen_ids.add(prev_id)

        return {"direct": direct, "chain": chain}

    def _stitch(self, direct_results, k=20):
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
                if not mem_id:
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
                        # Existing proximity check
                        raw_dist = math.sqrt(sum((a - b) ** 2
                                                 for a, b in zip(mem_pos, cand_pos)))
                        if raw_dist > 0.40:
                            continue
                        if tuple(cand_pos) in existing:
                            continue
                        should_link = True

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
                    results[f"{etype}:{etext}"] = json.loads(raw)
        return results

    def resolveContentWords(self, content_words: list, top_n: int = 20) -> dict:
        """Look up memories by content word overlap using the inverted index.
        Returns {mem_id: overlap_count} for top_n memories ranked by overlap."""
        if not content_words:
            return {}
        query_set = set(content_words)
        # Aggregate mem_id -> count across all query words via index (O(1) per word)
        mem_scores = {}
        with self._env.begin() as txn:
            for word in query_set:
                key = self._cw_key(word)
                raw = txn.get(key)
                if raw:
                    mem_ids = json.loads(raw)
                    for mid in mem_ids:
                        mem_scores[mid] = mem_scores.get(mid, 0) + 1
        # Sort by overlap count descending, return top_n
        sorted_results = sorted(mem_scores.items(), key=lambda x: x[1], reverse=True)
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
            prev_pos = mem.get("prevPos")
            if prev_pos:
                _, pI = self._index.search(self._encode(prev_pos), 1)
                pid = int(pI[0][0])
                if pid != -1 and pid not in visited:
                    visited.add(pid)
                    n = self.fetchById(pid)
                    if n:
                        queue.append(n)
            for lp in mem.get("linkedMemories", []):
                _, lI = self._index.search(self._encode(lp), 1)
                lid = int(lI[0][0])
                if lid != -1 and lid not in visited:
                    visited.add(lid)
                    n = self.fetchById(lid)
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

