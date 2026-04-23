import json
import os

import faiss
import lmdb
import numpy as np

FUZZINESS    = 0.4
DIM          = 6
LTM_MAP_SIZE = 10 * 1024 * 1024 * 1024


class longTermMemory:
    def __init__(self):
        base             = os.path.dirname(os.path.abspath(__file__))
        self.ltm_dir     = os.path.join(base, "MemoryStructures", "LTM")
        self.faiss_path  = os.path.join(self.ltm_dir, "asmc.faiss")
        self.counter_path = os.path.join(self.ltm_dir, "counter.json")
        os.makedirs(self.ltm_dir, exist_ok=True)

        self._env = lmdb.open(self.ltm_dir, map_size=LTM_MAP_SIZE)

        if os.path.exists(self.faiss_path):
            self._index = faiss.read_index(self.faiss_path)
        else:
            self._index = faiss.IndexIDMap(faiss.IndexFlatL2(DIM))

        self._next_id = self._load_counter()
        print(f"initialized {self.__class__.__name__}")

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

        vec = self._encode(memory["responsePos"])
        self._index.add_with_ids(vec, np.array([mem_id], dtype=np.int64))

        payload = json.dumps(memory, default=lambda o: list(o) if isinstance(o, tuple) else o).encode()
        with self._env.begin(write=True) as txn:
            txn.put(str(mem_id).encode(), payload)

        faiss.write_index(self._index, self.faiss_path)
        self._save_counter()
        return mem_id

    def fetchById(self, ltm_id: int) -> dict:
        with self._env.begin() as txn:
            raw = txn.get(str(ltm_id).encode())
            return json.loads(raw) if raw else None

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

    def queryMemory(self, inputCoord, k: int = 10, syn_coord=None) -> dict:
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

