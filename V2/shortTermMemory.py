import json
import math
import os
import threading


class shortTermMemory:

    STM_MAX    = 50
    STM_RADIUS = 0.05

    def __init__(self):
        base          = os.path.dirname(os.path.abspath(__file__))
        self.stm_dir  = os.path.join(base, "MemoryStructures", "STM")
        self.stm_path = os.path.join(self.stm_dir, "stm.json")
        os.makedirs(self.stm_dir, exist_ok=True)
        self._lock         = threading.Lock()
        self._ckpt_every   = 3
        self._ckpt_counter = 0
        self._entries      = []
        if os.path.isfile(self.stm_path):
            try:
                with open(self.stm_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded.get("entries"), list):
                    self._entries = loaded["entries"]
            except Exception:
                self._entries = []
        print(f"initialized {self.__class__.__name__} ({len(self._entries)} entries recovered)")

    # --- file helpers ---

    def _load(self) -> dict:
        return {"entries": list(self._entries)}

    def _checkpoint(self):
        tmp = self.stm_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"entries": self._entries}, f, ensure_ascii=False,
                      default=lambda o: list(o) if isinstance(o, tuple) else o)
        os.replace(tmp, self.stm_path)

    def _save(self, data: dict):
        """Compatibility shim — updates in-memory entries and checkpoints immediately."""
        self._entries = data.get("entries", [])
        self._checkpoint()

    # --- geometry helpers ---

    def _parse_6d(self, v) -> tuple:
        if isinstance(v, (list, tuple)) and len(v) == 6:
            try:
                return tuple(float(x) for x in v)
            except (TypeError, ValueError):
                pass
        return None

    def _dist(self, a, b) -> float:
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(6)))

    def _nearby(self, entries: list, coord: tuple, radius: float) -> list:
        results = []
        for e in entries:
            for key in ("inputPos", "responsePos"):
                p = self._parse_6d(e.get(key))
                if p is not None and self._dist(coord, p) <= radius:
                    results.append(e)
                    break
        return results

    # --- linkage builder (private, receives already-loaded entries) ---

    def _build_linkages(self, memory: dict, entries: list, radius: float) -> list:
        new_in  = self._parse_6d(memory.get("inputPos"))
        new_out = self._parse_6d(memory.get("responsePos"))
        if new_in is None or new_out is None:
            return []
        linked = []
        seen   = set()
        for e in entries:
            hit = False
            for key in ("inputPos", "responsePos"):
                p = self._parse_6d(e.get(key))
                if p is not None and (self._dist(new_in, p) <= radius or self._dist(new_out, p) <= radius):
                    hit = True
                    break
            if not hit:
                continue
            anchor = self._parse_6d(e.get("responsePos")) or self._parse_6d(e.get("inputPos"))
            if anchor and anchor not in seen:
                seen.add(anchor)
                linked.append(list(anchor))
        return linked

    # --- public API ---

    def addMemory(self, memory: dict) -> dict:
        """Add memory. Returns oldest entry to promote to LTM if over STM_MAX, else None."""
        with self._lock:
            if self._entries:
                memory["prevPos"] = self._entries[-1]["responsePos"]
            links = []
            for r in (self.STM_RADIUS, 0.15, 0.25, 0.40):
                links = self._build_linkages(memory, self._entries, r)
                if links:
                    break
            memory["linkedMemories"] = links
            self._entries.append(memory)
            self._ckpt_counter += 1
            if self._ckpt_counter >= self._ckpt_every:
                self._checkpoint()
                self._ckpt_counter = 0
            if len(self._entries) > self.STM_MAX:
                return self._entries.pop(0)
            return None

    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    def pop_oldest(self) -> dict:
        with self._lock:
            if not self._entries:
                return None
            oldest = self._entries.pop(0)
            self._checkpoint()
            return oldest

    def get_recent(self, n: int, metadata_filter: dict = None) -> list:
        with self._lock:
            entries = list(self._entries)
        if metadata_filter:
            entries = [e for e in entries if all(e.get("metaDataTag", {}).get(k) == v for k, v in metadata_filter.items())]
        return entries[-n:]

    def queryMemory(self, coord, radius=None) -> list:
        p = self._parse_6d(coord)
        if p is None:
            return []
        with self._lock:
            entries = list(self._entries)
        if radius is not None:
            return self._nearby(entries, p, radius)
        for r in (0.1, 0.2, 0.3, 0.5):
            results = self._nearby(entries, p, r)
            if results:
                return results
        return []


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
    ]

    _ms  = memorySpatial()
    _stm = shortTermMemory()

    # clear stm for a clean test
    _stm._save({"entries": []})

    print(f"\nInserting {len(PHRASES)} memories...")
    objects = []
    for phrase in PHRASES:
        obj = _ms.buildMemoryObject(
            inputText    = phrase,
            responseText = f"response: {phrase}",
            metaDataTag  = {"test": True},
        )
        _stm.addMemory(obj)
        objects.append(obj)
    print(f"Inserted {len(objects)} memories.\n")

    # --- exact retrieval test ---
    print("Exact retrieval (querying each responsePos, radius=0.0)...")
    hits, misses = 0, []
    for obj in objects:
        results = _stm.queryMemory(obj["responsePos"], radius=0.0)
        match = any(r["responseText"] == obj["responseText"] for r in results)
        if match:
            hits += 1
        else:
            misses.append(obj["inputText"])
    print(f"  Hits:   {hits} / {len(objects)}")
    print(f"  Misses: {len(misses)}")
    for m in misses:
        print(f"    - {m}")

    # --- radial neighbourhood test ---
    QUERIES = [
        "my body is in agony",
        "i feel pure joy and happiness",
        "grief has swallowed me whole",
        "i am filled with terror",
        "the system is broken and crashing",
    ]

    print(f"\n{'='*70}")
    print("RADIAL SEARCH  radius=0.3  — what lives near each query?")
    print(f"{'='*70}")
    for q in QUERIES:
        coord   = _ms._svc.computeSpatialValence(q)
        results = _stm.queryMemory(coord, radius=0.3)
        print(f"\n  QUERY: '{q}'")
        print(f"  coord: {coord}")
        print(f"  neighbors ({len(results)}):")
        for r in results:
            print(f"    - {r['inputText']}")
