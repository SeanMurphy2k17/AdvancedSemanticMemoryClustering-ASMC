import json
import threading
from memorySpatial import memorySpatial
from shortTermMemory import shortTermMemory
from longTermMemory import longTermMemory
from spatialCompMap import spatialCompMap


class memoryManager:
    def __init__(self):
        self._spatial = memorySpatial()
        self._stm     = shortTermMemory()
        self._ltm     = longTermMemory()
        self._scm     = spatialCompMap(self._ltm)
        # One lock around the two entry points. Callers that use threads raced
        # queryMemory against addMemory over a FAISS index that has no locking
        # of its own; an add/write racing a search is native undefined
        # behaviour and showed up as a segfault.
        self._lock = threading.RLock()
        self._drain_overflow()
        print(f"initialized {self.__class__.__name__}")

    def _ltm_has(self, entry: dict) -> bool:
        key = (entry.get("inputText") or "", entry.get("timeDate") or "")
        with self._ltm._env.begin() as txn:
            for k, v in txn.cursor():
                if not k.isdigit():
                    continue
                try:
                    m = json.loads(v)
                except ValueError:
                    continue
                if (m.get("inputText") or "", m.get("timeDate") or "") == key:
                    return True
        return False

    def _drain_overflow(self) -> None:
        """A short-term file left by an unclean death can hold more entries than
        the window allows (and, before checkpoint_if_due, an entry already
        promoted). Promote the extras on open, skipping ones long-term memory
        already holds, so the window never stays over its maximum."""
        promoted = skipped = 0
        while self._stm.count() > self._stm.STM_MAX:
            oldest = self._stm.pop_oldest()
            if oldest is None:
                break
            if self._ltm_has(oldest):
                skipped += 1
            else:
                self._ltm.addMemory(oldest)
                promoted += 1
        if promoted or skipped:
            self._stm._checkpoint()
            print(f"[STM] window drained on open: promoted={promoted} already_in_ltm={skipped}", flush=True)

    def get_recent(self, n: int, metadata_filter: dict = None) -> list:
        return self._stm.get_recent(n, metadata_filter=metadata_filter)

    def addMemory(self, inputText: str, responseText: str,
                  metaDataTag: dict = None, anchor_id: int = None):
        with self._lock:
            return self._addMemory(inputText, responseText, metaDataTag, anchor_id)

    def _addMemory(self, inputText: str, responseText: str,
                   metaDataTag: dict = None, anchor_id: int = None):
        meta   = dict(metaDataTag or {})
        memory = self._spatial.buildMemoryObject(
            inputText    = inputText,
            responseText = responseText,
            metaDataTag  = meta,
        )
        if anchor_id is not None:
            valence = memory["inputPos"][1]
            self._scm.visitAnchor(anchor_id, valence)
            memory["metaDataTag"]["scm_anchor_id"] = anchor_id

        to_promote = self._stm.addMemory(memory)
        if to_promote:
            self._ltm.addMemory(to_promote)
        # After the long-term write, and always on a promotion: the window file
        # then never names an entry long-term memory already holds, whatever
        # happens to the process next.
        self._stm.checkpoint_if_due(force=to_promote is not None)

    def queryMemory(self, text: str, k: int = 10, layer1_count: int = 6,
                    platform_tag: str = None, temp: float = 0.5) -> dict:
        with self._lock:
            return self._queryMemory(text, k, layer1_count, platform_tag, temp)

    def _queryMemory(self, text: str, k: int = 10, layer1_count: int = 6,
                     platform_tag: str = None, temp: float = 0.5) -> dict:
        import time as _t
        import threading as _th
        _tid = _th.current_thread().name
        _t0 = _t.perf_counter()
        print(f"[MEMORY QUERY:{_tid}] Step 1: converting query ({len(text.split())} words) to coordinates...", flush=True)
        coord      = self._spatial._svc.computeSpatialValence(text)
        print(f"[MEMORY QUERY:{_tid}] Step 2: searching recent memory... (step 1 took {_t.perf_counter()-_t0:.2f}s)", flush=True)
        _t1 = _t.perf_counter()
        stm_spatial = self._stm.queryMemory(coord)
        stm_recent  = self._stm.get_recent(layer1_count)
        # Exact-word overlap inside the short-term window too. The inverted index
        # covers long-term memory only, so without this a thing said an hour ago
        # could be found only by valence proximity. Words are weighted by how
        # few window entries contain them, so "do" does not outrank "faiss".
        stm_overlap = []
        q_words = set(self._spatial._svc.extractContentWords(text))
        if q_words:
            window = self._stm.get_recent(self._stm.STM_MAX)
            df = {}
            for m in window:
                for w in set(m.get("contentWords") or []) | set(m.get("responseWords") or []):
                    df[w] = df.get(w, 0) + 1
            scored = []
            for m in window:
                words = set(m.get("contentWords") or []) | set(m.get("responseWords") or [])
                score = sum(1.0 / df[w] for w in q_words & words)
                if score > 0:
                    scored.append((score, m))
            scored.sort(key=lambda t: -t[0])
            stm_overlap = [m for _, m in scored[:layer1_count]]
        seen        = {id(m) for m in stm_overlap}
        stm_result  = stm_overlap + [m for m in stm_spatial if id(m) not in seen]
        seen       |= {id(m) for m in stm_spatial}
        stm_result += [m for m in stm_recent if id(m) not in seen]
        print(f"[MEMORY QUERY:{_tid}] Step 3: searching long-term memory... (step 2 took {_t.perf_counter()-_t1:.2f}s)", flush=True)
        _t2 = _t.perf_counter()

        # Entity overlap channel — look up named entities from query text
        _t3 = _t.perf_counter()
        entity_ids = set()
        entity_results = self._ltm.resolveEntities(text)
        for entity_key, mem_ids in entity_results.items():
            for mid in mem_ids:
                entity_ids.add(mid)
        print(f"[MEMORY QUERY:{_tid}] entity overlap: {len(entity_ids)} candidate IDs from {len(entity_results)} entities ({_t.perf_counter()-_t3:.2f}s)", flush=True)

        # Content word overlap channel — match significant words from query
        _t4 = _t.perf_counter()
        content_words = self._spatial._svc.extractContentWords(text)
        content_id_results = self._ltm.resolveContentWords(content_words)
        content_ids = set(content_id_results.keys())
        print(f"[MEMORY QUERY:{_tid}] content overlap: {len(content_ids)} candidate IDs from {len(content_words)} words ({_t.perf_counter()-_t4:.2f}s)", flush=True)

        # --- Dynamic temp: compute HHI from recent STM actions ---
        # HHI measures action concentration in recent memory (0.0 = diverse, 1.0 = monopolized)
        # temp controls how much HHI shifts retrieval from entity-precision to semantic-drift
        hhi = 0.0
        has_actions = False
        if temp > 0.0:
            recent = self._stm.get_recent(10)
            actions = []
            for m in recent:
                meta = m.get("metaDataTag", {})
                action = meta.get("action", "")
                if action:
                    actions.append(action.split()[0] if action else "")
            if actions:
                has_actions = True
                from collections import Counter
                counts = Counter(actions)
                total = len(actions)
                hhi = sum(c * c for c in counts.values()) / (total * total)
                print(f"[MEMORY QUERY:{_tid}] HHI={hhi:.3f} temp={temp} ({len(actions)} actions, {len(counts)} unique)", flush=True)

        # Apply temp-weighted channel balance
        # No actions + high temp = full semantic drift (idle agent should wander)
        # Has actions + high temp + high HHI = drift (bored, repetitive)
        # Has actions + high temp + low HHI = precision (active, diverse)
        final_entity_ids = entity_ids  # default: use computed entity_ids
        if temp > 0.0:
            if not has_actions:
                # Idle agent — default to full semantic drift, skip entity channel
                entity_frac   = 1.0 - temp
                semantic_frac = temp
                if semantic_frac > 0.5:
                    final_entity_ids = None  # disable entity channel entirely
            else:
                entity_frac   = 1.0 - hhi * temp
                semantic_frac = hhi * temp
                if semantic_frac > 0.5:
                    # Dilute entity_ids when semantic weight is high
                    if entity_ids:
                        entity_list = list(entity_ids)
                        keep = max(1, int(len(entity_list) * entity_frac))
                        final_entity_ids = set(entity_list[:keep])
            print(f"[MEMORY QUERY:{_tid}] entity_frac={entity_frac:.2f} semantic_frac={semantic_frac:.2f} has_actions={has_actions} entity_ids={'DISABLED' if final_entity_ids is None else len(final_entity_ids)}", flush=True)

        # Merge both overlap channels into one ranked list: content-word hits
        # first (IDF-ranked, exact words), entity hits after. The entity tagger is
        # noisy on chat text ("alright" as a location), so its hits should not take
        # slots ahead of a rare-word match. The LTM query keeps this order and
        # interleaves it with the FAISS results instead of appending leftovers.
        ranked = list(content_id_results)
        ranked += [mid for mid in sorted(final_entity_ids or []) if mid not in ranked]
        _t5 = _t.perf_counter()
        syn_coord = self._spatial._svc.computeWorldValence(text)
        print(f"[MEMORY QUERY:{_tid}] computeWorldValence: {_t.perf_counter()-_t5:.2f}s", flush=True)
        _t6 = _t.perf_counter()
        ltm_result = self._ltm.queryMemory(coord, k=k,
                         syn_coord=syn_coord,
                         entity_ids=ranked or None)
        print(f"[MEMORY QUERY:{_tid}] ltm.queryMemory call: {_t.perf_counter()-_t6:.2f}s", flush=True)
        print(f"[MEMORY QUERY:{_tid}] done. Total query time: {_t.perf_counter()-_t0:.2f}s", flush=True)

        memories = [m for m in ltm_result["direct"] if not self._scm.isAnchor(m)]
        anchors  = [m for m in ltm_result["direct"] if self._scm.isAnchor(m)]
        chain    = [m for m in ltm_result["chain"]  if not self._scm.isAnchor(m)]
        query_words   = content_words  # already computed above — same text
        cursor_entry  = self._ltm.fetch_platform_cursor(platform_tag) if platform_tag else None
        seed          = ([cursor_entry] + ltm_result["direct"]) if cursor_entry else ltm_result["direct"]
        semantic_raw  = self._ltm.semanticTraverse(
            seed, query_words, self._spatial._svc.extractContentWords)
        semantic      = [m for m in semantic_raw if not self._scm.isAnchor(m)]
        if cursor_entry and not self._scm.isAnchor(cursor_entry):
            memories = [cursor_entry] + [m for m in memories if m.get("inputText") != cursor_entry.get("inputText")]

        # traverse anchor graph — one hop from each found anchor
        anchor_chain = []
        seen_ids     = {a["metaDataTag"]["ltm_id"] for a in anchors
                        if "ltm_id" in a.get("metaDataTag", {})}
        for anchor in anchors:
            for linked_id in anchor.get("linked_anchors", []):
                if linked_id in seen_ids:
                    continue
                seen_ids.add(linked_id)
                neighbour = self._ltm.fetchById(linked_id)
                if neighbour:
                    anchor_chain.append(neighbour)

        # Debug: show what each channel returned
        print(f"[MEMORY QUERY:{_tid}] RESULTS — stm:{len(stm_result)} ltm:{len(memories)} chain:{len(chain)} semantic:{len(semantic)} anchors:{len(anchors)} anchor_chain:{len(anchor_chain)}", flush=True)
        for i, m in enumerate(memories[:3]):
            meta = m.get("metaDataTag", {})
            print(f"  ltm[{i}] platform={meta.get('platform','?')} action={meta.get('action','')[:40]} input={m.get('inputText','')[:60]}", flush=True)
        for i, m in enumerate(semantic[:3]):
            meta = m.get("metaDataTag", {})
            print(f"  semantic[{i}] platform={meta.get('platform','?')} action={meta.get('action','')[:40]} input={m.get('inputText','')[:60]}", flush=True)

        return {
            "stm":          stm_result,
            "ltm_memories": memories,
            "ltm_chain":    chain,
            "ltm_semantic": semantic,
            "ltm_anchors":  anchors,
            "anchor_chain": anchor_chain,
        }


if __name__ == "__main__":
    MEMORIES = [
        ("my back is absolutely killing me",       "that sounds really painful",          None),
        ("i have a terrible headache",             "let me check your vitals",            None),
        ("my feet are sore and aching",            "have you been standing all day",      None),
        ("i am so incredibly happy today",         "that is wonderful to hear",           None),
        ("this is the best day of my life",        "tell me everything about it",         None),
        ("everything feels wonderful and bright",  "you seem really energised",           None),
        ("i feel completely broken and hollow",    "i am here, take your time",           None),
        ("i lost someone i truly loved",           "grief takes time, be kind to yourself",None),
        ("the sadness is slowly consuming me",     "let us talk through what you feel",   None),
        ("i cannot stop crying",                   "that is okay, let it out",            None),
        ("i am terrified and i cannot move",       "breathe slowly with me",              None),
        ("something is very wrong and i am scared","tell me what happened",               None),
        ("i am starving and i want pizza",         "shall i order some for you",          None),
        ("the meal was absolutely delicious",      "glad you enjoyed it",                 None),
        ("the server crashed and lost all data",   "do you have a backup",                None),
        ("the algorithm runs in polynomial time",  "that should be efficient enough",     None),
        ("the api keeps returning timeout errors", "check the connection pool settings",  None),
        ("we deployed the build to production",    "monitoring looks stable",             None),
        ("the sunset was breathtaking and vivid",  "nature has a way of doing that",      None),
        ("the mountains looked ancient and vast",  "they make everything feel small",     None),
        ("the ocean stretched on forever",         "the horizon is humbling",             None),
        ("i cannot sleep and it is three am",      "let us try a breathing exercise",     None),
        ("exhaustion has settled into my bones",   "your body needs proper rest",         None),
        ("the idea clicked all at once",           "insight can come at any moment",      None),
        ("suddenly everything made perfect sense", "keep that thread of thought",         None),
        ("she forgave him after years of silence", "forgiveness is for the forgiver too", None),
        ("they reconciled over a long dinner",     "shared meals heal a lot",             None),
        ("the apology came too late to matter",    "the hurt is still valid",             None),
        ("some wounds do not heal easily",         "time and care are the only tools",    None),
        ("we had been overthinking it entirely",   "simplicity was always the answer",    None),
        ("the experiment failed again",            "failure is data, keep going",         None),
        ("the hypothesis was proven wrong",        "revise and retest",                   None),
        ("she laughed until her sides ached",      "joy is the best medicine",            None),
        ("we danced all night long",               "sounds like a memorable night",       None),
        ("the children played in the garden",      "it is good to see them happy",        None),
        ("he smiled and everything was okay",      "sometimes a smile says it all",       None),
        ("the diagnosis was worse than expected",  "we will work through this together",  None),
        ("the news hit like a freight train",      "take a moment to breathe",            None),
        ("everything i built has fallen apart",    "we can rebuild from here",            None),
        ("i have nothing left to hold on to",      "you still have yourself",             None),
        ("the engine overheated and stalled",      "let it cool before restarting",       None),
        ("the brakes failed on the highway",       "are you safe right now",              None),
        ("i wrote a poem about the rain",          "i would love to hear it",             None),
        ("the painting took three weeks to finish","patience shows in the result",        None),
        ("the music filled the whole room",        "music has a way of changing the air", None),
        ("she sang and the crowd fell silent",     "that must have been extraordinary",   None),
        ("i cannot understand this concept",       "let us work through it together",     None),
        ("nothing makes sense to me anymore",      "that feeling is temporary",           None),
        ("i am completely lost and confused",      "let us find the thread together",     None),
        ("my muscles throb from the workout",      "rest and hydration will help",        None),
        ("my heart is overflowing with gratitude", "that is a beautiful feeling",         None),
    ]

    _mm = memoryManager()
    _mm._stm._save({"entries": []})
    _mm._ltm._clear()

    # build hospital anchor chain (anchors live in LTM immediately)
    print("\nBuilding anchor chain...")
    a_reception = _mm._scm.createAnchor("hospital", {"room": "reception"}, "reception area")
    a_ward      = _mm._scm.createAnchor("hospital", {"room": "ward"},      "ward seven",
                                        prev_anchor_id=a_reception)
    a_surgery   = _mm._scm.createAnchor("hospital", {"room": "surgery"},   "surgery theatre",
                                        prev_anchor_id=a_ward)
    a_icu       = _mm._scm.createAnchor("hospital", {"room": "icu"},       "intensive care unit",
                                        prev_anchor_id=a_surgery)
    _mm._scm.linkAnchors(a_ward, a_icu)    # shortcut: ward ↔ icu

    for aid, label in [(a_reception,"reception"),(a_ward,"ward"),(a_surgery,"surgery"),(a_icu,"icu")]:
        a = _mm._scm.getAnchor(aid)
        print(f"  id={aid}  {label:10s}  linked_anchors={a['linked_anchors']}")

    # push regular memories — some tagged to ward anchor
    print(f"\nPushing {len(MEMORIES)} memories...")
    ward_mems = [
        ("she is running a high fever",       "we will get her on fluids"),
        ("blood pressure is dropping fast",   "prepare the crash cart"),
        ("she is going into cardiac arrest",  "starting compressions now"),
        ("the patient is stabilising",        "keep monitoring closely"),
        ("she is asking for water",           "small sips only"),
    ]
    for inp, resp in ward_mems:
        _mm.addMemory(inp, resp, anchor_id=a_ward)

    for inp, resp, _ in MEMORIES:
        _mm.addMemory(inp, resp)

    stm_c = _mm._stm.count()
    ltm_c = _mm._ltm._index.ntotal
    print(f"  STM={stm_c}  LTM={ltm_c}  (anchors count in LTM)")

    QUERIES = [
        ("cardiac emergency chest pain",   "should surface ward anchor + chain to icu/surgery"),
        ("i feel pure joy",                "should surface casual memories"),
        ("grief and loss",                 "should surface therapy memories"),
    ]

    print(f"\n{'='*70}")
    print("QUERY TEST — memories + anchors + anchor chain")
    print(f"{'='*70}")
    for q, note in QUERIES:
        result = _mm.queryMemory(q, k=4)
        print(f"\n  QUERY: '{q}'")
        print(f"  ({note})")
        print(f"  STM ({len(result['stm'])}):")
        for m in result["stm"]:
            print(f"    - {m['inputText']}")
        print(f"  LTM memories ({len(result['ltm_memories'])}):")
        for m in result["ltm_memories"]:
            print(f"    - {m['inputText']}")
        print(f"  LTM chain ({len(result['ltm_chain'])}):")
        for m in result["ltm_chain"]:
            print(f"    ~ {m['inputText']}")
        print(f"  LTM semantic ({len(result['ltm_semantic'])}):")
        for m in result["ltm_semantic"]:
            print(f"    * {m['inputText']}")
        print(f"  LTM anchors ({len(result['ltm_anchors'])}):")
        for a in result["ltm_anchors"]:
            meta = a["metaDataTag"]
            print(f"    - [{meta['cluster_id']}/{meta['coords']}] {a['inputText']}  "
                  f"visits={meta['visit_count']}  valence={meta['aggregate_valence']:+.2f}")
        if result["anchor_chain"]:
            print(f"  Anchor chain ({len(result['anchor_chain'])}) — adjacent nodes:")
            for a in result["anchor_chain"]:
                meta = a["metaDataTag"]
                t    = meta.get("type", "memory")
                print(f"    - [{t}] {a['inputText']}")
