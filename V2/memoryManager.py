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
        print(f"initialized {self.__class__.__name__}")

    def addMemory(self, inputText: str, responseText: str,
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

    def queryMemory(self, text: str, k: int = 10) -> dict:
        import time as _t
        _t0 = _t.perf_counter()
        print(f"[MEMORY QUERY] Step 1: converting query ({len(text.split())} words) to coordinates...", flush=True)
        coord      = self._spatial._svc.computeSpatialValence(text)
        print(f"[MEMORY QUERY] Step 2: searching recent memory... (step 1 took {_t.perf_counter()-_t0:.2f}s)", flush=True)
        _t1 = _t.perf_counter()
        stm_result = self._stm.queryMemory(coord)
        print(f"[MEMORY QUERY] Step 3: searching long-term memory... (step 2 took {_t.perf_counter()-_t1:.2f}s)", flush=True)
        _t2 = _t.perf_counter()
        ltm_result = self._ltm.queryMemory(coord, k=k)
        print(f"[MEMORY QUERY] done. Total query time: {_t.perf_counter()-_t0:.2f}s", flush=True)

        memories = [m for m in ltm_result["direct"] if not self._scm.isAnchor(m)]
        anchors  = [m for m in ltm_result["direct"] if self._scm.isAnchor(m)]
        chain    = [m for m in ltm_result["chain"]  if not self._scm.isAnchor(m)]

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

        return {
            "stm":          stm_result,
            "ltm_memories": memories,
            "ltm_chain":    chain,
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
