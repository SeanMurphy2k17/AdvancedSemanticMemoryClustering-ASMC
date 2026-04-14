from datetime import datetime, timezone
from spatialValenceCompute import spatialValenceCompute


class spatialCompMap:
    """
    SCM anchor factory.
    Anchors are permanent fixtures stored directly in LTM (FAISS + LMDB).
    They act as gravitational nodes — memories cluster around them in semantic
    space and they link to adjacent anchors forming a navigational graph.

    Anchor memory object layout:
        inputText      = location_type label
        inputPos       = node_pos  (stable 6D semantic coord)
        responseText   = location_type label
        responsePos    = node_pos  (indexed in LTM FAISS)
        linkedMemories = []        (not used — memories find anchors via FAISS)
        linked_anchors = [ltm_id, ...]  structural graph edges
        metaDataTag    = { type, cluster_id, coords, entities,
                           visit_count, valence_sum, aggregate_valence }
        timeDate       = creation timestamp
    """

    def __init__(self, ltm):
        self._ltm = ltm
        self._svc = spatialValenceCompute()
        print(f"initialized {self.__class__.__name__}")

    # --- public API ---

    def createAnchor(self, cluster_id: str, coords: dict,
                     location_type: str = "",
                     entities: list = None,
                     prev_anchor_id: int = None) -> int:
        """
        Create a new anchor and store it permanently in LTM.
        Returns the assigned LTM integer ID.
        """
        node_pos = list(self._svc.computeSpatialValence(location_type) if location_type else [0.0] * 6)
        now      = datetime.now(timezone.utc).isoformat()

        anchor = {
            "inputText":       location_type,
            "inputPos":        node_pos,
            "responseText":    location_type,
            "responsePos":     node_pos,
            "prevPos":         None,
            "linkedMemories":  [],
            "linked_anchors":  [],
            "metaDataTag": {
                "type":               "scm_anchor",
                "cluster_id":         cluster_id,
                "coords":             coords,
                "entities":           entities or [],
                "visit_count":        0,
                "valence_sum":        0.0,
                "aggregate_valence":  0.0,
            },
            "timeDate": now,
        }

        if prev_anchor_id is not None:
            prev = self._ltm.fetchById(prev_anchor_id)
            if prev:
                anchor["prevPos"] = prev["responsePos"]

        ltm_id = self._ltm.addMemory(anchor)
        anchor["metaDataTag"]["ltm_id"] = ltm_id   # stamp own ID for chain traversal
        self._ltm.updatePayload(ltm_id, anchor)

        if prev_anchor_id is not None:
            self.linkAnchors(prev_anchor_id, ltm_id)

        return ltm_id

    def linkAnchors(self, ltm_id_a: int, ltm_id_b: int):
        """Bidirectional structural link between two anchors."""
        for src, dst in [(ltm_id_a, ltm_id_b), (ltm_id_b, ltm_id_a)]:
            data = self._ltm.fetchById(src)
            if data and dst not in data.get("linked_anchors", []):
                data.setdefault("linked_anchors", []).append(dst)
                self._ltm.updatePayload(src, data)

    def visitAnchor(self, ltm_id: int, valence: float = 0.0):
        """Record a visit — update visit_count and rolling valence."""
        data = self._ltm.fetchById(ltm_id)
        if not data:
            return
        meta = data["metaDataTag"]
        meta["visit_count"]       += 1
        meta["valence_sum"]       += valence
        meta["aggregate_valence"]  = meta["valence_sum"] / meta["visit_count"]
        self._ltm.updatePayload(ltm_id, data)

    def getAnchor(self, ltm_id: int) -> dict:
        return self._ltm.fetchById(ltm_id)

    def isAnchor(self, memory: dict) -> bool:
        return memory.get("metaDataTag", {}).get("type") == "scm_anchor"


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from longTermMemory import longTermMemory
    from memorySpatial import memorySpatial

    _ltm = longTermMemory()
    _ltm._clear()
    _scm = spatialCompMap(_ltm)
    _ms  = memorySpatial()

    # build a linear anchor chain: reception → ward_7 → surgery → icu
    print("\nCreating anchor chain...")
    a_reception = _scm.createAnchor("hospital", {"room": "reception"}, "reception area")
    a_ward      = _scm.createAnchor("hospital", {"room": "ward_7"},    "ward seven",
                                    prev_anchor_id=a_reception)
    a_surgery   = _scm.createAnchor("hospital", {"room": "surgery"},   "surgery theatre",
                                    prev_anchor_id=a_ward)
    a_icu       = _scm.createAnchor("hospital", {"room": "icu"},       "intensive care unit",
                                    prev_anchor_id=a_surgery)

    # also link ward directly to icu (shortcut — staff move between them often)
    _scm.linkAnchors(a_ward, a_icu)

    for aid, label in [(a_reception,"reception"), (a_ward,"ward_7"),
                       (a_surgery,"surgery"), (a_icu,"icu")]:
        anchor = _scm.getAnchor(aid)
        meta   = anchor["metaDataTag"]
        print(f"  id={aid:2d}  {label:12s}  node_pos={anchor['inputPos']}  "
              f"linked_anchors={anchor['linked_anchors']}")

    # add some memories near ward_7 (same semantic space as "ward seven")
    print("\nAdding memories near ward_7...")
    ward_anchor = _scm.getAnchor(a_ward)
    WARD_MEMS = [
        ("she is running a high fever",       "we will get her on fluids"),
        ("blood pressure is dropping fast",   "prepare the crash cart"),
        ("she is going into cardiac arrest",  "starting compressions now"),
    ]
    for inp, resp in WARD_MEMS:
        mem = _ms.buildMemoryObject(inputText=inp, responseText=resp,
                                    metaDataTag={"scm_anchor_id": a_ward})
        _ltm.addMemory(mem)

    # record visits
    _scm.visitAnchor(a_ward, valence=-0.6)
    _scm.visitAnchor(a_ward, valence=-0.8)
    updated = _scm.getAnchor(a_ward)
    print(f"  ward_7 visits={updated['metaDataTag']['visit_count']}  "
          f"valence={updated['metaDataTag']['aggregate_valence']:+.2f}")

    # query near "cardiac arrest chest pain" — should surface anchor + memories
    print(f"\nQuerying 'cardiac arrest chest pain' (k=6)...")
    result = _ltm.queryMemory(_ms._svc.computeSpatialValence("cardiac arrest chest pain"), k=6)
    for m in result["direct"]:
        t = m["metaDataTag"].get("type", "memory")
        print(f"  [{t:10s}] {m['inputText']}  linked_anchors={m.get('linked_anchors', [])}")
