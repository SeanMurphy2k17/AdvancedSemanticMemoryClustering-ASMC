"""
ADVANCED SEMANTIC MEMORY CLUSTERING — UNIFIED API

Thin backwards-compatible wrapper around the V2 backend:
    memoryManager → STM (JSON) + LTM (FAISS + LMDB) + SCM (anchor graph in LTM)

Public surface is identical to the previous API so existing consumers keep working.
New capabilities (anchor chains, anchor graph) are exposed as additional methods.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "V2"))

from memoryManager import memoryManager
from shortTermMemory import shortTermMemory


class AdvancedSemanticMemory:

    def __init__(self, max_stm_entries: int = 50, ltm_db_path: str = None,
                 verbose: bool = False, enable_scm: bool = True, defer_init: bool = False):
        self._mm           = memoryManager()
        self._verbose      = verbose
        self._anchor_cache = {}   # (cluster_id, coords_str) → ltm_id
        if verbose:
            print("Advanced Semantic Memory Clustering initialized")

    # --- internal helpers ---

    def _resolve_anchor(self, spatial_anchor) -> int:
        """
        Backwards compat: accepts either an int anchor_id or the old-style
        spatial_anchor dict (cluster_id + coordinates). Dict calls auto-create
        the anchor on first use and cache it for subsequent calls.
        """
        if isinstance(spatial_anchor, int):
            return spatial_anchor
        if not isinstance(spatial_anchor, dict):
            return None
        cluster_id  = spatial_anchor.get("cluster_id")
        coordinates = spatial_anchor.get("coordinates", {})
        if not cluster_id:
            return None
        cache_key = (cluster_id, str(sorted(coordinates.items())))
        if cache_key in self._anchor_cache:
            return self._anchor_cache[cache_key]
        location_type = (spatial_anchor.get("context_metadata") or {}).get("location_type", "")
        entities      = spatial_anchor.get("entities", [])
        ltm_id = self._mm._scm.createAnchor(
            cluster_id    = cluster_id,
            coords        = coordinates,
            location_type = location_type,
            entities      = entities,
        )
        self._anchor_cache[cache_key] = ltm_id
        if self._verbose:
            print(f"  [ASMC] created anchor id={ltm_id} for {cluster_id}/{coordinates}")
        return ltm_id

    # --- public API (backwards compatible) ---

    def add_experience(self, situation: str, response: str,
                       thought: str = "", objective: str = "",
                       action: str = "", result: str = "",
                       spatial_anchor=None, metadata: dict = None):
        """
        Store an experience. Writes to STM; oldest entry auto-promotes to LTM
        once STM_MAX is exceeded.

        spatial_anchor accepts:
            int  — a pre-created anchor LTM id (new style)
            dict — old-style {cluster_id, coordinates, context_metadata, entities}
                   anchor is auto-created on first call, cached thereafter
        """
        meta = dict(metadata or {})
        if thought:   meta["thought"]   = thought
        if objective: meta["objective"] = objective
        if action:    meta["action"]    = action
        if result:    meta["result"]    = result

        anchor_id = self._resolve_anchor(spatial_anchor) if spatial_anchor else None

        self._mm.addMemory(
            inputText    = situation,
            responseText = response,
            metaDataTag  = meta,
            anchor_id    = anchor_id,
        )
        return {"success": True, "anchor_id": anchor_id}

    def get_context(self, query: str, layer1_count: int = 6,
                    layer2_count: int = 6, complexity: int = 5) -> dict:
        """
        Retrieve layered context for a query.
            
        Returns:
            layer1        — STM results (recent + semantically close)
            layer2        — LTM memory results (semantic FAISS search)
            anchors       — LTM anchor nodes found in same FAISS search
            anchor_chain  — structurally adjacent anchors (one graph hop)
        """
        result = self._mm.queryMemory(query, k=layer2_count)
        return {
            "query":          query,
            "layer1":         result["stm"],
            "layer2":         result["ltm_memories"],
            "layer2_chain":   result["ltm_chain"],
            "semantic_chain": result["ltm_semantic"],
            "anchors":        result["ltm_anchors"],
            "anchor_chain":   result["anchor_chain"],
        }

    def get_raw_entries(self, n: int = None, metadata_filter: dict = None) -> list:
        """Return raw STM entries, optionally filtered and limited."""
        entries = self._mm._stm._load()["entries"]
        if metadata_filter:
            entries = [
                e for e in entries
                if all(e.get("metaDataTag", {}).get(k) == v
                       for k, v in metadata_filter.items())
            ]
        return entries[:n] if n else entries

    def get_statistics(self) -> dict:
        """Counts across all memory layers."""
        stm_count = self._mm._stm.count()
        anchors = memories = 0
        with self._mm._ltm._env.begin() as txn:
            for _, v in txn.cursor():
                entry = json.loads(v)
                if entry.get("metaDataTag", {}).get("type") == "scm_anchor":
                    anchors += 1
                else:
                    memories += 1
        return {
            "stm_entries":  stm_count,
            "stm_max":      shortTermMemory.STM_MAX,
            "ltm_memories": memories,
            "ltm_anchors":  anchors,
            "ltm_total":    memories + anchors,
        }

    def get_spatial_context(self, spatial_anchor, radius: int = 1) -> dict:
        """
        Retrieve anchor data + structurally adjacent anchors.
        spatial_anchor: int anchor_id or old-style dict.
        radius: how many graph hops to traverse (default 1).
        """
        ltm_id = self._resolve_anchor(spatial_anchor)
        if ltm_id is None:
            return {}
        anchor = self._mm._scm.getAnchor(ltm_id)
        if not anchor:
            return {}
        chain = []
        seen  = {ltm_id}
        queue = list(anchor.get("linked_anchors", []))
        for _ in range(radius):
            next_queue = []
            for linked_id in queue:
                if linked_id in seen:
                    continue
                seen.add(linked_id)
                neighbour = self._mm._ltm.fetchById(linked_id)
                if neighbour:
                    chain.append(neighbour)
                    next_queue.extend(neighbour.get("linked_anchors", []))
            queue = next_queue
        return {"anchor": anchor, "chain": chain}

    def get_spatial_context_string(self, spatial_anchor, radius: int = 1) -> str:
        """Human-readable spatial context."""
        ctx = self.get_spatial_context(spatial_anchor, radius)
        if not ctx:
            return "No anchor found."
        lines = []
        a    = ctx["anchor"]
        meta = a["metaDataTag"]
        lines.append(f"Anchor: {a['inputText']}  [{meta['cluster_id']}]  "
                     f"visits={meta['visit_count']}  "
                     f"valence={meta['aggregate_valence']:+.2f}")
        if ctx["chain"]:
            lines.append("Adjacent anchors:")
            for n in ctx["chain"]:
                nm = n["metaDataTag"]
                lines.append(f"  → {n['inputText']}  [{nm['cluster_id']}]")
        return "\n".join(lines)
    
    def create_anchor(self, cluster_id: str, coords: dict,
                      location_type: str = "", entities: list = None,
                      prev_anchor_id: int = None) -> int:
        """Create a permanent spatial anchor in LTM. Returns anchor LTM id."""
        ltm_id = self._mm._scm.createAnchor(cluster_id, coords, location_type,
                                             entities, prev_anchor_id)
        cache_key = (cluster_id, str(sorted(coords.items())))
        self._anchor_cache[cache_key] = ltm_id
        return ltm_id

    def link_anchors(self, id_a: int, id_b: int):
        """Bidirectional structural link between two anchors."""
        self._mm._scm.linkAnchors(id_a, id_b)

    def get_anchor(self, ltm_id: int) -> dict:
        return self._mm._scm.getAnchor(ltm_id)

    def clear_memory(self, confirm: bool = False) -> dict:
        if not confirm:
            return {"success": False,
                    "message": "Must set confirm=True to clear memory (DESTRUCTIVE)"}
        self._mm._stm._save({"entries": []})
        self._mm._ltm._clear()
        self._anchor_cache.clear()
        return {"success": True, "cleared": {"stm": True, "ltm": True}}
    
    def shutdown(self):
        pass  # LMDB environments close automatically on GC

    def MassDataUpload(self, folder_path: str,
                       file_extensions: list = None,
                       chunk_size: int = 300) -> dict:
        """
        Scan a folder and ingest all matching text files as memories.
        Each file is chunked into segments of chunk_size characters.
        """
        if file_extensions is None:
            file_extensions = [".txt", ".md"]
        uploaded = errors = 0
        for root, _, files in os.walk(folder_path):
            for fname in files:
                if not any(fname.endswith(ext) for ext in file_extensions):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    chunks = [text[i:i + chunk_size]
                              for i in range(0, len(text), chunk_size)]
                    for i, chunk in enumerate(chunks):
                        self.add_experience(
                            situation = chunk,
                            response  = f"[source: {fname} chunk {i+1}/{len(chunks)}]",
                            metadata  = {"source": fpath, "chunk": i},
                        )
                        uploaded += 1
                except Exception as e:
                    errors += 1
                    if self._verbose:
                        print(f"  [ASMC] error reading {fpath}: {e}")
        return {"files_processed": uploaded, "errors": errors}


def create_memory(max_entries: int = 50, db_path: str = None,
                  verbose: bool = False, defer_init: bool = False):
    """Factory function — drop-in replacement for old create_memory()."""
    return AdvancedSemanticMemory(
        max_stm_entries = max_entries,
        ltm_db_path     = db_path,
        verbose         = verbose,
        defer_init      = defer_init,
    )


if __name__ == "__main__":
    api = create_memory(verbose=True)
    api.clear_memory(confirm=True)

    # build anchors
    a_kitchen = api.create_anchor("home", {"room": "kitchen"}, "kitchen")
    a_living  = api.create_anchor("home", {"room": "living"},  "living room",
                                  prev_anchor_id=a_kitchen)
    api.link_anchors(a_kitchen, a_living)

    # add experiences
    api.add_experience("i am so hungry", "shall i make something", spatial_anchor=a_kitchen)
    api.add_experience("the meal was delicious", "glad you enjoyed it",   spatial_anchor=a_kitchen)
    api.add_experience("lets watch a film",      "great idea",            spatial_anchor=a_living)

    # old-style spatial_anchor dict (backwards compat)
    api.add_experience(
        "the sofa is comfortable",
        "good spot to relax",
        spatial_anchor={
            "cluster_id":       "home",
            "coordinates":      {"room": "living"},
            "context_metadata": {"location_type": "living room"},
        }
    )

    print("\n--- get_context ---")
    ctx = api.get_context("i am hungry", layer2_count=4)
    print(f"  layer1 ({len(ctx['layer1'])}):", [m['inputText'] for m in ctx['layer1']])
    print(f"  layer2 ({len(ctx['layer2'])}):", [m['inputText'] for m in ctx['layer2']])
    print(f"  anchors ({len(ctx['anchors'])}):", [a['inputText'] for a in ctx['anchors']])
    print(f"  chain ({len(ctx['anchor_chain'])}):", [a['inputText'] for a in ctx['anchor_chain']])

    print("\n--- get_spatial_context_string ---")
    print(api.get_spatial_context_string(a_kitchen))

    print("\n--- get_statistics ---")
    for k, v in api.get_statistics().items():
        print(f"  {k}: {v}")
