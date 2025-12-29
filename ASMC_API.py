#!/usr/bin/env python3
"""
🧠 ADVANCED SEMANTIC MEMORY CLUSTERING - UNIFIED API 🧠

CREATORS:
- Sean Murphy (Human Inventor & System Architect)
- Claude AI Models (AI Co-Inventor & Implementation Partner)

A complete two-layer semantic memory system combining:
- SpatialValienceToCoords: 9D semantic coordinate generation with NLTK SentiWordNet
- Short-Term Memory: Fast RAM-based storage with semantic clustering
- Long-Term Memory: Persistent LMDB storage with spatial linking

FEATURES:
- Two-layer context retrieval (immediate + depth)
- 117k word sentiment analysis via SentiWordNet
- 9D spatial semantic clustering
- Automatic STM → LTM promotion
- Zero-shot semantic understanding

License: MIT
Copyright (c) 2024 Sean Murphy
"""

import os
import sys

# Add submodule paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'SpatialShortTermContextAIMemory'))

from STM_API import create_stm_api
from .SpatialComprehensionMap import create_scm

class AdvancedSemanticMemory:
    """
    🧠 ADVANCED SEMANTIC MEMORY CLUSTERING
    
    Unified interface for two-layer semantic memory system
    """
    
    def __init__(self, max_stm_entries: int = 50, ltm_db_path: str = None, verbose: bool = False, enable_scm: bool = True):
        """
        Initialize the Advanced Semantic Memory Clustering system
        
        Args:
            max_stm_entries: Maximum short-term memory entries (default: 50)
            ltm_db_path: OPTIONAL custom LTM path (default: auto-managed in MemoryStructures/)
            verbose: Enable detailed logging (default: False)
            enable_scm: Enable Spatial Comprehension Map integration (default: True)
        """
        self.max_stm_entries = max_stm_entries
        self.verbose = verbose
        self.enable_scm = enable_scm
        
        # ASMC owns its storage - all under MemoryStructures/
        asmc_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_memory_path = os.path.join(asmc_dir, "MemoryStructures")
        
        # Clean, generic naming - no app-specific prefixes
        self.stm_path = os.path.join(self.base_memory_path, "STM")
        self.ltm_db_path = ltm_db_path if ltm_db_path else os.path.join(self.base_memory_path, "LTM", "ltm.lmdb")
        self.scm_path = os.path.join(self.base_memory_path, "SCM", "scm.lmdb")
        self.scm_log_path = os.path.join(self.base_memory_path, "SCM", "scm_operations.log")
        
        # Ensure directories exist
        os.makedirs(self.stm_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.ltm_db_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.scm_path), exist_ok=True)
        
        # Initialize STM (which handles SVC and LTM internally)
        self._stm_api = create_stm_api(
            max_entries=max_stm_entries,
            save_interval=30,
            data_directory=self.stm_path,
            ltm_db_path=self.ltm_db_path,
            verbose=verbose
        )
        
        # Initialize SCM (Spatial Comprehension Map)
        self.scm = None
        if enable_scm:
            self.scm = create_scm(db_path=self.scm_path, verbose=verbose)
        
        if verbose:
            print("🧠 Advanced Semantic Memory Clustering initialized!")
            print(f"   Memory Base: {self.base_memory_path}")
            print(f"   STM: {self.stm_path}")
            print(f"   LTM: {self.ltm_db_path}")
            print(f"   Features: Two-layer retrieval, 9D clustering, 117k word sentiment")
            if enable_scm:
                print(f"   SCM: {self.scm_path}")
    
    def _log_scm_operation(self, operation_type: str, details: dict):
        """Log SCM operations to file for diagnostics"""
        if not self.enable_scm:
            return
        try:
            os.makedirs(os.path.dirname(self.scm_log_path), exist_ok=True)
            with open(self.scm_log_path, 'a', encoding='utf-8') as f:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                f.write(f"[{timestamp}] {operation_type}\n")
                for key, value in details.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
        except Exception as e:
            if self.verbose:
                print(f"[ASMC] SCM log error: {e}")
    
    def add_experience(self, situation: str, response: str, 
                      thought: str = "", objective: str = "", action: str = "", result: str = "",
                      spatial_anchor: dict = None, metadata: dict = None):
        """
        Add an experience to memory (stores in STM, auto-promotes to LTM)
        
        Args:
            situation: The situation/context (e.g., sensor data, user input)
            response: The response/thought (e.g., LLM output, action taken)
            thought: Separated thought output from LLM
            objective: Separated objective output from LLM
            action: Action taken by agent
            result: Result of action
            spatial_anchor: Optional spatial context (dict with structure_type, cluster_id, coordinates, entities)
            metadata: Optional metadata dictionary
            
        Returns:
            Dict: Storage result with coordinate information
        """
        # Generate coordinate directly for SCM (bypass STM return issues)
        from spatial_valence import UltraEnhancedSpatialValenceToCoordGeneration
        coord_gen = UltraEnhancedSpatialValenceToCoordGeneration()
        full_context = f"User: {situation}\nAI: {response}"
        coord_result = coord_gen.process(full_context)
        coord_key = coord_result.get('coordinate_key')
        print(f"[ASMC DEBUG] Generated coord_key: {coord_key}")
        print(f"[ASMC DEBUG] spatial_anchor present: {spatial_anchor is not None}")
        
        # Still store in STM (let it do its internal thing)
        stm_result = self._stm_api.add_conversation(
            user_message=situation,
            ai_response=response,
            thought=thought,
            objective=objective,
            action=action,
            result=result,
            metadata=metadata
        )
        # SCM Integration: If spatial anchor provided, link memory to location
        if self.scm and spatial_anchor and coord_key:
            try:
                # Extract spatial info
                structure_type = spatial_anchor.get('structure_type', 'spatial')
                cluster_id = spatial_anchor.get('cluster_id')
                coordinates = spatial_anchor.get('coordinates', {})
                location_type = spatial_anchor.get('context_metadata', {}).get('location_type', '')
                entities = spatial_anchor.get('entities', [])
                neighbors = spatial_anchor.get('neighbors', {})
                
                if not cluster_id:
                    return result  # Skip if no cluster specified
                
                # Ensure cluster exists
                if not self.scm.cluster_exists(cluster_id):
                    self.scm.create_cluster(
                        cluster_id=cluster_id,
                        cluster_type=structure_type,
                        description=spatial_anchor.get('context_metadata', {}).get('description', '')
                    )
                
                # Create/update node with physical structure
                node_key = self.scm.create_or_update_node(
                    structure_type=structure_type,
                    cluster_id=cluster_id,
                    coordinates=coordinates,
                    location_type=location_type,
                    entities=entities,
                    neighbors=neighbors
                )
                
                # Record visit
                self.scm.visit_node(structure_type, cluster_id, coordinates)
                
                # Extract valence from ASMC sentiment
                valence = self._extract_valence_from_stm(coord_key)
                
                # Link STM memory to SCM node
                self.scm.link_stm_memory(node_key, coord_key, valence)
                
                # Log SCM storage operation
                self._log_scm_operation("SCM_STORE", {
                    "node_key": node_key,
                    "cluster_id": cluster_id,
                    "coordinates": coordinates,
                    "location_type": location_type,
                    "coord_key": coord_key,
                    "valence": f"{valence:.3f}",
                    "entities": str(entities)
                })
                
                # Add SCM info to coord_result
                coord_result['scm_node_key'] = node_key
                coord_result['scm_valence'] = valence
                
            except Exception as e:
                if self.verbose:
                    print(f"   [ASMC] Warning: SCM linking failed: {e}")
        
        # Return our coordinate result
        return {
            'success': True,
            'coordinate_key': coord_key,
            'coordinates': coord_result.get('coordinates'),
            'summary': coord_result.get('summary'),
            'scm_node_key': coord_result.get('scm_node_key'),
            'scm_valence': coord_result.get('scm_valence')
        }
    
    def get_context(self, query: str, layer1_count: int = 6, layer2_count: int = 6):
        """
        Get two-layer context for a query
        
        Layer 1 (Immediate): Recent conversations + STM relevant matches
        Layer 2 (Depth): LTM semantic associations + spatial neighbors
        
        Args:
            query: Query text to build context for
            layer1_count: Items from Layer 1 (split: recent + relevant)
            layer2_count: Items from Layer 2 (split: semantic + neighbors)
            
        Returns:
            Dict with layer1_immediate and layer2_depth context
        """
        # Layer 1 split: 50/50 between recent and relevant
        recent_count = layer1_count // 2
        relevant_count = layer1_count - recent_count
        
        return self._stm_api.get_context(
            user_input=query,
            recent_count=recent_count,
            relevant_count=relevant_count
        )
    
    def get_statistics(self):
        """Get comprehensive system statistics"""
        stats = self._stm_api.get_statistics()
        
        # Add SCM statistics if enabled
        if self.scm:
            scm_stats = self.scm.get_statistics()
            stats['scm'] = scm_stats
        
        return stats
    
    def _extract_valence_from_stm(self, coord_key: str) -> float:
        """
        Extract emotional significance from STM entry using ASMC's sentiment analysis
        
        Reuses existing NLTK SentiWordNet analysis from coordinate generation.
        
        Args:
            coord_key: 9D semantic coordinate key
            
        Returns:
            float: Valence (-1.0 to +1.0)
        """
        try:
            # Get STM entry
            stm_entry = self._stm_api._stm.stm_entries.get(coord_key)
            if not stm_entry:
                return 0.0
            
            # Check if coordinate result has fingerprint with sentiment
            # (ASMC already computed this during coordinate generation)
            coord_result = stm_entry.get('coord_result', {})
            
            # Try to extract from fingerprint (if available)
            if 'fingerprint' in coord_result:
                fingerprint = coord_result['fingerprint']
                if hasattr(fingerprint, 'semantic_features'):
                    sentiment = fingerprint.semantic_features.get('sentiment', {})
                    if sentiment:
                        pos_score = sentiment.get('positive', 0.0)
                        neg_score = sentiment.get('negative', 0.0)
                        
                        # Convert to -1 to +1 scale
                        if pos_score + neg_score > 0:
                            valence = (pos_score - neg_score) / (pos_score + neg_score)
                            return valence
            
            # Fallback: Analyze response text directly (simpler heuristic)
            response = stm_entry.get('ai_response', '')
            return self._simple_valence_extraction(response)
            
        except Exception as e:
            if self.verbose:
                print(f"   [ASMC] Warning: Valence extraction failed: {e}")
            return 0.0
    
    def _simple_valence_extraction(self, text: str) -> float:
        """
        Simple valence extraction using keyword matching
        (Fallback if ASMC sentiment not available)
        """
        text_lower = text.lower()
        
        positive_keywords = [
            'good', 'great', 'excellent', 'found', 'discovered', 'satisfied',
            'happy', 'like', 'love', 'beautiful', 'amazing', 'successful',
            'comfortable', 'safe', 'peaceful', 'enjoy', 'interesting'
        ]
        
        negative_keywords = [
            'bad', 'terrible', 'failed', 'stuck', 'lost', 'confused',
            'sad', 'hate', 'scary', 'frustrating', 'painful', 'unable',
            'trapped', 'starving', 'dangerous', 'avoid'
        ]
        
        pos_count = sum(1 for word in positive_keywords if word in text_lower)
        neg_count = sum(1 for word in negative_keywords if word in text_lower)
        
        total = pos_count + neg_count
        if total > 0:
            return (pos_count - neg_count) / total
        
        return 0.0
    
    def get_spatial_context(self, structure_type: str, cluster_id: str, coordinates: dict,
                           radius: int = 0, include_ltm: bool = True, max_memories: int = 5,
                           use_cone_search: bool = False, query_text: str = None):
        """
        Get integrated spatial + semantic context for a location
        
        Two modes:
        1. Radial Search (default): Returns memories near coordinates
        2. Cone Search (use_cone_search=True): Semantic cone search using query's 9D valence
        
        Cone search mimics hippocampal memory retrieval:
        - Extracts cone parameters from query's semantic coordinate (9D)
        - Searches semantic space using directional cone
        - Adapts based on memory density and system maturity
        - Returns emergently-relevant memories
        
        Returns:
        - SCM node (structure, entities, neighbors, statistics)
        - Recent STM memories at this location
        - LTM patterns at this location
        - Cluster-level context
        
        Args:
            structure_type: Type of structure ('spatial', 'linear', etc.)
            cluster_id: Cluster identifier
            coordinates: Position within cluster
            radius: Include nearby nodes (not yet implemented)
            include_ltm: Include LTM patterns
            max_memories: Maximum STM memories to return
            use_cone_search: Enable semantic cone search (default: False)
            query_text: Query text for semantic coordinate generation (required if use_cone_search=True)
            
        Returns:
            Dict with complete spatial + semantic context
        """
        if not self.scm:
            return None
        
        # ============================================================
        # CONE SEARCH MODE - Semantic space directional retrieval
        # ============================================================
        if use_cone_search:
            if query_text is None:
                raise ValueError("query_text required when use_cone_search=True")
            
            if self.verbose:
                print(f"\n[CONE SEARCH] Initiating semantic cone search")
                print(f"[CONE SEARCH] Query: {query_text}")
            
            # Generate query's semantic coordinate
            from spatial_valence import UltraEnhancedSpatialValenceToCoordGeneration
            coord_gen = UltraEnhancedSpatialValenceToCoordGeneration()
            query_result = coord_gen.process(query_text)
            query_coords = query_result['coordinates']
            
            # Extract semantic parameters (9D → 3x3D)
            semantic_position = (query_coords[0], query_coords[1], query_coords[2])
            semantic_direction = (query_coords[3], query_coords[4], query_coords[5])
            semantic_shape = (query_coords[6], query_coords[7], query_coords[8])
            
            if self.verbose:
                print(f"[CONE SEARCH] Semantic position: {semantic_position}")
                print(f"[CONE SEARCH] Semantic direction: {semantic_direction}")
                print(f"[CONE SEARCH] Semantic shape: {semantic_shape}")
            
            # === STAGE 1: RADIAL RETRIEVAL ===
            all_nodes = self.scm.get_cluster_nodes(cluster_id)
            
            # Count memories for density calculation
            total_memories = sum(len(node.get('stm_coord_keys', [])) for node in all_nodes)
            memory_count = len(all_nodes)
            memory_density = total_memories / max(1, memory_count)
            
            # Calculate tightening factor (exponential maturity curve)
            import math
            total_system_memories = self.get_statistics().get('total_stm_links', 0)
            k = 0.001  # Tightening rate
            tightening_factor = -1.0 + (2.0 / (1.0 + math.exp(-k * total_system_memories)))
            
            # Map semantic shape to cone parameters
            cone_radius = abs(semantic_shape[0]) * memory_density * (1.0 + tightening_factor)
            cone_taper = abs(semantic_shape[1])
            cone_length = abs(semantic_shape[2]) * memory_density * (1.0 + tightening_factor)
            
            # Clamp to reasonable ranges
            cone_radius = max(0.1, min(5.0, cone_radius))
            cone_taper = max(0.01, min(1.0, cone_taper))
            cone_length = max(0.5, min(10.0, cone_length))
            
            if self.verbose:
                print(f"[STAGE 1] Radial: {memory_count} nodes, {total_memories} memories, density={memory_density:.2f}")
                print(f"[STAGE 1] Tightening: {tightening_factor:.4f} (from {total_system_memories} total memories)")
                print(f"[STAGE 1] Cone params: radius={cone_radius:.2f}, taper={cone_taper:.2f}, length={cone_length:.2f}")
            
            # === STAGE 2: DIRECTION FILTER ===
            semantic_direction_normalized = self._normalize_vector(semantic_direction)
            direction_filtered = []
            
            for node in all_nodes:
                for memory_coord_key in node.get('stm_coord_keys', []):
                    # Parse memory's semantic coordinate
                    memory_coords = self._parse_coordinate_key(memory_coord_key)
                    if not memory_coords:
                        continue
                    
                    memory_position = (memory_coords[0], memory_coords[1], memory_coords[2])
                    
                    # Calculate semantic offset
                    offset = self._subtract_vectors(memory_position, semantic_position)
                    offset_mag = self._magnitude(offset)
                    
                    if offset_mag < 1e-6:
                        continue
                    
                    # Check directional alignment
                    offset_normalized = self._normalize_vector(offset)
                    alignment = self._dot_product(offset_normalized, semantic_direction_normalized)
                    
                    if alignment > 0:  # Forward hemisphere
                        direction_filtered.append({
                            'node': node,
                            'memory_coord_key': memory_coord_key,
                            'memory_coords': memory_coords,
                            'alignment': alignment,
                            'offset': offset
                        })
            
            if self.verbose:
                print(f"[STAGE 2] Direction: {len(direction_filtered)} memories in cone direction")
            
            # === STAGE 3: CONE SHAPE FILTER ===
            cone_filtered = []
            
            for item in direction_filtered:
                offset = item['offset']
                
                # Project onto direction axis
                projection = self._dot_product(offset, semantic_direction_normalized)
                
                # Check length bound
                if projection < 0 or projection > cone_length:
                    continue
                
                # Calculate cone radius at this projection distance
                cone_radius_at_d = cone_radius + (cone_taper * projection)
                
                # Calculate perpendicular distance from axis
                parallel_component = self._scale_vector(semantic_direction_normalized, projection)
                perpendicular = self._subtract_vectors(offset, parallel_component)
                perp_distance = self._magnitude(perpendicular)
                
                # Check if inside cone
                if perp_distance <= cone_radius_at_d:
                    item['projection'] = projection
                    item['perp_distance'] = perp_distance
                    item['cone_radius_at_d'] = cone_radius_at_d
                    cone_filtered.append(item)
            
            if self.verbose:
                print(f"[STAGE 3] Cone: {len(cone_filtered)} memories inside cone shape")
            
            # === STAGE 4: SCORE & SORT ===
            for item in cone_filtered:
                # 4-factor scoring
                angular_score = item['alignment']
                distance_score = 1.0 - (item['projection'] / cone_length)
                axis_score = 1.0 - (item['perp_distance'] / item['cone_radius_at_d'])
                
                # Density score
                node_memory_count = len(item['node'].get('stm_coord_keys', []))
                density_score = node_memory_count / max(1, memory_density)
                density_score = min(2.0, density_score)
                
                # Combined relevance
                item['relevance'] = (
                    0.30 * angular_score +
                    0.25 * distance_score +
                    0.20 * axis_score +
                    0.25 * density_score
                )
            
            # Sort by relevance
            cone_filtered.sort(key=lambda x: x['relevance'], reverse=True)
            
            # Build results
            memories = []
            for item in cone_filtered[:max_memories]:
                memory_entry = self._stm_api._stm.stm_entries.get(item['memory_coord_key'])
                if memory_entry:
                    memories.append({
                        'coord_key': item['memory_coord_key'],
                        'semantic_summary': memory_entry.get('semantic_summary', ''),
                        'relevance_score': item['relevance'],
                        'alignment': item['alignment'],
                        'projection': item['projection'],
                        'timestamp': memory_entry.get('timestamp', ''),
                        'valence': memory_entry.get('valence', 0.0)
                    })
            
            if self.verbose:
                print(f"[STAGE 4] Returning {len(memories)} top-scored memories")
            
            return {
                'success': True,
                'cone_search_used': True,
                'memories': memories,
                'cone_stats': {
                    'stage1_radial': memory_count,
                    'stage2_direction': len(direction_filtered),
                    'stage3_cone': len(cone_filtered),
                    'final_returned': len(memories),
                    'tightening_factor': tightening_factor,
                    'cone_radius': cone_radius,
                    'cone_taper': cone_taper,
                    'cone_length': cone_length,
                    'memory_density': memory_density
                }
            }
        
        # ============================================================
        # RADIAL SEARCH MODE (Default)
        # ============================================================
        
        # Get SCM node
        node = self.scm.get_node(structure_type, cluster_id, coordinates)
        if not node:
            return None
        
        # Get cluster info
        cluster = self.scm.get_cluster(cluster_id)
        
        # Log SCM retrieval operation
        self._log_scm_operation("SCM_RETRIEVE", {
            "node_key": node.get('node_key', 'unknown'),
            "cluster_id": cluster_id,
            "coordinates": coordinates,
            "location_type": node.get('location_type', 'unknown'),
            "visit_count": node.get('visit_count', 0),
            "stm_links": len(node.get('stm_coord_keys', [])),
            "ltm_links": len(node.get('ltm_engram_ids', [])),
            "valence": f"{node.get('aggregate_valence', 0.0):.3f}"
        })
        
        # Fetch recent STM memories
        stm_memories = []
        for coord_key in node.get('stm_coord_keys', [])[-max_memories:]:
            entry = self._stm_api._stm.stm_entries.get(coord_key)
            if entry:
                # Return full_context - complete interaction cycle with causation chain
                stm_memories.append({
                    'coord_key': coord_key,
                    'full_context': entry.get('full_context', ''),
                    'timestamp': entry.get('timestamp', ''),
                    'valence': entry.get('valence', 0.0)
                })
        
        # Fetch LTM patterns (if requested)
        ltm_patterns = []
        if include_ltm:
            for ltm_ref in node.get('ltm_engram_ids', []):
                ltm_patterns.append({
                    'engram_id': ltm_ref.get('engram_id'),
                    'concept': ltm_ref.get('concept'),
                    'strength': ltm_ref.get('strength', 0.0)
                })
        
        return {
            'node': node,
            'cluster': cluster,
            'stm_memories': stm_memories,
            'ltm_patterns': ltm_patterns,
            'visit_count': node.get('visit_count', 0),
            'aggregate_valence': node.get('aggregate_valence', 0.0),
            'cluster_valence': cluster.get('aggregate_valence', 0.0) if cluster else 0.0
        }
    
    def get_spatial_context_string(self, structure_type: str, cluster_id: str, coordinates: dict,
                                   max_memories: int = 3) -> str:
        """
        Get human-readable spatial context string for LLM prompts
        
        Returns formatted string with:
        - Current location info
        - Physical objects/entities
        - Recent memories
        - Long-term patterns
        - Emotional valence
        - Available exits
        """
        context = self.get_spatial_context(structure_type, cluster_id, coordinates, max_memories=max_memories)
        if not context:
            return ""
        
        node = context['node']
        cluster = context['cluster']
        
        # Build context string
        lines = []
        lines.append("=== SPATIAL CONTEXT ===")
        lines.append(f"Location: {node.get('location_type', 'Unknown')} at {coordinates}")
        lines.append(f"Region: {cluster.get('description', cluster_id) if cluster else cluster_id}")
        lines.append(f"Visits: {node.get('visit_count', 0)} times")
        
        # Valence
        valence = node.get('aggregate_valence', 0.0)
        if valence > 0.3:
            feeling = "positive (+{:.2f})".format(valence)
        elif valence < -0.3:
            feeling = "negative ({:.2f})".format(valence)
        else:
            feeling = "neutral ({:.2f})".format(valence)
        lines.append(f"Feeling: {feeling}")
        
        # Objects
        entities = node.get('entities', [])
        if entities:
            lines.append(f"Objects here: {', '.join(entities)}")
        
        # Recent memories - FULL context with complete causation chains
        if context['stm_memories']:
            lines.append("\nRecent experiences:")
            for mem in context['stm_memories'][:5]:  # Top 5 recent memories
                full_context = mem.get('full_context', '')
                if full_context:
                    lines.append(f"  {full_context}")
                    lines.append("")  # Blank line between memories for readability
        
        # Long-term patterns
        if context['ltm_patterns']:
            lines.append("\nKnown patterns:")
            for pat in context['ltm_patterns'][:5]:  # Top 5 patterns
                lines.append(f"  - {pat['concept']} (strength: {pat['strength']:.2f})")
        
        # Neighbors/exits
        neighbors = node.get('neighbors', {})
        if neighbors:
            lines.append("\nAvailable exits:")
            for direction, neighbor in neighbors.items():
                if neighbor:
                    lines.append(f"  - {direction}")
        
        return "\n".join(lines)
    
    def clear_memory(self, confirm: bool = False):
        """
        Clear all memories: STM, LTM, and SCM (DESTRUCTIVE - requires confirm=True)
        
        Properly handles LMDB directories (not just files) using shutil.rmtree
        """
        if not confirm:
            return {
                'success': False,
                'message': 'Must set confirm=True to clear memory (DESTRUCTIVE operation)'
            }
        
            import os
        import shutil
        
        cleared = {
            'stm': False,
            'ltm': False,
            'scm': False,
            'errors': []
        }
        
        print("\n[MEMORY] CLEARING ALL MEMORY SYSTEMS...")
        
        # 1. Clear STM (RAM + cache files)
        try:
            print("  [STM] Clearing STM...")
            stm_result = self._stm_api.clear_memory(confirm=True)
            cleared['stm'] = stm_result.get('success', False)
            
            # Also delete STM cache directory
            if os.path.exists(self.stm_path):
                shutil.rmtree(self.stm_path)
                os.makedirs(self.stm_path, exist_ok=True)
                print(f"     [OK] STM cache directory cleared: {self.stm_path}")
        except Exception as e:
            cleared['errors'].append(f"STM: {e}")
            print(f"     [FAIL] STM clearing failed: {e}")
        
        # 2. Clear LTM (LMDB directory)
        try:
            print("  [LTM] Clearing LTM...")
            
            # Close LTM connection first (via STM API)
            if hasattr(self._stm_api, '_ltm') and self._stm_api._ltm:
                self._stm_api._ltm.close()
            
            # Delete LTM directory
            if os.path.exists(self.ltm_db_path):
                if os.path.isdir(self.ltm_db_path):
                    shutil.rmtree(self.ltm_db_path)
                else:
                    os.remove(self.ltm_db_path)
                print(f"     [OK] LTM database cleared: {self.ltm_db_path}")
                cleared['ltm'] = True
            else:
                print(f"     [INFO] LTM database not found: {self.ltm_db_path}")
                cleared['ltm'] = True
        except Exception as e:
            cleared['errors'].append(f"LTM: {e}")
            print(f"     [FAIL] LTM clearing failed: {e}")
        
        # 3. Clear SCM (LMDB directory)
        if self.scm:
            try:
                print("  [SCM] Clearing SCM...")
                
                # Close SCM connection
                self.scm.close()
                
                # Delete SCM directory
                if os.path.exists(self.scm_path):
                    if os.path.isdir(self.scm_path):
                        shutil.rmtree(self.scm_path)
                    else:
                        os.remove(self.scm_path)
                    print(f"     [OK] SCM database cleared: {self.scm_path}")
                
                # Also clear SCM log
                if os.path.exists(self.scm_log_path):
                    os.remove(self.scm_log_path)
                    print(f"     [OK] SCM log cleared: {self.scm_log_path}")
                
                # Recreate fresh SCM
                self.scm = create_scm(db_path=self.scm_path, verbose=self.verbose)
                cleared['scm'] = True
                print(f"     [OK] Fresh SCM recreated")
            except Exception as e:
                cleared['errors'].append(f"SCM: {e}")
                print(f"     [FAIL] SCM clearing failed: {e}")
        
        success = cleared['stm'] and cleared['ltm'] and (cleared['scm'] or not self.enable_scm)
        
        if success:
            print("[SUCCESS] ALL MEMORY SYSTEMS CLEARED\n")
        else:
            print(f"[WARNING] MEMORY CLEARING INCOMPLETE: {cleared['errors']}\n")
        
        return {
            'success': success,
            'cleared': cleared,
            'errors': cleared['errors']
        }
    
    def shutdown(self):
        """Gracefully shutdown the memory system"""
        return self._stm_api.shutdown()
    
    def MassDataUpload(self, folder_path, file_extensions=['.txt', '.md'], chunk_size=300):
        """
        Matrix-style knowledge upload - scan folder and inject all text files as memories.
        Auto-chunks large files and uploads via existing STM pipeline (auto-promotes to LTM).
        
        Args:
            folder_path: Path to folder containing text files
            file_extensions: File types to process (default: ['.txt', '.md'])
            chunk_size: Size of text chunks in characters (default: 300)
        
        Returns:
            Dict: Upload statistics
        """
        import os
        
        print("🧠" * 30)
        print("🧠 ASMC MASS DATA UPLOAD - MATRIX MODE")
        print("🧠" * 30)
        print(f"📁 Scanning: {folder_path}")
        print(f"📄 File types: {', '.join(file_extensions)}")
        print(f"✂️ Chunk size: {chunk_size} chars")
        print("="*70 + "\n")
        
        # Find all files
        all_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in file_extensions):
                    all_files.append(os.path.join(root, file))
        
        if not all_files:
            print("❌ No files found!")
            return {'error': 'No files found', 'files_processed': 0}
        
        print(f"📚 Found {len(all_files)} files\n")
        
        total_chunks = 0
        files_processed = 0
        
        # Process each file
        for file_idx, filepath in enumerate(all_files, 1):
            filename = os.path.basename(filepath)
            print(f"📖 [{file_idx}/{len(all_files)}] Processing: {filename}")
            
            try:
                # Read file
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                
                if not text.strip():
                    print(f"   ⚠️ Empty file, skipping")
                    continue
                
                # Chunk text
                chunks = self._chunk_text(text, chunk_size)
                print(f"   ✂️ Created {len(chunks)} chunks")
                
                # Upload each chunk
                for chunk_idx, chunk in enumerate(chunks):
                    self.add_experience(
                        situation=f"Knowledge from {filename}",
                        response=chunk,
                        metadata={
                            'source': 'mass_upload',
                            'filename': filename,
                            'filepath': filepath,
                            'chunk_index': chunk_idx,
                            'total_chunks': len(chunks),
                            'is_innate': True
                        }
                    )
                    total_chunks += 1
                    
                    # Progress every 100 chunks
                    if total_chunks % 100 == 0:
                        stats = self.get_statistics()
                        promoted = stats.get('total_promoted_to_longterm', 0)
                        print(f"   📊 Progress: {total_chunks} chunks uploaded | {promoted} promoted to LTM")
                
                files_processed += 1
                print(f"   ✅ {filename} complete ({len(chunks)} chunks)\n")
                
            except Exception as e:
                print(f"   ❌ Error processing {filename}: {e}\n")
        
        # Final statistics
        final_stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("🎉 MASS UPLOAD COMPLETE!")
        print("="*70)
        print(f"📁 Files processed: {files_processed}/{len(all_files)}")
        print(f"📝 Total chunks: {total_chunks}")
        print(f"🧠 STM entries: {final_stats.get('current_entries', 0)}")
        print(f"📤 Promoted to LTM: {final_stats.get('total_promoted_to_longterm', 0)}")
        print(f"💾 Memory system ready with pre-loaded knowledge!")
        print("="*70)
        
        return {
            'success': True,
            'files_found': len(all_files),
            'files_processed': files_processed,
            'chunks_uploaded': total_chunks,
            'promoted_to_ltm': final_stats.get('total_promoted_to_longterm', 0),
            'current_stm_entries': final_stats.get('current_entries', 0)
        }
    
    def _chunk_text(self, text, chunk_size=300):
        """Split text into semantic chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        # Split on sentences
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            test_chunk = current_chunk + ". " + sentence if current_chunk else sentence
            
            if len(test_chunk) <= chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk + ".")
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk + ".")
        
        return chunks
    
    # ========================================================================
    # CONE SEARCH HELPER METHODS
    # ========================================================================
    
    def _normalize_vector(self, vec):
        """Normalize a 3D vector to unit length"""
        import math
        if isinstance(vec, dict):
            vec = (vec.get('x', 0), vec.get('y', 0), vec.get('z', 0))
        
        mag = math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
        if mag < 1e-8:
            return (0, 0, 1)  # Default to forward if zero vector
        return (vec[0]/mag, vec[1]/mag, vec[2]/mag)
    
    def _magnitude(self, vec):
        """Calculate magnitude of a 3D vector"""
        import math
        return math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    
    def _dot_product(self, vec1, vec2):
        """Calculate dot product of two 3D vectors"""
        return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]
    
    def _subtract_vectors(self, vec1, vec2):
        """Subtract two 3D vectors"""
        return (vec1[0]-vec2[0], vec1[1]-vec2[1], vec1[2]-vec2[2])
    
    def _scale_vector(self, vec, scalar):
        """Scale a 3D vector by a scalar"""
        return (vec[0]*scalar, vec[1]*scalar, vec[2]*scalar)
    
    def _parse_coordinate_key(self, coord_key):
        """Parse coordinate key string to list of floats"""
        try:
            # Format: '[0.45][-0.32][0.61]...'
            parts = coord_key.strip('[]').split('][')
            coords = [float(p) for p in parts]
            if len(coords) >= 9:
                return coords
        except:
            pass
        return None


# Convenience factory function
def create_memory(max_entries: int = 50, db_path: str = None, verbose: bool = False):
    """
    Quick factory function to create Advanced Semantic Memory system
    
    Args:
        max_entries: Maximum STM entries
        db_path: OPTIONAL custom LTM path (default: auto-managed in MemoryStructures/LTM/ltm.lmdb)
        verbose: Enable logging
        
    Returns:
        AdvancedSemanticMemory: Initialized memory system
    
    Note:
        SCM (Spatial Comprehension Map) is ALWAYS enabled - it's a core research feature.
    """
    return AdvancedSemanticMemory(
        max_stm_entries=max_entries,
        ltm_db_path=db_path,
        verbose=verbose,
        enable_scm=True  # ALWAYS enabled - SCM is core to spatial episodic memory research
    )


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("ADVANCED SEMANTIC MEMORY CLUSTERING - Example")
    print("=" * 60)
    
    # Create memory system
    memory = create_memory(max_entries=10, verbose=True)
    
    # Add some experiences
    print("\nAdding experiences...")
    memory.add_experience(
        situation="I took cursed amulet in Room (1,1)",
        response="Lost 25 HP! Health now 75/100"
    )
    
    memory.add_experience(
        situation="I took healing potion in Room (2,2)",
        response="Gained 30 HP! Health now 100/100"
    )
    
    # Get context for new situation
    print("\nGetting context for: 'I see a cursed amulet'")
    context = memory.get_context("I see a cursed amulet", layer1_count=6, layer2_count=6)
    
    if context['success']:
        print(f"\nLayer 1 (Immediate):")
        print(f"  Recent: {len(context.get('recent_context', []))} items")
        print(f"  Relevant: {len(context.get('relevant_context', []))} items")
        
        print(f"\nLayer 2 (Depth):")
        print(f"  Semantic: {len(context.get('ltm_semantic', []))} items")
        print(f"  Neighbors: {len(context.get('ltm_neighbors', []))} items")
        
        print(f"\nTotal context: {context['total_context_entries']} items")
    
    # Shutdown
    print("\nShutting down...")
    memory.shutdown()
    print("Complete!")

