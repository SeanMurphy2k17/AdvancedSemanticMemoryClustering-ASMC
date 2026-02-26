#!/usr/bin/env python3
"""
🗺️ SPATIAL COMPREHENSION MAP (SCM) 🗺️

CREATORS:
- Sean Murphy (Human Inventor & System Architect)
- Claude AI Models (AI Co-Inventor & Implementation Partner)

The third pillar of Ghost's cognitive architecture:
- STM/LTM (ASMC): Semantic clustering (9D valence space) - "WHAT happened?"
- SCM: Physical anchoring (spatial structure + metadata) - "WHERE did it happen?"
- GhostEngine: Executive planning (action selection) - "WHAT should I do?"

ARCHITECTURE:
- Lightweight metadata layer on top of ASMC
- Stores physical structure (maps, objects, neighbors)
- Stores statistics (visits, valence, timestamps)
- Stores references to STM/LTM (NOT the content itself)
- Bidirectional linking: STM ↔ SCM ↔ LTM

FEATURES:
- Cluster-based organization (apartment, streets, store)
- Node-level tracking (kitchen, bedroom, town square)
- Emotional spatial mapping (valence per location)
- Portal tracking (doors between clusters)
- Automatic STM/LTM integration
- NLTK-based significance extraction (reuses ASMC sentiment)

PURPOSE:
SCM provides the "WHERE" to ASMC's "WHAT", creating absolute contextual
grounding for semantic memories. This accelerates spatial learning by giving
Ghost a persistent map structure that semantic memories can anchor to.

License: MIT
Copyright (c) 2024 Sean Murphy
"""

import os
import lmdb
import msgpack
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class SpatialComprehensionMap:
    """
    🗺️ SPATIAL COMPREHENSION MAP
    
    Physical spatial structure with semantic memory anchoring.
    Lightweight metadata layer - stores structure + references, not memories.
    """
    
    # Structure type definitions
    STRUCTURE_TYPES = {
        'spatial': {
            'description': '3D physical space',
            'coord_names': ['x', 'y', 'z'],
            'neighbor_types': ['forward', 'back', 'left', 'right', 'up', 'down']
        },
        'linear': {
            'description': '1D sequence (timeline, pages)',
            'coord_names': ['position'],
            'neighbor_types': ['previous', 'next']
        },
        'grid': {
            'description': '2D discrete space',
            'coord_names': ['row', 'col'],
            'neighbor_types': ['north', 'south', 'east', 'west']
        },
        'tree': {
            'description': 'Hierarchical structure',
            'coord_names': ['path'],
            'neighbor_types': ['parent', 'children', 'siblings']
        },
        'graph': {
            'description': 'Network/concept space',
            'coord_names': ['node_id'],
            'neighbor_types': 'dynamic'
        },
        'temporal': {
            'description': 'Time-based sequence',
            'coord_names': ['timestamp'],
            'neighbor_types': ['before', 'after', 'concurrent']
        }
    }
    
    def __init__(self, db_path: str = "./scm_data/scm.lmdb", verbose: bool = False):
        """
        Initialize Spatial Comprehension Map
        
        Args:
            db_path: Path to LMDB database
            verbose: Enable detailed logging
        """
        self.db_path = db_path
        self.verbose = verbose
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Open LMDB environment
        self.env = lmdb.open(
            db_path,
            map_size=200 * 1024 * 1024,  # 200MB (plenty for metadata)
            max_dbs=2,  # 2 sub-databases
            sync=False,  # Async writes for speed
            writemap=True  # Memory-mapped writes
        )
        
        # Sub-databases
        self.clusters_db = self.env.open_db(b'clusters')  # Cluster metadata
        self.nodes_db = self.env.open_db(b'nodes')        # Node metadata + references
        
        # Statistics
        self.stats = {
            'total_clusters': 0,
            'total_nodes': 0,
            'total_visits': 0,
            'total_stm_links': 0,
            'total_ltm_links': 0
        }
        
        # Load stats from database
        self._load_stats()
        
        if verbose:
            print("🗺️ Spatial Comprehension Map initialized")
            print(f"   Database: {db_path}")
            print(f"   Clusters: {self.stats['total_clusters']}")
            print(f"   Nodes: {self.stats['total_nodes']}")
            print(f"   Total visits: {self.stats['total_visits']}")
    
    def _load_stats(self):
        """Load statistics from database"""
        try:
            with self.env.begin() as txn:
                # Count clusters
                cursor = txn.cursor(db=self.clusters_db)
                self.stats['total_clusters'] = sum(1 for _ in cursor)
                
                # Count nodes and aggregate stats
                cursor = txn.cursor(db=self.nodes_db)
                node_count = 0
                total_visits = 0
                total_stm = 0
                total_ltm = 0
                
                for key, value in cursor:
                    node = msgpack.unpackb(value)
                    node_count += 1
                    total_visits += node.get('visit_count', 0)
                    total_stm += len(node.get('stm_coord_keys', []))
                    total_ltm += len(node.get('ltm_engram_ids', []))
                
                self.stats['total_nodes'] = node_count
                self.stats['total_visits'] = total_visits
                self.stats['total_stm_links'] = total_stm
                self.stats['total_ltm_links'] = total_ltm
        except Exception as e:
            if self.verbose:
                print(f"   Note: Could not load stats (new database?): {e}")
    
    def _make_node_key(self, structure_type: str, cluster_id: str, coordinates: dict) -> str:
        """
        Generate unique node key from structure type, cluster, and coordinates
        
        Format: [structure_type][cluster_id][coord1][coord2]...
        Example: [spatial][level_1_apartment][1][1][1]
        """
        if structure_type == 'spatial':
            x = int(coordinates.get('x', 0))
            y = int(coordinates.get('y', 0))
            z = int(coordinates.get('z', 0))
            return f"[spatial][{cluster_id}][{x}][{y}][{z}]"
        
        elif structure_type == 'linear':
            pos = int(coordinates.get('position', 0))
            return f"[linear][{cluster_id}][{pos}]"
        
        elif structure_type == 'grid':
            row = int(coordinates.get('row', 0))
            col = int(coordinates.get('col', 0))
            return f"[grid][{cluster_id}][{row}][{col}]"
        
        elif structure_type == 'tree':
            path = coordinates.get('path', [])
            path_str = ']['.join(str(p) for p in path)
            return f"[tree][{cluster_id}][{path_str}]"
        
        elif structure_type == 'graph':
            node_id = coordinates.get('node_id', 'unknown')
            return f"[graph][{cluster_id}][{node_id}]"
        
        elif structure_type == 'temporal':
            timestamp = coordinates.get('timestamp', 0)
            return f"[temporal][{cluster_id}][{timestamp}]"
        
        else:
            raise ValueError(f"Unknown structure type: {structure_type}")
    
    # ========================================================================
    # CLUSTER MANAGEMENT
    # ========================================================================
    
    def create_cluster(self, cluster_id: str, cluster_type: str = 'spatial',
                      description: str = "", bounds: dict = None) -> str:
        """
        Create a spatial cluster (region, zone, level)
        
        Args:
            cluster_id: Unique identifier (e.g., 'level_1_apartment')
            cluster_type: Structure type ('spatial', 'linear', etc.)
            description: Human-readable description
            bounds: Optional spatial bounds for filtering
            
        Returns:
            cluster_id
        """
        if cluster_type not in self.STRUCTURE_TYPES:
            raise ValueError(f"Unknown cluster type: {cluster_type}")
        
        cluster = {
            'cluster_id': cluster_id,
            'cluster_type': cluster_type,
            'description': description,
            'bounds': bounds or {},
            'node_count': 0,
            'total_visits': 0,
            'first_visit': datetime.now().isoformat(),
            'last_visit': None,
            'aggregate_valence': 0.0,
            'valence_sum': 0.0,
            'ltm_cluster_concepts': [],  # High-level patterns for this region
            'portals': []  # Connections to other clusters
        }
        
        with self.env.begin(write=True) as txn:
            txn.put(cluster_id.encode(), msgpack.packb(cluster), db=self.clusters_db)
        
        self.stats['total_clusters'] += 1
        
        if self.verbose:
            print(f"   [SCM] Created cluster: {cluster_id} ({cluster_type})")
        
        return cluster_id
    
    def cluster_exists(self, cluster_id: str) -> bool:
        """Check if cluster exists"""
        with self.env.begin() as txn:
            return txn.get(cluster_id.encode(), db=self.clusters_db) is not None
    
    def get_cluster(self, cluster_id: str) -> Optional[Dict]:
        """Get cluster metadata"""
        with self.env.begin() as txn:
            cluster_data = txn.get(cluster_id.encode(), db=self.clusters_db)
            if cluster_data:
                return msgpack.unpackb(cluster_data)
        return None
    
    def get_all_clusters(self) -> List[Dict]:
        """Get all clusters"""
        clusters = []
        with self.env.begin() as txn:
            cursor = txn.cursor(db=self.clusters_db)
            for key, value in cursor:
                clusters.append(msgpack.unpackb(value))
        return clusters
    
    def update_cluster_valence(self, cluster_id: str, node_valence: float, visit_delta: int = 1):
        """
        Update cluster aggregate valence when a node is visited
        
        Args:
            cluster_id: Cluster to update
            node_valence: Valence from the node visit
            visit_delta: Number of visits to add (usually 1)
        """
        with self.env.begin(write=True) as txn:
            cluster_data = txn.get(cluster_id.encode(), db=self.clusters_db)
            if not cluster_data:
                return
            
            cluster = msgpack.unpackb(cluster_data)
            
            # Update visit count
            cluster['total_visits'] += visit_delta
            cluster['last_visit'] = datetime.now().isoformat()
            
            # Update aggregate valence
            cluster['valence_sum'] += node_valence
            if cluster['total_visits'] > 0:
                cluster['aggregate_valence'] = cluster['valence_sum'] / cluster['total_visits']
            
            txn.put(cluster_id.encode(), msgpack.packb(cluster), db=self.clusters_db)
    
    def add_cluster_concept(self, cluster_id: str, concept: str, strength: float):
        """
        Add a high-level LTM concept to a cluster
        
        Args:
            cluster_id: Cluster to update
            concept: Concept name (e.g., 'food_acquisition', 'safe_space')
            strength: Concept strength (0.0-1.0)
        """
        with self.env.begin(write=True) as txn:
            cluster_data = txn.get(cluster_id.encode(), db=self.clusters_db)
            if not cluster_data:
                return
            
            cluster = msgpack.unpackb(cluster_data)
            
            # Check if concept already exists
            existing = [c for c in cluster['ltm_cluster_concepts'] if c['concept'] == concept]
            if existing:
                # Update strength
                for c in cluster['ltm_cluster_concepts']:
                    if c['concept'] == concept:
                        c['strength'] = max(c['strength'], strength)  # Keep highest strength
            else:
                # Add new concept
                cluster['ltm_cluster_concepts'].append({
                    'concept': concept,
                    'strength': strength
                })
            
            # Sort by strength
            cluster['ltm_cluster_concepts'].sort(key=lambda x: x['strength'], reverse=True)
            
            txn.put(cluster_id.encode(), msgpack.packb(cluster), db=self.clusters_db)
    
    # ========================================================================
    # NODE MANAGEMENT
    # ========================================================================
    
    def create_or_update_node(self, structure_type: str, cluster_id: str,
                              coordinates: dict, location_type: str = "",
                              entities: list = None, neighbors: dict = None) -> str:
        """
        Create or update a spatial node
        
        Stores:
        - Physical structure (location_type, entities, neighbors)
        - Statistics (visit_count, timestamps)
        - References to STM/LTM (NOT the memories themselves)
        
        Args:
            structure_type: Type of structure ('spatial', 'linear', etc.)
            cluster_id: Parent cluster identifier
            coordinates: Position within cluster
            location_type: Description of this location (e.g., "Kitchen")
            entities: List of objects/entities at this location
            neighbors: Navigation links to adjacent nodes
            
        Returns:
            node_key: Unique node identifier
        """
        node_key = self._make_node_key(structure_type, cluster_id, coordinates)
        
        with self.env.begin(write=True) as txn:
            existing = txn.get(node_key.encode(), db=self.nodes_db)
            
            if existing:
                # Update existing node
                node = msgpack.unpackb(existing)
                
                # Update physical metadata if provided
                if location_type:
                    node['location_type'] = location_type
                if entities is not None:
                    node['entities'] = entities
                if neighbors is not None:
                    node['neighbors'] = neighbors
                
            else:
                # Create new node
                node = {
                    'structure_type': structure_type,
                    'cluster_id': cluster_id,
                    'coordinates': coordinates,
                    'node_key': node_key,
                    'location_type': location_type,
                    'entities': entities or [],
                    'neighbors': neighbors or {},
                    'visit_count': 0,
                    'first_visit': datetime.now().isoformat(),
                    'last_visit': None,
                    'aggregate_valence': 0.0,
                    'valence_sum': 0.0,
                    'valence_history': [],
                    'stm_coord_keys': [],  # References to STM memories
                    'ltm_engram_ids': [],  # References to LTM patterns
                    'portals': []  # Exits to other clusters
                }
                
                self.stats['total_nodes'] += 1
                
                # Update cluster node count
                cluster_data = txn.get(cluster_id.encode(), db=self.clusters_db)
                if cluster_data:
                    cluster = msgpack.unpackb(cluster_data)
                    cluster['node_count'] += 1
                    txn.put(cluster_id.encode(), msgpack.packb(cluster), db=self.clusters_db)
                
                if self.verbose:
                    print(f"   [SCM] Created node: {node_key}")
            
            txn.put(node_key.encode(), msgpack.packb(node), db=self.nodes_db)
        
        return node_key
    
    def visit_node(self, structure_type: str, cluster_id: str, coordinates: dict) -> str:
        """
        Record a visit to a node (updates statistics)
        
        Returns:
            node_key
        """
        node_key = self._make_node_key(structure_type, cluster_id, coordinates)
        
        with self.env.begin(write=True) as txn:
            node_data = txn.get(node_key.encode(), db=self.nodes_db)
            if not node_data:
                # Node doesn't exist, create it
                return self.create_or_update_node(structure_type, cluster_id, coordinates)
            
            node = msgpack.unpackb(node_data)
            node['visit_count'] += 1
            node['last_visit'] = datetime.now().isoformat()
            
            txn.put(node_key.encode(), msgpack.packb(node), db=self.nodes_db)
        
        self.stats['total_visits'] += 1
        
        return node_key
    
    def get_node(self, structure_type: str, cluster_id: str, coordinates: dict) -> Optional[Dict]:
        """
        Get node metadata (structure + statistics + references)
        
        Returns node with:
        - Physical structure (location_type, entities, neighbors)
        - Statistics (visit_count, valence)
        - References (stm_coord_keys, ltm_engram_ids)
        
        Does NOT return actual memories (fetch those from ASMC)
        """
        node_key = self._make_node_key(structure_type, cluster_id, coordinates)
        return self.get_node_by_key(node_key)
    
    def get_node_by_key(self, node_key: str) -> Optional[Dict]:
        """Get node by key"""
        with self.env.begin() as txn:
            node_data = txn.get(node_key.encode(), db=self.nodes_db)
            if node_data:
                return msgpack.unpackb(node_data)
        return None
    
    def get_cluster_nodes(self, cluster_id: str) -> List[Dict]:
        """Get all nodes within a cluster"""
        nodes = []
        with self.env.begin() as txn:
            cursor = txn.cursor(db=self.nodes_db)
            for key, value in cursor:
                node = msgpack.unpackb(value)
                if node.get('cluster_id') == cluster_id:
                    nodes.append(node)
        return nodes
    
    # ========================================================================
    # MEMORY LINKING (STM/LTM)
    # ========================================================================
    
    def link_stm_memory(self, node_key: str, stm_coord_key: str, valence: float):
        """
        Link a STM memory to a spatial node (reference only!)
        
        Args:
            node_key: SCM node identifier
            stm_coord_key: 9D ASMC coordinate key (reference)
            valence: Emotional significance (-1.0 to +1.0)
        """
        with self.env.begin(write=True) as txn:
            node_data = txn.get(node_key.encode(), db=self.nodes_db)
            if not node_data:
                if self.verbose:
                    print(f"   [SCM] Warning: Node {node_key} not found for STM link")
                return
            
            node = msgpack.unpackb(node_data)
            
            # Add reference (just the key!)
            if stm_coord_key not in node['stm_coord_keys']:
                node['stm_coord_keys'].append(stm_coord_key)
                self.stats['total_stm_links'] += 1
            
            # Update aggregate valence
            node['valence_sum'] += valence
            if node['visit_count'] > 0:
                old_valence = node['aggregate_valence']
                node['aggregate_valence'] = node['valence_sum'] / node['visit_count']
                
                # Track valence history (sample every 10 visits to avoid bloat)
                if node['visit_count'] % 10 == 0:
                    node['valence_history'].append({
                        'timestamp': datetime.now().isoformat(),
                        'valence': node['aggregate_valence'],
                        'visit_count': node['visit_count']
                    })
            
            txn.put(node_key.encode(), msgpack.packb(node), db=self.nodes_db)
            
            # Update cluster valence (in same transaction)
            cluster_id = node['cluster_id']
            cluster_data = txn.get(cluster_id.encode(), db=self.clusters_db)
            if cluster_data:
                cluster = msgpack.unpackb(cluster_data)
                cluster['valence_sum'] += valence
                if cluster['total_visits'] > 0:
                    cluster['aggregate_valence'] = cluster['valence_sum'] / cluster['total_visits']
                txn.put(cluster_id.encode(), msgpack.packb(cluster), db=self.clusters_db)
    
    def link_ltm_pattern(self, node_key: str, engram_id: str, concept: str, strength: float):
        """
        Link a LTM pattern to a spatial node (reference only!)
        
        Args:
            node_key: SCM node identifier
            engram_id: LTM engram identifier (reference)
            concept: Pattern name (for quick lookup)
            strength: Pattern strength (0.0-1.0)
        """
        with self.env.begin(write=True) as txn:
            node_data = txn.get(node_key.encode(), db=self.nodes_db)
            if not node_data:
                if self.verbose:
                    print(f"   [SCM] Warning: Node {node_key} not found for LTM link")
                return
            
            node = msgpack.unpackb(node_data)
            
            # Check if already linked
            existing_ids = [ref['engram_id'] for ref in node['ltm_engram_ids']]
            if engram_id not in existing_ids:
                node['ltm_engram_ids'].append({
                    'engram_id': engram_id,
                    'concept': concept,
                    'strength': strength
                })
                self.stats['total_ltm_links'] += 1
                
                # Sort by strength
                node['ltm_engram_ids'].sort(key=lambda x: x['strength'], reverse=True)
                
                # Also add to cluster-level concepts
                cluster_id = node['cluster_id']
                self.add_cluster_concept(cluster_id, concept, strength)
                
                if self.verbose:
                    print(f"   [SCM] Linked LTM pattern '{concept}' to {node_key}")
            
            txn.put(node_key.encode(), msgpack.packb(node), db=self.nodes_db)
    
    # ========================================================================
    # QUERY METHODS
    # ========================================================================
    
    def get_cluster_valence(self, cluster_id: str) -> float:
        """Get aggregate valence for entire cluster"""
        cluster = self.get_cluster(cluster_id)
        if cluster:
            return cluster.get('aggregate_valence', 0.0)
        return 0.0
    
    def get_statistics(self) -> Dict:
        """Get SCM statistics"""
        return {
            **self.stats,
            'avg_visits_per_node': self.stats['total_visits'] / max(1, self.stats['total_nodes']),
            'avg_stm_per_node': self.stats['total_stm_links'] / max(1, self.stats['total_nodes']),
            'avg_ltm_per_node': self.stats['total_ltm_links'] / max(1, self.stats['total_nodes'])
        }
    
    def close(self):
        """Close LMDB environment"""
        self.env.close()
        if self.verbose:
            print("🗺️ SCM closed")


def create_scm(db_path: str = "./scm_data/scm.lmdb", verbose: bool = False) -> SpatialComprehensionMap:
    """
    Factory function to create SCM instance
    
    Args:
        db_path: Path to LMDB database
        verbose: Enable detailed logging
        
    Returns:
        SpatialComprehensionMap instance
    """
    return SpatialComprehensionMap(db_path=db_path, verbose=verbose)


if __name__ == "__main__":
    # Simple test
    print("=== SCM Test ===")
    
    scm = create_scm(verbose=True)
    
    # Create a cluster
    scm.create_cluster('test_apartment', 'spatial', 'Test home interior')
    
    # Create some nodes
    scm.create_or_update_node(
        'spatial', 'test_apartment', {'x': 1, 'y': 1, 'z': 1},
        location_type='Kitchen',
        entities=['fridge', 'stove']
    )
    
    scm.create_or_update_node(
        'spatial', 'test_apartment', {'x': 1, 'y': 0, 'z': 1},
        location_type='Bedroom',
        entities=['bed']
    )
    
    # Visit nodes
    scm.visit_node('spatial', 'test_apartment', {'x': 1, 'y': 1, 'z': 1})
    scm.visit_node('spatial', 'test_apartment', {'x': 1, 'y': 1, 'z': 1})
    scm.visit_node('spatial', 'test_apartment', {'x': 1, 'y': 0, 'z': 1})
    
    # Link some fake memories
    scm.link_stm_memory("[spatial][test_apartment][1][1][1]", "[0.45][-0.32][0.61]", +0.42)
    scm.link_stm_memory("[spatial][test_apartment][1][1][1]", "[0.38][0.12][-0.42]", +0.38)
    
    # Get stats
    print("\n=== Statistics ===")
    stats = scm.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Get kitchen node
    print("\n=== Kitchen Node ===")
    kitchen = scm.get_node('spatial', 'test_apartment', {'x': 1, 'y': 1, 'z': 1})
    print(f"Location: {kitchen['location_type']}")
    print(f"Visits: {kitchen['visit_count']}")
    print(f"Valence: {kitchen['aggregate_valence']:.2f}")
    print(f"STM links: {len(kitchen['stm_coord_keys'])}")
    
    scm.close()
    print("\n✅ SCM test complete!")

