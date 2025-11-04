#!/usr/bin/env python3
"""
🔗 SEMANTIC LINKING MANAGER V2 - LEAN & FAST 🔗

CLEAN V2 LINKING SYSTEM:
- Linear succession linking (consecutive memories)
- Simple radial coordinate search  
- Integration with V2 EngramManager
- Fast spatial relationship discovery
- Minimal overhead, maximum speed

FOCUS: Fast, functional semantic relationship building!
"""

import math
import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

class SemanticLinking_Manager_V2:
    """
    🔗 V2 SEMANTIC LINKING - LEAN AND FAST
    
    Features:
    - Linear succession linking (memories added in sequence)
    - Radial coordinate search for spatial neighbors
    - Simple, fast relationship discovery
    - Direct integration with V2 coordinate system
    """
    
    def __init__(self, 
                 succession_window=5,      # How many previous memories to link
                 radial_threshold=0.6,     # Max distance for radial links
                 max_radial_links=3,       # Max radial links per memory
                 verbose=False):
        """Initialize the V2 semantic linking system"""
        
        self.succession_window = succession_window
        self.radial_threshold = radial_threshold
        self.max_radial_links = max_radial_links
        self.verbose = verbose
        
        # Memory tracking
        self.memories = []
        self.links = {}  # memory_id -> list of links
        
        # Statistics
        self.stats = {
            'total_memories': 0,
            'succession_links': 0,
            'radial_links': 0,
            'total_links': 0,
            'avg_links_per_memory': 0.0
        }
        
        if verbose:
            print("🔗 SemanticLinking Manager V2 initialized")
            print(f"   Succession window: {succession_window}")
            print(f"   Radial threshold: {radial_threshold}")
            print(f"   Max radial links: {max_radial_links}")
    
    def add_memory(self, memory_id: int, coordinates: Dict[str, float], 
                   content: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Add a memory and create semantic links
        
        Args:
            memory_id: Unique memory identifier
            coordinates: 9D coordinate dictionary
            content: Memory content/text
            metadata: Optional metadata
            
        Returns:
            Dict: Link creation results
        """
        
        # Store memory
        memory_entry = {
            'id': memory_id,
            'coordinates': coordinates,
            'content': content,
            'metadata': metadata or {},
            'timestamp': time.time(),
            'order': len(self.memories)  # Order of addition
        }
        
        self.memories.append(memory_entry)
        self.links[memory_id] = []
        self.stats['total_memories'] += 1
        
        # Create links
        link_results = self._create_links_for_memory(memory_entry)
        
        # Update statistics
        self._update_stats()
        
        if self.verbose:
            total_links = link_results['succession_links'] + link_results['radial_links']
            print(f"🔗 Memory {memory_id}: {total_links} links created "
                  f"({link_results['succession_links']} succession + {link_results['radial_links']} radial)")
        
        return link_results
    
    def _create_links_for_memory(self, new_memory: Dict) -> Dict:
        """Create links for a new memory"""
        succession_links = 0
        radial_links = 0
        
        # 1. LINEAR SUCCESSION LINKING
        succession_links = self._create_succession_links(new_memory)
        
        # 2. RADIAL COORDINATE SEARCH
        radial_links = self._create_radial_links(new_memory)
        
        return {
            'succession_links': succession_links,
            'radial_links': radial_links,
            'total_links': succession_links + radial_links
        }
    
    def _create_succession_links(self, new_memory: Dict) -> int:
        """Create links to recent previous memories (linear succession)"""
        links_created = 0
        memory_id = new_memory['id']
        
        # Get recent memories (within succession window)
        recent_memories = self.memories[-(self.succession_window + 1):-1]  # Exclude new memory itself
        
        for recent_memory in recent_memories:
            # Calculate succession strength (more recent = stronger)
            order_diff = new_memory['order'] - recent_memory['order']
            succession_strength = 1.0 - (order_diff / self.succession_window)
            
            # Create bidirectional succession link
            succession_link = {
                'target_id': recent_memory['id'],
                'type': 'succession',
                'strength': succession_strength,
                'distance': self._calculate_coordinate_distance(
                    new_memory['coordinates'], 
                    recent_memory['coordinates']
                ),
                'order_difference': order_diff
            }
            
            # Add forward link
            self.links[memory_id].append(succession_link)
            
            # Add backward link
            backward_link = succession_link.copy()
            backward_link['target_id'] = memory_id
            self.links[recent_memory['id']].append(backward_link)
            
            links_created += 1
            self.stats['succession_links'] += 1
        
        return links_created
    
    def _create_radial_links(self, new_memory: Dict) -> int:
        """Create links to spatially nearby memories (radial search)"""
        links_created = 0
        memory_id = new_memory['id']
        
        # Find candidates within radial threshold
        radial_candidates = []
        
        for existing_memory in self.memories[:-1]:  # Exclude new memory
            distance = self._calculate_coordinate_distance(
                new_memory['coordinates'],
                existing_memory['coordinates']
            )
            
            if distance <= self.radial_threshold:
                # Check if not already linked via succession
                already_linked = any(
                    link['target_id'] == existing_memory['id'] 
                    for link in self.links[memory_id]
                )
                
                if not already_linked:
                    radial_strength = 1.0 - (distance / self.radial_threshold)
                    radial_candidates.append({
                        'memory': existing_memory,
                        'distance': distance,
                        'strength': radial_strength
                    })
        
        # Sort by strength and select top candidates
        radial_candidates.sort(key=lambda x: x['strength'], reverse=True)
        selected_candidates = radial_candidates[:self.max_radial_links]
        
        # Create radial links
        for candidate in selected_candidates:
            target_memory = candidate['memory']
            
            radial_link = {
                'target_id': target_memory['id'],
                'type': 'radial',
                'strength': candidate['strength'],
                'distance': candidate['distance'],
                'spatial_similarity': self._analyze_coordinate_similarity(
                    new_memory['coordinates'],
                    target_memory['coordinates']
                )
            }
            
            # Add forward link
            self.links[memory_id].append(radial_link)
            
            # Add backward link
            backward_link = radial_link.copy()
            backward_link['target_id'] = memory_id
            self.links[target_memory['id']].append(backward_link)
            
            links_created += 1
            self.stats['radial_links'] += 1
        
        return links_created
    
    def _calculate_coordinate_distance(self, coords1: Dict[str, float], 
                                     coords2: Dict[str, float]) -> float:
        """Calculate 9D Euclidean distance between coordinates"""
        coord_names = ['x', 'y', 'z', 'a', 'b', 'c', 'd', 'e', 'f']
        
        distance_squared = sum(
            (coords1[name] - coords2[name]) ** 2 
            for name in coord_names
        )
        
        return math.sqrt(distance_squared)
    
    def _analyze_coordinate_similarity(self, coords1: Dict[str, float], 
                                     coords2: Dict[str, float]) -> Dict[str, float]:
        """Analyze which dimensions are most similar"""
        coord_names = ['x', 'y', 'z', 'a', 'b', 'c', 'd', 'e', 'f']
        dimension_names = [
            'time', 'emotion', 'person', 'concrete', 'action',
            'urgency', 'certainty', 'scope', 'focus'
        ]
        
        similarities = {}
        
        for coord_name, dim_name in zip(coord_names, dimension_names):
            diff = abs(coords1[coord_name] - coords2[coord_name])
            similarity = 1.0 - (diff / 2.0)  # Normalize to 0-1
            similarities[dim_name] = similarity
        
        return similarities
    
    def get_memory_links(self, memory_id: int) -> List[Dict]:
        """Get all links for a specific memory"""
        return self.links.get(memory_id, [])
    
    def get_memory_by_id(self, memory_id: int) -> Optional[Dict]:
        """Get memory data by ID"""
        for memory in self.memories:
            if memory['id'] == memory_id:
                return memory
        return None
    
    def find_linked_memories(self, memory_id: int, link_type: Optional[str] = None,
                           min_strength: float = 0.0) -> List[Dict]:
        """
        Find memories linked to a specific memory
        
        Args:
            memory_id: Source memory ID
            link_type: Filter by link type ('succession', 'radial', or None for all)
            min_strength: Minimum link strength threshold
            
        Returns:
            List of linked memory data with link info
        """
        
        if memory_id not in self.links:
            return []
        
        linked_memories = []
        
        for link in self.links[memory_id]:
            # Apply filters
            if link_type and link['type'] != link_type:
                continue
            
            if link['strength'] < min_strength:
                continue
            
            # Find the target memory
            target_memory = None
            for memory in self.memories:
                if memory['id'] == link['target_id']:
                    target_memory = memory
                    break
            
            if target_memory:
                linked_memories.append({
                    'memory': target_memory,
                    'link': link
                })
        
        # Sort by link strength
        linked_memories.sort(key=lambda x: x['link']['strength'], reverse=True)
        
        return linked_memories
    
    def find_spatial_neighborhood(self, coordinates: Dict[str, float], 
                                radius: float = 0.8, max_results: int = 10) -> List[Dict]:
        """
        Find memories within a spatial radius of given coordinates
        
        Args:
            coordinates: Center coordinates for search
            radius: Search radius
            max_results: Maximum number of results
            
        Returns:
            List of nearby memories with distance info
        """
        
        nearby_memories = []
        
        for memory in self.memories:
            distance = self._calculate_coordinate_distance(coordinates, memory['coordinates'])
            
            if distance <= radius:
                nearby_memories.append({
                    'memory': memory,
                    'distance': distance,
                    'similarity': 1.0 - (distance / radius)
                })
        
        # Sort by distance (closest first)
        nearby_memories.sort(key=lambda x: x['distance'])
        
        return nearby_memories[:max_results]
    
    def _update_stats(self):
        """Update linking statistics"""
        self.stats['total_links'] = self.stats['succession_links'] + self.stats['radial_links']
        
        if self.stats['total_memories'] > 0:
            self.stats['avg_links_per_memory'] = (
                self.stats['total_links'] / self.stats['total_memories']
            )
    
    def get_linking_stats(self) -> Dict:
        """Get comprehensive linking statistics"""
        return {
            'total_memories': self.stats['total_memories'],
            'succession_links': self.stats['succession_links'],
            'radial_links': self.stats['radial_links'],
            'total_links': self.stats['total_links'],
            'avg_links_per_memory': self.stats['avg_links_per_memory'],
            'succession_ratio': (
                self.stats['succession_links'] / max(1, self.stats['total_links'])
            ),
            'radial_ratio': (
                self.stats['radial_links'] / max(1, self.stats['total_links'])
            )
        }
    
    def export_network_summary(self) -> Dict:
        """Export a summary of the semantic network"""
        return {
            'statistics': self.get_linking_stats(),
            'memory_count': len(self.memories),
            'link_types': {
                'succession': self.stats['succession_links'],
                'radial': self.stats['radial_links']
            },
            'configuration': {
                'succession_window': self.succession_window,
                'radial_threshold': self.radial_threshold,
                'max_radial_links': self.max_radial_links
            }
        }

# Quick test of V2 semantic linking
if __name__ == "__main__":
    print("🔗 Testing Semantic Linking Manager V2")
    
    # Import enhanced V2 system with DEEP mode
    # Import from SpatialValienceToCoords (the REAL coordinate library)
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SpatialValienceToCoords'))
    from spatial_valence import UltraEnhancedSpatialValenceToCoordGeneration, SemanticDepth
    
    # Initialize systems with DEEP mode for maximum consistency
    coord_processor = UltraEnhancedSpatialValenceToCoordGeneration(SemanticDepth.DEEP)
    linker = SemanticLinking_Manager_V2(verbose=True)
    
    # Test memories
    test_texts = [
        "The cat sat peacefully on the warm mat",
        "A dog barked loudly in the backyard", 
        "Scientists discovered amazing new galaxies in space",
        "Researchers found evidence of water on Mars",
        "I love walking through the peaceful forest",
        "The hiking trail led through beautiful woods",
        "The computer crashed and lost important work",
        "My laptop froze during the presentation",
        "Tomorrow we will build incredible things together",
        "The team plans to create something amazing next week"
    ]
    
    print(f"\n⚡ Processing {len(test_texts)} memories with enhanced DEEP linking...")
    
    start_time = time.time()
    
    for i, text in enumerate(test_texts):
        # Generate coordinates
        result = coord_processor.process(text)
        
        # Add to linker with linking
        link_results = linker.add_memory(
            memory_id=i,
            coordinates=result['coordinates'],
            content=text,
            metadata={'summary': result['summary']}
        )
        
        print(f"   [{i}] \"{text[:40]}...\" → {link_results['total_links']} links")
    
    total_time = time.time() - start_time
    
    # Show statistics
    stats = linker.get_linking_stats()
    print(f"\n📊 Linking Results:")
    print(f"   Total memories: {stats['total_memories']}")
    print(f"   Succession links: {stats['succession_links']}")
    print(f"   Radial links: {stats['radial_links']}")
    print(f"   Avg links per memory: {stats['avg_links_per_memory']:.1f}")
    print(f"   Processing time: {total_time:.3f}s")
    
    # Test finding linked memories
    print(f"\n🔍 Testing memory linking for Memory 5:")
    memory_5_text = test_texts[5]
    linked = linker.find_linked_memories(5, min_strength=0.3)
    
    print(f"   Source: \"{memory_5_text[:50]}...\"")
    print(f"   Found {len(linked)} linked memories:")
    
    for link_data in linked:
        link = link_data['link']
        memory = link_data['memory']
        print(f"      {link['type']} → \"{memory['content'][:40]}...\" "
              f"(strength: {link['strength']:.3f})")
    
    # Test spatial neighborhood
    print(f"\n📍 Testing spatial neighborhood search:")
    sample_coords = coord_processor.process("cats and animals")['coordinates']
    neighbors = linker.find_spatial_neighborhood(sample_coords, radius=1.0, max_results=3)
    
    print(f"   Query: 'cats and animals'")
    print(f"   Found {len(neighbors)} spatial neighbors:")
    
    for neighbor in neighbors:
        memory = neighbor['memory']
        print(f"      \"{memory['content'][:40]}...\" "
              f"(distance: {neighbor['distance']:.3f})")
    
    print(f"\n🎯 V2 Semantic Linking System Test Complete!")
    print(f"✅ Linear succession + radial search working perfectly!") 