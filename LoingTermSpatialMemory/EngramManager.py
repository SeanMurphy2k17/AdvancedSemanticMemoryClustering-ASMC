#!/usr/bin/env python3
"""
🧠 ENGRAM MANAGER - CLEAN V2 SYSTEM 🧠

LEAN, FAST, FUNCTIONAL:
- Spatial Valence → Fast semantic summarization
- Hash Transformer → 9D coordinate generation 
- CoreData.lmdb → Clean key-based storage
- NO LLM BOTTLENECKS!

Main controller for the spatial memory system.
"""

import os
import time
import lmdb
import pickle
import json
import math
from typing import Dict, List, Optional
# Import from SpatialValienceToCoords (the REAL coordinate library)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SpatialValienceToCoords'))
from spatial_valence import UltraEnhancedSpatialValenceToCoordGeneration, SemanticDepth

# Import LTM-specific components (absolute imports - these libraries are always together)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from EnhancedDBManager import EnhancedDBManager
from SemanticLinking_Manager_V2 import SemanticLinking_Manager_V2

class EngramManager:
    """
    🎯 MAIN ENGRAM MANAGER - V2 CLEAN SYSTEM
    
    Coordinates between:
    1. SpatialValenceToCoordGeneration (semantic + 9D coords)
    2. EnhancedDBManager (database storage)
    3. Provides simple API for memory storage/retrieval
    """
    
    def __init__(self, 
                 db_path="DigitalEngramEdgeV2/CoreData.lmdb",
                 enable_linking=True,
                 turbo_mode=True,
                 verbose=True):
        """Initialize the clean Engram Manager"""
        
        if verbose:
            print("🧠" * 30)
            print("🧠 ENGRAM MANAGER V2 - CLEAN SYSTEM 🧠")
            print("🧠" * 30)
            print("🚀 Loading LEAN components with DEEP semantic analysis...")
        
        # Initialize REAL coordinate generation system with DEEP mode
        self.coord_system = UltraEnhancedSpatialValenceToCoordGeneration(SemanticDepth.DEEP)
        if verbose:
            print("✅ [1/3] Enhanced coordinate system ready with DEEP mode (Maximum consistency!)")
        
        # Initialize database manager
        self.db_manager = EnhancedDBManager(
            db_path=db_path,
            max_size=50 * 1024 * 1024 * 1024,  # 50GB for massive datasets
            turbo_mode=turbo_mode
        )
        if verbose:
            print("✅ [2/3] Database manager ready!")
        
        # Initialize semantic linking system
        self.enable_linking = enable_linking
        self.semantic_linker = None
        
        if enable_linking:
            self.semantic_linker = SemanticLinking_Manager_V2(
                succession_window=5,
                radial_threshold=0.6,
                max_radial_links=3,
                verbose=False  # SPEED: Always silent for bulk processing
            )
            if verbose:
                print("✅ [3/3] Semantic linking ready!")
        
        # Stats
        self.total_stored = 0
        self.total_retrieved = 0
        self.verbose = verbose
        
        # TURBO MODE: RAM cache for recent memories
        self.memory_cache = []  # Cache last N memories for fast linking
        self.cache_size = 10   # Keep last 10 memories in RAM (1 succession + 9 spatial candidates)
        self.pending_updates = {}  # Batch updates for efficiency
        
        # TURBO: Larger batches for MASSIVE databases
        self.batch_size = 500 if turbo_mode else 100  # Bigger batches for huge DBs
        
        # TURBO LINK TRACKING
        self.turbo_stats = {
            'succession_links': 0,
            'radial_links': 0,
            'total_links': 0,
            'memories_with_links': 0
        }
        
        if verbose:
            print("\n🎯 ENGRAM MANAGER V2 READY!")
            print("📍 Coordinate Keys: [x.xxx][y.yyy][z.zzz]...[f.fff]")
            print("🗄️ Database: Fast key-based LMDB storage")
            print(f"🔗 Semantic Linking: {'ENABLED' if enable_linking else 'DISABLED'}")
            print("⚡ Processing: Pure algorithmic (NO LLM bottlenecks)")
            if not verbose:
                print("🚀 SPEED MODE: Minimal logging for bulk processing")
            print("🧠" * 30)
    
    def store_memory(self, text: str, metadata: Optional[Dict] = None) -> Optional[int]:
        """
        Store text in spatial memory system
        
        Args:
            text: Input text to store
            metadata: Optional metadata dictionary
            
        Returns:
            int: Memory ID if successful, None if failed
        """
        try:
            # Process text through coordinate system
            result = self.coord_system.process(text)
            
            # Prepare storage data
            storage_data = {
                'input_text': text,
                'semantic_summary': result['summary'],
                'coordinates': result['coordinates'],
                'coordinate_key': result['coordinate_key'],
                'semantic_keys': result['semantic_keys'],
                'processing_time': result['processing_time']
            }
            
            # Add metadata if provided
            if metadata:
                storage_data['metadata'] = metadata
            
            # Generate memory_id first
            memory_id = self.db_manager.stats['total_memories']
            
            # TURBO MODE: Create links using RAM cache for speed
            embedded_links = {'succession_links': [], 'radial_links': [], 'total_links': 0}
            
            if self.enable_linking:
                embedded_links = self._create_turbo_links(memory_id, result['coordinates'], text)
                storage_data['semantic_links'] = embedded_links
            
            # Store in database WITH embedded links (single write)
            final_memory_id = self.db_manager.store_memory(
                input_text=text,
                semantic_summary=result['summary'],
                coordinates=result['coordinates'],
                metadata=storage_data
            )
            
            if final_memory_id is not None:
                self.total_stored += 1
                
                # Add to RAM cache for future linking
                cache_entry = {
                    'id': final_memory_id,
                    'coordinates': result['coordinates'],
                    'content': text,
                    'coord_key': result['coordinate_key'],
                    'storage_data': storage_data.copy()
                }
                
                self.memory_cache.append(cache_entry)
                
                # Keep cache at reasonable size
                if len(self.memory_cache) > self.cache_size:
                    self.memory_cache.pop(0)
                
                # TURBO: Batch update previous memories with backward links
                self._queue_backward_link_updates(final_memory_id, embedded_links)
                
                if self.verbose:
                    links_count = embedded_links.get('total_links', 0)
                    print(f"🧠 Memory {final_memory_id} stored: {result['coordinate_key']} | "
                          f"Links: {links_count} embedded")
            
            return final_memory_id
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Storage failed: {e}")
            return None
    
    def retrieve_by_coordinates(self, coordinates: Dict[str, float]) -> Optional[Dict]:
        """
        Retrieve memory by exact coordinates
        
        Args:
            coordinates: 9D coordinate dictionary
            
        Returns:
            Dict: Memory data if found, None if not found
        """
        try:
            result = self.db_manager.get_memory_by_coordinates(coordinates)
            
            if result:
                self.total_retrieved += 1
                if self.verbose:
                    coord_key = self.coord_system.generate_coordinate_key(coordinates)
                    print(f"🔍 Retrieved memory: {coord_key}")
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Retrieval failed: {e}")
            return None
    
    def search_similar(self, query_text: str, max_results: int = 5) -> List[Dict]:
        """
        Search for similar memories using coordinate-based search
        
        Args:
            query_text: Query text
            max_results: Maximum number of results
            
        Returns:
            List[Dict]: Similar memories
        """
        try:
            # Process query to get coordinates
            query_result = self.coord_system.process(query_text)
            query_coords = query_result['coordinates']
            
            # Search database
            results = self.db_manager.search_by_coordinates(
                query_coords=query_coords,
                radius=0.5,  # Search radius in coordinate space
                max_results=max_results,
                search_strategy='radius'
            )
            
            if results:
                self.total_retrieved += len(results)
                if self.verbose:
                    print(f"🔍 Found {len(results)} similar memories")
            
            return results
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Search failed: {e}")
            return []
    
    def process_text_list(self, text_list: List[str], show_progress: bool = True) -> Dict:
        """
        Process a list of texts for bulk storage
        
        Args:
            text_list: List of texts to process
            show_progress: Show progress during processing
            
        Returns:
            Dict: Processing results and statistics
        """
        start_time = time.time()
        stored_count = 0
        failed_count = 0
        
        if show_progress:
            print(f"📚 Processing {len(text_list)} texts...")
        
        for i, text in enumerate(text_list):
            if show_progress and i % 100 == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                print(f"   Progress: {i}/{len(text_list)} | Rate: {rate:.1f} texts/sec")
            
            memory_id = self.store_memory(text)
            
            if memory_id is not None:
                stored_count += 1
            else:
                failed_count += 1
        
        total_time = time.time() - start_time
        
        results = {
            'total_processed': len(text_list),
            'stored_successfully': stored_count,
            'failed': failed_count,
            'processing_time': total_time,
            'rate': len(text_list) / total_time if total_time > 0 else 0
        }
        
        if show_progress:
            print(f"✅ Bulk processing complete!")
            print(f"   Stored: {stored_count}/{len(text_list)}")
            print(f"   Rate: {results['rate']:.1f} texts/second")
        
        return results
    
    def get_linked_memories(self, memory_id: int, link_type: Optional[str] = None,
                           min_strength: float = 0.0) -> List[Dict]:
        """
        Get memories linked to a specific memory (requires semantic linking)
        
        Args:
            memory_id: Source memory ID
            link_type: Filter by link type ('succession', 'radial', or None for all)
            min_strength: Minimum link strength threshold
            
        Returns:
            List of linked memory data
        """
        if not self.enable_linking or not self.semantic_linker:
            if self.verbose:
                print("⚠️ Semantic linking not enabled")
            return []
        
        return self.semantic_linker.find_linked_memories(
            memory_id=memory_id,
            link_type=link_type,
            min_strength=min_strength
        )
    
    def find_spatial_neighbors(self, coordinates: Dict[str, float], 
                              radius: float = 0.8, max_results: int = 10) -> List[Dict]:
        """
        Find memories within spatial radius (requires semantic linking)
        
        Args:
            coordinates: Center coordinates for search
            radius: Search radius
            max_results: Maximum number of results
            
        Returns:
            List of nearby memories
        """
        if not self.enable_linking or not self.semantic_linker:
            if self.verbose:
                print("⚠️ Semantic linking not enabled")
            return []
        
        return self.semantic_linker.find_spatial_neighborhood(
            coordinates=coordinates,
            radius=radius,
            max_results=max_results
        )
    
    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        coord_stats = self.coord_system.get_stats()
        db_stats = self.db_manager.get_memory_statistics()
        
        stats = {
            'total_stored': self.total_stored,
            'total_retrieved': self.total_retrieved,
            'coordinate_cache_size': coord_stats['cache_size'],
            'coordinate_cache_rate': coord_stats['cache_rate'],
            'database_memories': db_stats['total_memories'],
            'database_size_mb': db_stats['database_size_mb'],
            'semantic_linking_enabled': self.enable_linking
        }
        
        # Add semantic linking stats if enabled
        if self.enable_linking:
            # TURBO: Use TURBO stats instead of semantic_linker
            avg_links = self.turbo_stats['total_links'] / max(1, self.total_stored)
            stats.update({
                'succession_links': self.turbo_stats['succession_links'],
                'radial_links': self.turbo_stats['radial_links'],
                'total_semantic_links': self.turbo_stats['total_links'],
                'avg_links_per_memory': round(avg_links, 2)
            })
        
        return stats
    
    def _create_embedded_links_data(self, links: List[Dict]) -> Dict:
        """
        🔗 CREATE EMBEDDED LINKS DATA WITH SUMMARIES
        
        Creates the embedded links structure that will be stored with the memory
        
        Args:
            links: List of link dictionaries from semantic linker
            
        Returns:
            Dict: Embedded links structure with summaries
        """
        embedded_links = {
            'succession_links': [],
            'radial_links': [],
            'total_links': len(links)
        }
        
        for link in links:
            # Find the target memory to get its content
            target_memory = self.semantic_linker.get_memory_by_id(link['target_id'])
            
            if target_memory:
                # Create coordinate key for the linked memory
                target_coord_key = self.coord_system.generate_coordinate_key(target_memory['coordinates'])
                
                # Generate summary title from target memory content
                target_content = target_memory.get('content', '')
                target_summary = target_content[:100] + "..." if len(target_content) > 100 else target_content
                
                embedded_link = {
                    'target_memory_id': link['target_id'],
                    'target_coordinate_key': target_coord_key,
                    'target_coordinates': target_memory['coordinates'],
                    'summary': target_summary,  # Add summary title!
                    'link_type': link['type'],
                    'strength': round(link['strength'], 3),
                    'distance': round(link['distance'], 3)
                }
                
                # Add to appropriate category
                if link['type'] == 'succession':
                    embedded_links['succession_links'].append(embedded_link)
                elif link['type'] == 'radial':
                    embedded_links['radial_links'].append(embedded_link)
        
        return embedded_links
    
    def _create_turbo_links(self, memory_id: int, coordinates: Dict[str, float], content: str) -> Dict:
        """
        🚀 TURBO LINKING: Create links using RAM cache for maximum speed
        
        Uses recently cached memories instead of database reads for linking.
        Creates succession links to last 5 memories and spatial links within threshold.
        
        Args:
            memory_id: New memory ID
            coordinates: New memory coordinates  
            content: New memory content
            
        Returns:
            Dict: Embedded links structure
        """
        embedded_links = {'succession_links': [], 'radial_links': [], 'total_links': 0}
        
        if not self.memory_cache:
            return embedded_links
        
        # LINEAR SUCCESSION LINKS: Link ONLY to immediate predecessor (true chain)
        if self.memory_cache:
            # Get the immediately previous memory (last in cache)
            previous_memory = self.memory_cache[-1]
            
            # Calculate distance to immediate predecessor
            distance = self._calculate_coordinate_distance(coordinates, previous_memory['coordinates'])
            
            # Create single backward link to immediate predecessor
            succession_link = {
                'target_memory_id': previous_memory['id'],
                'target_coordinate_key': previous_memory['coord_key'],
                'target_coordinates': previous_memory['coordinates'],
                'summary': previous_memory['content'][:100] + "...",
                'link_type': 'succession',
                'strength': 0.9,  # High strength for immediate succession
                'distance': round(distance, 3)
            }
            
            embedded_links['succession_links'].append(succession_link)
        
        # RADIAL LINKS: Find spatially similar memories within threshold
        radial_threshold = 0.6
        max_radial_links = 3
        radial_candidates = []
        
        for cached_memory in self.memory_cache[:-1]:  # Exclude immediate predecessor
            distance = self._calculate_coordinate_distance(coordinates, cached_memory['coordinates'])
            
            if distance <= radial_threshold:
                radial_strength = 1.0 - (distance / radial_threshold)
                radial_candidates.append({
                    'memory': cached_memory,
                    'distance': distance,
                    'strength': radial_strength
                })
        
        # Sort by strength and take top candidates
        radial_candidates.sort(key=lambda x: x['strength'], reverse=True)
        selected_radial = radial_candidates[:max_radial_links]
        
        for candidate in selected_radial:
            cached_memory = candidate['memory']
            
            radial_link = {
                'target_memory_id': cached_memory['id'],
                'target_coordinate_key': cached_memory['coord_key'],
                'target_coordinates': cached_memory['coordinates'],
                'summary': cached_memory['content'][:100] + "...",
                'link_type': 'radial',
                'strength': round(candidate['strength'], 3),
                'distance': round(candidate['distance'], 3)
            }
            
            embedded_links['radial_links'].append(radial_link)
        
        # Update total count
        embedded_links['total_links'] = len(embedded_links['succession_links']) + len(embedded_links['radial_links'])
        
        # TURBO: Update link statistics
        self.turbo_stats['succession_links'] += len(embedded_links['succession_links'])
        self.turbo_stats['radial_links'] += len(embedded_links['radial_links'])
        self.turbo_stats['total_links'] += embedded_links['total_links']
        if embedded_links['total_links'] > 0:
            self.turbo_stats['memories_with_links'] += 1
        
        return embedded_links
    
    def _calculate_coordinate_distance(self, coords1: Dict[str, float], coords2: Dict[str, float]) -> float:
        """Calculate 9D Euclidean distance between coordinates"""
        distance = 0.0
        axes = ['x', 'y', 'z', 'a', 'b', 'c', 'd', 'e', 'f']
        
        for axis in axes:
            val1 = coords1.get(axis, 0.0)
            val2 = coords2.get(axis, 0.0)
            distance += (val1 - val2) ** 2
        
        return math.sqrt(distance)
    
    def _queue_backward_link_updates(self, new_memory_id: int, forward_links: Dict):
        """
        🚀 TURBO: Queue backward link updates for batch processing
        
        Instead of immediately writing each backward link, queue them
        for batch processing every N memories for speed.
        """
        for link in forward_links['succession_links'] + forward_links['radial_links']:
            target_id = link['target_memory_id']
            
            # Find target in cache
            target_cache_entry = None
            for cached_memory in self.memory_cache:
                if cached_memory['id'] == target_id:
                    target_cache_entry = cached_memory
                    break
            
            if target_cache_entry:
                # Create backward link
                backward_link = {
                    'target_memory_id': new_memory_id,
                    'target_coordinate_key': self.coord_system.generate_coordinate_key(self._get_last_coordinates()),
                    'target_coordinates': self._get_last_coordinates(),
                    'summary': self._get_last_content()[:100] + "...",
                    'link_type': link['link_type'],
                    'strength': link['strength'],
                    'distance': link['distance']
                }
                
                # Queue update for target memory
                if target_id not in self.pending_updates:
                    self.pending_updates[target_id] = {
                        'coordinates': target_cache_entry['coordinates'],
                        'new_links': []
                    }
                
                self.pending_updates[target_id]['new_links'].append(backward_link)
        
        # Process batch updates with dynamic batch size for efficiency
        if len(self.pending_updates) >= self.batch_size:
            self._process_batch_updates()
    
    def _get_last_coordinates(self) -> Dict[str, float]:
        """Get coordinates of the most recently added memory"""
        if self.memory_cache:
            return self.memory_cache[-1]['coordinates']
        return {}
    
    def _get_last_content(self) -> str:
        """Get content of the most recently added memory"""
        if self.memory_cache:
            return self.memory_cache[-1]['content']
        return ""
    
    def _process_batch_updates(self):
        """
        🚀 TURBO: Process queued backward link updates in a single transaction
        
        This dramatically reduces database writes by batching all updates.
        """
        if not self.pending_updates:
            return
        
        updated_count = 0
        
        # Use single transaction for all updates
        with self.db_manager.env.begin(write=True) as txn:
            for target_id, update_info in self.pending_updates.items():
                try:
                    # Retrieve current memory
                    coord_key = self.db_manager._create_coordinate_key(update_info['coordinates'])
                    memory_value = txn.get(coord_key)
                    
                    if memory_value:
                        stored_memory = pickle.loads(memory_value)
                        
                        # Initialize semantic_links if not exists
                        if 'semantic_links' not in stored_memory:
                            stored_memory['semantic_links'] = {
                                'succession_links': [],
                                'radial_links': [],
                                'total_links': 0
                            }
                        
                        # Add new backward links
                        for new_link in update_info['new_links']:
                            if new_link['link_type'] == 'succession':
                                stored_memory['semantic_links']['succession_links'].append(new_link)
                            elif new_link['link_type'] == 'radial':
                                stored_memory['semantic_links']['radial_links'].append(new_link)
                        
                        # Update total count
                        total_links = (len(stored_memory['semantic_links']['succession_links']) + 
                                     len(stored_memory['semantic_links']['radial_links']))
                        stored_memory['semantic_links']['total_links'] = total_links
                        
                        # Write back to database
                        updated_value = pickle.dumps(stored_memory)
                        txn.put(coord_key, updated_value)
                        updated_count += 1
                        
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ Failed to update memory {target_id}: {e}")
        
        if self.verbose and updated_count > 0:
            print(f"🚀 TURBO: Batch updated {updated_count} memories with backward links")
        
        # Clear processed updates
        self.pending_updates.clear()
    
    def _embed_links_in_stored_memory(self, memory_id: int, coordinates: Dict[str, float], links: List[Dict]):
        """
        🔗 EMBED LINKS DIRECTLY INTO STORED MEMORY DATA WITH SUMMARIES
        
        This ensures that when you retrieve a memory by coordinates,
        you get ALL related coordinate keys AND summaries in a single lookup!
        
        Args:
            memory_id: The memory ID to update
            coordinates: The memory's coordinates
            links: List of link dictionaries from semantic linker
        """
        try:
            # Retrieve the current memory data
            stored_memory = self.db_manager.get_memory_by_coordinates(coordinates)
            
            if stored_memory:
                # Create embedded link data with coordinate keys AND summaries
                embedded_links = {
                    'succession_links': [],
                    'radial_links': [],
                    'total_links': len(links)
                }
                
                for link in links:
                    # Find the target memory to get its coordinates and content
                    target_memory = self.semantic_linker.get_memory_by_id(link['target_id'])
                    
                    if target_memory:
                        # Create coordinate key for the linked memory
                        target_coord_key = self.coord_system.generate_coordinate_key(target_memory['coordinates'])
                        
                        # Generate summary title from target memory content
                        target_content = target_memory.get('content', '')
                        target_summary = target_content[:100] + "..." if len(target_content) > 100 else target_content
                        
                        embedded_link = {
                            'target_memory_id': link['target_id'],
                            'target_coordinate_key': target_coord_key,
                            'target_coordinates': target_memory['coordinates'],
                            'summary': target_summary,  # Add summary title!
                            'link_type': link['type'],
                            'strength': round(link['strength'], 3),
                            'distance': round(link['distance'], 3)
                        }
                        
                        # Add to appropriate category
                        if link['type'] == 'succession':
                            embedded_links['succession_links'].append(embedded_link)
                        elif link['type'] == 'radial':
                            embedded_links['radial_links'].append(embedded_link)
                
                # Add embedded links to the stored memory data
                stored_memory['semantic_links'] = embedded_links
                
                # Re-store the memory with embedded links using coordinate key
                coord_key = self._create_coordinate_key(coordinates)
                
                # Store using enhanced DB manager with the coordinate key
                with self.db_manager.env.begin(write=True) as txn:
                    memory_value = pickle.dumps(stored_memory)
                    txn.put(coord_key, memory_value)
                
                if self.verbose:
                    print(f"      🔗 Embedded {len(links)} links with summaries into memory {memory_id}")
                    
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to embed links for memory {memory_id}: {e}")
    
    def _create_coordinate_key(self, coordinates):
        """Create coordinate key for direct database storage"""
        import json
        axes = ['x', 'y', 'z', 'a', 'b', 'c', 'd', 'e', 'f']
        coord_values = []
        
        for axis in axes:
            if axis in coordinates:
                coord_values.append(round(float(coordinates[axis]), 3))
            else:
                coord_values.append(0.0)
        
        coord_key = json.dumps(coord_values)
        return coord_key.encode()

    def _update_backward_links(self, new_memory_id: int):
        """
        🔄 UPDATE PREVIOUSLY STORED MEMORIES WITH BACKWARD LINKS
        
        When a new memory links to previous memories, we need to update
        those previous memories to include backward links to the new memory.
        """
        try:
            # Get links from the new memory
            new_memory_links = self.semantic_linker.get_memory_links(new_memory_id)
            
            for link in new_memory_links:
                target_id = link['target_id']
                
                # Get the target memory from semantic linker
                target_memory_data = self.semantic_linker.get_memory_by_id(target_id)
                
                if target_memory_data:
                    target_coords = target_memory_data['coordinates']
                    
                    # Retrieve stored target memory
                    stored_target = self.db_manager.get_memory_by_coordinates(target_coords)
                    
                    if stored_target:
                        # Initialize semantic_links if not exists
                        if 'semantic_links' not in stored_target:
                            stored_target['semantic_links'] = {
                                'succession_links': [],
                                'radial_links': [],
                                'total_links': 0
                            }
                        
                        # Get new memory data for backward link
                        new_memory_data = self.semantic_linker.get_memory_by_id(new_memory_id)
                        
                        if new_memory_data:
                            new_coord_key = self.coord_system.generate_coordinate_key(new_memory_data['coordinates'])
                            
                            # Create backward link
                            backward_link = {
                                'target_memory_id': new_memory_id,
                                'target_coordinate_key': new_coord_key,
                                'target_coordinates': new_memory_data['coordinates'],
                                'link_type': link['type'],
                                'strength': link['strength'],
                                'distance': link['distance']
                            }
                            
                            # Add to appropriate category
                            if link['type'] == 'succession':
                                stored_target['semantic_links']['succession_links'].append(backward_link)
                            elif link['type'] == 'radial':
                                stored_target['semantic_links']['radial_links'].append(backward_link)
                            
                            # Update total count
                            total_links = (len(stored_target['semantic_links']['succession_links']) + 
                                         len(stored_target['semantic_links']['radial_links']))
                            stored_target['semantic_links']['total_links'] = total_links
                            
                            # Re-store the updated memory
                            self.db_manager.store_memory_engram(stored_target)
                            
                            if self.verbose:
                                print(f"      🔄 Updated backward links for memory {target_id}")
                        
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to update backward links: {e}")

    def switch_to_safe_mode(self):
        """
        🛡️ SWITCH TO SAFE MODE - Enable data durability
        
        Call this after bulk loading to ensure data safety for normal operations.
        """
        # Process any pending updates first
        if self.pending_updates:
            print(f"🔄 Processing {len(self.pending_updates)} pending updates before mode switch...")
            self._process_batch_updates()
        
        self.db_manager.switch_to_safe_mode()
        print("🛡️ Engram Manager switched to SAFE MODE")
    
    def switch_to_turbo_mode(self):
        """
        🚀 SWITCH TO TURBO MODE - Enable bulk loading optimizations
        
        Use this when you need maximum speed for bulk operations.
        """
        self.db_manager.switch_to_turbo_mode()
        print("🚀 Engram Manager switched to TURBO MODE")
    
    def force_sync(self):
        """
        💾 FORCE SYNC - Create safety checkpoint
        
        Manually flush all pending writes to disk during bulk loading.
        """
        # Process pending updates first
        if self.pending_updates:
            print(f"🔄 Processing {len(self.pending_updates)} pending updates...")
            self._process_batch_updates()
        
        self.db_manager.force_sync()
        print("💾 Engram Manager synced to disk")
    
    def get_mode_info(self):
        """Get current database mode and performance info"""
        db_info = self.db_manager.get_mode_info()
        return {
            **db_info,
            'batch_size': self.batch_size,
            'cache_size': self.cache_size,
            'pending_updates': len(self.pending_updates),
            'cached_memories': len(self.memory_cache)
        }

    def cleanup(self):
        """Clean up system resources"""
        
        # TURBO: Process any remaining batch updates
        if self.pending_updates:
            if self.verbose:
                print(f"🚀 TURBO: Processing final {len(self.pending_updates)} batch updates...")
            self._process_batch_updates()
        
        self.db_manager.close()
        if self.verbose:
            print("🧹 Engram Manager V2 cleanup complete")
            print(f"📊 Cache processed: {len(self.memory_cache)} recent memories")

# Quick test
if __name__ == "__main__":
    print("🧠 Testing Engram Manager V2")
    
    # Initialize manager
    manager = EngramManager(db_path="test_coredata.lmdb", verbose=True)
    
    # Test storage
    test_texts = [
        "The cat sat on the mat peacefully",
        "Scientists discovered amazing new galaxies",
        "I love walking through peaceful forests",
        "The computer crashed and lost important work",
        "Tomorrow we will build incredible things together"
    ]
    
    print(f"\n📚 Storing {len(test_texts)} test memories...")
    
    for text in test_texts:
        memory_id = manager.store_memory(text)
        print(f"   Stored: {memory_id}")
    
    # Test search
    print(f"\n🔍 Testing search...")
    results = manager.search_similar("cats and animals", max_results=2)
    for result in results:
        print(f"   Found: {result['input_text'][:50]}...")
    
    # Show stats
    stats = manager.get_system_stats()
    print(f"\n📊 System Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    manager.cleanup()
    print("🎯 Engram Manager V2 test complete!") 