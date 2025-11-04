#!/usr/bin/env python3
"""
🗄️ ENHANCED DATABASE MANAGER - SPATIAL MEMORY STORAGE 🗄️

MIT License

Copyright (c) 2025 Spatial Memory Project Contributors
                   AI Democratization Initiative

🚀 SIMPLIFIED COORDINATE STORAGE 🚀
Easy peasy pumpkin squeezy coordinate lookups using [x][y][z] keys!
Based on the clean DBManager.py approach that actually works!

🎯 CORE FEATURES 🎯
- Simple coordinate-based keys
- Fast LMDB storage
- Clean 9D coordinate indexing  
- No complex KD-trees - just simple lookups!
"""

import lmdb
import json
import pickle
import os
import math
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

class EnhancedDBManager:
    def __init__(self, db_path="enhanced_memory.lmdb", max_size=50 * 1024 * 1024 * 1024, turbo_mode=True):
        """
        Enhanced database manager with SIMPLE coordinate-based storage
        
        Args:
            db_path: Path to the LMDB database
            max_size: Maximum database size in bytes (50GB default)
            turbo_mode: Use TURBO settings for bulk loading vs SAFE settings for production
        """
        self.db_path = db_path
        self.turbo_mode = turbo_mode
        os.makedirs(db_path, exist_ok=True)
        
        # SIMPLE APPROACH - Single database with coordinate keys!
        if turbo_mode:
            # TURBO SETTINGS for MASSIVE databases (882k+ memories)
            self.env = lmdb.open(
                db_path, 
                map_size=max_size, 
                writemap=True,
                sync=False,        # TURBO: Disable sync for bulk loading
                metasync=False,    # TURBO: Disable metadata sync  
                map_async=True,    # TURBO: Async memory mapping
                readahead=False    # TURBO: Disable readahead for random access
            )
        else:
            # SAFE SETTINGS for production use
            self.env = lmdb.open(
                db_path, 
                map_size=max_size, 
                writemap=True,
                sync=True,         # SAFE: Enable sync for data durability
                metasync=True,     # SAFE: Enable metadata sync
                map_async=False,   # SAFE: Synchronous memory mapping
                readahead=True     # SAFE: Enable readahead for sequential access
            )
        
        self.stats = {
            'total_memories': 0,
            'last_access_time': time.time(),
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        self._load_stats()
    
    def _create_coordinate_key(self, coordinates):
        """
        Create a simple coordinate key like DBManager.py
        Uses JSON encoding of coordinate values for exact lookups
        """
        # Extract coordinate values in consistent order
        axes = ['x', 'y', 'z', 'a', 'b', 'c', 'd', 'e', 'f']
        coord_values = []
        
        for axis in axes:
            if axis in coordinates:
                # Round to 3 decimal places for consistent keys
                coord_values.append(round(float(coordinates[axis]), 3))
            else:
                coord_values.append(0.0)
        
        # Create JSON key like DBManager.py
        coord_key = json.dumps(coord_values)
        return coord_key.encode()
    
    def _decode_coordinate_key(self, coord_key_bytes):
        """Decode coordinate key back to dictionary"""
        coord_values = json.loads(coord_key_bytes.decode())
        axes = ['x', 'y', 'z', 'a', 'b', 'c', 'd', 'e', 'f']
        
        coordinates = {}
        for i, axis in enumerate(axes):
            coordinates[axis] = coord_values[i] if i < len(coord_values) else 0.0
        
        return coordinates
    
    def store_memory_engram(self, memory_data):
        """
        Store memory using COORDINATE KEY approach - Simple and clean!
        
        Args:
            memory_data: Dict containing input, semantic, coordinates, id, etc.
        """
        # Get coordinates and create key
        coordinates = memory_data['coordinates']
        coord_key = self._create_coordinate_key(coordinates)
        
        # NUCLEAR SANITIZATION - Convert everything to safe types
        sanitized_memory_data = {
            'id': int(memory_data.get('id', self.stats['total_memories'])),
            'input': str(memory_data.get('input', '')),
            'input_text': str(memory_data.get('input_text', memory_data.get('input', ''))),
            'semantic': str(memory_data.get('semantic', '')),
            'semantic_summary': str(memory_data.get('semantic_summary', memory_data.get('semantic', ''))),
            'coordinates': coordinates,
            'timestamp': float(memory_data.get('timestamp', time.time())),
            'metadata': self._sanitize_metadata(memory_data.get('metadata', {}))
        }
        
        # PRESERVE SEMANTIC LINKS AT TOP LEVEL - CRITICAL FOR SINGLE LOOKUP!
        if 'semantic_links' in memory_data:
            sanitized_memory_data['semantic_links'] = memory_data['semantic_links']
        
        # Store directly with coordinate key - SIMPLE!
        with self.env.begin(write=True) as txn:
            memory_value = pickle.dumps(sanitized_memory_data)
            txn.put(coord_key, memory_value)
            
            # Update stats
            self.stats['total_memories'] += 1
            self._save_stats(txn)
        
        return sanitized_memory_data['id']
    
    def store_memory(self, input_text, semantic_summary, coordinates, metadata=None):
        """
        Store a memory with the simplified interface
        
        Args:
            input_text: The original input text
            semantic_summary: Semantic summary of the text
            coordinates: 9D coordinate dictionary
            metadata: Optional metadata dictionary
            
        Returns:
            memory_id: The ID of the stored memory
        """
        # Generate a new memory ID
        memory_id = self.stats['total_memories']
        
        # Create memory data structure
        memory_data = {
            'id': memory_id,
            'input': input_text,
            'input_text': input_text,
            'semantic': semantic_summary,
            'semantic_summary': semantic_summary,
            'coordinates': coordinates,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        # Store using coordinate key
        return self.store_memory_engram(memory_data)
    
    def get_memory_by_coordinates(self, coordinates, tolerance=0.001):
        """
        Retrieve memory by coordinates - SIMPLE LOOKUP!
        
        Args:
            coordinates: Dict with x,y,z,a,b,c,d,e,f values
            tolerance: Tolerance for approximate matching
        """
        # Try exact match first
        coord_key = self._create_coordinate_key(coordinates)
        
        with self.env.begin() as txn:
            memory_value = txn.get(coord_key)
            
            if memory_value:
                self.stats['cache_hits'] += 1
                return pickle.loads(memory_value)
            
            # If no exact match and tolerance > 0, try approximate
            if tolerance > 0:
                return self._find_approximate_match(coordinates, tolerance, txn)
        
        self.stats['cache_misses'] += 1
        return None
    
    def get_memory_by_id(self, memory_id):
        """
        Retrieve memory by ID - scan through all memories
        (Less efficient but sometimes needed)
        """
        with self.env.begin() as txn:
            cursor = txn.cursor()
            
            for coord_key, memory_value in cursor:
                try:
                    memory_data = pickle.loads(memory_value)
                    if memory_data.get('id') == memory_id:
                        self.stats['cache_hits'] += 1
                        return memory_data
                except:
                    continue  # Skip corrupted entries
        
        self.stats['cache_misses'] += 1
        return None
    
    def find_memories_in_region(self, center_coords, radius=1.0, max_results=50):
        """
        Find memories within a radius - Simple distance calculation
        
        Args:
            center_coords: Center coordinates for search
            radius: Search radius in 9D space
            max_results: Maximum number of results to return
        """
        found_memories = []
        
        with self.env.begin() as txn:
            cursor = txn.cursor()
            
            for coord_key, memory_value in cursor:
                try:
                    # Decode coordinates from key
                    coords = self._decode_coordinate_key(coord_key)
                    
                    # Calculate distance
                    distance = self._calculate_distance(center_coords, coords)
                    
                    if distance <= radius:
                        memory_data = pickle.loads(memory_value)
                        found_memories.append({
                            'memory': memory_data,
                            'distance': distance,
                            'coordinates': coords
                        })
        
                        if len(found_memories) >= max_results:
                            break
                            
                except:
                    continue  # Skip corrupted entries
        
        # Sort by distance
        found_memories.sort(key=lambda x: x['distance'])
        return found_memories
    
    def find_nearest_memories(self, query_coords, k=10):
        """
        Find k nearest memories - Simple brute force approach
        
        Args:
            query_coords: Query coordinates
            k: Number of nearest neighbors to find
        """
        all_distances = []
        
        with self.env.begin() as txn:
            cursor = txn.cursor()
            
            for coord_key, memory_value in cursor:
                try:
                    # Decode coordinates from key
                    coords = self._decode_coordinate_key(coord_key)
                    
                    # Calculate distance
                    distance = self._calculate_distance(query_coords, coords)
                    memory_data = pickle.loads(memory_value)
                    
                    all_distances.append({
                        'memory': memory_data,
                        'distance': distance,
                        'coordinates': coords
                    })
                    
                except:
                    continue  # Skip corrupted entries
        
        # Sort by distance and return top k
        all_distances.sort(key=lambda x: x['distance'])
        return all_distances[:k]
    
    def search_semantic_content(self, query_text, max_results=20):
        """
        Search memories by semantic content
        """
        query_lower = query_text.lower()
        matching_memories = []
        
        with self.env.begin() as txn:
            cursor = txn.cursor()
            
            for coord_key, memory_value in cursor:
                try:
                    memory_data = pickle.loads(memory_value)
                    
                    # Check input text and semantic summary
                    input_text = memory_data.get('input', '').lower()
                    semantic_text = memory_data.get('semantic', '').lower()
                    
                    if (query_lower in input_text or query_lower in semantic_text):
                        matching_memories.append(memory_data)
                        
                        if len(matching_memories) >= max_results:
                            break
                                
                except:
                    continue  # Skip corrupted entries
        
        return matching_memories
    
    def list_all_coordinate_keys(self):
        """
        List all coordinate keys in the database - like DBManager.py
        """
        keys = []
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for coord_key, _ in cursor:
                try:
                    coords = self._decode_coordinate_key(coord_key)
                    keys.append(coords)
                except:
                    continue  # Skip corrupted keys
        return keys
    
    def get_memory_statistics(self):
        """Get database statistics"""
        memory_count = 0
        
        with self.env.begin() as txn:
            cursor = txn.cursor()
            memory_count = sum(1 for _ in cursor)
            stat = txn.stat()
            
        return {
            'total_memories': memory_count,
            'database_size_mb': stat.get('ms_entries', 0) * stat.get('ms_psize', 1) / (1024 * 1024),
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['cache_hits'] + self.stats['cache_misses'], 1),
            'last_access': self.stats['last_access_time']
        }

    def search_by_coordinates(self, query_coords, radius=1.0, max_results=50, 
                             search_strategy='radius'):
        """
        🎯 SIMPLE COORDINATE SEARCH - Clean and fast!
        
        Args:
            query_coords: Query coordinates (9D dict)
            radius: Search radius 
            max_results: Maximum results to return
            search_strategy: 'radius' or 'nearest'
            
        Returns:
            List of memories with format: [{'data': memory, 'distance': float}]
        """
        if search_strategy == 'radius':
            raw_results = self.find_memories_in_region(
                center_coords=query_coords,
                radius=radius,
                max_results=max_results
            )
            
            return [
                {
                    'data': result['memory'],
                    'distance': result['distance'],
                    'coordinates': result['coordinates'],
                    'search_type': 'radius_simple'
                }
                for result in raw_results
            ]
            
        elif search_strategy == 'nearest':
            raw_results = self.find_nearest_memories(
                query_coords=query_coords,
                k=max_results
            )
            
            return [
                {
                    'data': result['memory'],
                    'distance': result['distance'],
                    'coordinates': result['coordinates'],
                    'search_type': 'nearest_simple'
                }
                for result in raw_results
            ]
        else:
            raise ValueError(f"Unknown search strategy: {search_strategy}")
    
    def _sanitize_metadata(self, metadata):
        """Convert metadata to safe types"""
        if not metadata:
            return {}
        
        sanitized = {}
        for key, value in metadata.items():
            try:
                safe_key = str(key)
                
                if isinstance(value, (str, int, float, bool)):
                    sanitized[safe_key] = value
                elif isinstance(value, dict):
                    sanitized[safe_key] = self._sanitize_metadata(value)
                elif isinstance(value, (list, tuple)):
                    sanitized[safe_key] = [str(item) for item in value]
                else:
                    sanitized[safe_key] = str(value)
            except Exception:
                continue
        
        return sanitized
    
    def _calculate_distance(self, coords1, coords2):
        """Calculate 9D Euclidean distance"""
        axes = ['x', 'y', 'z', 'a', 'b', 'c', 'd', 'e', 'f']
        distance_squared = sum(
            (coords1.get(axis, 0) - coords2.get(axis, 0)) ** 2 
            for axis in axes
        )
        return math.sqrt(distance_squared)
    
    def _find_approximate_match(self, target_coords, tolerance, txn):
        """Find memories with coordinates within tolerance"""
        cursor = txn.cursor()
        
        for coord_key, memory_value in cursor:
            try:
                coords = self._decode_coordinate_key(coord_key)
                distance = self._calculate_distance(target_coords, coords)
                
                if distance <= tolerance:
                    return pickle.loads(memory_value)
            except:
                continue  # Skip corrupted entries
        
        return None
    
    def _load_stats(self):
        """Load stats from database"""
        try:
            with self.env.begin() as txn:
                stats_value = txn.get(b'__stats__')
                if stats_value:
                    stored_stats = pickle.loads(stats_value)
                    self.stats.update(stored_stats)
        except:
            pass  # Use default stats
    
    def _save_stats(self, txn):
        """Save stats to database"""
        self.stats['last_access_time'] = time.time()
        stats_value = pickle.dumps(self.stats)
        txn.put(b'__stats__', stats_value)
    
    def switch_to_safe_mode(self):
        """
        🛡️ SWITCH TO SAFE MODE - Enable data durability settings
        
        Call this after bulk loading is complete to ensure data safety.
        Requires closing and reopening the database.
        """
        if self.turbo_mode:
            print("🛡️ Switching database to SAFE MODE...")
            
            # Save stats and close current environment
            with self.env.begin(write=True) as txn:
                self._save_stats(txn)
            self.env.sync()  # Force final sync
            self.env.close()
            
            # Reopen with safe settings
            self.turbo_mode = False
            self.env = lmdb.open(
                self.db_path, 
                map_size=50 * 1024 * 1024 * 1024, 
                writemap=True,
                sync=True,         # SAFE: Enable sync for data durability
                metasync=True,     # SAFE: Enable metadata sync
                map_async=False,   # SAFE: Synchronous memory mapping
                readahead=True     # SAFE: Enable readahead for sequential access
            )
            
            print("✅ Database switched to SAFE MODE - data durability enabled!")
        else:
            print("ℹ️ Database already in SAFE MODE")
    
    def switch_to_turbo_mode(self):
        """
        🚀 SWITCH TO TURBO MODE - Enable bulk loading optimizations
        
        Use this for bulk loading operations where speed is critical.
        """
        if not self.turbo_mode:
            print("🚀 Switching database to TURBO MODE...")
            
            # Save stats and close current environment
            with self.env.begin(write=True) as txn:
                self._save_stats(txn)
            self.env.close()
            
            # Reopen with turbo settings
            self.turbo_mode = True
            self.env = lmdb.open(
                self.db_path, 
                map_size=50 * 1024 * 1024 * 1024, 
                writemap=True,
                sync=False,        # TURBO: Disable sync for bulk loading
                metasync=False,    # TURBO: Disable metadata sync  
                map_async=True,    # TURBO: Async memory mapping
                readahead=False    # TURBO: Disable readahead for random access
            )
            
            print("✅ Database switched to TURBO MODE - maximum speed enabled!")
        else:
            print("ℹ️ Database already in TURBO MODE")
    
    def force_sync(self):
        """
        💾 FORCE SYNC - Manually flush all pending writes to disk
        
        Use this periodically during bulk loading for safety checkpoints.
        """
        if hasattr(self, 'env') and self.env:
            self.env.sync()
            print("💾 Database synced to disk")
    
    def get_mode_info(self):
        """Get current database mode information"""
        return {
            'mode': 'TURBO' if self.turbo_mode else 'SAFE',
            'sync_enabled': not self.turbo_mode,
            'optimized_for': 'bulk_loading' if self.turbo_mode else 'data_safety'
        }

    def close(self):
        """Close the database"""
        with self.env.begin(write=True) as txn:
            self._save_stats(txn)
        if self.turbo_mode:
            # Force final sync in TURBO mode before closing
            self.env.sync()
        self.env.close() 