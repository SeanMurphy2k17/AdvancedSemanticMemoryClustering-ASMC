#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Long-Term Memory (LTM) API

Professional API wrapper for the revolutionary 9D Spatial Long-Term Memory System.

CREATORS:
- Sean Murphy (Human Inventor & System Architect)
- Claude AI Models (AI Co-Inventor & Implementation Partner)
  - Claude-3.7-Sonnet: Core system design and implementation
  - Claude-4-Sonnet: Advanced optimization and API development
  - Claude-4-Opus: Conceptual breakthroughs and testing

Copyright (c) 2024 Sean Murphy & Claude AI
License: MIT
"""

import os
import sys
import json
import time
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from EngramManager import EngramManager

class LongTermMemory_API:
    """
    Long-Term Memory API
    
    Professional API for massive-scale 9D spatial memory management.
    Provides intelligent long-term storage, semantic search, and memory reasoning.
    """
    
    def __init__(self, 
                 db_path: str = "LTM_CoreData.lmdb",
                 enable_linking: bool = True,
                 turbo_mode: bool = False,
                 max_size_gb: int = 50,
                 verbose: bool = False):
        """
        Initialize the Long-Term Memory API
        
        Args:
            db_path: Path to LMDB database file (default: LTM_CoreData.lmdb)
            enable_linking: Enable semantic linking between memories (default: True)
            turbo_mode: Use TURBO mode for bulk operations vs SAFE mode (default: False)
            max_size_gb: Maximum database size in GB (default: 50GB)
            verbose: Enable detailed logging (default: False)
        """
        self.version = "1.0.0"
        self.db_path = db_path
        self.enable_linking = enable_linking
        self.turbo_mode = turbo_mode
        self.max_size_gb = max_size_gb
        self.verbose = verbose
        
        # Initialize the core LTM manager
        self._ltm = EngramManager(
            db_path=db_path,
            enable_linking=enable_linking,
            turbo_mode=turbo_mode,
            verbose=verbose
        )
        
        if verbose:
            print(f"Long-Term Memory API v{self.version} initialized")
            print(f"Database: {db_path}")
            print(f"Max size: {max_size_gb}GB")
            print(f"Semantic linking: {'ENABLED' if enable_linking else 'DISABLED'}")
            print(f"Mode: {'TURBO' if turbo_mode else 'SAFE'}")
    
    def store_memory(self, 
                    text: str, 
                    metadata: Optional[Dict] = None) -> Dict:
        """
        Store a memory in the long-term spatial database
        
        Args:
            text: Text content to store
            metadata: Optional metadata dictionary
            
        Returns:
            Dict: Storage result with coordinate information
        """
        try:
            memory_id = self._ltm.store_memory(text, metadata)
            
            if memory_id is not None:
                # Get the stored memory details
                stats = self._ltm.get_system_stats()
                
                return {
                    "success": True,
                    "memory_id": memory_id,
                    "text": text,
                    "metadata": metadata,
                    "timestamp": time.time(),
                    "total_memories": stats['database_memories'],
                    "database_size_mb": stats['database_size_mb']
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to store memory",
                    "timestamp": time.time()
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def search_similar(self, 
                      query_text: str,
                      max_results: int = 10,
                      radius: float = 0.5) -> Dict:
        """
        Search for semantically similar memories
        
        Args:
            query_text: Query text to search for
            max_results: Maximum number of results to return
            radius: Search radius in 9D coordinate space
            
        Returns:
            Dict: Search results with similarity scores
        """
        try:
            results = self._ltm.search_similar(query_text, max_results)
            
            formatted_results = []
            for result in results:
                data = result.get('data', {})
                formatted_results.append({
                    "memory_id": data.get('id', 'unknown'),
                    "text": data.get('input_text', ''),
                    "semantic_summary": data.get('semantic_summary', ''),
                    "coordinates": data.get('coordinates', {}),
                    "distance": result.get('distance', 0.0),
                    "similarity_score": 1.0 - result.get('distance', 1.0),
                    "timestamp": data.get('timestamp', 0)
                })
            
            return {
                "success": True,
                "query": query_text,
                "results": formatted_results,
                "total_found": len(formatted_results),
                "search_timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query_text,
                "search_timestamp": time.time()
            }
    
    def get_system_statistics(self) -> Dict:
        """
        Get comprehensive system statistics and performance metrics
        
        Returns:
            Dict: System statistics and health information
        """
        try:
            stats = self._ltm.get_system_stats()
            mode_info = self._ltm.get_mode_info()
            
            return {
                "success": True,
                "version": self.version,
                "database": {
                    "path": self.db_path,
                    "total_memories": stats['database_memories'],
                    "size_mb": stats['database_size_mb'],
                    "max_size_gb": self.max_size_gb
                },
                "performance": {
                    "total_stored": stats['total_stored'],
                    "total_retrieved": stats['total_retrieved'],
                    "cache_hit_rate": stats['coordinate_cache_rate'],
                    "cache_size": stats['coordinate_cache_size']
                },
                "semantic_linking": {
                    "enabled": self.enable_linking,
                    "succession_links": stats.get('succession_links', 0),
                    "radial_links": stats.get('radial_links', 0),
                },
                "mode": {
                    "current_mode": mode_info['current_mode'],
                    "turbo_enabled": mode_info.get('turbo_mode', False),
                    "linking_enabled": mode_info['linking_enabled']
                },
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def bulk_store_memories(self, 
                           text_list: List[str],
                           show_progress: bool = True) -> Dict:
        """
        Store multiple memories efficiently in bulk
        
        Args:
            text_list: List of text strings to store
            show_progress: Show progress during bulk storage
            
        Returns:
            Dict: Bulk storage results and statistics
        """
        try:
            start_time = time.time()
            successful_stores = 0
            failed_stores = 0
            memory_ids = []
            
            for i, text in enumerate(text_list):
                if show_progress and i % 10 == 0:
                    print(f"Storing memory {i+1}/{len(text_list)}...")
                
                memory_id = self._ltm.store_memory(text)
                if memory_id is not None:
                    successful_stores += 1
                    memory_ids.append(memory_id)
                else:
                    failed_stores += 1
            
            duration = time.time() - start_time
            
            return {
                "success": True,
                "total_attempted": len(text_list),
                "successful_stores": successful_stores,
                "failed_stores": failed_stores,
                "memory_ids": memory_ids,
                "duration_seconds": duration,
                "average_time_per_memory": duration / len(text_list),
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def cleanup(self) -> Dict:
        """
        Clean up resources and close database connections
        
        Returns:
            Dict: Cleanup status
        """
        try:
            self._ltm.cleanup()
            return {
                "success": True,
                "message": "LTM API cleaned up successfully",
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }

def create_ltm_api(db_path: str = "LTM_CoreData.lmdb",
                   enable_linking: bool = True,
                   turbo_mode: bool = False,
                   max_size_gb: int = 50,
                   verbose: bool = False) -> LongTermMemory_API:
    """
    Factory function to create LTM API instance
    
    Args:
        db_path: Database file path
        enable_linking: Enable semantic linking
        turbo_mode: Use turbo mode for performance
        max_size_gb: Maximum database size
        verbose: Enable verbose logging
        
    Returns:
        LongTermMemory_API: Initialized API instance
    """
    return LongTermMemory_API(
        db_path=db_path,
        enable_linking=enable_linking,
        turbo_mode=turbo_mode,
        max_size_gb=max_size_gb,
        verbose=verbose
    )

# Example usage
if __name__ == "__main__":
    print("Long-Term Memory API - Example Usage")
    print("=" * 60)
    
    # Create LTM API instance
    ltm_api = create_ltm_api(
        db_path="demo_ltm.lmdb",
        enable_linking=True,
        turbo_mode=False,
        verbose=True
    )
    
    # Store some memories
    print("\nStoring sample memories...")
    
    sample_memories = [
        "The ancient library contained thousands of scrolls about mathematics and astronomy",
        "Scientists discovered a new method for quantum error correction using topological qubits",
        "The forest was filled with the sound of birds singing in the early morning light",
        "Machine learning algorithms can now process natural language with remarkable accuracy",
        "The philosopher contemplated the nature of consciousness and free will"
    ]
    
    stored_count = 0
    for memory_text in sample_memories:
        result = ltm_api.store_memory(memory_text)
        if result['success']:
            stored_count += 1
            print(f"   Stored memory {result['memory_id']}")
        else:
            print(f"   Failed: {result['error']}")
    
    print(f"\nStored {stored_count} memories in long-term database")
    
    # Test semantic search
    print("\nTesting semantic search...")
    search_result = ltm_api.search_similar("artificial intelligence and computing", max_results=3)
    
    if search_result['success']:
        print(f"   Found {search_result['total_found']} relevant memories:")
        for i, result in enumerate(search_result['results'], 1):
            print(f"      [{i}] Similarity: {result['similarity_score']:.3f}")
            print(f"          Text: {result['text'][:60]}...")
    
    # Show system statistics
    print("\nSystem statistics...")
    stats = ltm_api.get_system_statistics()
    
    if stats['success']:
        db_stats = stats['database']
        perf_stats = stats['performance']
        print(f"   Total memories: {db_stats['total_memories']}")
        print(f"   Database size: {db_stats['size_mb']:.2f} MB")
        print(f"   Cache hit rate: {perf_stats['cache_hit_rate']}")
        print(f"   Total stored: {perf_stats['total_stored']}")
        print(f"   Total retrieved: {perf_stats['total_retrieved']}")
    
    # Graceful shutdown
    print("\nShutting down...")
    cleanup_result = ltm_api.cleanup()
    
    if cleanup_result['success']:
        print("   LTM API shutdown successfully")
    
    print("\nExample complete!") 