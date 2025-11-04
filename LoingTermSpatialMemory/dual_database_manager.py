#!/usr/bin/env python3
"""
🧠 DUAL DATABASE MANAGER - KNOWLEDGE vs EXPERIENCE SEPARATION 🧠

REVOLUTIONARY TWO-DATABASE ARCHITECTURE:
- Knowledge Database: Static reference materials (books, docs, facts)
- Experience Database: Dynamic personal experiences (conversations, learned behaviors)

BENEFITS:
- Clean separation of concerns
- Optimized access patterns
- Prevents contamination of knowledge base
- Enables different retention policies
- Supports STM → Experience pipeline

CREATOR: Sean Murphy's brilliant architectural insight!
"""

import os
import time
from typing import Dict, List, Optional, Union, Any
from EngramManager import EngramManager
from LTM_API import LongTermMemory_API

class DualDatabaseManager:
    """
    🧠 DUAL DATABASE MANAGER
    
    Manages two separate LTM databases:
    1. Knowledge Database - Static reference materials
    2. Experience Database - Dynamic personal experiences
    """
    
    def __init__(self,
                 knowledge_db_path: str = "ltm_knowledge.lmdb",
                 experience_db_path: str = "ltm_experience.lmdb",
                 enable_linking: bool = True,
                 verbose: bool = False):
        """
        Initialize dual database system
        
        Args:
            knowledge_db_path: Path to knowledge database (static reference)
            experience_db_path: Path to experience database (dynamic personal)
            enable_linking: Enable semantic linking within each database
            verbose: Enable detailed logging
        """
        self.verbose = verbose
        
        if verbose:
            print("🧠" * 30)
            print("🧠 DUAL DATABASE MANAGER - INITIALIZING 🧠")
            print("🧠" * 30)
        
        # Initialize Knowledge Database (optimized for reading)
        self.knowledge_db = EngramManager(
            db_path=knowledge_db_path,
            enable_linking=enable_linking,
            turbo_mode=False,  # SAFE mode for knowledge preservation
            verbose=verbose
        )
        
        # Initialize Experience Database (optimized for writing)
        self.experience_db = EngramManager(
            db_path=experience_db_path,
            enable_linking=enable_linking,
            turbo_mode=False,  # SAFE mode for personal memories
            verbose=verbose
        )
        
        # Statistics tracking
        self.stats = {
            'knowledge_queries': 0,
            'experience_queries': 0,
            'knowledge_stores': 0,
            'experience_stores': 0,
            'cross_database_searches': 0
        }
        
        if verbose:
            print(f"📚 Knowledge Database: {knowledge_db_path}")
            print(f"🧠 Experience Database: {experience_db_path}")
            print(f"🔗 Semantic Linking: {'ENABLED' if enable_linking else 'DISABLED'}")
            print("✅ Dual Database Manager Ready!")
    
    def store_knowledge(self, text: str, metadata: Optional[Dict] = None) -> Optional[int]:
        """
        Store static knowledge (books, documentation, facts)
        
        Args:
            text: Knowledge content to store
            metadata: Optional metadata (source, category, etc.)
            
        Returns:
            int: Memory ID if successful, None if failed
        """
        if metadata is None:
            metadata = {}
        
        # Add knowledge-specific metadata
        metadata.update({
            'database_type': 'knowledge',
            'storage_timestamp': time.time(),
            'is_static_knowledge': True
        })
        
        memory_id = self.knowledge_db.store_memory(text, metadata)
        
        if memory_id is not None:
            self.stats['knowledge_stores'] += 1
            if self.verbose:
                print(f"📚 Knowledge stored: ID {memory_id}")
        
        return memory_id
    
    def store_experience(self, text: str, metadata: Optional[Dict] = None) -> Optional[int]:
        """
        Store personal experience (conversations, learned behaviors)
        
        Args:
            text: Experience content to store
            metadata: Optional metadata (context, importance, etc.)
            
        Returns:
            int: Memory ID if successful, None if failed
        """
        if metadata is None:
            metadata = {}
        
        # Add experience-specific metadata
        metadata.update({
            'database_type': 'experience',
            'storage_timestamp': time.time(),
            'is_personal_experience': True
        })
        
        memory_id = self.experience_db.store_memory(text, metadata)
        
        if memory_id is not None:
            self.stats['experience_stores'] += 1
            if self.verbose:
                print(f"🧠 Experience stored: ID {memory_id}")
        
        return memory_id
    
    def search_knowledge(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search only the knowledge database
        
        Args:
            query: Search query
            max_results: Maximum results to return
            
        Returns:
            List[Dict]: Search results from knowledge database
        """
        self.stats['knowledge_queries'] += 1
        results = self.knowledge_db.search_similar(query, max_results)
        
        # Add database source to results
        for result in results:
            if 'data' in result:
                result['data']['source_database'] = 'knowledge'
        
        if self.verbose:
            print(f"📚 Knowledge search: '{query}' → {len(results)} results")
        
        return results
    
    def search_experience(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search only the experience database
        
        Args:
            query: Search query
            max_results: Maximum results to return
            
        Returns:
            List[Dict]: Search results from experience database
        """
        self.stats['experience_queries'] += 1
        results = self.experience_db.search_similar(query, max_results)
        
        # Add database source to results
        for result in results:
            if 'data' in result:
                result['data']['source_database'] = 'experience'
        
        if self.verbose:
            print(f"🧠 Experience search: '{query}' → {len(results)} results")
        
        return results
    
    def search_both(self, query: str, 
                   knowledge_results: int = 3, 
                   experience_results: int = 3) -> Dict[str, List[Dict]]:
        """
        Search both databases and return categorized results
        
        Args:
            query: Search query
            knowledge_results: Max results from knowledge database
            experience_results: Max results from experience database
            
        Returns:
            Dict: Categorized results from both databases
        """
        self.stats['cross_database_searches'] += 1
        
        knowledge_hits = self.search_knowledge(query, knowledge_results)
        experience_hits = self.search_experience(query, experience_results)
        
        return {
            'query': query,
            'knowledge_results': knowledge_hits,
            'experience_results': experience_hits,
            'total_knowledge': len(knowledge_hits),
            'total_experience': len(experience_hits),
            'search_timestamp': time.time()
        }
    
    def promote_stm_to_experience(self, stm_memories: List[Dict]) -> Dict:
        """
        Promote memories from STM to Experience database
        
        Args:
            stm_memories: List of STM memory objects
            
        Returns:
            Dict: Promotion results and statistics
        """
        promoted_count = 0
        failed_count = 0
        
        for stm_memory in stm_memories:
            # Extract text content
            text = stm_memory.get('content', stm_memory.get('text', ''))
            
            if not text:
                failed_count += 1
                continue
            
            # Create experience metadata
            metadata = {
                'promoted_from_stm': True,
                'stm_timestamp': stm_memory.get('timestamp', time.time()),
                'stm_context': stm_memory.get('context', {}),
                'promotion_timestamp': time.time(),
                'importance_score': stm_memory.get('importance', 0.5)
            }
            
            # Store in experience database
            memory_id = self.store_experience(text, metadata)
            
            if memory_id is not None:
                promoted_count += 1
            else:
                failed_count += 1
        
        results = {
            'total_processed': len(stm_memories),
            'promoted_successfully': promoted_count,
            'failed_promotions': failed_count,
            'success_rate': promoted_count / len(stm_memories) if stm_memories else 0,
            'promotion_timestamp': time.time()
        }
        
        if self.verbose:
            print(f"🔄 STM→Experience promotion: {promoted_count}/{len(stm_memories)} successful")
        
        return results
    
    def bulk_load_knowledge(self, file_path_or_folder: str) -> Dict:
        """
        Bulk load knowledge from files using mass data uploader
        
        Args:
            file_path_or_folder: Path to file or folder to load
            
        Returns:
            Dict: Bulk loading results
        """
        # Import mass data uploader functionality
        from mass_data_uploader import process_mass_data
        
        # Use knowledge database path
        knowledge_path = self.knowledge_db.db_manager.db_path
        
        if self.verbose:
            print(f"📚 Bulk loading knowledge into: {knowledge_path}")
        
        # Process with mass uploader
        results = process_mass_data(
            folder_path=file_path_or_folder,
            db_path=knowledge_path,
            enable_linking=True,
            chunk_size=300
        )
        
        # Update our statistics
        if 'memories_stored' in results:
            self.stats['knowledge_stores'] += results['memories_stored']
        
        return results
    
    def get_system_statistics(self) -> Dict:
        """
        Get comprehensive statistics for both databases
        
        Returns:
            Dict: Complete system statistics
        """
        knowledge_stats = self.knowledge_db.get_system_stats()
        experience_stats = self.experience_db.get_system_stats()
        
        return {
            'dual_database_stats': self.stats,
            'knowledge_database': {
                'path': self.knowledge_db.db_manager.db_path,
                'memories': knowledge_stats['database_memories'],
                'size_mb': knowledge_stats['database_size_mb'],
                'purpose': 'Static reference materials'
            },
            'experience_database': {
                'path': self.experience_db.db_manager.db_path,
                'memories': experience_stats['database_memories'],
                'size_mb': experience_stats['database_size_mb'],
                'purpose': 'Dynamic personal experiences'
            },
            'total_memories': (knowledge_stats['database_memories'] + 
                             experience_stats['database_memories']),
            'architecture': 'dual_database_separation'
        }
    
    def intelligent_search(self, query: str, context: str = "general") -> Dict:
        """
        Intelligent search that decides which database(s) to query based on context
        
        Args:
            query: Search query
            context: Context hint ("knowledge", "experience", "general")
            
        Returns:
            Dict: Intelligent search results
        """
        query_lower = query.lower()
        
        # Determine search strategy based on query content and context
        if context == "knowledge" or any(word in query_lower for word in 
                                       ['what is', 'define', 'explain', 'how to', 'documentation']):
            # Knowledge-focused search
            return {
                'strategy': 'knowledge_focused',
                'results': self.search_knowledge(query, max_results=5),
                'reasoning': 'Query appears to be seeking factual/reference information'
            }
        
        elif context == "experience" or any(word in query_lower for word in 
                                          ['remember when', 'last time', 'we discussed', 'you said']):
            # Experience-focused search
            return {
                'strategy': 'experience_focused',
                'results': self.search_experience(query, max_results=5),
                'reasoning': 'Query appears to be seeking personal/conversational context'
            }
        
        else:
            # Search both with balanced results
            return {
                'strategy': 'balanced_search',
                'results': self.search_both(query, knowledge_results=3, experience_results=3),
                'reasoning': 'General query - searching both databases'
            }
    
    def cleanup(self):
        """Clean up both database connections"""
        self.knowledge_db.cleanup()
        self.experience_db.cleanup()
        
        if self.verbose:
            print("🧹 Dual Database Manager cleanup complete")

# Convenience function for easy initialization
def create_dual_ltm(knowledge_db: str = "ltm_knowledge.lmdb",
                   experience_db: str = "ltm_experience.lmdb",
                   enable_linking: bool = True,
                   verbose: bool = False) -> DualDatabaseManager:
    """
    Create a dual database LTM system
    
    Args:
        knowledge_db: Path to knowledge database
        experience_db: Path to experience database
        enable_linking: Enable semantic linking
        verbose: Enable detailed logging
        
    Returns:
        DualDatabaseManager: Configured dual database system
    """
    return DualDatabaseManager(
        knowledge_db_path=knowledge_db,
        experience_db_path=experience_db,
        enable_linking=enable_linking,
        verbose=verbose
    )

# Example usage
if __name__ == "__main__":
    print("🧠 Testing Dual Database Manager")
    
    # Initialize dual system
    dual_ltm = create_dual_ltm(verbose=True)
    
    # Store some knowledge
    dual_ltm.store_knowledge(
        "Python is a high-level programming language known for its simplicity",
        metadata={"category": "programming", "source": "reference"}
    )
    
    # Store some experience
    dual_ltm.store_experience(
        "User asked about Python programming and seemed interested in learning",
        metadata={"context": "conversation", "user_interest": "programming"}
    )
    
    # Test intelligent search
    results = dual_ltm.intelligent_search("tell me about Python")
    print(f"Intelligent search results: {results['strategy']}")
    
    # Get statistics
    stats = dual_ltm.get_system_statistics()
    print(f"Knowledge memories: {stats['knowledge_database']['memories']}")
    print(f"Experience memories: {stats['experience_database']['memories']}")
    
    dual_ltm.cleanup()
    print("🎯 Dual Database test complete!")

# 🧠 DUAL DATABASE ARCHITECTURE - SEAN'S BRILLIANT INSIGHT! 🧠
#
# BENEFITS:
# 📚 Knowledge Database:
#   - Static reference materials
#   - Mass-loaded via bulk uploader
#   - Optimized for fast lookup
#   - Immutable during conversations
#
# 🧠 Experience Database:
#   - Dynamic personal experiences
#   - STM promotion pipeline
#   - Conversational context
#   - Grows with interactions
#
# 🎯 INTELLIGENT ROUTING:
#   - Context-aware search decisions
#   - Prevents knowledge contamination
#   - Optimized access patterns
#   - Clean separation of concerns 