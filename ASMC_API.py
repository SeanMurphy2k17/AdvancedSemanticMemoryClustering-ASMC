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

class AdvancedSemanticMemory:
    """
    🧠 ADVANCED SEMANTIC MEMORY CLUSTERING
    
    Unified interface for two-layer semantic memory system
    """
    
    def __init__(self, max_stm_entries: int = 50, ltm_db_path: str = "ASMC_memory.lmdb", verbose: bool = False):
        """
        Initialize the Advanced Semantic Memory Clustering system
        
        Args:
            max_stm_entries: Maximum short-term memory entries (default: 50)
            ltm_db_path: Path to long-term memory database (default: ASMC_memory.lmdb)
            verbose: Enable detailed logging (default: False)
        """
        self.max_stm_entries = max_stm_entries
        self.ltm_db_path = ltm_db_path
        self.verbose = verbose
        
        # Initialize STM (which handles SVC and LTM internally)
        self._stm_api = create_stm_api(
            max_entries=max_stm_entries,
            save_interval=30,
            data_directory="./asmc_stm_data",
            verbose=verbose
        )
        
        if verbose:
            print("🧠 Advanced Semantic Memory Clustering initialized!")
            print(f"   STM Capacity: {max_stm_entries} entries")
            print(f"   LTM Database: {ltm_db_path}")
            print(f"   Features: Two-layer retrieval, 9D clustering, 117k word sentiment")
    
    def add_experience(self, situation: str, response: str, metadata: dict = None):
        """
        Add an experience to memory (stores in STM, auto-promotes to LTM)
        
        Args:
            situation: The situation/context (e.g., sensor data, user input)
            response: The response/thought (e.g., LLM output, action taken)
            metadata: Optional metadata dictionary
            
        Returns:
            Dict: Storage result with coordinate information
        """
        return self._stm_api.add_conversation(
            user_message=situation,
            ai_response=response,
            metadata=metadata
        )
    
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
        return self._stm_api.get_statistics()
    
    def clear_memory(self, confirm: bool = False):
        """Clear all memories (DESTRUCTIVE - requires confirm=True)"""
        return self._stm_api.clear_memory(confirm=confirm)
    
    def shutdown(self):
        """Gracefully shutdown the memory system"""
        return self._stm_api.shutdown()


# Convenience factory function
def create_memory(max_entries: int = 50, db_path: str = "ASMC_memory.lmdb", verbose: bool = False):
    """
    Quick factory function to create Advanced Semantic Memory system
    
    Args:
        max_entries: Maximum STM entries
        db_path: LTM database path  
        verbose: Enable logging
        
    Returns:
        AdvancedSemanticMemory: Initialized memory system
    """
    return AdvancedSemanticMemory(
        max_stm_entries=max_entries,
        ltm_db_path=db_path,
        verbose=verbose
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

