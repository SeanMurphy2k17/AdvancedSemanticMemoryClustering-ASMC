#!/usr/bin/env python3
"""
🧠 DUAL DATABASE QUICK TEST
Test Sean's brilliant dual database architecture
"""

from dual_database_manager import create_dual_ltm

def test_dual_database():
    print("🧠 Testing Dual Database Architecture")
    print("=" * 40)
    
    # Initialize dual system
    dual = create_dual_ltm(
        knowledge_db="test_knowledge.lmdb",
        experience_db="test_experience.lmdb",
        verbose=False
    )
    
    # Store knowledge
    k_id = dual.store_knowledge(
        "Python is a high-level programming language", 
        {"source": "documentation", "category": "programming"}
    )
    print(f"📚 Knowledge stored: ID {k_id}")
    
    # Store experience  
    e_id = dual.store_experience(
        "Sean loves the dual database architecture idea!",
        {"user": "Sean", "context": "conversation", "importance": 0.9}
    )
    print(f"🧠 Experience stored: ID {e_id}")
    
    # Test intelligent search
    print("\n🔍 Testing Intelligent Search:")
    
    # Knowledge query
    results = dual.intelligent_search("What is Python?")
    print(f"   Query: 'What is Python?' → Strategy: {results['strategy']}")
    
    # Experience query
    results = dual.intelligent_search("What did Sean say?")
    print(f"   Query: 'What did Sean say?' → Strategy: {results['strategy']}")
    
    # Get statistics
    stats = dual.get_system_statistics()
    print(f"\n📊 System Statistics:")
    print(f"   Knowledge memories: {stats['knowledge_database']['memories']}")
    print(f"   Experience memories: {stats['experience_database']['memories']}")
    print(f"   Total memories: {stats['total_memories']}")
    
    # Cleanup
    dual.cleanup()
    print("\n✅ Dual database test successful!")
    print("🎯 Sean's architectural insight: BRILLIANT! 🧠✨")

if __name__ == "__main__":
    test_dual_database() 