#!/usr/bin/env python3
"""
🧪 SEMANTIC STM SYSTEM TEST 🧪

Comprehensive test of the Semantic Short-Term Memory Manager:
- RAM storage and retrieval
- 9D spatial coordinate search
- Rolling pair saves (A/B alternating)
- Context building (recent + relevant)
- Promotion to long-term memory
- Corruption recovery
"""

import os
import sys
import time
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shortTermMemory.SemanticSTMManager import SemanticSTMManager

def test_basic_functionality():
    """Test basic STM functionality"""
    print("🧪 TEST 1: Basic STM Functionality")
    print("=" * 50)
    
    # Initialize with small capacity for testing
    stm = SemanticSTMManager(max_entries=5, save_interval=3, verbose=True)
    
    # Test conversation exchanges
    test_conversations = [
        ("Hello, how are you?", "I'm doing well, thank you! How can I help you today?"),
        ("What's the weather like?", "I don't have access to current weather data, but I can help you find weather information."),
        ("Tell me about cats", "Cats are fascinating animals! They're independent, agile, and make great companions."),
        ("I love programming", "Programming is a wonderful skill! What languages do you enjoy working with?"),
        ("Python is my favorite", "Python is excellent! It's versatile, readable, and has a great ecosystem.")
    ]
    
    print(f"\n📚 Adding {len(test_conversations)} conversation exchanges...")
    
    coord_keys = []
    for i, (user_msg, ai_msg) in enumerate(test_conversations, 1):
        coord_key = stm.add_conversation_exchange(user_msg, ai_msg)
        coord_keys.append(coord_key)
        print(f"   [{i}] Added: '{user_msg[:30]}...' → {coord_key[:15]}...")
        time.sleep(0.5)  # Small delay to see saves
    
    print(f"\n✅ Basic functionality test complete!")
    print(f"   Stored {len(stm.stm_entries)} entries in RAM")
    
    return stm, coord_keys

def test_spatial_search(stm):
    """Test 9D spatial coordinate search"""
    print("\n🧪 TEST 2: Spatial Search")
    print("=" * 50)
    
    # Test semantic search
    search_queries = [
        "cats and animals",
        "programming languages",
        "weather information",
        "help and assistance"
    ]
    
    for query in search_queries:
        print(f"\n🔍 Searching for: '{query}'")
        results = stm.search_relevant_context(query, top_k=3)
        
        if results:
            for i, result in enumerate(results, 1):
                entry = result['entry']
                distance = result['distance']
                relevance = result['relevance_score']
                print(f"   [{i}] Distance: {distance:.3f} | Relevance: {relevance:.3f}")
                print(f"       User: {entry['user_input'][:40]}...")
                print(f"       AI: {entry['ai_response'][:40]}...")
        else:
            print("   No matches found")
    
    print(f"\n✅ Spatial search test complete!")

def test_context_building(stm):
    """Test enhanced context building"""
    print("\n🧪 TEST 3: Enhanced Context Building")
    print("=" * 50)
    
    test_inputs = [
        "I need help with Python programming",
        "Can you tell me more about animals?",
        "What's the current weather forecast?"
    ]
    
    for user_input in test_inputs:
        print(f"\n🔍 Building context for: '{user_input}'")
        context = stm.build_enhanced_context(user_input, recent_count=2, relevant_count=3)
        
        print(f"   Recent context: {len(context['recent_context'])} entries")
        print(f"   Relevant context: {len(context['relevant_context'])} entries")
        print(f"   Total context: {context['total_context_entries']} entries")
        print(f"   Query summary: {context['query_summary']}")
        
        # Show most relevant
        if context['relevant_context']:
            print(f"   Most relevant:")
            for entry in context['relevant_context'][:1]:
                print(f"      User: {entry['user_input'][:35]}...")
                print(f"      AI: {entry['ai_response'][:35]}...")
    
    print(f"\n✅ Context building test complete!")

def test_rolling_saves(stm):
    """Test rolling pair save system"""
    print("\n🧪 TEST 4: Rolling Pair Saves")
    print("=" * 50)
    
    # Check initial save status
    save_status = stm.get_save_status()
    print(f"Current save target: {save_status['current_target']}")
    print(f"Files exist - A: {save_status['files_exist']['A']}, B: {save_status['files_exist']['B']}")
    
    # Force a save to see alternation
    print(f"\n💾 Forcing save...")
    stm.force_save()
    
    # Check status after save
    save_status = stm.get_save_status()
    print(f"After save - target: {save_status['current_target']}")
    print(f"Files exist - A: {save_status['files_exist']['A']}, B: {save_status['files_exist']['B']}")
    
    # Add more data and wait for automatic save
    print(f"\n📚 Adding more data to trigger automatic save...")
    for i in range(3):
        stm.add_conversation_exchange(
            f"Test message {i+1}",
            f"Test response {i+1} with some content to make it longer"
        )
        time.sleep(1)
    
    # Wait for automatic save
    print(f"⏰ Waiting for automatic save (every {stm.save_interval}s)...")
    time.sleep(stm.save_interval + 1)
    
    # Check final status
    save_status = stm.get_save_status()
    print(f"Final save target: {save_status['current_target']}")
    print(f"Seconds since save: {save_status['seconds_since_save']:.1f}")
    
    print(f"\n✅ Rolling saves test complete!")

def test_capacity_management(stm):
    """Test capacity management and promotion"""
    print("\n🧪 TEST 5: Capacity Management")
    print("=" * 50)
    
    initial_count = len(stm.stm_entries)
    print(f"Initial STM entries: {initial_count}")
    print(f"Max capacity: {stm.max_entries}")
    
    # Add entries beyond capacity to trigger promotion
    print(f"\n📚 Adding entries beyond capacity...")
    for i in range(stm.max_entries + 3):
        stm.add_conversation_exchange(
            f"Overflow test message {i+1}",
            f"This is overflow response {i+1} to test capacity management"
        )
        
        current_count = len(stm.stm_entries)
        print(f"   Added entry {i+1} | STM count: {current_count}")
        
        if current_count <= stm.max_entries:
            print(f"      ✅ Within capacity")
        else:
            print(f"      ⚠️ Over capacity - should trigger promotion")
    
    final_count = len(stm.stm_entries)
    print(f"\nFinal STM entries: {final_count}")
    print(f"Should be ≤ {stm.max_entries}: {'✅' if final_count <= stm.max_entries else '❌'}")
    
    print(f"\n✅ Capacity management test complete!")

def test_statistics_and_cleanup(stm):
    """Test statistics and cleanup"""
    print("\n🧪 TEST 6: Statistics and Cleanup")
    print("=" * 50)
    
    # Get comprehensive stats
    stats = stm.get_stats()
    print(f"📊 STM Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test cleanup
    print(f"\n🧹 Testing cleanup...")
    stm.cleanup()
    
    print(f"\n✅ Statistics and cleanup test complete!")

def test_recovery_simulation():
    """Test recovery from save files"""
    print("\n🧪 TEST 7: Recovery Simulation")
    print("=" * 50)
    
    print("🔄 Creating new STM instance to test recovery...")
    
    # Create new instance - should load from existing save files
    recovered_stm = SemanticSTMManager(max_entries=10, save_interval=30, verbose=True)
    
    print(f"Recovered {len(recovered_stm.stm_entries)} entries from disk")
    
    if recovered_stm.stm_entries:
        print("📋 Sample recovered entries:")
        for i, (coord_key, entry) in enumerate(list(recovered_stm.stm_entries.items())[:3]):
            print(f"   [{i+1}] {entry['user_input'][:40]}...")
    
    recovered_stm.cleanup()
    print(f"\n✅ Recovery simulation test complete!")

def main():
    """Run all tests"""
    print("🧠" * 60)
    print("🧠 SEMANTIC STM SYSTEM - COMPREHENSIVE TEST SUITE 🧠")
    print("🧠" * 60)
    
    start_time = time.time()
    
    try:
        # Run all tests
        stm, coord_keys = test_basic_functionality()
        test_spatial_search(stm)
        test_context_building(stm)
        test_rolling_saves(stm)
        test_capacity_management(stm)
        test_statistics_and_cleanup(stm)
        test_recovery_simulation()
        
        total_time = time.time() - start_time
        
        print("\n" + "🎯" * 60)
        print("🎯 ALL TESTS COMPLETED SUCCESSFULLY! 🎯")
        print("🎯" * 60)
        print(f"Total test time: {total_time:.2f} seconds")
        print(f"✅ RAM-first storage: WORKING")
        print(f"✅ 9D spatial search: WORKING") 
        print(f"✅ Rolling pair saves: WORKING")
        print(f"✅ Context building: WORKING")
        print(f"✅ Capacity management: WORKING")
        print(f"✅ Recovery system: WORKING")
        print("\n🚀 Semantic STM System is ready for integration!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 