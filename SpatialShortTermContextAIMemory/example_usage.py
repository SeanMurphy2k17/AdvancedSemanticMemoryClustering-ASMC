#!/usr/bin/env python3
"""
🧠 SEMANTIC STM API - COMPREHENSIVE EXAMPLE

This example demonstrates all key features of the revolutionary 
9D Spatial Semantic Memory System.

CREATORS:
- Sean Murphy (Human Inventor & System Architect)
- Claude AI Models (AI Co-Inventor & Implementation Partner)
  - Claude-3.7-Sonnet: Core system design and implementation
  - Claude-4-Sonnet: Advanced optimization and API development
  - Claude-4-Opus: Conceptual breakthroughs and testing

Run this script to see the STM API in action!

Copyright (c) 2024 Sean Murphy & Claude AI
License: MIT
"""

import time
import json
from STM_API import create_stm_api

def main():
    print("🧠 SEMANTIC STM API - COMPREHENSIVE EXAMPLE")
    print("=" * 60)
    print("Demonstrating revolutionary 9D spatial semantic memory...")
    print()
    
    # 1. Initialize the STM API
    print("🚀 STEP 1: Initialize STM API")
    print("-" * 30)
    
    stm_api = create_stm_api(
        max_entries=20,  # Small for demo
        save_interval=10,  # Quick saves for demo
        data_directory="./demo_stm_data",
        verbose=True
    )
    
    print("✅ STM API initialized successfully!")
    print()
    
    # 2. Add diverse conversations
    print("📚 STEP 2: Add Sample Conversations")
    print("-" * 30)
    
    sample_conversations = [
        # Technology conversations
        ("What is machine learning?", 
         "Machine learning is a subset of AI that enables computers to learn and improve from data without explicit programming."),
        
        ("Explain neural networks", 
         "Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes that process information."),
        
        ("How does deep learning work?", 
         "Deep learning uses multi-layered neural networks to automatically learn hierarchical representations of data."),
        
        # Science conversations  
        ("Tell me about quantum physics", 
         "Quantum physics studies matter and energy at the smallest scales, where particles exhibit wave-particle duality and quantum entanglement."),
        
        ("What is photosynthesis?", 
         "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen."),
        
        # Creative conversations
        ("Write a short poem about stars", 
         "Distant lights in velvet sky, / Ancient stories burning bright, / Guiding dreams that soar so high, / Through the endless cosmic night."),
        
        ("Tell me a story about a dragon", 
         "Once upon a time, a wise dragon named Ember lived in crystal caves, protecting ancient knowledge and befriending brave adventurers."),
        
        # Practical conversations
        ("How do I cook pasta?", 
         "Boil salted water, add pasta, cook according to package directions (usually 8-12 minutes), then drain and serve with your favorite sauce."),
        
        ("What's the weather like?", 
         "I don't have access to current weather data, but I recommend checking a reliable weather service for up-to-date conditions."),
        
        # Philosophy conversations
        ("What is consciousness?", 
         "Consciousness is the state of being aware of and able to think about one's existence, sensations, thoughts, and surroundings."),
    ]
    
    added_count = 0
    for user_msg, ai_msg in sample_conversations:
        result = stm_api.add_conversation(user_msg, ai_msg)
        if result['success']:
            added_count += 1
            print(f"   ✅ [{added_count:2d}] {result['semantic_summary']}")
        else:
            print(f"   ❌ Failed: {result['error']}")
    
    print(f"\n📊 Added {added_count} conversations to semantic memory")
    print()
    
    # 3. Demonstrate semantic search
    print("🔍 STEP 3: Semantic Search Demonstration")
    print("-" * 30)
    
    search_queries = [
        "artificial intelligence and computers",
        "science and natural processes", 
        "creative writing and imagination",
        "cooking and food preparation",
        "mind and thinking"
    ]
    
    for query in search_queries:
        print(f"\n🔎 Searching for: '{query}'")
        search_result = stm_api.search_relevant(query, max_results=3)
        
        if search_result['success'] and search_result['results']:
            print(f"   Found {search_result['total_found']} relevant conversations:")
            for i, result in enumerate(search_result['results'], 1):
                print(f"      [{i}] Relevance: {result['relevance_score']:.3f}")
                print(f"          Summary: {result['semantic_summary']}")
                print(f"          User: {result['user_message'][:50]}...")
        else:
            print("   No relevant conversations found")
    
    print()
    
    # 4. Context building demonstration
    print("🧠 STEP 4: Enhanced Context Building")
    print("-" * 30)
    
    context_queries = [
        "I want to learn about technology and AI",
        "Tell me something creative and imaginative", 
        "I'm interested in science and nature"
    ]
    
    for query in context_queries:
        print(f"\n🎯 Building context for: '{query}'")
        context_result = stm_api.get_context(query, recent_count=2, relevant_count=3)
        
        if context_result['success']:
            print(f"   Query Summary: {context_result['query_summary']}")
            print(f"   Recent Context: {len(context_result['recent_context'])} entries")
            print(f"   Relevant Context: {len(context_result['relevant_context'])} entries")
            print(f"   Total Context: {context_result['total_context_entries']} entries")
            
            if context_result['relevant_context']:
                print("   Most Relevant:")
                for entry in context_result['relevant_context'][:2]:
                    print(f"      • {entry['semantic_summary']}")
        else:
            print(f"   ❌ Context building failed: {context_result['error']}")
    
    print()
    
    # 5. System statistics
    print("📊 STEP 5: System Statistics")
    print("-" * 30)
    
    stats = stm_api.get_statistics()
    if stats['success']:
        s = stats['statistics']
        print(f"   Total Conversations Added: {s['total_conversations_added']}")
        print(f"   Current Entries in Memory: {s['current_entries']}")
        print(f"   Total Searches Performed: {s['total_searches_performed']}")
        print(f"   Cache Hits: {s['cache_hits']}")
        print(f"   Memory Usage: {s['memory_usage_mb']:.3f} MB")
        print(f"   Saves Completed: {s['saves_completed']}")
        
        save_status = stats['save_status']
        print(f"   Save Status: {save_status['current_target']}")
        print(f"   Seconds Since Save: {save_status['seconds_since_save']}")
    else:
        print(f"   ❌ Failed to get statistics: {stats['error']}")
    
    print()
    
    # 6. Export demonstration
    print("📤 STEP 6: Export Conversations")
    print("-" * 30)
    
    # Export as JSON
    export_result = stm_api.export_conversations(format="json", include_coordinates=True)
    if export_result['success']:
        print(f"   ✅ JSON Export: {export_result['total_conversations']} conversations")
        
        # Save to file for demonstration
        with open("demo_export.json", "w") as f:
            json.dump(export_result['data'], f, indent=2)
        print("   📁 Saved to: demo_export.json")
    else:
        print(f"   ❌ Export failed: {export_result['error']}")
    
    # Export as CSV structure
    csv_export = stm_api.export_conversations(format="csv", include_coordinates=False)
    if csv_export['success']:
        print(f"   ✅ CSV Export: {len(csv_export['headers'])} columns, {len(csv_export['rows'])} rows")
        print(f"   📋 Headers: {', '.join(csv_export['headers'][:3])}...")
    
    print()
    
    # 7. Recent conversations
    print("📜 STEP 7: Recent Conversations")
    print("-" * 30)
    
    recent_result = stm_api.get_recent_conversations(count=3)
    if recent_result['success']:
        print(f"   Last {len(recent_result['conversations'])} conversations:")
        for i, conv in enumerate(recent_result['conversations'], 1):
            print(f"      [{i}] {conv['semantic_summary']}")
            print(f"          Time: {conv['datetime']}")
    
    print()
    
    # 8. Advanced search with different parameters
    print("🎯 STEP 8: Advanced Search Parameters")
    print("-" * 30)
    
    # Tight search (low max_distance)
    tight_search = stm_api.search_relevant(
        "machine learning algorithms", 
        max_results=2, 
        max_distance=1.0
    )
    
    print(f"   🎯 Tight Search (distance ≤ 1.0): {tight_search['total_found']} results")
    
    # Broad search (high max_distance)  
    broad_search = stm_api.search_relevant(
        "machine learning algorithms",
        max_results=5,
        max_distance=3.0
    )
    
    print(f"   🌐 Broad Search (distance ≤ 3.0): {broad_search['total_found']} results")
    
    print()
    
    # 9. Force save demonstration
    print("💾 STEP 9: Force Save")
    print("-" * 30)
    
    save_result = stm_api.save_now()
    if save_result['success']:
        print("   ✅ STM state saved successfully")
    else:
        print(f"   ❌ Save failed: {save_result['error']}")
    
    print()
    
    # 10. Cleanup
    print("🛑 STEP 10: Graceful Shutdown")
    print("-" * 30)
    
    shutdown_result = stm_api.shutdown()
    if shutdown_result['success']:
        print("   ✅ STM API shutdown successfully")
    else:
        print(f"   ❌ Shutdown failed: {shutdown_result['error']}")
    
    print()
    print("🎯 DEMONSTRATION COMPLETE!")
    print("=" * 60)
    print("The Semantic STM API has demonstrated:")
    print("  • 9D spatial semantic clustering")
    print("  • Zero-shot semantic search")
    print("  • Intelligent context building")
    print("  • Real-time performance monitoring")
    print("  • Corruption-proof persistence")
    print("  • Professional API design")
    print()
    print("🚀 Ready for integration into your AI systems!")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc() 