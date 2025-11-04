#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠🗺️ COMPREHENSIVE LTM SYSTEM TEST

Tests the Long-Term Memory system with conversational and generic inputs,
then verifies retrieval functionality with similar queries.

This test validates:
- Memory storage across different content types
- Semantic search accuracy and relevance
- Database performance and reliability
- 9D spatial coordinate generation
- System statistics and health monitoring
"""

import sys
import os
import time
import json
from typing import List, Dict, Any

# Add LTM directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from EngramManager import EngramManager

class LTMComprehensiveTest:
    """Comprehensive test suite for the LTM system"""
    
    def __init__(self):
        self.test_db_path = "comprehensive_test.lmdb"
        self.ltm = None
        self.stored_memories = []
        self.test_results = {
            'storage_tests': [],
            'retrieval_tests': [],
            'performance_metrics': {},
            'errors': []
        }
    
    def setup(self):
        """Initialize the LTM system for testing"""
        print("🧠🗺️ COMPREHENSIVE LTM SYSTEM TEST")
        print("=" * 50)
        print("Setting up test environment...")
        
        try:
            self.ltm = EngramManager(
                db_path=self.test_db_path,
                enable_linking=True,  # Enable semantic linking
                turbo_mode=False,     # Use SAFE mode for testing
                verbose=True
            )
            print("✅ LTM system initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize LTM: {e}")
            self.test_results['errors'].append(f"Setup failed: {e}")
            return False
    
    def generate_test_data(self) -> Dict[str, List[str]]:
        """Generate diverse test data for comprehensive testing"""
        return {
            'conversational': [
                "Hey, how are you doing today? I've been thinking about our last conversation.",
                "I really enjoyed our discussion about artificial intelligence and consciousness.",
                "Can you help me understand quantum physics? I'm particularly interested in entanglement.",
                "What's your opinion on the future of renewable energy technologies?",
                "I'm feeling a bit overwhelmed with work lately. Any advice on managing stress?",
                "Do you remember when we talked about machine learning algorithms last week?",
                "I've been reading about space exploration and Mars colonization. Fascinating stuff!",
                "Could you explain the difference between supervised and unsupervised learning?",
                "I'm planning a trip to Japan next year. Any recommendations for places to visit?",
                "What do you think about the ethical implications of genetic engineering?"
            ],
            'technical': [
                "Neural networks utilize backpropagation for gradient descent optimization.",
                "LMDB provides ACID transactions with memory-mapped file storage efficiency.",
                "Transformer architectures employ self-attention mechanisms for sequence modeling.",
                "Quantum entanglement demonstrates non-local correlations between particle states.",
                "Distributed systems require consensus algorithms for fault-tolerant coordination.",
                "Cryptographic hash functions ensure data integrity through one-way transformations.",
                "Reinforcement learning agents maximize cumulative reward through policy optimization.",
                "Database indexing structures like B-trees enable logarithmic search complexity.",
                "Convolutional neural networks extract hierarchical features from spatial data.",
                "Microservice architectures promote scalability through service decomposition."
            ],
            'factual': [
                "The Great Wall of China stretches over 13,000 miles across northern China.",
                "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
                "The human brain contains approximately 86 billion neurons.",
                "Mount Everest is 8,848.86 meters above sea level.",
                "The speed of light in vacuum is 299,792,458 meters per second.",
                "DNA consists of four nucleotide bases: adenine, thymine, guanine, and cytosine.",
                "The Amazon rainforest produces about 20% of the world's oxygen.",
                "Saturn has 146 known moons, with Titan being the largest.",
                "The periodic table contains 118 confirmed chemical elements.",
                "Photosynthesis converts carbon dioxide and water into glucose using sunlight."
            ],
            'creative': [
                "In a world where colors had flavors, purple tasted like midnight dreams.",
                "The old lighthouse keeper collected stories from passing ships like seashells.",
                "Time moved differently in the library, where books whispered ancient secrets.",
                "She painted with emotions, each brushstroke carrying a memory of summer rain.",
                "The clockmaker's workshop existed between seconds, in the pause of time.",
                "Music flowed from her fingertips like liquid starlight across piano keys.",
                "The garden grew thoughts instead of flowers, blooming with ideas at dawn.",
                "He wrote letters to the future, sealing them in bottles of crystallized hope.",
                "The mirror showed not reflections, but the dreams of those who gazed within.",
                "Words danced in the air above the poet's desk, forming constellations of meaning."
            ],
            'emotional': [
                "I felt overwhelming joy when I saw my daughter take her first steps.",
                "The loss of my grandmother left an emptiness that words cannot describe.",
                "Standing at the mountain peak, I experienced pure exhilaration and freedom.",
                "Her unexpected kindness during my difficult time restored my faith in humanity.",
                "The anxiety before the presentation was almost paralyzing, but I pushed through.",
                "Watching the sunset with my partner filled me with deep contentment and love.",
                "The frustration of repeated failures made me question my abilities and worth.",
                "Finding my lost childhood diary brought back waves of nostalgic memories.",
                "The fear of uncertainty about the future keeps me awake at night sometimes.",
                "Achieving my lifelong dream felt surreal, like floating on clouds of accomplishment."
            ]
        }
    
    def run_storage_tests(self):
        """Test memory storage across different content types"""
        print("\n📝 STORAGE TESTS")
        print("-" * 30)
        
        test_data = self.generate_test_data()
        storage_start_time = time.time()
        
        for category, texts in test_data.items():
            print(f"\nTesting {category} content...")
            category_results = {
                'category': category,
                'total_items': len(texts),
                'successful_stores': 0,
                'failed_stores': 0,
                'memory_ids': []
            }
            
            for i, text in enumerate(texts):
                try:
                    metadata = {
                        'category': category,
                        'test_index': i,
                        'content_length': len(text),
                        'timestamp': time.time()
                    }
                    
                    memory_id = self.ltm.store_memory(text, metadata)
                    
                    if memory_id is not None:
                        category_results['successful_stores'] += 1
                        category_results['memory_ids'].append(memory_id)
                        self.stored_memories.append({
                            'id': memory_id,
                            'text': text,
                            'category': category,
                            'metadata': metadata
                        })
                        print(f"  ✅ Stored memory {memory_id}: {text[:50]}...")
                    else:
                        category_results['failed_stores'] += 1
                        print(f"  ❌ Failed to store: {text[:50]}...")
                        
                except Exception as e:
                    category_results['failed_stores'] += 1
                    print(f"  ❌ Error storing memory: {e}")
                    self.test_results['errors'].append(f"Storage error in {category}: {e}")
            
            self.test_results['storage_tests'].append(category_results)
            success_rate = (category_results['successful_stores'] / category_results['total_items']) * 100
            print(f"  📊 {category}: {category_results['successful_stores']}/{category_results['total_items']} stored ({success_rate:.1f}%)")
        
        storage_duration = time.time() - storage_start_time
        print(f"\n⏱️ Total storage time: {storage_duration:.2f} seconds")
        print(f"📈 Average storage time: {storage_duration/len(self.stored_memories):.4f} seconds per memory")
    
    def run_retrieval_tests(self):
        """Test memory retrieval with various query types"""
        print("\n🔍 RETRIEVAL TESTS")
        print("-" * 30)
        
        # Define test queries that should match stored content
        test_queries = {
            'conversational_queries': [
                "conversation about artificial intelligence",
                "help with quantum physics",
                "discussion about stress management",
                "trip planning to Japan",
                "machine learning algorithms"
            ],
            'technical_queries': [
                "neural networks and backpropagation",
                "database storage systems",
                "transformer attention mechanisms",
                "quantum particle entanglement",
                "reinforcement learning optimization"
            ],
            'factual_queries': [
                "Great Wall of China length",
                "water boiling temperature",
                "human brain neurons",
                "Mount Everest height",
                "speed of light physics"
            ],
            'creative_queries': [
                "colors with flavors",
                "lighthouse keeper stories",
                "library with time",
                "painting with emotions",
                "clockmaker workshop"
            ],
            'emotional_queries': [
                "joy from daughter's first steps",
                "grandmother's loss and grief",
                "mountain peak exhilaration",
                "unexpected kindness during difficulty",
                "presentation anxiety and fear"
            ]
        }
        
        retrieval_start_time = time.time()
        
        for query_category, queries in test_queries.items():
            print(f"\nTesting {query_category}...")
            category_results = {
                'category': query_category,
                'total_queries': len(queries),
                'successful_retrievals': 0,
                'total_results_found': 0,
                'average_relevance': 0.0,
                'query_details': []
            }
            
            for query in queries:
                try:
                    search_start = time.time()
                    results = self.ltm.search_similar(query, max_results=5)
                    search_duration = time.time() - search_start
                    
                    query_detail = {
                        'query': query,
                        'results_count': len(results),
                        'search_time': search_duration,
                        'results': []
                    }
                    
                    if results:
                        category_results['successful_retrievals'] += 1
                        category_results['total_results_found'] += len(results)
                        
                        print(f"  🎯 Query: '{query}' -> {len(results)} results")
                        
                        for j, result in enumerate(results[:3]):  # Show top 3
                            data = result.get('data', {})
                            distance = result.get('distance', 1.0)
                            similarity = 1.0 - distance
                            
                            result_detail = {
                                'text': data.get('input_text', '')[:100],
                                'similarity': similarity,
                                'category': data.get('metadata', {}).get('category', 'unknown')
                            }
                            query_detail['results'].append(result_detail)
                            
                            print(f"    {j+1}. [{similarity:.3f}] {data.get('input_text', '')[:80]}...")
                    else:
                        print(f"  ❌ Query: '{query}' -> No results found")
                    
                    category_results['query_details'].append(query_detail)
                    
                except Exception as e:
                    print(f"  ❌ Error with query '{query}': {e}")
                    self.test_results['errors'].append(f"Retrieval error: {e}")
            
            # Calculate average relevance
            if category_results['total_results_found'] > 0:
                total_similarity = sum(
                    sum(r['similarity'] for r in qd['results'])
                    for qd in category_results['query_details']
                )
                category_results['average_relevance'] = total_similarity / category_results['total_results_found']
            
            self.test_results['retrieval_tests'].append(category_results)
            success_rate = (category_results['successful_retrievals'] / category_results['total_queries']) * 100
            print(f"  📊 {query_category}: {category_results['successful_retrievals']}/{category_results['total_queries']} successful ({success_rate:.1f}%)")
            print(f"  🎯 Average relevance: {category_results['average_relevance']:.3f}")
        
        retrieval_duration = time.time() - retrieval_start_time
        total_queries = sum(len(queries) for queries in test_queries.values())
        print(f"\n⏱️ Total retrieval time: {retrieval_duration:.2f} seconds")
        print(f"📈 Average query time: {retrieval_duration/total_queries:.4f} seconds per query")
    
    def run_performance_tests(self):
        """Test system performance and generate metrics"""
        print("\n📊 PERFORMANCE ANALYSIS")
        print("-" * 30)
        
        try:
            stats = self.ltm.get_system_stats()
            mode_info = self.ltm.get_mode_info()
            
            self.test_results['performance_metrics'] = {
                'database_memories': stats['database_memories'],
                'database_size_mb': stats['database_size_mb'],
                'total_stored': stats['total_stored'],
                'total_retrieved': stats['total_retrieved'],
                'cache_hit_rate': stats['coordinate_cache_rate'],
                'cache_size': stats['coordinate_cache_size'],
                'mode': mode_info['current_mode'],
                'linking_enabled': mode_info['linking_enabled']
            }
            
            print(f"📈 Database Statistics:")
            print(f"  Total memories: {stats['database_memories']}")
            print(f"  Database size: {stats['database_size_mb']:.2f} MB")
            print(f"  Total stored: {stats['total_stored']}")
            print(f"  Total retrieved: {stats['total_retrieved']}")
            print(f"  Cache hit rate: {stats['coordinate_cache_rate']:.1f}%")
            print(f"  Cache size: {stats['coordinate_cache_size']}")
            print(f"  Mode: {mode_info['current_mode']}")
            print(f"  Linking: {'ENABLED' if mode_info['linking_enabled'] else 'DISABLED'}")
            
        except Exception as e:
            print(f"❌ Error getting performance metrics: {e}")
            self.test_results['errors'].append(f"Performance test error: {e}")
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n📋 TEST REPORT")
        print("=" * 50)
        
        # Storage summary
        total_storage_attempts = sum(t['total_items'] for t in self.test_results['storage_tests'])
        total_successful_stores = sum(t['successful_stores'] for t in self.test_results['storage_tests'])
        storage_success_rate = (total_successful_stores / total_storage_attempts) * 100 if total_storage_attempts > 0 else 0
        
        print(f"📝 STORAGE RESULTS:")
        print(f"  Total attempts: {total_storage_attempts}")
        print(f"  Successful: {total_successful_stores}")
        print(f"  Success rate: {storage_success_rate:.1f}%")
        
        # Retrieval summary
        total_retrieval_attempts = sum(t['total_queries'] for t in self.test_results['retrieval_tests'])
        total_successful_retrievals = sum(t['successful_retrievals'] for t in self.test_results['retrieval_tests'])
        retrieval_success_rate = (total_successful_retrievals / total_retrieval_attempts) * 100 if total_retrieval_attempts > 0 else 0
        
        print(f"\n🔍 RETRIEVAL RESULTS:")
        print(f"  Total queries: {total_retrieval_attempts}")
        print(f"  Successful: {total_successful_retrievals}")
        print(f"  Success rate: {retrieval_success_rate:.1f}%")
        
        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        if storage_success_rate >= 95 and retrieval_success_rate >= 80:
            print("  ✅ EXCELLENT - LTM system performing optimally")
        elif storage_success_rate >= 90 and retrieval_success_rate >= 70:
            print("  ✅ GOOD - LTM system performing well")
        elif storage_success_rate >= 80 and retrieval_success_rate >= 60:
            print("  ⚠️ ACCEPTABLE - LTM system needs optimization")
        else:
            print("  ❌ POOR - LTM system requires attention")
        
        # Error summary
        if self.test_results['errors']:
            print(f"\n⚠️ ERRORS ENCOUNTERED: {len(self.test_results['errors'])}")
            for error in self.test_results['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
        else:
            print(f"\n✅ NO ERRORS - Clean test execution")
    
    def cleanup(self):
        """Clean up test resources"""
        print(f"\n🧹 CLEANUP")
        print("-" * 30)
        
        try:
            if self.ltm:
                self.ltm.cleanup()
                print("✅ LTM system cleaned up")
            
            # Remove test database
            if os.path.exists(self.test_db_path):
                os.remove(self.test_db_path)
                print("✅ Test database removed")
                
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
    
    def run_full_test_suite(self):
        """Run the complete test suite"""
        if not self.setup():
            return False
        
        try:
            self.run_storage_tests()
            self.run_retrieval_tests()
            self.run_performance_tests()
            self.generate_test_report()
            return True
        except Exception as e:
            print(f"❌ Test suite failed: {e}")
            return False
        finally:
            self.cleanup()

def main():
    """Main test execution"""
    test_suite = LTMComprehensiveTest()
    success = test_suite.run_full_test_suite()
    
    if success:
        print(f"\n🎉 COMPREHENSIVE LTM TEST COMPLETED SUCCESSFULLY!")
    else:
        print(f"\n❌ COMPREHENSIVE LTM TEST FAILED!")
    
    return success

if __name__ == "__main__":
    main()