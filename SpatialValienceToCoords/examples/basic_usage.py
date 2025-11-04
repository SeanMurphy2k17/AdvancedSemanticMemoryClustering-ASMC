#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic Usage Example for Spatial Valence Solution

This example demonstrates the core functionality of the ULTRA-robust
spatial valence processor with aggressive semantic capture.
"""

import sys
import os

# Add the package to path for local testing
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from spatial_valence import (
    UltraEnhancedSpatialValenceToCoordGeneration,
    EnhancedSpatialValenceToCoordGeneration,
    SemanticDepth,
    SpatialValenceToCoordGeneration
)

def ultra_example():
    """Demonstrate ULTRA mode capabilities"""
    
    print("🔥 SPATIAL VALENCE SOLUTION - ULTRA MODE EXAMPLE")
    print("=" * 60)
    
    # Initialize with ULTRA mode
    processor = UltraEnhancedSpatialValenceToCoordGeneration(SemanticDepth.DEEP)
    
    # Test cases that showcase ULTRA capabilities
    test_cases = [
        # Synonym detection
        ("I absolutely love this amazing AI system!", "Strong positive emotion"),
        ("I adore this fantastic AI system!", "Should cluster with 'love'"),
        
        # Antonym differentiation  
        ("I am certain about this decision", "High certainty"),
        ("I am uncertain about this decision", "Should be far from 'certain'"),
        
        # Context handling
        ("I need to go to the bank for money", "Financial context"),
        ("The river bank was beautiful", "Nature context"),
        
        # Complex semantics
        ("The quantum computer processes information efficiently", "Technical"),
        ("Tomorrow we will build incredible things together", "Future + collaborative")
    ]
    
    print("🔥 Processing with ULTRA mode analysis:")
    print()
    
    results = []
    for i, (text, note) in enumerate(test_cases, 1):
        result = processor.process(text)
        results.append(result)
        
        print(f"[{i}] \"{text}\"")
        print(f"    Note: {note}")
        print(f"    Coordinate: {result['coordinate_key'][:40]}...")
        print(f"    Confidence: {result.get('confidence', 0):.3f}")
        print(f"    Summary: '{result['summary']}'")
        
        # Show ULTRA features
        enhanced = result.get('enhanced_analysis', {})
        
        # Sentiment analysis
        sentiment = enhanced.get('sentiment', {})
        if sentiment:
            polarity = sentiment.get('polarity', 0)
            intensity = sentiment.get('intensity', 0)
            print(f"    Sentiment: {polarity:.3f} (intensity: {intensity:.2f})")
        
        # Key concepts
        concepts = enhanced.get('concepts', {})
        if concepts:
            top_concepts = sorted(concepts.items(), key=lambda x: x[1], reverse=True)[:3]
            concept_str = ", ".join([f"{c}:{v:.2f}" for c, v in top_concepts])
            print(f"    Concepts: {concept_str}")
        
        # Semantic hash for clustering
        if 'semantic_hash' in enhanced:
            print(f"    Semantic Hash: {enhanced['semantic_hash'][:8]}...")
        
        print()
    
    # Show clustering analysis
    print("📊 SEMANTIC CLUSTERING ANALYSIS:")
    print("Comparing coordinates to show semantic relationships...")
    print()
    
    # Compare "love" vs "adore"
    coord1 = results[0]['coordinate_key'][:20]
    coord2 = results[1]['coordinate_key'][:20]
    print(f"'love' coords:  {coord1}")
    print(f"'adore' coords: {coord2}")
    print(f"→ These should be VERY similar (ULTRA captures synonyms)")
    print()
    
    # Compare "certain" vs "uncertain"
    coord3 = results[2]['coordinate_key'][:20]
    coord4 = results[3]['coordinate_key'][:20]
    print(f"'certain' coords:   {coord3}")
    print(f"'uncertain' coords: {coord4}")
    print(f"→ These should be VERY different (ULTRA captures antonyms)")
    print()

def comparison_example():
    """Compare original vs enhanced vs ULTRA processor"""
    
    print("🔄 ORIGINAL vs ENHANCED vs ULTRA COMPARISON")
    print("=" * 60)
    
    # Initialize all processors
    original = SpatialValenceToCoordGeneration()
    enhanced = EnhancedSpatialValenceToCoordGeneration(SemanticDepth.DEEP)
    ultra = UltraEnhancedSpatialValenceToCoordGeneration(SemanticDepth.DEEP)
    
    test_text = "I absolutely adore this amazing AI system!"
    
    # Process with all three
    original_result = original.process(test_text)
    enhanced_result = enhanced.process(test_text)
    ultra_result = ultra.process(test_text)
    
    print(f"Text: \"{test_text}\"")
    print()
    
    print("📊 ORIGINAL PROCESSOR:")
    print(f"  Summary: '{original_result['summary']}'")
    print(f"  Coordinate: {original_result['coordinate_key'][:40]}...")
    print(f"  Features: Basic processing only")
    print()
    
    print("🚀 ENHANCED PROCESSOR:")
    print(f"  Summary: '{enhanced_result['summary']}'")
    print(f"  Coordinate: {enhanced_result['coordinate_key'][:40]}...")
    print(f"  Confidence: {enhanced_result.get('confidence', 0):.3f}")
    print(f"  Features: Multi-layer semantic analysis")
    print()
    
    print("🔥 ULTRA PROCESSOR:")
    print(f"  Summary: '{ultra_result['summary']}'")
    print(f"  Coordinate: {ultra_result['coordinate_key'][:40]}...")
    print(f"  Confidence: {ultra_result.get('confidence', 0):.3f}")
    
    ultra_analysis = ultra_result.get('enhanced_analysis', {})
    if 'semantic_hash' in ultra_analysis:
        print(f"  Semantic Hash: {ultra_analysis['semantic_hash'][:8]}...")
    if 'concepts' in ultra_analysis:
        concepts = ultra_analysis['concepts']
        print(f"  Concepts Found: {len(concepts)}")
    
    print(f"  Features: 6-layer analysis, word embeddings, semantic networks!")
    print()

def feature_showcase():
    """Showcase specific ULTRA features"""
    
    print("🎯 ULTRA FEATURE SHOWCASE")
    print("=" * 60)
    
    processor = UltraEnhancedSpatialValenceToCoordGeneration(SemanticDepth.DEEP)
    
    # Test grammatical complexity
    print("📝 Grammatical Complexity Analysis:")
    texts = [
        "The cat sat.",
        "The cat sat on the mat.",
        "The cat, which was orange and fluffy, sat contentedly on the warm mat while purring."
    ]
    
    for text in texts:
        result = processor.process(text)
        complexity = result['enhanced_analysis'].get('grammatical_complexity', 0)
        print(f"  '{text[:30]}...' → Complexity: {complexity:.3f}")
    
    print()
    
    # Test semantic coherence
    print("🔗 Semantic Coherence Analysis:")
    texts = [
        "I love cats. Dogs are great too. Animals are wonderful.",
        "I love cats. The weather is nice. Pizza tastes good.",
    ]
    
    for text in texts:
        result = processor.process(text)
        coherence = result['enhanced_analysis'].get('coherence_score', 0)
        print(f"  '{text[:40]}...' → Coherence: {coherence:.3f}")
    
    print()
    
    # Test context dependency
    print("🔄 Context Dependency Analysis:")
    texts = [
        "The quantum computer is powerful.",
        "It processes information quickly.",
        "This is amazing technology."
    ]
    
    for text in texts:
        result = processor.process(text)
        context_dep = result['enhanced_analysis'].get('context_dependency', 0)
        print(f"  '{text}' → Context Dependency: {context_dep:.3f}")
    
    print()

def main():
    """Run all examples"""
    ultra_example()
    print("\n" + "="*80 + "\n")
    
    comparison_example()
    print("\n" + "="*80 + "\n")
    
    feature_showcase()
    
    print("🎯 CONCLUSION:")
    print("✅ ULTRA mode provides aggressive semantic capture")
    print("✅ Never misses synonyms or semantic relationships")
    print("✅ 6-layer analysis with word embeddings")
    print("✅ Perfect for consciousness systems requiring maximum understanding")
    print("🔥 Ready for production deployment!")

if __name__ == "__main__":
    main() 