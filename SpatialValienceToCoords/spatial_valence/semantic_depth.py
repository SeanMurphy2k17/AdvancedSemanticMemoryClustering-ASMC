#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Depth Configuration for Spatial Valence Processing

This module defines the different levels of semantic analysis depth
available in the spatial valence processor.
"""

from enum import Enum

class SemanticDepth(Enum):
    """
    Semantic analysis depth levels for spatial valence processing
    
    FAST: Basic linguistic analysis optimized for real-time processing
    STANDARD: Enhanced semantic analysis with relationship mapping  
    DEEP: Maximum analysis with full context integration and advanced features
    """
    
    FAST = "fast"
    STANDARD = "standard" 
    DEEP = "deep"
    
    def __str__(self):
        return self.value
    
    def __repr__(self):
        return f"SemanticDepth.{self.name}"
    
    @property
    def description(self):
        """Get a description of this semantic depth level"""
        descriptions = {
            SemanticDepth.FAST: "Basic linguistic analysis for real-time processing (<1ms)",
            SemanticDepth.STANDARD: "Enhanced semantic analysis with relationships (2-5ms)",
            SemanticDepth.DEEP: "Maximum analysis with full context integration (10-20ms)"
        }
        return descriptions[self]
    
    @property
    def typical_processing_time_ms(self):
        """Get typical processing time in milliseconds for this depth"""
        times = {
            SemanticDepth.FAST: 1,
            SemanticDepth.STANDARD: 3, 
            SemanticDepth.DEEP: 15
        }
        return times[self]
    
    @property
    def features(self):
        """Get list of features enabled at this depth level"""
        features = {
            SemanticDepth.FAST: [
                "Basic word extraction",
                "Simple coordinate generation",
                "Fast deterministic processing"
            ],
            SemanticDepth.STANDARD: [
                "Enhanced word extraction", 
                "Basic relationship detection",
                "Improved coordinate precision",
                "Confidence scoring"
            ],
            SemanticDepth.DEEP: [
                "Advanced semantic analysis",
                "Emotional intelligence",
                "Temporal relationship detection", 
                "Context-aware processing",
                "Confidence scoring",
                "Semantic complexity analysis",
                "Enhanced grammatical parsing"
            ]
        }
        return features[self]

# Convenience constants for backward compatibility
FAST = SemanticDepth.FAST
STANDARD = SemanticDepth.STANDARD  
DEEP = SemanticDepth.DEEP

def get_all_depths():
    """Get all available semantic depth levels"""
    return list(SemanticDepth)

def get_recommended_depth(use_case: str) -> SemanticDepth:
    """
    Get recommended semantic depth for a specific use case
    
    Args:
        use_case: One of 'stm', 'ltm', 'consciousness', 'realtime', 'analysis'
        
    Returns:
        Recommended SemanticDepth level
    """
    recommendations = {
        'stm': SemanticDepth.DEEP,  # Universal DEEP mode
        'ltm': SemanticDepth.DEEP,  # Universal DEEP mode
        'consciousness': SemanticDepth.DEEP,  # Universal DEEP mode
        'realtime': SemanticDepth.FAST,  # When speed is critical
        'analysis': SemanticDepth.DEEP,  # When analysis quality matters
        'default': SemanticDepth.DEEP  # Universal recommendation
    }
    
    return recommendations.get(use_case.lower(), SemanticDepth.DEEP)

def compare_depths():
    """Print a comparison of all semantic depth levels"""
    print("🧠 SEMANTIC DEPTH COMPARISON")
    print("=" * 50)
    
    for depth in SemanticDepth:
        print(f"\n{depth.name} Mode:")
        print(f"  Description: {depth.description}")
        print(f"  Processing Time: ~{depth.typical_processing_time_ms}ms")
        print(f"  Features:")
        for feature in depth.features:
            print(f"    • {feature}")

if __name__ == "__main__":
    # Demo the semantic depth system
    compare_depths()
    
    print(f"\n🎯 UNIVERSAL RECOMMENDATION:")
    print(f"Use {SemanticDepth.DEEP} for maximum deterministic consistency!") 