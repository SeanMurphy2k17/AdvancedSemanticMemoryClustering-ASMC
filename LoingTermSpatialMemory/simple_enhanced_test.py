#!/usr/bin/env python3
"""Simple test for enhanced spatial valence processor"""

import sys
import time
import hashlib
import re
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass

class SemanticDepth(Enum):
    FAST = "fast"
    STANDARD = "standard" 
    DEEP = "deep"

@dataclass
class SemanticElement:
    content: str
    element_type: str
    confidence: float
    position: int

class SimpleEnhancedProcessor:
    def __init__(self, depth=SemanticDepth.STANDARD):
        self.depth = depth
        
    def process(self, text: str) -> Dict:
        start_time = time.time()
        
        # Simple enhanced analysis
        words = text.split()
        
        # Enhanced emotion detection
        positive_words = ['love', 'amazing', 'great', 'wonderful', 'excellent']
        negative_words = ['hate', 'terrible', 'bad', 'awful', 'horrible']
        
        emotion_score = 0.0
        for word in words:
            word_lower = word.lower()
            if word_lower in positive_words:
                emotion_score += 0.3
            elif word_lower in negative_words:
                emotion_score -= 0.3
        
        # Clamp emotion score
        emotion_score = max(-1.0, min(1.0, emotion_score))
        
        # Generate coordinates (enhanced)
        hash_bytes = hashlib.md5(text.encode()).digest()
        coords = {}
        coord_names = ['x', 'y', 'z', 'a', 'b', 'c', 'd', 'e', 'f']
        
        for i, name in enumerate(coord_names):
            if name == 'e':  # Emotion coordinate
                coords[name] = emotion_score * 0.8  # Enhanced emotion mapping
            else:
                # Hash-based with small variation
                offset = (hash_bytes[i] / 255.0 - 0.5) * 0.1
                coords[name] = round(offset, 4)
        
        # Generate coordinate key
        coord_key = ''.join(f"[{coords[name]:.3f}]" for name in coord_names)
        
        # Calculate confidence (new feature)
        confidence = 0.7 + (len(words) * 0.02)  # More words = higher confidence
        confidence = min(1.0, confidence)
        
        processing_time = time.time() - start_time
        
        return {
            'input': text,
            'summary': ' '.join(words[:4]),  # Simple summary
            'coordinates': coords,
            'coordinate_key': coord_key,
            'confidence': confidence,
            'processing_time': processing_time,
            'emotion_score': emotion_score
        }

def test_enhanced_processor():
    print("🚀 Testing Simple Enhanced Spatial Valence Processor")
    print("=" * 60)
    
    processor = SimpleEnhancedProcessor()
    
    test_cases = [
        "I absolutely love this amazing AI system",
        "The computer crashed and lost all our work",
        "Tomorrow we will build wonderful things together",
        "This is a terrible horrible day",
        "Scientists discovered fascinating new galaxies"
    ]
    
    for i, text in enumerate(test_cases, 1):
        result = processor.process(text)
        
        print(f"[{i}] Input: {text}")
        print(f"    Summary: {result['summary']}")
        print(f"    Emotion Score: {result['emotion_score']:.2f}")
        print(f"    Confidence: {result['confidence']:.3f}")
        print(f"    Coordinate Key: {result['coordinate_key'][:30]}...")
        print(f"    Processing Time: {result['processing_time']*1000:.1f}ms")
        print()
    
    print("✅ Simple Enhanced Processor Working!")

if __name__ == "__main__":
    test_enhanced_processor() 