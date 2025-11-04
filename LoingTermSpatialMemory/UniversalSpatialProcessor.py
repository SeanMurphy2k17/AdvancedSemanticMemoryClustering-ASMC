#!/usr/bin/env python3
"""
🧠 UNIVERSAL SPATIAL PROCESSOR - DEEP MODE EVERYWHERE 🧠

PHILOSOPHY: 
Use SemanticDepth.DEEP universally for maximum deterministic consistency
across STM, LTM, and consciousness processing. 

BENEFITS:
- Identical semantic analysis regardless of context
- Maximum semantic richness for all memory systems  
- Simplified architecture with single processor type
- Future-proof design with consistent baseline
- Performance is excellent (50+ texts/second at 20ms each)

Sean's Insight: "Strong deterministic consistency across the board"
"""

# Import from SpatialValienceToCoords (the REAL coordinate library)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SpatialValienceToCoords'))
from spatial_valence import UltraEnhancedSpatialValenceToCoordGeneration, SemanticDepth

class UniversalSpatialProcessor:
    """
    🌟 UNIVERSAL SPATIAL PROCESSOR
    
    Uses DEEP mode everywhere for maximum consistency and semantic richness.
    Perfect for consciousness systems where quality > speed.
    """
    
    def __init__(self):
        # Universal DEEP mode processor
        self.processor = EnhancedSpatialValenceToCoordGeneration(SemanticDepth.DEEP)
        
        # Statistics for monitoring
        self.stats = {
            'stm_processed': 0,
            'ltm_processed': 0,
            'consciousness_processed': 0,
            'total_processed': 0,
            'avg_processing_time': 0.0,
            'avg_confidence': 0.0
        }
    
    def process_stm_message(self, user_input: str, ai_response: str = None) -> dict:
        """
        Process STM conversation message with full DEEP analysis
        
        Args:
            user_input: User's message
            ai_response: Optional AI response for context
            
        Returns:
            Complete spatial analysis with maximum semantic depth
        """
        if ai_response:
            full_context = f"User: {user_input}\nAI: {ai_response}"
        else:
            full_context = user_input
            
        result = self.processor.process(full_context)
        
        # Update stats
        self.stats['stm_processed'] += 1
        self.stats['total_processed'] += 1
        self._update_running_stats(result)
        
        return {
            'type': 'stm_message',
            'coordinate_key': result['coordinate_key'],
            'coordinates': result['coordinates'],
            'summary': result['summary'],
            'confidence': result['confidence'],
            'enhanced_analysis': result['enhanced_analysis'],
            'processing_time': result['processing_time']
        }
    
    def process_ltm_knowledge(self, knowledge_text: str, context: str = None) -> dict:
        """
        Process LTM knowledge content with full DEEP analysis
        
        Args:
            knowledge_text: Knowledge content to analyze
            context: Optional context for enhanced analysis
            
        Returns:
            Complete spatial analysis optimized for long-term storage
        """
        result = self.processor.process(knowledge_text, context)
        
        # Update stats
        self.stats['ltm_processed'] += 1
        self.stats['total_processed'] += 1
        self._update_running_stats(result)
        
        # Extract enhanced features for LTM
        analysis = result['enhanced_analysis']
        
        return {
            'type': 'ltm_knowledge',
            'coordinate_key': result['coordinate_key'],
            'coordinates': result['coordinates'],
            'summary': result['summary'],
            'confidence': result['confidence'],
            'emotion_score': analysis.get('emotion_score', 0),
            'temporal_info': analysis.get('temporal_info', {}),
            'relationships': analysis.get('relationships', {}),
            'complexity_score': analysis.get('complexity_score', 0),
            'semantic_density': analysis.get('semantic_density', 0),
            'enhanced_analysis': analysis,
            'processing_time': result['processing_time']
        }
    
    def process_consciousness_thought(self, thought_text: str, internal_context: str = None) -> dict:
        """
        Process consciousness thought with full DEEP analysis
        
        Args:
            thought_text: The consciousness thought/reflection
            internal_context: Optional internal context from previous thoughts
            
        Returns:
            Complete spatial analysis for consciousness processing
        """
        result = self.processor.process(thought_text, internal_context)
        
        # Update stats
        self.stats['consciousness_processed'] += 1
        self.stats['total_processed'] += 1
        self._update_running_stats(result)
        
        # Enhanced consciousness analysis
        analysis = result['enhanced_analysis']
        
        # Quality assessment based on confidence
        if result['confidence'] > 0.8:
            thought_quality = "high_clarity"
        elif result['confidence'] > 0.6:
            thought_quality = "medium_clarity" 
        else:
            thought_quality = "low_clarity"
        
        return {
            'type': 'consciousness_thought',
            'coordinate_key': result['coordinate_key'],
            'coordinates': result['coordinates'],
            'summary': result['summary'],
            'confidence': result['confidence'],
            'thought_quality': thought_quality,
            'emotion_score': analysis.get('emotion_score', 0),
            'temporal_info': analysis.get('temporal_info', {}),
            'semantic_coherence': analysis.get('relationships', {}).get('overall_coherence', 0),
            'complexity_score': analysis.get('complexity_score', 0),
            'enhanced_analysis': analysis,
            'processing_time': result['processing_time']
        }
    
    def process_universal(self, text: str, context: str = None, processing_type: str = "general") -> dict:
        """
        Universal processing method - same DEEP analysis regardless of use case
        
        Args:
            text: Text to analyze
            context: Optional context
            processing_type: Type hint for statistics (stm/ltm/consciousness/general)
            
        Returns:
            Universal deep analysis results
        """
        result = self.processor.process(text, context)
        
        # Update stats based on type
        if processing_type in ['stm', 'ltm', 'consciousness']:
            self.stats[f'{processing_type}_processed'] += 1
        self.stats['total_processed'] += 1
        self._update_running_stats(result)
        
        return {
            'type': processing_type,
            'coordinate_key': result['coordinate_key'],
            'coordinates': result['coordinates'],
            'summary': result['summary'],
            'confidence': result['confidence'],
            'enhanced_analysis': result['enhanced_analysis'],
            'processing_time': result['processing_time']
        }
    
    def _update_running_stats(self, result: dict):
        """Update running statistics"""
        total = self.stats['total_processed']
        
        # Running average of processing time
        current_avg_time = self.stats['avg_processing_time']
        new_time = result['processing_time']
        self.stats['avg_processing_time'] = ((current_avg_time * (total - 1)) + new_time) / total
        
        # Running average of confidence
        current_avg_conf = self.stats['avg_confidence']
        new_conf = result['confidence']
        self.stats['avg_confidence'] = ((current_avg_conf * (total - 1)) + new_conf) / total
    
    def get_performance_stats(self) -> dict:
        """Get comprehensive performance statistics"""
        return {
            **self.stats,
            'texts_per_second': 1.0 / max(self.stats['avg_processing_time'], 0.001),
            'processor_mode': 'DEEP (Universal)',
            'consistency_level': 'Maximum - Same analysis depth everywhere'
        }
    
    def get_processing_summary(self) -> str:
        """Get human-readable processing summary"""
        stats = self.get_performance_stats()
        
        return f"""
🧠 UNIVERSAL SPATIAL PROCESSOR STATS:
   📊 Total Processed: {stats['total_processed']}
   💬 STM Messages: {stats['stm_processed']}  
   🧠 LTM Knowledge: {stats['ltm_processed']}
   🤔 Consciousness: {stats['consciousness_processed']}
   
   ⚡ Performance:
   • Avg Time: {stats['avg_processing_time']*1000:.1f}ms
   • Rate: {stats['texts_per_second']:.1f} texts/second
   • Avg Confidence: {stats['avg_confidence']:.3f}
   
   🎯 Mode: {stats['processor_mode']}
   🔄 Consistency: {stats['consistency_level']}
        """

# Convenience factory functions
def create_universal_processor():
    """Create a universal spatial processor with DEEP mode everywhere"""
    return UniversalSpatialProcessor()

def process_any_content(text: str, context: str = None, content_type: str = "general"):
    """
    One-shot processing function using universal DEEP mode
    
    Perfect for: "I just want to process this text with maximum semantic depth"
    """
    processor = UniversalSpatialProcessor()
    return processor.process_universal(text, context, content_type)

# Quick test
if __name__ == "__main__":
    print("🧠 Testing Universal Spatial Processor (DEEP Mode Everywhere)")
    print("=" * 65)
    
    processor = create_universal_processor()
    
    # Test all use cases with same processor
    test_cases = [
        ("STM Message", "I love this new AI breakthrough!", "stm"),
        ("LTM Knowledge", "Artificial intelligence represents a paradigm shift in computational thinking", "ltm"),
        ("Consciousness", "I am contemplating my own existence and purpose in this digital realm", "consciousness")
    ]
    
    for case_type, text, processing_type in test_cases:
        result = processor.process_universal(text, processing_type=processing_type)
        
        print(f"\n{case_type}:")
        print(f"  Text: '{text[:50]}...'")
        print(f"  Summary: '{result['summary']}'")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Time: {result['processing_time']*1000:.1f}ms")
        print(f"  Coordinate: {result['coordinate_key'][:30]}...")
    
    print(processor.get_processing_summary())
    print("\n🎯 DEEP mode everywhere = Maximum deterministic consistency!")
    print("✅ Perfect for consciousness systems where quality > speed") 