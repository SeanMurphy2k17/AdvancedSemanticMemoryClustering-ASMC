"""
Spatial Valence Solution - ULTRA semantic analysis for AI consciousness systems

This package provides ULTRA-ROBUST spatial valence processing with:
- 6-Layer semantic analysis (lexical, syntactic, semantic, contextual, embedding, relational)
- Aggressive semantic capture that never misses relationships
- Word embeddings and semantic networks
- Emotional intelligence and confidence scoring
- Temporal relationship detection
- Universal deterministic consistency
- 100% backward compatibility

Example Usage:
    from spatial_valence import SpatialValenceToCoordGeneration
    
    # Automatically uses ULTRA mode for maximum semantic capture!
    processor = SpatialValenceToCoordGeneration()
    result = processor.process("I absolutely love this amazing AI system!")
    
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Emotion: {result['enhanced_analysis']['sentiment']['polarity']:.3f}")
"""

# Import SemanticDepth first
from .semantic_depth import SemanticDepth

# Import ULTRA processor as the core implementation
from .ultra_processor import EnhancedSpatialValenceToCoordGeneration, UltraRobustSemanticEncoder

# Import universal processor 
from .universal_processor import UniversalSpatialProcessor

# Backward compatibility aliases
SpatialValenceToCoordGeneration = EnhancedSpatialValenceToCoordGeneration

__version__ = "3.0.0"
__author__ = "Sean"
__email__ = "your-email@example.com"
__description__ = "Spatial valence processor for AI consciousness systems"

__all__ = [
    "SpatialValenceToCoordGeneration",
    "EnhancedSpatialValenceToCoordGeneration",
    "UltraRobustSemanticEncoder",
    "SemanticDepth",
    "UniversalSpatialProcessor",
]

# Version info
VERSION_INFO = {
    'major': 3,
    'minor': 0,
    'patch': 0,
    'release': 'stable'
}

def get_version():
    """Get the package version string"""
    return __version__

def get_info():
    """Get package information"""
    return {
        'name': 'spatial-valence-solution',
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'capabilities': [
            'ULTRA mode with 6-layer analysis',
            'Aggressive semantic capture',
            'Word embeddings and semantic networks', 
            'Emotional intelligence',
            'Confidence scoring',
            'Temporal relationship detection',
            'Never misses semantic relationships',
            'Advanced NLP techniques'
        ]
    } 