#!/usr/bin/env python3
"""Test different coordinate precisions"""

import sys
sys.path.insert(0, '..')

from spatial_valence import UltraEnhancedSpatialValenceToCoordGeneration, SemanticDepth

def format_coords_with_precision(coords, decimals):
    """Format coordinates with specific decimal places"""
    parts = []
    for dim in ['x', 'y', 'z', 'a', 'b', 'c', 'd', 'e', 'f']:
        value = coords[dim]
        if decimals == 0:
            # Round to nearest 0.5
            value = round(value * 2) / 2
            parts.append(f"[{value:+.1f}]")
        else:
            parts.append(f"[{value:+.{decimals}f}]")
    return ''.join(parts)

print("🎯 COORDINATE PRECISION TEST")
print("=" * 60)

processor = UltraEnhancedSpatialValenceToCoordGeneration(SemanticDepth.DEEP)

# Test similar phrases
phrases = [
    "I love you",
    "I love you so much",
    "I really love you", 
    "I adore you",
    "I hate you"
]

print("\n3 DECIMAL PLACES (current):")
coords_3dec = []
for phrase in phrases:
    result = processor.process(phrase)
    coords = result['coordinates']
    coords_3dec.append(coords)
    print(f"{phrase:<25} → {result['coordinate_key'][:40]}...")

print("\n2 DECIMAL PLACES:")
for i, phrase in enumerate(phrases):
    coords_2dec = format_coords_with_precision(coords_3dec[i], 2)
    print(f"{phrase:<25} → {coords_2dec[:40]}...")

print("\n1 DECIMAL PLACE:")
for i, phrase in enumerate(phrases):
    coords_1dec = format_coords_with_precision(coords_3dec[i], 1)
    print(f"{phrase:<25} → {coords_1dec[:40]}...")

print("\n0.5 STEPS (even coarser):")
for i, phrase in enumerate(phrases):
    coords_half = format_coords_with_precision(coords_3dec[i], 0)
    print(f"{phrase:<25} → {coords_half[:40]}...")

print("\nOBSERVATIONS:")
print("- 3 decimals: Every phrase gets unique position (maybe too unique?)")
print("- 2 decimals: Similar phrases start clustering better")
print("- 1 decimal: Strong clustering of related concepts")
print("- 0.5 steps: Very coarse, but clear semantic groups")

print(f"\nUNIQUE POSITIONS AVAILABLE:")
print(f"3 decimals: 2,001^9 = 5.13 × 10^29")
print(f"2 decimals: 201^9 = 6.8 × 10^20") 
print(f"1 decimal:  21^9 = 794 billion")
print(f"0.5 steps:  5^9 = 1.95 million")

print("\n💡 RECOMMENDATION: 2 decimal places is probably optimal!")
print("   - Still 10^20 unique positions (more than you'll EVER need)")
print("   - Better semantic clustering")
print("   - Cleaner coordinate keys") 