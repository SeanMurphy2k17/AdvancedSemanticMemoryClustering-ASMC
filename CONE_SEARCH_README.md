# 🧠 Cone Search - Semantic Memory Retrieval

## Overview
Cone search is a hippocampus-inspired memory retrieval system that uses directional semantic filtering to return contextually-relevant memories. Instead of simple radial search, it uses the query's 9D semantic coordinate to create a cone-shaped search pattern in semantic space.

## How It Works

### 1. Semantic Coordinate Extraction
When you query the system, it generates a 9D semantic coordinate using spatial valence:
```python
query = "What dangerous things are ahead?"
coords = [0.45, -0.32, 0.61, 0.12, -0.45, 0.78, 0.33, 0.21, -0.12]
         └─ position ─┘  └─ direction ─┘  └─── shape ───┘
```

### 2. Cone Parameters
The 9D coordinate is split into three components:
- **Position (dims 0-2)**: Semantic origin point
- **Direction (dims 3-5)**: Semantic search direction (normalized)
- **Shape (dims 6-8)**: Cone geometry (radius, taper, length)

### 3. Adaptive Scaling
Cone size adapts based on two factors:

**Memory Density:**
```python
memory_density = total_memories / node_count
cone_radius = shape[0] * memory_density
cone_length = shape[2] * memory_density
```

**Tightening Factor** (system maturity):
```python
# Exponential curve: -1.0 (loose) → +0.9999 (tight)
k = 0.001
tightening = -1.0 + (2.0 / (1.0 + exp(-k * total_system_memories)))

# Apply to cone
cone_radius *= (1.0 + tightening)
cone_length *= (1.0 + tightening)
```

**Result:** System starts with wide searches (exploratory) and tightens over time (precise).

### 4. Three-Stage Filtering

**STAGE 1: Radial Retrieval**
- Get all nodes in cluster
- Count memories for density calculation

**STAGE 2: Direction Filter**
- Calculate semantic offset for each memory
- Check alignment using dot product
- Keep only memories in forward hemisphere (alignment > 0)

**STAGE 3: Cone Shape Filter**
- Project offset onto direction axis
- Check if within cone length
- Calculate perpendicular distance from axis
- Keep only memories inside cone geometry

### 5. Relevance Scoring
Surviving memories are scored by 4 factors:
```python
relevance = (
    0.30 * angular_alignment +    # Pointing same semantic direction
    0.25 * distance_falloff +     # Closer to origin
    0.20 * axis_alignment +       # Closer to cone axis
    0.25 * density_boost          # More memories in this region
)
```

## Usage

### Basic Radial Search (Default)
```python
context = memory.get_spatial_context(
    structure_type='spatial',
    cluster_id='dungeon_level_1',
    coordinates={'x': 10, 'y': 5, 'z': 0},
    radius=5,
    max_memories=10
)
```

### Cone Search (New)
```python
context = memory.get_spatial_context(
    structure_type='spatial',
    cluster_id='dungeon_level_1',
    coordinates={'x': 10, 'y': 5, 'z': 0},
    use_cone_search=True,           # Enable cone search
    query_text="What's ahead?",     # Query for semantic coord generation
    max_memories=10
)
```

## Return Format

```python
{
    'success': True,
    'cone_search_used': True,
    'memories': [
        {
            'coord_key': '[0.45][-0.32][0.61]...',
            'semantic_summary': 'Explored corridor ahead',
            'relevance_score': 0.87,
            'alignment': 0.95,
            'projection': 3.2,
            'timestamp': '2024-12-24T10:30:00',
            'valence': 0.42
        },
        # ... more memories
    ],
    'cone_stats': {
        'stage1_radial': 45,          # Nodes in cluster
        'stage2_direction': 23,       # Passed direction filter
        'stage3_cone': 12,            # Inside cone shape
        'final_returned': 10,         # Top scored
        'tightening_factor': 0.73,    # System maturity
        'cone_radius': 2.1,           # Computed radius
        'cone_taper': 0.2,            # Computed taper
        'cone_length': 8.5,           # Computed length
        'memory_density': 3.8         # Memories per node
    }
}
```

## Why This Works

### Hippocampal Analogy
Real hippocampal memory retrieval:
- Query creates activation pattern (semantic coordinate)
- Activation propagates along neural pathways (direction)
- Denser neural regions get stronger activation (density boost)
- Similar patterns reinforce (dot product alignment)

### Emergent Behavior
- **Not intentional:** The system doesn't "choose" to search in a direction
- **Emergent:** The semantic coordinate naturally biases retrieval
- **Adaptive:** Cone tightens as system gains experience
- **Contextual:** Similar queries get similar memory patterns

## Benefits

1. **More Relevant Memories:** Directional bias returns contextually-aligned memories
2. **Adaptive Learning:** System tightens search as it matures
3. **Density-Aware:** Boosts important (memory-rich) semantic regions
4. **Emergent Intelligence:** No manual tuning required

## Implementation Details

**File:** `AdvancedSemanticMemoryClustering/ASMC_API.py`

**New Parameters:**
- `use_cone_search: bool = False` - Enable cone mode
- `query_text: str = None` - Query for semantic coord generation

**Helper Methods:**
- `_normalize_vector()` - Normalize 3D vectors
- `_magnitude()` - Calculate vector magnitude
- `_dot_product()` - Vector dot product
- `_subtract_vectors()` - Vector subtraction
- `_scale_vector()` - Vector scaling
- `_parse_coordinate_key()` - Parse semantic coordinate strings

**Total Addition:** ~200 lines of code

## Testing

Test results show correct behavior:
- ✅ Vector normalization (magnitude = 1.0)
- ✅ Dot product (1.0 = aligned, 0.0 = perpendicular, -1.0 = opposite)
- ✅ Coordinate parsing (9D → position, direction, shape)
- ✅ Cone geometry (memories inside cone pass, outside fail)
- ✅ Direction filter (forward hemisphere only)

## Future Enhancements

Potential improvements:
- [ ] Ellipsoid shaping (non-uniform cone cross-section)
- [ ] Multi-cone search (parallel semantic directions)
- [ ] Temporal decay (older memories fade)
- [ ] Cross-cluster cone search
- [ ] Semantic attractor points (gravity wells in semantic space)

---

**Authors:** Sean Murphy (Human Inventor) & Claude AI (Implementation Partner)  
**License:** MIT  
**Date:** December 24, 2024





