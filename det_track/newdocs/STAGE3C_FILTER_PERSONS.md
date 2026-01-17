# Stage 3c: Filter Persons & Extract Final Crops

**Implementation**: `stage3c_filter_persons.py`

## Purpose
Filters canonical persons from Stage 3b (~40+ persons) to top 8-10 persons based on ranking scores, applies late-appearance penalty, and extracts 60 representative crops per person using 3-bin contiguous selection.

## Inputs
- `canonical_persons.npz`: 40+ canonical persons from Stage 3b
- `crops_cache.pkl`: 527 MB cache with all 8832 crops (created in Stage 1)

## Outputs
- `canonical_persons_3c.npz`: Filtered 8-10 persons (input for Stage 3d)
- `final_crops_3c.pkl`: 39 MB (60 crops per person × 8-10 persons, input for Stage 3d)
- `crops_cache.pkl`: Deleted (auto-cleanup)

## Processing Flow

```
canonical_persons.npz (40+ persons from Stage 3b)
    ↓
Load crops_cache.pkl
    ↓
Rank all persons (duration, coverage, center, smoothness)
    ↓
Select TOP 10 persons
    ↓
Apply late-appearance penalty to top 10 (may reduce to ~8)
    ↓
For each selected person:
    ├─→ Extract person's detection_indices
    ├─→ O(1) lookup: crops_by_idx[detection_idx]
    ├─→ Collect all person's crops (100-500 crops)
    ├─→ Sort by frame_number (temporal order)
    ├─→ Apply 3-bin contiguous selection
    └─→ Store 60 final crops
    ↓
canonical_persons_3c.npz (8-10 persons) + final_crops_3c.pkl (60 × 8-10 = 480-600 crops)
    ↓
Delete crops_cache.pkl (527 MB freed)
```

## Performance

| Metric | Value |
|--------|-------|
| Time | 0.95s |
| Input persons (from 3b) | 40+ |
| Output persons (to 3d) | 8-10 |
| Output crops/person | 60 |
| Total final crops | 480-600 |
| Memory freed | 527 MB (cache deletion) |

**Key Achievement**: 11× faster than naive approach (1.0s vs 10-12s)

## 3-Bin Contiguous Selection Algorithm

### Overview
Divide person's timeline into 3 equal sections, extract 20 consecutive frames from middle of each section.

### Algorithm
```python
person_crops.sort(key=lambda x: x['frame_number'])  # Temporal order

total_crops = len(person_crops)
bin_size = total_crops // 3

# Divide into 3 bins
bins = [
    person_crops[:bin_size],                    # 0-33% (beginning)
    person_crops[bin_size:2*bin_size],          # 33-67% (middle)
    person_crops[2*bin_size:]                   # 67-100% (end)
]

# Extract 20 contiguous frames from middle of each bin
crops_per_bin = 20
selected_crops = []

for bin_crops in bins:
    bin_len = len(bin_crops)
    mid_point = bin_len // 2
    start_idx = max(0, mid_point - crops_per_bin // 2)
    end_idx = min(bin_len, start_idx + crops_per_bin)
    
    selected_crops.extend(bin_crops[start_idx:end_idx])

return selected_crops  # 60 crops total
```

### Example
Person appears for 240 frames:
```
Bin 1 (0-80):     Extract frames [30-50] (20 consecutive) = beginning
Bin 2 (80-160):   Extract frames [110-130]             = middle
Bin 3 (160-240):  Extract frames [190-210]             = end

Result: 60 crops with temporal diversity
```

## Design Decisions

### Why 3 Bins (Not Hybrid 10-Bin)?
**Tested approaches**:
1. **Pure quality scoring** (v1): Slow O(n log n), requires ranking
2. **Hybrid 10-bin** (v2): Complex, diminishing returns
3. **3-bin quality** (v3): Select top 20 by quality per bin
4. **3-bin contiguous** (v4): ✅ **SELECTED** - Simple, deterministic

**Why contiguous over quality?**
- Quality scoring adds complexity + compute
- Contiguous frames provide smooth animations in WebP
- Deterministic = reproducible (same person → same crops)
- No quality bias = unbiased representation

### Crop Size Consistency
All crops guaranteed to be <400px (aspect ratio preserved):
```
Crop height: 1-400px
Crop width: 1-400px
Result: No "jumpy" resizing in Stage 4 WebP generation
```

### Why Delete crops_cache After?
```
Stage 1 output: crops_cache.pkl (527 MB)
Stage 3c input:  Load 527 MB
Stage 3c output: final_crops_3c.pkl (39 MB - only top crops)
After output:    DELETE 527 MB ✅

Net disk usage: 527 MB → 39 MB (93% reduction)
```

Automatic cleanup on success ensures no stale data.

## O(1) Lookup Optimization

**Critical**: Speed from detection_idx linkage.

```python
# Build lookup table (single pass through cache)
crops_by_idx = {crop['detection_idx']: crop for crop in all_crops}
# O(n) build time, then O(1) per lookup

# Extract person's crops
for detection_idx in person['detection_indices']:
    crop = crops_by_idx[detection_idx]  # O(1) ✅
    selected_crops.append(crop)

# Result: 0.95s for 600 crops (vs 10-12s if re-reading video)
```

## Configuration

```yaml
stage3c_filter:
  selection:
    crops_per_person: 60      # Top crops per person
    selection_method: contiguous  # or quality (legacy)
    
    # Contiguous selection
    bins: 3                    # Beginning/middle/end
    crops_per_bin: 20          # 60 ÷ 3 = 20 per bin
    
  # Auto-cleanup
  delete_crops_cache_after_success: true
```

## Output Data Structure

```python
final_crops_3c = {
    'crops_with_quality': [
        {
            'person_id': 0,
            'crops': [
                {
                    'crop': np.ndarray,        # Resized image
                    'frame_number': 45,
                    'detection_idx': 42,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': 0.92
                },
                ...  # 60 crops
            ]
        },
        ...  # 10 persons
    ]
}
```

## Performance Comparison

| Approach | Time | Quality | Determinism |
|----------|------|---------|-------------|
| Pure quality (top 60) | 1.2s | Highest | ✅ |
| Hybrid 10-bin | 0.9s | High | ✅ |
| 3-bin quality | 0.88s | High | ✅ |
| **3-bin contiguous** | **0.95s** | **Good** | **✅** |
| Naive re-read | 10-12s | Best | ❌ |

**Selected**: 3-bin contiguous (best balance of simplicity + speed)

## Design Rationale

### Why Not Just Top-60 by Quality?
- Biased toward well-lit, frontal views
- Misses interesting motion moments
- Users prefer temporal variety over perfect quality

### Why Not Re-Read Video?
```
Option A: Re-read video in Stage 3c
  Cost: 10-12s (video decode + crop extract)
  Benefit: Perfect crops (no stale cache)
  
Option B: Eager extract in Stage 1, cache
  Cost: +5s in Stage 1
  Benefit: -11s in Stage 3c (O(1) lookup)
  Net: -6s ✅
  
Selected: Option B
```

## Cache Auto-Cleanup

On successful Stage 3c completion:
```python
if success:
    try:
        os.remove(crops_cache_path)
        print(f"✓ Deleted {size_mb:.1f} MB cache")
    except Exception as e:
        print(f"⚠ Failed to delete cache: {e}")
        # Dont fail pipeline, just warn
```

**Why graceful error handling?** In case of permission issues or file locks.

## Performance Notes

- **Time**: 0.95s (includes file I/O)
- **Memory peak**: 527 MB (loading crops_cache)
- **Determinism**: 100% (same input → same output)
- **Reproducibility**: Can re-run safely, no randomness

---

**Related**: [Back to Master](README_MASTER.md) | [← Stage 3b](STAGE3B_GROUPING.md) | [Stage 3d →](STAGE3D_VISUAL_REFINEMENT.md)
