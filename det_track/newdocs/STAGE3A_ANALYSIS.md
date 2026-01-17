# Stage 3a: Tracklet Analysis

**Implementation**: `stage3a_analyze_tracklets.py`

## Purpose
Compute statistics for each tracklet to enable ranking and identity resolution in later stages.

## Inputs
- `tracklets_raw.npz`: 67 tracklets from ByteTrack

## Outputs
- `tracklet_stats.npz`: Statistics for each tracklet (duration, coverage, center position, smoothness, velocity)

## Processing Flow

```
tracklets_raw.npz (67 tracklets)
    ↓
For each tracklet:
    ├─→ Compute duration (# frames present)
    ├─→ Compute confidence statistics (mean, std, min)
    ├─→ Compute bbox coverage (area × duration)
    ├─→ Estimate appearance (mean of crops)
    └─→ Create ranking score
    ↓
Identify potential merges
    ├─→ Group tracklets by temporal overlap
    ├─→ Calculate visual similarity (OSNet features)
    └─→ Flag high-similarity pairs for Stage 3d
    ↓
tracklet_stats.npz + reid_candidates.json
```

## Performance

| Metric | Value |
|--------|-------|
| Time | 0.23s |
| Tracklets analyzed | 67 |
| Statistics computed | 7 per tracklet |
| ReID candidates | ~5-10 pairs |

## Computed Statistics

```python
tracklet_stats = {
    'tracklet_id': [0, 1, 2, ...],
    'duration': np.array([120, 45, 200, ...]),      # Frames present
    'confidence_mean': np.array([0.92, 0.88, ...]), # Mean detection score
    'confidence_min': np.array([0.75, 0.60, ...]),  # Worst detection
    'bbox_area': np.array([50000, 30000, ...]),     # Avg pixel area
    'coverage': np.array([6e6, 1.35e6, ...]),       # area × duration
    'visibility_ratio': np.array([0.98, 0.95, ...]), # (present frames / total)
    'rank_score': np.array([0.89, 0.72, ...])       # Combined ranking
}
```

## Ranking Algorithm

### Combined Rank Score
```
rank_score = (
    0.5 × (duration / max_duration)           # Prefer long tracklets
    + 0.3 × confidence_mean                    # Prefer high confidence
    + 0.2 × (coverage / max_coverage)         # Prefer large, long tracklets
)
```

**Rationale**: Prioritize tracklets that represent substantial parts of the video.

### Late-Appearance Penalty
Tracklets appearing only in final frames are penalized:
```python
if first_frame / total_frames > 0.75:  # Appears in last 25% only
    rank_score *= 0.7  # 30% penalty
```

**Why?** Users prefer seeing people throughout video, not just at end.

## ReID Candidates Identification

Tracklets are flagged for potential visual merging if:
1. Temporal overlap: `|start_time - other_start_time| < 200 frames`
2. Geometric separation: Not in same bbox area consistently
3. Feature similarity: Cosine distance < 0.4 (will be verified in Stage 3d)

```json
{
  "candidates": [
    {"tracklet_id_a": 5, "tracklet_id_b": 12, "confidence": 0.68},
    {"tracklet_id_a": 3, "tracklet_id_b": 19, "confidence": 0.74},
    ...
  ]
}
```

## Configuration

```yaml
stage3a_analyze:
  ranking:
    duration_weight: 0.5       # How much to favor long tracklets
    confidence_weight: 0.3     # How much to favor high-confidence detections
    coverage_weight: 0.2       # How much to favor large bboxes
    
    # Penalty for late appearances
    late_appearance_threshold: 0.75  # If > 75% into video
    late_appearance_penalty: 0.7     # Multiply score by 0.7
  
  reid:
    temporal_overlap_frames: 200     # Max frame gap for candidate pairs
    min_feature_similarity: 0.6      # Min cosine similarity
```

## Output Files

### tracklet_stats.npz
- Used by Stage 3b for canonical grouping
- Enables deterministic ranking (always same top-N persons)

### reid_candidates.json
- Used by Stage 3d (if enabled) for visual ReID matching
- If Stage 3d disabled: ignored

## Design Rationale

### Why Separate Analysis Stage?
1. **Decoupling**: Ranking logic independent of detection/tracking
2. **Reproducibility**: Same video = same statistics = same ranking
3. **Tuning**: Users can adjust weights without re-running detection

### Why Rank by Duration + Coverage?
- Users want to see people who are significantly present
- Quick cameos (1-2 frames) are less interesting
- Geometric coverage (large bbox × long duration) = more data

### Why OSNet Features?
- Lightweight appearance model (11M params)
- Fast inference: ~0.1ms per crop
- Pre-trained on person ReID dataset (good generalization)
- Enables Stage 3d to merge visually similar tracklets

## Performance Notes

- **Time**: <0.3s (can process 1000+ tracklets easily)
- **Memory**: Minimal (just statistics arrays)
- **Bottleneck**: Feature extraction for ReID candidates (skipped in fast mode)

---

**Related**: [Back to Master](README_MASTER.md) | [← Stage 2](STAGE2_TRACKING.md) | [Stage 3b →](STAGE3B_GROUPING.md)
