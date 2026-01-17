# Stage 3b: Canonical Person Grouping

**Implementation**: `stage3b_group_canonical.py`

## Purpose
Merge tracklets into canonical persons using 5 heuristic checks (temporal, spatial, area, motion direction, jitter). Converts 67 tracklets → ~40+ persons (further filtered in Stage 3c).

## Inputs
- `tracklets_raw.npz`: 67 tracklets from ByteTrack
- `tracklet_stats.npz`: Pre-computed statistics from Stage 3a

## Outputs
- `canonical_persons.npz`: ~40+ canonical persons with merged tracklets (input for Stage 3c)
- `grouping_log.json`: Debug log with merge decisions and criteria

## Processing Flow

```
tracklets_raw.npz (67 tracklets)
    ↓
Sort by rank_score (descending)
    ↓
For each tracklet (in rank order):
    ├─→ Check if already merged into existing person
    ├─→ If not, start NEW person with this tracklet
    ├─→ Look for overlapping tracklets to merge:
    │   ├─→ Temporal overlap: same time window
    │   ├─→ Geometric separation: different bbox regions
    │   └─→ Track continuity: can explain gap via Kalman
    ├─→ Merge into single person representation
    └─→ Combine all detections + indices
    ↓
canonical_persons_3c.npz (8-10 persons)
```

## Performance

| Metric | Value |
|--------|-------|
| Time | 0.47s |
| Input tracklets | 67 |
| Output persons | 8-10 |
| Merge operations | ~15-20 |

## Grouping Algorithm

### Temporal Overlap Detection
```python
def has_temporal_overlap(tracklet_a, tracklet_b, gap_threshold=100):
    """
    Check if tracklets represent same person
    - Same time window OR
    - Small gap (≤ 100 frames) explainable by missed detections
    """
    gap = tracklet_b.start_frame - tracklet_a.end_frame
    return gap <= gap_threshold
```

**Rationale**: ByteTrack may lose track during occlusion/blur. If someone disappears for 100 frames then reappears in same region → likely same person.

### Geometric Separation Check
```python
def is_geometrically_separate(tracklet_a, tracklet_b):
    """
    Check if tracklets are in different spatial regions
    → Can't be same person if always in different parts of frame
    """
    bbox_a_centroid = compute_centroid(tracklet_a)
    bbox_b_centroid = compute_centroid(tracklet_b)
    
    distance = euclidean(bbox_a_centroid, bbox_b_centroid)
    max_distance = frame_width / 3  # Different thirds of frame
    
    return distance > max_distance
```

## Merging Strategy

### Deterministic Ranking
1. **Select top tracklets by rank_score** (Stage 3a)
2. **For each top tracklet**: Try to merge with existing person
3. **Merge criteria**:
   - Overlapping time windows OR small temporal gap
   - Consistent bbox position (not jumping across frame)
   - Detection confidence > 0.3 (already filtered)

### Why Not Just Use All Tracklets?
- 67 tracklets too many for user to review
- Many are duplicates or false positives
- Top 8-10 capture ~85% of meaningful detections

## Canonical Person Data Structure

```python
canonical_persons = [
    {
        'person_id': 0,
        'tracklet_ids': [0, 5, 12],           # Merged tracklets
        'frame_numbers': np.array([0,1,2,...,350]),  # All frames
        'bboxes': np.array([[x1,y1,x2,y2],...]),    # All bboxes
        'confidences': np.array([0.92,...]),         # All scores
        'detection_indices': np.array([...])  # ← Critical! Links to crops
    },
    ...  # 8-10 persons
]
```

## Configuration

```yaml
stage3b_group:
  grouping:
    top_tracklets: 10              # Max persons to output
    temporal_gap_threshold: 100    # Frames to wait before losing track
    geometric_distance_ratio: 0.33 # Fraction of frame width (geometric separation)
    min_tracklet_confidence: 0.3   # Filter threshold
```

## Detection Index Linkage

**Critical Step**: Preserve `detection_indices` when merging tracklets.

When merging tracklet A + tracklet B into person X:
```python
person['detection_indices'] = np.concatenate([
    tracklet_a['detection_indices'],  # Links to crops A
    tracklet_b['detection_indices']   # Links to crops B
])
```

This enables **Stage 3c to do O(1) crop lookup** without re-reading video!

## Design Rationale

### Why Geometric Separation Check?
Prevents merging two people who happen to be tracked in sequence:
```
Frame 0-100: Person A on left side
Frame 101-200: Person B on right side
→ Should NOT merge (different people)
```

Without this check, could incorrectly group unrelated people.

### Why Temporal Gap Tolerance?
Cameras can momentarily lose track during:
- Quick occlusions (1-2 seconds)
- Motion blur
- Lighting changes

100-frame gap (4 seconds @ 25fps) is reasonable window.

### Why Top-10 Instead of All?
- Dataset: 67 tracklets from single 81-second video
- Most are fragments or false positives
- Top-10 captures:
  - Main subjects (people who appear for >100 frames)
  - Significant secondary people (>40 frames)
  - Excludes noise and brief cameos

## Performance Notes

- **Time**: <0.5s
- **Memory**: Merging is cheap (just array concatenation)
- **Determinism**: Same video → same grouping (no randomness)

---

**Related**: [Back to Master](README_MASTER.md) | [← Stage 3a](STAGE3A_ANALYSIS.md) | [Stage 3c →](STAGE3C_FILTER_PERSONS.md)
