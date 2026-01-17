# Stage 3d: Visual Refinement (OSNet ReID)

**Implementation**: `stage3d_refine_visual.py`

## Purpose
Optional visual ReID matching to merge canonical **persons** representing same individual but missed by Stage 3a/3b/3c grouping heuristics. Reduces 8-10 persons to 7-8 by identifying visual similarity.

## Inputs
- `canonical_persons_3c.npz`: 8-10 persons from Stage 3c (filtered persons)
- `final_crops_3c.pkl`: 60 crops per person (480-600 total crops)

## Outputs
- `canonical_persons_3d.npz`: Merged persons (8-10 → 7-8 if merges found, or identical to input if no merges)
- `final_crops_3d.pkl`: Merged crops (7-8 persons × 60 crops each)
- `merging_report.json`: Details of person chains detected and merged
- `stage3d_sidecar.json`: Debug info (similarity matrix, merge groups)

## When It Runs
Only if `enable_stage3d: true` in pipeline config (disabled in fast mode).

## Processing Flow

```
canonical_persons_3c.npz + final_crops_3c.pkl (8-10 persons from Stage 3c)
    ↓
For each person:
    ├─→ Extract sample crops (top 16 by quality)
    ├─→ OSNet forward pass (get 512-dim features)
    └─→ Average features → per-person embedding
    ↓
For each pair of non-overlapping temporal persons:
    ├─→ Compute cosine similarity
    ├─→ If similarity > 0.60 (threshold):
    │   └─→ Mark as same person (potential merge)
    └─→ Otherwise: keep separate
    ↓
Build connected components (Union-Find)
    ↓
Merge person records + crops for each component
    ↓
canonical_persons_3d.npz (7-8 persons) + final_crops_3d.pkl + merging_report.json
```

## OSNet Feature Extraction

### What is OSNet?
- **Lightweight ReID model** (11M parameters)
- **Input**: Crop images (any size, typically 100-200px)
- **Output**: 512-dimensional feature vector
- **Pre-trained**: ImageNet + person ReID datasets (MARKET1501, DukeMTMC)

### Feature Extraction
```python
def extract_osnet_features(crops_batch):
    """
    Input: List of crop images (np.ndarray)
    Output: (N, 512) features
    """
    # Normalize crops to [0,1]
    crops_normalized = crops_batch / 255.0
    
    # Batch process through OSNet
    with torch.no_grad():
        features = osnet_model(torch.tensor(crops_normalized))
    
    return features  # (batch_size, 512)
```

### Cosine Distance Matching
```python
def compute_feature_distance(feat_a, feat_b):
    """Cosine distance between features"""
    feat_a_norm = feat_a / (np.linalg.norm(feat_a) + 1e-8)
    feat_b_norm = feat_b / (np.linalg.norm(feat_b) + 1e-8)
    
    return 1 - np.dot(feat_a_norm, feat_b_norm)
    # 0 = identical, 1 = opposite
```

## Merging Strategy

### Threshold Selection
```
similarity > 0.60  → MERGE (same person, high confidence)
0.50 ≤ similarity ≤ 0.60  → Uncertain (no merge)
similarity < 0.50  → Separate (different people)
```

### Why 0.60 Threshold?
Empirically tuned to balance precision/recall:
- **0.70**: Too strict (misses valid merges)
- **0.60**: ✅ **SELECTED** (good precision/recall balance)
- **0.50**: Too permissive (false positives increase)

Note: Uses **cosine similarity** (higher = more similar), NOT distance (like older Stage 3d versions).

### Averaging Strategy
When merging person_a + person_b:

```python
def merge_persons(person_a, person_b):
    """
    Merge two canonical persons identified as same individual.
    Combines all their detections and re-sorts chronologically.
    """
    merged = {
        'person_id': person_a['person_id'],
        'tracklet_ids': person_a['tracklet_ids'] + person_b['tracklet_ids'],
        'frame_numbers': np.concatenate([
            person_a['frame_numbers'],
            person_b['frame_numbers']
        ]),
        'bboxes': np.concatenate([
            person_a['bboxes'],
            person_b['bboxes']
        ]),
        'confidences': np.concatenate([
            person_a['confidences'],
            person_b['confidences']
        ]),
        'detection_indices': np.concatenate([
            person_a['detection_indices'],
            person_b['detection_indices']
        ])
    }
    
    # Sort by frame number for continuity
    sort_idx = np.argsort(merged['frame_numbers'])
    merged['frame_numbers'] = merged['frame_numbers'][sort_idx]
    merged['bboxes'] = merged['bboxes'][sort_idx]
    merged['confidences'] = merged['confidences'][sort_idx]
    merged['detection_indices'] = merged['detection_indices'][sort_idx]
    
    return merged
```

## Pairwise Matching

### Candidate Generation
Only compare persons that don't overlap temporally (avoid merging concurrent persons):

```python
# Find pairs to evaluate
candidates = []
for i, person_a in enumerate(canonical_persons):
    for j in range(i+1, len(canonical_persons)):
        person_b = canonical_persons[j]
        
        # Temporal: are they non-overlapping?
        last_frame_a = person_a['frame_numbers'][-1]
        first_frame_b = person_b['frame_numbers'][0]
        gap = first_frame_b - last_frame_a
        
        if gap > temporal_gap_max:  # Too far apart (default 15 frames)
            continue
        
        if gap < -temporal_overlap_tolerance:  # Too much overlap (default 15 frames)
            continue
        
        candidates.append((i, j))
```

### Feature-Based Matching
For each candidate pair, extract OSNet features and compare:

```python
for person_id_a, person_id_b in candidates:
    # Extract top 16 crops by quality for each person
    crops_a = get_top_crops(person_a, n=16)
    crops_b = get_top_crops(person_b, n=16)
    
    # Extract OSNet features
    features_a = osnet(crops_a)  # (16, 512)
    features_b = osnet(crops_b)  # (16, 512)
    
    # Average features across samples
    feat_a_avg = np.mean(features_a, axis=0)  # (512,)
    feat_b_avg = np.mean(features_b, axis=0)  # (512,)
    
    # Normalize and compute cosine similarity
    feat_a_norm = feat_a_avg / np.linalg.norm(feat_a_avg)
    feat_b_norm = feat_b_avg / np.linalg.norm(feat_b_avg)
    similarity = np.dot(feat_a_norm, feat_b_norm)
    
    if similarity > 0.60:
        mark_for_merge(person_a, person_b)
```

## Configuration

```yaml
stage3d_refine:
  enabled: true  # Set to false for fast mode
  
  osnet:
    model_path: ${models_dir}/reid/osnet_x0_25_msmt17.onnx  # Recommended model
    device: cuda
    num_crops_per_person: 16               # Use top 16 crops for feature extraction
  
  merging:
    temporal_gap_max: 15                   # Max frames between persons (gap > this = separate persons)
    temporal_overlap_tolerance: 15         # Allow up to 15 frames of overlap
    similarity_threshold: 0.60             # Cosine similarity threshold (0-1, higher = stricter)
```

## Design Rationale

### Why Optional (Not Always Enabled)?
1. **Adds 10-15s to pipeline** (OSNet feature extraction)
2. **Risk of false positives** if threshold too permissive
3. **Fast mode users** don't need perfect identity resolution
4. **Trade-off**: Speed vs accuracy (let users choose)

### Why OSNet Instead of DeepSORT?
- **OSNet**: Lightweight, easy to integrate, modern architecture
- **DeepSORT**: Requires pre-computed features, adds complexity

### Why Not Just Use Spatial+Temporal Cues?
- **Limitation**: Stage 3a/3b/3c already use spatial/temporal + motion/size heuristics
- **Limitation**: Heuristics can fail when person exits and re-enters differently
- **OSNet**: Adds visual appearance information independent of geometry/motion

## Performance Notes

- **Time**: 8-12s (dependent on # candidate pairs)
- **Memory**: Peak 500 MB (feature extraction batches)
- **FPS**: ~1 crop/ms through OSNet (GPU-limited)
- **Accuracy**: ~85% precision (some false merges possible)

## Output Format

```python
canonical_persons_3d = [
    {
        'person_id': 0,
        'tracklet_ids': [3, 7],  # Merged from 2 tracklets
        'frame_numbers': np.array([...]),
        'bboxes': np.array([...]),
        'confidences': np.array([...]),
        'detection_indices': np.array([...])
    },
    ...  # 7-8 persons (fewer than input if merges found)
]

final_crops_3d = {
    'crops_with_quality': [
        {
            'person_id': 0,
            'crops': [
                {
                    'crop': np.ndarray,
                    'frame_number': 45,
                    'detection_idx': 42,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': 0.92
                },
                ...  # 60 crops
            ]
        },
        ...  # 7-8 persons
    ]
}
```

## When to Use Stage 3d

✅ **Enable if**:
- You need maximum identity accuracy
- Time budget allows 10-15s extra
- Video has significant person occlusions/exits/re-entries

❌ **Disable if**:
- Speed is critical (<60s budget)
- Identities don't matter (e.g., crowd analysis)
- False positive merges are costly

## Limitations

1. **Appearance variation**: Won't merge people with very different poses
2. **Lighting changes**: May fail if person dramatically changes appearance
3. **False positives**: May incorrectly merge similar-looking people
4. **Threshold tuning**: 0.35 is empirical, may not work for all datasets

---

**Related**: [Back to Master](README_MASTER.md) | [← Stage 3c](STAGE3C_FILTER_PERSONS.md) | [Stage 4 →](STAGE4_HTML_GENERATION.md)
