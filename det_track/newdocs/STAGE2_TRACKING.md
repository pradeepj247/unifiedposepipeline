# Stage 2: Tracking (ByteTrack Offline)

**Implementation**: `stage2_track.py`

## Purpose
Link detections across frames into consistent tracklets (sequences of same person).

## Inputs
- `detections_raw.npz`: All detections with bboxes

## Outputs
- `tracklets_raw.npz`: 67 tracklets with bboxes linked by ByteTrack
- Timing: Tracking performance metrics

## Processing Flow

```
detections_raw.npz (8832 detections)
    ↓
ByteTrack Initialize
    ├─→ Kalman filter tracker
    └─→ Hungarian assignment matcher
    ↓
For each frame (2025 frames):
    ├─→ Get detections for frame
    ├─→ Kalman predict tracklet positions
    ├─→ Hungarian match detections → tracklets
    ├─→ Create new tracklets if unmatched detections
    ├─→ Remove dead tracklets (30-frame timeout)
    └─→ Store matched tracklet IDs
    ↓
tracklets_raw.npz (67 tracklets)
```

## Performance

| Metric | Value |
|--------|-------|
| Time | 7.91s |
| Frames processed | 2025 |
| FPS | 256 (with overhead), 694.7 (tracking only) |
| Tracklets created | 67 |
| Tracked detections | 8646 / 8832 (98%) |

**ByteTrack alone**: 694.7 FPS ← Highly optimized

## Key Optimizations

### 1. Reused Dummy Frame (100×100)
**Before Optimization**:
```python
for frame_idx in range(num_frames):
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)  # 2025×
    results = tracker.update(frame, detections)
```
Problem: Creating 2025 × 6.2MB arrays = 12.7GB wasted memory allocations

**After Optimization**:
```python
dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)  # Create once

for frame_idx in range(num_frames):
    results = tracker.update(dummy_frame, detections)  # Reuse
```
Result: 192× smaller per allocation, 715 FPS (27% faster)

**Why?** ByteTrack only uses Kalman filters and Hungarian matching—it never reads pixel data from the frame!

### 2. Progress Bar Optimization
**Before**:
```python
pbar = tqdm(total=num_frames, desc="Tracking", mininterval=1.0)
# tqdm adds ~5-10% overhead
```

**After**:
```python
pbar = tqdm(..., disable=not verbose)  # Conditional
# Disabled when verbose=false
```

Result: Slightly cleaner output in normal mode.

## Configuration

```yaml
stage2_track:
  tracker:
    track_buffer: 30        # Frames to keep live tracklets
    track_thresh: 0.5       # Min IoU for assignment
    match_thresh: 0.8       # Association threshold
    frame_rate: 25          # Video fps
    mot20: false            # Motion model complexity
  
  verbose: false  # Disable progress bar
```

## Tracklet Data Structure

### tracklets_raw.npz
```python
{
    'tracklets': [
        {
            'tracklet_id': 0,
            'frame_numbers': np.array([5, 6, 7, 8, ...]),
            'bboxes': np.array([[x1,y1,x2,y2], ...]),
            'confidences': np.array([0.9, 0.88, ...]),
            'detection_indices': np.array([42, 43, 44, 45, ...])  # Links to crops!
        },
        ...  # 67 tracklets
    ]
}
```

## Identity Switches
ByteTrack operates "offline" (all frames available), so identity switches are minimal:
- **ID switches observed**: ~3-5 per video (1.5% of 67 tracklets)
- These are corrected in Stage 3d (OSNet visual refinement)

## Design Rationale

### Why ByteTrack?
1. **Offline mode**: All detections available upfront → globally optimal matching
2. **Kalman + Hungarian**: Simple, fast, interpretable
3. **SOTA for low-data regime**: Works with detections alone (no appearance features yet)

### Why Not DeepSORT?
- DeepSORT requires pre-computed appearance features → added complexity
- ByteTrack achieves similar tracking quality with simpler architecture
- Stage 3d handles appearance refinement separately

## Performance Notes

- **Tracking overhead**: 7.91s for 8832 detections
- **ByteTrack algorithm**: O(n²) for Hungarian matching (n = detections/frame ≈ 4-15)
- **Kalman state**: Each tracklet stores [x, y, w, h, vx, vy] + covariance
- **Memory**: ~50 MB for 67 tracklets

---

**Related**: [Back to Master](README_MASTER.md) | [← Stage 1](STAGE1_DETECTION.md) | [Stage 3a →](STAGE3A_ANALYSIS.md)
