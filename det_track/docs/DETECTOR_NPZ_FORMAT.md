# Detector NPZ Output Format

Complete specification of the detection output format used throughout the unified pose estimation pipeline.

---

## NPZ Structure

The detector outputs a compressed NumPy archive (`.npz` file) with the following keys:

```python
np.savez_compressed(
    output_path,
    frame_numbers=np.array([...], dtype=np.int64),  # Shape: (N,)
    bboxes=np.array([...], dtype=np.int64)          # Shape: (N, 4)
)
```

---

## Data Format Specification

| Key | Type | Shape | Contents | Notes |
|-----|------|-------|----------|-------|
| `frame_numbers` | int64 | (N,) | Frame index for each detection | 0-based frame indices |
| `bboxes` | int64 | (N, 4) | `[x1, y1, x2, y2]` | Top-left and bottom-right corners in pixel coordinates |

### Key Details

#### `frame_numbers` (int64, shape N)
- Array of frame indices where detections exist
- 0-based indexing
- One entry per frame processed
- Example: `[0, 1, 2, 3, 4, ..., 2847]`

#### `bboxes` (int64, shape N × 4)
- Bounding box coordinates in format `[x1, y1, x2, y2]`
- Integer pixel coordinates (not float)
- Coordinate system:
  - `x1, y1`: Top-left corner
  - `x2, y2`: Bottom-right corner
- Empty detection (no person found): `[0, 0, 0, 0]`

---

## Detection Logic

### Single Detection Per Frame

The detector selects **one detection per frame** (the largest bounding box):

```python
# From run_detector.py lines 206-212
frame_numbers.append(frame_idx)
if len(detections) > 0:
    largest_bbox = select_largest_bbox(detections)  # Takes largest bbox
    bboxes_list.append(largest_bbox)                # [x1, y1, x2, y2]
else:
    bboxes_list.append([0, 0, 0, 0])                # Empty bbox if no detection
```

### Empty Detection Handling

When no person is detected in a frame:
- Frame index is still recorded in `frame_numbers`
- Corresponding bbox is `[0, 0, 0, 0]`
- Validation: rest of pipeline checks `if bbox[2] > 0` to detect invalid entries

---

## Example NPZ Contents

When loaded in Python:

```python
import numpy as np

data = np.load('detections.npz')

# Frame numbers (2848 frames total)
print(data['frame_numbers'])
# Output: array([0, 1, 2, 3, 4, ..., 2847], dtype=int64)

# Bounding boxes (2848 detections)
print(data['bboxes'])
# Output:
# array([
#     [150, 200, 450, 850],  # Frame 0: person detected at [150,200] to [450,850]
#     [152, 198, 455, 860],  # Frame 1: person detected
#     [0, 0, 0, 0],          # Frame 2: no detection
#     [155, 205, 460, 875],  # Frame 3: person detected
#     ...
# ], dtype=int64)
# Shape: (2848, 4)
```

---

## Validation in Downstream Code

The rest of the pipeline validates detections using `x2 > 0`:

```python
# From run_posedet.py
for frame_idx, bbox in zip(frame_numbers, bboxes):
    if bbox[2] > 0:  # Valid detection (x2 > 0)
        keypoints, scores = pose_model(frame, bboxes=[bbox])
    else:
        # No valid detection, store empty keypoints
        all_keypoints.append(np.zeros((17, 2)))
        all_scores.append(np.zeros(17))
```

---

## Related Variations

### Stage 1 in Detection & Tracking Pipeline

The integrated pipeline (`det_track/run_pipeline.py`) Stage 1 produces a similar but extended format:

```python
# Stage 1 output includes additional field
np.savez(
    detections_file,
    frame_numbers=frame_numbers,      # (N,) int64
    bboxes=bboxes,                    # (N, 4) int64
    confidences=confidences           # (N,) float32 - detection confidence scores
)
```

**Key difference:** Stage 1 includes confidence scores, but `run_posedet.py` only requires `frame_numbers` and `bboxes`.

---

## File Location

### Default Output Paths

**Standalone detector:**
```
demo_data/outputs/detections.npz
```

**Pipeline Stage 1:**
```
demo_data/outputs/{video_name}/detections_raw.npz
```

Example (for video `kohli_nets.mp4`):
```
demo_data/outputs/kohli_nets/detections_raw.npz
```

---

## Usage Examples

### Loading Detections

```python
import numpy as np

# Load compressed NPZ
data = np.load('detections.npz')
frame_numbers = data['frame_numbers']
bboxes = data['bboxes']

print(f"Total frames: {len(frame_numbers)}")
print(f"Bboxes shape: {bboxes.shape}")
```

### Iterating Through Detections

```python
for frame_idx, bbox in zip(frame_numbers, bboxes):
    x1, y1, x2, y2 = bbox
    
    # Validate detection
    if x2 > 0:  # Valid bbox
        print(f"Frame {frame_idx}: Person at ({x1}, {y1}) - ({x2}, {y2})")
    else:
        print(f"Frame {frame_idx}: No detection")
```

### Computing Statistics

```python
# Count valid detections
valid_detections = np.sum(bboxes[:, 2] > 0)
detection_rate = valid_detections / len(frame_numbers) * 100

print(f"Frames with detections: {valid_detections}/{len(frame_numbers)}")
print(f"Detection rate: {detection_rate:.1f}%")

# Bbox dimensions
widths = bboxes[:, 2] - bboxes[:, 0]
heights = bboxes[:, 3] - bboxes[:, 1]
print(f"Average person width: {widths[widths > 0].mean():.0f}px")
print(f"Average person height: {heights[heights > 0].mean():.0f}px")
```

---

## Pipeline Integration

### Stage Flow

```
Video File
    ↓
[run_detector.py]
    ↓
detections.npz (frame_numbers, bboxes)
    ↓
[run_posedet.py] or [Stage 1: run_pipeline.py]
    ↓
[2D Pose Estimation with bbox-guided cropping]
    ↓
keypoints.npz (kps_2d_rtm.npz, kps_2d_vit.npz, etc.)
```

### Consumer Scripts

Scripts that read this format:

- **`run_posedet.py`**: 2D pose estimation using saved detections
- **`det_track/stage1_detect.py`**: Detection stage of full pipeline
- **`det_track/stage2_track.py`**: Uses detections for tracking
- **`run_posedet.py`**: Uses detections for pose inference
- **`stage12_keyperson_selector.py`**: Uses tracking results derived from detections

---

## Backward Compatibility

### With `udp_video.py`

The old unified script `udp_video.py` produces detections in the same NPZ format internally, ensuring compatibility with both the new modular pipeline and the legacy all-in-one approach.

### Data Consistency

All pipelines maintain:
- Consistent bbox format: `[x1, y1, x2, y2]`
- Consistent dtype: int64 for coordinates
- Consistent empty detection value: `[0, 0, 0, 0]`

---

## Common Issues

### Issue: Pipeline fails to read detections

**Cause:** Missing NPZ file or wrong path
- Check that `detections_file` in config points to correct location
- Verify that `run_detector.py` or Stage 1 has completed successfully

### Issue: All bboxes are zeros

**Cause:** No person detections found in video
- Verify video contains visible people
- Check YOLO confidence threshold is not too high
- Try lowering `confidence` parameter in config

### Issue: Pose estimation produces all zeros

**Cause:** Invalid bboxes (all are `[0, 0, 0, 0]`)
- Ensure detections were computed correctly
- Check detection file was saved properly
- Run detector again with `--verbose` flag

---

## Performance Notes

- **File size:** ~10-50 KB for typical video (depending on compression)
- **Load time:** <100 ms for any video length (due to lazy loading)
- **Memory:** Minimal (arrays loaded on demand)

---

## Related Documentation

- [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md) - Full pipeline architecture
- [run_detector.py](../run_detector.py) - Detection script source
- [run_posedet.py](../run_posedet.py) - Pose estimation with detections
- [pipeline_config.yaml](configs/pipeline_config.yaml) - Pipeline configuration
