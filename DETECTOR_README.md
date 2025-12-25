# Detector System

## Overview

The detector system (`run_detector.py`) is a modular detection and tracking component for the unified pose estimation pipeline. It can be used standalone or integrated into the full pipeline.

## Features

- **Detection**: YOLOv8 (currently supported), RTMDet (planned)
- **Tracking**: BoT-SORT, DeepOCSORT, ByteTrack, StrongSORT, OCSORT (planned)
- **ReID**: OSNet-based re-identification (planned)
- **Output Format**: Compatible with `udp_video.py` Stage 1 (`detections.npz`)

## Quick Start

### 1. Configure Detection

Edit `configs/detector.yaml`:

```yaml
detector:
  type: yolo
  model_path: models/yolo/yolov8s.pt
  confidence: 0.3
  device: cuda

input:
  video_path: demo_data/videos/dance.mp4
  max_frames: 300

output:
  detections_file: demo_data/outputs/detections.npz
```

### 2. Run Detection

```bash
python run_detector.py --config configs/detector.yaml
```

### 3. Validate Output

```bash
python test_detector.py
```

## Configuration Options

### Detector Settings

| Parameter | Options | Description |
|-----------|---------|-------------|
| `type` | `yolo`, `rtmdet` | Detector type |
| `model_path` | Path string | Path to detector model |
| `confidence` | 0.0 - 1.0 | Confidence threshold |
| `device` | `cuda`, `cpu` | Device to use |
| `detect_only_humans` | `true`, `false` | Filter for person class only |

### Tracking Settings

| Parameter | Options | Description |
|-----------|---------|-------------|
| `enabled` | `true`, `false` | Enable tracking (default: `false`) |
| `tracker` | `botsort`, `deepocsort`, `bytetrack`, `strongsort`, `ocsort` | Tracker type |
| `largest_bbox_only` | `true`, `false` | Select only largest bbox (when tracking disabled) |

### ReID Settings

| Parameter | Options | Description |
|-----------|---------|-------------|
| `apply` | `true`, `false` | Enable ReID |
| `model_path` | Path string | Path to ReID model |

### Input/Output

| Parameter | Description |
|-----------|-------------|
| `video_path` | Path to input video |
| `max_frames` | Max frames to process (0 = all) |
| `detections_file` | Output NPZ file path |
| `save_visualization` | Save annotated video |
| `visualization_path` | Visualization output path |

## Output Format

The detector outputs a standard NPZ file with the following keys:

```python
{
    'frame_numbers': np.array(shape=(N,), dtype=int64),
    'bboxes': np.array(shape=(N, 4), dtype=int64)  # [x1, y1, x2, y2]
}
```

This format is **identical** to `udp_video.py` Stage 1 output and can be used interchangeably.

**Key Details:**
- `frame_numbers`: Sequential frame indices (0, 1, 2, ...)
- `bboxes`: Bounding box coordinates in `[x1, y1, x2, y2]` format
  - **Integer coordinates** (int64 dtype)
  - Empty detections stored as `[0, 0, 0, 0]`
  - Valid detections have `x2 > 0`

## Usage Modes

### Mode 1: Detection Only (Default)

**Purpose:** Single person tracking with largest bbox selection

**Configuration:**
```yaml
tracking:
  enabled: false
  largest_bbox_only: true
```

**Behavior:**
- Detects all persons in frame
- Selects largest bbox by area
- Outputs one detection per frame

**Use case:** Single-person pose estimation (e.g., dance videos)

### Mode 2: Multi-Object Tracking (Planned)

**Purpose:** Track multiple persons across frames

**Configuration:**
```yaml
tracking:
  enabled: true
  tracker: botsort
```

**Behavior:**
- Detects all persons
- Assigns unique IDs
- Maintains IDs across frames
- Outputs all tracked persons

**Use case:** Multi-person scenarios with occlusions

## Integration with Pipeline

### Standalone Usage

```bash
# Step 1: Detection
python run_detector.py --config configs/detector.yaml

# Step 2: 2D Pose Estimation (using detections.npz)
python udp_video.py --config configs/udp_video.yaml

# Step 3: 3D Lifting
python udp_3d_lifting.py --keypoints demo_data/outputs/kps_2d_rtm.npz \
    --video demo_data/videos/dance.mp4 \
    --output demo_data/outputs/kps_3d_magf.npz
```

### Integrated Usage

The detector can also be called directly from `udp_video.py` as Stage 1.

## Model Downloads

### YOLOv8 Models

```bash
# YOLOv8s (recommended for speed/accuracy balance)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt \
    -P models/yolo/

# YOLOv8m (higher accuracy)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt \
    -P models/yolo/

# YOLOv8n (fastest)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt \
    -P models/yolo/
```

## Troubleshooting

### Issue: No detections found

**Solutions:**
1. Lower confidence threshold: `confidence: 0.2`
2. Check video path is correct
3. Verify model is downloaded: `ls models/yolo/`

### Issue: CUDA out of memory

**Solutions:**
1. Use CPU: `device: cpu`
2. Use smaller model: `yolov8n.pt` instead of `yolov8s.pt`
3. Process fewer frames: `max_frames: 100`

### Issue: Wrong person detected

**Solutions:**
1. Enable tracking (when implemented)
2. Manually crop video to focus on target person
3. Adjust confidence threshold

## Future Enhancements

- [ ] Multi-object tracking implementation
- [ ] ReID integration
- [ ] RTMDet detector support
- [ ] Visualization output (annotated video)
- [ ] Track filtering by size/position
- [ ] Export to other formats (COCO JSON, MOT format)

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [BoT-SORT Paper](https://arxiv.org/abs/2206.14651)
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)
