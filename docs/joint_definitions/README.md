# Joint Definition Files

This directory contains JSON files defining the joint formats used in the unified pose estimation pipeline.

## Available Formats

### 2D Keypoint Formats

| File | Format | Joints | Usage |
|------|--------|--------|-------|
| `coco17_2d.json` | COCO-17 | 17 | RTMPose, ViTPose (2D detection) |
| `halpe26_2d.json` | Halpe-26 | 26 | Halpe (COCO-17 + feet) |
| `dwpose133_2d.json` | DWPose-133 | 133 | DWPose wholebody (body + hands + face) |

### 3D Keypoint Formats

| File | Format | Joints | Usage |
|------|--------|--------|-------|
| `h36m17_3d_magf.json` | H36M-17-MAGF | 17 | MotionAGFormer 3D lifting |
| `dwpose133_3d.json` | Wholebody3D-133 | 133 | Wholebody3D end-to-end 3D estimation |

## File Structure

Each JSON file contains:

```json
{
  "format_name": "Format identifier",
  "description": "Human-readable description",
  "num_joints": 17,
  "joint_names": ["joint_0", "joint_1", ...],
  "skeleton_connections": [[0, 1], [1, 2], ...],
  "left_right_pairs": [[1, 2], [3, 4], ...],
  "color_scheme": {
    "left_joints": [1, 3, 5, ...],
    "right_joints": [2, 4, 6, ...],
    "center_joints": [0],
    "left_color": [0, 0, 255],
    "right_color": [255, 0, 0],
    "center_color": [0, 255, 0]
  },
  "coordinate_system": {
    "type": "coordinate_space_type",
    "origin": "origin_description",
    "x_axis": "x_axis_description",
    "y_axis": "y_axis_description",
    "z_axis": "z_axis_description (3D only)",
    "units": "units_description"
  },
  "usage": "Pipeline stage/model name"
}
```

## Usage in Python

```python
import json
from pathlib import Path

# Load joint definition
with open('joint_definitions/coco17_2d.json', 'r') as f:
    coco17 = json.load(f)

# Access joint information
num_joints = coco17['num_joints']
joint_names = coco17['joint_names']
skeleton = coco17['skeleton_connections']

# Get joint name by index
joint_idx = 5
joint_name = coco17['joint_names'][joint_idx]  # "left_shoulder"

# Get color for visualization
left_joints = coco17['color_scheme']['left_joints']
left_color = coco17['color_scheme']['left_color']  # [0, 0, 255] (blue)
```

## Coordinate Systems

### 2D Formats (COCO-17, Halpe-26, DWPose-133)
- **Type**: Image pixels
- **Origin**: Top-left corner of image
- **X-axis**: Left to right (positive = right)
- **Y-axis**: Top to bottom (positive = down)
- **Units**: Pixels

### 3D MotionAGFormer (H36M-17-MAGF)
- **Type**: Body-relative
- **Origin**: Pelvis (joint 0) at [0, 0, 0]
- **X-axis**: Right to left (positive = left)
- **Y-axis**: Down to up (positive = up)
- **Z-axis**: Back to front (positive = forward)
- **Units**: Normalized (typically -1 to 1 range)

### 3D Wholebody3D (Wholebody3D-133)
- **Type**: Camera space hybrid
- **Origin**: Camera optical center
- **X-axis**: Left to right in image plane (positive = right)
- **Y-axis**: Top to bottom in image plane (positive = down)
- **Z-axis**: Camera to scene (positive = away from camera)
- **Units**: Mixed (X/Y in pixels, Z in millimeters or relative depth)
- **Note**: X/Y coordinates match 2D pixel coordinates for overlay

## Format Conversions

### COCO-17 → H36M-17
The pipeline automatically converts COCO-17 2D keypoints to H36M-17 format using:
- `udp_3d_lifting.py` - Contains `coco_h36m()` conversion function
- Maps COCO joint indices to H36M semantic positions
- Computes synthetic joints (pelvis, thorax, spine, head)

### Joint Mapping: COCO-17 → H36M-17

| COCO Index | COCO Name | → | H36M Index | H36M Name |
|------------|-----------|---|------------|-----------|
| 0 | nose | → | (computed) | chin |
| 1-4 | eyes, ears | → | (computed) | head_top |
| 5 | left_shoulder | → | 11 | left_shoulder |
| 6 | right_shoulder | → | 14 | right_shoulder |
| 7 | left_elbow | → | 12 | left_elbow |
| 8 | right_elbow | → | 15 | right_elbow |
| 9 | left_wrist | → | 13 | left_wrist |
| 10 | right_wrist | → | 16 | right_wrist |
| 11 | left_hip | → | 4 | left_hip |
| 12 | right_hip | → | 1 | right_hip |
| 13 | left_knee | → | 5 | left_knee |
| 14 | right_knee | → | 2 | right_knee |
| 15 | left_ankle | → | 6 | left_ankle |
| 16 | right_ankle | → | 3 | right_ankle |
| (computed) | hip center | → | 0 | pelvis_root |
| (computed) | shoulder avg | → | 8 | spine_top_neck |
| (computed) | mid torso | → | 7 | spine_mid |

## Important Notes

### ⚠️ Critical Semantic Differences

1. **H36M Joint 0 (Pelvis)**
   - **Not** "Hip" - it's the root/center of the pelvis
   - Set to origin [0, 0, 0] by MotionAGFormer
   - All other joints are relative to this point

2. **H36M Joint 8 (Spine Top / Thorax)**
   - This is the **vertex** for shoulder angle calculations
   - Shoulder angles: `(8, 11, 12)` and `(8, 14, 15)` - NOT using joint 0!

3. **H36M Joint 9 (Chin / Neck-Nose)**
   - Not just "nose" - represents neck/chin area
   - Connects thorax to head (8 → 9 → 10)

4. **Skeleton Connections**
   - H36M has **16 connections** (not 15!)
   - Missing connections (8→9→10) in old code caused visualization errors

## Version History

- **2024-12-24**: Initial creation with corrected H36M-17 semantics
- Joint definitions extracted from working pipeline code
- Verified against MotionAGFormer `demo/vis.py` reference implementation

## References

- COCO: [https://cocodataset.org/#keypoints-2017](https://cocodataset.org/#keypoints-2017)
- Human3.6M: [http://vision.imar.ro/human3.6m/](http://vision.imar.ro/human3.6m/)
- Halpe: [https://github.com/Fang-Haoshu/Halpe-FullBody](https://github.com/Fang-Haoshu/Halpe-FullBody)
- DWPose: [https://github.com/IDEA-Research/DWPose](https://github.com/IDEA-Research/DWPose)
- MotionAGFormer: [https://github.com/TaatiTeam/MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer)
