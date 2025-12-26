# NPZ File Naming Convention Update

## Summary

Updated all Stage 2 pose estimation save functions to use model-specific naming and include joint format metadata.

## Changes Made

### 1. File Naming Convention

**Old Naming:**
- All models saved to generic `keypoints_2d.npz` or `keypoints_3d.npz`
- Impossible to identify which model generated the file

**New Naming:**
- `kps_2d_rtm.npz` - RTMPose COCO-17 (2D)
- `kps_2d_vit.npz` - ViTPose COCO-17 (2D)
- `kps_2d_rtm_halpe26.npz` - RTMPose Halpe-26 (2D)
- `kps_2d_wb3d.npz` - Wholebody3D DWPose-133 (2D)
- `kps_3d_wb3d.npz` - Wholebody3D DWPose-133 (3D)
- `kps_3d_magf.npz` - MotionAGFormer H36M-17 (3D)

### 2. Metadata Fields Added

All NPZ files now include two new metadata fields:

```python
np.savez_compressed(
    output_path,
    frame_numbers=frame_numbers,
    keypoints=np.array(all_keypoints),
    scores=np.array(all_scores),
    joint_format="coco17_2d.json",  # ← NEW: Reference to joint definition file
    model_type="rtmpose"              # ← NEW: Model identifier
)
```

### 3. Joint Format References

| Model | File Name | joint_format | model_type |
|-------|-----------|--------------|------------|
| RTMPose | `kps_2d_rtm.npz` | `coco17_2d.json` | `rtmpose` |
| ViTPose | `kps_2d_vit.npz` | `coco17_2d.json` | `vitpose` |
| RTMPose Halpe26 | `kps_2d_rtm_halpe26.npz` | `halpe26_2d.json` | `rtmpose_halpe26` |
| Wholebody3D (2D) | `kps_2d_wb3d.npz` | `dwpose133_2d.json` | `wb3d` |
| Wholebody3D (3D) | `kps_3d_wb3d.npz` | `dwpose133_3d.json` | `wb3d` |
| MotionAGFormer | `kps_3d_magf.npz` | `h36m17_3d_magf.json` | `motionagformer` |

## Usage Example

### Loading NPZ with Metadata

```python
import numpy as np
import json
from pathlib import Path

# Load NPZ file
data = np.load('demo_data/outputs/kps_2d_rtm.npz')

# Access keypoints
keypoints = data['keypoints']  # (N, 17, 2)
scores = data['scores']         # (N, 17)

# Access metadata
joint_format = str(data['joint_format'])  # "coco17_2d.json"
model_type = str(data['model_type'])      # "rtmpose"

# Load joint definitions
with open(f'joint_definitions/{joint_format}', 'r') as f:
    joint_def = json.load(f)
    joint_names = joint_def['joint_names']
    skeleton = joint_def['skeleton_connections']

# Now you know exactly:
# - Which model generated this file
# - What the joint semantics are
# - How to visualize the skeleton correctly

print(f"Model: {model_type}")
print(f"Joint format: {joint_format}")
print(f"Joint 5: {joint_names[5]}")  # "left_shoulder"
```

### Automatic Format Detection

```python
def load_keypoints_auto(npz_path):
    """Load keypoints with automatic format detection"""
    data = np.load(npz_path)
    
    # Extract metadata
    joint_format = str(data['joint_format'])
    model_type = str(data['model_type'])
    
    # Load corresponding joint definitions
    joint_def_path = Path('joint_definitions') / joint_format
    with open(joint_def_path) as f:
        joint_def = json.load(f)
    
    return {
        'keypoints': data['keypoints'],
        'scores': data['scores'],
        'frame_numbers': data['frame_numbers'],
        'model_type': model_type,
        'joint_format': joint_format,
        'joint_definition': joint_def
    }

# Usage
data = load_keypoints_auto('demo_data/outputs/kps_2d_rtm.npz')
print(f"Loaded {data['keypoints'].shape[0]} frames from {data['model_type']}")
print(f"Joints: {data['joint_definition']['joint_names']}")
```

## Files Modified

1. **`udp_video.py`** (4 functions updated):
   - `stage2_estimate_poses_rtmpose()` → saves to `kps_2d_rtm.npz`
   - `stage2_estimate_poses_rtmpose_halpe26()` → saves to `kps_2d_rtm_halpe26.npz`
   - `stage2_estimate_poses_vitpose()` → saves to `kps_2d_vit.npz`
   - `stage2_estimate_poses_wb3d()` → saves to `kps_2d_wb3d.npz` and `kps_3d_wb3d.npz`

2. **`udp_3d_lifting.py`** (1 function updated):
   - `main()` → saves to `kps_3d_magf.npz`

## Benefits

1. ✅ **Clear file identification** - Filename tells you which model generated it
2. ✅ **Self-documenting** - NPZ contains reference to joint format
3. ✅ **No confusion** - Can't mix up different model outputs
4. ✅ **Automatic validation** - Can programmatically verify format matches expected
5. ✅ **Version control friendly** - Different model outputs don't overwrite each other

## Migration Notes

### For Existing Code

If you have existing code that loads `keypoints_2d.npz`:

```python
# Old code
data = np.load('demo_data/outputs/keypoints_2d.npz')

# Update to new naming (choose model):
data = np.load('demo_data/outputs/kps_2d_rtm.npz')  # RTMPose
# OR
data = np.load('demo_data/outputs/kps_2d_vit.npz')  # ViTPose
```

### For verify_3dlifting.py

The script should be updated to use the new naming convention:

```python
# Old
--keypoints demo_data/outputs/keypoints_2D_rtm.npz

# New
--keypoints demo_data/outputs/kps_2d_rtm.npz
```

## Related Files

- `joint_definitions/` - Directory containing all joint format definitions
- `joint_definitions/README.md` - Comprehensive documentation of all formats
- All `.json` files in `joint_definitions/` - Machine-readable joint definitions

## Date

December 24, 2025
