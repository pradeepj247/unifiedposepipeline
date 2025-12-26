# Unified Pose Estimation Pipeline - Project Context

**Last Updated:** December 24, 2025

---

## 1. Core Objective

Build a **unified, configurable pose estimation pipeline** that integrates multiple 2D and 3D pose estimation methods under a consistent API. The pipeline enables easy switching between different backends while maintaining similar function calls and data formats.

### Key Features:
- **Multi-stage architecture**: Detection → 2D Pose → 3D Lifting → Visualization
- **Model flexibility**: Switch between RTMPose, ViTPose, RTMPose-Halpe26, Wholebody
- **Consistent outputs**: Standardized NPZ format across all models
- **Integrated visualization**: Automatic video generation with skeleton overlays

---

## 2. Pipeline Architecture

### Stage 1: Object Detection (YOLOv8)
**Purpose:** Detect persons in video frames and extract bounding boxes

**Input:**
- Video file (e.g., `demo_data/videos/dance.mp4`)

**Output:**
- `demo_data/outputs/detections.npz`
  - Keys: `frame_numbers`, `bboxes`
  - Format: `bboxes` shape `(N_frames, 4)` → `[x1, y1, x2, y2]`

**Script:** `udp_video.py` → `stage1_detect_yolo()`

---

### Stage 2: 2D Pose Estimation

**Purpose:** Estimate 2D keypoints from detected person crops

#### Model Options (Configurable via `configs/udp_video.yaml`):

| Model Name | Keypoints | Output File Pattern | Description |
|------------|-----------|---------------------|-------------|
| `rtmpose` | 17 (COCO) | `kps_2d_rtm.npz` | Standard COCO body joints |
| `rtmpose_h26` | 26 (Halpe26) | `kps_2d_halpe26.npz` | COCO + foot keypoints |
| `vitpose` | 17 (COCO) | `kps_2d_vit.npz` | ViTPose detector |
| `wholebody` | 133 (COCO-WholeBody) | `kps_2d_wholebody.npz` + `kps_3d_wholebody.npz` | Body + face + hands + 3D |

**Input:**
- Video file
- `detections.npz` (from Stage 1)

**Output (2D):**
- NPZ file with keys:
  - `frame_numbers`: `(N_frames,)` array
  - `keypoints`: `(N_frames, N_joints, 2)` array (x, y coordinates)
  - `scores`: `(N_frames, N_joints)` confidence scores
  - `joint_format`: JSON filename describing keypoint semantics
  - `model_type`: String identifier (e.g., `"rtmpose"`, `"wholebody"`)

**Output (3D - Wholebody only):**
- `kps_3d_wholebody.npz`
  - `keypoints_3d`: `(N_frames, 133, 3)` array (x, y, z coordinates)
  - Same metadata structure as 2D

**Script:** `udp_video.py` → `stage2_estimate_poses_*()`

---

### Stage 3: 3D Pose Lifting (MotionAGFormer)

**Purpose:** Lift 2D keypoints to 3D using temporal model

**Model:** MotionAGFormer (MAGF)
- Temporal window: 243 frames
- Uses past/future context for smoothing
- Trained on Human3.6M dataset

**Input:**
- 2D keypoints NPZ (17 joints in H36M format after conversion)
- Video metadata (width, height)

**Output:**
- `kps_3d_magf.npz`
  - `poses_3d`: `(N_frames, 17, 3)` array in H36M-17 format
  - Normalized coordinates (root-centered)

**Script:** `udp_3d_lifting.py` or integrated in `udp_video.py`

**Conversion Pipeline:**
1. COCO-17 → H36M-17 format conversion
2. Screen coordinate normalization
3. Temporal resampling (to 243 frames if needed)
4. Model inference with test-time augmentation (horizontal flip)
5. Camera rotation applied for proper orientation

---

## 3. Joint Format Specifications

### 3.1 COCO-17 Format
**Used by:** `rtmpose`, `vitpose`

**JSON:** `demo_data/joint_formats/coco17_2d.json`

```
0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
```

### 3.2 Halpe26 Format
**Used by:** `rtmpose_h26`

**JSON:** `demo_data/joint_formats/halpe26_2d.json`

COCO-17 + 6 foot keypoints + 3 face keypoints = 26 total

### 3.3 COCO-WholeBody-133 Format
**Used by:** `wholebody`

**JSON:** `demo_data/joint_formats/dwpose133_2d.json`, `dwpose133_3d.json`

```
Joints 0-16:   COCO-17 body (17 keypoints)
Joints 17-22:  Foot (6 keypoints)
Joints 23-90:  Face landmarks (68 keypoints)
Joints 91-111: Left hand (21 keypoints)
Joints 112-132: Right hand (21 keypoints)
Total: 133 keypoints
```

### 3.4 H36M-17 Format
**Used by:** MotionAGFormer 3D output

**JSON:** `demo_data/joint_formats/h36m17_3d.json`

```
0: Hip (pelvis center), 1: RHip, 2: RKnee, 3: RAnkle,
4: LHip, 5: LKnee, 6: LAnkle, 7: Spine, 8: Thorax,
9: Nose, 10: Head, 11: LShoulder, 12: LElbow, 13: LWrist,
14: RShoulder, 15: RElbow, 16: RWrist
```

**Key Differences from COCO:**
- Joint 0 (Hip): Computed as midpoint of left/right hips
- Joint 7 (Spine): Computed between hips and shoulders
- Joint 8 (Thorax): Computed from shoulder center
- Joint 10 (Head): Approximated from nose + upward offset

---

## 4. Visualization System

### 4.1 2D Visualization
**Script:** `udp_video.py` → `stage3_visualize()`

**Logic:**
- Loads video + 2D keypoints NPZ
- Draws skeleton connections based on joint format JSON
- Color coding: Green for left side, blue for right side
- Outputs annotated video: `demo_data/outputs/result_2D.mp4`

**Auto-detection:** Determines COCO-17 vs Halpe26 vs Wholebody based on keypoint count

### 4.2 3D Visualization (Wholebody)
**Script:** `udp_video.py` → `stage3_visualize_3d()`

**Layout:** Side-by-side 2-panel video
- **Left panel:** Original frame + 2D keypoints overlay
- **Right panel:** 3D matplotlib skeleton (rotating view)

**Output:** `demo_data/outputs/result_wb_2D3D.mp4`

**Technical Details:**
- Uses matplotlib 3D projection
- Camera rotation: Quaternion `[0.1407, -0.1500, -0.7552, 0.6223]`
- Root-centered at pelvis (joint 0)
- Normalized to unit scale

### 4.3 Standalone 3D Visualization Tool
**Script:** `vis_wb3d.py`

**Purpose:** Visualize pre-computed 3D keypoints with frame-by-frame inspection

**Usage:**
```bash
python vis_wb3d.py \
    --video demo_data/videos/dance.mp4 \
    --kps2d demo_data/outputs/kps_2d_wholebody.npz \
    --kps3d demo_data/outputs/kps_3d_wholebody.npz \
    --output demo_data/outputs/debug_wb3d.mp4 \
    --max-frames 120
```

**Features:**
- 2-panel layout: 2D skeleton (left) + 3D skeleton (right)
- No PNG debug output (removed)
- Minimal logging
- Fixed keypoint loading: Uses correct keys (`keypoints` for 2D, `keypoints_3d` for 3D)

---

## 5. Configuration System

### Main Config: `configs/udp_video.yaml`

**Structure:**
```yaml
video:
  input_path: demo_data/videos/dance.mp4
  max_frames: 120

detection:
  model_path: models/yolo/yolov8x.pt
  confidence: 0.3
  device: cuda

pose_estimation:
  method: rtmpose  # Options: rtmpose, vitpose, rtmpose_h26, wholebody
  
  rtmpose:
    pose_model_path: models/rtmlib/rtmpose-l-coco-384x288.onnx
    pose_input_size: [288, 384]
    backend: onnxruntime
    device: cuda
  
  rtmpose_h26:
    pose_model_path: models/rtmlib/rtmpose-l-halpe26-384x288.onnx
    pose_input_size: [288, 384]
    backend: onnxruntime
    device: cuda
  
  vitpose:
    model_path: models/vitpose/vitpose-b.pth
    model_name: b
    dataset: coco
    device: cuda
  
  wholebody:
    pose_model_path: models/wb3d/rtmw3d-l.onnx
    pose_input_size: [288, 384]
    backend: onnxruntime
    device: cuda

output:
  stage1_detections: demo_data/outputs/detections.npz
  stage2_keypoints_2d: demo_data/outputs/keypoints_2D.npz
  stage2_keypoints_3d: demo_data/outputs/keypoints_3D.npz  # Only for wholebody
  video_output_2d: demo_data/outputs/result_2D.mp4
  video_output_3d: demo_data/outputs/result_3D.mp4  # Only for wholebody
  plot: true  # Set to false to skip visualization stage
```

---

## 6. Key Scripts and Their Roles

### 6.1 Main Pipeline Scripts

| Script | Purpose | Key Functions |
|--------|---------|---------------|
| `udp_video.py` | Main unified pipeline | `stage1_detect_yolo()`, `stage2_estimate_poses_*()`, `stage3_visualize()` |
| `udp_3d_lifting.py` | Standalone 3D lifting | Converts 2D keypoints → 3D using MAGF |
| `vis_wb3d.py` | Wholebody visualization tool | 2-panel video with 2D+3D skeletons |
| `run_detector.py` | Standalone detector/tracker | Detection with YOLO/RTMDet, optional tracking |

### 6.1.1 Standalone Detector (`run_detector.py`)

**Purpose:** Modular detection and tracking script that can be used independently or as part of the pipeline.

**Configuration:** `configs/detector.yaml`

**Key Features:**
- **Detector Support:** YOLOv8, RTMDet (planned)
- **Tracking Support:** BoT-SORT, DeepOCSORT, ByteTrack, StrongSORT, OCSORT (planned)
- **ReID Support:** OSNet-based re-identification (planned)
- **Modes:**
  - Detection only (default): Selects largest bbox per frame
  - Tracking: Multi-object tracking across frames (future)

**Usage:**
```bash
python run_detector.py --config configs/detector.yaml
```

**Output:** Same format as `udp_video.py` Stage 1 (`detections.npz`)

### 6.2 Recent Bug Fixes

**Issue 1: vis_wb3d.py KeyError** ✅ Fixed (Dec 24)
- **Problem:** Loading 3D keypoints with wrong key name
- **Fix:** Changed from `keypoints` → `keypoints_3d` on line 311
- **Commit:** c197c63

**Issue 2: Output Clutter in vis_wb3d.py** ✅ Fixed (Dec 24)
- **Changes:**
  - Removed PNG debug output (`debug_wb3d_001.png`)
  - Simplified title: "3D Skeleton" (removed frame number from 3D panel)
  - Reduced verbose logging (debug=False)
- **Commit:** c197c63

**Issue 3: Model Naming Inconsistency** ✅ Fixed (Dec 24)
- **Problem:** Inconsistent naming (`rtmpose_halpe26`, `wb3d`)
- **Solution:** Renamed throughout codebase:
  - `rtmpose_halpe26` → `rtmpose_h26`
  - `wb3d` → `wholebody`
- **Impact:** Output files now named `kps_2d_wholebody.npz`, `kps_3d_wholebody.npz`
- **Commit:** a60b59b

---

## 7. Directory Structure

```
unifiedposepipeline/
├── configs/
│   └── udp_video.yaml           # Main configuration
├── demo_data/
│   ├── videos/
│   │   └── dance.mp4            # Input video
│   ├── images/
│   │   └── sample.jpg           # Input image (for testing)
│   ├── outputs/                 # All generated outputs
│   │   ├── detections.npz
│   │   ├── kps_2d_rtm.npz
│   │   ├── kps_2d_wholebody.npz
│   │   ├── kps_3d_wholebody.npz
│   │   ├── kps_3d_magf.npz
│   │   ├── result_2D.mp4
│   │   └── result_wb_2D3D.mp4
│   └── joint_formats/           # Joint semantic definitions
│       ├── coco17_2d.json
│       ├── halpe26_2d.json
│       ├── dwpose133_2d.json
│       ├── dwpose133_3d.json
│       └── h36m17_3d.json
├── models/                      # Downloaded model weights
│   ├── yolo/
│   ├── rtmlib/
│   ├── vitpose/
│   ├── wb3d/
│   └── motionagformer/
├── lib/                         # Library code
│   └── motionagformer/          # MotionAGFormer model code
├── snippets/                    # Debug/analysis scripts (not in repo)
├── udp_video.py                 # Main pipeline
├── udp_3d_lifting.py            # 3D lifting script
├── vis_wb3d.py                  # Wholebody visualization
└── README.md
```

---

## 8. Google Colab Environment

**Paths:**
- Root: `/content/`
- Models: `/content/models/`
- Code: `/content/unifiedposepipeline/`
- Demo data: `/content/unifiedposepipeline/demo_data/`

**Input Defaults:**
- Video: `/content/unifiedposepipeline/demo_data/videos/dance.mp4`
- Image: `/content/unifiedposepipeline/demo_data/images/sample.jpg`

**Output Location:**
- `/content/unifiedposepipeline/demo_data/outputs/`

---

## 9. NPZ File Key Reference

### Detection Output (`detections.npz`)
```python
{
    'frame_numbers': np.array(shape=(N,), dtype=int64),
    'bboxes': np.array(shape=(N, 4), dtype=int64)  # [x1, y1, x2, y2]
}
```

### 2D Keypoints Output
```python
{
    'frame_numbers': np.array(shape=(N,), dtype=int),
    'keypoints': np.array(shape=(N, J, 2), dtype=float),  # J = 17/26/133
    'scores': np.array(shape=(N, J), dtype=float),
    'joint_format': str,  # e.g., "coco17_2d.json"
    'model_type': str     # e.g., "rtmpose", "wholebody"
}
```

### 3D Keypoints Output (Wholebody)
```python
{
    'frame_numbers': np.array(shape=(N,), dtype=int),
    'keypoints_3d': np.array(shape=(N, 133, 3), dtype=float),  # Note: keypoints_3d!
    'scores': np.array(shape=(N, 133), dtype=float),
    'joint_format': str,  # "dwpose133_3d.json"
    'model_type': str     # "wholebody"
}
```

### 3D Lifted Output (MotionAGFormer)
```python
{
    'poses_3d': np.array(shape=(N, 17, 3), dtype=float),  # H36M-17 format
    # Additional metadata may be present
}
```

---

## 10. Recent Work Summary (Last 200 Messages)

### Phase 1: WB3D Skeleton Visualization Issues
- **Problem:** MotionAGFormer normalization caused head clipping
- **Solution:** Changed from `np.max(np.abs())` → `np.max()` and implemented adaptive Z-limits
- **Files Modified:** Normalization logic in 3D visualization

### Phase 2: vis_wb3d.py Runtime Error
- **Problem:** `KeyError: 'keypoints is not a file in the archive'`
- **Root Cause:** 2D file uses key `keypoints`, 3D file uses `keypoints_3d`
- **Solution:** Fixed line 311 in `vis_wb3d.py`

### Phase 3: Output Cleanup
- **User Request:** Remove PNG output, simplify title, reduce logging
- **Changes Applied:**
  1. Removed PNG save logic (debug_wb3d_001.png)
  2. Changed title from "WB3D Skeleton (Frame X)" → "3D Skeleton"
  3. Set debug=False for all frames
  4. Simplified success messages

### Phase 4: Model Naming Refactor (Current)
- **Motivation:** Improve naming clarity and consistency
- **Changes:**
  - `rtmpose_halpe26` → `rtmpose_h26` (shorter, consistent)
  - `wb3d` → `wholebody` (descriptive, clear)
- **Scope:** Updated both `configs/udp_video.yaml` and `udp_video.py`
- **Impact:** Output file names changed, config files need updating

### All Changes Committed to GitHub ✅
- Commit c197c63: vis_wb3d.py fixes
- Commit a60b59b: Model naming refactor

---

## 11. Development Workflow

### Running the Pipeline (Colab)
```bash
# Install dependencies
pip install -r requirements.txt

# Run standalone detector (new modular approach)
python run_detector.py --config configs/detector.yaml

# Run full pipeline
python udp_video.py --config configs/udp_video.yaml

# Run 3D lifting only (after Stage 2)
python udp_3d_lifting.py \
    --keypoints demo_data/outputs/kps_2d_rtm.npz \
    --video demo_data/videos/dance.mp4 \
    --output demo_data/outputs/kps_3d_magf.npz \
    --max-frames 360

# Visualize wholebody 3D
python vis_wb3d.py \
    --video demo_data/videos/dance.mp4 \
    --kps2d demo_data/outputs/kps_2d_wholebody.npz \
    --kps3d demo_data/outputs/kps_3d_wholebody.npz \
    --output demo_data/outputs/debug_wb3d.mp4 \
    --max-frames 120
```

### Switching Models
Edit `configs/udp_video.yaml`:
```yaml
pose_estimation:
  method: wholebody  # Change to: rtmpose, vitpose, rtmpose_h26, or wholebody
```

---

## 12. Known Issues & Future Work

### Current Limitations
1. **Backward Compatibility:** Old config files with `rtmpose_halpe26` or `wb3d` need manual updating
2. **File Naming:** Existing output files with old names (`kps_2d_wb3d.npz`) won't be auto-detected

### Potential Improvements
1. Add deprecation warnings for old model names
2. Implement file name migration tool
3. Add config validation
4. Support batch video processing
5. Add more 3D lifting models (beyond MAGF)

---

## 13. Git Repository

- **GitHub:** `pradeepj247/unifiedposepipeline`
- **Branch:** `main`
- **Latest Commit:** a60b59b (Model naming refactor)
- **Working Directory (Local):** `D:\trials\unifiedpipeline\newrepo\`

---

**Document generated from conversation history spanning pipeline development, bug fixes, and refactoring efforts through December 24, 2025.**
