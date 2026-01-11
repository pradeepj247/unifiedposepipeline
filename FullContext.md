# Unified Pose Estimation Pipeline - Algorithms & Data Formats Reference

**Last Updated:** January 11, 2026  
**Focus:** Core algorithms, data formats, and model specifications (framework-independent)

---

## 1. Detection & Tracking Overview

The pipeline includes a **multi-stage detection and tracking system** (11 configurable stages) for:
- Person detection using YOLOv8
- Multi-person tracking with ByteTrack offline tracking
- Tracklet analysis and canonical person grouping
- Person ranking and selection

**For detailed implementation of the 11-stage detection & tracking pipeline:**
- See: [det_track/PIPELINE_DESIGN.md](det_track/PIPELINE_DESIGN.md)
- See: Individual stage scripts in `det_track/` folder
- See: Configuration in [det_track/configs/pipeline_config.yaml](det_track/configs/pipeline_config.yaml)

This document focuses on the **core algorithms** and **data formats** used throughout the pipeline.

---

## 2. 2D Pose Estimation & 3D Lifting

### 2.1 2D Pose Estimation Models

The pipeline supports four model families for estimating 2D human keypoints:

#### RTMPose (Real-Time Multi-Person Pose Estimation)
- **Format:** ONNX model
- **Keypoints:** 17 (COCO body joints) or 26 (Halpe26 with feet)
- **Input Size:** 288×384 or 256×192 (configurable)
- **Backend:** ONNX Runtime
- **Device:** GPU (CUDA) or CPU
- **Speed:** ~48 FPS on T4 GPU (384×288 input)
- **Strengths:** Fast inference, real-time capable, good accuracy
- **Use Case:** Real-time applications, production deployments

#### ViTPose (Vision Transformer Pose)
- **Format:** PyTorch (.pth)
- **Keypoints:** 17 (COCO body joints)
- **Input Size:** 384×384 (fixed)
- **Architecture:** Vision Transformer backbone
- **Device:** GPU (CUDA) or CPU, auto-detects
- **Speed:** ~39 FPS on T4 GPU
- **Strengths:** Highest accuracy among body-only models, robust to occlusions
- **Use Case:** High-precision applications, when accuracy matters more than speed

#### RTMPose-Halpe26
- **Format:** ONNX model
- **Keypoints:** 26 (COCO-17 + 6 foot keypoints + 3 face keypoints)
- **Input Size:** 288×384
- **Strengths:** Includes foot keypoints for complete lower-body coverage
- **Use Case:** Applications requiring foot tracking (sports, dance, physical analysis)

#### RTMPose3D Wholebody
- **Format:** ONNX model
- **Keypoints:** 133 (COCO-WholeBody: body + face + hands)
- **Output:** Both 2D and 3D coordinates simultaneously
- **Input Size:** 288×384
- **Strengths:** Single-model whole-body detection, includes hand/face details
- **Use Case:** Full-body analysis, gesture recognition, detailed interaction understanding

### 2.2 Model Comparison

| Aspect | RTMPose | ViTPose | RTMPose-H26 | Wholebody |
|--------|---------|---------|-------------|-----------|
| **Keypoints** | 17 | 17 | 26 | 133 |
| **Speed (FPS)** | 48 | 39 | ~45 | ~25 |
| **Accuracy** | High | Highest | High | Medium |
| **Feet Tracking** | ❌ | ❌ | ✅ | ✅ |
| **Hand Tracking** | ❌ | ❌ | ❌ | ✅ |
| **Face Landmarks** | ❌ | ❌ | Limited | ✅ |
| **Backend** | ONNX | PyTorch | ONNX | ONNX |
| **Inference Time** | ~21ms | ~26ms | ~22ms | ~40ms |

### 2.3 3D Pose Lifting (MotionAGFormer)

**Model:** MotionAGFormer (Motion Attention-based Graphical Former)

**Purpose:** Lift 2D keypoints to 3D using temporal context

**Input:**
- 2D keypoints in H36M-17 format (17 joints, 2D coordinates)
- Temporal window: 243 frames

**Output:**
- 3D keypoints in H36M-17 format (17 joints, 3D coordinates)
- Root-centered normalization (pelvis at origin)

**Algorithm Details:**
- **Temporal Processing:** Uses 243-frame window with past/future context
- **Architecture:** Graph-based attention mechanism with motion modeling
- **Training:** Human3.6M dataset
- **Test-Time Augmentation:** Horizontal flip averaging for robustness

**Processing Pipeline:**
1. **Format Conversion:** COCO-17 → H36M-17 format mapping
2. **Screen Normalization:** Convert pixel coordinates to normalized space
3. **Temporal Resampling:** Interpolate/extrapolate to 243-frame window if needed
4. **Model Inference:** Graph-based temporal lifting with attention
5. **Post-Processing:** Apply camera rotation matrix for proper orientation

**Key Features:**
- Smooth temporal consistency (exploits motion patterns)
- Handles occlusions via temporal interpolation
- Normalizes output to unit scale (root-centered at pelvis)
- Bidirectional temporal modeling (uses future frames)

---

## 3. Joint Format Specifications

### 3.1 COCO-17 (OpenPose Convention)

**Structure:** 17 joints, hierarchical skeleton

| ID | Joint Name | Parent | Description |
|:--:|-----------|--------|-------------|
| 0 | Nose | - | Face center |
| 1 | Left Eye | Nose | Eye region |
| 2 | Right Eye | Nose | Eye region |
| 3 | Left Ear | Left Eye | Ear region |
| 4 | Right Ear | Right Eye | Ear region |
| 5 | Left Shoulder | - | Upper limb |
| 6 | Right Shoulder | - | Upper limb |
| 7 | Left Elbow | Left Shoulder | Arm joint |
| 8 | Right Elbow | Right Shoulder | Arm joint |
| 9 | Left Wrist | Left Elbow | Hand joint |
| 10 | Right Wrist | Right Elbow | Hand joint |
| 11 | Left Hip | - | Torso-leg junction |
| 12 | Right Hip | - | Torso-leg junction |
| 13 | Left Knee | Left Hip | Leg joint |
| 14 | Right Knee | Right Hip | Leg joint |
| 15 | Left Ankle | Left Knee | Foot joint |
| 16 | Right Ankle | Right Knee | Foot joint |

**Skeleton Edges (16 connections):**
```
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Arms
    (5, 11), (6, 12), (11, 12),               # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)    # Legs
]
```

**Usage:** RTMPose (17-kpt variant), ViTPose standard output

---

### 3.2 Halpe26 Format

**Structure:** COCO-17 + 9 additional keypoints (including feet)

| ID | Joint Name | Parent | Notes |
|:--:|-----------|--------|-------|
| 0-16 | COCO joints | - | Same as COCO-17 |
| 17 | Neck | - | Midpoint between shoulders |
| 18-19 | Head top, Head bottom | - | Additional head landmarks |
| 20-22 | Left foot (3 points) | Left Ankle | Foot structure |
| 23-25 | Right foot (3 points) | Right Ankle | Foot structure |

**Skeleton Edges (32 connections):** Extends COCO edges with foot tracking

**Usage:** RTMPose-Halpe26 model

---

### 3.3 H36M-17 Format

**Structure:** 17 joints, different naming convention (Human3.6M protocol)

| ID | Joint Name | H36M ID |
|:--:|-----------|---------|
| 0 | Pelvis (root) | Hip/Pelvis center |
| 1 | Right Hip | 1 |
| 2 | Right Knee | 2 |
| 3 | Right Ankle | 3 |
| 4 | Left Hip | 4 |
| 5 | Left Knee | 5 |
| 6 | Left Ankle | 6 |
| 7 | Spine | 7 (mid-body) |
| 8 | Thorax | 8 (chest) |
| 9 | Neck/Nose | 9 |
| 10 | Head | 10 |
| 11 | Left Shoulder | 11 |
| 12 | Left Elbow | 12 |
| 13 | Left Wrist | 13 |
| 14 | Right Shoulder | 14 |
| 15 | Right Elbow | 15 |
| 16 | Right Wrist | 16 |

**Key Differences from COCO:**
- Pelvis-centered (ID 0)
- More torso joints (7, 8, 9 represent spine/chest/neck)
- No separate eyes/ears
- Standard for 3D lifting models (MotionAGFormer)

**Conversion:** COCO → H36M requires joint remapping (ID transformation)

---

### 3.4 COCO-WholeBody-133

**Structure:** 17 body + 68 face + 21 left hand + 21 right hand + 6 foot = 133 total

| Region | Count | IDs | Description |
|--------|-------|-----|-------------|
| Body | 17 | 0-16 | COCO-17 joints |
| Left Hand | 21 | 17-37 | Finger keypoints |
| Right Hand | 21 | 38-58 | Finger keypoints |
| Face | 68 | 59-126 | Dense face landmarks |
| Feet | 6 | 127-132 | Extended foot coverage |

**Usage:** RTMPose3D Wholebody model (simultaneous 2D + 3D output)

---

## 4. Visualization System

### 4.1 2D Skeleton Rendering

**Color Scheme:** Rainbow HSV gradient applied to skeleton edges
- Automatic color assignment: `get_edge_colors(num_edges)` function
- HSV space → RGB conversion ensures perceptual uniformity
- Different color palettes for different skeleton types:
  - COCO-17: 16 edge colors
  - Halpe26: 32 edge colors
  - WholeBody: 50+ edge colors

**Rendering Pipeline:**
1. Extract bounding box region from frame
2. Draw pose skeleton (line segments between joints)
3. Draw joint circles (configurable radius based on confidence)
4. Apply alpha blending for semi-transparency
5. Overlay on original frame

**Frame Types:**
- **Single-person mode:** One skeleton per frame
- **Multi-person mode:** Multiple skeletons (person_id → unique color)
- **2D+3D overlay:** 2D skeleton + projected 3D pose

### 4.2 3D Rendering

**Output:** GIF or MP4 with rotating 3D skeleton
- Matplotlib 3D plotting with rotation animation
- Automatic axis scaling to frame dimensions
- View angle: Gradually rotates 0° → 360°
- Frame rate: 30 FPS (configurable)

**Coordinate System:**
- X: Left-right
- Y: Up-down
- Z: Depth (forward-backward)
- Origin: Pelvis (root joint)

---

## 5. NPZ File Format Reference

### 5.1 Detections NPZ (`detections_raw.npz`)

**Keys:**
- `frame_numbers`: `int64 (N,)` — frame index for each detection
- `bboxes`: `int64 (N, 4)` — bounding boxes `[x1, y1, x2, y2]`
- `confidences`: `float32 (N,)` — detection confidence scores
- `classes`: `int64 (N,)` — class IDs (0 = person)

**Guarantee:** One row per detection, multiple detections per frame possible

**Example:**
```python
import numpy as np
data = np.load('detections_raw.npz')
frame_0_detections = data['bboxes'][data['frame_numbers'] == 0]  # All detections in frame 0
```

### 5.2 Keypoints NPZ (2D/3D Pose)

**Keys:**
- `keypoints`: `float32 (N_frames, N_persons, N_joints, 2)` or `(N, N, N, 3)` for 3D
- `scores`: `float32 (N_frames, N_persons, N_joints)` — confidence per joint
- `frame_numbers`: `int64 (N_frames,)` — frame indices
- `joint_format`: `str` — joint naming convention (COCO, Halpe26, H36M, etc.)
- `model_type`: `str` — model identifier (rtmpose, vitpose, wholebody, etc.)

**Guarantee:** One row per frame (no repeated frames)

**Access Pattern:**
```python
kps = np.load('keypoints.npz')
frame_5_pose = kps['keypoints'][5]  # Shape: (N_persons, N_joints, 2)
frame_5_confidence = kps['scores'][5]  # Shape: (N_persons, N_joints)
```

### 5.3 Tracklets NPZ (`tracklets_raw.npz`)

**Structure:** List of tracklet dictionaries

**Keys (per tracklet):**
- `tracklet_id`: `int` — unique identifier
- `frame_numbers`: `int64 (M,)` — frames this tracklet appears in
- `bboxes`: `int64 (M, 4)` — bounding boxes for each frame
- `confidences`: `float32 (M,)` — detection confidences

**Format:**
```python
tracklets_data = np.load('tracklets_raw.npz')
tracklets_list = tracklets_data['tracklets'].tolist()  # Decode object array
for tracklet in tracklets_list:
    print(f"ID {tracklet['tracklet_id']}: frames {tracklet['frame_numbers']}")
```

### 5.4 Canonical Persons NPZ (`canonical_persons.npz`)

**Structure:** Aggregated persons from merged tracklets

**Keys (per person):**
- `person_id`: `int` — canonical person identifier
- `tracklet_ids`: `list[int]` — which tracklets were merged
- `frame_numbers`: `int64 (M,)` — all frames this person appears in
- `bboxes`: `int64 (M, 4)` — continuous trajectory
- `confidences`: `float32 (M,)` — confidence scores

**Guarantee:** Person appears continuously across merged tracklets

---

## 6. Data Flow & Integration

### Overall Pipeline Sequence:

1. **Detection (Stage 1):** Video → YOLOv8 → `detections_raw.npz`
2. **Pose Estimation (Stage 2):** Detections → RTMPose/ViTPose/Wholebody → `keypoints_*.npz`
3. **3D Lifting (Stage 3):** 2D Keypoints → MotionAGFormer → `keypoints_3d.npz`
4. **Visualization:** Keypoints → Skeleton rendering → Video/GIF output

### Cross-Format Compatibility:

- **COCO ↔ H36M conversion:** Required for MotionAGFormer (expects H36M-17 input)
- **Multiple backends:** Select via config, output format standardized
- **Frame indexing:** All stages use consistent frame numbering for frame-by-frame alignment

---

**This document serves as a timeless reference for core algorithms and data formats, independent of implementation details. For the current 11-stage detection/tracking pipeline architecture and implementation specifics, see [det_track/PIPELINE_DESIGN.md](det_track/PIPELINE_DESIGN.md).**
