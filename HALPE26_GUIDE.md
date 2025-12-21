# Halpe26 Support - 26 Keypoint Pose Estimation

## Overview

The Unified Pose Pipeline now supports **Halpe26** models with **26 keypoints** in addition to the standard COCO **17 keypoints**.

## What is Halpe26?

Halpe26 extends the standard COCO 17-keypoint format with 9 additional keypoints:
- **6 foot keypoints** (indices 17-22): 3 per foot for detailed foot pose
- **3 body keypoints** (indices 23-25): neck, chest/spine, and pelvis for detailed torso

### Keypoint Layout

```
COCO Body (0-16):
0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
5: Left Shoulder, 6: Right Shoulder
7: Left Elbow, 8: Right Elbow
9: Left Wrist, 10: Right Wrist
11: Left Hip, 12: Right Hip
13: Left Knee, 14: Right Knee
15: Left Ankle, 16: Right Ankle

Feet (17-22):
17-19: Left foot (big toe, small toe, heel)
20-22: Right foot (big toe, small toe, heel)

Additional Body (23-25):
23: Neck
24: Chest/Spine
25: Pelvis
```

## Usage

### Video Processing

**COCO-17 (standard):**
```bash
python udp_video.py --config configs/udp_video.yaml
```

**Halpe26 (26 keypoints):**
```bash
python udp_video.py --config configs/udp_video_halpe26.yaml
```

### Image Processing

**COCO-17 (standard):**
```bash
python udp_image.py --config configs/udp_image.yaml
```

**Halpe26 (26 keypoints):**
```bash
python udp_image.py --config configs/udp_image_halpe26.yaml
```

### Benchmarking

Compare COCO-17 vs Halpe26 performance:
```bash
python benchmark_halpe26.py --video demo_data/videos/dance.mp4 --frames 360
```

## Model Configurations

### Available Halpe26 Models

| Model | Resolution | PCK@0.1 | AUC | Params | URL |
|-------|-----------|---------|-----|--------|-----|
| RTMPose-t | 256×192 | 91.89 | 66.35 | 3.51M | [Download](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7-halpe26_700e-256x192-6020f8a6_20230605.zip) |
| RTMPose-s | 256×192 | 93.01 | 68.62 | 5.70M | [Download](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.zip) |
| RTMPose-m | 256×192 | 94.75 | 71.91 | 13.93M | [Download](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip) |
| RTMPose-l | 256×192 | 95.37 | 73.19 | 28.11M | [Download](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.zip) |
| RTMPose-m | 384×288 | 95.15 | 73.56 | 14.06M | [Download](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-384x288-89e6428b_20230605.zip) |
| **RTMPose-l** | **384×288** | **95.56** | **74.38** | **28.24M** | **[Download](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-body7_pt-body7-halpe26_700e-384x288-734182ce_20230605.zip)** ⭐ |
| RTMPose-x | 384×288 | 95.74 | 74.82 | 50.00M | [Download](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip) |

**Default in config:** RTMPose-L 384×288 (best accuracy/speed tradeoff)

## Visualization

The skeleton drawing automatically detects the number of keypoints:

**COCO-17:**
- All keypoints: **Red**
- Rainbow-colored skeleton edges

**Halpe26:**
- Body keypoints (0-16): **Red**
- Feet keypoints (17-22): **Green**
- Additional body (23-25): **Blue**
- Rainbow-colored skeleton edges

## Expected Performance

Based on RTMPose-L 384×288:

| Metric | COCO-17 | Halpe26 | Difference |
|--------|---------|---------|------------|
| Keypoints | 17 | 26 | +9 |
| Expected FPS | ~48 FPS | ~45-48 FPS | -0 to -3 FPS |
| Model Size | 106 MB | ~110 MB | +4 MB |
| Accuracy (PCK) | 95.08 | 95.56 | +0.48 |

*Note: Halpe26 provides 53% more keypoints with minimal speed impact!*

## Use Cases

**Use COCO-17 when:**
- Standard body pose is sufficient
- Maximum speed is critical
- Deploying on resource-constrained devices

**Use Halpe26 when:**
- Detailed foot tracking is needed (dance, sports, gait analysis)
- Detailed torso tracking is important (biomechanics, exercise form)
- Extra accuracy is valuable (high-quality motion capture)
- Device has sufficient resources (~5% slower)

## Technical Details

### Skeleton Definitions

**COCO Edges (16 connections):**
Head, arms, torso, legs

**Halpe26 Edges (28 connections):**
- All COCO edges (16)
- Foot edges (6): 3 per foot
- Additional body edges (6): neck-head, neck-pelvis, pelvis-hips

### Output Format

**NPZ file structure:**
```python
{
    'frame_numbers': (N,),      # Frame indices
    'keypoints': (N, K, 2),     # K=17 or 26, [x, y] coords
    'scores': (N, K)            # Confidence scores
}
```

### Code Implementation

The pipeline automatically handles both formats:
```python
def draw_skeleton_unified(image, keypoints, scores):
    num_keypoints = len(keypoints)
    
    if num_keypoints == 26:
        edges = HALPE26_EDGES
        # ... use Halpe26 skeleton
    else:  # 17 keypoints
        edges = COCO_EDGES
        # ... use COCO skeleton
```

## References

- **Halpe Dataset:** https://github.com/Fang-Haoshu/Halpe-FullBody/
- **RTMPose Paper:** https://arxiv.org/abs/2303.07399
- **MMPose Halpe26 Models:** https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose

## Citation

```bibtex
@misc{jiang2023rtmpose,
      title={RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose},
      author={Tao Jiang and Peng Lu and Li Zhang and Ningsheng Ma and Rui Han and Chengqi Lyu and Yining Li and Kai Chen},
      year={2023},
      eprint={2303.07399},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
