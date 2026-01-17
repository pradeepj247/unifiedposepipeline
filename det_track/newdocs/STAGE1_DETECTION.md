# Stage 1: Detection + Eager Crop Extraction

**Implementation**: `stage1_detect.py`

## Purpose
Run YOLO to detect all persons in video, extract person crops, and build searchable cache for fast Stage 3c access.

## Inputs
- Canonical video from Stage 0

## Outputs
- `detections_raw.npz`: All bboxes with `detection_idx` linkage
- `crops_cache.pkl`: 527 MB cache with all 8832 crops (deleted after Stage 3c)
- Timing: Stage 1 breakdown

## Processing Pipeline

```
Video Stream (25 fps, 1920×1080)
    ↓
[H.264 CPU Decode] ← BOTTLENECK: 18-20ms per frame
    ↓
Batch of Frames
    ↓
[YOLO Inference] ← GPU: 8-10ms per frame
    ├─→ Filter: confidence ≥ 0.3
    ├─→ Filter: class=person (ignore other classes)
    ├─→ Limit: max 15 detections/frame
    └─→ Create detection_idx (sequential counter)
    ↓
[Crop Extraction] ← CPU: 2-3ms per crop
    ├─→ Extract bbox region from frame
    ├─→ Aspect-ratio-preserving resize (max 400px)
    ├─→ Store with detection_idx linkage
    └─→ Save to crops_cache.pkl
    ↓
[Frame Save] ← Disk I/O: 5-6s total
    ├─→ Save detections_raw.npz
    └─→ Save crops_cache.pkl (527 MB)
```

## Performance Breakdown (2025 frames)

| Component | Time | FPS |
|-----------|------|-----|
| Model load | 2.68s | - |
| Detection + extraction | 42.87s | 47.2 |
| File saving | 5.89s | - |
| **Total** | **51.16s** | **39.5** |

**Actual YOLO FPS**: 53.6 (pure inference only)  
**Overhead from crop extraction**: 4-5s (12%)

## Key Design Decisions

### 1. YOLOv8s Model Selection
**Why YOLOv8s (not YOLOv8n)?**
- Tested both models back-to-back in same Colab session
- YOLOv8n: 37.78s (53.6 FPS)
- YOLOv8s: 38.36s (52.8 FPS)
- **Difference**: 0.58s (within Colab variance)
- **Root cause**: Video decoding bottleneck, not model inference
- **Decision**: Use YOLOv8s for better accuracy with no speed penalty

### 2. Eager Crop Extraction
**Why extract crops in Stage 1?**
```
Option A: Extract here, reuse in Stage 3c
  Stage 1: +5s overhead (crop extraction + save to disk)
  Stage 3c: -11s savings (O(1) lookup, no re-reading video)
  Net: -6s ✅

Option B: Extract crops on-demand in Stage 3c
  Stage 1: -5s (fast)
  Stage 3c: +12s (re-read video, re-decode, extract)
  Net: +7s ❌
```
**Selected**: Option A (eager extraction)

### 3. Crop Resizing Strategy
**Size Decision**: Max 400 pixels (preserves aspect ratio)
```python
if max(height, width) > 400:
    scale = 400 / max(height, width)
    new_w = int(width * scale)
    new_h = int(height * scale)
    crop_resized = cv2.resize(crop, (new_w, new_h))
```

**Why?**
- Typical person bbox: 100-200px wide
- 400px max preserves quality without bloat
- Result: 527 MB total (manageable for Stage 3c loading)
- Smaller = faster WebP encoding in Stage 4

### 4. detection_idx Linkage
**Every crop tagged with sequential detection_idx**
```python
all_detection_indices.append(detection_global_idx)
all_crops.append({
    'detection_idx': detection_global_idx,  ← Critical!
    'frame_idx': frame_idx,
    'bbox': [x1, y1, x2, y2],
    'crop': crop_resized,
    'confidence': conf
})
detection_global_idx += 1  # Always increment
```

**Why?** Enables unambiguous tracking through all stages:
- detection_idx is preserved in Stage 2 (tracklets)
- detection_idx is preserved in Stage 3b (canonical persons)
- Stage 3c does O(1) lookup: `crops_by_idx[detection_idx]` to fetch exact crop

## Configuration

```yaml
stage1_detect:
  detector:
    model_path: ${models_dir}/yolo/yolov8s.pt
    confidence: 0.3           # Detection threshold
    device: cuda              # or cpu
    detect_only_humans: true  # Filter to class=person
  
  detection_limit:
    method: hybrid            # Limit strategy
    max_count: 15             # Max detections per frame
    min_confidence: 0.3
```

## Output Data Format

### detections_raw.npz
```python
{
    'frame_numbers': np.array([0, 0, 1, 1, ...]),     # Frame indices
    'bboxes': np.array([[x1,y1,x2,y2], ...]),         # Bounding boxes
    'confidences': np.array([0.92, 0.87, ...]),       # Detection scores
    'classes': np.array([0, 0, ...]),                 # All are person (0)
    'detection_indices': np.array([0, 1, 2, 3, ...])  # ← NEW: Critical linkage
}
```

### crops_cache.pkl
```python
all_crops = [
    {
        'detection_idx': 0,
        'frame_idx': 0,
        'bbox': [x1, y1, x2, y2],
        'confidence': 0.92,
        'class_id': 0,
        'crop': np.ndarray  # Resized image (uint8)
    },
    ...  # 8832 entries
]
```

## File Sizes

| File | Size | Lifetime |
|------|------|----------|
| detections_raw.npz | 0.16 MB | Permanent (used by Stage 2, 3b) |
| crops_cache.pkl | 527 MB | **Deleted after Stage 3c** ✅ |

## Performance Notes

- **Video Decoding**: 18-20ms per frame (CPU H.264 bottleneck)
- **YOLO Inference**: 8-10ms per frame (GPU)
- **Crop Extraction**: 2-3ms per crop (CPU, linear with bbox count)
- **Total**: Dominated by video I/O, not model inference

---

**Related**: [Back to Master](README_MASTER.md) | [Stage 2 →](STAGE2_TRACKING.md)
