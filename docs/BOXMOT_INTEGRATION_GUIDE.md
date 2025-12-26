# BoxMOT Tracking Integration Guide

**Last Updated:** December 25, 2024  
**BoxMOT Version:** 16.0.4  
**Environment:** Google Colab

---

## Overview

BoxMOT is a powerful multi-object tracking library that provides:
- 7 different tracking algorithms
- Pluggable ReID (Re-Identification) models for appearance-based tracking
- Motion-only and Motion+Appearance tracking modes
- ONNX/TensorRT export for speed optimization

---

## Available Trackers on Our System

### ✅ ALL 7 Trackers Confirmed Available!

**Important:** BoxMOT v16.0.4 uses **case-sensitive** class names. The issue was incorrect casing:
- ✅ Correct: `DeepOcSort`, `StrongSort`, `OcSort`, `HybridSort`
- ❌ Wrong: `DeepOCSORT`, `StrongSORT`, `OCSORT`, `HybridSORT`

| Tracker | HOTA | MOTA | IDF1 | FPS | Description | Class Name |
|---------|------|------|------|-----|-------------|------------|
| **botsort** | 69.4 | 78.2 | 81.8 | 46 | **Best accuracy** - BoT-SORT with appearance + motion | `BotSort` |
| **bytetrack** | 67.7 | 78.0 | 79.2 | 1265 | **Fastest** - Motion-only, lightweight | `ByteTrack` |
| **boosttrack** | 69.3 | 75.9 | 83.2 | 25 | **Best IDF1** - Strong identity consistency | `BoostTrack` |
| **deepocsort** | 67.8 | 75.9 | 80.5 | 12 | Deep learning based | `DeepOcSort` ⚠️ |
| **strongsort** | 68.0 | 76.2 | 80.8 | 17 | Strong baseline | `StrongSort` ⚠️ |
| **ocsort** | 66.4 | 74.5 | 77.9 | 1483 | Fast observation-centric | `OcSort` ⚠️ |
| **hybridsort** | 67.4 | 74.1 | 79.1 | 25 | Hybrid approach | `HybridSort` ⚠️ |

**⚠️ Note on Case Sensitivity:** When importing in Python, use exact case: `from boxmot import DeepOcSort` (not `DeepOCSORT`)

---

## Tracking Modes

### 1. Motion-Only Tracking (Fast)
- Uses Kalman filtering for motion prediction
- No appearance features (no ReID model needed)
- Very fast (46-1265 Hz depending on tracker)
- Good for simple scenarios with no occlusions

**Configuration:**
```yaml
tracking:
  enabled: true
  tracker: bytetrack  # Fastest motion-only tracker
  reid:
    enabled: false  # No ReID
```

**Best For:**
- High FPS requirements
- CPU-only systems
- Simple tracking scenarios
- When identity switches are acceptable

### 2. Motion + Appearance Tracking (Accurate)
- Combines Kalman filtering with visual appearance features
- Requires ReID model (downloads automatically on first use)
- Slower but maintains identity through occlusions
- Better for crowded scenes

**Configuration:**
```yaml
tracking:
  enabled: true
  tracker: botsort  # Best tracker with ReID support
  reid:
    enabled: true
    weights_path: models/reid/osnet_x0_25_msmt17.pt
```

**Best For:**
- Crowded scenes with occlusions
- When maintaining consistent IDs is critical
- Long-term tracking
- Re-identification after temporary disappearance

---

## ReID Models (Appearance Features)

BoxMOT supports multiple ReID models with different speed/accuracy tradeoffs:

### Lightweight (Recommended for Colab):
| Model | Parameters | Speed | Notes |
|-------|-----------|-------|-------|
| `lmbn_n_cuhk03_d` | ~1M | Fast | Lightest option |
| `osnet_x0_25_market1501` | ~0.5M | Fast | Good balance |
| `osnet_x0_25_msmt17` | ~0.5M | Fast | **Default choice** |

### Medium Weight:
| Model | Parameters | Speed | Notes |
|-------|-----------|-------|-------|
| `mobilenetv2_x1_4_msmt17` | ~6M | Medium | MobileNet backbone |
| `osnet_x1_0_msmt17` | ~2M | Medium | Better accuracy |
| `resnet50_msmt17` | ~25M | Slow | ResNet backbone |

### Heavy (Not recommended for Colab):
| Model | Parameters | Speed | Notes |
|-------|-----------|-------|-------|
| `clip_market1501` | ~150M | Very Slow | CLIP-based |
| `clip_vehicleid` | ~150M | Very Slow | For vehicles |

**Training Datasets:**
- `market1501`: Person re-identification dataset (small)
- `msmt17`: Large-scale person re-identification dataset (diverse)
- `duke`: DukeMTMC dataset (similar to market1501)
- `cuhk03`: CUHK person re-identification dataset

**Recommendation:** Start with `osnet_x0_25_msmt17` (default) - good balance of speed and accuracy.

---

## Model Export (Speed Optimization)

BoxMOT supports exporting ReID models to faster formats:

### ONNX Export (CPU/GPU):
```bash
boxmot export --weights osnet_x0_25_msmt17.pt --include onnx --device cpu
```

### TensorRT Export (GPU only, dynamic batch):
```bash
boxmot export --weights osnet_x0_25_msmt17.pt --include engine --device 0 --dynamic
```

### OpenVINO Export (Intel CPUs):
```bash
boxmot export --weights osnet_x0_25_msmt17.pt --include openvino --device cpu
```

**Benefits:**
- 2-5x faster inference
- Lower memory usage
- Better for production deployment

**Usage in Config:**
```yaml
reid:
  enabled: true
  weights_path: models/reid/osnet_x0_25_msmt17.onnx  # Use .onnx instead of .pt
```

---

## Recommended Configurations

### Configuration 1: Fast Motion-Only (1265 Hz)
**Use Case:** Real-time processing, simple scenes, CPU-friendly

```yaml
tracking:
  enabled: true
  tracker: bytetrack
  reid:
    enabled: false
```

**Pros:** Extremely fast, no model downloads needed  
**Cons:** May lose identity during occlusions

---

### Configuration 2: Balanced (46 Hz)
**Use Case:** Most scenarios, good accuracy/speed tradeoff

```yaml
tracking:
  enabled: true
  tracker: botsort
  reid:
    enabled: false  # Start with motion-only
```

**Pros:** Good tracking quality, reasonable speed  
**Cons:** Still motion-only, may have ID switches

---

### Configuration 3: Best Accuracy (46 Hz with ReID)
**Use Case:** Complex scenes, crowded environments, critical ID consistency

```yaml
tracking:
  enabled: true
  tracker: botsort  # Best overall (HOTA: 69.4)
  reid:
    enabled: true
    weights_path: models/reid/osnet_x0_25_msmt17.pt
```

**Pros:** Best ID consistency, handles occlusions  
**Cons:** Slower, downloads ReID model (~10MB) on first run

---

### Configuration 4: Best IDF1 Score (83.2)
**Use Case:** When identity preservation is most important

```yaml
tracking:
  enabled: true
  tracker: boosttrack  # Best IDF1 score
  reid:
    enabled: true
    weights_path: models/reid/osnet_x0_25_msmt17.pt
```

**Pros:** Highest IDF1 (identity F1 score)  
**Cons:** Slower than botsort

---

### Configuration 5: Deep Learning Tracker
**Use Case:** When you want deep learning based tracking

```yaml
tracking:
  enabled: true
  tracker: deepocsort  # Or strongsort (all 7 trackers available!)
  reid:
    enabled: true
    weights_path: models/reid/osnet_x0_25_msmt17.pt
```

**Pros:** Deep learning based, good accuracy  
**Cons:** Slower than motion-only (12-17 Hz)

---

## Integration with Our Pipeline

### Current Architecture:
```
Video → YOLO Detection → BoxMOT Tracking → Largest BBox Selection → NPZ Output
```

### Key Integration Points:

**1. Detection Format (Input to Tracker):**
```python
# BoxMOT expects: (N, 6) array
# [x1, y1, x2, y2, confidence, class_id]
dets = np.array([
    [100, 200, 300, 400, 0.95, 0],  # bbox1: person with 95% confidence
    [150, 180, 320, 420, 0.87, 0],  # bbox2: person with 87% confidence
])
```

**2. Tracking Format (Output from Tracker):**
```python
# BoxMOT returns: (M, 8) array
# [x1, y1, x2, y2, track_id, confidence, class_id, detection_index]
tracks = np.array([
    [100, 200, 300, 400, 1, 0.95, 0, 0],  # track_id=1
    [150, 180, 320, 420, 2, 0.87, 0, 1],  # track_id=2
])
```

**3. Pipeline Output (Our Format):**
```python
# We output: (N_frames, 4) array
# [x1, y1, x2, y2] - largest bbox per frame
output = np.array([
    [100, 200, 300, 400],  # frame 0
    [105, 202, 302, 398],  # frame 1 - same track_id (smoothed)
    [110, 204, 304, 396],  # frame 2 - consistent tracking
])
```

### Benefits of Tracking:
1. **Smooth bboxes**: Less jitter between frames
2. **Occlusion handling**: Maintains ID when person temporarily hidden
3. **Better downstream**: More consistent 2D pose → Better 3D lifting

---

## CLI Reference (for manual testing)

BoxMOT also provides a CLI for testing:

```bash
# Test tracking on webcam (motion-only)
boxmot track yolov8n bytetrack --source 0

# Test tracking on video (with ReID)
boxmot track yolov8n osnet_x0_25_msmt17 botsort --source video.mp4 --save

# Show trajectories and lost tracks
boxmot track yolov8n osnet_x0_25_msmt17 botsort \
  --source video.mp4 --save --show-trajectories --show-lost
```

---

## Troubleshooting

### Issue: Tracker Not Available
**Symptom:** `ValueError: Tracker 'strongsort' not available`  
**Solution:** Use one of the 3 confirmed trackers: `botsort`, `bytetrack`, `boosttrack`

### Issue: ReID Model Download Fails
**Symptom:** ReID weights not found, but tracking still works  
**Explanation:** Automatic fallback to motion-only tracking  
**Solution:** Pre-download models or set `reid.enabled: false`

### Issue: Slow Performance
**Symptom:** Low FPS, laggy tracking  
**Solutions:**
1. Switch to `bytetrack` (fastest: 1265 Hz)
2. Disable ReID: `reid.enabled: false`
3. Export ReID model to ONNX: `boxmot export --weights ... --include onnx`
4. Reduce video resolution

### Issue: Identity Switches
**Symptom:** Track IDs change frequently, same person gets different IDs  
**Solutions:**
1. Enable ReID: `reid.enabled: true`
2. Switch to `botsort` or `boosttrack` (better for occlusions)
3. Increase detection confidence threshold (fewer false positives)

---

## Performance Benchmarks (MOT17 Dataset)

From official BoxMOT README:

| Tracker | HOTA↑ | MOTA↑ | IDF1↑ | FPS | Use Case |
|---------|-------|-------|-------|-----|----------|
| **botsort** ✅ | **69.4** | 78.2 | 81.8 | 46 | Best overall accuracy |
| **boosttrack** ✅ | 69.3 | 75.9 | **83.2** | 25 | Best identity consistency |
| strongsort ⚠️ | 68.0 | 76.2 | 80.8 | 17 | Good balance (unavailable) |
| deepocsort ⚠️ | 67.8 | 75.9 | 80.5 | 12 | Deep learning (unavailable) |
| **bytetrack** ✅ | 67.7 | 78.0 | 79.2 | **1265** | **Fastest**, motion-only |
| hybridsort ⚠️ | 67.4 | 74.1 | 79.1 | 25 | Hybrid approach (unavailable) |
| ocsort ⚠️ | 66.4 | 74.5 | 77.9 | 1483 | Fast motion-only (unavailable) |

**Metrics:**
- **HOTA** (Higher Order Tracking Accuracy): Overall tracking quality
- **MOTA** (Multiple Object Tracking Accuracy): Detection + tracking
- **IDF1** (Identity F1 Score): Identity preservation
- **FPS**: Frames per second (higher = faster)

---

## Next Steps

1. **Run API exploration script:**
   ```bash
   python snippets/explore_boxmot_api.py
   ```
   This will reveal the exact Python API structure of BoxMOT v16.0.4

2. **Test motion-only tracking:**
   ```bash
   # Edit detector.yaml: tracking.enabled=true, reid.enabled=false
   python snippets/run_detector_tracking.py --config configs/detector.yaml
   ```

3. **Test with ReID:**
   ```bash
   # Edit detector.yaml: tracking.enabled=true, reid.enabled=true
   python snippets/run_detector_tracking.py --config configs/detector.yaml
   ```

4. **Compare outputs:**
   - Detection-only: `run_detector.py` (baseline)
   - Motion tracking: `run_detector_tracking.py` (smooth bboxes)
   - Motion+ReID: `run_detector_tracking.py` (best ID consistency)

5. **Optimize (optional):**
   ```bash
   boxmot export --weights osnet_x0_25_msmt17.pt --include onnx
   # Then update config: weights_path: models/reid/osnet_x0_25_msmt17.onnx
   ```

---

## References

- **BoxMOT GitHub:** https://github.com/mikel-brostrom/boxmot
- **BoxMOT Paper:** https://doi.org/10.5281/zenodo.8132989
- **ReID Zoo:** https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO
- **Local README:** `D:\trials\unifiedpipeline\notebooks\BOXMOT_README.md`

---

**Generated from conversation analysis on December 25, 2024**
