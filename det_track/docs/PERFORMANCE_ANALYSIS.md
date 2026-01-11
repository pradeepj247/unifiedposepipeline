# Pipeline Performance Analysis

**Test Environment**: Google Colab (T4 GPU)  
**Video**: kohli_nets.mp4 (2027 frames, 1920×1080, 25 fps)  
**Test Date**: January 10, 2026  
**Total Execution Time**: 151.78 seconds (2m 32s)

---

## 📊 Stage-by-Stage Performance

| Stage | Name | Time (s) | % Total | Input | Output | FPS/Throughput |
|-------|------|----------|---------|-------|--------|----------------|
| **1** | YOLO Detection | **52.51** | **34.6%** | 2027 frames | 8782 detections | **46.5 FPS** |
| **2** | ByteTrack Tracking | 9.35 | 6.2% | 8782 detections | 49 tracklets | **582.9 FPS** |
| **3** | Tracklet Analysis | 0.23 | 0.2% | 49 tracklets | stats + 12 candidates | - |
| **4** | Load Crops Cache | 0.80 | 0.5% | 823.7 MB cache | In-memory | 10.2 MB/s |
| **5** | Canonical Grouping | 0.39 | 0.3% | 49 tracklets | 46 persons | - |
| **6** | Enrich Crops (HDF5) | **50.46** | **33.2%** | 8661 crops | 731.74 MB HDF5 | **14.5 MB/s** |
| **7** | Rank Persons | 0.21 | 0.1% | 46 persons | 1 primary person | - |
| **9** | Output Video | **33.41** | **22.0%** | 2027 frames | 80.49 MB MP4 | **61.2 FPS** |
| **11** | Generate WebPs | 3.46 | 2.3% | 8661 crops | 10 WebPs (1.51 MB) | - |
| **10** | HTML Report | 0.94 | 0.6% | WebPs + metadata | 2.35 MB HTML | - |
| | | | | | | |
| **TOTAL** | **Full Pipeline** | **151.78** | **100%** | - | - | - |

---

## 🔴 Performance Bottlenecks (Top 3)

### 1️⃣ **Stage 1: YOLO Detection** (52.51s = 34.6%)
**Problem**: GPU inference is inherently slow for real-time processing
- Detection: 43.57s @ 46.5 FPS
- Crop extraction: 41.84s
- Model loading + overhead: ~7.1s

**Optimization Options**:
- ✅ **Use TensorRT quantized model** (2-3x speedup possible) - Change `model_path` to `.engine` file
- ✅ **Reduce detection confidence threshold** (skip low-confidence frames)
- ✅ **Batch processing** (not currently implemented)
- ⚠️ **Use smaller YOLO model** (yolov8n instead of yolov8s) - Trade accuracy for speed

---

### 2️⃣ **Stage 6: Enrich Crops (HDF5)** (50.46s = 33.2%)
**Problem**: Writing 8661 crops to HDF5 with gzip compression is slow
- 8661 crops stored across 46 persons
- File size: 731.74 MB (14.5 MB/s write speed)
- Gzip compression overhead

**Optimization Options**:
- ✅ **Reduce compression level** (gzip → no compression, or lz4)
  ```yaml
  stage6:
    enrichment:
      compression: none  # Instead of: gzip
  ```
- ✅ **Parallel HDF5 writing** (not currently implemented)
- ✅ **Skip enrichment if not needed** (only needed for Stage 10)
  ```yaml
  pipeline:
    stages:
      stage6: false  # Skip HDF5, use crops_cache.pkl directly
  ```

---

### 3️⃣ **Stage 9: Output Video** (33.41s = 22.0%)
**Problem**: Video encoding (H.264) is CPU-heavy
- 2027 frames @ 22.5 FPS output
- 16 person bounding boxes drawn per frame
- MP4 encoding overhead

**Optimization Options**:
- ✅ **Skip visualization** (only needed for review)
  ```yaml
  pipeline:
    stages:
      stage9: false  # Skip output video
  ```
- ✅ **Reduce output resolution** (720×405 → lower)
- ✅ **Use faster codec** (replace H.264 with ProRes or rawvideo)
- ⚠️ **GPU-accelerated encoding** (requires NVENC setup)

---

## ✅ Fast Stages (Highly Optimized)

| Stage | Time | Why It's Fast |
|-------|------|---------------|
| Stage 2: ByteTrack | 9.35s | Pure Python, vectorized operations |
| Stage 3: Analysis | 0.23s | Simple statistics, O(n) |
| Stage 4: Load Cache | 0.80s | Pickle deserialization, well-optimized |
| Stage 5: Grouping | 0.39s | Heuristic matching, only 49 tracklets |
| Stage 7: Ranking | 0.21s | Weighted scoring, only 46 persons |
| Stage 10: HTML | 0.94s | Simple template rendering |
| Stage 11: WebPs | 3.46s | PIL/Pillow optimized, only 10 files |

---

## 🎯 Recommended Optimization Strategy

### **Option A: Minimal Changes (Fastest)**
```yaml
# In pipeline_config.yaml
pipeline:
  stages:
    stage9: false   # Skip video visualization
    stage6: false   # Skip HDF5 enrichment
# Expected time: ~62s (60% reduction)
```
- Output: HTML report + WebPs (all you need)
- Savings: 33.41s + 50.46s = **83.87s**

### **Option B: Medium Effort (Good Balance)**
```yaml
# Use TensorRT model + reduce compression
stage1:
  detector:
    model_path: ${models_dir}/yolo/yolov8s.engine  # TensorRT quantized

stage6:
  enrichment:
    compression: none  # Remove gzip overhead

pipeline:
  stages:
    stage9: false  # Skip video
# Expected time: ~70s (50% reduction)
```
- Setup: Convert YOLO to TensorRT once (5 min)
- Savings: ~82s

### **Option C: Full Power (Production)**
- Implement GPU-accelerated video encoding
- Add parallel HDF5 writing
- Use batch detection
- Expected time: ~40s (75% reduction)

---

## 📈 Scaling Analysis

**Current**: 2027 frames = 151.78s = **0.075 s/frame**

| Scenario | Est. Time | Notes |
|----------|-----------|-------|
| 10-min video (15,000 frames) | ~18.75 min | Linear scaling |
| With Stage 9/6 disabled | ~7.5 min | **Recommended for production** |
| With TensorRT + no compression | ~10.5 min | Best real-world setup |

---

## 🚀 Quick Win: Disable Stages 9 & 6

The easiest optimization is to skip output video and HDF5 enrichment since they're not essential for the interactive HTML report:

**Current Output**:
- `person_selection_report.html` (2.35 MB) ← Keep
- `top_persons_visualization.mp4` (80.49 MB) ← Optional
- `crops_enriched.h5` (731.74 MB) ← Optional

**Modify config**:
```yaml
pipeline:
  stages:
    stage6: false   # No HDF5 enrichment (Stage 10 uses crops_cache instead)
    stage9: false   # No video visualization
```

**Result**: 151.78s → **~67s** (56% faster) ⚡

---

## 📝 Summary Table

| Category | Stage | Time | Recommendation |
|----------|-------|------|-----------------|
| 🔴 **Critical** | Stage 1 (YOLO) | 52.51s | Use TensorRT or smaller model |
| 🔴 **Critical** | Stage 6 (HDF5) | 50.46s | Disable or remove compression |
| 🟠 **High** | Stage 9 (Video) | 33.41s | **DISABLE** (not essential) |
| 🟢 **Good** | Others | ~12s | No changes needed |

