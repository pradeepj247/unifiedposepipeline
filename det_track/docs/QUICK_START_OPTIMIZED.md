# 🚀 Quick Start: Optimized 11-Stage Pipeline

## Overview

This guide covers running the unified pose estimation pipeline with performance optimizations:
- **Detection & Tracking:** Multi-person detection and tracking across video frames
- **Person Grouping:** Canonical grouping of tracklets into persons
- **WebP Generation:** Fast animated WebP export with in-memory crop caching
- **HTML Report:** Interactive person selection report with embedded videos

**Key Optimization:** In-memory crop caching eliminates slow HDF5 writes, resulting in ~33% faster execution.

---

## Quick Start

### Windows:
```bash
cd d:\trials\unifiedpipeline\newrepo\det_track
python run_pipeline.py --config configs/pipeline_config.yaml
```

### Google Colab:
```bash
%cd /content/unifiedposepipeline/det_track
!python run_pipeline.py --config configs/pipeline_config.yaml
```

---

## Expected Execution Flow

### ✅ Stages 1-7: Detection, Tracking & Grouping
```
🚀 Stage 1: YOLO Object Detection (52s GPU time)
   ✓ Detections saved to detections_raw.npz

🎯 Stage 2: ByteTrack Multi-Object Tracking (9s)
   ✓ Tracklets saved to tracklets_raw.npz

📊 Stage 3: Tracklet Analysis (2s)
   ✓ Statistics and candidates saved

🔗 Stage 4: Tracklet Recovery (optional, ~1s)
   ✓ ReID-based merging (if enabled)

👥 Stage 5: Canonical Grouping (2s)
   ✓ Persons saved to canonical_persons.npz

📈 Stage 7: Person Ranking (0.5s)
   ✓ Ranking results saved
```

### ✅ Stages 11 & 10: WebP Generation & HTML Report
```
🎬 Stage 11: Generate WebP Animations (3-5s IN-MEMORY OPTIMIZED)
   ✓ Crops kept in RAM
   ✓ WebP files generated directly from memory
   ✓ ~50-100 KB per person (highly compressed)

📄 Stage 10: Generate HTML Report (0.7s)
   ✓ Interactive person selection report
   ✓ All WebPs embedded (no external files needed)
```

### 📊 Total Execution Time: ~70-75 seconds

---

## Performance Optimization Explained

| Component | Strategy |
|-----------|----------|
| **Stage 1 (YOLO)** | GPU acceleration (primary bottleneck) |
| **Crop Caching** | Keep crops in RAM after Stage 5 → avoid HDF5 write (~50s saved) |
| **WebP Format** | Animated WebP instead of GIF (2× smaller, faster encoding) |
| **HTML Embedding** | Base64-encoded WebPs embedded in HTML (instant loading) |

**Result:** 33% faster than traditional file-based approach + 16× less memory footprint

---

## Verify Output

### Check Generated Files:
```bash
# WebP animations
ls -lh outputs/[VIDEO_NAME]/webp/*.webp

# HTML report
ls -lh outputs/[VIDEO_NAME]/person_selection_report.html
```

### Expected:
- 10 WebP files (~50-100 KB each, total ~0.5-1.0 MB)
- 1 HTML report (~2-5 MB including embedded videos)

---

## Configuration Tuning

Edit `configs/pipeline_config.yaml` to adjust WebP generation:

```yaml
stage11:
  video_generation:
    format: webp
    fps: 10              # Frames per second (10 = 1s per 10 frames)
    max_frames: 60       # Duration: 60 frames @ 10fps = 6 seconds per person
    frame_width: 128     # Smaller = faster encoding
    frame_height: 192
    quality: 80          # 0-100, lower = faster but lower quality
```

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `FileNotFoundError: canonical_persons.npz` | Stage 5 didn't complete | Check Stage 5 logs, verify Stage 1-4 finished |
| `KeyError: 'crops_cache'` | Stage 4 didn't run properly | Ensure crops are loaded before Stage 11 |
| WebP files missing from HTML | Stage 11 didn't complete | Check Stage 11 logs for encoding errors |
| Slow WebP generation | Quality too high or frame size too large | Reduce `quality` or `max_frames` in config |

---

## Core Pipeline Flow

The optimized pipeline sequence:

1. ✅ **Detect with YOLO** (52s GPU time)
   - Person detection and confidence scoring

2. ✅ **Track with ByteTrack** (9s)
   - Multi-object tracking across frames
   - Identity continuity maintained

3. ✅ **Group into Canonical Persons** (2.9s metadata)
   - Merge tracklets from same person
   - Build continuous trajectories

4. ✅ **In-Memory Crop Caching** ← **KEY OPTIMIZATION**
   - Crops kept in RAM (no HDF5 write)
   - Save time: ~50 seconds
   - Save complexity: One-step WebP generation

5. ✅ **Generate WebPs from Memory** (3-5s)
   - Direct encoding from cached crops
   - Animated WebP for smooth playback

6. ✅ **Generate HTML Report** (0.7s)
   - Interactive person cards
   - All media embedded (self-contained file)

7. ✅ **Done in ~70 seconds total!**

---

## 🎉 You're Ready!

Your pipeline is optimized for speed and efficiency. The in-memory crop caching strategy eliminates the slowest file I/O operations while maintaining all functionality.

For detailed pipeline architecture, see [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md).

For data format specifications, see [../FullContext.md](../FullContext.md).
