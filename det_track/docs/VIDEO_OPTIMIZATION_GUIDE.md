# Video Processing Optimization Guide

## Current State
We've already implemented several optimizations:
- ✅ **Smart Frame Selection (Stage 6b)**: Select only 3-9 frames instead of seeking 30+ times
- ✅ **Sequential Access**: Sort frames by number before seeking (avoids random seeks)
- ✅ **Removed Downscaling**: Process at 1920×1080 directly (19% speedup)

## Video Encoding Bottlenecks

### 1. **Frame Seeking** (Potential Bottleneck)
- **Problem**: `cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)` can be slow for non-keyframe seeks
- **Cause**: Video codec has to decode from last keyframe to reach target frame

**Current Optimization:**
```python
# Sort frame_plan by frame number for sequential access (CRITICAL for speed)
frame_plan_sorted = sorted(frame_plan, key=lambda x: x[0])
```

**Why this works:**
- Sequential reads (frame N → N+1) just call `cap.read()` (fast)
- Random seeks require decoding from keyframes (slow)

### 2. **Container Format**
- **Current**: MP4 (H.264) - standard but keyframe-heavy
- **Issue**: H.264 keyframes typically every 2-5 seconds; non-keyframe seeks trigger full decoding
- **Better option**: MOV or ProRes for intra-coded (every frame is keyframe) = fast seeks

### 3. **Memory Usage**
- **Current**: Single frame buffered at a time (good)
- **Pre-caching**: Could load 3-5 frames ahead to reduce wait for I/O

## Optimization Opportunities

### Tier 1: Easy Wins (No Code Changes)
1. **Ensure video is H.264 with low keyframe interval**
   - Command: `ffmpeg -i input.mp4 -g 1 -vf scale=1920:1080 -c:v libx264 output.mp4`
   - `-g 1` = keyframe every frame (slow to encode but instant seeks)

2. **Use hardware video decoder if available**
   - OpenCV can use NVIDIA CUVID, Intel MFX, or Apple VideoToolbox
   - Set: `cap.set(cv2.CAP_PROP_AUTOFOCUS, cv2.CAP_PROP_HW_ACCELERATION)`

### Tier 2: Medium Effort
1. **Frame Caching with Look-ahead**
   - Cache next 5 frames while processing current frame
   - Overlap I/O with computation
   - Code location: `stage6b_create_selection_grid.py:extract_crops()`

2. **Batch Seeks**
   - Group seeks by proximity before extracting
   - Current implementation already does this via `sorted(frame_plan)`

3. **Multi-threaded Frame Loading**
   - Background thread loads frames while main thread processes
   - Skip for now (complexity vs benefit)

### Tier 3: Heavy Lift
1. **Pre-extract Frames to Disk**
   - Extract all frames once as PNG/JPG
   - Then load from disk (random access fast)
   - Trade: 2-3GB disk space for 20x faster seeks

2. **Use Video Codec Library Directly**
   - ffmpeg/libav with direct memory access (faster than OpenCV)
   - Skip for now (complexity)



## Files to Monitor
- `det_track/stage1_detect.py`: Uses sequential reads (already optimal)
- `det_track/stage2_track.py`: Minimal video reads (optional - mostly motion-based)
- `det_track/stage6b_create_selection_grid.py`: Heavy seeking (biggest opportunity)

## Summary Table

| Optimization | Effort | Speed Gain | Current Status |
|---|---|---|---|
| Smart frame selection | Done | 3-9 frames instead of 30+ | ✅ Implemented |
| Sequential seek ordering | Done | 2-5x via batching | ✅ Implemented |
| Remove downscaling | Done | 19% YOLO speedup | ✅ Just completed |
| Video codec (keyframe every frame) | 5 min | **5-10x seek speedup** | ⚠️ Not done |
| Frame caching/pre-load | 2 hours | 20-30% via I/O overlap | 📋 Proposed |
| Hardware decoding (GPU) | 3 hours | 5x decode speedup | 📋 Proposed |

**Recommendation**: Fix video codec first (5 min, 5-10x gain), then evaluate if caching needed.
