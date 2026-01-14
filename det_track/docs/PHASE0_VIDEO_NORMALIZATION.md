# Phase 0: Video Ingestion & Normalization

**Date:** January 14, 2026  
**Status:** Design/Proposal Stage  
**Priority:** Critical - Foundation for production system  
**Related:** This is Phase 0 (runs BEFORE Phase 1) - discovered after Phase 3 analysis

---

## 🎯 Executive Summary

**Problem:** Current pipeline assumes well-formatted videos. Reality: users upload anything.  
**Solution:** Add Stage 0 normalization layer before YOLO detection.  
**Benefits:** Robust input handling, consistent behavior, fast seeking, predictable performance.

---

## 📋 Problem Statement

### Current Weakness

```python
# Stage 1 currently does this:
video = cv2.VideoCapture(user_uploaded_video)  # 🚨 Dangerous!

# What if:
- 4K @ 60fps? (4× more frames, GPU OOM)
- Variable frame rate? (ByteTrack Δt breaks)
- H.265 with GOP=250? (slow seeking)
- 10 minutes long? (hours to process)
- Exotic codec? (decoder bugs)
```

### Real-World Upload Risks

| Risk | Impact | Example |
|------|--------|---------|
| **High resolution** | GPU OOM, slow detection | 4K video → 8× pixels vs 1080p |
| **Variable FPS** | Tracking breaks | VFR video → inconsistent Δt |
| **Long GOP** | Slow seeking | GOP=250 → decode 250 frames for 1 frame |
| **Exotic formats** | Decoder bugs | Old AVI, MOV variants |
| **Large files** | Cost, latency | 10 min video = 15,000 frames |

**The insight:**
> "The moment you said 'users might upload any kind of video', you crossed from researcher to platform architect."

---

## 🎬 Proposed Solution

### New Pipeline Architecture

```
User Upload (any format, any resolution, any FPS)
    ↓
[STAGE 0: NORMALIZE] ← NEW STAGE
    - Validate (reject if too large/long)
    - Transcode to canonical format
    - Standardize FPS, resolution, codec
    ↓
Canonical Video (MP4, H.264, 25 FPS, ≤1080p)
    ↓
Stage 1: YOLO Detection
    ↓
... rest of pipeline (unchanged)
```

### Stage 0 Responsibilities

#### 1. Validation (Reject Bad Inputs)
```python
def validate_video(filepath):
    info = get_video_info(filepath)
    
    # Check duration
    if info['duration'] > 120:  # 2 minutes
        return False, "Video too long (max 2 minutes)"
    
    # Check file size
    if info['filesize'] > 200 * 1024 * 1024:  # 200 MB
        return False, "File too large (max 200 MB)"
    
    # Check if readable
    if not info['is_valid']:
        return False, "Corrupt or unsupported format"
    
    return True, "OK"
```

#### 2. Format Normalization
```bash
ffmpeg -i user_upload.* \
  -c:v libx264 \
  -preset veryfast \
  -profile:v main \
  -pix_fmt yuv420p \
  canonical.mp4
```
- **Output:** Always MP4 + H.264
- **Benefit:** Browser compatible, stable decoder

#### 3. Temporal Normalization (FPS)
```bash
ffmpeg -i input.mp4 \
  -r 25 \
  -vsync cfr \
  canonical.mp4
```
- **Output:** Constant 25 FPS
- **Benefit:** ByteTrack Δt = 0.04s (consistent), Kalman filters stable

#### 4. Spatial Normalization (Resolution)
```bash
ffmpeg -i input.mp4 \
  -vf "scale='min(1920,iw)':'min(1080,ih)':force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2" \
  canonical.mp4
```
- **Output:** Max 1920×1080, padded to 16:9, aspect preserved
- **Benefit:** GPU memory predictable, YOLO batch size stable

#### 5. Keyframe Structure (Seeking Optimization)

**Option A: Small GOP (Recommended)**
```bash
ffmpeg -i input.mp4 \
  -x264-params keyint=30:scenecut=0 \
  canonical.mp4
```
- Keyframe every 30 frames (~1.2 seconds @ 25 FPS)
- File size: +50% vs normal GOP
- Seeking: Fast (< 1 second)

**Option B: All-I (Maximum Speed)**
```bash
ffmpeg -i input.mp4 \
  -c:v libx264 \
  -x264-params keyint=1 \
  -preset veryfast \
  allI.mp4
```
- Keyframe every frame (GOP=1)
- File size: +150% vs normal GOP
- Seeking: Instant (O(1), no GOP decoding)

---

## 📊 Performance Analysis

### Re-encoding Time (CPU, veryfast preset)

| Video Length | Resolution | Time to Normalize |
|-------------|------------|-------------------|
| 30 sec | 1080p | 0.3-0.6s |
| 1 min | 1080p | 0.6-1.2s |
| 2 min | 1080p | 1.2-2.4s |

**With GPU (NVENC):** Near real-time or faster (0.5-1.0× video length)

### File Size Impact

| GOP Setting | File Size | Seeking Speed |
|------------|-----------|---------------|
| Normal (GOP=250) | 100 MB | Slow (decode chain) |
| Small GOP (keyint=30) | 150 MB | Fast (<1s) |
| All-I (keyint=1) | 250 MB | Instant (O(1)) |
| MJPEG | 300 MB | Instant (O(1)) |

**Recommendation:** Start with small GOP (keyint=30)
- Good balance of size vs speed
- Can upgrade to All-I later if seeking is bottleneck

---

## 🛡️ Upload Restrictions (Product Design)

### Recommended Limits

**Hard Limits (Reject Upload):**
- **Max file size:** 200 MB
- **Max duration:** 2 minutes (120 seconds)
- **Max resolution:** Auto-downscale to 1080p (don't reject)
- **Max frame rate:** Auto-clamp to 30 FPS (don't reject)

**Soft Warnings:**
- File size > 100 MB: "Large file may take longer to process"
- Duration > 60 sec: "Longer videos may incur additional charges"

**Accepted Formats (UI Level):**
- Accept: MP4, MOV, AVI, WebM, MKV, FLV (anything ffmpeg can read)
- **Backend:** Convert everything to canonical format
- **Don't rely on user compliance** - always normalize

### Rationale for Limits

| Limit | Reason |
|-------|--------|
| 200 MB | Prevent abuse, control storage costs |
| 2 minutes | 3,000 frames @ 25fps = reasonable YOLO cost |
| 1080p | Fits in 8GB GPU with batch processing |
| 25-30 FPS | Balance of smoothness vs processing cost |

**Tiered Limits (Future):**
- Free tier: 60 sec, 100 MB
- Paid tier: 5 min, 500 MB
- Enterprise: Custom

---

## 🔧 Implementation Plan

### Stage 0: Video Normalization Module

**New file: `stage0_normalize_video.py`**

```python
def validate_video(filepath, config):
    """Check if video meets requirements."""
    # Get video info (duration, size, resolution, codec)
    # Compare against limits
    # Return: (is_valid, error_message)

def normalize_video(input_path, output_path, config):
    """Convert to canonical format."""
    # Build ffmpeg command
    # Execute with progress callback
    # Verify output
    # Return: canonical_video_path

def get_video_info(filepath):
    """Extract metadata using ffprobe."""
    # Duration, resolution, FPS, codec, size
    # Return: dict

def cleanup_temp_files(paths):
    """Remove temporary files."""
    pass
```

**Configuration: `pipeline_config.yaml`**
```yaml
stage0_normalize:
  enabled: true  # Can disable for development
  
  validation:
    max_duration_seconds: 120
    max_filesize_mb: 200
    reject_invalid: true  # or auto-fix
  
  normalization:
    target_fps: 25
    max_resolution: [1920, 1080]
    maintain_aspect: true
    pad_to_16_9: true
    
  encoding:
    codec: libx264
    preset: veryfast  # veryfast, fast, medium
    profile: main
    pix_fmt: yuv420p
    keyframe_interval: 30  # 1=All-I, 30=small GOP
    
  output:
    canonical_video_file: ${outputs_dir}/${current_video}/canonical.mp4
    keep_original: true  # Store both original and canonical
```

### Integration with Orchestrator

**Update `run_pipeline.py`:**
```python
# Add Stage 0 before Stage 1
if config['stage0_normalize']['enabled']:
    print("Stage 0: Normalizing video...")
    canonical_video = run_stage0_normalize(
        input_video=user_uploaded_video,
        config=config
    )
    video_for_detection = canonical_video
else:
    video_for_detection = user_uploaded_video

# Pass canonical video to Stage 1
run_stage1_detection(video_for_detection, config)
```

### Dependencies

```bash
pip install ffmpeg-python  # Python wrapper for ffmpeg
# OR use subprocess with ffmpeg CLI (already installed)
```

**ffmpeg must be installed:**
```bash
# Ubuntu/Debian
apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
choco install ffmpeg
# OR download from ffmpeg.org
```

---

## 🎯 Benefits

### For Development
- ✅ **Reproducible results** (consistent input format)
- ✅ **Easier debugging** (standardized pipeline)
- ✅ **Predictable performance** (no surprises)

### For Production
- ✅ **Controlled costs** (capped processing time)
- ✅ **Better UX** (fast turnaround)
- ✅ **Abuse prevention** (limits enforced)
- ✅ **Browser compatibility** (MP4 + H.264 universal)

### For Algorithms
- ✅ **ByteTrack Δt consistent** (tracking more accurate)
- ✅ **Kalman filters stable** (smooth trajectories)
- ✅ **Speed estimates meaningful** (pixels/frame → m/s)
- ✅ **GPU memory predictable** (no OOM crashes)

---

## 🔗 Integration with Phase 3 (LMDB)

### Complementary Optimizations

**Phase 0 (Normalization):**
- Solves: Input variability, fast seeking
- Output: Canonical video (small-GOP H.264)

**Phase 3 (LMDB Crops):**
- Solves: Fast person-level access
- Output: Compressed crops database

**Together:**
```
Phase 0: User video → Canonical video (fast seeking)
    ↓
Phase 1: YOLO detection → LMDB crops (fast access)
    ↓
Phase 3: WebP generation from LMDB (no seeking!)

Combined benefit: Robust + Fast
```

### Recommended Implementation Order

**Option 1: Sequential**
1. Phase 0A (validation only) - 2-3 hours
2. Phase 3 (LMDB crops) - 8-12 hours
3. Phase 0B (full normalization) - 4-6 hours

**Option 2: Parallel (if 2 developers)**
- Developer A: Phase 0
- Developer B: Phase 3
- Integrate after both complete

**Option 3: Gradual**
1. Phase 0A (validation) - Quick win
2. Test with real user uploads
3. Phase 3 (LMDB) - Performance gain
4. Phase 0B (normalization) - Complete robustness

---

## 📈 Expected Results

### Before (Current)

```
User uploads 4K @ 60fps, 5 min video
    ↓
Pipeline tries to process (FAILS or takes hours)
OR
User uploads VFR video
    ↓
ByteTrack Δt inconsistent → bad tracking
```

### After (Phase 0 + Phase 3)

```
User uploads anything
    ↓
Stage 0: Validate (reject if > 2 min)
Stage 0: Normalize (2s) → 1080p @ 25fps
    ↓
Stage 1: YOLO detection → LMDB crops (50s)
    ↓
Pipeline completes in ~67s (predictable!)
    ↓
Results: Consistent tracking, fast access
```

### Performance Summary

| Metric | Current | After Phase 0+3 | Improvement |
|--------|---------|-----------------|-------------|
| Accepts any format | ❌ | ✅ | Robust |
| Consistent FPS | ❌ | ✅ | Stable tracking |
| Fast seeking | ❌ | ✅ | Small-GOP |
| Crop storage | 812 MB | 100 MB | 85% smaller |
| Pipeline time | 74.7s | 67s | 10% faster |
| Predictable | ❌ | ✅ | Production-ready |

---

## ❓ Open Questions

1. **Keyframe interval?**
   - Small GOP (30): +50% size, fast seeking
   - All-I (1): +150% size, instant seeking
   - **Recommendation:** Start with 30

2. **Resolution target?**
   - 1080p: Best quality, slower
   - 720p: Faster, good enough
   - **Recommendation:** Configurable, default 1080p

3. **Upload limits?**
   - 2 min might be too short
   - Tiered limits by user plan?
   - **Recommendation:** Start 2 min, adjust based on usage

4. **Stage 0 always-on?**
   - Development: Optional (skip normalization)
   - Production: Mandatory (always normalize)
   - **Recommendation:** Config flag `enabled: true/false`

5. **Storage of both videos?**
   - Keep original + canonical (debugging)
   - Delete original after normalization (save space)
   - **Recommendation:** Configurable, default keep both

---

## ✅ Success Criteria

**Phase 0 will be considered successful if:**
1. ✅ Accepts any video format (MP4, MOV, AVI, WebM, etc.)
2. ✅ Rejects videos > 2 min or > 200 MB
3. ✅ Normalizes to 1080p @ 25 FPS in < 2s
4. ✅ ByteTrack Δt consistent (±1ms)
5. ✅ No GPU OOM errors
6. ✅ Browser can play canonical video
7. ✅ Pipeline time increase < 2s (normalization overhead acceptable)

---

## 🚀 Next Steps

1. **Review this design** with stakeholders
2. **Approve implementation** approach
3. **Prioritize**: Phase 0A (validation) vs full normalization
4. **Coordinate** with Phase 3 (LMDB) implementation
5. **Test** with real user uploads

---

**Status:** Awaiting approval to proceed with implementation
