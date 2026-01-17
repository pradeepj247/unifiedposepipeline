# Stage 0: Video Normalization & Validation

**Implementation**: `stage0_normalize_video.py`

## Purpose
**This is the FIRST stage** - runs BEFORE any detection or tracking. Ensures all videos are in a consistent, optimal format for processing.

## When It Runs
Always first (Stage 0) - required before Stage 1 (detection).

## Inputs
- Raw video file from user: `${video_dir}${video_file}`
- Example: `/content/unifiedposepipeline/demo_data/videos/kohli_nets.mp4`
- Accepts any format: MP4, MOV, AVI, MKV, etc.
- Any resolution, FPS, codec

## Outputs
- `canonical_video.mp4`: Normalized video in canonical format
- `stage0_timing.json`: Timing and validation metrics
- `stage0_metadata.json`: Video properties (resolution, FPS, codec)

## Processing Flow

```
Input Video (any format)
    ↓
[Validation Checks]
    ├─→ Resolution: min 640×480, max 1920×1080
    ├─→ Duration: min 5 frames
    ├─→ File size: check reasonable limits
    └─→ Codec: validate readable by OpenCV
    ↓
[Check if already canonical]
    ├─ Format: MP4
    ├─ Codec: H.264
    ├─ Resolution: ≤1080p
    ├─ FPS: Constant frame rate
    └─ GOP structure: Keyframe interval reasonable
    ↓
[Decision]
    ├─→ Already canonical: Create symlink (instant)
    └─→ Needs re-encoding: FFmpeg normalization
        ├─→ Convert to MP4
        ├─→ Re-encode with H.264
        ├─→ Set constant FPS
        ├─→ Set GOP=30 (keyframe every 30 frames)
        └─→ Optimize for seeking
    ↓
Output: canonical_video.mp4
```

## Canonical Format Specification

| Property | Value | Rationale |
|----------|-------|-----------|
| **Container** | MP4 | Universal compatibility |
| **Video Codec** | H.264 | Best hardware support, fast decode |
| **Resolution** | ≤1920×1080 | Limits memory usage |
| **FPS** | Constant | Enables frame-accurate seeking |
| **GOP Size** | 30 frames | Balance seeking speed vs compression |
| **Pixel Format** | yuv420p | Standard for compatibility |

## Configuration

```yaml
stage0_normalize:
  enabled: true
  
  limits:
    min_width: 640           # Minimum video width
    min_height: 480          # Minimum video height
    max_width: 1920          # Maximum video width
    max_height: 1080         # Maximum video height
    min_duration_seconds: 1  # Minimum 1 second
    max_duration_seconds: 300  # Maximum 5 minutes
  
  ffmpeg:
    preset: medium           # Encoding speed (ultrafast to veryslow)
    bitrate: 8000k           # Target bitrate
    crf: 23                  # Constant Rate Factor (quality)
    keyframe_interval: 30    # GOP size
    pix_fmt: yuv420p         # Pixel format
  
  input:
    video_file: ${video_dir}${video_file}
  
  output:
    canonical_video_file: ${outputs_dir}/${current_video}/canonical_video.mp4
    timing_file: ${outputs_dir}/${current_video}/stage0_timing.json
    symlink_if_canonical: true  # Create symlink if already canonical
```

## Performance

| Scenario | Time | Notes |
|----------|------|-------|
| Already canonical | <1s | Just create symlink |
| Re-encode (81s video) | 15-25s | Depends on resolution |
| High-res (4K) downscale | 30-40s | Resizing overhead |

**Bottleneck**: FFmpeg encoding (CPU-bound on most systems)

## Validation Checks

### Pre-Processing Checks
```python
# Resolution
if width < 640 or height < 480:
    raise ValueError("Video too small")
if width > 1920 or height > 1080:
    print("Will downscale to 1080p")

# Duration
if num_frames < 30:  # ~1 second at 30fps
    raise ValueError("Video too short")
if num_frames > 18000:  # 5 minutes at 60fps
    print("Warning: Long video")

# Codec
if codec not in ['h264', 'hevc']:
    print("Will re-encode to H.264")
```

## Why This Stage Exists

### Problem Without Normalization
- **Variable FPS**: ByteTrack fails with frame timing inconsistencies
- **Poor GOP structure**: Slow seeking, wasted time in Stage 3c
- **Unsupported codecs**: OpenCV fails to decode some formats
- **High resolution**: OOM errors, slow processing

### Solution With Normalization
- ✅ Consistent format across all videos
- ✅ Fast seeking (Stage 3c benefits)
- ✅ Predictable memory usage
- ✅ No codec surprises

## Output Files

### canonical_video.mp4
Standard MP4 with H.264 video. Used by all downstream stages.

### stage0_timing.json
```json
{
  "validation_time": 0.12,
  "encoding_time": 18.45,
  "total_time": 18.57,
  "was_already_canonical": false
}
```

### stage0_metadata.json
```json
{
  "input": {
    "path": "/content/.../kohli_nets.mp4",
    "resolution": "1920x1080",
    "fps": 25.0,
    "codec": "h264",
    "duration_seconds": 81.06
  },
  "output": {
    "path": "/content/.../canonical_video.mp4",
    "resolution": "1920x1080",
    "fps": 25.0,
    "gop_size": 30
  },
  "action": "re-encoded"
}
```

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Video too small" | Resolution <640×480 | Upscale or reject |
| "FFmpeg not found" | Missing dependency | `apt install ffmpeg` |
| Slow re-encoding | CPU bottleneck | Use faster preset or GPU encoding |
| Large output file | Bitrate too high | Reduce to 4000k-6000k |

## Key Design Decision

**Why separate normalization stage?**
- **Consistency**: One-time cost ensures all stages see same format
- **Debugging**: Canonical video can be inspected independently
- **Caching**: Re-run detection without re-normalizing
- **User feedback**: Validate video before heavy processing

---

**Related**: [Back to Master](README_MASTER.md)
