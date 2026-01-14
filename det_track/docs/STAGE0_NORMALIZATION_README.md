# Stage 0: Video Normalization & Validation

## Overview
Stage 0 is the **foundation layer** of the pipeline - it runs **BEFORE** YOLO detection to ensure all input videos are in a consistent, optimal format for processing.

## Purpose
- **Validate** user uploads (size, duration, format)
- **Normalize** to canonical format (MP4, H.264, constant FPS, optimal GOP)
- **Enable** fast seeking and consistent tracking
- **Prevent** issues like OOM, tracking breaks, slow processing

## What It Does

### 1. Metadata Extraction
Uses `ffprobe` to extract video metadata:
- Resolution (width x height)
- FPS (frames per second)
- Duration (seconds)
- Codec (H.264, HEVC, etc.)
- Pixel format (yuv420p, etc.)
- Variable frame rate (VFR) detection

### 2. Validation
Checks video against limits (configured in `pipeline_config.yaml`):
- ✅ **Duration**: Max 120 seconds (2 minutes)
- ✅ **File size**: Max 200 MB
- ✅ **Resolution**: Max 1920x1080
- ⚠️ **FPS**: Warns if < 10 or > 60
- ⚠️ **Codec**: Warns if unusual codec

### 3. Normalization Decision
Determines if re-encoding is needed based on:
- Resolution > 1920x1080 → **Downscale**
- FPS != 25 → **Resample** to constant 25 FPS
- Variable FPS → **Convert** to constant FPS
- Codec != H.264 → **Transcode**
- GOP structure unknown → **Normalize**

### 4. Normalization (if needed)
Uses `ffmpeg` to transcode to canonical format:
- **Container**: MP4
- **Codec**: H.264 (libx264)
- **Profile**: main
- **Pixel format**: yuv420p
- **FPS**: 25 constant (CFR)
- **GOP**: Keyframe every 30 frames (1.2s at 25fps)
- **Audio**: Removed (not needed for pose estimation)
- **Preset**: veryfast (for speed)

### 5. Output
- **Canonical video**: `outputs/<video_name>/canonical_video.mp4`
- **Timing JSON**: `outputs/<video_name>/stage0_timing.json`
- **Smart**: If already canonical, creates symlink/copy (no re-encoding)

## Configuration

In `det_track/configs/pipeline_config.yaml`:

```yaml
stage0_normalize:
  enabled: true  # Set false to skip normalization
  
  limits:
    max_duration_seconds: 120        # Max video length (2 minutes)
    max_filesize_mb: 200             # Max file size (200 MB)
    max_resolution: [1920, 1080]     # Max width and height
  
  normalization:
    target_fps: 25                   # Constant FPS
    force_constant_fps: true         # Convert VFR to CFR
  
  encoding:
    codec: libx264                   # H.264 codec
    preset: veryfast                 # Encoding speed
    profile: main                    # H.264 profile
    pix_fmt: yuv420p                 # Pixel format
    keyframe_interval: 30            # GOP size (30 frames)
  
  output:
    canonical_video_file: ${outputs_dir}/${current_video}/canonical_video.mp4
    timing_file: ${outputs_dir}/${current_video}/stage0_timing.json
    symlink_if_canonical: true       # Skip re-encoding if already good
```

## Usage

### Run Stage 0 alone:
```bash
python stage0_normalize_video.py --config configs/pipeline_config.yaml
```

### Run full pipeline (includes Stage 0):
```bash
python run_pipeline.py --config configs/pipeline_config.yaml
```

### Run specific stages:
```bash
# Run Stage 0 then Stage 1 (detection)
python run_pipeline.py --config configs/pipeline_config.yaml --stages 0,1
```

### Disable Stage 0:
In `pipeline_config.yaml`:
```yaml
pipeline:
  stages:
    stage0: false  # Disable normalization
```

## Performance

### Already Canonical Video (e.g., kohli_nets.mp4):
- **Input**: 1920x1080 @ 25fps, H.264, 45.1 MB
- **Processing**: Metadata extraction + validation only
- **Output**: Symlink/copy (no re-encoding)
- **Time**: ~1.3 seconds
- **Overhead**: 0s (no transcode)

### Needs Normalization:
Depends on video length and operations needed:
- **30-second 1080p video**:
  - Downscale from 4K: +2-3s
  - FPS conversion: +1-2s
  - Codec transcode: +3-5s
  - Total: 1-10s depending on operations

- **60-second 1080p video**:
  - Same operations, ~2x time

The `veryfast` preset prioritizes speed over file size.

## Benefits

### For Production Systems:
- ✅ **Robust input handling**: Accept any format/resolution/FPS
- ✅ **Consistent tracking**: Constant FPS ensures stable ByteTrack delta-t
- ✅ **Fast seeking**: Keyframe every 1.2s enables efficient frame access
- ✅ **Predictable performance**: Normalized inputs = no surprises
- ✅ **Cost control**: Upload limits prevent abuse/OOM
- ✅ **Quality assurance**: Validation catches problematic inputs early

### For Development:
- ✅ **Faster iteration**: Test with any video, Stage 0 handles it
- ✅ **Reproducibility**: Same canonical format = consistent results
- ✅ **Debugging**: Validation logs identify input issues

## Output Files

### `canonical_video.mp4`
The normalized video ready for Stage 1 (YOLO detection). If input was already canonical, this is a symlink/copy.

### `stage0_timing.json`
Contains detailed timing and metadata:
```json
{
  "stage": "stage0_normalize",
  "input_video": "path/to/input.mp4",
  "total_time": 1.33,
  "metadata_extraction_time": 0.94,
  "validation_time": 0.001,
  "normalization_time": 0.0,
  "normalization_needed": false,
  "original_metadata": {
    "width": 1920,
    "height": 1080,
    "fps": 25.0,
    "duration": 81.06,
    "codec": "h264",
    "pix_fmt": "yuv420p",
    "is_vfr": false
  },
  "canonical_metadata": { ... }
}
```

## Requirements

### System Dependencies:
- **ffmpeg**: For video transcoding and metadata extraction
- **ffprobe**: For video metadata (usually comes with ffmpeg)

Check if available:
```bash
ffmpeg -version
ffprobe -version
```

### Python Dependencies:
All in standard library:
- `subprocess` - Run ffmpeg/ffprobe
- `json` - Parse ffprobe output
- `pathlib` - Path handling
- `yaml` - Config loading
- `re` - Variable resolution

## Error Handling

### Validation Failures:
- **Duration > 120s**: ❌ FAIL - Video too long
- **File size > 200 MB**: ❌ FAIL - File too large
- Resolution, FPS, codec: ⚠️ WARN - Will normalize

### Processing Failures:
- **ffprobe error**: ❌ FAIL - Cannot read video metadata
- **ffmpeg error**: ❌ FAIL - Transcoding failed
- **Missing input**: ❌ FAIL - Video file not found

All failures exit with code 1 and clear error messages.

## Integration with Other Stages

### Stage 1 (YOLO Detection)
Stage 1 now reads from `canonical_video.mp4`:
```yaml
stage1:
  input:
    video_file: ${outputs_dir}/${current_video}/canonical_video.mp4
```

If Stage 0 is disabled, this path won't exist, so Stage 1 should fall back to the original video path.

### Future Stages
All stages benefit from:
- **Consistent FPS**: ByteTrack delta-t is predictable
- **Fast seeking**: GOP structure enables efficient frame access
- **Known format**: No codec surprises

## Troubleshooting

### "Input video not found"
- Check `video_dir` and `video_file` in `pipeline_config.yaml`
- Ensure path resolution is correct for your environment (Colab vs Windows)

### "ffprobe not found"
- Install ffmpeg: https://ffmpeg.org/download.html
- Windows: Add ffmpeg to PATH

### "Validation failed"
- Check video duration and file size
- Reduce video length or compress before upload

### "ffmpeg failed"
- Check ffmpeg error message in output
- Verify input video is not corrupted
- Try with different encoding preset

## Future Enhancements

### Possible Additions:
- [ ] Magic bytes validation (detect file type tampering)
- [ ] Bitrate normalization (control output file size)
- [ ] Aspect ratio handling (letterbox/pillarbox to 16:9)
- [ ] GPU acceleration (NVENC/QSV for faster encoding)
- [ ] Multi-pass encoding (better quality)
- [ ] Progress reporting (for long videos)
- [ ] Configurable GOP interval (all-I vs balanced)
- [ ] Audio preservation option (if needed)

## Status

**Implementation**: ✅ Complete  
**Testing**: ✅ Tested on Windows with kohli_nets.mp4  
**Colab Testing**: ⏳ Pending (need to test on Google Colab)  
**Production Ready**: ⚠️ Needs more testing with diverse inputs

## Related Documentation

- [Phase 0 Design Document](docs/PHASE0_VIDEO_NORMALIZATION.md) - Full design rationale
- [Pipeline Reorganization](docs/PIPELINE_REORGANIZATION_DESIGN.md) - Overall pipeline architecture
- [Stage 1 Documentation](stage1_detect.py) - YOLO detection using canonical video
