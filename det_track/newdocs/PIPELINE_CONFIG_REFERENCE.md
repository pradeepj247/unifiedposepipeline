# pipeline_config.yaml Reference Guide

**Location**: `det_track/configs/pipeline_config.yaml`

## Overview

Single YAML file controls entire pipeline:
- **Path resolution**: Where to find videos, models, outputs
- **Mode selection**: Fast vs balanced vs full
- **Stage execution**: Enable/disable each stage
- **Stage parameters**: Thresholds, limits, tuning knobs

## Global Configuration

```yaml
global:
  repo_root: /content/unifiedposepipeline  # Colab
  # repo_root: D:/trials/unifiedpipeline/newrepo  # Windows (uncomment)
  
  models_dir: ${repo_root}/models
  demo_data_dir: ${repo_root}/demo_data
  outputs_dir: ${demo_data_dir}/outputs
  
  video_dir: ${demo_data_dir}/videos/
  video_file: kohli_nets.mp4  # ← CHANGE THIS for different videos
  
  verbose: false  # Set true for detailed debug output
```

### Path Resolution

Uses `${variable}` syntax for cross-platform compatibility:

```yaml
models_dir: ${repo_root}/models
# Resolves to: /content/unifiedposepipeline/models  (Colab)
# Or:          D:/trials/unifiedpipeline/newrepo/models  (Windows)

video_path: ${video_dir}${video_file}
# Resolves to: /content/unifiedposepipeline/demo_data/videos/kohli_nets.mp4
```

**Multi-pass resolution**: Variables can reference other variables (resolved iteratively).

## Pipeline Mode Configuration

```yaml
mode: full  # Options: fast | balanced | full

modes:
  fast:
    description: "Quick analysis (~59s) - 60 crops per person, no ReID merging"
    crops_per_person: 60
    enable_stage3d: false
    stage4_dual_row: false
  
  balanced:
    description: "Moderate analysis (~67s) - 30 crops per person, no ReID merging"
    crops_per_person: 30
    enable_stage3d: false
    stage4_dual_row: false
  
  full:
    description: "Thorough analysis (~77s) - 50 crops per person, with ReID merging"
    crops_per_person: 50
    enable_stage3d: true
    stage4_dual_row: true
```

**Mode Impact**:
- **fast**: No OSNet ReID, single-row HTML, fewer crops
- **balanced**: Moderate crops, still fast
- **full**: Maximum accuracy, includes ReID, dual-row HTML

## Stage Execution Control

```yaml
pipeline:
  stages:
    stage0: true   # Video normalization
    stage1: true   # Detection
    stage2: true   # Tracking
    stage3a: true  # Tracklet analysis
    stage3b: true  # Canonical grouping
    stage3c: true  # Filter persons & extract crops
    stage3d: true  # Visual ReID refinement (uses enable_stage3d from mode)
    stage4: true   # HTML generation
  
  advanced:
    verbose: false  # Overall verbosity (can override per-stage)
```

**Usage**:
- Set to `false` to skip stages
- Stages are cached (won't re-run if output exists)
- Use `--force` flag to re-run: `python run_pipeline.py --force`

## Stage-Specific Configuration

### Stage 0: Video Normalization

```yaml
stage0_normalize:
  enabled: true
  
  limits:
    min_width: 640
    min_height: 480
    max_width: 1920
    max_height: 1080
  
  ffmpeg_preset: medium  # ultrafast | superfast | veryfast | faster | fast | medium | slow | slower | veryslow
  bitrate: 8000k         # Output bitrate
  
  input:
    video_file: ${video_dir}${video_file}
  
  output:
    canonical_video_file: ${outputs_dir}/${current_video}/canonical_video.mp4
```

### Stage 1: Detection

```yaml
stage1_detect:
  detector:
    model_path: ${models_dir}/yolo/yolov8s.pt  # Change to yolov8n.pt for faster (but less accurate)
    confidence: 0.3       # Detection threshold (0-1)
    device: cuda          # or cpu
    detect_only_humans: true  # Filter to person class
  
  detection_limit:
    method: hybrid        # top_n | confidence | hybrid
    max_count: 15         # Max detections per frame
    min_confidence: 0.3   # Minimum confidence
  
  input:
    video_file: ${video_dir}${video_file}
  
  output:
    detections_file: ${outputs_dir}/${current_video}/detections_raw.npz
```

### Stage 2: Tracking

```yaml
stage2_track:
  tracker:
    track_buffer: 30      # Frames to keep alive (tracklet timeout)
    track_thresh: 0.5     # IoU threshold for assignment
    match_thresh: 0.8     # Association threshold
    frame_rate: 25        # Video FPS
    mot20: false          # Use MOT20 settings (for crowded scenes)
  
  verbose: false  # Show progress bar
  
  input:
    detections_file: ${outputs_dir}/${current_video}/detections_raw.npz
  
  output:
    tracklets_file: ${outputs_dir}/${current_video}/tracklets_raw.npz
```

### Stage 3a: Tracklet Analysis

```yaml
stage3a_analyze:
  ranking:
    duration_weight: 0.5       # How much to favor long tracklets
    confidence_weight: 0.3     # How much to favor high confidence
    coverage_weight: 0.2       # How much to favor large bboxes
    
    late_appearance_threshold: 0.75  # Penalty if appears after 75% through video
    late_appearance_penalty: 0.7     # Multiply score by this
  
  reid:
    temporal_overlap_frames: 200  # Max frame gap for candidate pairs
    min_feature_similarity: 0.6    # Min cosine similarity
  
  input:
    tracklets_file: ${outputs_dir}/${current_video}/tracklets_raw.npz
  
  output:
    stats_file: ${outputs_dir}/${current_video}/tracklet_stats.npz
    reid_candidates_file: ${outputs_dir}/${current_video}/reid_candidates.json
```

### Stage 3b: Canonical Grouping

```yaml
stage3b_group:
  grouping:
    top_tracklets: 10              # Max persons to output
    temporal_gap_threshold: 100    # Frames to wait before losing track
    geometric_distance_ratio: 0.33 # Fraction of frame width for separation
    min_tracklet_confidence: 0.3   # Minimum confidence
  
  input:
    tracklets_file: ${outputs_dir}/${current_video}/tracklets_raw.npz
    stats_file: ${outputs_dir}/${current_video}/tracklet_stats.npz
  
  output:
    canonical_file: ${outputs_dir}/${current_video}/canonical_persons_3b.npz
```

### Stage 3c: Filter Persons & Extract Crops

```yaml
stage3c_filter:
  selection:
    crops_per_person: 60      # From mode config (fast: 60, balanced: 30, full: 50)
    selection_method: contiguous  # contiguous | quality
    
    # 3-bin contiguous selection
    bins: 3
    crops_per_bin: 20
  
  delete_crops_cache_after_success: true  # Auto-cleanup 527 MB cache
  
  input:
    canonical_file: ${outputs_dir}/${current_video}/canonical_persons_3b.npz
    crops_cache_file: ${outputs_dir}/${current_video}/crops_cache.pkl
  
  output:
    final_crops_file: ${outputs_dir}/${current_video}/final_crops_3c.pkl
    canonical_3c_file: ${outputs_dir}/${current_video}/canonical_persons_3c.npz
```

### Stage 3d: Visual Refinement (Optional)

```yaml
stage3d_refine:
  enabled: ${enable_stage3d}  # From mode config
  
  osnet:
    model_name: osnet_x1_0    # osnet_x1_0 | osnet_x0_5 | osnet_x0_25
    pretrained: true
    device: cuda
  
  matching:
    distance_threshold: 0.35    # Cosine distance cutoff
    min_samples_per_tracklet: 3 # Crops to average
    temporal_window: 200        # Max frame gap
    spatial_distance_ratio: 0.5 # Min separation
  
  input:
    tracklets_file: ${outputs_dir}/${current_video}/tracklets_raw.npz
    stats_file: ${outputs_dir}/${current_video}/tracklet_stats.npz
    crops_cache_file: ${outputs_dir}/${current_video}/crops_cache.pkl
  
  output:
    tracklets_recovered_file: ${outputs_dir}/${current_video}/tracklets_recovered.npz
```

### Stage 4: HTML Generation

```yaml
stage4_html:
  webp:
    duration_ms: 200        # 200ms = 5 FPS, 60 frames = 12 seconds
    quality: 90             # 0-100 (higher = better)
    target_width: 256
    target_height: 256
  
  html:
    dual_row: ${stage4_dual_row}  # From mode config
    include_metadata: true
    responsive_layout: true
  
  input:
    final_crops_file: ${outputs_dir}/${current_video}/final_crops_3c.pkl
  
  output:
    viewer_html: ${outputs_dir}/${current_video}/person_selection_slideshow.html
    webp_viewer_dir: ${outputs_dir}/${current_video}/webp_viewer/
```

## Common Configuration Changes

### Process Different Video
```yaml
global:
  video_file: your_video.mp4  # Change from kohli_nets.mp4
```

### Switch Pipeline Mode
```yaml
mode: fast  # or balanced, or full
```

### Disable Video Normalization
```yaml
pipeline:
  stages:
    stage0: false  # Skip if video already canonical
```

### Enable OSNet ReID (Full Mode)
```yaml
mode: full  # Includes stage3d: true
```

### Increase Detection Limit
```yaml
stage1_detect:
  detection_limit:
    max_count: 20  # Was 15, increase for crowded scenes
```

### Adjust ByteTrack Sensitivity
```yaml
stage2_track:
  tracker:
    track_buffer: 50      # Longer tracklet lifetime (was 30)
    track_thresh: 0.6     # More lenient assignment (was 0.5)
```

## File Reference

All output files use `${outputs_dir}/${current_video}/` prefix:

| File | Stage | Purpose |
|------|-------|---------|
| `canonical_video.mp4` | 0 | Normalized video |
| `detections_raw.npz` | 1 | All bboxes with detection_idx |
| `crops_cache.pkl` | 1 | 527 MB cache (deleted after 3c) |
| `tracklets_raw.npz` | 2 | 67 tracklets with bboxes |
| `tracklet_stats.npz` | 3a | Ranking scores |
| `reid_candidates.json` | 3a | Pairs for OSNet matching |
| `canonical_persons_3b.npz` | 3b | 8-10 canonical persons |
| `tracklets_recovered.npz` | 3d | Merged tracklets (if ReID enabled) |
| `final_crops_3c.pkl` | 3c | 60 crops × 10 persons (39 MB) |
| `canonical_persons_3c.npz` | 3c | Canonical persons with 3c crops |
| `person_selection_slideshow.html` | 4 | Interactive viewer (5 MB) |
| `webp_viewer/` | 4 | Individual WebP files (temporary) |

---

**Related**: [Back to Master](README_MASTER.md) | [run_pipeline.py →](RUN_PIPELINE_EXECUTION.md)
