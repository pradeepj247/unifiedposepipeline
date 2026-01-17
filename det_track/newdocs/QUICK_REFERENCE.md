# Quick Reference & Cheat Sheet

**Latest Version**: January 17, 2026

## One-Minute Overview

**What**: Detect → Track → Select persons from video  
**Input**: Video file (any resolution)  
**Output**: Interactive HTML viewer with animated crop sequences  
**Time**: ~60 seconds for 2025 frames (1920×1080)  
**Modes**: Fast (59s), Balanced (67s), Full (77s)

---

## Essential Commands

### Run Everything
```bash
cd det_track
python run_pipeline.py --config configs/pipeline_config.yaml
```

### Run Specific Stages
```bash
python run_pipeline.py --config configs/pipeline_config.yaml --stages 1,2,3c,4
```

### Force Re-Run
```bash
python run_pipeline.py --config configs/pipeline_config.yaml --stages 4 --force
```

### Change Video
Edit `configs/pipeline_config.yaml` line 19:
```yaml
video_file: your_video.mp4
```

### Switch Mode
Edit line 25:
```yaml
mode: fast  # or balanced, or full
```

---

## Pipeline at a Glance

| Stage | File | Input | Output | Time |
|-------|------|-------|--------|------|
| **0** | stage0_normalize_video.py | Raw video | Canonical video | 2-8s |
| **1** | stage1_detect.py | Video | detections_raw.npz + crops_cache.pkl | 48.75s |
| **2** | stage2_track.py | detections | tracklets_raw.npz | 7.91s |
| **3a** | stage3a_analyze_tracklets.py | tracklets | tracklet_stats.npz | 0.23s |
| **3b** | stage3b_group_canonical.py | stats | canonical_persons.npz | 0.47s |
| **3c** | stage3c_filter_persons.py | canonical | final_crops_3c.pkl | 0.95s |
| **3d** | stage3d_refine_visual.py | tracklets | tracklets_recovered.npz | 8-12s (optional) |
| **4** | stage4_generate_html.py | crops | HTML viewer + WebPs | 2.51s |

**Total**: 60.24s (Stage 3d not included by default)

---

## File Locations

```
det_track/
├── configs/
│   └── pipeline_config.yaml          ← Edit this
├── run_pipeline.py                   ← Run this
├── stage0_normalize_video.py
├── stage1_detect.py
├── stage2_track.py
├── stage3a_analyze_tracklets.py
├── stage3b_group_canonical.py
├── stage3c_filter_persons.py
├── stage3d_refine_visual.py
└── stage4_generate_html.py

newdocs/  ← Full documentation
├── README_MASTER.md                  ← Start here
├── STAGE0_VIDEO_VALIDATION.md
├── STAGE1_DETECTION.md
├── STAGE2_TRACKING.md
├── STAGE3A_ANALYSIS.md
├── STAGE3B_GROUPING.md
├── STAGE3C_FILTER_PERSONS.md
├── STAGE3D_VISUAL_REFINEMENT.md
├── STAGE4_HTML_GENERATION.md
├── PIPELINE_CONFIG_REFERENCE.md
└── RUN_PIPELINE_EXECUTION.md

demo_data/
├── videos/
│   └── kohli_nets.mp4
├── images/
│   └── sample.jpg
└── outputs/
    └── kohli_nets/
        ├── canonical_video.mp4
        ├── detections_raw.npz
        ├── tracklets_raw.npz
        ├── tracklet_stats.npz
        ├── canonical_persons_3c.npz
        ├── final_crops_3c.pkl
        └── person_selection_slideshow.html  ← View this!
```

---

## Configuration Quickstart

### File: `configs/pipeline_config.yaml`

```yaml
# Line 10: Change repo root if needed
global:
  repo_root: /content/unifiedposepipeline

# Line 19: Change video input
  video_file: kohli_nets.mp4

# Line 25: Switch mode
mode: fast  # or balanced, or full

# Lines 66-73: Enable/disable stages
pipeline:
  stages:
    stage0: true
    stage1: true
    stage2: true
    stage3a: true
    stage3b: true
    stage3c: true
    stage3d: false  # Optional OSNet ReID
    stage4: true

# Lines 117: YOLO model (detected speed vs accuracy)
stage1_detect:
  detector:
    model_path: ${models_dir}/yolo/yolov8s.pt  # Try yolov8n for same speed

# Line 175: Crops per person
stage3c_filter:
  selection:
    crops_per_person: 60  # Controlled by mode (fast: 60, balanced: 30)

# Line 199: WebP animation speed
stage4_html:
  webp:
    duration_ms: 200  # 200ms = 5 FPS, 60 frames = 12s total
```

---

## Performance Breakdown

```
┌─────────────────────────────────────────┐
│ Total: 60.24s                           │
├─────────────────────────────────────────┤
│ Stage 1: 48.75s (79.3%) ████████████━  │
│ Stage 2:  7.91s (13.1%) ██━━━━━━━━━━━  │
│ Stage 3c: 0.95s (1.5%)  ━━━━━━━━━━━━━  │
│ Stage 4:  2.51s (3.9%)  ━━━━━━━━━━━━━  │
│ Stage 3a: 0.23s (0.4%)  ━━━━━━━━━━━━━  │
│ Stage 3b: 0.47s (0.7%)  ━━━━━━━━━━━━━  │
└─────────────────────────────────────────┘

Bottleneck: Stage 1 (YOLO + crop extraction)
  - Video decoding: 18-20ms per frame (CPU H.264)
  - YOLO inference: 8-10ms per frame (GPU)
  - Crop extraction: 2-3ms per crop (CPU)
  → 53 FPS limit (video I/O limited, not GPU)
```

---

## Data Flow & Key Linkage

**Critical**: `detection_idx` enables cross-stage tracking

```
Stage 1: Create detection_idx
  ├─ Each detection gets sequential ID (0, 1, 2, ...)
  └─ Stored in detections_raw.npz AND crops_cache.pkl

Stage 2: Propagate to tracklets
  └─ Tracklets inherit detection_indices

Stage 3b: Merge into canonical persons
  └─ Persons collect all child detection_indices

Stage 3c: O(1) lookup
  ├─ Build: crops_by_idx = {detection_idx: crop, ...}
  └─ Fetch: crop = crops_by_idx[detection_idx]  ← Fast!
```

Result: Can extract specific crops without re-reading video (11× faster Stage 3c)

---

## Output Interpretation

### File: `person_selection_slideshow.html`
- Open in browser
- View 60-frame animated WebP for each person
- Click "Select This Person" to proceed to pose estimation (Stage 5+)
- Shows rank, duration, confidence for each person

### Crop Selection Strategy
**3-Bin Contiguous**:
- Divide person's timeline into 3 sections
- Take 20 consecutive frames from middle of each section
- Result: Beginning + Middle + End perspectives
- No quality scoring = deterministic + reproducible

### File Sizes
| File | Size | Purpose |
|------|------|---------|
| detections_raw.npz | 0.16 MB | Bboxes |
| tracklets_raw.npz | 0.18 MB | Tracked sequences |
| canonical_persons_3c.npz | 0.13 MB | Top persons |
| final_crops_3c.pkl | 39 MB | 600 crop images |
| HTML viewer | 5.09 MB | WebPs + UI |
| **Total output** | **45 MB** | All results |

---

## Common Troubleshooting

| Problem | Solution |
|---------|----------|
| "Stage X not found" | Check det_track/ directory has stagex.py file |
| "${repo_root} in error" | Edit pipeline_config.yaml, uncomment Windows line if needed |
| Slow Stage 1 | Normal (53 FPS is video I/O limit), Stage 3d optional if too slow |
| HTML not loading | Open person_selection_slideshow.html locally (not drive) |
| 60 crops too many/few | Edit config: crops_per_person (fast: 60, balanced: 30, full: 50) |
| ByteTrack merges wrong people | That's why Stage 3d exists (enable_stage3d: true) |
| No output files | Check pipeline: stages: all true in config |

---

## Design Highlights

### Why Eager Crop Extraction?
**Trade-off**: Spend +5s in Stage 1 → Save -11s in Stage 3c = **-6s net gain**

### Why 3-Bin Selection?
**Deterministic** (no quality bias), **Temporal diversity** (beginning/middle/end), **Simple** (O(n) not O(n log n))

### Why ByteTrack Reused Frame?
**Insight**: ByteTrack only uses Kalman filters, doesn't read pixels → Can use 100×100 dummy instead of 1920×1080 dummy = **27% faster**

### Why Two Formats?
- **Fast mode** (single-row HTML): For quick preview
- **Full mode** (dual-row HTML + ReID): For maximum accuracy

---

## Next Steps (Stage 5+)

1. **User selects person** from HTML viewer
2. **Stage 5**: Extract all bboxes for that person
3. **Stage 6**: Run 2D pose estimation (RTMPose/ViTPose)
4. **Stage 7**: Run 3D pose lifting (HybrIK/MotionAGFormer)
5. **Stage 8+**: Biomechanics analysis (joint angles, velocities, etc.)

---

## Key Files to Modify

### To change behavior:
```
pipeline_config.yaml
├─ Line 10: repo_root (environment)
├─ Line 19: video_file (input)
├─ Line 25: mode (fast/balanced/full)
├─ Line 66-73: stages (enable/disable)
├─ Line 117: model (yolov8s vs yolov8n)
├─ Line 175: crops_per_person (quality vs speed)
└─ Line 199: duration_ms (animation speed)
```

### To understand code:
```
Start here: newdocs/README_MASTER.md
Then: Individual stage docs (STAGE1_DETECTION.md, etc.)
Then: run_pipeline.py and pipeline_config.yaml references
```

---

## Performance Summary

```
Video:    kohli_nets.mp4 (2025 frames, 1920×1080, 81 seconds)
GPU:      NVIDIA T4 (Colab)
Output:   10 persons, 60 crops each, 12-second WebP animations
Time:     60 seconds total (~75% of video length)
Accuracy: ~92% correct person identification

Breakdown:
- Detection+crops: 48.75s (79.3%) ← Bottleneck
- Tracking:        7.91s  (13.1%)
- Analysis:        0.70s  (1.2%)
- HTML generation: 2.51s  (4.2%)
- Overhead:        0.37s  (0.6%)
```

---

**Last Updated**: January 17, 2026  
**Status**: Production Ready ✅  
**Questions?** See [README_MASTER.md](README_MASTER.md) for detailed docs
