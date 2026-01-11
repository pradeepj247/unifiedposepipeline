# 🎬 Unified Detection & Tracking Pipeline - Complete Guide

**Last Updated**: January 10, 2026  
**Status**: ✅ Production Ready  
**Repository**: https://github.com/pradeepj247/unifiedposepipeline

---

## 🚀 Quick Start

**👉 To run this pipeline on Google Colab, follow the step-by-step guide:**

➡️ **[COLAB_QUICKSTART.md](COLAB_QUICKSTART.md)**

This guide covers:
1. Fresh git clone
2. Installing libraries & packages
3. Fetching model files
4. Installing demo data
5. Verifying environment
6. Running the complete pipeline

---

## 📋 Executive Summary

This is a **multi-stage offline pose estimation pipeline** that detects, tracks, and identifies persons in video footage. It integrates YOLO detection with ByteTrack tracking, creates an interactive HTML selection interface with animated WebP previews, and outputs structured person data for downstream pose estimation tasks.

### Pipeline Overview
- **11 Configurable Stages** (1-11, no gaps)
- **Fully Offline Processing** - detect once, track/group independently
- **Cross-Platform** - Colab and Windows compatible
- **Production Performance** - 2.96s for Stages 11+10 on Colab (T4 GPU)
- **Minimal Output** - 2.35 MB self-contained HTML with all WebP animations embedded

---

## 🏗️ Architecture: 11-Stage Pipeline

### Stage Overview Table

| Stage | Name | Input | Output | Time | Purpose |
|-------|------|-------|--------|------|---------|
| **1** | YOLO Detection | Video file | `detections_raw.npz` | ~5s | Find all people in each frame |
| **2** | ByteTrack Tracking | `detections_raw.npz` | `tracklets_raw.npz` | ~3s | Link detections across frames |
| **3** | Tracklet Analysis | `tracklets_raw.npz` | `tracklet_stats.npz` | ~1s | Compute statistics, find gaps |
| **4** | Load Crops Cache | Cached crops | In-memory cache | ~1s | Pre-load person crops (no ReID) |
| **5** | Canonical Grouping | `tracklets_raw.npz` | `canonical_persons.npz` | ~1s | Merge fragmented tracks → persons |
| **6** | HDF5 Enrichment | Crops + persons | `crops_enriched.h5` | ~2s | Index crops by person_id (O(1) lookup) |
| **7** | Rank Persons | `canonical_persons.npz` | `ranking_report.json` | ~1s | Sort by duration, coverage, position |
| **8** | Visualize Grouping | `canonical_persons.npz` | Debug video | ~30s | Optional: show all persons in video |
| **9** | Output Video | Top 10 persons | `output_video.mp4` | ~2s | Generate video of top 10 persons |
| **11** | Generate WebPs | Crops + persons | `webp/` folder | ~1.9s | Create 10 animated WebP files |
| **10** | HTML Report | WebPs + data | `.html` file | ~0.6s | Interactive selection UI (2.35 MB) |

**Note**: Stage numbering has intentional gaps (no stage 4a, no stage 4b, etc.) for backward compatibility.

---

## 🎯 Critical Design Decisions

### 1. **Offline Processing Philosophy**
- **YOLO runs once** (expensive, GPU-heavy)
- **Tracking, grouping, ranking are cheap** - iterate without re-detecting
- Enables rapid experimentation with tracking parameters
- All intermediate outputs saved as NPZ files for inspection

### 2. **HDF5 Enrichment (Stage 6)**
- Person crop images indexed by: `person_id → frame_id → {image_bgr, bbox, metadata}`
- **O(1) lookup time** instead of O(n) scanning
- Native numpy array storage (fast I/O)
- Enables WebP generation without re-reading crops cache

### 3. **WebP Format (Stage 11)**
- **2.4x faster** generation than GIF (1.9s vs 12s)
- **7.4x smaller** file sizes (1 MB vs 7.4 MB for 10 files)
- Auto-animates in browser (no JavaScript control needed)
- Embedded as base64 in HTML (single file deployment)

### 4. **Horizontal Tape Layout (Stage 10)**
- **1 row × 10 columns** person cards
- Horizontally scrollable on mobile
- Click any card to select person
- WebP animations auto-play without controls

### 5. **Stage Execution Order**
```
Detection (1-3) → Grouping (4-7) → Output (8-9) → WebPs (11) → HTML (10)
                                      └─→ HDF5 (6) ─────────────┘
```
**Stage 11 MUST run before Stage 10** (generates WebPs that HTML embeds)

---

## 📂 Path Resolution & Variables

### Global Variables (Single Source of Truth)

All paths use `${variable}` syntax in `configs/pipeline_config.yaml`:

```yaml
global:
  repo_root: /content/unifiedposepipeline      # Colab (comment for Windows)
  # repo_root: D:/trials/unifiedpipeline/newrepo  # Windows (uncomment locally)
  
  # Derived paths (multi-pass resolution)
  models_dir: ${repo_root}/models
  demo_data_dir: ${repo_root}/demo_data
  outputs_dir: ${demo_data_dir}/outputs
  video_dir: ${demo_data_dir}/videos/
  video_file: kohli_nets.mp4
```

### Path Resolution Process

1. **Load config** → Extract all `${...}` variables
2. **Multi-pass substitution**:
   - Pass 1: Resolve `${repo_root}` → `/content/unifiedposepipeline`
   - Pass 2: Resolve `${models_dir}` → `/content/unifiedposepipeline/models`
   - Pass 3: Resolve `${outputs_dir}` → `/content/unifiedposepipeline/demo_data/outputs`
3. **Runtime substitution**: `${current_video}` → `kohli_nets` (video filename without .mp4)
4. **Final paths**:
   ```
   /content/unifiedposepipeline/demo_data/outputs/kohli_nets/detections_raw.npz
   /content/unifiedposepipeline/demo_data/outputs/kohli_nets/webp/person_03.webp
   ```

### Environment Detection
- **Colab**: `repo_root: /content/unifiedposepipeline`
- **Windows**: `repo_root: D:/trials/unifiedpipeline/newrepo`
- Switch by uncommenting appropriate line in config

---

## 📊 Output Data Formats

### Stage 1: Detections NPZ
```python
{
  'frame_numbers': np.array([0, 0, 1, 1, ...]),  # Frame index per detection
  'bboxes': np.array([[x1,y1,x2,y2], ...]),      # Bounding box coords
  'confidences': np.array([0.92, 0.87, ...]),    # Detection scores
  'classes': np.array([0, 0, ...])                # Class ID (0=person)
}
```

### Stage 2: Tracklets NPZ
```python
{
  'tracklets': [  # List of tracklet dictionaries
    {
      'tracklet_id': 3,
      'frame_numbers': np.array([5, 6, 7, ...]),
      'bboxes': np.array([[x1,y1,x2,y2], ...]),
      'confidences': np.array([0.9, 0.88, ...])
    },
    ...
  ]
}
```

### Stage 5: Canonical Persons NPZ
```python
{
  'persons': [  # List of person dictionaries
    {
      'person_id': 3,                       # Unique person identifier
      'tracklet_ids': [5, 12, 18],         # Which tracklets merged into this person
      'frame_numbers': np.array([...]),    # All frames this person appears in
      'bboxes': np.array([...]),           # Aggregated bounding boxes
      'confidences': np.array([...]),      # Per-frame detection scores
      'duration_frames': 347,              # Total frames (for ranking)
      'coverage': 0.1714                   # Percentage of total video
    },
    ...
  ]
}
```

### Stage 6: HDF5 Crops Enriched
```
crops_enriched.h5
├── person_03/
│   ├── frame_000023/
│   │   ├── image_bgr: np.ndarray (192, 128, 3) BGR uint8
│   │   ├── bbox: [x1, y1, x2, y2]
│   │   ├── width: 128
│   │   └── height: 192
│   ├── frame_000024/
│   │   ├── image_bgr
│   │   ├── bbox
│   │   ...
│   └── ...
├── person_04/
│   ├── frame_000045/
│   │   ...
└── ...
```

### Stage 7: Ranking Report JSON
```json
{
  "ranking": [
    {
      "rank": 1,
      "person_id": 3,
      "duration_frames": 347,
      "coverage_percent": 17.14,
      "score": 0.893
    },
    ...
  ],
  "total_persons": 87,
  "total_frames": 2025,
  "algorithm": "weighted (duration=0.4, coverage=0.3, center=0.2, smoothness=0.1)"
}
```

### Stage 11: WebP Files
```
outputs/kohli_nets/webp/
├── person_03.webp  (0.30 MB, 50 frames @ 10 fps)
├── person_65.webp  (0.11 MB, 50 frames @ 10 fps)
├── person_37.webp  (0.15 MB, 50 frames @ 10 fps)
└── ...
```
- **Format**: Animated WebP (PIL-generated)
- **Quality**: 80/100
- **Frame rate**: 10 FPS (~5 second playback)
- **Resolution**: 128×192 (reduced from original)
- **Max frames**: 50 per person

### Stage 10: HTML Selection Report
```
outputs/kohli_nets/person_selection_report.html (2.35 MB)
```
- **Self-contained**: All WebP images embedded as base64
- **Layout**: 1 row × 10 columns horizontal scrollable tape
- **Per-card**: Rank, person ID, frame count, coverage %, animated WebP
- **Interaction**: Click to select, shows selection state
- **No external dependencies**: Pure HTML+CSS+minimal JS

---

## 🔧 Configuration: Pipeline Control

### Enable/Disable Stages (`pipeline_config.yaml`)

```yaml
pipeline:
  stages:
    stage1: true    # YOLO detection (MUST run first)
    stage2: true    # ByteTrack tracking
    stage3: true    # Tracklet analysis
    stage4: true    # Load crops (required for later stages)
    stage5: true    # Canonical grouping
    stage6: true    # HDF5 enrichment (required for WebP generation)
    stage7: true    # Ranking
    stage8: false   # Debug visualization (optional, slow)
    stage9: true    # Output video
    stage11: true   # WebP generation (MUST run before stage10)
    stage10: true   # HTML report (runs last, uses WebPs from stage11)
```

**Critical Rules**:
- Stage 1 must always run (no detections = pipeline fails)
- Stage 4 must run (loads crops cache for later use)
- Stage 6 must run before Stage 11 (HDF5 needed for WebP generation)
- Stage 11 must run before Stage 10 (WebPs embedded in HTML)

### Stage-Specific Configuration

#### Stage 1: YOLO Detection
```yaml
stage1:
  detector:
    model_path: ${models_dir}/yolo/yolov8s.pt
    confidence: 0.3        # Detection threshold
    device: cuda
    detect_only_humans: true
  
  detection_limit:
    method: hybrid         # hybrid | top_n | confidence
    max_count: 15          # Max detections per frame
    min_confidence: 0.3
  
  input:
    max_frames: 0          # 0 = all frames (change for quick tests: 100, 500, etc.)
```

#### Stage 2: ByteTrack Tracking
```yaml
stage2:
  params:
    track_thresh: 0.15     # Detection confidence for tracking
    track_buffer: 30       # Frames to keep lost tracks
    match_thresh: 0.8      # IOU threshold for matching
    min_hits: 1            # Min consecutive detections to create track
```
**Tuning Tips**:
- Lower `track_thresh` → more detections tracked (better recall, more noise)
- Increase `track_buffer` → longer fragmented tracks (better for slow people)
- Lower `match_thresh` → looser matching (more merges, fewer ID switches)

#### Stage 5: Canonical Grouping
```yaml
stage5:
  grouping:
    method: heuristic      # heuristic | clustering | hybrid
    heuristic_criteria:
      max_temporal_gap: 30         # Frames between tracklet end→start
      max_spatial_distance: 200    # Pixels between last→first center
      area_ratio_range: [0.7, 1.3] # Bbox area consistency
```

#### Stage 7: Ranking
```yaml
stage7:
  ranking:
    method: auto
    weights:
      duration: 0.4       # 40% - longest presence
      coverage: 0.3       # 30% - percentage of frames
      center: 0.2         # 20% - proximity to frame center
      smoothness: 0.1     # 10% - motion stability (less jitter)
```

#### Stage 11: WebP Generation
```yaml
stage11:
  video_generation:
    format: webp           # webp (was: gif)
    fps: 10                # Frames per second
    max_frames: 50         # Max frames per animation
    max_persons: 10        # Top N persons to generate
    frame_width: 128       # Output resolution
    frame_height: 192
    quality: 80            # WebP quality (0-100)
  
  output:
    webp_dir: ${outputs_dir}/${current_video}/webp
```

---

## 🚀 Running the Pipeline

### Full Pipeline Execution
```bash
cd /content/unifiedposepipeline/det_track  # Colab
# OR
cd d:\trials\unifiedpipeline\newrepo\det_track  # Windows

python run_pipeline.py --config configs/pipeline_config.yaml
```

### Run Specific Stages Only
```bash
# Just WebP generation + HTML (skip detection/tracking)
python run_pipeline.py --config configs/pipeline_config.yaml --stages 11,10

# Run everything except debug visualization
python run_pipeline.py --config configs/pipeline_config.yaml --stages 1,2,3,4,5,6,7,9,11,10

# Force re-run (skip caching checks)
python run_pipeline.py --config configs/pipeline_config.yaml --force
```

### Quick Test (100 frames only)
```bash
# Edit pipeline_config.yaml
stage1:
  input:
    max_frames: 100  # Change from 0 to 100

python run_pipeline.py --config configs/pipeline_config.yaml
```

---

## 📈 Performance Benchmarks

### Colab (T4 GPU, 360 frames, 720p video)

| Stage | Operation | Time | Notes |
|-------|-----------|------|-------|
| 1 | YOLO Detection (YOLOv8s) | ~5s | 72 FPS (4.6s actual) |
| 2 | ByteTrack Tracking | ~3s | Offline processing |
| 3 | Tracklet Analysis | ~1s | Statistics only |
| 4 | Load Crops Cache | ~1s | In-memory loading |
| 5 | Canonical Grouping | ~1s | Heuristic merge |
| 6 | HDF5 Enrichment | ~2s | Compression gzip |
| 7 | Rank Persons | ~1s | Auto-weighting |
| 8 | Visualize Grouping | ~30s | Optional, slow |
| 9 | Output Video | ~2s | MP4 encoding |
| 11 | WebP Generation | 1.9s | 10 WebPs @ quality=80 |
| 10 | HTML Report | 0.6s | Base64 encoding + layout |
| **Total (1-7,9,11,10)** | **Full Pipeline** | **~18s** | Skip stage 8 |

### Per-Stage Details

**Stage 1 Breakdown** (YOLO):
- Model load: 2s
- Detection: 77 FPS actual
- Total: ~4.6s for 360 frames

**Stage 11 Breakdown** (WebP generation):
- Load HDF5: 0.3s
- Resize frames: 0.8s
- PIL encode: 0.8s
- Total: 1.9s for 10 persons × 50 frames

---

## 📁 Output Directory Structure

```
outputs/
├── kohli_nets/                          # Video name folder
│   ├── detections_raw.npz               # Stage 1 output
│   ├── crops_cache.pkl                  # Stage 1 output (large ~800 MB)
│   ├── tracklets_raw.npz                # Stage 2 output
│   ├── tracklet_stats.npz               # Stage 3 output
│   ├── reid_candidates.json             # Stage 3 output
│   ├── canonical_persons.npz            # Stage 5 output
│   ├── grouping_log.json                # Stage 5 output
│   ├── crops_enriched.h5                # Stage 6 output (fast indexed lookup)
│   ├── primary_person.npz               # Stage 7 output
│   ├── ranking_report.json              # Stage 7 output
│   ├── top_persons_visualization.mp4    # Stage 9 output (optional)
│   ├── webp/                            # Stage 11 output
│   │   ├── person_03.webp
│   │   ├── person_65.webp
│   │   ├── person_37.webp
│   │   └── ... (10 total)
│   └── person_selection_report.html     # Stage 10 output (2.35 MB, self-contained)
└── dance.mp4/
    ├── ... (same structure)
```

### File Size Guide
- `detections_raw.npz`: 2-5 MB
- `crops_cache.pkl`: 500-1000 MB (LARGE! Frames × bboxes)
- `tracklets_raw.npz`: 1-2 MB
- `canonical_persons.npz`: 1 MB
- `crops_enriched.h5`: 800 MB (indexed, compressed)
- `webp/` folder: 1-1.5 MB total (10 WebPs)
- `person_selection_report.html`: 2-3 MB (embedded WebPs)

**Disk Estimate**: ~2.5-3 GB per video (mostly crops)

---

## 🔍 Debugging & Troubleshooting

### Common Issues

#### 1. Path Resolution Fails
**Symptom**: Error message shows `${repo_root}` still in path
```
FileNotFoundError: [Errno 2] No such file or directory: '${outputs_dir}/detections_raw.npz'
```
**Fix**: Check `pipeline_config.yaml` `global.repo_root` matches your environment
```yaml
global:
  repo_root: /content/unifiedposepipeline  # Colab (uncomment this)
  # repo_root: D:/trials/unifiedpipeline/newrepo  # Windows (comment out)
```

#### 2. Crops Cache Not Found
**Symptom**: Stage 4 fails because `crops_cache.pkl` doesn't exist
**Fix**: Re-run Stage 1 (generates crops cache)
```bash
python run_pipeline.py --config configs/pipeline_config.yaml --stages 1,4,5,6,7,9,11,10
```

#### 3. WebP Generation Fails
**Symptom**: `ModuleNotFoundError: No module named 'PIL'`
**Fix**: PIL is installed as fallback in `stage9_generate_person_webps.py`, but manually:
```bash
pip install Pillow
```

#### 4. HTML Shows Missing WebPs
**Symptom**: HTML generated but WebP animations appear as placeholders
**Fix**: Ensure Stage 11 ran successfully before Stage 10
```bash
# Check webp/ folder exists and has files
ls /content/unifiedposepipeline/demo_data/outputs/kohli_nets/webp/
# Should show: person_03.webp, person_65.webp, ...

# Re-run both stages
python run_pipeline.py --config configs/pipeline_config.yaml --stages 11,10
```

#### 5. Memory Out of Bounds (crops_cache)
**Symptom**: RAM usage spike to >16 GB when loading crops cache
**Fix**: Reduce `max_frames` for quick testing
```yaml
stage1:
  input:
    max_frames: 100  # Test with 100 frames instead of all
```

#### 6. HTML File Too Large
**Symptom**: HTML file is >5 MB, takes long to load/edit
**Fix**: This is normal (2.35 MB for 10 WebPs embedded). Options:
- Keep as-is (still single file, offline-friendly)
- Set `stage10: false` to skip HTML generation
- Reduce WebP quality (trade file size for visual quality)
```yaml
stage11:
  video_generation:
    quality: 70  # Lower = smaller files, ~7.4x reduction at quality=70
```

---

## 📝 Console Output & Logging

### Stage 11 Output Example
```
🚀 Running Stage 11: Generate Person Animated WebPs...

======================================================================
🎬 STAGE 11: GENERATE PERSON ANIMATED WEBP FILES
======================================================================

📂 Loading canonical persons...
📁 Output directory: /content/unifiedposepipeline/demo_data/outputs/kohli_nets/webp

🎬 Generating animated WebP files for top 10 persons...

  ✅ Rank 1: P3 - person_03.webp (50 frames, 128x192, 0.30 MB)
  ✅ Rank 2: P65 - person_65.webp (50 frames, 128x192, 0.11 MB)
  ✅ Rank 3: P37 - person_37.webp (50 frames, 128x192, 0.15 MB)
  ...

======================================================================
📊 WebP Generation Summary:
  ✅ Successful: 10/10
  📁 Output: /content/unifiedposepipeline/demo_data/outputs/kohli_nets/webp
======================================================================

✅ Stage 11 completed in 1.90s
```

### Stage 10 Output Example
```
🚀 Running Stage 10: HTML Selection Report (Horizontal Tape)...

======================================================================
📄 STAGE 10: CREATE HORIZONTAL TAPE LAYOUT SELECTION REPORT
======================================================================

📂 Loading canonical persons...
   Calculated video_duration_frames from data: 2025
📂 Loading crops cache...
🎬 Encoding WebP files and generating HTML...

  🎬 Rank 1: P3 - Encoding WebP... ✅
  🎬 Rank 2: P65 - Encoding WebP... ✅
  ...

✅ Report created: 2.35 MB

✅ Stage 10 completed in 0.59s
   HTML: person_selection_report.html
   Open: file:///content/unifiedposepipeline/demo_data/outputs/kohli_nets/person_selection_report.html
```

---

## 🎨 HTML Selection UI Features

### Layout: Horizontal Tape
- **1 row, 10 columns** of person cards
- **Horizontally scrollable** on mobile devices
- **Sticky header** with title and info
- **Footer** with statistics and instructions

### Per-Card Features
- **Rank badge** (#1, #2, ... #10)
- **Person ID** (P3, P65, P37, etc.)
- **Animated WebP** (auto-plays, ~5 second loop)
- **Frame count** (50 frames, for example)
- **Coverage %** (percentage of total video)
- **Hover effect** - slight elevation and shadow
- **Click handler** - shows selection state (TODO in next iteration)

### HTML Styling
- **Colors**: Purple gradient header, light gray cards, dark text
- **Typography**: Segoe UI, responsive font sizes
- **Responsive**: Works on desktop (>1200px), tablet (800px), mobile (320px)
- **No dependencies**: Pure HTML/CSS/JS (no libraries)

### Expected File Size Breakdown (2.35 MB total)
- HTML structure: ~5 KB
- CSS styling: ~10 KB
- JavaScript: ~2 KB
- Base64 WebP images: ~2.33 MB (10 × ~233 KB each)
- Metadata/margins: ~5 KB

---

## 🎯 Design Decisions & Rationale

### Why WebP Instead of MP4?
| Format | Gen Time | File Size | Browser | Auto-play |
|--------|----------|-----------|---------|-----------|
| MP4 | 2s | 0.08-0.15 MB | ✅ | ❌ Complex JS |
| GIF | 12s | 0.57-0.80 MB | ✅ | ✅ Yes |
| **WebP** | **1.9s** | **0.11-0.30 MB** | **✅** | **✅ Yes** |

**Decision**: WebP provides best balance of speed, file size, browser support, and auto-animation without JavaScript complexity.

### Why HDF5 Instead of Dict Cache?
- **Dict/pickle**: Entire cache loaded into RAM (~1-2 GB)
- **HDF5**: Indexed access, only requested crops loaded (~100 MB)
- **Lookup time**: O(n) scanning → O(1) HDF5 key access
- **Enables**: Stage 11 WebP generation without 10-minute load time

### Why Horizontal Tape Instead of Grid?
- **Grid**: Wastes space, requires scrolling in both directions
- **Tape**: Familiar "film strip" metaphor, easy horizontal scroll
- **Touch-friendly**: Swipe to browse on mobile
- **10 person limit**: Matches single-row optimization

### Why Offline Processing?
- **Detection is expensive** (~5s for 360 frames)
- **Tracking is cheap** (~1-3s, many parameter options)
- **Iteration workflow**: Change tracking settings, re-run stages 2-7 (~5s) without re-detecting
- **Reproducibility**: All intermediate outputs saved for inspection

---

## 🔮 Future Enhancements (TODOs)

### High Priority
- [ ] **Selection Feedback (Stage 10)**
  - Visual highlight when person card clicked
  - Show "Currently selected: P3" message
  - Export selected person ID to JSON for downstream use
  
- [ ] **3D Pose Lifting (Post-Pipeline)**
  - Extract selected person's crops from HDF5
  - Run ViTPose/RTMPose on crops
  - Generate 3D skeleton using HybrIK/MotionAGFormer

### Medium Priority
- [ ] **ReID Recovery (Stage 4a)**
  - Optional ONNX ReID model to merge similar-looking persons
  - Currently disabled due to `boxmot.appearance` dependency
  - Would improve canonical grouping for similar-height people

- [ ] **Batch Processing**
  - Process multiple videos in sequence
  - Aggregate rankings across videos
  - Parallel stage execution (1-7 independent, 8-10 sequential)

- [ ] **Web UI Backend**
  - Replace static HTML with Flask/FastAPI server
  - Realtime person selection and preview
  - Export selected crop sequences

### Lower Priority
- [ ] **Alternative Backends**
  - Try different tracking: DeepSORT, TrackletNet
  - Try different grouping: Hungarian algorithm, graph clustering
  - Benchmark against ByteTrack baseline

- [ ] **Video Format Options**
  - AVIF format (even smaller than WebP)
  - HE-AAC audio for MP4 variant
  - Configurable output video codec (H.264 vs H.265)

- [ ] **Dataset Export**
  - COCO format for pose annotation
  - MOT format for tracking evaluation
  - Custom JSON with full metadata

---

## 📚 Code Structure

### Main Files

**Orchestration**
- `run_pipeline.py` - Main pipeline runner, stage orchestration, timing

**Stages**
- `stage1_detect.py` - YOLO detection
- `stage2_track.py` - ByteTrack tracking
- `stage3_analyze.py` - Tracklet statistics
- `stage4_load_crops.py` - Load crops cache (stub, no output)
- `stage4_load_crops_cache.py` - Load crops cache (Stage 4)
- `stage5_group_canonical.py` - Canonical person grouping (Stage 5)
- `stage6_enrich_crops.py` - HDF5 enrichment (Stage 6)
- `stage7_rank_persons.py` - Ranking (Stage 7)
- `stage8_visualize_grouping.py` - Debug visualization (Stage 8)
- `stage9_create_output_video.py` - Output video (Stage 9)
- `stage10_generate_person_webps.py` - WebP generation (Stage 10, was GIF)
- `stage6b_create_selection_html_horizontal.py` - HTML report (Stage 10)

**Configuration**
- `configs/pipeline_config.yaml` - Single source of truth for paths, stages, parameters
- `configs/tracking_params.yaml` - Legacy tracking parameters (not used, all in pipeline_config)

**Utilities**
- `path_resolver.py` - Multi-pass `${variable}` substitution
- Various imports from parent modules (YOLO, ByteTrack, etc.)

---

## 🚀 Quick Start Guide

### 1. One-Time Setup (Colab)
```bash
# Clone and install
!git clone https://github.com/pradeepj247/unifiedposepipeline.git /content/unifiedposepipeline
cd /content/unifiedposepipeline
!pip install -r requirements.txt
```

### 2. Run Full Pipeline
```bash
cd /content/unifiedposepipeline/det_track
!python run_pipeline.py --config configs/pipeline_config.yaml
```

### 3. View Results
```bash
# Find output folder
import os
output_dir = '/content/unifiedposepipeline/demo_data/outputs/kohli_nets/'
print("Files created:")
for root, dirs, files in os.walk(output_dir):
    for f in files:
        size_mb = os.path.getsize(os.path.join(root, f)) / 1e6
        print(f"  {f}: {size_mb:.1f} MB")

# Open HTML in Colab
from IPython.display import IFrame
IFrame(src='file:///content/unifiedposepipeline/demo_data/outputs/kohli_nets/person_selection_report.html', 
       width=1200, height=800)
```

### 4. Next: Pose Estimation (Post-Pipeline)
```bash
# Use selected person's crops from HDF5 for pose estimation
python udp_video.py --config configs/udp_video.yaml \
  --person_id 3 \
  --crops_file demo_data/outputs/kohli_nets/crops_enriched.h5
```

---

## 📞 Support & Questions

### If Something Breaks
1. **Check console output** - error messages are detailed with stage context
2. **Verify paths** - ensure `global.repo_root` matches environment
3. **Check intermediate files** - do earlier stage outputs exist? (`detections_raw.npz` for stage 2, etc.)
4. **Reduce scope** - test with `max_frames: 100` first
5. **Re-run failing stage** - sometimes transient I/O errors

### To Understand a Stage
1. **Find stage file** in `det_track/` directory (e.g., `stage2_track.py`)
2. **Read docstring** at top of file (describes input/output)
3. **Check config section** in `pipeline_config.yaml` (parameters explained)
4. **Look at console output** (shows what stage is doing)
5. **Inspect output NPZ file** in `outputs/` folder

### To Modify a Stage
1. **Read this guide** (understand pipeline context)
2. **Find config parameters** (in `pipeline_config.yaml`)
3. **Modify config** (safest way - no code changes)
4. **Test with `max_frames: 100`** (quick iterations)
5. **Re-run downstream stages** (stage 11 + 10 are cheap, ~2.5s)
6. **Commit changes** with detailed message explaining rationale

---

## 📜 Version History

| Date | Version | Changes |
|------|---------|---------|
| Jan 3, 2026 | 1.0 | 🎉 **Production Release**: WebP format, HDF5 enrichment, HTML UI |
| Jan 2, 2026 | 0.9 | GIF format implementation, horizontal tape layout |
| Jan 1, 2026 | 0.8 | MP4 format investigation, video playback debugging |
| Dec 31, 2025 | 0.7 | HDF5 enrichment and person-crop association |
| Dec 30, 2025 | 0.6 | Stage numbering refactor (1-11) |

---

## ✅ Pre-Deployment Checklist

Before deploying to production:

- [ ] All 11 stages configured in `pipeline_config.yaml`
- [ ] Path variables correct for your environment (Colab vs Windows)
- [ ] Test run with `max_frames: 100` completes without errors
- [ ] Output folder structure matches expected layout
- [ ] HTML file generated and loads in browser
- [ ] WebP animations auto-play in HTML
- [ ] Person selection click handler works
- [ ] All console messages are clean (no warnings)
- [ ] Timing logs show reasonable performance
- [ ] Output files committed to results folder

---

**Happy Tracking! 🎬**

For questions or updates, refer to this document or the GitHub repository.
