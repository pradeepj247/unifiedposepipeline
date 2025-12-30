# Detection & Tracking Pipeline - Design Document

**Date Created:** December 30, 2025  
**Status:** Design Phase  
**Location:** `det_track/`

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Directory Structure](#directory-structure)
4. [Configuration System](#configuration-system)
5. [Stage Specifications](#stage-specifications)
6. [Data Formats](#data-formats)
7. [Design Decisions](#design-decisions)
8. [Implementation Plan](#implementation-plan)

---

## 🎯 Overview

### Purpose
Multi-stage pipeline for person detection, tracking, and identity resolution that separates concerns and enables flexible benchmarking of different tracking strategies.

### Key Features
- **Offline Processing:** Run detection once, experiment with tracking parameters
- **Toggleable Stages:** Enable/disable ReID recovery and canonical grouping independently
- **Hybrid Approach:** Combines motion-based tracking (ByteTrack) with selective appearance-based recovery (ReID)
- **Modular Design:** Each stage is independent and testable
- **Path Variables:** Centralized path management for environment flexibility

### Problem Solved
Traditional tracking pipelines run detection + tracking in one pass, making it expensive to tune tracking parameters or fix ID switches. This pipeline:
1. Runs YOLO detection once → saves to NPZ
2. Experiments with tracking offline (no re-detection needed)
3. Applies selective ReID only where ID switches occur (not entire video)
4. Groups fragmented tracklets into canonical person identities

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DETECTION & TRACKING PIPELINE                 │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ Stage 1: YOLO Detection (Offline)                               │
│ ───────────────────────────────────────────────────────────────  │
│ Input:  video.mp4                                                │
│ Action: Run YOLO detector, extract top-N persons per frame       │
│ Output: detections_raw.npz                                       │
│         - frame_numbers: [0, 0, 0, 1, 1, ...]                   │
│         - bboxes: [[x1,y1,x2,y2], ...]                          │
│         - confidences: [0.92, 0.87, ...]                        │
│         - classes: [0, 0, 0, ...]                               │
└──────────────────────────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────────┐
│ Stage 2: ByteTrack Tracking (Offline, Motion-Only)              │
│ ───────────────────────────────────────────────────────────────  │
│ Input:  detections_raw.npz                                       │
│ Action: Run ByteTrack on stored detections (Kalman filter + IoU)│
│         Expects ID switches during occlusions (normal behavior)  │
│ Output: tracklets_raw.npz                                        │
│         - tracklet_id: int                                       │
│         - frame_numbers: [5, 6, 7, ..., 780]                    │
│         - bboxes: [[x1,y1,x2,y2], ...]                          │
│         - confidences: [0.92, 0.91, ...]                        │
│ Note:   Typically produces 30-40 short tracklets                 │
└──────────────────────────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────────┐
│ Stage 3: Tracklet Analysis                                       │
│ ───────────────────────────────────────────────────────────────  │
│ Input:  tracklets_raw.npz                                        │
│ Action: Compute geometric statistics per tracklet:               │
│         - Temporal: start_frame, end_frame, duration             │
│         - Spatial: mean_center, center_jitter, mean_area         │
│         - Motion: velocity, smoothness                           │
│         Identify candidate pairs for ReID recovery               │
│ Output: tracklet_stats.npz (embedded statistics)                 │
│         reid_candidates.json (suspicious pairs)                  │
└──────────────────────────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────────┐
│ Stage 4a: Selective ReID Recovery [OPTIONAL - TOGGLEABLE]       │
│ ───────────────────────────────────────────────────────────────  │
│ Input:  video.mp4 + tracklets_raw.npz + reid_candidates.json    │
│ Action: For each suspicious tracklet pair:                       │
│         1. Extract frames at transition points (last/first)      │
│         2. Crop person ROIs from bboxes                          │
│         3. Extract ReID features (OSNet)                         │
│         4. Compute cosine similarity                             │
│         5. Merge if similarity > threshold                       │
│ Output: tracklets_recovered.npz (fewer, longer tracklets)        │
│         reid_merge_log.json (which tracklets were merged)        │
│ Note:   Only processes ~10-50 frames (not entire video)          │
│         Toggleable for speed/accuracy benchmarking               │
└──────────────────────────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────────┐
│ Stage 4b: Canonical Person Grouping [OPTIONAL - TOGGLEABLE]     │
│ ───────────────────────────────────────────────────────────────  │
│ Input:  tracklets_recovered.npz (if 4a ran) OR                   │
│         tracklets_raw.npz (if 4a skipped)                        │
│ Action: Further grouping using geometric heuristics:             │
│         - Temporal continuity (gap < N frames)                   │
│         - Spatial proximity (last bbox near first bbox)          │
│         - Size consistency (area ratio)                          │
│         - Motion continuity (velocity alignment)                 │
│         OR clustering (DBSCAN, Agglomerative)                    │
│ Output: canonical_persons.npz                                    │
│         - canonical_id: ["A", "B", ...]                         │
│         - tracklet_ids: [[3, 24], [7, 15], ...]                 │
│         - frame_ranges: [[5, 2100], [10, 500], ...]             │
│ Note:   Produces human-interpretable identities                  │
│         Toggleable for speed/accuracy benchmarking               │
└──────────────────────────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────────┐
│ Stage 5: Primary Person Ranking & Selection                     │
│ ───────────────────────────────────────────────────────────────  │
│ Input:  canonical_persons.npz                                    │
│ Action: Rank canonical persons by:                               │
│         - Total duration (longest presence)                      │
│         - Temporal coverage (% of video frames)                  │
│         - Center bias (proximity to frame center)                │
│         - Motion smoothness (less jitter)                        │
│         Select primary person automatically or manually          │
│ Output: primary_person.npz                                       │
│         ranking_report.json                                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📂 Directory Structure

```
unifiedpipeline/newrepo/
│
├── run_detector_tracking.py          # Existing (untouched)
├── udp_video.py                       # Existing
├── udp_image.py                       # Existing
│
├── configs/                           # Existing configs (untouched)
│   ├── detector.yaml
│   └── udp_video.yaml
│
└── det_track/                         # NEW SUBDIRECTORY ⭐
    │
    ├── PIPELINE_DESIGN.md             # This document
    ├── README.md                      # Usage guide
    │
    ├── stage1_detect.py               # Stage 1: Detection
    ├── stage2_track.py                # Stage 2: Tracking
    ├── stage3_analyze_tracklets.py    # Stage 3: Analysis
    ├── stage4a_reid_recovery.py       # Stage 4a: ReID recovery
    ├── stage4b_group_canonical.py     # Stage 4b: Canonical grouping
    ├── stage5_rank_persons.py         # Stage 5: Ranking
    │
    ├── run_pipeline.py                # Master orchestrator
    ├── benchmark_pipeline.py          # Benchmarking tool
    │
    ├── utils/                         # Shared utilities
    │   ├── __init__.py
    │   ├── detection_utils.py         # Detection helpers
    │   ├── tracking_utils.py          # Tracking helpers
    │   ├── reid_utils.py              # ReID feature extraction
    │   ├── geometry_utils.py          # Bbox/spatial calculations
    │   └── visualization_utils.py     # Debug visualizations
    │
    └── configs/                       # Pipeline configs
        ├── pipeline_config.yaml       # Master config
        ├── pipeline_fast.yaml         # Speed preset
        ├── pipeline_accurate.yaml     # Accuracy preset
        └── pipeline_hybrid.yaml       # Balanced preset
```

---

## ⚙️ Configuration System

### Path Variables (Option C)

**Centralized path management for environment flexibility.**

```yaml
# det_track/configs/pipeline_config.yaml

# ============================================================================
# Global Path Variables (Single Source of Truth)
# ============================================================================
global:
  # Repository root (auto-detected or specified)
  repo_root: /content/unifiedposepipeline  # Colab
  # repo_root: D:/trials/unifiedpipeline/newrepo  # Local Windows
  
  # Derived paths (use ${variable} syntax)
  models_dir: ${repo_root}/models
  demo_data_dir: ${repo_root}/demo_data
  outputs_dir: ${demo_data_dir}/outputs
  
  # Current video being processed
  current_video: dance  # Used for per-video output directories

# ============================================================================
# Stage 1: Detection
# ============================================================================
stage1_detect:
  detector:
    type: yolo
    model_path: ${models_dir}/yolo/yolov8s.pt  # Uses variable
    confidence: 0.3
    device: cuda
  
  input:
    video_path: ${demo_data_dir}/videos/dance.mp4  # Uses variable
  
  output:
    detections_file: ${outputs_dir}/${current_video}/detections_raw.npz

# ============================================================================
# Stage 4a: ReID Recovery
# ============================================================================
stage4a_reid_recovery:
  reid:
    model_path: ${models_dir}/reid/osnet_x0_25_msmt17.pt  # Uses variable
  
  input:
    video_path: ${demo_data_dir}/videos/dance.mp4  # Uses variable
```

### Variable Interpolation in Code

```python
# In run_pipeline.py or stage scripts
import os
import re

def resolve_path_variables(config, global_vars=None):
    """Recursively resolve ${variable} in config"""
    if global_vars is None:
        global_vars = config.get('global', {})
    
    if isinstance(config, dict):
        return {k: resolve_path_variables(v, global_vars) for k, v in config.items()}
    elif isinstance(config, list):
        return [resolve_path_variables(v, global_vars) for v in config]
    elif isinstance(config, str):
        # Replace ${variable} with actual value
        def replace_var(match):
            var_name = match.group(1)
            return str(global_vars.get(var_name, match.group(0)))
        return re.sub(r'\$\{(\w+)\}', replace_var, config)
    else:
        return config

# Usage:
config = yaml.safe_load(open('configs/pipeline_config.yaml'))
config = resolve_path_variables(config)
# Now all paths are resolved
```

### Benefits

| Benefit | Example |
|---------|---------|
| **Environment Switch** | Change `repo_root` once: `/content/` (Colab) → `D:/trials/` (Windows) |
| **Maintainability** | Update model location: change `models_dir` once, applies to all stages |
| **Clarity** | `${models_dir}/yolo/yolov8s.pt` is clearer than `../../models/yolo/yolov8s.pt` |
| **Per-video Outputs** | `${outputs_dir}/${current_video}/` creates organized structure |

---

## 📊 Stage Specifications

### Stage 1: Detection

**Purpose:** Run YOLO detector and save all detections to NPZ.

**Input:**
- Video file (MP4, AVI, etc.)

**Output:**
```python
detections_raw.npz:
  frame_numbers: (M,) int64          # Flat array: [0, 0, 0, 1, 1, 2, ...]
  bboxes: (M, 4) float32             # [[x1, y1, x2, y2], ...]
  confidences: (M,) float32          # [0.92, 0.87, 0.81, ...]
  classes: (M,) int64                # [0, 0, 0, ...] (all "person")
  num_detections_per_frame: (N,) int64  # [3, 2, 5, ...] for reconstruction
```

**Configuration Options (TBD):**

| Option | Description | Status |
|--------|-------------|--------|
| **Detection Limit** | How to filter detections per frame | ⚠️ **TBD** |
| - `top_n` | Take top N by confidence (e.g., 10) | Option A |
| - `confidence` | All above threshold (e.g., > 0.3) | Option B |
| - `hybrid` | Top N with minimum confidence | Option C ✅ Recommended |

**Recommendation:** `hybrid` (max 15 detections, min 0.3 confidence)

---

### Stage 2: Tracking

**Purpose:** Run ByteTrack offline on stored detections (motion-only).

**Input:**
- `detections_raw.npz`

**Output:**
```python
tracklets_raw.npz:
  tracklets: list[dict]
    Each tracklet:
      tracklet_id: int
      frame_numbers: (T,) int64      # Frames where this tracklet appears
      bboxes: (T, 4) float32
      confidences: (T,) float32
```

**Key Points:**
- **No video frames needed** (pure motion model)
- **ID switches expected** (occlusions, re-entries)
- **Only ByteTrack/OCSORT supported** (other trackers need ReID frames)

---

### Stage 3: Analysis

**Purpose:** Compute tracklet statistics and identify ReID candidates.

**Input:**
- `tracklets_raw.npz`

**Output:**
```python
tracklet_stats.npz:
  tracklet_id: (K,) int64
  start_frame: (K,) int64
  end_frame: (K,) int64
  duration: (K,) int64
  mean_center: (K, 2) float32       # [x_center, y_center]
  center_jitter: (K,) float32       # Spatial stability
  mean_area: (K,) float32
  area_variance: (K,) float32
  mean_velocity: (K, 2) float32
  velocity_magnitude: (K,) float32

reid_candidates.json:
  [
    {
      "tracklet_1": 3,
      "tracklet_2": 24,
      "gap": 15,                     # Frames between tracklets
      "distance": 120.5,             # Pixels between last/first bbox
      "area_ratio": 0.95,
      "transition_frame_1": 768,
      "transition_frame_2": 783
    },
    ...
  ]
```

**Candidate Criteria:**
- Temporal gap < 50 frames
- Spatial distance < 300 pixels
- Area ratio within [0.6, 1.4]

---

### Stage 4a: ReID Recovery [TOGGLEABLE]

**Purpose:** Selectively merge tracklets using appearance features.

**Input:**
- `video.mp4` (only read transition frames)
- `tracklets_raw.npz`
- `reid_candidates.json`

**Output:**
```python
tracklets_recovered.npz:
  # Same format as tracklets_raw.npz, but with merged tracklets
  tracklets: list[dict]

reid_merge_log.json:
  [
    {
      "tracklet_pair": [3, 24],
      "similarity": 0.82,
      "merged_into": 3,
      "removed": 24,
      "frame_gap": 15
    },
    ...
  ]
```

**Toggle Control:**
```yaml
pipeline:
  stages:
    stage4a_reid_recovery: true  # Set false to skip

stage4a_reid_recovery:
  enabled: true  # Independent toggle
```

**Performance:**
- Processes only ~10-50 frames (transition points)
- ReID feature extraction: ~25-50ms per crop (OSNet x0.25)
- Similarity computation: <1ms per pair
- **Total overhead:** ~1-5 seconds for typical video

---

### Stage 4b: Canonical Grouping [TOGGLEABLE]

**Purpose:** Further group tracklets using geometric heuristics or clustering.

**Input:**
- `tracklets_recovered.npz` (if Stage 4a ran)
- `tracklets_raw.npz` (if Stage 4a skipped)
- `tracklet_stats.npz`

**Output:**
```python
canonical_persons.npz:
  # Option A: Metadata Only (TBD)
  canonical_id: ["A", "B", "C", ...]
  tracklet_ids: [[3, 24], [7, 15], ...]  # Which tracklets per person
  frame_ranges: [[5, 2100], [10, 500], ...]
  durations: [2095, 490, ...]

  # Option B: Full Trajectories (TBD)
  canonical_id: ["A", "B", ...]
  frame_numbers: [[5,6,7,...,2100], [...], ...]
  bboxes: [[(x1,y1,x2,y2), ...], [...], ...]
  confidences: [[0.9, 0.88, ...], [...], ...]
  tracklet_ids: [[3, 24], [7, 15], ...]
```

**Configuration Options (TBD):**

| Option | Description | Status |
|--------|-------------|--------|
| **Canonical Format** | What data to include | ⚠️ **TBD** |
| - `metadata_only` | Lightweight (tracklet IDs + frame ranges) | Option A ✅ Recommended |
| - `full_trajectories` | Complete bbox sequences | Option B |
| **Grouping Method** | Algorithm choice | ⚠️ **TBD** |
| - `heuristic` | Rule-based (fast, interpretable) | Option A ✅ Recommended |
| - `clustering` | ML-based (DBSCAN, Agglomerative) | Option B |
| - `hybrid` | Rules + clustering | Option C |

**Toggle Control:**
```yaml
pipeline:
  stages:
    stage4b_group_canonical: true  # Set false to skip

stage4b_group_canonical:
  enabled: true  # Independent toggle
  grouping:
    method: heuristic  # or clustering, hybrid
```

---

### Stage 5: Ranking

**Purpose:** Select primary person for single-person pipelines.

**Input:**
- `canonical_persons.npz`

**Output:**
```python
primary_person.npz:
  canonical_id: "A"
  frame_numbers: [5, 6, 7, ..., 2100]  # Full trajectory
  bboxes: [(x1,y1,x2,y2), ...]
  score: 0.92  # Ranking score

ranking_report.json:
  [
    {
      "canonical_id": "A",
      "duration": 2095,
      "coverage": 0.89,
      "center_bias": 0.75,
      "smoothness": 0.88,
      "total_score": 0.92
    },
    {
      "canonical_id": "B",
      "duration": 490,
      "coverage": 0.21,
      ...
    }
  ]
```

**Configuration Options (TBD):**

| Option | Description | Status |
|--------|-------------|--------|
| **Selection Method** | How to choose primary | ⚠️ **TBD** |
| - `auto` | Weighted ranking | Option A ✅ Recommended |
| - `manual` | User specifies canonical_id | Option B |
| - `both` | Auto with manual override | Option C |

**Ranking Weights (if auto):**
```yaml
stage5_rank:
  ranking:
    weights:
      duration: 0.4       # Longest presence
      coverage: 0.3       # % of frames
      center_bias: 0.2    # Frame center proximity
      smoothness: 0.1     # Motion stability
```

---

## 📦 Data Formats

### NPZ File Specifications

**NPZ (NumPy Compressed Archive):**
- Efficient binary format
- Supports multiple arrays in one file
- Compresses automatically (smaller than pickle)
- Easy to load: `data = np.load('file.npz')`

**Naming Convention (TBD):**

| Option | Example | Status |
|--------|---------|--------|
| **Descriptive** | `detections_raw.npz`, `tracklets_recovered.npz` | Option A ✅ Recommended |
| **Versioned** | `stage1_detections_20251230_143522.npz` | Option B |
| **Per-video** | `dance/detections_raw.npz`, `kohli_nets/detections_raw.npz` | Option C ✅ Recommended |

**Recommendation:** Combine Option A + C (descriptive names in per-video directories)

```
demo_data/outputs/
├── dance/
│   ├── detections_raw.npz
│   ├── tracklets_raw.npz
│   ├── tracklets_recovered.npz
│   ├── canonical_persons.npz
│   └── primary_person.npz
└── kohli_nets/
    └── ...
```

---

## 🎯 Design Decisions

### ✅ Confirmed Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Subdirectory** | `det_track/` | Isolates new pipeline, prevents conflicts |
| **Path Strategy** | Variables (Option C) | Single source of truth, environment flexibility |
| **Stage 4a/4b Toggle** | Independent enable flags | Benchmarking speed vs accuracy |
| **ByteTrack Only** | For offline mode | Only motion-based trackers work without frames |
| **Selective ReID** | Extract ~10-50 frames only | Efficient hybrid approach |

### ⚠️ Pending Decisions

**Need User Input:**

1. **Detection Limit Method** (Stage 1)
   - [ ] Top N (e.g., 10)
   - [ ] Confidence threshold (e.g., > 0.3)
   - [x] Hybrid (top 15, min 0.3) ✅ **Recommended**

2. **Canonical Person Format** (Stage 4b)
   - [x] Metadata only (tracklet IDs + ranges) ✅ **Recommended**
   - [ ] Full trajectories (complete bboxes)

3. **Grouping Method** (Stage 4b)
   - [x] Heuristic rules ✅ **Recommended for v1**
   - [ ] ML clustering
   - [ ] Hybrid

4. **Primary Person Selection** (Stage 5)
   - [x] Auto ranking ✅ **Recommended**
   - [ ] Manual selection
   - [ ] Both (auto + override)

5. **Output Directory Structure**
   - [ ] Flat (`demo_data/outputs/`)
   - [x] Per-video subdirectories ✅ **Recommended**
   - [ ] Timestamped

6. **Utils Module**
   - [x] Separate `det_track/utils/` ✅ **Recommended**
   - [ ] Inline in each stage

7. **README Depth**
   - [ ] Minimal (quick start)
   - [x] Comprehensive (architecture + examples) ✅ **Recommended**

---

## 🔄 Benchmark Scenarios

### Four Scenarios to Test

```yaml
# Scenario 1: Baseline (Fastest)
stage4a_reid_recovery: false
stage4b_group_canonical: false
# Output: Raw ByteTrack tracklets (ID switches present)

# Scenario 2: Geometric Only (Fast)
stage4a_reid_recovery: false
stage4b_group_canonical: true
# Output: Geometrically grouped tracklets (motion-based merge)

# Scenario 3: ReID Only (Accurate)
stage4a_reid_recovery: true
stage4b_group_canonical: false
# Output: ReID-merged tracklets (appearance-based merge)

# Scenario 4: Hybrid (Most Accurate)
stage4a_reid_recovery: true
stage4b_group_canonical: true
# Output: ReID + geometric merge (best of both)
```

### Expected Performance

| Scenario | Speed | Accuracy | Use Case |
|----------|-------|----------|----------|
| Baseline | 100% | Low | Quick draft, debugging |
| Geometric | 105% | Medium | Good enough for simple videos |
| ReID Only | 120% | High | Crowded scenes, occlusions |
| Hybrid | 125% | Highest | Production quality |

*Speed relative to baseline (100% = fastest)*

---

## 🚀 Implementation Plan

### Phase 1: Core Pipeline (Stages 1-3)
- [ ] Create `stage1_detect.py`
- [ ] Create `stage2_track.py`
- [ ] Create `stage3_analyze_tracklets.py`
- [ ] Create `det_track/utils/` module
- [ ] Create basic config YAML
- [ ] Test on sample video

### Phase 2: Optional Stages (4a, 4b)
- [ ] Create `stage4a_reid_recovery.py`
- [ ] Create `stage4b_group_canonical.py`
- [ ] Implement toggleable logic
- [ ] Test all 4 benchmark scenarios

### Phase 3: Ranking & Orchestration
- [ ] Create `stage5_rank_persons.py`
- [ ] Create `run_pipeline.py` (master)
- [ ] Create `benchmark_pipeline.py`
- [ ] Create preset configs (fast/accurate/hybrid)

### Phase 4: Documentation & Polish
- [ ] Create README.md
- [ ] Add visualization utilities
- [ ] Add error handling
- [ ] Add progress bars and logging
- [ ] Create usage examples

---

## 📝 Usage Examples

### Run Full Pipeline
```bash
cd det_track
python run_pipeline.py --config configs/pipeline_config.yaml
```

### Run Individual Stage
```bash
python stage1_detect.py --config configs/pipeline_config.yaml
python stage2_track.py --config configs/pipeline_config.yaml
```

### Run Benchmark
```bash
python benchmark_pipeline.py --config configs/pipeline_config.yaml
# Outputs comparison of all 4 scenarios
```

### Use Presets
```bash
# Fast preset (speed optimized)
python run_pipeline.py --config configs/pipeline_fast.yaml

# Accurate preset (accuracy optimized)
python run_pipeline.py --config configs/pipeline_accurate.yaml

# Hybrid preset (balanced)
python run_pipeline.py --config configs/pipeline_hybrid.yaml
```

---

## 🔧 Technical Notes

### ByteTrack Offline Capability

**Confirmed:** ByteTrack can run on pre-stored detections without video frames.

**How it works:**
```python
# ByteTrack only needs bbox coordinates and confidences
detections = np.column_stack([bboxes, confidences, classes])
# Shape: (N, 6) = [x1, y1, x2, y2, conf, cls]

tracked = bytetrack_tracker.update(detections, frame=None)
# frame=None is OK for motion-only tracking!
```

**Kalman Filter Math:**
- Predicts next position from current bbox + velocity
- Updates prediction when new detection matches (IoU)
- No pixel data needed, pure geometry

### ReID Feature Extraction

**OSNet Models:**
- `osnet_x0_25_msmt17.pt`: 0.25M params, ~25 FPS, good accuracy
- `osnet_x1_0_msmt17.pt`: 2.2M params, ~12 FPS, best accuracy

**Feature Vector:**
- Output: 512-dim embedding
- Normalized to unit length
- Cosine similarity for matching

**Similarity Threshold:**
- 0.5-0.6: Very lenient (many false positives)
- 0.7-0.8: Sweet spot (balanced)
- 0.85-0.95: Very strict (may miss valid merges)

### Path Variable Implementation

```python
import re

def resolve_variables(config):
    """Recursively resolve ${variable} in config"""
    global_vars = config.get('global', {})
    
    def resolve_string(s):
        return re.sub(
            r'\$\{(\w+)\}',
            lambda m: str(global_vars.get(m.group(1), m.group(0))),
            s
        )
    
    def resolve_recursive(obj):
        if isinstance(obj, dict):
            return {k: resolve_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_recursive(v) for v in obj]
        elif isinstance(obj, str):
            return resolve_string(obj)
        return obj
    
    return resolve_recursive(config)
```

---

## 📚 References

- **ByteTrack Paper:** [arXiv:2110.06864](https://arxiv.org/abs/2110.06864)
- **OSNet ReID:** [arXiv:1905.00953](https://arxiv.org/abs/1905.00953)
- **BoxMOT Library:** [GitHub](https://github.com/mikel-brostrom/boxmot)

---

## 📅 Changelog

**2025-12-30:** Initial design document created
- Defined 5-stage pipeline architecture
- Established path variable system (Option C)
- Documented toggleable Stage 4a/4b
- Listed pending decisions (7 items)
- Created benchmark scenarios

---

## 🤝 Next Steps

1. **User Review:** Get feedback on pending decisions
2. **Implementation:** Start with Phase 1 (core pipeline)
3. **Testing:** Validate on sample videos
4. **Iteration:** Refine based on real-world performance

---

*This document is a living design specification and will be updated as implementation progresses.*
