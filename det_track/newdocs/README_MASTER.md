# Detection & Tracking Pipeline - Complete Documentation
**Last Updated: January 17, 2026 (Latest)**

## Core Objective
To detect and track persons in video, perform identity resolution through multi-stage analysis, and present curated results to the end user for final visual selection.

---

## Pipeline Overview

### Main Architecture
```
Video Input
    ↓
[Stage 0] Video Normalization & Validation
    ↓
[Stage 1] YOLO Detection (YOLOv8s)
    ↓
[Stage 2] ByteTrack Offline Tracking
    ↓
[Stage 3] Multi-stage Analysis & Refinement
    ├─ Stage 3a: Tracklet Analysis (compute statistics)
    ├─ Stage 3b: Canonical Grouping (merge tracklets into persons)
    ├─ Stage 3c: Filter & Extract Crops (top 8-10 persons, eager crop extraction)
    └─ Stage 3d: Visual Refinement (optional OSNet ReID matching)
    ↓
[Stage 4] HTML Viewer Generation (WebP + interactive UI)
    ↓
[Stage 5] Person Selection (User picks person_id from HTML viewer)
    ↓
Output: selected_person.npz (ready for pose estimation)
```

**Total Pipeline Time**: ~60 seconds (2025 frames, 1920×1080, T4 GPU)

---

## Pipeline Execution Modes

### Mode Selection in Config
```yaml
mode: full  # Options: fast | balanced | full
```

| Mode | Duration | Crops/Person | ReID (Stage 3d) | HTML Output | Use Case |
|------|----------|--------------|-----------------|------------|----------|
| **fast** | ~59s | 60 | ❌ No | Single row | Quick preview |
| **balanced** | ~67s | 30 | ❌ No | Single row | Moderate throughput |
| **full** | ~77s | 50 | ✅ Yes | Dual rows | Maximum accuracy |

### Stage Execution Options
- **Full pipeline**: `python run_pipeline.py --config configs/pipeline_config.yaml`
- **Specific stages**: `python run_pipeline.py --config configs/pipeline_config.yaml --stages 1,2,3c,4`
- **Force re-run**: `python run_pipeline.py --config configs/pipeline_config.yaml --stages 4 --force`
- **Skip 3D ReID**: Edit config `enable_stage3d: false` to skip Stage 3d and get single-row HTML

---

## Detailed Stage Documentation

Each stage is documented in its own file with inputs, outputs, and design rationale:

| Stage | File | Purpose | Docs |
|-------|------|---------|------|
| **0** | `stage0_normalize_video.py` | Video validation & normalization | [📄 Stage 0](STAGE0_VIDEO_VALIDATION.md) |
| **1** | `stage1_detect.py` | YOLO detection + eager crop extraction | [📄 Stage 1](STAGE1_DETECTION.md) |
| **2** | `stage2_track.py` | ByteTrack offline tracking | [📄 Stage 2](STAGE2_TRACKING.md) |
| **3a** | `stage3a_analyze_tracklets.py` | Tracklet statistics & ranking | [📄 Stage 3a](STAGE3A_ANALYSIS.md) |
| **3b** | `stage3b_group_canonical.py` | Canonical person grouping | [📄 Stage 3b](STAGE3B_GROUPING.md) |
| **3c** | `stage3c_filter_persons.py` | Filter top persons + 3-bin crop selection | [📄 Stage 3c](STAGE3C_FILTER_PERSONS.md) |
| **3d** | `stage3d_refine_visual.py` | OSNet visual ReID matching | [📄 Stage 3d](STAGE3D_VISUAL_REFINEMENT.md) |
| **4** | `stage4_generate_html.py` | WebP generation + HTML viewer | [📄 Stage 4](STAGE4_HTML_GENERATION.md) |
| **5** | `stage5_extract_person.py` | User selects person → extract bboxes | [📄 Stage 5](STAGE5_PERSON_SELECTION.md) |

---

## Configuration & Execution

### Main Files
- **Pipeline orchestrator**: [`run_pipeline.py`](RUN_PIPELINE_EXECUTION.md) - Loads config, resolves paths, calls stages
- **Configuration**: [`pipeline_config.yaml`](PIPELINE_CONFIG_REFERENCE.md) - All settings in one place

### Key Config Features
1. **Path Resolution**: Multi-pass `${variable}` substitution for cross-platform compatibility
2. **Mode Selection**: `fast | balanced | full` controls speed/quality tradeoff
3. **Stage Control**: Enable/disable individual stages
4. **Video Input**: Change `video_file` to process different videos

---

## Performance Summary

### Comprehensive Timing Breakdown (2025 frames, 1920×1080, T4 GPU)

| Stage | Component | Time | % | FPS | Details |
|-------|-----------|------|---|-----|---------|
| **0** | Video Validation | ~2s | 3.2% | - | Check codec, resolution, GOP; normalize if needed |
| **1** | YOLO Detection | 41.23s | 66.8% | 49.1 | YOLOv8s inference on GPU |
| **1** | Crop Extraction (eager) | 7.52s | 12.2% | 269.3 | Extract 8832 crops during detection |
| **1** | **Stage 1 Total** | **48.75s** | **79.0%** | **41.5** | **Detection + eager extraction** |
| **2** | ByteTrack | 7.91s | 12.8% | 694.7 | Optimized: dummy frame reuse + no tqdm |
| **3a** | Tracklet Analysis | 0.23s | 0.4% | - | Compute statistics (duration, coverage, etc.) |
| **3b** | Canonical Grouping | 0.47s | 0.8% | - | 67 tracklets → 40+ persons |
| **3c** | Filter + Crop Selection | 0.95s | 1.5% | - | 40+ → 8-10 persons, 3-bin selection |
| **3d** | OSNet ReID (optional) | 8-12s | 13-16% | - | Visual matching: 8-10 → 7-8 persons |
| **4** | WebP Generation | 2.51s | 4.1% | - | 10 persons × 60 crops → WebP + HTML |
| | **TOTAL (without 3d)** | **60.24s** | **100%** | **33.6** | **Standard fast mode** |
| | **TOTAL (with 3d)** | **~77s** | **-** | **26.3** | **Full mode with ReID** |

### Performance Notes
- **Video I/O bottleneck**: Stage 1 limited by H.264 CPU decoding (~53 FPS cap), not GPU
- **YOLOv8s vs v8n**: No speed difference (both 53 FPS) due to decode bottleneck
- **ByteTrack optimization**: 694.7 FPS (27% faster via dummy frame reuse)
- **Eager extraction ROI**: +7.5s in Stage 1, but saves 10-12s in Stage 3c = net +5-6s gain
- **Stage 3d optional**: Disabled in fast mode (saves 8-12s), enabled in full mode

### Pipeline Mode Comparison

| Mode | Total Time | Stage 3d | Crops/Person | HTML Rows | Accuracy |
|------|------------|----------|--------------|-----------|----------|
| **fast** | ~60s | ❌ Off | 60 | 1 row (Stage 3c only) | Good |
| **balanced** | ~67s | ❌ Off | 30 | 1 row (Stage 3c only) | Good |
| **full** | ~77s | ✅ On | 50 | 2 rows (3c vs 3d comparison) | Best |

---

## Key Design Decisions & Rationale

### 1. YOLOv8s for Detection (Not v8n)
**Decision**: Use YOLOv8s despite being "larger"  
**Rationale**: Testing showed YOLOv8n ≈ YOLOv8s speed (both ~53 FPS) because:
- Video decoding (H.264 CPU) is the bottleneck, not model inference
- YOLOv8s provides better accuracy with no speed penalty
- Pure YOLO test: 38.24s for 2025 frames (53 FPS), same with both models

### 2. Eager Crop Extraction in Stage 1
**Decision**: Extract all 8832 crops during detection, save to `crops_cache.pkl`  
**Rationale**:
- Adds 4-5s in Stage 1 (12% overhead)
- Saves 10-11s in Stage 3c (O(1) lookup instead of re-reading video)
- **Net gain: 5-6 seconds** for entire pipeline

### 3. 3-Bin Contiguous Frame Selection
**Decision**: Divide person's timeline into 3 bins, take 20 consecutive frames from middle of each bin  
**Rationale**:
- Ensures temporal diversity (beginning/middle/end of person's appearance)
- 60 crops total per person = 12s WebP loop at 200ms/frame
- No quality scoring = deterministic, reproducible crops
- Simpler than hybrid binning, same temporal coverage

### 4. ByteTrack Optimizations
**Decision**: Reuse single 100×100 dummy frame, disable tqdm progress bar  
**Rationale**:
- ByteTrack only uses Kalman filters (doesn't read pixel data)
- Creating 2025 × 1920×1080 arrays was wasteful
- Result: 715 FPS peak (27% improvement from baseline)

### 5. No Frame Skipping
**Decision**: Process all 2025 frames despite video decoding overhead  
**Rationale**:
- Skipping frames saves YOLO time but loses crop diversity
- Video must be decoded for crop extraction anyway
- Net savings: only 19s for added complexity + tracking quality loss
- Current 60s is good enough for production

---

## Data Flow & Intermediate Outputs

### Critical Linkage: `detection_idx`
Every crop and bbox is tagged with `detection_idx` to enable unambiguous tracking through all stages:
```
Stage 1: detection_idx created when YOLO runs
  └─→ stored in detections_raw.npz & crops_cache.pkl

Stage 2: detection_indices propagated to tracklets_raw.npz
  └─→ enables linking detections → tracklets

Stage 3b: detection_indices merged into canonical_persons.npz
  └─→ enables stage3c to read original detection_idx

Stage 3c: O(1) lookup: crops_by_idx[detection_idx]
  └─→ directly fetch crops without re-reading video
```

### Complete File Listing (All Pipeline Outputs)

| File | Stage | Size (2025f) | Lifetime | Purpose |
|------|-------|--------------|----------|---------|
| **canonical_video.mp4** | 0 | ~50-150 MB | ✅ Kept | Normalized video (H.264, GOP=30, ≤1080p) |
| **detections_raw.npz** | 1 | 0.16 MB | ✅ Kept | All YOLO detections (frame_numbers, bboxes, confidences) |
| **crops_cache.pkl** | 1 | 527 MB | ❌ Deleted | All 8832 crops (temp cache for Stage 3c) |
| **tracklets_raw.npz** | 2 | 0.18 MB | ✅ Kept | ByteTrack output (67 tracklets with detection_indices) |
| **tracklet_stats.npz** | 3a | 0.11 MB | ✅ Kept | Statistics (duration, coverage, center, smoothness) |
| **canonical_persons.npz** | 3b | 0.15 MB | ✅ Kept | Grouped persons (~40+ persons from 67 tracklets) |
| **grouping_log.json** | 3b | 0.02 MB | ✅ Kept | Merge decisions and criteria used |
| **canonical_persons_3c.npz** | 3c | 0.13 MB | ✅ Kept | Filtered top 8-10 persons (input for Stage 3d) |
| **final_crops_3c.pkl** | 3c | 39 MB | ✅ Kept | 60 crops × 8-10 persons (480-600 crops total) |
| **canonical_persons_3d.npz** | 3d | 0.12 MB | ✅ Kept | OSNet-merged persons (7-8 persons, optional) |
| **final_crops_3d.pkl** | 3d | 34 MB | ✅ Kept | Crops for 3d persons (420-480 crops, optional) |
| **merging_report.json** | 3d | 0.01 MB | ✅ Kept | Person chains detected by OSNet (optional) |
| **stage3d_sidecar.json** | 3d | 0.05 MB | ✅ Kept | Similarity matrix debug info (optional) |
| **webp_viewer/** | 4 | 5.09 MB | ✅ Kept | Directory with WebP files + HTML |
| └─ **person_000.webp** | 4 | ~0.5 MB each | ✅ Kept | Animated WebP (60 frames @ 200ms = 12s loop) |
| └─ **person_selection.html** | 4 | 0.08 MB | ✅ Kept | Interactive viewer (click person → auto-select) |
| **selected_person.npz** | 5 | 2-3 MB | ✅ Kept | User-selected person bboxes (for pose estimation) |

**Total Disk Usage**:
- **Without Stage 3d**: ~95 MB (canonical_video + detections + tracklets + crops_3c + webp)
- **With Stage 3d**: ~130 MB (adds canonical_persons_3d + final_crops_3d + reports)
- **After Stage 5**: +2-3 MB (selected_person.npz)
- **Peak during Stage 3c**: ~622 MB (includes 527 MB crops_cache before deletion)

**Cleanup Strategy**:
- `crops_cache.pkl` auto-deleted after successful Stage 3c completion (frees 527 MB)
- All other files kept for reproducibility and debugging
- If re-running pipeline: existing files overwritten (no manual cleanup needed)

---

## Common Operations

### Run Full Pipeline
```bash
cd det_track
python run_pipeline.py --config configs/pipeline_config.yaml
```

### Test Single Stage
```bash
# Re-run Stage 4 (force bypass cache)
python run_pipeline.py --config configs/pipeline_config.yaml --stages 4 --force

# Run tracking only (skips Stage 0-1)
python run_pipeline.py --config configs/pipeline_config.yaml --stages 2

# Run all analysis stages
python run_pipeline.py --config configs/pipeline_config.yaml --stages 3a,3b,3c,3d
```

### Change Video Input
Edit `det_track/configs/pipeline_config.yaml`:
```yaml
global:
  video_file: your_video.mp4  # Change this
```

### Switch Pipeline Mode
Edit `det_track/configs/pipeline_config.yaml`:
```yaml
mode: fast  # or balanced, or full
```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `${repo_root}` in error | Path resolution failed | Check YAML syntax, run with `verbose: true` |
| Stage 4 JPEG errors | Crops have different sizes | Stage 3c 3-bin selection ensures consistent dims |
| Memory errors | crops_cache.pkl not deleted | Stage 3c auto-cleanup; check for exceptions |
| No output files | Stage disabled in config | Enable in `pipeline: stages:` section |

---

## Next Steps (Stage 6+)

**Stage 6 (2D Pose Estimation)**: Use `selected_person.npz` to extract person crops → run RTMPose/ViTPose → generate 2D keypoints  
**Stage 7 (3D Lifting)**: Use 2D keypoints → run HybrIK/MotionAGFormer → generate 3D keypoints  
**Stage 8+ (Biomechanics)**: Joint angles, center of mass, ground reaction forces, etc.

---

## References

- [run_pipeline.py Execution](RUN_PIPELINE_EXECUTION.md)
- [pipeline_config.yaml Reference](PIPELINE_CONFIG_REFERENCE.md)
- Individual stage documentation (Stage 0-4 links above)

---

**Document Version**: 1.0  
**Last Updated**: January 17, 2026  
**Status**: Production Ready ✅
