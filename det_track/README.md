# Detection & Tracking Pipeline

Simplified 5-stage pipeline for person detection, tracking, and selection.

## Pipeline Stages

### Automated Stages (0-4)

Run these sequentially using the orchestrator:

```bash
python run_pipeline.py --config configs/pipeline_config.yaml
```

**Stage 0: Video Normalization**
- Validates and normalizes video format (H.264, constant FPS)
- Ensures optimal format for detection

**Stage 1: Detection**
- YOLOv8 person detection
- Saves: `detections_raw.npz`

**Stage 2: Tracking**
- ByteTrack offline tracking
- Saves: `tracklets_raw.npz`

**Stage 3: Analysis & Ranking**
- **3a**: Compute tracklet statistics
- **3b**: Group tracklets into canonical persons
- **3c**: Rank persons by duration/coverage/position
- Saves: `tracklet_stats.npz`, `canonical_persons.npz`, `primary_person.npz`

**Stage 4: HTML Viewer Generation**
- On-demand crop extraction (no intermediate storage)
- Generates WebP animations for top 10 persons
- Creates interactive HTML viewer
- Saves: `webp_viewer/` directory with HTML + WebPs

### Manual Stage (5)

Run AFTER viewing the HTML report from Stage 4:

**Stage 5: Person Selection** *(Manual)*
```bash
# View HTML first to identify person IDs
# Then extract selected person(s):
python stage5_select_person.py --config configs/pipeline_config.yaml --persons p3

# Or select multiple persons:
python stage5_select_person.py --config configs/pipeline_config.yaml --persons p3,p7,p12
```

- Extracts selected person(s) tracklet data
- Handles overlapping frames (later person gets priority)
- Saves: `final_tracklet.npz` (detector-compatible format)

## Complete Workflow

1. **Run automated pipeline:**
   ```bash
   python run_pipeline.py --config configs/pipeline_config.yaml
   ```

2. **View HTML report:**
   - Open `demo_data/outputs/{video_name}/webp_viewer/person_selection.html`
   - Review WebP animations for each person
   - Note the person ID(s) you want (e.g., P3)

3. **Extract selected person:**
   ```bash
   python stage5_select_person.py --config configs/pipeline_config.yaml --persons p3
   ```

4. **Use extracted data:**
   - `final_tracklet.npz` contains frame_numbers and bboxes for selected person(s)
   - Ready for downstream pose estimation pipeline

## Directory Structure

```
det_track/
├── run_pipeline.py              # Orchestrator (runs stages 0-4)
├── stage0_normalize_video.py    # Video normalization
├── stage1_detect.py             # YOLO detection
├── stage2_track.py              # ByteTrack tracking
├── stage3a_analyze_tracklets.py # Compute statistics
├── stage3b_group_canonical.py   # Group tracklets
├── stage3c_rank_persons.py      # Rank persons
├── stage4_generate_html.py      # HTML viewer (on-demand extraction)
├── stage5_select_person.py      # Person selection (manual)
├── configs/
│   └── pipeline_config.yaml     # Central configuration
├── docs/                        # Documentation
├── debug/                       # Debug scripts
└── deprecated/                  # Old/deprecated code
```

## Configuration

Edit `configs/pipeline_config.yaml`:

```yaml
global:
  video_file: your_video.mp4     # Change input video
  
pipeline:
  stages:
    stage0: true    # Enable/disable each stage
    stage1: true
    stage2: true
    stage3a: true
    stage3b: true
    stage3c: true
    stage4: true
```

## Performance

Typical runtime (360 frames, 720p, T4 GPU):
- Stage 0: ~2s (normalization)
- Stage 1: ~5s (detection)
- Stage 2: ~3s (tracking)
- Stage 3a-c: ~2s (analysis)
- Stage 4: ~6s (HTML generation)
- **Total: ~18s** (excluding video I/O overhead)

Storage:
- Detections: ~10 MB
- Tracklets: ~5 MB
- Stats/Persons: ~2 MB
- WebPs: ~5-10 MB (10 persons × 50 frames)
- **Total: ~25 MB** (vs 812 MB old approach)

## Key Improvements (Phase 3)

✅ **On-demand extraction**: No intermediate crop storage (-812 MB)  
✅ **Simplified pipeline**: 5 stages vs 11 stages (-6 deprecated stages)  
✅ **Faster execution**: 18s vs 73s (-75% time)  
✅ **Clean codebase**: 562 lines orchestrator vs 695 lines (-19%)  
✅ **Clear workflow**: HTML → Select → Extract (intuitive UX)
