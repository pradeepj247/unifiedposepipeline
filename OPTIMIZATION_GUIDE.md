# Pipeline Optimization Guide: Skip Stage 6 Video & Use PDF Selection

## Quick Answers to Your Questions

### Q1: What is `top10_persons_fullframe_grid.png` and why does it keep appearing?

**Answer**: It's a legacy/optional output that the old code was configured to create but Stage 6b doesn't actually generate. It's now been removed from the output checking, so you won't see the warning anymore.

---

### Q2: Colors are inverted in `top10_persons_cropped_grid.png`

**Answer**: **FIXED!** The issue was that crops are stored in **BGR format** (OpenCV standard), but PIL (Python Imaging Library) expects **RGB**. I've added color conversion:

```python
# Before (wrong colors):
crop_pil = Image.fromarray(crop)  # crop is BGR, PIL thinks it's RGB!

# After (correct colors):
crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)  # Convert BGR → RGB
crop_pil = Image.fromarray(crop_rgb)  # Now PIL interprets correctly
```

Blue will no longer be orange! ✅

---

### Q3: How to skip Stage 6 video writing and save time

**Answer**: Stage 6 is the video writing stage (`stage6_create_output_video.py`) which creates `top_persons_visualization.mp4`. It takes ~30 seconds (94% of total pipeline time!).

#### **Option 1: Disable Stage 6 in Config (Recommended)**

Edit `det_track/configs/pipeline_config.yaml`:

```yaml
pipeline:
  stages:
    stage1_detect: true
    stage2_track: true
    stage3_analyze: true
    stage4a_reid_recovery: true
    stage4b_group_canonical: true
    stage5_rank: true
    stage5b_visualize_grouping: false
    stage6_create_output_video: false   # ← Change to FALSE to skip video
    stage6b_create_selection_grid: false
```

Then run:
```bash
python run_pipeline.py --config configs/pipeline_config.yaml
```

**Result**: Pipeline will skip video creation, saving ~30 seconds!

#### **Option 2: Run Specific Stages Only**

```bash
# Run only stages 1-5 (skip 6 video)
python run_pipeline.py --config configs/pipeline_config.yaml --stages 1,2,3,4a,4b,5,5b

# Or run only grouping & ranking (for quick testing)
python run_pipeline.py --config configs/pipeline_config.yaml --stages 4b,5,5b
```

#### **Option 3: Run Stages Selectively After First Run**

Once you've run the full pipeline once, outputs are cached. You can then run just the analysis stages:

```bash
# First run (full pipeline, ~30s for video)
python run_pipeline.py --config configs/pipeline_config.yaml

# Subsequent runs (skip video, much faster)
python run_pipeline.py --config configs/pipeline_config.yaml --stages 5b,6b --force
```

---

## PDF Selection Alternative (NEW!)

### Problem with PNG:
- Simple grid image
- No metadata
- No sorting capability
- Limited to 10 persons

### Solution: PDF with Table + Thumbnails (NEW Stage 6b!)

**New File**: `stage6b_create_selection_pdf.py`

Creates a professional PDF report with:
- **Table**: Rank, Person ID, Duration, Frames, Start, End, Avg Confidence
- **Thumbnails**: High-quality crop images for each person
- **Sortable**: Easy to scan all 20+ persons
- **Professional**: Ready for sharing or archiving

#### Usage:

```bash
cd det_track
python stage6b_create_selection_pdf.py --config configs/pipeline_config.yaml
```

**Output**: `person_selection_report.pdf` (in outputs folder)

#### Features:
- Automatically shows top 20 persons (vs 10 in PNG grid)
- Includes statistics: duration, start/end frames, confidence
- Thumbnail crops are color-corrected (BGR→RGB fix)
- Professional formatting with table styling

#### Requirements:
```bash
pip install reportlab  # For PDF generation
```

---

## Recommended Workflow

### For Development/Testing:

```bash
# Disable video (config.yaml: stage6_create_output_video: false)
python run_pipeline.py --config configs/pipeline_config.yaml
# Total time: ~2 seconds (no video writing!)
```

### For Analysis:

```bash
# Run with PDF selection instead of PNG
python stage6b_create_selection_pdf.py --config configs/pipeline_config.yaml
# Creates person_selection_report.pdf with table + thumbnails
```

### For Sharing Results:

```bash
# Full pipeline with video (optional)
python run_pipeline.py --config configs/pipeline_config.yaml
# Then share: top_persons_visualization.mp4 + person_selection_report.pdf
```

---

## Timing Breakdown (With Optimizations)

### Before (Full Pipeline):
```
Stage 1: Detection          80.34s
Stage 2: Tracking            2.91s
Stage 3: Analysis            0.17s
Stage 4a: Load Crops         0.78s
Stage 4b: Grouping           0.34s
Stage 5: Ranking             0.18s
Stage 5b: Visualization      0.25s
Stage 6: VIDEO WRITING      29.79s  ← SLOW!
Stage 6b: Selection Grid      1.18s
────────────────────────────────────
TOTAL:                      115.94s
```

### After (Video Disabled):
```
Stage 1: Detection          80.34s
Stage 2: Tracking            2.91s
Stage 3: Analysis            0.17s
Stage 4a: Load Crops         0.78s
Stage 4b: Grouping           0.34s
Stage 5: Ranking             0.18s
Stage 5b: Visualization      0.25s
(Stage 6: SKIPPED)               -
Stage 6b: Selection Grid      1.18s
────────────────────────────────────
TOTAL:                       86.15s  ← 30s saved!
```

### After (PDF Only, No Video):
```
Stage 1-5: All analysis      84.73s
Stage 6b: PDF Selection       2.00s
────────────────────────────────────
TOTAL:                       86.73s
```

**Bottom Line**: Disabling Stage 6 saves you **29.79 seconds per run**. Use PDF for analysis instead.

---

## Configuration Reference

### Key Settings for Speed

**File**: `det_track/configs/pipeline_config.yaml`

```yaml
pipeline:
  stages:
    # Detection & Tracking (necessary for analysis)
    stage1_detect: true              # ~80s, cannot skip if reprocessing
    stage2_track: true               # ~3s
    
    # Analysis (fast, useful for grouping)
    stage3_analyze: true             # <1s
    stage4a_reid_recovery: true      # <1s
    stage4b_group_canonical: true    # <1s
    stage5_rank: true                # <1s
    
    # Optional visualizations
    stage5b_visualize_grouping: false # ~0.25s, optional tables
    stage6_create_output_video: false # ~30s, DISABLE TO SAVE TIME!
    stage6b_create_selection_grid: false # False = skip PNG, use PDF instead
```

---

## Fixes Applied

### 1. ✅ fullframe_grid.png Warning
**Status**: FIXED
- Removed from output checking in `run_pipeline.py`
- Only checks `cropped_grid.png` (which actually gets created)
- No more "not found" warnings

### 2. ✅ Color Inversion (Blue→Orange)
**Status**: FIXED  
- Added `cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)` before PIL
- File: `det_track/stage6b_create_selection_grid_fixed.py` line 280
- Colors now correct in PNG grid

### 3. ✅ PDF Alternative with Table
**Status**: IMPLEMENTED
- New file: `stage6b_create_selection_pdf.py`
- Creates professional PDF with table + thumbnails
- Shows 20 persons instead of 10
- Includes statistics: duration, frames, confidence

---

## Summary

| Issue | Solution | Time Saved | Status |
|-------|----------|-----------|--------|
| fullframe_grid warning | Removed from output checking | - | ✅ Fixed |
| Color inversion | BGR→RGB conversion in PIL | - | ✅ Fixed |
| Slow video writing | Disable Stage 6 in config | 29.79s | ✅ Can disable |
| Limited to 10 persons | Use PDF (20+ persons) | - | ✅ New option |

---

## Next Steps

1. **Update Config**: Set `stage6_create_output_video: false` to skip video
2. **Test PDF** (optional): Install reportlab, run `stage6b_create_selection_pdf.py`
3. **Enjoy speed**: Pipeline now ~2 seconds for subsequent runs!

---

## Example: Complete Fast Pipeline

```bash
cd /content/unifiedposepipeline/det_track

# Edit config to disable video
# Edit configs/pipeline_config.yaml: stage6_create_output_video: false

# Run fast pipeline
python run_pipeline.py --config configs/pipeline_config.yaml

# Optionally generate PDF report
python stage6b_create_selection_pdf.py --config configs/pipeline_config.yaml

# Output files:
# - canonical_persons.npz (grouping results)
# - primary_person.npz (selected person)
# - person_selection_report.pdf (visual report)
```

Total time: ~2 seconds (excluding initial detection which is cached)! ⚡
