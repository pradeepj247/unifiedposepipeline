# Phase 3 Cleanup Plan - Pipeline Simplification

**Goal:** Finalize the simplified 5-stage pipeline (0→1→2→3→4) by removing deprecated code and renumbering Stage 10b to Stage 4.

**Status:** 🟡 Ready for execution
**Created:** 2026-01-14
**Estimated Time:** 1-2 hours

---

## Overview

### Current State (Messy):
```
✅ stage0_normalize.py           (Stage 0)
✅ stage1_detect.py               (Stage 1)
✅ stage2_track.py                (Stage 2)
✅ stage3a_analyze_tracklets.py   (Stage 3a)
✅ stage3b_canonical_grouping.py  (Stage 3b)
✅ stage3c_rank_persons.py        (Stage 3c)
❌ stage4_load_crops_cache.py     [DEPRECATED]
❌ stage4b_reorganize_crops.py    [DEPRECATED]
❌ stage10_generate_webps.py      [DEPRECATED - old method]
🔧 stage10b_ondemand_webps.py     (Needs renumber → stage4)
❌ stage11_create_selection_grid.py [DEPRECATED]
```

### Target State (Clean):
```
✅ stage0_normalize.py
✅ stage1_detect.py
✅ stage2_track.py
✅ stage3a_analyze_tracklets.py
✅ stage3b_canonical_grouping.py
✅ stage3c_rank_persons.py
✅ stage4_generate_html.py        (renamed from stage10b)
✅ ondemand_crop_extraction.py    (utility, no renumber needed)
```

---

## Phase 1: Inventory & Verification

### 1.1 Identify All Deprecated Files

**Action:** Search for all files related to deprecated stages

```bash
# Files to ARCHIVE (move to deprecated/ folder):
det_track/stage4_load_crops_cache.py
det_track/stage4b_reorganize_crops.py
det_track/stage10_generate_webps.py
det_track/stage11_create_selection_grid.py
det_track/test_ondemand_extraction.py          # Test file
det_track/check_top_persons.py                 # Debug script
det_track/test_data/                           # Test data folder
det_track/test_output/                         # Test output folder
```

**Verification Steps:**
- [ ] Confirm no active imports of these files in current pipeline
- [ ] Check git history to ensure no critical logic needs extraction
- [ ] Verify test files are truly temporary

---

## Phase 2: File Operations

### 2.1 Rename Stage 10b → Stage 4

**Primary rename:**
```bash
git mv det_track/stage10b_ondemand_webps.py det_track/stage4_generate_html.py
```

**Update internal references in stage4_generate_html.py:**
- [ ] Line 1: Module docstring
- [ ] Line 113: PipelineLogger name
- [ ] Any other self-references

**Keep unchanged:**
- `ondemand_crop_extraction.py` - utility module, no stage number

### 2.2 Create Archive Folder

```bash
mkdir det_track/deprecated
```

### 2.3 Archive Deprecated Stage Files

```bash
# Move deprecated stages to archive
git mv det_track/stage4_load_crops_cache.py det_track/deprecated/
git mv det_track/stage4b_reorganize_crops.py det_track/deprecated/
git mv det_track/stage10_generate_webps.py det_track/deprecated/
git mv det_track/stage11_create_selection_grid.py det_track/deprecated/
```

**Note:** These files have `[DEPRECATED]` warnings already added in Phase 3A.

### 2.4 Archive Temporary Test Files

```bash
# If these exist in git:
git mv det_track/test_ondemand_extraction.py det_track/deprecated/ 2>/dev/null || true
git mv det_track/check_top_persons.py det_track/deprecated/ 2>/dev/null || true

# If local only:
mv det_track/test_ondemand_extraction.py det_track/deprecated/ 2>/dev/null || true
mv det_track/check_top_persons.py det_track/deprecated/ 2>/dev/null || true
mv det_track/test_data det_track/deprecated/ 2>/dev/null || true
mv det_track/test_output det_track/deprecated/ 2>/dev/null || true
```

### 2.5 Create Archive README

Create `det_track/deprecated/README.md`:

```markdown
# Deprecated Pipeline Components

**Archive Date:** 2026-01-14
**Reason:** Phase 3 pipeline simplification (5-stage architecture)

## Archived Files

### Stage Files (replaced by on-demand extraction):
- `stage4_load_crops_cache.py` - Replaced by on-demand extraction in stage4
- `stage4b_reorganize_crops.py` - No longer needed
- `stage10_generate_webps.py` - Replaced by stage4_generate_html.py
- `stage11_create_selection_grid.py` - Merged into stage4_generate_html.py

### Test/Debug Files:
- `test_ondemand_extraction.py` - Development testing
- `check_top_persons.py` - Debug script
- `test_data/` - Test data folder
- `test_output/` - Test output folder

## Why Archived?

These files implemented the old pipeline approach:
1. Extract ALL crops during detection (812 MB storage)
2. Load crops from cache
3. Generate WebPs from cached crops
4. Separate HTML generation

New approach:
1. Detection without crop extraction
2. On-demand crop extraction for top N persons only
3. Combined WebP + HTML generation
4. Zero intermediate storage (808 MB saved)

## Can I Delete This Folder?

Yes, after 1-2 weeks of stable operation with the new pipeline.

## Restore Instructions

If you need to revert to old approach:
```bash
git log --all --full-history -- "**/stage10_generate_webps.py"
git checkout <commit-hash> -- det_track/stage10_generate_webps.py
```
```

---

## Phase 3: Configuration Updates

### 3.1 Update pipeline_config.yaml

**File:** `det_track/configs/pipeline_config.yaml`

**Changes needed:**

#### A. Stage Control Section (~line 46-56)
```yaml
# OLD:
pipeline:
  stages:
    stage0: true
    stage1: true
    stage2: true
    stage3a: true
    stage3b: true
    stage3c: true
    stage4: false          # ← REMOVE (deprecated)
    stage4b: false         # ← REMOVE (deprecated)
    stage10: false         # ← REMOVE (deprecated)
    stage10b_ondemand: true  # ← RENAME to stage4
    stage11: false         # ← REMOVE (deprecated)

# NEW:
pipeline:
  stages:
    stage0: true
    stage1: true
    stage2: true
    stage3a: true
    stage3b: true
    stage3c: true
    stage4: true           # ← Renamed from stage10b_ondemand
```

#### B. Remove Deprecated Config Blocks
Delete entire sections:
- [ ] `stage4_load_crops:` section (~line 200-230)
- [ ] `stage4b_reorganize:` section (~line 231-260)
- [ ] `stage10_generate_webps:` section (~line 350-390)
- [ ] `stage11_create_grid:` section (~line 420-450)

#### C. Rename stage10b_ondemand Section (~line 396-418)
```yaml
# OLD:
stage10b_ondemand:
  enabled: true
  video_file: ${outputs_dir}/${current_video}/canonical_video.mp4
  ...

# NEW:
stage4_generate_html:
  enabled: true
  video_file: ${outputs_dir}/${current_video}/canonical_video.mp4
  ...
```

### 3.2 Update Stage Output Mappings

**File:** `det_track/configs/pipeline_config.yaml`

Check if any `output:` fields reference `stage10b_ondemand_file` or similar. Update to `stage4_output_file`.

---

## Phase 4: Orchestrator Updates

### 4.1 Update run_pipeline.py

**File:** `det_track/run_pipeline.py`

**Line ~292:** Update stage list tuple

```python
# OLD:
all_stages = [
    (stage0_normalize, 'stage0', 'Stage 0: Video Normalization'),
    (stage1_detect, 'stage1', 'Stage 1: YOLO Detection'),
    (stage2_track, 'stage2', 'Stage 2: ByteTrack Tracking'),
    (stage3a_analyze, 'stage3a', 'Stage 3a: Tracklet Analysis'),
    (stage3b_grouping, 'stage3b', 'Stage 3b: Enhanced Canonical Grouping'),
    (stage3c_ranking, 'stage3c', 'Stage 3c: Person Ranking'),
    (stage4_load, 'stage4', 'Stage 4: Load Crops Cache'),              # ← REMOVE
    (stage4b_reorg, 'stage4b', 'Stage 4b: Reorganize Crops'),          # ← REMOVE
    (stage10_webps, 'stage10', 'Stage 10: Generate Person WebPs'),     # ← REMOVE
    (stage10b_ondemand, 'stage10b_ondemand', 'Stage 10b: On-Demand WebP Generation (Phase 3)'),  # ← RENAME
    (stage11_grid, 'stage11', 'Stage 11: Create Selection Grid'),      # ← REMOVE
]

# NEW:
all_stages = [
    (stage0_normalize, 'stage0', 'Stage 0: Video Normalization'),
    (stage1_detect, 'stage1', 'Stage 1: Detection'),
    (stage2_track, 'stage2', 'Stage 2: Tracking'),
    (stage3a_analyze, 'stage3a', 'Stage 3a: Tracklet Analysis'),
    (stage3b_grouping, 'stage3b', 'Stage 3b: Canonical Grouping'),
    (stage3c_ranking, 'stage3c', 'Stage 3c: Person Ranking'),
    (stage4_html, 'stage4', 'Stage 4: Generate HTML Viewer'),
]
```

**Line ~585-640:** Update stage10b_ondemand handler to stage4 handler

```python
# OLD (~line 595):
elif stage_key == 'stage10b_ondemand':
    stage_config = config.get('stage10b_ondemand', {})
    # ...

# NEW:
elif stage_key == 'stage4':
    stage_config = config.get('stage4_generate_html', {})
    # ...
```

**Remove deprecated handlers:**
- [ ] Delete `stage4` handler (load crops cache)
- [ ] Delete `stage4b` handler (reorganize crops)
- [ ] Delete `stage10` handler (old webp generation)
- [ ] Delete `stage11` handler (selection grid)

### 4.2 Update Import Statements

**File:** `det_track/run_pipeline.py`

```python
# OLD (~line 15-30):
from stage4_load_crops_cache import main as stage4_load
from stage4b_reorganize_crops import main as stage4b_reorg
from stage10_generate_webps import main as stage10_webps
from stage10b_ondemand_webps import main as stage10b_ondemand
from stage11_create_selection_grid import main as stage11_grid

# NEW:
from stage4_generate_html import main as stage4_html
```

---

## Phase 5: Documentation Updates

### 5.1 Update README.md

**File:** `det_track/README.md`

Update pipeline overview section to show simplified 5-stage flow:

```markdown
## Pipeline Stages

0. **Video Normalization** - Validate and normalize input video
1. **Detection** - YOLO object detection
2. **Tracking** - ByteTrack multi-object tracking
3. **Analysis & Ranking** - Tracklet analysis, grouping, ranking
4. **HTML Generation** - On-demand crop extraction + WebP viewer
```

Remove all references to:
- Stage 4 (Load Crops Cache)
- Stage 4b (Reorganize Crops)
- Stage 10 (Old WebP generation)
- Stage 11 (Selection Grid)

### 5.2 Update PIPELINE_DESIGN.md

**Files to update:**
- `det_track/docs/PIPELINE_DESIGN.md` - Remove deprecated stages
- `det_track/docs/STAGE10B_INTEGRATION.md` - Rename to `STAGE4_HTML_VIEWER.md`
- `det_track/IMPLEMENTATION_COMPLETE.md` - Update stage list

### 5.3 Update Main README

**File:** `README.md` (root)

Update quick start and architecture sections to reflect 5-stage pipeline.

---

## Phase 6: Testing & Validation

### 6.1 Syntax Check

```bash
cd det_track
python -m py_compile stage4_generate_html.py
python -m py_compile run_pipeline.py
```

### 6.2 Config Validation

```bash
python -c "import yaml; yaml.safe_load(open('configs/pipeline_config.yaml'))"
```

### 6.3 Import Check

```bash
cd det_track
python -c "from stage4_generate_html import main; print('✅ Import OK')"
```

### 6.4 Full Pipeline Test (Colab)

```bash
cd /content/unifiedposepipeline/det_track
python run_pipeline.py --config configs/pipeline_config.yaml
```

**Expected output:**
```
Running enabled stages: stage0, stage1, stage2, stage3a, stage3b, stage3c, stage4
...
✅ Stage 4: Generate HTML Viewer completed in X.XXs
```

### 6.5 Verify Archive

Ensure old stages moved to archive:
```bash
# Should NOT exist in main directory:
ls det_track/stage4_load_crops_cache.py 2>/dev/null && echo "ERROR: Not moved!" || echo "✅ Moved"
ls det_track/stage4b_reorganize_crops.py 2>/dev/null && echo "ERROR: Not moved!" || echo "✅ Moved"
ls det_track/stage10_generate_webps.py 2>/dev/null && echo "ERROR: Not moved!" || echo "✅ Moved"
ls det_track/stage11_create_selection_grid.py 2>/dev/null && echo "ERROR: Not moved!" || echo "✅ Moved"

# Should exist in archive:
ls det_track/deprecated/stage4_load_crops_cache.py || echo "ERROR: Not archived!"
ls det_track/deprecated/stage10_generate_webps.py || echo "ERROR: Not archived!"
ls det_track/deprecated/README.md || echo "ERROR: No archive README!"
```

---

## Phase 7: Git Cleanup & Commit

### 7.1 Stage Changes

```bash
cd d:/trials/unifiedpipeline/newrepo

# Create archive folder
mkdir det_track/deprecated

# Rename active stage
git mv det_track/stage10b_ondemand_webps.py det_track/stage4_generate_html.py

# Archive deprecated stages
git mv det_track/stage4_load_crops_cache.py det_track/deprecated/
git mv det_track/stage4b_reorganize_crops.py det_track/deprecated/
git mv det_track/stage10_generate_webps.py det_track/deprecated/
git mv det_track/stage11_create_selection_grid.py det_track/deprecated/

# Add archive README
git add det_track/deprecated/README.md

# Modified files (stage later)
git add det_track/configs/pipeline_config.yaml
git add det_track/run_pipeline.py
git add det_track/stage4_generate_html.py
git add det_track/README.md
git add README.md
```

### 7.2 Commit Strategy

**Option A: Single atomic commit**
```bash
git commit -m "Phase 3 Cleanup: Simplify to 5-stage pipeline

- Rename: stage10b_ondemand → stage4_generate_html
- Archive deprecated stages to deprecated/ folder: 4, 4b, 10, 11
- Update config to use stage4 naming
- Update orchestrator and documentation
- Archive test files and temporary scripts

Pipeline now follows clean 0→1→2→3→4 flow:
  0. Video Normalization
  1. Detection
  2. Tracking
  3. Analysis & Ranking
  4. HTML Generation

Deprecated files preserved in det_track/deprecated/ for reference.
Closes Phase 3 implementation."
```

**Option B: Multi-step commits** (if you want history separation)
```bash
# Commit 1: Rename
git commit -m "Rename stage10b_ondemand to stage4_generate_html"

# Commit 2: Delete deprecated
git commit -m "Remove deprecated stages 4, 4b, 10, 11"

# Commit 3: Update config
git commit -m "Update config for simplified 5-stage pipeline"

# Commit 4: Update orchestrator
git commit -m "Update run_pipeline.py for stage4 naming"

# Commit 5: Documentation
git commit -m "Update documentation for simplified pipeline"
```

### 7.3 Push to GitHub

```bash
git push origin main
```

---

## Phase 8: Post-Cleanup Validation

### 8.1 Colab Re-test

Clone fresh repo and run full pipeline:
```bash
cd /content
rm -rf unifiedposepipeline
git clone https://github.com/pradeepj247/unifiedposepipeline.git
cd unifiedposepipeline/det_track
python run_pipeline.py --config configs/pipeline_config.yaml
```

### 8.2 Verify Outputs

Check that all expected files are generated:
- [ ] `canonical_video.mp4`
- [ ] `detections_raw.npz`
- [ ] `tracklets_raw.npz`
- [ ] `tracklet_stats.npz`
- [ ] `canonical_persons.npz`
- [ ] `primary_person.npz`
- [ ] `ranking_report.json`
- [ ] `webp_ondemand/viewer.html` (or renamed path)

### 8.3 Verify HTML Viewer

- [ ] Open `viewer.html` in browser
- [ ] Check WebP animations load
- [ ] Verify rank badges and stats display correctly

---

## Risk Mitigation

### Backup Strategy
Before starting Phase 2 (file operations):
```bash
cd d:/trials/unifiedpipeline/newrepo
git branch phase3-cleanup-backup
```

If something breaks:
```bash
git checkout main
git reset --hard phase3-cleanup-backup
```

### Rollback Points

After each phase, create a checkpoint:
```bash
git add -A
git commit -m "Checkpoint: Phase X complete"
```

---

## Success Criteria

- [ ] No deprecated files in main `det_track/` directory
- [ ] Deprecated files moved to `det_track/deprecated/` with README
- [ ] Config only has stages 0, 1, 2, 3a, 3b, 3c, 4
- [ ] Pipeline runs successfully with clean output
- [ ] Stage numbering is intuitive (0→1→2→3→4)
- [ ] Documentation reflects simplified architecture
- [ ] All tests pass
- [ ] HTML viewer works correctly
- [ ] Timing performance maintained (~66s total)

---

## Execution Checklist

Ready to execute? Check off as you go:

- [ ] **Phase 1:** Inventory & Verification
- [ ] **Phase 2:** File Operations (rename, delete)
- [ ] **Phase 3:** Configuration Updates
- [ ] **Phase 4:** Orchestrator Updates
- [ ] **Phase 5:** Documentation Updates
- [ ] **Phase 6:** Testing & Validation
- [ ] **Phase 7:** Git Cleanup & Commit
- [ ] **Phase 8:** Post-Cleanup Validation

---

## Notes & Issues Log

*(Use this section to track any issues encountered during cleanup)*

**Issue 1:**
- **Problem:** 
- **Solution:** 
- **Status:** 

---

**End of Cleanup Plan**
