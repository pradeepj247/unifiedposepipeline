# Current Issues Analysis & Fix Plan

## Executive Summary
The pipeline has 5 critical issues preventing proper execution. The plan below addresses all of them systematically.

---

## Issue 1: Late-Appearance Penalty Logic is BROKEN ❌

### Problem
Debug output shows: `total_frames=10000` (a default value, not the actual video length)
- All 10 persons have start frames < 1634 (max is person_89 @ frame 1632)
- With `max_appearance_ratio=0.5`, threshold is frame 5000
- Since all persons start before frame 5000, **NONE get penalized**
- Result: All 10 persons kept, 0 removed

### Root Cause
The `total_frames` calculation in stage3c_filter_persons.py is wrong:
```python
# Current (WRONG):
video_duration = config.get('global', {}).get('video_duration_seconds', 0)
total_frames = int(video_duration * video_fps) if video_duration > 0 else 10000
```

The config doesn't have `video_duration_seconds` set, so it defaults to **10000 frames** instead of actual video length.

### Correct Logic
Must calculate `total_frames` from the actual VIDEO, not config defaults:
```python
# Get actual video metadata
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
```

### Expected Behavior
- Actual cricket video has ~360 frames (known from earlier testing)
- With `max_appearance_ratio=0.5`, threshold = frame 180
- Person_87 starts at frame 1534 → way beyond 50% → gets penalized
- Person_89 starts at frame 1632 → way beyond 50% → gets penalized
- Result: 8 persons kept (2 removed by penalty)

---

## Issue 2: Stage Parsing is COMPLETELY BROKEN ❌

### Problem
User ran: `python run_pipeline.py --config configs/pipeline_config.yaml --stages 3b,3c,3d,4`

Output shows: `Running pipeline stages: 5, 6, 7, 4`

Pipeline executed: **Stage 3b → 3c → 3d → 3a** (then 3a again?!)

### Root Cause
Stage key parsing is still broken. The shorthand-to-full-key conversion fixed "3c" → "stage3c" but something else is wrong.

The output says "stages: 5, 6, 7, 4" which suggests:
- "3b" → parsed as number 5 (wrong offset)
- "3c" → parsed as number 6 (wrong offset)
- "3d" → parsed as number 7 (wrong offset)  
- "4" → parsed as number 4 (correct)

### Root Cause Analysis
Need to check run_pipeline.py line 215-230:
```python
for spec in stage_specs:
    try:
        stage_num = int(spec)  # "3b" fails, "3c" fails, "3d" fails
        if 1 <= stage_num <= len(all_stages):
            stages.append(all_stages[stage_num - 1])  # This uses 1-based indexing
    except ValueError:
        # Try as stage key - handle shorthand (3c, 3d)
        search_key = spec if spec.startswith('stage') else f'stage{spec}'
        # This SHOULD convert "3b" → "stage3b"
```

**Hypothesis**: The shorthand conversion isn't working, OR the stage list is incomplete.

### Solution
Need to debug stage parsing by:
1. Verify all_stages list has correct entries
2. Test conversion logic: "3b" → "stage3b" → find in list
3. Add debug prints to confirm matching

---

## Issue 3: Stage Execution Order is WRONG ❌

### Problem
Even after stages are parsed, they execute in wrong order:
- Expected: 3b → 3c → 3d → 4
- Actual: 3b → 3c → 3d → 3a

Stage 3a ran unexpectedly and after stages we didn't request.

### Root Cause
Likely in run_pipeline.py where stages are filtered/ordered.

Need to check if Stage 3a is:
1. Being auto-enabled despite not being requested
2. Being re-run due to dependency logic
3. Being appended to stage list somewhere

---

## Issue 4: Output Files NOT SAVED/LISTED ❌

### Problem
Final output shows:
```
📦 Output Files:
  ✅ canonical_persons.npz (0.16 MB)
  ✅ grouping_log.json (0.01 MB)
  ⚠️  A (not found)
  ⚠️  A (not found)
  ✅ tracklet_stats.npz (0.17 MB)
```

Missing:
- `final_crops.pkl` (should be saved by Stage 3c)
- `canonical_persons_filtered.npz` (should be saved by Stage 3c)
- WebP files (not created since Stage 4 didn't run)
- HTML file (not created since Stage 4 didn't run)

### Root Cause
1. **Stage 3c** might not be saving `final_crops.pkl` correctly
2. **Output file list** in run_pipeline.py might be looking for wrong filenames
3. **Stage 4 never ran**, so no WebP/HTML files

---

## Issue 5: Stage 4 Did NOT RUN ❌

### Problem
User requested: `--stages 3b,3c,3d,4`
Only executed: 3b, 3c, 3d, 3a

Stage 4 (Visualization) completely skipped.

### Root Cause
Likely one of:
1. Stage 4 was filtered out by stage parsing bug
2. Stage 4 had a dependency check that failed
3. Stage 4 outputs already exist (skip logic triggered)

---

## Data Flow Problem Summary

According to PIPELINE_RESTRUCTURE_PLAN.md:

**Stage 3c should output:**
- `canonical_persons.npz` (v2, filtered to 8 persons)
- `final_crops.pkl` (crops for 8 persons)

**Stage 3d should read:**
- `canonical_persons.npz` (v2 from 3c)
- `final_crops.pkl` (from 3c)

**Stage 3d should output:**
- `canonical_persons.npz` (v3, merged to 6 persons) - **OVERWRITES v2**
- `final_crops.pkl` (merged crops) - **OVERWRITES 3c version**

**Stage 4 should read:**
- `canonical_persons.npz` (v3 from 3d)
- `final_crops.pkl` (from 3d)

**Current Problem**: File naming is correct in theory, but:
1. Stage 3c saves with wrong keys (might be using v1 name instead of v2)
2. Stage 3d config fallbacks using wrong paths
3. Stage 4 never runs so can't verify

---

# ACTION PLAN

## Phase 1: FIX TOTAL_FRAMES CALCULATION (Issue 1) 🔧

**Files to modify**: `det_track/stage3c_filter_persons.py`

**Changes**:
1. Calculate `total_frames` from actual video using cv2
2. Log actual frame count for debugging
3. Re-run Stage 3c to verify penalty filtering now removes 2 persons

**Expected outcome**: Persons 87 and 89 get penalized, output shows 8 persons instead of 10

---

## Phase 2: DEBUG STAGE PARSING (Issue 2) 🔧

**Files to modify**: `det_track/run_pipeline.py`

**Changes**:
1. Add debug prints to stage parsing loop
2. Print: input spec, converted key, matched stage
3. Verify stage list is complete
4. Test that "3b", "3c", "3d" all match correctly

**Expected outcome**: Confirm stages are parsed correctly, see correct stage numbers printed

---

## Phase 3: FIX STAGE EXECUTION ORDER (Issue 3) 🔧

**Files to modify**: `det_track/run_pipeline.py`

**Changes**:
1. Fix any auto-enable logic for Stage 3a
2. Ensure stages execute in requested order
3. Verify Stage 4 is included in final stage list

**Expected outcome**: Pipeline executes in correct order: 3b → 3c → 3d → 4

---

## Phase 4: VERIFY OUTPUT FILES (Issue 4) 🔧

**Files to modify**: 
- `det_track/stage3c_filter_persons.py` (verify save logic)
- `det_track/run_pipeline.py` (verify output file list)

**Changes**:
1. Add debug log in stage3c after saving `final_crops.pkl`
2. Verify file exists and has size > 0
3. Update output file list in run_pipeline to look for correct filenames
4. Remove the "⚠️ A (not found)" errors

**Expected outcome**: Output file list shows `final_crops.pkl` and other files

---

## Phase 5: VERIFY STAGE 4 RUNS (Issue 5) 🔧

**Files to modify**: `det_track/run_pipeline.py`

**Changes**:
1. After fixing stage parsing, Stage 4 should automatically run
2. Verify Stage 4 completes successfully
3. Verify WebP and HTML files generated

**Expected outcome**: Stage 4 executes after Stage 3d, generates WebP and HTML files

---

# STEP-BY-STEP IMPLEMENTATION ORDER

1. **First**: Fix Issue 1 (total_frames) - this is the most obvious bug
2. **Then**: Add debug output to see stage parsing (Issue 2 diagnosis)
3. **Then**: Fix stage parsing based on debug output (Issue 2 fix)
4. **Then**: Verify execution order (Issue 3)
5. **Then**: Fix output file saving (Issue 4)
6. **Finally**: Verify Stage 4 runs (Issue 5)

---

# SUCCESS CRITERIA FOR THIS FIX

✅ Stage 3c output shows **8 persons** instead of 10 (penalty filtering works)
✅ Pipeline executes in correct order: 3b → 3c → 3d → 4
✅ Output file list shows `final_crops.pkl` and all expected files
✅ Stage 4 completes successfully
✅ No "⚠️ A (not found)" errors
✅ Run time should be faster (only 8 persons to crop, not 10)

---

# DISCUSSION POINTS FOR USER APPROVAL

1. **Is the late-appearance penalty logic correct?**
   - Should persons starting after 50% of video be penalized?
   - Should penalty threshold be 0.7 (removes if penalty < 0.7)?
   - For actual 360-frame video, threshold would be frame 180

2. **Should output files be named differently for Stage 3c vs 3d?**
   - Current plan: Both use same filenames (3d overwrites 3c)
   - Alternative: Use version numbers (canonical_persons_v2.npz, canonical_persons_v3.npz)

3. **What should happen if Stage 4 detects missing inputs?**
   - Current: Falls back to old paths, might read old data
   - Should it fail loudly instead?

