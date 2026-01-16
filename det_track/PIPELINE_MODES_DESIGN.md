# Pipeline Modes Design Document

**Created:** January 16, 2026  
**Last Updated:** January 16, 2026  
**Status:** Pre-Implementation (Prerequisites Complete ✅)  
**Purpose:** Add configurable fast/balanced/full modes to optimize pipeline performance

---

## 0. Prerequisites Verification ✅

### File Renaming Completed (Jan 16, 2026)
✅ **Stage 3c outputs renamed for consistency:**
- `canonical_persons_filtered.npz` → `canonical_persons_3c.npz`
- `final_crops.pkl` → `final_crops_3c.pkl`
- `stage3c_filtering.json` → `stage3c_sidecar.json`

✅ **Stage 3d outputs already use _3d suffix:**
- `canonical_persons_3d.npz`
- `final_crops_3d.pkl`
- `merging_report.json` (functional, read by Stage 4)
- `stage3d_sidecar.json`

✅ **All impacts verified:**
- Config file updated (3 sections: 3c output, 3d input, 4 input)
- Stage 3d reads inputs via config keys (lines 360-361)
- Stage 4 reads inputs via config keys (lines 114-117)
- No hardcoded paths found

### Configurability Verified ✅
✅ **`crops_per_person` is already configurable:**
- Defined in config: `stage3c_filter.crops_per_person: 50`
- Read by code: `stage3c_filter_persons.py` line 431
- **No hardcoded values** - fully dynamic

✅ **Stage 3d handles variable crop counts:**
- Uses `num_crops_per_person: 16` from config (default)
- Safely handles fewer crops: `crops[:min(num_crops, len(crops))]`
- **No breakage** if given 10, 30, or 50 crops

✅ **Stage 4 handles variable crop counts:**
- Caps at 50 max, uses all available if fewer
- Dynamically counts: `len(person_buckets_3c[person_id])`
- **No hardcoded 500 (50×10) anywhere**

✅ **Result:** Changing `crops_per_person` from 50→10 will work seamlessly across all stages without code changes!

---

## 1. Executive Summary

### Problem Statement
Current pipeline takes 77.53s to process a video. Stages 3c (14.7%), 3d (10.4%), and 4 (6.9%) account for 33% of total time. Users need flexibility to trade thoroughness for speed based on their use case.

### Solution
Implement three pipeline modes (fast, balanced, full) that configure key parameters:
- Number of crops per person (10/30/50)
- Enable/disable Stage 3d ReID merging
- Enable/disable Stage 4 dual-row HTML viewer

### Expected Benefits
- **Fast mode:** 59s (24% faster) - Quick analysis
- **Balanced mode:** 67s (14% faster) - Moderate detail
- **Full mode:** 77s (current) - Maximum thoroughness

---

## 2. Current State Analysis

### Timing Breakdown (77.53s total)
```
Stage 0: Video Normalization              0.38s   (0.5%)
Stage 1: Detection                        42.61s  (55.0%)   ← Non-negotiable
Stage 2: Tracking                         9.19s   (11.9%)   ← Non-negotiable
Stage 3a: Tracklet Analysis               0.21s   (0.3%)    ← Core, fast
Stage 3b: Canonical Grouping              0.41s   (0.5%)    ← Core, fast
Stage 3c: Filter & Extract Crops          11.37s  (14.7%)   ← OPTIMIZATION TARGET
Stage 3d: Visual Refinement (OSNet)       8.04s   (10.4%)   ← OPTIMIZATION TARGET
Stage 4: Generate HTML Viewer             5.33s   (6.9%)    ← OPTIMIZATION TARGET
```

### Optimization Targets (24.74s, 33% of total)

#### Stage 3c: 11.37s
- **Crop extraction:** 10.83s (95% of stage time)
  - Current: 500 crops (10 persons × 50 crops)
  - Time per crop: ~22ms
  - **Optimization:** Reduce crops_per_person
    - 50 → 30 crops: Save ~4.3s
    - 50 → 10 crops: Save ~8.6s

#### Stage 3d: 8.04s
- **OSNet ReID merging:** Entire stage
  - Current: Merges 10→8 persons (2 merge pairs)
  - **Optimization:** Make optional
    - Skip entirely: Save 8.04s
    - Auto-skip if persons ≤8: Conditional savings

#### Stage 4: 5.33s
- **WebP generation:** 4.89s
  - Current: 18 WebPs (10 from 3c + 8 from 3d)
  - **Optimization:** Single-row mode
    - Only 10 WebPs (skip 3c comparison): Save ~2.7s

---

## 3. Proposed Design

### 3.1 Mode Definitions

| Mode      | Time  | Crops/Person | Stage 3d | Stage 4 Dual Row | Use Case                |
|-----------|-------|--------------|----------|------------------|-------------------------|
| **fast**  | 59s   | 10           | ❌       | ❌               | Quick preview           |
| **balanced** | 67s | 30           | ❌       | ❌               | Standard analysis       |
| **full**  | 77s   | 50           | ✅       | ✅               | Maximum detail, debug   |

### 3.2 Configuration Structure

```yaml
# configs/pipeline_config.yaml

# === NEW SECTION: Mode Configuration ===
mode: full  # default mode (fast | balanced | full)

modes:
  fast:
    description: "Quick analysis (~59s) - 10 crops, no ReID merging"
    crops_per_person: 10
    enable_stage3d: false
    stage4_dual_row: false
    
  balanced:
    description: "Moderate analysis (~67s) - 30 crops, no ReID merging"
    crops_per_person: 30
    enable_stage3d: false
    stage4_dual_row: false
    
  full:
    description: "Thorough analysis (~77s) - 50 crops, with ReID merging"
    crops_per_person: 50
    enable_stage3d: true
    stage4_dual_row: true

# === EXISTING SECTIONS (unchanged structure) ===
stage3c_filter:
  crops_per_person: 50  # Will be overridden by mode
  # ... rest of config

pipeline:
  stages:
    stage3d:
      enabled: true  # Will be overridden by mode
      # ... rest of config

# stage4_html section doesn't exist yet - will add
```

### 3.3 CLI Interface

```bash
# Use default mode from config (full)
python run_pipeline.py --config pipeline_config.yaml

# Override to fast mode
python run_pipeline.py --config pipeline_config.yaml --mode fast

# Override to balanced mode
python run_pipeline.py --config pipeline_config.yaml --mode balanced

# Explicit full mode
python run_pipeline.py --config pipeline_config.yaml --mode full
```

### 3.4 Application Logic

```python
# Pseudocode for run_pipeline.py

def apply_mode_overrides(config, mode):
    """Apply mode presets to config (non-destructive)"""
    if 'modes' not in config or mode not in config['modes']:
        logger.info(f"No mode '{mode}' defined, using config as-is")
        return config
    
    mode_settings = config['modes'][mode]
    logger.info(f"Applying mode: {mode} - {mode_settings.get('description', '')}")
    
    # Override stage 3c
    config['stage3c_filter']['crops_per_person'] = mode_settings['crops_per_person']
    
    # Override stage 3d enable/disable
    if 'pipeline' in config and 'stages' in config['pipeline']:
        if 'stage3d' not in config['pipeline']['stages']:
            config['pipeline']['stages']['stage3d'] = {}
        config['pipeline']['stages']['stage3d']['enabled'] = mode_settings['enable_stage3d']
    
    # Override stage 4 dual row (if config section exists)
    if 'stage4_html' in config:
        config['stage4_html']['dual_row'] = mode_settings['stage4_dual_row']
    
    return config

def main():
    # Parse arguments (add --mode flag)
    parser.add_argument('--mode', type=str, choices=['fast', 'balanced', 'full'],
                       help='Pipeline mode (fast/balanced/full)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Determine active mode (CLI overrides config)
    active_mode = args.mode if args.mode else config.get('mode', 'full')
    
    # Apply mode overrides
    config = apply_mode_overrides(config, active_mode)
    
    # Continue with normal pipeline execution...
```

---

## 4. Impact Analysis

### 4.1 Files to Modify

#### **HIGH IMPACT (Core Changes)**

1. **`det_track/configs/pipeline_config.yaml`**
   - **Location:** Lines 1-50 (add new mode section at top)
   - **Changes:**
     - Add `mode: full` default
     - Add `modes:` dictionary with fast/balanced/full definitions
   - **Risk:** Low (purely additive, backward compatible)

2. **`det_track/run_pipeline.py`**
   - **Location 1:** Argument parser (~line 50-80)
     - Add `--mode` CLI argument
   - **Location 2:** After config load, before stage execution (~line 180-200)
     - Add `apply_mode_overrides()` function
     - Call it after loading config
     - Print active mode
   - **Location 3:** Stage execution loop (~line 250-300)
     - Respect `enabled: false` for stage3d
   - **Risk:** Medium (modifies main execution flow)

3. **`det_track/stage4_generate_html.py`**
   - **Location 1:** Config loading (~line 100-130)
     - Add logic to read `stage4_html.dual_row` setting
   - **Location 2:** HTML generation (~line 400-500)
     - Add conditional: if dual_row, generate both 3c+3d, else only 3d
   - **Risk:** Medium (changes output behavior)

#### **LOW IMPACT (No Direct Changes, Reads Config)**

4. **`det_track/stage3c_filter_persons.py`**
   - **Changes:** NONE (already reads `crops_per_person` from config)
   - **Risk:** None

5. **`det_track/stage3d_refine_visual.py`**
   - **Changes:** NONE (enable/disable handled by run_pipeline.py)
   - **Risk:** None

### 4.2 Backward Compatibility Matrix

| Scenario | Config Has Modes? | CLI Has --mode? | Behavior |
|----------|-------------------|-----------------|----------|
| Old config, old command | ❌ | ❌ | ✅ Works as before (full mode implicit) |
| Old config, new command | ❌ | ✅ | ✅ Ignores --mode, uses config values |
| New config, old command | ✅ | ❌ | ✅ Uses config.mode default (full) |
| New config, new command | ✅ | ✅ | ✅ CLI overrides config.mode |

### 4.3 Testing Requirements

**Unit Tests:**
- `apply_mode_overrides()` with valid modes
- `apply_mode_overrides()` with missing modes (should not crash)
- CLI argument parsing

**Integration Tests:**
- Run pipeline with `--mode fast` (verify 10 crops, no stage3d)
- Run pipeline with `--mode balanced` (verify 30 crops, no stage3d)
- Run pipeline with `--mode full` (verify 50 crops, stage3d runs)
- Run old config without modes section (verify backward compatibility)

**Performance Validation:**
- Measure actual time for fast mode (~59s expected)
- Measure actual time for balanced mode (~67s expected)
- Verify full mode time unchanged (~77s)

---

## 5. Implementation Plan

### ✅ Phase 0: Prerequisites (COMPLETED)
**Goal:** Ensure file naming consistency and verify configurability

- [x] Rename Stage 3c outputs with _3c suffix
- [x] Update all config references (3c output, 3d input, 4 input)
- [x] Update all code docstrings
- [x] Verify Stage 3d reads from config (not hardcoded paths)
- [x] Verify Stage 4 reads from config (not hardcoded paths)
- [x] Confirm `crops_per_person` is configurable
- [x] Confirm no hardcoded "500" (50×10) values exist
- [x] Verify stages handle variable crop counts gracefully
- [x] Commit: "Rename Stage 3c outputs for consistency with _3c suffix"

### Phase 1: Core Infrastructure 🎯 (NEXT - Estimated 45 min)
**Goal:** Add mode system without breaking existing behavior

#### Task 1.1: Update Config File
- [ ] Open `det_track/configs/pipeline_config.yaml`
- [ ] Add mode section at top (after global section):
  ```yaml
  # === Pipeline Mode Configuration ===
  mode: full  # default mode (fast | balanced | full)
  
  modes:
    fast:
      description: "Quick analysis (~59s) - 10 crops, no ReID"
      crops_per_person: 10
      enable_stage3d: false
      stage4_dual_row: false
    balanced:
      description: "Moderate analysis (~67s) - 30 crops, no ReID"
      crops_per_person: 30
      enable_stage3d: false
      stage4_dual_row: false
    full:
      description: "Thorough analysis (~77s) - 50 crops, with ReID"
      crops_per_person: 50
      enable_stage3d: true
      stage4_dual_row: true
  ```
- [ ] Commit: "Add pipeline mode definitions to config"

#### Task 1.2: Add Mode Override Function
- [ ] Open `det_track/run_pipeline.py`
- [ ] Add `apply_mode_overrides(config, mode, logger)` function after imports (~line 40):
  ```python
  def apply_mode_overrides(config, mode, logger):
      """Apply mode presets to config (non-destructive)"""
      if 'modes' not in config or mode not in config['modes']:
          logger.info(f"⚙️  No mode '{mode}' defined, using config as-is")
          return config
      
      mode_settings = config['modes'][mode]
      logger.info(f"⚙️  Applying mode: {mode.upper()} - {mode_settings.get('description', '')}")
      
      # Override stage 3c
      config['stage3c_filter']['crops_per_person'] = mode_settings['crops_per_person']
      logger.info(f"   └─ crops_per_person: {mode_settings['crops_per_person']}")
      
      # Override stage 3d enable/disable
      if 'pipeline' in config and 'stages' in config['pipeline']:
          if 'stage3d' not in config['pipeline']['stages']:
              config['pipeline']['stages']['stage3d'] = {}
          config['pipeline']['stages']['stage3d']['enabled'] = mode_settings['enable_stage3d']
          logger.info(f"   └─ stage3d enabled: {mode_settings['enable_stage3d']}")
      
      # Override stage 4 dual row
      if 'stage4_html' not in config:
          config['stage4_html'] = {}
      config['stage4_html']['dual_row'] = mode_settings['stage4_dual_row']
      logger.info(f"   └─ stage4 dual_row: {mode_settings['stage4_dual_row']}")
      
      return config
  ```
- [ ] Commit: "Add apply_mode_overrides function"

#### Task 1.3: Add CLI Argument
- [ ] In `run_pipeline.py`, find argument parser (~line 60-80)
- [ ] Add `--mode` argument:
  ```python
  parser.add_argument('--mode', type=str, choices=['fast', 'balanced', 'full'],
                     help='Pipeline mode: fast (10 crops, no ReID), balanced (30 crops), full (50 crops + ReID)')
  ```
- [ ] Commit: "Add --mode CLI argument"

#### Task 1.4: Integrate Mode Application
- [ ] In `run_pipeline.py` main(), after config load (~line 180-200)
- [ ] Add mode application logic:
  ```python
  # Determine active mode (CLI overrides config)
  active_mode = args.mode if args.mode else config.get('mode', 'full')
  
  # Apply mode overrides
  config = apply_mode_overrides(config, active_mode, logger)
  ```
- [ ] Commit: "Integrate mode override system into pipeline"

### Phase 2: Stage 3d Enable/Disable 🎯 (Estimated 15 min)
**Goal:** Make stage3d skippable based on mode

#### Task 2.1: Verify Stage Gating Logic
- [ ] Review `run_pipeline.py` stage execution loop (~line 250-300)
- [ ] Check if `pipeline.stages.stage3d.enabled` flag is already respected
- [ ] If yes, verify fast mode skips stage3d
- [ ] If no, add check:
  ```python
  # In stage execution loop
  if stage_key == 'stage3d':
      if not config.get('pipeline', {}).get('stages', {}).get('stage3d', {}).get('enabled', True):
          logger.info("⏭️  Stage 3d: SKIPPED (disabled by mode)")
          continue
  ```
- [ ] Test: Set `enabled: false` manually, verify stage3d skips
- [ ] Commit: "Ensure stage3d respects enabled flag"

### Phase 3: Stage 4 Dual Row Toggle 🎯 (Estimated 30 min)
**Goal:** Add single-row mode for Stage 4

#### Task 3.1: Modify Stage 4 HTML Generation
- [ ] Open `det_track/stage4_generate_html.py`
- [ ] Find config loading section (~line 100-130)
- [ ] Add dual_row setting read:
  ```python
  dual_row_mode = stage_config.get('dual_row', True)  # Default to True (current behavior)
  logger.info(f"Dual-row mode: {'✅ Enabled' if dual_row_mode else '❌ Disabled'}")
  ```
- [ ] Find WebP generation section (~line 230-270)
- [ ] Wrap Stage 3c WebP generation in conditional:
  ```python
  if dual_row_mode:
      logger.info("Generating WebPs for Stage 3c...")
      # ... existing 3c WebP code ...
  else:
      logger.info("⏭️  Skipping Stage 3c WebPs (single-row mode)")
      webp_base64_dict_3c = {}  # Empty dict
  ```
- [ ] Find HTML generation section (~line 400-500)
- [ ] Add conditional for dual-row vs single-row HTML:
  ```python
  if dual_row_mode:
      # Generate dual-row HTML (existing logic)
  else:
      # Generate single-row HTML (only 3d data)
  ```
- [ ] Commit: "Add single-row mode to Stage 4 HTML viewer"

### Phase 4: Testing & Validation 🎯 (Estimated 45 min)

#### Task 4.1: Manual Testing - Fast Mode
- [ ] Run: `python run_pipeline.py --config configs/pipeline_config.yaml --mode fast`
- [ ] Verify console shows "Applying mode: FAST"
- [ ] Verify "Extracting crops for 10 persons, 10 crops each" (or similar)
- [ ] Verify "Stage 3d: SKIPPED (disabled by mode)"
- [ ] Verify HTML has only single row (no 3c comparison)
- [ ] Measure time (expect ~59s)

#### Task 4.2: Manual Testing - Balanced Mode
- [ ] Run: `python run_pipeline.py --config configs/pipeline_config.yaml --mode balanced`
- [ ] Verify "30 crops" in console
- [ ] Verify Stage 3d skipped
- [ ] Verify single-row HTML
- [ ] Measure time (expect ~67s)

#### Task 4.3: Manual Testing - Full Mode
- [ ] Run: `python run_pipeline.py --config configs/pipeline_config.yaml --mode full`
- [ ] Verify "50 crops" in console
- [ ] Verify Stage 3d runs
- [ ] Verify dual-row HTML (both 3c and 3d)
- [ ] Measure time (expect ~77s)

#### Task 4.4: Backward Compatibility Testing
- [ ] Remove `mode:` and `modes:` section from config temporarily
- [ ] Run: `python run_pipeline.py --config configs/pipeline_config.yaml`
- [ ] Should work without errors (uses config values as-is)
- [ ] Restore mode section

#### Task 4.5: Edge Case Testing
- [ ] Test invalid mode: `--mode invalid` (should error or fallback)
- [ ] Test config without mode key (should default to 'full')
- [ ] Test CLI override: config says balanced, CLI says fast (CLI should win)

#### Task 4.6: Performance Validation
- [ ] Record actual times for each mode
- [ ] Compare against estimates
- [ ] Document discrepancies if >10% difference

### Phase 5: Documentation & Polish 🎯 (Estimated 30 min)

#### Task 5.1: Update README
- [ ] Open main README.md
- [ ] Add "Pipeline Modes" section
- [ ] Document fast/balanced/full modes with timing
- [ ] Add CLI examples
- [ ] Add use case recommendations

#### Task 5.2: Update Run Script Comments
- [ ] Add mode info to run_pipeline.py header docstring
- [ ] Update examples in file comments

#### Task 5.3: Update TIMING SUMMARY Output
- [ ] In run_pipeline.py, find timing summary section
- [ ] Add active mode to summary:
  ```python
  print(f"Pipeline Mode: {active_mode.upper()}")
  print(f"Total Time: {total_time:.2f}s")
  ```

#### Task 5.4: Final Review & Commit
- [ ] Review PIPELINE_MODES_DESIGN.md (this file)
- [ ] Check all checkboxes are marked
- [ ] Test all three modes one final time
- [ ] Commit: "Complete pipeline modes implementation (fast/balanced/full)"

---

## 6. Remaining TODOs - Quick Reference

### 🎯 IMMEDIATE NEXT STEPS (In Order):

1. **Add mode config section** → Edit `pipeline_config.yaml` (5 min)
2. **Add mode override function** → Edit `run_pipeline.py` (15 min)
3. **Add --mode CLI arg** → Edit `run_pipeline.py` (5 min)
4. **Integrate mode application** → Edit `run_pipeline.py` (10 min)
5. **Verify stage3d gating** → Check/fix `run_pipeline.py` (10 min)
6. **Add dual-row toggle** → Edit `stage4_generate_html.py` (25 min)
7. **Test all 3 modes** → Run pipeline 3 times (30 min)
8. **Update documentation** → Edit README (20 min)

**Total Estimated Time:** ~2 hours

### Summary Checklist:
- [x] Prerequisites (file renaming, configurability verification)
- [ ] Phase 1: Core Infrastructure (4 tasks)
- [ ] Phase 2: Stage 3d Enable/Disable (1 task)
- [ ] Phase 3: Stage 4 Dual Row Toggle (1 task)
- [ ] Phase 4: Testing (6 tasks)
- [ ] Phase 5: Documentation (4 tasks)

**Total Tasks Remaining:** 16  
**Prerequisites Complete:** ✅ All 9 prerequisite verifications done!

---

## 7. Rollback Plan
  - Verify Stage 3d skipped
  - Measure time (~67s expected)
- [ ] Test full mode: `--mode full`
  - Verify 50 crops
  - Verify Stage 3d runs
  - Verify dual-row HTML
  - Measure time (~77s expected)
- [ ] Test backward compatibility: Run old config without modes
  - Should work without errors

#### Task 4.2: Edge Case Testing
- [ ] Test with invalid mode: `--mode invalid`
  - Should show error or fall back gracefully
- [ ] Test old config + new command with --mode
  - Should ignore mode or use defaults
- [ ] Test config without mode key
  - Should default to 'full'

#### Task 4.3: Performance Validation
- [ ] Record actual times for each mode
- [ ] Compare against estimates (59s/67s/77s)
- [ ] Document any discrepancies

### Phase 5: Documentation & Polish ✅ (TO DO)

#### Task 5.1: Update README
- [ ] Add "Pipeline Modes" section to README.md
- [ ] Document fast/balanced/full modes
- [ ] Add CLI examples
- [ ] Document expected timing for each mode

#### Task 5.2: Update Console Output
- [ ] Add mode announcement at pipeline start
  - e.g., "Running in FAST mode (10 crops, no ReID)"
- [ ] Update TIMING SUMMARY to show active mode

#### Task 5.3: Final Review
- [ ] Re-read this design document
- [ ] Verify all tasks completed
- [ ] Check for any missed edge cases
- [ ] Commit: "Complete pipeline modes implementation"

---

## 6. Rollback Plan

If implementation causes issues:

1. **Immediate Rollback:**
   ```bash
   git revert HEAD~N  # Revert last N commits
   ```

2. **Partial Rollback:**
   - Remove mode section from config
   - Remove --mode argument from CLI
   - Remove apply_mode_overrides() call
   - Keep individual features (they're backward compatible)

3. **Safe State:**
   - Config without `modes:` section → works as before
   - Code without mode logic → uses config values directly

---

## 7. Future Enhancements

### Auto-Detection Mode (Optional)
```python
def suggest_mode(config, canonical_persons):
    """Auto-suggest mode based on video characteristics"""
    num_persons = len(canonical_persons)
    
    if num_persons <= 8:
        return 'fast'  # Already clean, skip ReID
    elif num_persons <= 12:
        return 'balanced'
    else:
        return 'full'  # Many persons, need ReID
```

### Additional Modes
- **debug:** Max crops, verbose logging, keep intermediates
- **production:** Minimal output, cleanup intermediates

### Per-Stage Mode Overrides
```yaml
modes:
  custom:
    stage3c:
      crops_per_person: 25
    stage3d:
      enabled: true
      similarity_threshold: 0.7  # Higher threshold
    stage4:
      dual_row: false
```

---

## 8. Success Criteria

✅ **Must Have:**
- [ ] All three modes (fast/balanced/full) work correctly
- [ ] Fast mode: ~59s, 10 crops, no stage3d
- [ ] Balanced mode: ~67s, 30 crops, no stage3d
- [ ] Full mode: 77s, 50 crops, with stage3d
- [ ] CLI `--mode` flag works
- [ ] Backward compatible with old configs
- [ ] No errors in any mode

✅ **Nice to Have:**
- [ ] Mode shown in console output
- [ ] Mode recorded in timing JSON
- [ ] Documentation updated
- [ ] Performance matches estimates ±10%

---

## 9. Timeline

**Estimated Effort:** 2-3 hours

- Phase 1 (Core): 45 min
- Phase 2 (Stage 3d): 15 min
- Phase 3 (Stage 4): 30 min
- Phase 4 (Testing): 45 min
- Phase 5 (Documentation): 30 min

---

## 10. Sign-off

**Design Approved By:** [Pending User Review]  
**Implementation Start:** [After Approval]  
**Expected Completion:** [TBD]

---

## Appendix A: Files Reference

```
det_track/
├── configs/
│   └── pipeline_config.yaml          [MODIFY: Add modes section]
├── run_pipeline.py                   [MODIFY: Add mode logic]
├── stage3c_filter_persons.py         [NO CHANGE: Reads config]
├── stage3d_refine_visual.py          [NO CHANGE: Gated by run_pipeline]
├── stage4_generate_html.py           [MODIFY: Add dual_row toggle]
└── PIPELINE_MODES_DESIGN.md          [THIS FILE]
```

## Appendix B: Key Variables to Override

| Variable | Current Location | Override Source |
|----------|-----------------|-----------------|
| `crops_per_person` | `config['stage3c_filter']['crops_per_person']` | `modes[mode]['crops_per_person']` |
| `stage3d.enabled` | `config['pipeline']['stages']['stage3d']['enabled']` | `modes[mode]['enable_stage3d']` |
| `stage4.dual_row` | `config['stage4_html']['dual_row']` (NEW) | `modes[mode]['stage4_dual_row']` |

---

**END OF DESIGN DOCUMENT**
