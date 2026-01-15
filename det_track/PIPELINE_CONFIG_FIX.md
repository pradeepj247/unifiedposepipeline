# Pipeline Configuration Fix - Jan 15, 2026

## Issue

When running the full pipeline (`run_pipeline.py`), Stage 1 (Detection) failed with:

```
KeyError: 'stage1'
File "/content/unifiedposepipeline/det_track/stage1_detect.py", line 104, in load_config
    if resolved_config['stage1'].get('advanced', {}).get('verbose', False):
       ~~~~~~~~~~~~~~~^^^^^^^^^^
```

## Root Cause

During the Phase 3 cleanup (Jan 14), config keys were renamed to be more descriptive:

**Old Keys** → **New Keys**
- `stage0` → `stage0_normalize`
- `stage1` → `stage1_detect`
- `stage2` → `stage2_track`
- `stage3a` → `stage3a_analyze`
- `stage3b` → `stage3b_group`
- `stage3c` → `stage3c_rank`
- `stage4` → `stage4_generate_html`

The pipeline_config.yaml file was correctly updated with new keys, but the load_config() function in stage1_detect.py still referenced the old `stage1` key.

## Solution

### Fixed Files

**stage1_detect.py** (2 line fixes):
- Line 104: `resolved_config['stage1']` → `resolved_config['stage1_detect']`
- Line 106: `resolved_config['stage1']` → `resolved_config['stage1_detect']`

### Verified (Already Correct)

All other stage files already use the correct key names:
- ✅ stage2_track.py - uses `config['stage2_track']`
- ✅ stage3a_analyze_tracklets.py - uses `config['stage3a_analyze']`
- ✅ stage3b_group_canonical.py - uses `config['stage3b_group']`
- ✅ stage3c_rank_persons.py - uses `config['stage3c_rank']`
- ✅ stage4_generate_html.py - uses `config.get('stage4_generate_html', {})`

## Testing

The pipeline should now run successfully with:

```bash
cd /content/unifiedposepipeline/det_track
python run_pipeline.py --config configs/pipeline_config.yaml
```

Expected flow:
1. Stage 0: Video Normalization ✓ (was working)
2. Stage 1: Detection ← **NOW FIXED**
3. Stage 2: Tracking
4. Stage 3a: Analysis
5. Stage 3b: Grouping
6. Stage 3c: Ranking
7. Stage 4: HTML Viewer (with OSNet clustering)

## Git Commit

```
556f018 - Fix: stage1_detect.py config key references
```

All fixes have been pushed to GitHub main branch.

## Prevention

In the future, when renaming config keys:
1. Update both pipeline_config.yaml AND all stage files that reference them
2. Run `grep -r "config\['stage" det_track/*.py` to find all references
3. Test full pipeline before committing changes
