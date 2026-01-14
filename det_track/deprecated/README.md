# Deprecated Pipeline Components

**Archive Date:** 2026-01-14  
**Reason:** Phase 3 pipeline simplification (5-stage architecture)

## Archived Files

### Stage Files (replaced by on-demand extraction):
- `stage4_load_crops_cache.py` - Replaced by on-demand extraction in stage4
- `stage4b_reorganize_crops.py` - No longer needed
- `stage10_generate_person_webps.py` - Replaced by stage4_generate_html.py
- `stage11_create_selection_html_horizontal.py` - Merged into stage4_generate_html.py

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

## Performance Comparison

| Metric | Old Pipeline | New Pipeline | Improvement |
|--------|-------------|--------------|-------------|
| Total Time | 72.89s | 66.48s | **-6.41s (-8.8%)** ⚡ |
| Storage | 812 MB crops | 0 MB crops | **-812 MB** 💾 |
| Stage 1 | 57.62s | 43.31s | **-14.31s (-24.8%)** |

## Can I Delete This Folder?

Yes, after 1-2 weeks of stable operation with the new pipeline.

## Restore Instructions

If you need to revert to old approach:
```bash
git log --all --full-history -- "**/stage10_generate_person_webps.py"
git checkout <commit-hash> -- det_track/stage10_generate_person_webps.py
```
