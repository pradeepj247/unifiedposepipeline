# Phase 2: HDF5 Optimization Implementation

## Overview
Phase 2 implements HDF5 storage format for Stage 4b output and Stage 10b input to address the 812 MB pickle I/O bottleneck identified in Phase 1 validation.

## Performance Problem (Phase 1)
- **Observation**: New pipeline 9.38s vs old 4.98s (+4.4s overhead)
- **Root Cause**: 90% of overhead is pickle I/O
  - Stage 4b save: 2.69s for 812 MB
  - Stage 10b load: ~2s to load entire file for 10 persons
- **Analysis**: Pickle inefficient for large binary data (crops)

## Solution Strategy
**Conservative Approach**: Optimize Stage 4b and 10b only, don't touch Stage 1 yet.

### Key Optimizations
1. **HDF5 Format**: Binary-efficient storage
2. **Smart Compression**: Gzip level 1 on metadata only (not crops - images don't compress well)
3. **Partial Loading**: Stage 10b loads only top 10 persons, not all 48
4. **Backward Compatibility**: Pickle fallback maintained

### Expected Results
- File size: 812 MB → ~800 MB (minimal change - crops are already compressed JPEGs internally)
- Stage 4b save: 2.69s → 1.5-2s (faster I/O, no compression overhead)
- Stage 10b load: ~2s → 0.3-0.5s (80-85% faster via partial loading)
- Total pipeline: 9.38s → 7-7.5s (20-25% faster)

## Implementation Details

### HDF5 File Structure
```
crops_by_person.h5
├── person_003/
│   ├── frame_numbers [dataset: int64, gzip]
│   ├── crops/
│   │   ├── 0 [dataset: uint8 [H,W,3], gzip level 4]
│   │   ├── 1 [dataset: uint8 [H,W,3], gzip level 4]
│   │   └── ... (N crops total)
│   ├── bboxes [dataset: float32 [N,4], gzip]
│   └── confidences [dataset: float32 [N], gzip]
├── person_007/
│   └── ... (same structure)
└── metadata/
    ├── num_persons (attribute)
    ├── created_timestamp (attribute)
    └── format_version (attribute)
```

### Files Modified

#### 1. stage4b_reorganize_crops.py
**New Functions:**
- `save_crops_to_hdf5(crops_by_person, output_file, compression_level=4)`
  - Iterates persons with tqdm progress bar
  - Creates HDF5 groups per person
  - Stores crops as individual datasets with gzip compression
  - Preserves metadata (frame_numbers, bboxes, confidences)
  - Adds global metadata (num_persons, timestamp, format_version)

- `save_crops_to_pickle(crops_by_person, output_file)`
  - Legacy fallback for backward compatibility
  - Simple pickle.dump wrapper

**Updated Functions:**
- `run_reorganize_crops()`
  - Added format detection from output file extension
  - Added compression_level parameter
  - Enhanced console output with format info
  - Updated save logic to use appropriate function

**Dependencies:**
- h5py: Auto-installed if missing
- datetime, timezone: For timestamp metadata
- json: For metadata serialization

#### 2. stage10b_generate_webps.py
**New Functions:**
- `load_top_n_persons_from_hdf5(hdf5_file, max_persons=10)`
  - KEY OPTIMIZATION: Only reads needed persons, not all 48!
  - Scans HDF5 groups to find person IDs and sizes
  - Sorts by size (duration) and takes top N
  - Loads only selected persons with progress bar
  - Returns crops_by_person dict

- `load_crops_by_person(file_path, max_persons=10)`
  - Auto-detects format (.h5 vs .pkl)
  - Routes to appropriate loader
  - Returns normalized crops_by_person dict

**Updated Functions:**
- `create_webps_for_top_persons()`
  - Uses new load_crops_by_person() function
  - No longer loads entire file for pickle format
  - Simplified logic (loader handles top N selection)

**Dependencies:**
- h5py: Auto-installed if missing

#### 3. pipeline_config.yaml
**Stage 4b Configuration:**
```yaml
stage4b:
  output:
    format: hdf5                    # hdf5 (default) or pickle
    crops_by_person_file: ${outputs_dir}/${current_video}/crops_by_person.h5
    compression_level: 4            # Gzip level 1-9 (4=balanced)
```

**Changes:**
- Added `format: hdf5` parameter
- Changed file extension: .pkl → .h5
- Added `compression_level: 4` parameter

## Compression Level Guidance
**IMPORTANT**: Crops are NOT compressed (images don't benefit from gzip).

Only metadata is compressed:
- **frame_numbers**: gzip level 1 (fast)
- **bboxes**: gzip level 1 (fast)
- **confidences**: gzip level 1 (fast)

**Why no crop compression?**
- Crops are already compressed as JPEG internally (cv2.imencode)
- Gzip on JPEG data provides minimal benefit (~5% reduction)
- Compression overhead adds 40+ seconds (42s observed in testing)
- Uncompressed HDF5 is still efficient due to contiguous storage

## Testing Protocol

### Test on Google Colab
```bash
# Update code
cd /content/unifiedposepipeline
git pull origin main

# Delete old pickle to force HDF5 regeneration
cd det_track
rm demo_data/outputs/kohli_nets/crops_by_person.pkl

# Run pipeline
python run_pipeline.py --config configs/pipeline_config.yaml
```

### Validation Checklist
- [ ] Stage 4b completes successfully
- [ ] crops_by_person.h5 file created
- [ ] File size ~800 MB (verify with `ls -lh`)
- [ ] Stage 4b timing ~2s in timing sidecar
- [ ] Stage 10b completes successfully
- [ ] Stage 10b timing < 1s in timing sidecar
- [ ] Total pipeline time 7-7.5s (verify in orchestrator output)
- [ ] WebP files created correctly
- [ ] HTML report loads and displays correctly

## Backward Compatibility

### Pickle Fallback
Both stages support pickle format:
- **Stage 4b**: Set `format: pickle` in config or use .pkl extension
- **Stage 10b**: Auto-detects based on file extension

### Migration Path
1. Run pipeline with HDF5 config
2. Verify output quality
3. Delete old .pkl files if satisfied
4. Pickle files can coexist with HDF5 files

## Troubleshooting

### Issue: h5py not found
**Symptom**: ImportError: No module named 'h5py'
**Solution**: Both stages auto-install h5py on first run

### Issue: File format mismatch
**Symptom**: Stage 10b fails to load crops
**Solution**: Verify stage4b output file extension matches format parameter

### Issue: Slower than expected
**Symptom**: Stage 4b takes > 5s
**Solution**: 
- Verify crops are NOT compressed (compression=None for crops)
- Check that only metadata uses compression (level 1)
- Verify SSD storage (HDD will be slower)
- File size should be ~800 MB (similar to pickle)

### Issue: HDF5 file corrupted
**Symptom**: h5py.File() raises error
**Solution**:
- Delete crops_by_person.h5
- Re-run Stage 4b
- If persistent, switch to pickle format

## Performance Benchmarks

### Baseline (Phase 1 - Pickle)
```
Stage 4b:  3.28s total (0.59s load + 2.69s save)
Stage 10b: ~2s load + 1s process + 1s save = ~4s
File size: 812 MB
```

### Expected (Phase 2 - HDF5)
```
Stage 4b:  ~2s total (0.59s load + 1.5s save)
Stage 10b: 0.3-0.5s load + 1s process + 1s save = ~2.5s
File size: ~800 MB (similar to pickle - crops don't compress)
Improvement: 2s saved (20-25% faster for these stages)
```

**Note**: File size similar because crops are already JPEG-compressed internally.

## Next Steps (Future Phases)

### Phase 3: Stage 1 Optimization (If Needed)
- **Current**: Stage 1 saves 90 MB detections_raw.npz
- **Consideration**: HDF5 may not provide significant benefit
- **Decision**: Wait for Phase 2 validation

### Phase 4: Video Frame Caching
- **Opportunity**: Eliminate video seeking overhead
- **Approach**: Pre-extract frames to HDF5 or memory map
- **Expected**: 1-2s improvement in Stage 4b

## Documentation Updates
- [x] PHASE2_HDF5_IMPLEMENTATION.md (this file)
- [ ] PIPELINE_REORGANIZATION_DESIGN.md (update with Phase 2 results)
- [ ] README.md (update performance benchmarks)

## Commit Plan
```bash
git add det_track/stage4b_reorganize_crops.py
git add det_track/stage10b_generate_webps.py
git add det_track/configs/pipeline_config.yaml
git add det_track/docs/PHASE2_HDF5_IMPLEMENTATION.md
git commit -m "fix: Phase 2 HDF5 - remove crop compression for 20x speedup

CRITICAL FIX: Remove gzip compression from crops
- Testing revealed: gzip on crops = 42s save time (9.4x slower!)
- Root cause: Crops are already JPEG-compressed internally
- Solution: Only compress metadata (frame_numbers, bboxes, confidences)

Performance after fix:
- Expected Stage 4b save: 42s → ~2s (20x faster)
- File size: ~800 MB (similar to pickle - expected)
- Compression benefit minimal for pre-compressed image data

Additional fixes:
- stage10b: Update input path to crops_by_person.h5
- Config: Change stage10b input from .pkl to .h5
- Docs: Update expected results based on testing

Changes:
- stage4b: Set compression=None for crops datasets
- stage4b: Use gzip level 1 for metadata (fast)
- config: Fix stage10b input path mismatch
"
git push origin main
```

## Success Criteria
1. ✅ Code committed and pushed
2. ⏳ Pipeline runs successfully on Colab
3. ⏳ File size ~800 MB (similar to pickle - expected)
4. ⏳ Stage 4b save time < 3s (target: ~2s)
5. ⏳ Total pipeline time < 8s (target: 7-7.5s)
6. ⏳ WebP output quality unchanged
7. ⏳ HTML report displays correctly

---
**Status**: Implementation complete, awaiting Colab validation
**Created**: 2025-01-XX
**Last Updated**: 2025-01-XX
