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
1. **HDF5 Format**: Binary-efficient storage with compression
2. **Gzip Compression**: Level 4 (balanced speed/size)
3. **Partial Loading**: Stage 10b loads only top 10 persons, not all 48
4. **Backward Compatibility**: Pickle fallback maintained

### Expected Results
- File size: 812 MB → 250-350 MB (60-70% reduction)
- Stage 4b save: 2.69s → 0.5-0.8s (70-80% faster)
- Stage 10b load: ~2s → 0.3-0.5s (80-85% faster)
- Total pipeline: 9.38s → 6-6.5s (30-35% faster)

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
- **Level 1**: Fastest, ~50% compression
- **Level 4**: Balanced (RECOMMENDED)
- **Level 6**: Default, slower
- **Level 9**: Maximum compression, slowest

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
- [ ] File size 250-350 MB (verify with `ls -lh`)
- [ ] Stage 4b timing < 1s in timing sidecar
- [ ] Stage 10b completes successfully
- [ ] Stage 10b timing < 1s in timing sidecar
- [ ] Total pipeline time 6-6.5s (verify in orchestrator output)
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
**Symptom**: Stage 4b takes > 1s
**Solution**: 
- Check compression_level (try level 1 or 2)
- Verify SSD storage (HDD will be slower)
- Check file size (should be 250-350 MB)

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
Stage 4b:  ~1.3s total (0.59s load + 0.5-0.8s save)
Stage 10b: 0.3-0.5s load + 1s process + 1s save = ~2.5s
File size: 250-350 MB (60-70% reduction)
Improvement: 3.5s saved (30-35% faster)
```

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
git commit -m "feat: Phase 2 HDF5 optimization for Stage 4b and 10b

- Add HDF5 storage with gzip compression (level 4)
- Implement partial loading in Stage 10b (top 10 persons only)
- Maintain pickle fallback for backward compatibility
- Expected: 60-70% file size reduction, 30-35% speedup
- File size: 812 MB → 250-350 MB
- Pipeline time: 9.38s → 6-6.5s

Changes:
- stage4b: Add save_crops_to_hdf5() function
- stage10b: Add load_top_n_persons_from_hdf5() function
- config: Change output to crops_by_person.h5 with format parameter
"
git push origin main
```

## Success Criteria
1. ✅ Code committed and pushed
2. ⏳ Pipeline runs successfully on Colab
3. ⏳ File size < 400 MB (target: 250-350 MB)
4. ⏳ Total pipeline time < 7s (target: 6-6.5s)
5. ⏳ WebP output quality unchanged
6. ⏳ HTML report displays correctly

---
**Status**: Implementation complete, awaiting Colab validation
**Created**: 2025-01-XX
**Last Updated**: 2025-01-XX
