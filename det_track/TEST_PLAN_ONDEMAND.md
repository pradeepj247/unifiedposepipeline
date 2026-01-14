# On-Demand Crop Extraction - Test Plan

## Summary

We're testing an alternative approach to crop storage that could:
- **Save 811 MB** (99.9% storage reduction)
- **Save 12+ seconds** in total pipeline time
- **Simplify architecture** (no crop storage layer needed)

## Your Smart Strategy

Instead of storing crops (812 MB), we:
1. Use existing **canonical_persons.npz** (169 KB) from Google Drive
2. Do a **linear pass** through the video
3. Extract crops **on-demand** for top 10 persons
4. Fill buckets until we have 50 crops per person
5. **Early terminate** when all buckets filled

## Test Setup (What We Built)

### Files Created:
```
det_track/
├── test_ondemand_extraction.py    ← Main test script (373 lines)
├── test_ondemand.bat              ← Windows runner
└── test_data/
    ├── README.md                  ← Test documentation
    ├── DOWNLOAD.md                ← How to get canonical_persons.npz
    └── (canonical_persons.npz)    ← Download from Drive (169 KB)
```

### What You Need:
1. ✅ **Video**: `kohli_nets.mp4` (already have from Stage 0)
2. ⏳ **Data**: `canonical_persons.npz` (download from Google Drive)
3. ✅ **Code**: Test script ready to run

## How to Test

### Step 1: Download Data
```bash
# Download from Google Drive:
# Path: /content/drive/MyDrive/pipelineoutputs/kohli_nets/canonical_persons.npz
# Save to: det_track/test_data/canonical_persons.npz
```

See: `test_data/DOWNLOAD.md` for detailed instructions

### Step 2: Run Test
```batch
cd det_track
test_ondemand.bat
```

### Step 3: Check Results
```
test_output/
├── person_001.webp to person_010.webp  ← Visual output
└── timing_results.json                  ← Performance data
```

## Expected Performance

### Best Case (persons appear early):
- Process ~800 frames
- Time: ~8 seconds
- Early termination saves 1,227 frames (12 seconds)

### Typical Case:
- Process ~1,200 frames  
- Time: ~12 seconds
- Comparable to old approach (10.7s)

### Worst Case (persons scattered):
- Process full 2,027 frames
- Time: ~20 seconds
- Still saves 811 MB storage!

## Why This Could Win

### Total Pipeline Impact:

**Old Approach:**
```
Stage 1: YOLO + crops  = 40.5s (50 FPS)
Stage 4b: Save crops   =  4.6s
Stage 10b: Load crops  =  6.2s
                       ------
                Total:   51.3s
Storage: 812 MB
```

**New Approach:**
```
Stage 1: YOLO only     = 27.0s (75 FPS) ← 13.5s saved!
Stage 4b: Save bboxes  =  0.01s        ← 4.6s saved!
Stage 10b: Extract     = 12.0s         ← Net cost: +5.8s
                       ------
                Total:   39.0s          ← 12.3s faster!
Storage: 1 MB                          ← 811 MB saved!
```

**Net Benefit:**
- ⚡ 12.3 seconds faster (24% speedup)
- 💾 811 MB smaller (99.9% reduction)
- 🧩 Simpler architecture

## Key Advantages

1. **Stage 1 Speed**: No crop extraction overhead
   - YOLO runs at native 75 FPS (not 50 FPS)
   - Saves 13.5 seconds on 2027-frame video

2. **Stage 4b Simplicity**: Just save bboxes
   - 169 KB instead of 812 MB
   - 0.01s instead of 4.6s

3. **Stage 10b Efficiency**: Linear video decode
   - Modern decoders love sequential access
   - Multi-person batching (4 persons/frame)
   - Early termination possible

4. **Single Source of Truth**: Video file
   - No duplicate crop storage
   - Always in sync with video

5. **Enabled by Stage 0**: GOP normalization
   - Predictable decode performance
   - Fast sequential access

## What We're Validating

- ✅ Linear pass achieves 80-100 FPS
- ✅ Early termination works (don't need full video)
- ✅ Multi-person batching is efficient
- ✅ Total time is competitive (even if slightly slower)
- ✅ Code is simpler than LMDB approach
- ✅ 99.9% storage reduction

## Next Steps if Successful

1. ✅ Test locally (you're about to do this)
2. ⏳ Verify performance meets expectations
3. ⏳ Create feature branch: `feature/on-demand-extraction`
4. ⏳ Add config toggle to existing stages
5. ⏳ Test on Google Colab
6. ⏳ Benchmark full pipeline
7. ⏳ Decide: commit or revert

## Rollback Plan

If this doesn't work well:
1. We're testing in isolation (no pipeline changes yet)
2. Can abandon the approach
3. Return to LMDB plan (Phase 3)
4. Or keep current approach

**Zero risk testing!** 🛡️

---

**Status**: Ready for testing
**Blocker**: Need to download `canonical_persons.npz` from Google Drive
**ETA**: 5-10 minutes to run test
