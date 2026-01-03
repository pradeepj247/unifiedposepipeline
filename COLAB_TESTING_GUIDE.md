# Colab Testing Instructions - Cleaned Config

## Quick Start

```bash
# In Colab cell, pull latest changes:
cd /content/unifiedposepipeline
git pull origin main

# Run pipeline with cleaned config:
cd det_track
python run_pipeline.py --config configs/pipeline_config.yaml
```

## What to Verify

### ✅ Pipeline Execution
- [ ] All 9 stages execute without errors
- [ ] No warnings about missing config keys
- [ ] No "KeyError" for config parameters
- [ ] No references to removed settings in output

### ✅ Stage-by-Stage
```
Stage 1: DETECTION
  ✓ YOLO model loaded
  ✓ Detection running
  ✓ detections_raw.npz saved
  ✓ crops_cache.pkl saved

Stage 2: TRACKING (BYTETRACK OFFLINE)
  ✓ ByteTrack initialized
  ✓ Tracking running
  ✓ tracklets_raw.npz saved

Stage 3: TRACKLET ANALYSIS
  ✓ Statistics computed
  ✓ Candidates identified
  ✓ tracklet_stats.npz saved
  ✓ reid_candidates.json saved

Stage 4a: LOAD CROPS CACHE
  ✓ Crops cache loaded
  ✓ Ready for extraction

Stage 4b: CANONICAL PERSON GROUPING
  ✓ Grouping tracklets
  ✓ canonical_persons.npz saved
  ✓ grouping_log.json saved

Stage 5: RANKING AND PERSON SELECTION
  ✓ Ranking persons
  ✓ primary_person.npz saved
  ✓ ranking_report.json saved

Stage 6: CREATE OUTPUT VIDEO
  ✓ Video created
  ✓ top_persons_visualization.mp4 saved

Stage 6b: CREATE SELECTION GRID
  ✓ Grid image created
  ✓ top10_persons_fullframe_grid.png saved
```

### ✅ Output Files
```
demo_data/outputs/{video_name}/
├── detections_raw.npz              ✓ Stage 1
├── crops_cache.pkl                 ✓ Stage 1
├── tracklets_raw.npz               ✓ Stage 2
├── tracklet_stats.npz              ✓ Stage 3
├── reid_candidates.json            ✓ Stage 3
├── canonical_persons.npz           ✓ Stage 4b
├── grouping_log.json               ✓ Stage 4b
├── primary_person.npz              ✓ Stage 5
├── ranking_report.json             ✓ Stage 5
├── top_persons_visualization.mp4   ✓ Stage 6
├── top10_persons_fullframe_grid.png✓ Stage 6b
└── selection_grid_info.json        ✓ Stage 6b
```

### ✅ Performance Metrics
- Total pipeline time: ___ seconds
- Detection FPS: ___
- Tracking FPS: ___
- Analysis time: ___
- Grouping time: ___
- Ranking time: ___
- Video creation time: ___

### ✅ Log Inspection
Look for these in the output logs:
- "✅ YOLO model loaded" (Stage 1)
- "ByteTrack initialized" (Stage 2)
- "Detected ### candidates" (Stage 3)
- "Grouping tracklets" (Stage 4b)
- "Ranking persons" (Stage 5)
- "Video created" (Stage 6)

### ✅ No Errors
Make sure there are NO:
- `KeyError` for config keys
- `AttributeError` for config access
- Warnings about "processing_resolution", "output_fps", etc.
- References to removed settings

## Troubleshooting

### If you see "KeyError: 'processing_resolution'"
- ❌ This means code still tries to read removed settings
- Check if stage1_detect.py was recently modified
- Verify the code doesn't read this setting

### If you see "video file not found"
- Check `global.repo_root` matches your Colab environment
- Check `global.video_file` is set to an existing video
- Verify path: `/content/unifiedposepipeline/demo_data/videos/{video_file}`

### If you see "CUDA out of memory"
- Reduce `max_frames` in stage1_detect.input
- Set `device: cpu` in stage1_detect.detector
- Check GPU availability: `nvidia-smi`

### If any stage fails
- Check the error message carefully
- Look for removed config keys in the error
- Verify the video file exists and is readable
- Check that models are downloaded to correct paths

## Comparison with Previous Run

Before cleanup (if you have logs):
```
Lines in config:    249
Dead settings:      9
Unused parameters:  ~20%
```

After cleanup:
```
Lines in config:    194
Dead settings:      0
Unused parameters:  0%
```

**Expected result**: Same output files, same quality, cleaner config

## Success Criteria

✅ Pipeline completes successfully
✅ All stages produce expected output files
✅ No warnings about missing or removed config keys
✅ No errors related to configuration
✅ Performance metrics captured
✅ Output quality unchanged from before cleanup

---

## Report Back

After testing, please share:
1. Did all stages execute successfully? (Y/N)
2. Any errors or warnings? (describe)
3. Total pipeline time?
4. Performance compared to before cleanup?
5. Any issues with removed settings?
6. Confidence level for production use? (1-10)

---

**Config Status**: ✅ VERIFIED - All 194 lines checked for actual usage

**Ready to Deploy**: YES - Safe to use in production after Colab test
