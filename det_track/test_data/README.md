# On-Demand Crop Extraction - Proof of Concept

## What This Tests

This is a **proof-of-concept** for the on-demand crop extraction approach, which eliminates the need to store 812 MB of crops on disk.

### Old Approach (Current):
```
Stage 1: Extract crops during YOLO  → crops_cache.pkl (824 MB)
Stage 4b: Reorganize by person      → crops_by_person.pkl (812 MB)
Stage 10b: Load crops               → Generate WebPs

Overhead: 10.7 seconds (4.55s save + 6.15s load)
Storage: 812 MB
```

### New Approach (Being Tested):
```
Stage 1: YOLO only, no crops        → detections_raw.npz (153 KB)
Stage 4b: Organize bboxes only      → person_bboxes.npz (~100 KB)
Stage 10b: Linear pass through video → Extract on-demand → Generate WebPs

Expected: ~5-12 seconds, 1 MB storage
```

## Prerequisites

1. **Video File**: `kohli_nets.mp4`
   - Should be at: `../demo_data/videos/kohli_nets.mp4`
   - Already have this from Stage 0 testing ✅

2. **Canonical Persons Data**: `canonical_persons.npz` (169 KB)
   - Download from Google Drive:
     - Path: `/content/drive/MyDrive/pipelineoutputs/kohli_nets/canonical_persons.npz`
   - Save to: `test_data/canonical_persons.npz`
   - Contains: Top 10 persons with their frame numbers and bboxes

3. **Python Dependencies**:
   ```bash
   pip install numpy opencv-python imageio
   ```

## How to Run

### Windows:
```batch
# 1. Download canonical_persons.npz from Google Drive
#    Save to: test_data\canonical_persons.npz

# 2. Run the test
test_ondemand.bat
```

### Manual Run:
```bash
python test_ondemand_extraction.py \
    --video ../demo_data/videos/kohli_nets.mp4 \
    --data test_data/canonical_persons.npz \
    --output test_output/ \
    --crops-per-person 50
```

## What It Does

1. **Loads canonical persons data** (169 KB npz file)
   - Top 10 persons
   - Their frame numbers and bboxes

2. **Prepares extraction plan**
   - Maps frame numbers to persons
   - Example: Frame 0 → [Person 3, Person 12, Person 15, Person 17]

3. **Linear pass through video**
   - Opens video once
   - Reads frames sequentially (fast!)
   - Extracts crops for multiple persons per frame
   - Fills 10 buckets (50 crops each)
   - **Early termination** when all buckets filled

4. **Generates WebP animations**
   - 10 files: `person_001.webp` to `person_010.webp`

5. **Compares performance**
   - Time: Old vs New
   - Storage: 812 MB vs 1 MB

## Expected Results

### Timing Breakdown:

**Old Approach:**
- Stage 4b save: 4.55s
- Stage 10b load: 6.15s
- **Total: 10.7s**

**New Approach:**
- Linear pass extraction: ~5-12s (depends on person distribution)
- WebP generation: ~1-2s
- **Total: ~6-14s**

**Note**: We also save 13.5s in Stage 1 by not extracting crops during YOLO!

### Storage:
- Old: 812 MB (crops_by_person.pkl)
- New: 1 MB (just bboxes)
- **Savings: 811 MB (99.9%)**

## Output Files

After running, check:

```
test_output/
├── person_001.webp
├── person_002.webp
├── ...
├── person_010.webp
└── timing_results.json  ← Performance metrics
```

### timing_results.json:
```json
{
  "extraction": {
    "frames_processed": 1200,
    "total_frames": 2027,
    "crops_extracted": 500,
    "elapsed_seconds": 12.5,
    "processing_fps": 96.0
  },
  "webp_generation": {
    "webp_count": 10,
    "elapsed_seconds": 1.8
  },
  "comparison": {
    "old_time": 10.7,
    "new_time": 14.3,
    "time_saved": -3.6,
    "old_storage_mb": 812,
    "new_storage_mb": 1,
    "storage_saved_mb": 811
  }
}
```

## Key Insights to Validate

1. **Linear pass speed**: Should be 80-100 FPS
2. **Early termination**: Should NOT need full 2,027 frames
3. **Multi-person batching**: Extracting 3-4 persons per frame
4. **Total time**: Competitive with old approach (even if slightly slower)
5. **Storage savings**: Massive (99.9% reduction)

## Questions to Answer

- ✅ Is linear pass fast enough? (Target: >80 FPS)
- ✅ Does early termination work? (Stop before frame 2027?)
- ✅ Is total time acceptable? (Target: <15 seconds)
- ✅ Are WebPs generated correctly?
- ✅ Is the code simpler than LMDB approach?

## Next Steps

If this test is successful:
1. Modify Stage 1 to make crop extraction optional
2. Create Stage 4b variant for bbox organization
3. Create Stage 10b variant with linear pass
4. Add configuration toggle to switch between approaches
5. Test on Google Colab
6. Benchmark full pipeline with both approaches

## Troubleshooting

### "canonical_persons.npz not found"
- Download from Google Drive path listed above
- Or run the full pipeline once to generate it

### "Video not found"
- Check: `../demo_data/videos/kohli_nets.mp4`
- Or update path in command

### "Module not found: imageio"
```bash
pip install imageio
```

### WebP generation fails
- OpenCV might not support WebP animation
- imageio is used as fallback (requires pillow)
- Install: `pip install imageio[ffmpeg]`

## Status

- ✅ Script created
- ⏳ Waiting for canonical_persons.npz download
- ⏳ Pending: Local testing
- ⏳ Pending: Performance comparison

---

**This is a proof-of-concept to validate the approach before modifying the main pipeline.**
