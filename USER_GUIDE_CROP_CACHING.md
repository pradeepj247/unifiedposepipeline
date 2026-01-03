# User Guide: Crop Caching & Manual Selection Pipeline

## Quick Start

This guide walks you through using the new crop caching pipeline to identify and select a primary person from your video.

---

## Overview

The new pipeline solves the **P14/P29 merging issue** and many similar tracking problems by:

1. **Extracting crops during detection** (Stage 1) - single video pass
2. **Caching crops to disk** (~200MB for fast access)
3. **Creating visual selection table** (Stage 7b) - see all persons with crops
4. **Letting you manually select** (Stage 7) - you decide, not the algorithm

**Key advantage**: Eliminates 70+ seconds of video seeking per comparison.

---

## Step-by-Step Workflow

### Step 1: Configure Your Video

Edit `det_track/configs/pipeline_config.yaml`:

```yaml
global:
  repo_root: /path/to/unifiedposepipeline
  video_file: /path/to/your/video.mp4
  models_dir: /path/to/models
```

### Step 2: Run Detection Pipeline (Stages 1-6)

This will:
- ✅ Detect persons in video (YOLO)
- ✅ Track them over time (ByteTrack)
- ✅ **Extract and cache crops for each detection**
- ✅ Group tracklets into canonical persons
- ✅ Create visualization video

```bash
cd det_track
python run_pipeline.py --config configs/pipeline_config.yaml --stages 1,2,3,4,5,6
```

**Expected output files**:
```
outputs/video_name/
├── detections_raw.npz           # All detections
├── tracklets_raw.npz            # Tracked persons
├── canonical_persons.npz        # Grouped persons (46 persons)
├── crops_cache.pkl              # ← NEW: Cached crops (~200MB)
├── ranking_results.json         # Person rankings
└── output_video.mp4             # Visualization
```

**Timing**:
- Stage 1 (detection + crops): ~80-120 seconds
- Stages 2-6: ~5-10 seconds total
- **Total: ~90-130 seconds** (was 15+ minutes with old ReID approach)

### Step 3: Create Visual Selection Table

This generates a PNG table with all persons and their high-confidence crops.

```bash
python stage7_create_selection_table.py --config configs/pipeline_config.yaml
```

**Output**: `selection_table.png`

The table looks like:
```
┌─────┬─────────┬──────────────┬────────────┬──────────┬────────┐
│  #  │ Person  │     Crop     │ Start Frm  │ End Frm  │ Appears│
├─────┼─────────┼──────────────┼────────────┼──────────┼────────┤
│  1  │    7    │   [image]    │     0      │   200    │  140   │
│  2  │   14    │   [image]    │   103      │   365    │  250   │
│  3  │   29    │   [image]    │   360      │   785    │  425   │
│  4  │    3    │   [image]    │    45      │   189    │  120   │
└─────┴─────────┴──────────────┴────────────┴──────────┴────────┘
```

### Step 4: View the Selection Table

Open `selection_table.png` in your image viewer:

1. **Look at all persons and their crops**
2. **Identify which one you want** (note their Person ID)
3. **Examples**:
   - "Person 14 is the main subject, Person 29 is another person"
   - "Person 3 appears later, Person 7 is primary"

### Step 5: Select Your Primary Person

Use the Person ID from Step 4 to create `primary_person.npz`:

```bash
python stage7_select_person.py --config configs/pipeline_config.yaml --person-id 14
```

**Output**: `primary_person.npz` (contains only Person 14 data)

### Step 6: Continue with Pose Estimation

Now use `primary_person.npz` for downstream tasks:

```python
# Load primary person
data = np.load('primary_person.npz', allow_pickle=True)
primary = data['primary_person']

frame_numbers = primary['frame_numbers']
bboxes = primary['bboxes']
confidences = primary['confidences']

# Extract crops from cache
crops_cache_path = 'outputs/video_name/crops_cache.pkl'
with open(crops_cache_path, 'rb') as f:
    crops_cache = pickle.load(f)

# Use for pose estimation, ReID training, etc.
```

---

## Example: P14 vs P29

### Scenario

Your video has two people:
- **Person 14**: Main subject, appears frames 103-365 (250 frames)
- **Person 29**: Another person, appears frames 360-785 (425 frames)
- **Overlap**: Frames 360-365 (5 frames where both are visible)

### Why They Stayed Separate

The pipeline's Stage 3 area ratio check rejected them:
- P14 mean area: 164,020 pixels
- P29 mean area: 84,658 pixels
- Ratio: 0.52 (threshold [0.6, 1.4] rejects < 0.6)

**Reason**: They might be at different depths, different scales, or different people.

### How You Decide

1. **View selection_table.png**
   - See crops of both Person 14 and Person 29 side-by-side
   - Visually compare: same person or different?

2. **You choose**:
   - Option A: Both are different people → select Person 14 (main subject)
   - Option B: Same person, scale changed → select based on better pose coverage

3. **Run Stage 7**:
   ```bash
   # Your choice from visual inspection
   python stage7_select_person.py --config ... --person-id 14  # OR
   python stage7_select_person.py --config ... --person-id 29
   ```

---

## Tips & Troubleshooting

### Tip 1: Viewing Selection Table at High Resolution

If `selection_table.png` is large (many persons):
- Use an image viewer that supports zoom (most do)
- Or extract specific crops:

```python
import pickle
import numpy as np
from PIL import Image

# Load cache
with open('crops_cache.pkl', 'rb') as f:
    crops = pickle.load(f)

# View specific person
canonical = np.load('canonical_persons.npz', allow_pickle=True)['persons']
person_14 = [p for p in canonical if p['person_id'] == 14][0]

best_frame = int(person_14['frame_numbers'][np.argmax(person_14['confidences'])])
crop = crops[best_frame][list(crops[best_frame].keys())[0]]

Image.fromarray(crop).show()
```

### Tip 2: Crops Not Showing in Table?

The `stage7_create_selection_table.py` script requires `pillow`:

```bash
pip install pillow
```

If crops still don't appear:
- Check that `crops_cache.pkl` was created in Stage 1
- Check file size (~200MB expected for typical video)
- Re-run Stage 1 to regenerate

### Tip 3: Comparing Specific Persons

To compare two persons visually:

```python
import pickle
import numpy as np
from PIL import Image, ImageDraw

# Load cache and persons
with open('crops_cache.pkl', 'rb') as f:
    crops = pickle.load(f)

canonical = np.load('canonical_persons.npz', allow_pickle=True)['persons']

# Get best crop for each person
def get_best_crop(person, crops):
    best_frame = int(person['frame_numbers'][np.argmax(person['confidences'])])
    for crop in crops.get(best_frame, {}).values():
        if crop is not None:
            return crop
    return None

# Create comparison image
p14 = [p for p in canonical if p['person_id'] == 14][0]
p29 = [p for p in canonical if p['person_id'] == 29][0]

crop14 = get_best_crop(p14, crops)
crop29 = get_best_crop(p29, crops)

# Side-by-side
comparison = Image.new('RGB', (crop14.shape[1] + crop29.shape[1] + 10, crop14.shape[0]))
comparison.paste(Image.fromarray(crop14), (0, 0))
comparison.paste(Image.fromarray(crop29), (crop14.shape[1] + 10, 0))
comparison.show()
```

### Tip 4: Re-running a Specific Stage

If you want to regenerate crops without re-running detection:

```bash
# Re-run only Stage 1 (fast, detects + extracts crops)
python run_pipeline.py --config ... --stages 1

# Regenerate table (uses cached crops)
python stage7_create_selection_table.py --config ...
```

### Issue: "Crops cache not found"

**Fix 1**: Check that Stage 1 completed successfully
```bash
# Check output files
ls -lh outputs/video_name/crops_cache.pkl
```

**Fix 2**: Re-run Stage 1
```bash
python run_pipeline.py --config ... --stages 1
```

**Fix 3**: Check config file
```bash
# Verify pipeline_config.yaml has correct paths
grep crops_cache_file det_track/configs/pipeline_config.yaml
```

### Issue: "Person ID not found"

**Fix**: List all available persons
```bash
# This command will show all available person IDs
python stage7_select_person.py --config ... --person-id 999
# (Will fail but show all available persons)
```

---

## Performance Expectations

### Typical 30-minute Video (1080p)

| Stage | Time | Notes |
|-------|------|-------|
| Stage 1 (detect + crops) | 80-120s | Single video pass, YOLO at 51.8 FPS |
| Stage 2 (tracking) | 1-2s | ByteTrack motion-only |
| Stage 3 (analysis) | 2-3s | Temporal/spatial checks |
| Stage 4a (load crops) | 1-2s | Pickle load |
| Stage 4b (grouping) | 1s | Geometric grouping |
| Stage 5 (ranking) | <1s | Sort by duration |
| Stage 6 (visualization) | 5-10s | Draw all persons |
| Stage 7b (table) | 5-10s | Create PNG table |
| **Total** | **100-150s** | ~2-2.5 minutes |

**Old approach (ReID validation)**: 15+ minutes (70s × 12 candidates)
**New approach (manual selection)**: 2.5 minutes + user visual inspection

---

## Advanced: Batch Processing Multiple Videos

If you have multiple videos:

```bash
for video in /path/to/videos/*.mp4; do
  echo "Processing: $video"
  
  # Update config
  sed -i "s|video_file:.*|video_file: $video|" configs/pipeline_config.yaml
  
  # Run pipeline
  python run_pipeline.py --config configs/pipeline_config.yaml --stages 1,2,3,4,5,6,7b
  
  # View and manually select
  echo "View: outputs/$(basename $video .mp4)/selection_table.png"
  echo "Select: python stage7_select_person.py --config configs/pipeline_config.yaml --person-id ?"
done
```

---

## FAQ

**Q: What if I made the wrong selection?**
A: Re-run `stage7_select_person.py` with a different `--person-id`. The `primary_person.npz` will be overwritten.

**Q: Can I use multiple persons for pose estimation?**
A: Yes, run Stage 7 once per person ID. This creates separate `primary_person.npz` files (you'll need to modify naming). Then merge in post-processing.

**Q: The crops look blurry in the table. Why?**
A: The table resizes crops to fit in cells. The full-resolution crops are in `crops_cache.pkl`. Use the comparison script above to view full resolution.

**Q: Do I need to keep crops_cache.pkl?**
A: Yes, if you want to re-run Stage 7b or view crops later. It's ~200MB, which is reasonable.

**Q: Can I delete intermediate files?**
A: After selecting a person, you can delete:
- `detections_raw.npz` (only used by Stage 2)
- `tracklets_raw.npz` (only used by Stages 3-4)
- `canonical_persons.npz` (only used by Stages 5-7)

But keep `crops_cache.pkl` and `primary_person.npz`.

**Q: What if I want to try different area_ratio thresholds?**
A: Edit `pipeline_config.yaml`:
```yaml
stage3_analyze:
  tracklet_merging:
    area_ratio_range: [0.4, 2.5]  # More permissive (was [0.6, 1.4])
```
Then re-run Stages 3-7.

---

## Configuration Reference

### pipeline_config.yaml

```yaml
# Stage 1: Detection + Crop Extraction
stage1_detect:
  detector:
    model: yolov8s.pt
    confidence: 0.7
  
  output:
    detections_file: ${outputs_dir}/${current_video}/detections_raw.npz
    crops_cache_file: ${outputs_dir}/${current_video}/crops_cache.pkl
    # crops are (256, 128, 3) uint8 arrays

# Stage 4a: Load Crops Cache
stage4a_reid_recovery:
  input:
    crops_cache_file: ${outputs_dir}/${current_video}/crops_cache.pkl
  
  # No outputs - just loads for Stage 7

# Stage 7: Manual Selection
stage7_select_person:
  # Use: python stage7_select_person.py --config ... --person-id <ID>
  # Inputs: canonical_persons.npz
  # Outputs: primary_person.npz

# Stage 7b: Create Selection Table
stage7_create_selection_table:
  # Use: python stage7_create_selection_table.py --config ...
  # Inputs: canonical_persons.npz, crops_cache.pkl
  # Outputs: selection_table.png
```

---

## Summary

The new crop caching pipeline gives you:

1. **Speed**: 2.5 min vs 15+ min (6× faster)
2. **Reliability**: Your visual judgment, not ML models
3. **Simplicity**: Clear workflow with visual feedback
4. **Flexibility**: Easy to re-select if first choice was wrong

**For the P14/P29 case**: You can now quickly see both persons side-by-side in the selection table and decide which one(s) to use, instead of waiting 15+ minutes for automatic validation.

---

## Next Steps

1. Configure your video path in `pipeline_config.yaml`
2. Run the pipeline: `python run_pipeline.py --config ... --stages 1,2,3,4,5,6`
3. Create selection table: `python stage7_create_selection_table.py --config ...`
4. View `selection_table.png` and note Person ID you want
5. Select: `python stage7_select_person.py --config ... --person-id <ID>`
6. Use `primary_person.npz` for pose estimation

**Time investment**: 2.5 minutes automated + 1 minute visual inspection = 3.5 minutes total.

---

For questions or issues, refer to `CROP_CACHING_IMPLEMENTATION.md` for technical details.
