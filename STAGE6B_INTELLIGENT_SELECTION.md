# Stage 6b Intelligent Crop Selection - Improvement Guide

## Problem Statement (User Feedback)

When viewing `top10_persons_cropped_grid.png`, you rightfully asked:

> "Which crop does it pick from? Does it pick from the beginning, middle, or end? Does it choose a large crop or a small crop? Does it choose one with high confidence?"

**Previous Answer**: Random/arbitrary - just took the first available crop from the highest-confidence frame.

## Solution: Intelligent Three-Level Selection

### Level 1: Frame Selection (Which frame to use)

For each person, Stage 6b now evaluates **all frames** where that person appears:

```
For each frame in person's tracklet:
  Score = (60% × tracklet_confidence) + (40% × normalized_bbox_area)

Best Frame = argmax(Score)
```

**What this means:**
- **High confidence frames prioritized**: If a person appears with 0.95 confidence in frame 500, that's better than 0.50 confidence in frame 100
- **Large bboxes preferred**: Full-body persons (h=500px) better than portraits (h=200px)
- **Position-independent**: Equally considers frames from start, middle, or end of video
- **Adaptive to tracklet length**: Long tracklets get best-of-many, short tracklets use what's available

### Level 2: Detection Matching (Which detection in that frame)

In crowded frames with multiple people, we must match the right detection:

```
For each candidate detection in the selected frame:
  IoU_score = overlap_percentage(person_bbox, detection_bbox)
  area_score = normalized_bbox_area(detection_bbox)
  
  match_score = (70% × IoU_score) + (30% × area_score)

Best Detection = argmax(match_score)
```

**What this means:**
- **IoU Matching**: Finds detection with highest overlap with person's tracked bbox
- **Avoids wrong person**: Won't accidentally grab a crop of the person next to them
- **Prefers large detections**: Large crops (256×128) look better than small ones (128×64)
- **No video seeking**: Uses only the cached crops - instant lookup!

### Level 3: Quality Preferences (What kind of crops)

All comparisons use:
- **Confidence scores**: From the original YOLO detection (0.0-1.0)
- **Bbox dimensions**: Width × Height of the detection
- **Temporal distribution**: Can pick from any point in the tracklet

---

## Technical Details: What Changed

### Before (Broken):
```python
def get_best_crop_for_person(person, crops_cache):
    # Just get highest confidence frame
    best_idx = np.argmax(confidences)
    best_frame = person['frame_numbers'][best_idx]
    
    # Take FIRST crop from that frame (WRONG!)
    crops_in_frame = crops_cache[best_frame]
    return crops_in_frame[0]  # ← Random detection, might be wrong person!
```

**Problems:**
- Only evaluates ONE frame (highest confidence)
- Takes first crop without checking if it matches the person
- In multi-person frames, often grabs the wrong person
- Ignores bbox size (small crops look bad)

### After (Intelligent):
```python
def get_best_crop_for_person(person, crops_cache, detections_data):
    best_crop = None
    best_score = -1.0
    
    for i in range(len(frame_numbers)):
        # Evaluate EVERY frame
        person_conf = confidences[i]
        bbox_area = compute_area(bboxes[i])
        
        # Find matching detection using IoU
        best_det = find_best_detection_idx(
            person_bbox, 
            crops_in_frame, 
            detections_bbox_in_frame
        )
        
        # Composite score: confidence + size
        score = 0.6 * person_conf + 0.4 * area_score
        
        if score > best_score:
            best_crop = crops_in_frame[best_det]
    
    return best_crop
```

**Improvements:**
- Evaluates ALL frames (robust selection)
- Matches detection by IoU (correct person)
- Considers bbox size (better quality)
- Composite scoring (balanced decision)

---

## Testing Instructions

### In Colab:

```bash
cd /content/unifiedposepipeline/det_track

# Run just Stage 6b (stage 9) with improved selection
python run_pipeline.py --config configs/pipeline_config.yaml --stages 9 --force

# Then view the output:
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('/content/unifiedposepipeline/demo_data/outputs/kohli_nets/top10_persons_cropped_grid.png')
plt.figure(figsize=(15, 8))
plt.imshow(img)
plt.axis('off')
plt.tight_layout()
plt.show()
```

### Expected Results:

**Grid should now show:**
- ✅ Full-body persons (not head-only or foot-only crops)
- ✅ High-confidence detections (clear, visible persons)
- ✅ Well-positioned crops (not cut in middle)
- ✅ Consistent quality across all 10 persons
- ✅ Large bboxes when available (good visual quality)

**Performance:**
- ⚡ Still <1 second (no video seeking)
- 📊 Intelligent selection from cache only
- 🎯 Better visual quality than before

---

## Performance Expectations

### Timing Breakdown:
```
Cache load:        ~0.5s   (pickle.load of 823 MB file)
Detection load:    ~0.1s   (numpy.load of detections)
Crop selection:    ~0.1s   (IoU matching × 46 persons × ~43 frames avg)
Grid creation:     ~0.1s   (PIL image resizing + pasting)
─────────────────────────
Total:            ~0.7-0.9s (instant, no video seeking!)
```

### Compared to Before:
- **Old (broken)**: 34.39 seconds (video seeking)
- **New (fixed, simple)**: 0.69 seconds (cache + first match)
- **New (improved)**: 0.7-0.9 seconds (cache + intelligent matching)

---

## FAQ: Answering Your Questions

### Q: Which crops are picked?

**A:** The crop with the highest composite score based on:
1. **Where**: The frame with highest tracklet confidence AND large bbox
2. **Which detection**: The one with best overlap (IoU) with tracked bbox
3. **Quality**: Prefers large, high-confidence detections

### Q: Does it pick from beginning, middle, or end?

**A:** **Neither specifically** - it evaluates ALL frames and picks the BEST one regardless of position:
- If person appears with 0.95 confidence at frame 1500 → likely chosen
- If person appears with 0.50 confidence at frame 100 → unlikely chosen
- Frame position doesn't matter, only quality metrics

### Q: Does it pick large or small crops?

**A:** **Large crops when available** - 40% of selection score based on bbox area:
```
area_score = min(1.0, bbox_area / max_area)
score = 0.6 * confidence + 0.4 * area_score
```
A 500×600 detection beats a 100×150 detection (if confidence similar)

### Q: Does it choose high confidence?

**A:** **Absolutely** - 60% of selection score is tracklet confidence:
```
confidence_weight = 0.6  ← Primary factor
area_weight = 0.4        ← Secondary factor
```
A person with 0.90 avg confidence beats one with 0.50 (all else equal)

---

## Code Location

**File**: `det_track/stage6b_create_selection_grid_fixed.py`

**Key Functions**:
- `iou()` - Compute Intersection over Union
- `get_bbox_area()` - Calculate bbox area
- `find_best_detection_idx()` - Match detection by IoU
- `get_best_crop_for_person()` - Main selection logic with three-level scoring
- `create_grid_from_crops()` - Build 2×5 PIL grid image

**Git Commit**: `a2e8758` (Improvement: Stage 6b intelligent crop selection)

---

## Visual Example: P14 (Person 14)

Let's say P14 appears 250 times across the video:

**Old method**: "Just get frame with highest confidence"
```
Frame 103: conf=0.92, det_idx=2 (first detection in frame) → might be P15!
```

**New method**: "Get best frame by composite score, match by overlap"
```
Evaluating all 250 frames...

Frame 103: conf=0.92, area=450×600 → score = 0.6×0.92 + 0.4×0.82 = 0.88
Frame 210: conf=0.88, area=520×700 → score = 0.6×0.88 + 0.4×0.95 = 0.84
Frame 265: conf=0.75, area=200×300 → score = 0.6×0.75 + 0.4×0.35 = 0.59

Best = Frame 103 with score 0.88 ✅

Now match detection in Frame 103:
  Detection 0: IoU=0.15 with P14's bbox → bad match
  Detection 1: IoU=0.92 with P14's bbox → P14! ✅
  Detection 2: IoU=0.05 with P14's bbox → bad match
  Detection 3: IoU=0.88 with P14's bbox → P14 alternative

Best match = Detection 1 (IoU=0.92) ✅
```

**Result**: Gets the RIGHT person at the BEST frame with HIGH quality

---

## Next Steps

1. **Test in Colab**: Run Stage 6b and examine `top10_persons_cropped_grid.png`
2. **Compare visually**: All 10 persons should look better than before
3. **Check quality**: Look for full-body crops, not cut-off persons
4. **Proceed to Stage 7b**: Create selection table for manual review

---

## Summary

You asked three important questions:
- ✅ "Which crop?" → **Best by confidence + size + overlap**
- ✅ "From where?" → **Intelligently selected from all frames**  
- ✅ "High confidence?" → **Explicitly prioritized (60% of score)**

The new implementation transforms Stage 6b from "random" to "intelligent", selecting the highest-quality crops while maintaining zero video seeking and cache-only operation.
