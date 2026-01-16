# IMPLEMENTATION SUMMARY: Filtering & Ranking Fixes

## Changes Made (January 16, 2026)

### 1. Configuration Update
**File**: `det_track/configs/pipeline_config.yaml`

**Changes**:
- Added `min_duration_frames: 150` - Filters out persons visible for less than 5 seconds
- Increased `penalty_threshold` from 0.7 to 0.75 - Stricter late-appearance filtering
- Added detailed comments explaining each weight and threshold

**Impact**: 
- Removes short-lived detections before ranking
- Stricter filtering of persons appearing late in video (>50%)

### 2. Code Enhancement
**File**: `det_track/stage3c_filter_persons.py`

**Changes**:

#### A. Added STEP 0: Minimum Duration Filter
```python
# Filter persons by minimum duration (150 frames)
filtered_by_duration = [p for p in all_persons if len(p['frame_numbers']) >= min_duration_frames]
```

**Impact**: Removes persons with <150 frames early, before ranking

#### B. Updated Ranking to Use Filtered Persons
```python
# Rank from filtered_by_duration instead of all_persons
ranked_indices, scores = rank_persons(filtered_by_duration, ...)
```

**Impact**: Only top-quality candidates ranked high

#### C. Enhanced Debug Output
Added detailed breakdown for each of top 10 persons showing:
- Duration (frames)
- Coverage (% of timespan detected)
- Center distance (pixels)
- Appearance ratio & frame number
- Late-appearance penalty
- Final score

**Example output**:
```
   === TOP 10 SCORING BREAKDOWN ===

   1. PERSON 3:
      Duration: 290 frames
      Coverage: 99.0% (detected in 99.0% of timespan)
      Center dist: 120.5 pixels
      Appearance: frame 0/360 (ratio=0.000)
      ✓  EARLY: ratio 0.000 <= 0.5 → no penalty
      Final score: 0.3845

   2. PERSON 65:
      Duration: 180 frames
      Coverage: 91.5% (detected in 91.5% of timespan)
      Center dist: 250.0 pixels
      Appearance: frame 220/360 (ratio=0.611)
      ⚠️  LATE: ratio 0.611 > 0.5 → penalty=0.767
      ✓  KEPT: penalty 0.767 >= threshold 0.75
      Final score: 0.2956

   === FILTERING RESULT ===
   DEBUG: After filtering - KEEP: 8, REMOVE: 2
   DEBUG: Removed persons: [20, 87]
```

---

## What You'll See When Running Stage 3c

### Step 0: Duration Filtering
```
   DEBUG: Removed 5 persons with <150 frames
      - person_87: 70 frames
      - person_20: 45 frames
      - person_99: 120 frames
      - person_44: 85 frames
      - person_56: 140 frames
   STAT After min_duration filter: 43 persons (threshold: 150 frames)
```

### Step 1: Ranking
```
   INFO Step 1: Ranking all persons...
   FOUND Selected TOP 10 persons (from 43 filtered candidates)
```

### Step 2: Penalty & Filtering
```
   === TOP 10 SCORING BREAKDOWN ===
   [Detailed breakdown for each person]
   === FILTERING RESULT ===
   DEBUG: After filtering - KEEP: 8, REMOVE: 2
```

---

## How This Answers Your Questions

### Q: Duration vs Coverage?
- **Duration**: Total frames person appears in (e.g., 150 frames)
- **Coverage**: % of timespan they're detected (e.g., 85% of frame 100-500)
- **Together**: Rewards long, consistent presence

### Q: What is Center?
- Distance from frame center
- Closer = better (main action usually center)
- Example: 120 pixels from center = 0.0082 contribution

### Q: Where's the 150-frame filter?
- ✅ NOW IMPLEMENTED (was missing)
- Filters BEFORE ranking, catches short persons early

### Q: Will it catch persons at frame 1632/2100?
- ✅ YES, with updated threshold
- frame 1632/2100 = 77.6% → 16.6% penalty
- With threshold 0.75: removed if penalty < 0.75 (strict!)

---

## Testing Checklist

Before running full pipeline on Colab:

- [ ] Run `stage3c_filter_persons.py` on your video
- [ ] Check Step 0 output: Are short persons removed?
- [ ] Check Step 1 output: Top 10 persons make sense?
- [ ] Check Step 2 output: Anyone unexpectedly penalized/removed?
- [ ] Verify final 8 persons are high-quality

---

## Configuration Reference

```yaml
stage3c_filter:
  filtering:
    min_duration_frames: 150         # Minimum 5 seconds (150 frames @ 30fps)
    top_n: 10                        # Select top 10 by composite score
    penalty_threshold: 0.75          # Remove if penalty < 0.75 (max 25% penalty allowed)
    crops_per_person: 50             # Extract 50 crops per person
    
    weights:
      duration: 0.4                  # 40% weight: longer presence is better
      coverage: 0.3                  # 30% weight: consistent coverage is better
      center: 0.2                    # 20% weight: center alignment is better
      smoothness: 0.1                # 10% weight: smooth motion is better
      max_appearance_ratio: 0.5      # 50% threshold: penalty after halfway through video
```

---

## Next Steps

1. ✅ Config updated with new thresholds
2. ✅ Code updated with minimum duration filter & enhanced output
3. ⏳ Test on Colab with your current video
4. ⏳ Verify final persons are as expected
5. ⏳ Adjust weights/thresholds if needed

---

## Files Modified

- `det_track/configs/pipeline_config.yaml` (+13 lines, -5 lines) 
- `det_track/stage3c_filter_persons.py` (+73 lines, -26 lines)
- `FILTERING_LOGIC_ANALYSIS.md` (new documentation)
- `FILTERING_FIX_COMPLETE.md` (new documentation)
- `STAGE3C_FILTERING_LOGIC.md` (new documentation)

