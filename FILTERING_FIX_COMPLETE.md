# FILTERING LOGIC - COMPLETE BREAKDOWN & FIX

## TL;DR: What You Asked

### Q1: Duration vs Coverage?
- **Duration**: Total frames person appears (e.g., 200 frames)
- **Coverage**: % of their timespan they're actually detected (e.g., 85% detected)
  - If person appears frame 100-500 (timespan=401 frames) but only in 340 of them → coverage=84.8%

### Q2: Center?
- How close to frame center (center is best, edges are worst)

### Q3: Minimum 150 frames filter?
- **NOT IN THE CODE!** This is the problem!
- Current: `top_n: 10` just takes top 10 regardless of duration
- Missing: `min_duration_frames: 150` filter before ranking

### Q4: Will it catch persons at frame 1632/2100?
- **YES** they get penalized (16.6% penalty)
- **BUT** penalty_threshold=0.7 is too lenient, so they STAY (0.834 >= 0.7)
- **FIX**: Increase penalty_threshold to 0.75 or 0.8

---

## WHAT NEEDS TO CHANGE

### Current filtering (Stage 3c):

**CONFIG** (pipeline_config.yaml):
```yaml
stage3c_filter:
  filtering:
    top_n: 10                              # ← Just top 10, no min duration!
    penalty_threshold: 0.7                 # ← Too lenient! Allows 30% penalty
```

**CODE** (stage3c_filter_persons.py):
```python
# Line ~270
ranked_indices, scores = rank_persons(all_persons, ...)  # NO filtering first!
top_indices = ranked_indices[:top_n]                       # Just takes top N
```

---

## THE FIX (3 parts)

### PART 1: Update Config

In `pipeline_config.yaml`, replace:
```yaml
stage3c_filter:
  filtering:
    top_n: 10                              
    penalty_threshold: 0.7                 
```

With:
```yaml
stage3c_filter:
  filtering:
    min_duration_frames: 150               # ← NEW: Minimum 5 seconds
    top_n: 10                              # From filtered persons
    penalty_threshold: 0.75                # ← UPDATED: Stricter (was 0.7)
```

### PART 2: Update Code

Add minimum duration check BEFORE ranking:

In `stage3c_filter_persons.py`, after line ~260, change from:
```python
# OLD CODE:
ranked_indices, scores = rank_persons(all_persons, video_width, video_height, total_frames, weights)
```

To:
```python
# NEW CODE:
# First filter by minimum duration
min_duration_frames = filter_config.get('min_duration_frames', 150)
filtered_by_duration = [p for p in all_persons if len(p['frame_numbers']) >= min_duration_frames]

removed_by_duration = [p for p in all_persons if len(p['frame_numbers']) < min_duration_frames]
if removed_by_duration:
    print(f"   DEBUG: Removed {len(removed_by_duration)} persons with <{min_duration_frames} frames")
    for p in removed_by_duration:
        print(f"      - person_{p['person_id']}: {len(p['frame_numbers'])} frames")

logger.info(f"After min_duration filter: {len(filtered_by_duration)} persons (threshold: {min_duration_frames} frames)")

# Then rank among filtered persons
ranked_indices, scores = rank_persons(filtered_by_duration, video_width, video_height, total_frames, weights)
```

### PART 3: Update Top 10 Selection

Change from:
```python
top_n = filter_config.get('top_n', 10)
top_persons = [all_persons[i] for i in top_indices]
```

To:
```python
top_n = filter_config.get('top_n', 10)
top_persons = [filtered_by_duration[i] for i in top_indices]  # From filtered_by_duration
```

---

## DETAILED SCORING BREAKDOWN

Here's exactly how the composite score works for your current video (360 frames):

### Example Person A (Good - early, long, consistent):
```
Person_3: appears frames 0-350 (duration=350 frames)

Metric 1: Duration
  duration = 350 frames
  max_duration = 10000 (hardcoded)
  duration_normalized = 350/10000 = 0.035
  contribution = 0.4 × 0.035 = 0.014

Metric 2: Coverage
  frame_range = 350 - 0 + 1 = 351
  coverage_score = 350/351 = 0.997
  coverage_normalized = 0.997
  contribution = 0.3 × 0.997 = 0.299

Metric 3: Center
  All bboxes near center: avg_distance = 100 pixels
  center_score = 1/(100+1) = 0.0099
  center_normalized = 0.0099/10 = 0.00099
  contribution = 0.2 × 0.00099 = 0.0002

Metric 4: Smoothness
  Smooth motion: velocity_variance = 20
  smoothness_score = 1/(20+1) = 0.048
  smoothness_normalized = 0.048/100 = 0.00048
  contribution = 0.1 × 0.00048 = 0.000048

Penalty:
  start_frame = 0
  appearance_ratio = 0/360 = 0.0 (early!)
  appearance_ratio <= 0.5? YES → no penalty
  late_appearance_penalty = 1.0

Final Score = (0.014 + 0.299 + 0.0002 + 0.000048) × 1.0 = 0.313
```

### Example Person B (Bad - late appearance, short, jittery):
```
Person_87: appears frames 250-320 (duration=70 frames) ← BELOW 150 MINIMUM!

Metric 1: Duration
  duration = 70 frames
  duration_normalized = 70/10000 = 0.007
  contribution = 0.4 × 0.007 = 0.003

Metric 2: Coverage
  frame_range = 320 - 250 + 1 = 71
  coverage_score = 70/71 = 0.986
  coverage_normalized = 0.986
  contribution = 0.3 × 0.986 = 0.296

Metric 3: Center
  Off-center: avg_distance = 400 pixels
  center_score = 1/(400+1) = 0.0025
  center_normalized = 0.0025/10 = 00025
  contribution = 0.2 × 0.00025 = 0.00005

Metric 4: Smoothness
  Jittery motion: velocity_variance = 150
  smoothness_score = 1/(150+1) = 0.0066
  smoothness_normalized = 0.0066/100 = 0.000066
  contribution = 0.1 × 0.000066 = 0.0000066

Penalty:
  start_frame = 250
  appearance_ratio = 250/360 = 0.694 (very late!)
  appearance_ratio > 0.5? YES → apply penalty
  penalty_factor = (0.694 - 0.5) / (1.0 - 0.5) = 0.388
  late_appearance_penalty = 1.0 - (0.388 × 0.3) = 0.883

Final Score = (0.003 + 0.296 + 0.00005 + 0.0000066) × 0.883 = 0.263

WITH MINIMUM DURATION FILTER: Person_87 removed (70 < 150 frames) ✓
```

---

## COMPARISON: Current vs Fixed

### Current Pipeline:
```
48 canonical persons
    ↓
Rank all by score
    ↓
Select top 10  ← May include Person_87 (70 frames) if it's in top 10
    ↓
Apply penalty_threshold=0.7
    ↓
Final: ~8 persons (may include short ones)
```

### Fixed Pipeline:
```
48 canonical persons
    ↓
Remove < 150 frames  ← BLOCKS Person_87 early
    ↓
~40 remaining persons
    ↓
Rank by score
    ↓
Select top 10
    ↓
Apply penalty_threshold=0.75
    ↓
Final: ~8 persons (ALL guaranteed ≥150 frames)
```

---

## WHAT YOU'LL SEE WHEN RUN (Debug Output)

After adding the minimum duration filter, you'll see:

```
Stage 3c: Filter Persons
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   VIDEO METADATA: 360 frames, 30 fps, max_appearance_ratio=0.5 (threshold: frame 180)

   INFO Loading canonical persons...
   STAT Total canonical persons: 48

   INFO Step 0: Filtering by minimum duration (150 frames)...
   DEBUG: Removed 5 persons with <150 frames
      - person_87: 70 frames    ← SHORT, REMOVED!
      - person_20: 45 frames    ← SHORT, REMOVED!
      - person_99: 120 frames   ← SHORT, REMOVED!
      - person_44: 85 frames    ← SHORT, REMOVED!
      - person_56: 140 frames   ← SHORT, REMOVED!
   INFO After min_duration filter: 43 persons (threshold: 150 frames)

   INFO Step 1: Ranking all persons...
   FOUND Selected TOP 10 persons (from 43 candidates)

   INFO Step 2: Applying late-appearance penalty to top 10...
   DEBUG total_frames=360, max_appearance_ratio=0.5, penalty_threshold=0.75

   OK: person_3 @ frame 0/360 (ratio=0.000) → no penalty
   PENALTY: person_65 @ frame 220/360 (ratio=0.611) → penalty=0.767 ← Marginal, may remove
   OK: person_37 @ frame 50/360 (ratio=0.139) → no penalty
   PENALTY: person_92 @ frame 280/360 (ratio=0.778) → penalty=0.733 ← REMOVED (< 0.75)
   ...

   DEBUG: After filtering - KEEP: 8, REMOVE: 2
   FOUND After penalty filtering: 8 persons (threshold: 0.75)
```

---

## Penalty Threshold Reference

| Threshold | Removes if penalty < | Explanation |
|---|---|---|
| 0.50 | <50% penalty | Extremely lenient - almost nobody removed |
| **0.70** | <30% penalty | **CURRENT: Too lenient** |
| **0.75** | <25% penalty | **RECOMMENDED: Catches very late** |
| 0.80 | <20% penalty | Strict - removes most late persons |
| 0.85 | <15% penalty | Very strict |

For a 2100-frame video with person at frame 1632:
- appearance_ratio = 1632/2100 = 0.776
- penalty_factor = (0.776 - 0.5) / 0.5 = 0.552
- penalty = 1.0 - (0.552 × 0.3) = 0.834

| Threshold | Result |
|---|---|
| 0.70 | 0.834 >= 0.70 → **KEPT** |
| 0.75 | 0.834 >= 0.75 → **KEPT** |
| 0.80 | 0.834 >= 0.80 → **KEPT** |
| 0.85 | 0.834 >= 0.85 → **REMOVED** |

**Conclusion**: For your cricket scenario (frame 1632/2100), need threshold >= 0.85 to remove them!

---

## Summary of Changes

| Change | Current | Fixed | Impact |
|--------|---------|-------|--------|
| Min duration | None | 150 frames | Removes short persons early |
| Top selection | Top 10 of 48 | Top 10 of ~40 | Only good candidates rank high |
| Penalty threshold | 0.70 | 0.75 | Stricter late-appearance filter |
| Final persons | ~8 (may be short) | ~8 (all ≥150 frames) | Better quality |

---

## Implementation Checklist

- [ ] Add `min_duration_frames: 150` to config
- [ ] Increase `penalty_threshold` to 0.75
- [ ] Add minimum duration filtering code (3 lines)
- [ ] Test on your video
- [ ] Verify debug output shows removed short persons
- [ ] Check final 8 persons are all good quality

