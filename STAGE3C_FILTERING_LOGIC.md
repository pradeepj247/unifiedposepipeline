# Stage 3c: Complete Filtering & Ranking Logic

## Overview

Stage 3c performs **TWO-STEP filtering** to reduce persons from 40+ → 8-10:
1. **Step 1**: Rank ALL persons by composite score, select **TOP 10**
2. **Step 2**: Apply late-appearance penalty to top 10, filter by threshold → **~8 persons**

---

## STEP 1: Ranking Score Calculation

### The Composite Score Formula

For each person, calculate 4 independent metrics, normalize them, then combine with weights:

```
final_score = (
    0.4 × duration_normalized +
    0.3 × coverage_normalized +
    0.2 × center_normalized +
    0.1 × smoothness_normalized
) × late_appearance_penalty
```

### Metric 1: Duration (Weight: 0.4 = 40%)

**What it measures**: How long the person is visible in the video

```python
duration = len(frames)  # Number of frames person appears in
duration_normalized = min(duration / 10000, 1.0)  # Normalize to 0-1 (assuming max 10000 frames)
```

**Example**:
- Person appears in 300 frames → 300/10000 = 0.03
- Person appears in 200 frames → 200/10000 = 0.02
- Winner: First person (longer presence)

---

### Metric 2: Coverage (Weight: 0.3 = 30%)

**What it measures**: Percentage of the timespan when person is actually detected

```python
start_frame = frames[0]
end_frame = frames[-1]
frame_range = end_frame - start_frame + 1
coverage_score = duration / frame_range  # Ratio of frames present in timespan
coverage_normalized = coverage_score  # Already 0-1
```

**Example**:
- Person appears frame 100-200: timespan = 101 frames
  - If detected in 90 frames → coverage = 90/101 = 0.89 ✓ Good
  - If detected in 30 frames → coverage = 30/101 = 0.30 ✗ Bad
- Person appears frame 50-500: timespan = 451 frames
  - If detected in 100 frames → coverage = 100/451 = 0.22 ✗ Bad

---

### Metric 3: Center Bias (Weight: 0.2 = 20%)

**What it measures**: How close the person is to the frame center

```python
centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2  # Centroid of each bbox
frame_center = (width/2, height/2)  # Center of frame
distances = ||centers - frame_center||  # Euclidean distance
center_score = 1.0 / (distances.mean() + 1)  # Inverse distance
center_normalized = min(center_score / 10, 1.0)
```

**Example**:
- Person always near center: mean_distance = 50 pixels → score = 1/(50+1) = 0.0196, normalized ≈ 0.002
- Person always at edge: mean_distance = 500 pixels → score = 1/(500+1) = 0.002, normalized ≈ 0.0002
- Winner: Person near center (higher score)

---

### Metric 4: Smoothness (Weight: 0.1 = 10%)

**What it measures**: How stable the motion is (less jitter = higher score)

```python
velocities = diff(centers)  # Frame-to-frame motion vectors
velocity_variance = var(||velocities||)  # Variance of motion magnitude
smoothness_score = 1.0 / (velocity_variance + 1)  # Inverse variance
smoothness_normalized = min(smoothness_score / 100, 1.0)
```

**Example**:
- Smooth motion (person walking straight): variance = 10 → score = 1/11 = 0.091
- Jittery motion (person twitching): variance = 100 → score = 1/101 = 0.01
- Winner: Smooth motion person (higher score)

---

## STEP 2: Late-Appearance Penalty

Applied **AFTER** calculating the base score to penalize persons detected late in video.

### Penalty Calculation

```python
max_appearance_ratio = 0.5  # Persons appearing after 50% of video get penalized
appearance_ratio = start_frame / total_frames

if appearance_ratio > 0.5:
    # Person started after 50% of video
    penalty_factor = (appearance_ratio - 0.5) / (1.0 - 0.5)  # Scale to 0-1
    penalty = 1.0 - (penalty_factor × 0.3)  # Linear penalty, max 30%
else:
    # Person started before 50% of video
    penalty = 1.0  # No penalty
```

### Penalty Values by Appearance Ratio

| appearance_ratio | start_frame (for 360-frame video) | penalty_factor | penalty | Description |
|---|---|---|---|---|
| 0.00 | frame 0 | 0.00 | 1.0 (0% penalty) | Early appearance → no penalty |
| 0.25 | frame 90 | 0.00 | 1.0 (0% penalty) | Still early |
| **0.50** | **frame 180** | **0.00** | **1.0 (0% penalty)** | **THRESHOLD: no penalty** |
| 0.60 | frame 216 | 0.20 | 0.94 (6% penalty) | Starting to get late |
| 0.75 | frame 270 | 0.50 | 0.85 (15% penalty) | Getting very late |
| **0.90** | **frame 324** | **0.80** | **0.76 (24% penalty)** | **VERY LATE** |
| 1.00 | frame 360+ | 1.00 | 0.70 (30% penalty) | Latest possible → max penalty |

### Example with Actual Cricket Data

For a **~360 frame** video with threshold at 0.5:
- **Person_3**: starts frame 0 → ratio=0.0 → **NO penalty**
- **Person_87**: starts frame 1534? (wait, this doesn't make sense with 360-frame video)

**WAIT - This reveals the problem with the debug output!** The user showed frame 1534 but if video is 360 frames, that's impossible. The `total_frames` was being set to 10000 by default.

---

## STEP 1 RESULT: Top 10 Persons

After calculating composite scores with late-appearance penalty, select the **top 10 by final_score**.

Example ranking:
1. Person_65: score=0.85 (high duration, good coverage, smooth)
2. Person_37: score=0.82 (good on all metrics)
3. Person_3: score=0.78 (early, stable)
...
10. Person_20: score=0.62 (barely made top 10)

---

## STEP 2: Penalty Threshold Filtering

From the top 10, remove any person whose **penalty < 0.7**.

```python
penalty_threshold = 0.7  # Keep if penalty >= 0.7

# With 360-frame video:
# This removes persons with appearance_ratio > 0.6 (frame 216+)
```

### Penalty Threshold Interpretation

| Threshold | Removes if penalty < | Explanation |
|---|---|---|
| 0.5 | persons with >75% penalty | Very lenient - almost no one removed |
| **0.7** | persons with >30% penalty | **CURRENT: removes only VERY late persons** |
| 0.85 | persons with >15% penalty | Moderate - removes late persons |
| 0.95 | persons with >5% penalty | Strict - removes almost anyone not super early |

---

## FINAL RESULT: ~8 Persons

After both steps:
- Input: 40+ canonical persons
- After Step 1 (top 10): 10 persons
- After Step 2 (penalty threshold 0.7): ~8 persons (2-3 removed if they appeared very late)

---

## Configuration Values (pipeline_config.yaml)

```yaml
stage3c_filter:
  filtering:
    top_n: 10                         # STEP 1: Select top 10
    penalty_threshold: 0.7            # STEP 2: Keep if penalty >= 0.7
    crops_per_person: 50              # Extract 50 crops per person
    
    weights:
      duration: 0.4                   # Weight for duration metric (40%)
      coverage: 0.3                   # Weight for coverage metric (30%)
      center: 0.2                     # Weight for center metric (20%)
      smoothness: 0.1                 # Weight for smoothness metric (10%)
      max_appearance_ratio: 0.5       # Penalty threshold: 50% of video
```

---

## What Can Go Wrong

### Problem 1: Wrong total_frames (NOW FIXED ✅)
- **Was**: Hardcoded to 10000
- **Effect**: Persons at frame 1632 were treated as 16.3% of video (should be way over 100%)
- **Result**: Nobody got penalized
- **Fix**: Read actual frame count from video file

### Problem 2: Penalty threshold too low
- **If threshold = 0.5**: Almost nobody removed (too lenient)
- **If threshold = 0.99**: Almost everyone removed (too strict)
- **Current: 0.7**: Removes persons with >30% penalty (appears after ~75% of video)

### Problem 3: Ranking weights don't sum to 1.0
- **Current**: 0.4 + 0.3 + 0.2 + 0.1 = 1.0 ✓ Good
- **If wrong**: Scores become unpredictable

### Problem 4: Center bias favors center persons
- Cricket: Batsman usually center, might filter out fielders on edges
- Trade-off: Center bias is intentional to focus on main action

---

## Summary Table

| Step | Input | Filter | Output |
|---|---|---|---|
| **3b: Grouping** | 49 tracklets | Geometric merging | 48 canonical persons |
| **3c.1: Top 10** | 48 persons | Rank by 4 metrics + penalty | 10 persons |
| **3c.2: Threshold** | 10 persons | Keep if penalty ≥ 0.7 | ~8 persons |
| **3d: ReID Merge** | ~8 persons | OSNet similarity | ~6 persons |
| **4: Visualization** | ~6 persons | WebP generation | HTML + 6 WebPs |

---

## Key Changes Made

1. ✅ **Fixed total_frames**: Now reads from actual video (Phase 1)
2. ✅ **Added debug output**: Shows exact penalty values per person (Phase 2)
3. ⏳ **Pending**: Run on Colab to verify filtering works with correct frame count

