# Filtering Logic Analysis & Missing Piece

## Your Questions Answered

### Q1: What is the difference between Duration and Coverage?

**Duration** = How many total frames a person appears in
- Example: Person appears in 200 frames (doesn't matter if scattered)

**Coverage** = What percentage of their timespan they're actually detected
- Timespan = from first appearance to last appearance
- Example: Person appears frame 100 to frame 500 (timespan = 401 frames)
  - If detected in 200 of those 401 frames → coverage = 200/401 = **49.9%**
  - If detected in 380 of those 401 frames → coverage = 380/401 = **94.8%** ✓

**Why both?**
- Duration alone: rewards people who appear for a long time but with many gaps
- Coverage: rewards consistent presence (no gaps)
- Together: "Person should appear long AND consistently"

---

### Q2: What is Center?

**Center Bias** = How close to the frame center the person is

```python
# Calculate distance from frame center
frame_center = (width/2, height/2)  # e.g., (960, 540) for 1920x1080

for each bounding box:
    center = (bbox center point)
    distance = ||center - frame_center||
    
center_score = 1 / (average_distance + 1)
```

**Example (1920x1080 video)**:
- Frame center = (960, 540)
- Person always at center: distance=0 → score = 1/(0+1) = **1.0** ✓ High
- Person at corner: distance=1000 → score = 1/(1000+1) = **0.001** ✗ Low

**Why?** In sports videos, main action is usually center-frame.

---

### Q3: WHERE IS THE MINIMUM 150 FRAME FILTER??

**This is what you're looking for!** The code currently **DOES NOT** have this!

Current code:
```python
# ❌ NO FILTER FOR minimum duration
top_n = filter_config.get('top_n', 10)
top_persons = [all_persons[i] for i in ranked_indices[:top_n]]
# Just takes top 10, regardless of how short they are!
```

**This should be:**
```python
# ✅ SHOULD HAVE: Minimum 150 frames filter
min_duration_frames = filter_config.get('min_duration_frames', 150)  # 5 seconds at 30fps
filtered_by_duration = [p for p in all_persons if len(p['frame_numbers']) >= min_duration_frames]

# Then rank among those
ranked_indices, scores = rank_persons(filtered_by_duration, ...)
```

---

### Q4: Will the late-appearance threshold (50%) catch persons appearing after frame 1050?

**YES**, but with caveats. Let's work through your example:

**Scenario:**
- Video has 2100 frames
- 50% threshold = frame 1050
- Person appears starting frame 1632

```python
appearance_ratio = 1632 / 2100 = 0.776  (77.6% of video)

if 0.776 > 0.5:  # YES, over threshold
    penalty_factor = (0.776 - 0.5) / (1.0 - 0.5)
    penalty_factor = 0.276 / 0.5 = 0.552
    penalty = 1.0 - (0.552 × 0.3)
    penalty = 1.0 - 0.166 = 0.834  (16.6% penalty)
```

**Then:**
- Person needs `penalty >= penalty_threshold` to be kept
- If threshold = 0.7: 0.834 >= 0.7 → **KEPT** ✓
- If threshold = 0.85: 0.834 >= 0.85 → **REMOVED** ✗

**The problem:** threshold=0.7 is too lenient! It only removes persons with >30% penalty.

---

## THE MISSING FILTER: Minimum Duration

Here's what should be in the config:

```yaml
stage3c_filter:
  filtering:
    min_duration_frames: 150           # ← THIS IS MISSING!
    top_n: 10
    penalty_threshold: 0.7
    crops_per_person: 50
    
    weights:
      duration: 0.4
      coverage: 0.3
      center: 0.2
      smoothness: 0.1
      max_appearance_ratio: 0.5        # Threshold for late-appearance penalty
```

---

## Current Filtering Logic (What Actually Happens)

```
ALL PERSONS (40+)
    ↓
STEP 1: Rank by composite score
  - Duration: 40% weight
  - Coverage: 30% weight
  - Center: 20% weight
  - Smoothness: 10% weight
  - Multiply by late_appearance_penalty
    ↓
SELECT TOP 10  ← No minimum duration check!
    ↓
STEP 2: Apply penalty threshold (0.7)
  - Remove if penalty < 0.7
  - Keeps persons with >30% penalty
    ↓
FINAL: ~8 persons  (may include very short persons)
```

---

## What SHOULD Happen

```
ALL PERSONS (40+)
    ↓
FILTER 1: Minimum duration (150 frames = 5 seconds)  ← MISSING!
  - Remove all persons with <150 frames
    ↓
STEP 1: Rank remaining by composite score
    ↓
SELECT TOP 10
    ↓
STEP 2: Apply penalty threshold
    ↓
FINAL: ~8 persons (all guaranteed ≥150 frames)
```

---

## Debug Output You Asked For

You want to see when running on your video:
1. **Top 10 scores** - all metrics broken down
2. **Which ones got penalized** - penalty value for each

The code ALREADY HAS this! Look at the debug print statements in `stage3c_filter_persons.py`:

```python
# This line shows EXACTLY what you want:
print(f"   PENALTY: person_{person['person_id']} @ frame {start_frame}/{total_frames} (ratio={appearance_ratio:.3f}) → penalty={penalty:.3f}")
print(f"   OK: person_{person['person_id']} @ frame {start_frame}/{total_frames} (ratio={appearance_ratio:.3f}) → no penalty")
```

This gets printed for each of the top 10 persons!

---

## Action Items

### MUST FIX:
1. **Add minimum duration filter** before ranking
2. **Add to config** - `min_duration_frames: 150`
3. **Consider adjusting penalty_threshold** - maybe 0.75 or 0.8 instead of 0.7

### SHOULD CHECK:
1. Run on your current video and look at the `PENALTY:` debug lines
2. Verify that the top 10 matches what you expect
3. Check if any short persons are sneaking through

---

## Example Output (What You'll See)

When you run `stage3c_filter_persons.py`, you'll see:

```
Stage 3c: Filter Persons
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   VIDEO METADATA: 360 frames, 30 fps, max_appearance_ratio=0.5 (threshold: frame 180)

   INFO Loading canonical persons...
   STAT Total canonical persons: 48

   INFO Step 1: Ranking all persons...
   FOUND Selected TOP 10 persons (from 48 candidates)

   INFO Step 2: Applying late-appearance penalty...
   DEBUG total_frames=360, max_appearance_ratio=0.5, penalty_threshold=0.7

   OK: person_3 @ frame 0/360 (ratio=0.000) → no penalty
   OK: person_65 @ frame 50/360 (ratio=0.139) → no penalty
   PENALTY: person_87 @ frame 220/360 (ratio=0.611) → penalty=0.767
   PENALTY: person_20 @ frame 250/360 (ratio=0.694) → penalty=0.621
   OK: person_37 @ frame 100/360 (ratio=0.278) → no penalty
   ...

   DEBUG: After filtering - KEEP: 8, REMOVE: 2
   DEBUG: Removed persons: [20]
```

---

## Summary Table

| Metric | What It Measures | Weight | Example |
|--------|------------------|--------|---------|
| Duration | Total frames person appears | 40% | 200 frames |
| Coverage | % of timespan detected | 30% | 85% detected in their timespan |
| Center | Distance from frame center | 20% | Close to center = better |
| Smoothness | Motion stability | 10% | Smooth walking = better |
| **Late-appearance penalty** | **When they first appear** | **×1.0 or ×0.7-1.0** | **Frame 1632/2100 = 16% penalty** |

---

## Code Locations

- **Ranking logic**: [stage3c_filter_persons.py](stage3c_filter_persons.py#L80-L140)
- **Penalty calculation**: [stage3c_filter_persons.py](stage3c_filter_persons.py#L112-L125)
- **Filtering threshold**: [stage3c_filter_persons.py](stage3c_filter_persons.py#L289-L295)
- **Config**: `configs/pipeline_config.yaml` → `stage3c_filter:`

