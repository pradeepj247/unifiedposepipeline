# Stage 3D: ReID Merging Logic Explained

## Overview

Stage 3D uses OSNet (ReID model) to identify persons who were split into multiple detections and merge them back together. This happens when:
- A person leaves the frame and comes back later
- Tracking loses the person temporarily
- Person is occluded and reappears

---

## The 2-Phase Process

### Phase 1: Candidate Pair Selection (Who to Compare?)
Only compare persons that **could** be the same person based on temporal constraints.

### Phase 2: Visual Similarity Matching (Are They Actually the Same?)
Use OSNet embeddings to determine if candidates are visually similar enough to merge.

---

## Phase 1: Candidate Pair Selection

### Rule 1: Minimal Temporal Overlap
**Logic**: If two persons appear at the same time for too long, they're probably different people.

```python
overlap_tolerance = 15  # frames

Person A: frames 0-300
Person B: frames 50-400
Overlap: frames 50-300 = 251 frames > 15 → NOT a candidate pair ✗

Person A: frames 0-300
Person C: frames 310-600
Overlap: 0 frames → IS a candidate pair ✓
```

**Why?** Two persons detected simultaneously (beyond 15-frame tolerance) are likely different people.

**Example**: Bowler and batsman both visible → different persons

---

### Rule 2: Reasonable Gap Size
**Logic**: If the gap between appearances is too large, they're probably different people.

```python
temporal_gap_max = 60  # frames (2 seconds at 30fps)

Person A: frames 0-300 (ends at 300)
Person D: frames 500-800 (starts at 500)
Gap: 500 - 300 = 200 frames > 60 → NOT a candidate pair ✗

Person A: frames 0-300 (ends at 300)
Person E: frames 310-600 (starts at 310)
Gap: 310 - 300 = 10 frames ≤ 60 → IS a candidate pair ✓
```

**Why?** If someone is gone for more than 2 seconds, it's probably a different person appearing.

**Example**: Player walks off screen for 10 seconds → probably a new player when someone appears

---

### Candidate Selection Summary

For ALL pairs of persons, check:
1. ✓ Overlap ≤ 15 frames (not simultaneously visible for long)
2. ✓ Gap ≤ 60 frames (reappearance within 2 seconds)

Only pairs meeting BOTH conditions are analyzed.

---

## Phase 2: Visual Similarity Matching

### Step 1: Extract OSNet Features
For each person, take their crops and extract visual embeddings:

```python
# Use top 16 crops per person (best quality)
crops_selected = person_crops[:16]

# Extract 512-dimensional feature vectors
features = osnet_model(crops_selected)  # Shape: (16, 512)

# Average across all crops → single feature per person
person_embedding = features.mean(axis=0)  # Shape: (512,)
```

**Result**: Each person has a 512-D vector representing their appearance.

---

### Step 2: Compute Cosine Similarity
For each candidate pair, measure how similar their embeddings are:

```python
similarity_threshold = 0.60  # 60% similarity required

# Normalize embeddings
feat1_norm = feat1 / ||feat1||
feat2_norm = feat2 / ||feat2||

# Cosine similarity (dot product of normalized vectors)
similarity = feat1_norm · feat2_norm  # Range: -1 to 1 (typically 0.3 to 0.95)

# Decision
if similarity >= 0.60:
    MERGE them ✓
else:
    Keep separate ✗
```

**Similarity ranges**:
- **0.85-1.0**: Very high similarity (same person, different angles)
- **0.70-0.85**: High similarity (same person, lighting changes)
- **0.60-0.70**: Moderate similarity (same person, significant pose/clothing changes)
- **0.40-0.60**: Low similarity (probably different persons)
- **< 0.40**: Very low similarity (definitely different persons)

---

### Step 3: Build Connected Components
If multiple persons are all similar to each other, merge them into one group:

```python
# Example similarities:
person_3 ↔ person_65: 0.82 ✓
person_3 ↔ person_20: 0.73 ✓
person_65 ↔ person_20: 0.68 ✓

# Union-Find algorithm creates connected component:
Group 1: [person_3, person_20, person_65]
→ Merged into: person_3
```

**Why Union-Find?** Handles transitive relationships:
- If A ~ B and B ~ C, then merge A, B, C together
- Even if A and C aren't directly similar (A ↔ C < 0.60), they're connected through B

---

## Configuration Parameters

From `pipeline_config.yaml`:

```yaml
stage3d_refine:
  merging:
    temporal_gap_max: 60                   # Max 60 frames gap (2 sec)
    temporal_overlap_tolerance: 15         # Allow 15 frames overlap
    similarity_threshold: 0.60             # 60% cosine similarity required
    num_crops_per_person: 16               # Use top 16 crops for feature extraction
```

**Tuning guidance**:

| Parameter | Current | Strict (fewer merges) | Lenient (more merges) |
|-----------|---------|----------------------|---------------------|
| `temporal_gap_max` | 60 | 30 (1 sec) | 120 (4 sec) |
| `temporal_overlap_tolerance` | 15 | 5 frames | 30 frames |
| `similarity_threshold` | 0.60 | 0.70 (70%) | 0.50 (50%) |

---

## Complete Example Walkthrough

### Input: 10 persons from Stage 3C

```
Person 1:  frames 0-155
Person 3:  frames 0-2019  (main batsman - throughout video)
Person 4:  frames 0-560
Person 14: frames 103-352
Person 20: frames 201-793  (bowler)
Person 29: frames 360-784  (fielder)
Person 37: frames 479-1133
Person 40: frames 555-1050
Person 65: frames 898-1634
Person 66: frames 980-1505
```

---

### Phase 1: Candidate Pair Selection

Check all 45 pairs (10 choose 2):

**Example 1: Person 3 ↔ Person 65**
```
Person 3:  frames 0-2019
Person 65: frames 898-1634

Overlap: frames 898-1634 = 737 frames (both visible)
737 > 15 → OVERLAPS TOO MUCH → NOT a candidate ✗
```

**Example 2: Person 4 ↔ Person 40**
```
Person 4:  frames 0-560 (ends at 560)
Person 40: frames 555-1050 (starts at 555)

Overlap: frames 555-560 = 6 frames
6 ≤ 15 → Minimal overlap ✓

Gap (if we ignore overlap): 555 - 560 = -5 (they barely overlap)
Since overlap is small, treat as near-sequential
Gap ≤ 60 → Reasonable gap ✓

→ IS a candidate pair ✓
```

**Example 3: Person 1 ↔ Person 20**
```
Person 1:  frames 0-155 (ends at 155)
Person 20: frames 201-793 (starts at 201)

Overlap: 0 frames (no overlap) ✓
Gap: 201 - 155 = 46 frames ≤ 60 ✓

→ IS a candidate pair ✓
```

**Result**: Only 3 pairs meet criteria:
- Person 1 ↔ Person 20
- Person 4 ↔ Person 40
- Person 14 ↔ Person 29

---

### Phase 2: Visual Similarity

Extract OSNet features and compute similarities:

```
=== REID SIMILARITY ANALYSIS ===
Threshold: 0.600
Analyzed 3 pairs:

person_1 ↔ person_20:  similarity = 0.6750 ✓ MERGE
person_4 ↔ person_40:  similarity = 0.7875 ✓ MERGE
person_14 ↔ person_29: similarity = 0.6857 ✓ MERGE
```

All 3 pairs exceed 0.60 threshold → merge them!

---

### Phase 3: Build Groups & Merge

```
=== MERGE GROUPS ===
Found 3 group(s) to merge:

Group 1: person_1 + person_20 → person_1
Group 2: person_4 + person_40 → person_4
Group 3: person_14 + person_29 → person_14

Final: 7 persons (4 singles + 3 merged groups)
```

**Singles (not merged)**:
- person_3, person_37, person_65, person_66

**Merged**:
- person_1 (absorbed person_20)
- person_4 (absorbed person_40)
- person_14 (absorbed person_29)

---

## Why This Design?

### 1. **Temporal Constraints First** (Efficient)
- Reduces comparisons from 45 pairs to 3 pairs (93% reduction!)
- Only compare persons who COULD be the same based on timing
- Avoids expensive OSNet computation for obviously different persons

### 2. **Visual Similarity Second** (Accurate)
- Among candidates, use deep learning to verify
- OSNet trained on person re-identification (designed for this task)
- Robust to pose, lighting, angle changes

### 3. **Union-Find for Transitivity** (Complete)
- Handles chains: A→B→C all merged together
- Ensures consistent grouping even with multiple splits

---

## Common Issues & Solutions

### Issue 1: Too Many Merges (Over-merging)
**Symptom**: Different persons merged together

**Solutions**:
- ↑ Increase `similarity_threshold` (0.65 or 0.70)
- ↓ Decrease `temporal_gap_max` (30 frames = 1 sec)
- ↓ Decrease `temporal_overlap_tolerance` (5 frames)

### Issue 2: Too Few Merges (Under-merging)
**Symptom**: Same person stays split across multiple IDs

**Solutions**:
- ↓ Decrease `similarity_threshold` (0.55 or 0.50)
- ↑ Increase `temporal_gap_max` (120 frames = 4 sec)
- ↑ Increase `temporal_overlap_tolerance` (30 frames)

### Issue 3: No Candidate Pairs Found
**Symptom**: "No candidate pairs found (all persons overlap temporally)"

**Cause**: All persons visible simultaneously → no one to merge

**Solution**: This is correct behavior! If everyone is on screen at once, they're all different people.

---

## Summary

**Stage 3D merges split detections using 2 phases**:

1. **Temporal filtering**: Only compare persons who appear sequentially (not simultaneously)
2. **Visual matching**: Use OSNet ReID to verify visual similarity

**Key parameters** (from config):
- `temporal_gap_max: 60` - Max gap between appearances (2 seconds)
- `temporal_overlap_tolerance: 15` - Allow small overlaps (tracking noise)
- `similarity_threshold: 0.60` - Visual similarity required (60%)

**Output**: Fewer persons (merged duplicates) with consistent identities across the video.

