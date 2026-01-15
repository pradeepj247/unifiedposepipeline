# Per-Crop Features Approach - Implementation Summary

**Commit**: c8ee107  
**Date**: Jan 15, 2026  
**Status**: ✅ Ready for testing

---

## 🎯 The Problem We Identified

**High Similarities (0.99+) Despite Different People**:
- Previous approach: Average 16 crops per person → 1 averaged embedding
- Result: All 8 persons had ~0.99+ similarity (no discrimination)
- Root cause: **Averaging was collapsing feature variation**

### Why Averaging Failed

When you average 16 similar crops (same person, similar pose/lighting):
```
16 crop features → average → 1 embedding
All persons' averaged embeddings → very similar to each other
Similarity: 0.99+ (no discrimination)
```

---

## ✅ The Solution: Per-Crop Features

**New approach: Keep ALL crop features, compute similarities between feature sets**

### How it Works

For each pair of persons:
1. Extract 16 crop features from person A: (16, 256)
2. Extract 16 crop features from person B: (16, 256)
3. Compute all pairwise similarities: (16×16 = 256 comparisons)
4. Take **mean** of all pairwise similarities
5. Result: More nuanced similarity values

### Code Changes

**Old Flow**:
```python
# Per person:
16 crops → extract features → average → 1 embedding (256,)
# Result: 8 averaged embeddings → very similar to each other (0.99+)
```

**New Flow**:
```python
# Per person:
16 crops → extract features → keep ALL (16, 256) per person
# Similarity computation:
Person A (16, 256) vs Person B (16, 256) → mean pairwise similarity
# Result: More discriminative similarities
```

---

## 📋 Implementation Details

### New Function: `compute_similarity_matrix_from_features()`

```python
def compute_similarity_matrix_from_features(features_dict, person_ids=None, threshold=0.70, verbose=False):
    """
    Compute similarity from per-crop features (NO averaging).
    
    For each pair of persons:
    - Compute all pairwise similarities between their crops
    - Take mean of all similarities
    - This preserves feature variation better than averaging
    """
```

### Batch Size Updated

- ONNX (x0_25): batch_size=16 (model requirement)
- PyTorch (x1_0): batch_size=8 (was 8, now consistently used)

### Feature Storage

Instead of `embeddings.json` (averaged), now save `all_features.json`:
- Per-crop features for each person
- Crop counts tracked
- Full feature data preserved for clustering

---

## 🔬 Diagnostic Script

**Run on Colab to verify improvement**:

```bash
python snippets/diagnostic_averaging_collapse.py
```

This will:
1. Load current (averaged) embeddings
2. Simulate per-crop features
3. Compare similarity distributions
4. Show if per-crop approach has better variation

**Expected output**:
```
CURRENT (averaged):    std=0.002 (very narrow - PROBLEM)
MEDIAN aggregation:    std=0.150 (better)
PER-CROP (new):        std=0.200+ (BEST)
```

---

## 🚀 Expected Improvements

| Metric | Before (Averaged) | After (Per-Crop) |
|--------|---|---|
| Similarity range | 0.96-1.00 | 0.30-0.95 (expected) |
| Std deviation | ~0.002 | ~0.200+ (expected) |
| High-similarity pairs | 28 (ALL) | 8-12 (genuine duplicates) |
| Discrimination | ❌ None | ✅ Good |

---

## 🔄 Next Phase: Agglomerative Clustering

Once per-crop similarities show better variation:

1. **Input**: 8 persons × 16 crops each = 128 feature vectors
2. **Clustering**: Agglomerative Hierarchical Clustering
3. **Output**: 2-4 merged persons (true identities)

With better similarity values, clustering will:
- ✅ Find natural groupings
- ✅ Merge fragmented person instances
- ✅ Produce clean canonical persons

---

## 📁 Files Changed

**det_track/osnet_clustering.py**:
- ✅ New: `compute_similarity_matrix_from_features()` (per-crop approach)
- ✅ Updated: `create_similarity_matrix()` (uses new function)
- ✅ Updated: `save_similarity_results()` (saves all_features.json)
- ✅ Changed: Removed averaging step entirely

**snippets/diagnostic_averaging_collapse.py** (NEW):
- ✅ Test averaging vs per-crop
- ✅ Compare similarity distributions
- ✅ Verify improvement

---

## 🧪 Testing Instructions

### On Colab (after pulling latest code)

**Step 1: Run pipeline with new approach**
```bash
python det_track/run_pipeline.py --config det_track/configs/pipeline_config.yaml
```

**Step 2: Check output**
```bash
# Should see:
# [OSNet] ✓ Loaded PyTorch model
# ✅ OSNet clustering completed in 0.8s
# Total features stored: 128 (no averaging)
```

**Step 3: Verify similarities improved**
```bash
# Check the similarity_matrix.json
# Look for:
# - Similarities range: 0.3-0.95 (not 0.96-1.00)
# - Std deviation: > 0.15 (not 0.002)
# - High-similarity pairs: 8-12 (not 28)
```

**Step 4: Run diagnostic** (optional)
```bash
python snippets/diagnostic_averaging_collapse.py
```

---

## ⚠️ Known Limitations

1. **Slightly slower similarity computation**:
   - Before: 1×1 comparisons per pair
   - Now: 16×16 = 256 comparisons per pair
   - Expected overhead: ~100ms for 8 persons

2. **Output files changed**:
   - Old: `embeddings.json` (averaged)
   - New: `all_features.json` (per-crop)
   - HTML viewer may need update to handle new format

3. **Batch size**:
   - PyTorch now uses batch_size=8 consistently
   - Slightly slower inference (~0.57s vs X)

---

## 🎓 Why Per-Crop Works Better

**Mathematical intuition**:

**Averaging approach** (collapsed):
```
Person A: crops [1,2,3,4,5,6,7,8,...] → avg → 1 value
Person B: crops [1,2,3,4,5,6,7,8,...] → avg → 1 value
Similarity: just 1 comparison
```

If all crops are naturally similar (video background, lighting), averaging preserves too much similarity.

**Per-crop approach** (preserved):
```
Person A: [1,2,3,4,5,6,7,8,...] (16 values)
Person B: [1,2,3,4,5,6,7,8,...] (16 values)
Similarity: 16×16=256 comparisons → mean
```

If person A and B differ even slightly, some crop pairs will have lower similarity, pulling down the mean.

---

## 📊 Commit Info

**c8ee107**: "Switch to per-crop features instead of averaging"
- Files: 8 changed, 1618 insertions
- New: diagnostic_averaging_collapse.py
- Modified: osnet_clustering.py (major refactor)

---

## ✅ Checklist Before Production

- [ ] Run on Colab with new code
- [ ] Verify similarities show better distribution (not 0.99+)
- [ ] Confirm clustering timing is acceptable (<2s)
- [ ] Run diagnostic script to quantify improvement
- [ ] Update HTML viewer if needed (all_features vs embeddings format)
- [ ] Proceed to Agglomerative Clustering (Phase 2)

---

## 🎯 Next Action

**Run pipeline on Colab and report**:
1. New similarity range (should be 0.3-0.95, not 0.96-0.99)
2. Std deviation (should be > 0.15, not 0.002)
3. High-similarity pairs count (should be 8-12, not 28)
4. Execution time for Stage 4

If improved, ready to proceed with Agglomerative Clustering! 🚀
