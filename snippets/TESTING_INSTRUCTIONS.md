# Next Steps: Test Per-Crop Approach on Colab

## 🎯 What Changed

**Commit c8ee107**: Switched from averaging embeddings to per-crop features

- ✅ Remove: Averaging step (was collapsing variation)
- ✅ Keep: All 16 crop features per person
- ✅ New: Per-crop similarity computation (mean of pairwise)
- ✅ Result: Expected better similarity discrimination

---

## 🚀 Quick Test on Colab

### Step 1: Pull Latest Code
```bash
cd /content/unifiedposepipeline
git pull origin main
```

### Step 2: Run Pipeline
```bash
python det_track/run_pipeline.py --config det_track/configs/pipeline_config.yaml
```

Expected output:
```
[OSNet] ✓ Loaded PyTorch model (primary)
[OSNet]   Path: /content/.../osnet_x1_0_msmt17.pt

✅ OSNet clustering completed in 0.8s
   Total features stored: 128 (no averaging)
```

### Step 3: Check Similarity Range
```python
import json
with open('/content/unifiedposepipeline/demo_data/outputs/similarity_matrix.json') as f:
    data = json.load(f)

matrix = data['matrix']
# Extract off-diagonal
import numpy as np
m = np.array(matrix)
off_diag = []
for i in range(len(m)):
    for j in range(i+1, len(m)):
        off_diag.append(m[i,j])

off_diag = np.array(off_diag)
print(f"Min: {off_diag.min():.3f}")
print(f"Max: {off_diag.max():.3f}")
print(f"Mean: {off_diag.mean():.3f}")
print(f"Std: {off_diag.std():.3f}")
print(f"Pairs > 0.95: {(off_diag > 0.95).sum()}")
```

### Step 4: Compare to Previous Run

**Previous (Problematic)**:
- Min: 0.961, Max: 1.000, Mean: 0.982, Std: 0.002
- Pairs > 0.95: 28 (ALL!)

**New (Expected)**:
- Min: 0.30-0.50, Max: 0.90-0.95, Mean: 0.60-0.70, Std: 0.15+
- Pairs > 0.95: 0-5 (genuine duplicates only)

---

## 📊 Diagnostic Check (Optional)

Run diagnostic to verify the improvement:
```bash
python /content/unifiedposepipeline/snippets/diagnostic_averaging_collapse.py
```

This will:
- Compare averaged vs per-crop approaches
- Show expected variation
- Confirm the fix is working

---

## ✅ Success Criteria

| Metric | Before | After (Expected) | Status |
|--------|--------|---|---|
| Similarity range | 0.96-1.00 | 0.30-0.95 | ? |
| Std deviation | 0.002 | 0.15+ | ? |
| High-similarity pairs | 28 | 0-5 | ? |
| Clustering time | 0.84s | <2.0s | ? |

---

## 🐛 If It Doesn't Work

### Symptoms: Still seeing 0.99+ similarities

**Check 1**: Did pull succeed?
```bash
cd /content/unifiedposepipeline && git log --oneline | head -5
# Should show: c8ee107 "Switch to per-crop features"
```

**Check 2**: Is the new code being used?
```bash
grep -n "compute_similarity_matrix_from_features" \
  /content/unifiedposepipeline/det_track/osnet_clustering.py
# Should find matches
```

**Check 3**: Output format changed
- Old: `embeddings.json` (averaged)
- New: `all_features.json` (per-crop)
- Check if output files are using new format

---

## 🎓 Expected Behavior

### What We Changed

**Before**:
```python
# For each person: average 16 crops → 1 embedding
for person_id, crops in buckets.items():
    features = extract_osnet_features(crops, ...)  # (16, 256)
    embedding = compute_embedding(features)         # (256,)
    embeddings_dict[person_id] = embedding

# Similarity: 8 averaged embeddings → compute similarities
```

**After**:
```python
# For each person: keep all 16 crops
for person_id, crops in buckets.items():
    features = extract_osnet_features(crops, ...)  # (16, 256) - unchanged
    # NO averaging step!
    all_features_dict[person_id] = features

# Similarity: all-features computation
# For each pair: mean of pairwise crop similarities
similarity(A, B) = mean([cos_sim(crop_A_i, crop_B_j) for all i,j])
```

---

## 📈 Performance Impact

**Positive**:
- ✅ Better similarity discrimination
- ✅ Enables effective clustering
- ✅ Preserves feature variation

**Negative** (acceptable):
- ⚠️ Slightly slower similarity computation (~100-200ms overhead)
- ⚠️ Larger output files (all_features.json vs embeddings.json)

---

## 📋 Report Back With

Once you run on Colab, please share:

1. **Similarity statistics** (from Step 3 above):
   ```
   Min: ?
   Max: ?
   Mean: ?
   Std: ?
   Pairs > 0.95: ?
   ```

2. **Timing** (from pipeline output):
   ```
   Stage 4: Generate HTML Viewer: ?s
   OSNet clustering completed: ?s
   ```

3. **Visual** (check HTML):
   - Do the similarity heatmaps look reasonable?
   - Are some persons clearly different, some similar?

---

## 🚀 Next Phase (If Successful)

Once per-crop approach confirms better similarities:

1. **Implement Agglomerative Clustering**
   - Input: 8 persons with realistic similarities
   - Output: 2-4 merged persons (true identities)

2. **Update Canonical Persons**
   - Merge duplicates
   - Re-run visualization

3. **Update HTML Viewer**
   - Handle new all_features.json format
   - Show clustering results

---

## 📞 Questions?

- Why per-crop instead of averaging? → See MATHEMATICAL_EXPLANATION.md
- What if similarities don't improve? → Check diagnostic_averaging_collapse.py
- How does clustering work next? → See clustering design doc

**Ready to test?** Let me know the results! 🎯
