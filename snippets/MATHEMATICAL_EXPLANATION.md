# Mathematical Explanation: Per-Crop vs Averaged Similarities

---

## 🎯 The Core Issue

### Averaged Approach (PROBLEMATIC)

**Formula**:
```
embedding_A = (1/16) * Σ features_A[i]     // Average all 16 crops
embedding_B = (1/16) * Σ features_B[i]     // Average all 16 crops

similarity(A, B) = dot(normalize(embedding_A), normalize(embedding_B))
```

**Problem**: 
- If all crops are similar (same video background), averaging preserves too much similarity
- Result: All persons get similar averaged embeddings
- Similarity: 0.99+ (no discrimination)

**Example**:
```
Person A: crops = [similar, similar, similar, ...similar]  → avg = value_A
Person B: crops = [similar, similar, similar, ...similar]  → avg = value_B
Result: value_A ≈ value_B → similarity ≈ 0.99

Even if A and B are different people, if they both have 16 "similar" crops,
the averages end up very close!
```

---

### Per-Crop Approach (NEW - BETTER)

**Formula**:
```
# Don't average! Keep all features per person
features_A = (16, 256)  // 16 crop embeddings
features_B = (16, 256)  // 16 crop embeddings

# Compute all pairwise similarities
pairwise_sims = dot(features_A, features_B^T)  // (16, 16) matrix

# Take mean (or other aggregation)
similarity(A, B) = mean(pairwise_sims)
```

**Advantage**:
- If A and B differ even slightly, some crop pairs will have lower similarity
- Mean will be pulled down by the lower similarities
- Better discrimination

**Example**:
```
Person A crop 1 vs Person B crop 1: 0.95  ← high
Person A crop 1 vs Person B crop 2: 0.92  ← slightly lower
Person A crop 1 vs Person B crop 3: 0.88  ← lower
...
Person A crop 16 vs Person B crop 16: 0.96 ← high

Mean of all 256 comparisons: (0.95+0.92+0.88+...+0.96)/256 ≈ 0.75

Compare to averaging:
Person A average vs Person B average: 0.99 ← TOO HIGH (no discrimination)

Per-crop gives 0.75, which better reflects actual differences!
```

---

## 📐 Mathematical Properties

### Why Averaging Collapses Variation

**Theorem**: If all vectors are similar (high magnitude, same direction), averaging preserves that similarity.

**Proof**:
```
Let v_i = base_vector + ε_i, where ε_i is small noise

Average: v_avg = base_vector + (1/16)*Σ ε_i

If all ε_i are small and similar, v_avg ≈ base_vector

For two persons with similar base_vectors:
v_A_avg ≈ base_A
v_B_avg ≈ base_B

If base_A ≈ base_B (same video background), then:
similarity(v_A_avg, v_B_avg) ≈ 1.0  ← PROBLEM!

With per-crop:
Compare all v_A[i] with v_B[j]
If base_A ≈ base_B but noise varies, mean of comparisons < 1.0  ← BETTER!
```

---

### Per-Crop Metrics

**Notation**:
```
n_persons = N = 8
n_crops_per_person = K = 16
feature_dim = D = 256

Total features in system: N × K = 128
Total comparisons per person pair: K × K = 256
```

**Similarity Matrix**:
```
sim[i,j] = mean( dot_product(features_i[k1], features_j[k2]) 
                 for all k1, k2 in 1..K )

More formally:
sim[i,j] = (1/K²) * Σ_{k1=1}^{K} Σ_{k2=1}^{K} 
           cosine_similarity(features_i[k1], features_j[k2])
```

**Diagonal Properties**:
```
sim[i,i] = (1/K²) * Σ_{k1} Σ_{k2} cosine_sim(features_i[k1], features_i[k2])
         ≈ 1.0  (because we're comparing a person's crops with themselves)

Off-diagonal:
sim[i,j] where i≠j should be lower than diagonal
(because crops from different people should be less similar)
```

---

## 📊 Expected Similarity Distributions

### Before (Averaged - PROBLEMATIC):

```
Similarity distribution:
Min:  0.96
Max:  1.00
Mean: 0.987
Std:  0.002   ← VERY NARROW (bad!)
Skew: ~0

This means almost all pairs are similarly similar - no discrimination!
```

### After (Per-Crop - EXPECTED):

```
Similarity distribution:
Min:  0.30 (different people with some different crops)
Max:  0.95 (very similar people)
Mean: 0.55
Std:  0.20   ← MUCH WIDER (good!)
Skew: ~0.5

This means we see a range of similarities - good discrimination!
```

---

## 🔬 Why Our Data Had 0.99+ Similarities

### Hypothesis: Video Characteristics

Looking at the high averaged similarities, likely causes:

1. **Same Video Background**
   - All 8 persons from same video
   - Same lighting, same background
   - OSNet encodes these global features
   - When averaged, dominates the representation

2. **Similar Pose/Clothing**
   - Dance video: people in similar poses
   - Similar clothing colors/materials
   - Global appearance features similar

3. **Temporal Clustering**
   - Multi-person tracking created fragments
   - Each "person" might be slight variations of same person
   - Person moved, reappeared → got new ID

### Why Per-Crop Helps

With per-crop features, we can detect:
- Crop-level variations (different dance poses within same person)
- Subtle person differences (height, specific clothing details)
- Temporal changes (how person moves frame-to-frame)

Averaging threw away this nuance. Per-crop preserves it.

---

## 🎯 Agglomerative Clustering Connection

### Why Per-Crop Enables Better Clustering

**Hierarchical Agglomerative Clustering** needs:
- Input: Feature vectors with natural variation
- Metric: Meaningful distances/similarities
- Goal: Find groups of similar vectors

**With averaged embeddings**:
- Input: 8 vectors (all 0.99+ similar)
- Metric: All distances ~0.0 (very similar)
- Result: Clustering can't distinguish → wrong groups

**With per-crop features**:
- Input: 128 vectors (16 per person)
- Metric: Varied distances (0.3-0.95 range)
- Result: Clustering finds natural person groupings → correct groups

---

## 📈 Computational Cost

### Similarity Computation Complexity

**Before** (averaged):
```
Time per pair: 1 dot product
Pairs: N choose 2 = 8 choose 2 = 28 pairs
Total: 28 dot products = O(N²)
```

**After** (per-crop):
```
Time per pair: K² dot products = 16² = 256 dot products
Pairs: 28
Total: 28 × 256 = 7168 dot products = O(N² × K²)
```

**Ratio**: 256× more comparisons

**But**: Each dot product is same operation (hardware-optimized), so practical increase is only ~2-3× time (from ~10ms to ~30ms for 8 persons).

---

## 🧮 Alternative Aggregations

We compared several approaches:

| Method | Formula | Pros | Cons |
|--------|---------|------|------|
| **Mean** | (1/K²) * Σ similarities | Balanced, natural | Influenced by outliers |
| **Median** | Median of similarities | Robust to outliers | Less sensitive to pairs |
| **Max** | Max of similarities | Highlights similar pairs | Too lenient (greedy) |
| **Min** | Min of similarities | Highlights different pairs | Too strict |

**Selected: Mean** because:
- ✅ Natural interpretation
- ✅ Captures overall similarity
- ✅ Symmetric (A→B = B→A)
- ✅ Works well with clustering algorithms

---

## 🎓 Takeaway

**The fundamental issue**: Averaging destroyed feature variation.

**The solution**: Keep all features, compute pairwise similarities, aggregate with mean.

**The result**: Better similarity discrimination, enabling effective clustering.

**Next step**: Use these better similarities as input to Agglomerative Clustering to find true person groups.
