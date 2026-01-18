# Stage 4 OSNet Clustering - TODO List & Issues

**Date**: January 15, 2026
**Status**: Phase 4 Enhancement Planning

---

## 🔴 ISSUE #1: HTML Person Count Mismatch (8 vs 10)

**Problem**: 
- Analysis shows 8 persons in similarity matrix
- HTML viewer displays 10 persons
- Early appearance filter (50% threshold) should eliminate late-appearing persons
- This filter is NOT being applied consistently

**Root Cause Analysis**:
- Location of filter: `stage4_generate_html.py` line ~175
- Filter logic: `max_first_appearance_ratio: 0.5` 
- Applied in: `extract_crops_from_video()` function in `ondemand_crop_extraction.py`
- NOT applied in: HTML generation (`_generate_html_viewer()`)

**Affected Files**:
- `ondemand_crop_extraction.py` - Crop extraction applies filter
- `stage4_generate_html.py` - Passes person_buckets to HTML (already filtered)
- `osnet_clustering.py` - Uses filtered buckets (correct)

**Why Mismatch Occurs**:
- Crop extraction correctly filters to 8 persons for WebP/clustering
- But HTML gets embedded image count from somewhere else (raw person_info?)
- Need to verify metadata passed to `_generate_html_viewer()`

---

## 🟡 ISSUE #2: Use Stronger OSNet Model

**Current**: `osnet_x0_25_msmt17.onnx`
- Lightweight model
- Fast inference

**Available**: `osnet_x1_0_msmt17.pt` (PyTorch format)
- Larger model (better feature extraction)
- Still minimal overhead (~1-2 seconds for 8 persons)

**Benefits**:
- Stronger embeddings → More discriminative features
- Better separation of distinct persons
- Potentially identify more nuanced duplicates

**Considerations**:
- Must use PyTorch backend (current code supports both ONNX and PyTorch)
- May need to update default model path in config
- Batch size handling differs (PyTorch = variable, ONNX = fixed)

---

## 🟠 ISSUE #3: Implement Automatic Clustering Algorithm

**Current Approach**:
- Manual threshold (70% similarity)
- Connected components graph traversal
- Limited to pairwise comparison analysis

**Proposed Approach**:
- Use **Agglomerative Clustering** (hierarchical)
  - Deterministic, interpretable dendrograms
  - No hyperparameter tuning (distance threshold)
  - Works well for small datasets (N=8)
- Alternative: **DBScan**
  - Requires epsilon and min_samples tuning
  - Better for outlier detection (person 3 = isolated?)

**Implementation**:
```python
from scipy.cluster.hierarchy import linkage, fcluster
# Or sklearn alternatives if scipy unavailable

# Agglomerative approach:
Z = linkage(condensed_distances, method='average')
clusters = fcluster(Z, t=0.25, criterion='distance')
```

**Advantages**:
- Automatic cluster detection (no manual threshold)
- Captures multi-level relationships
- Produces interpretable dendrograms
- No random initialization (unlike K-means)

---

## 📋 TODO LIST (Prioritized)

### **TIER 1 - CRITICAL BUGS** 🔴

- [ ] **TODO 1.1**: Investigate HTML person count mismatch (8 vs 10)
  - Check metadata passed to `_generate_html_viewer()`
  - Verify early appearance filter is applied before HTML generation
  - Trace flow: canonical_persons → extract_crops_from_video → person_buckets → HTML
  - **Files**: ondemand_crop_extraction.py, stage4_generate_html.py
  - **Time est**: 30-45 min

- [ ] **TODO 1.2**: Fix HTML to show only filtered persons
  - Update metadata dictionary to match filtered person_buckets
  - Ensure HTML person count = num of filtered persons
  - Verify WebP files count matches
  - **Files**: ondemand_crop_extraction.py (metadata creation)
  - **Time est**: 15-20 min

### **TIER 2 - MODEL IMPROVEMENTS** 🟡

- [ ] **TODO 2.1**: Add support for stronger OSNet model (x1_0)
  - Create function to detect model type (.onnx vs .pt)
  - Update `load_osnet_model()` to handle both automatically
  - Add config option: `osnet_model_variant: x0_25 | x1_0`
  - **Files**: osnet_clustering.py, pipeline_config.yaml
  - **Time est**: 20-30 min

- [ ] **TODO 2.2**: Benchmark both models
  - Time x0_25 vs x1_0 on current dataset
  - Compare embedding quality (visual inspection of similarities)
  - Measure accuracy improvement for duplicate detection
  - **Files**: New benchmark script
  - **Time est**: 10-15 min (automated)

- [ ] **TODO 2.3**: Switch default model to x1_0 (if tests pass)
  - Update pipeline_config.yaml default path
  - Update documentation
  - Document batch size requirements for x1_0 if different
  - **Files**: pipeline_config.yaml, docs
  - **Time est**: 5-10 min

### **TIER 3 - ALGORITHMIC IMPROVEMENTS** 🟠

- [ ] **TODO 3.1**: Implement Agglomerative Clustering
  - Create function `agglomerative_cluster_persons()` in osnet_clustering.py
  - Input: person_buckets, similarity_matrix
  - Output: cluster_assignments, dendrogram data
  - Use 'average' linkage method (good balance)
  - **Files**: osnet_clustering.py
  - **Time est**: 45-60 min

- [ ] **TODO 3.2**: Integrate clustering into pipeline
  - Replace connected components with agglomerative results
  - Store dendogram data for visualization
  - Update output JSON to include cluster assignments
  - **Files**: osnet_clustering.py, stage4_generate_html.py
  - **Time est**: 30-45 min

- [ ] **TODO 3.3**: Add clustering visualization to HTML
  - Display dendrogram in HTML viewer
  - Show final cluster assignments
  - Highlight intra-cluster vs inter-cluster distances
  - **Files**: stage4_generate_html.py, ondemand_crop_extraction.py
  - **Time est**: 60-90 min

- [ ] **TODO 3.4**: Add automatic cluster merging
  - Define merging criteria: "clusters with avg similarity > 0.80"
  - Create merged_persons output showing consolidated identities
  - Update canonical_persons.npz with merged IDs
  - **Files**: New stage or post-processing module
  - **Time est**: 60-90 min

### **TIER 4 - VALIDATION & DOCUMENTATION** 📚

- [ ] **TODO 4.1**: Document person filtering logic
  - Create flowchart: canonical_persons → filtered_persons → HTML
  - Explain early appearance ratio
  - Document why 10 → 8 reduction happens
  - **Files**: docs/STAGE4_PERSON_FILTERING.md
  - **Time est**: 20-30 min

- [ ] **TODO 4.2**: Document clustering algorithm choice
  - Why Agglomerative vs DBScan vs K-means
  - Expected performance on different dataset sizes
  - Hyperparameter sensitivity analysis
  - **Files**: docs/OSNET_CLUSTERING_ALGORITHM.md
  - **Time est**: 30-45 min

- [ ] **TODO 4.3**: Create clustering best practices guide
  - When to use which model (x0_25 vs x1_0)
  - Similarity threshold tuning
  - Cluster merging decisions
  - **Files**: docs/CLUSTERING_BEST_PRACTICES.md
  - **Time est**: 30-45 min

---

## 📊 Implementation Order

**Recommended sequence** (dependencies):

1. **Week 1**: Fix critical bugs
   - TODO 1.1, 1.2 → Verify data integrity
   
2. **Week 1-2**: Model improvements
   - TODO 2.1, 2.2, 2.3 → Switch to stronger model
   
3. **Week 2-3**: Clustering algorithm
   - TODO 3.1 → Core algorithm
   - TODO 3.2 → Pipeline integration
   - TODO 3.3 → Visualization
   - TODO 3.4 → Automatic merging
   
4. **Week 3**: Documentation
   - TODO 4.1, 4.2, 4.3 → Knowledge capture

---

## 🎯 Success Criteria

- [ ] HTML displays correct filtered person count (8, not 10)
- [ ] Both OSNet models (x0_25, x1_0) are supported
- [ ] Agglomerative clustering produces stable clusters
- [ ] Automatic merging reduces canonical_persons from 8 to 3-4 unique identities
- [ ] All documentation is complete and accurate
- [ ] No regressions in existing pipeline stages

---

## 💾 Affected Files Summary

| File | Issues | TODOs |
|------|--------|-------|
| ondemand_crop_extraction.py | Metadata mismatch | 1.1, 1.2, 4.1 |
| stage4_generate_html.py | Person count display | 1.2, 2.3, 3.2, 3.3 |
| osnet_clustering.py | Model selection, algorithm | 2.1, 3.1, 3.2 |
| pipeline_config.yaml | Model path, config | 2.1, 2.3 |
| Documentation | Missing | 4.1, 4.2, 4.3 |

---

## ⏱️ Total Time Estimate

- **Tier 1 (Critical)**: ~1.5 hours
- **Tier 2 (Model)**: ~1.5 hours  
- **Tier 3 (Algorithm)**: ~4.5 hours
- **Tier 4 (Docs)**: ~2.5 hours
- **Total**: ~10 hours of work

---

**Next Action**: Start with TODO 1.1 to fix the HTML person count mismatch
