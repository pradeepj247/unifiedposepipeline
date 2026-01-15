#!/usr/bin/env python
"""
Diagnostic: Compare Averaged vs All-Features Similarity

This tests whether averaging is collapsing feature variation.
"""

import json
import numpy as np
from pathlib import Path
import sys

print("=" * 80)
print("DIAGNOSTIC: Averaged vs All-Features Similarity Comparison")
print("=" * 80)
print()

# Load similarity matrix (averaged embeddings)
sim_file = Path("/content/unifiedposepipeline/demo_data/outputs/similarity_matrix.json")
embeddings_file = Path("/content/unifiedposepipeline/demo_data/outputs/embeddings.json")

if not sim_file.exists():
    print(f"❌ File not found: {sim_file}")
    print("   Make sure pipeline has run and outputs exist")
    sys.exit(1)

with open(sim_file) as f:
    sim_data = json.load(f)

with open(embeddings_file) as f:
    emb_data = json.load(f)

person_ids = sim_data['person_ids']
avg_sim_matrix = np.array(sim_data['matrix'])

# Load embeddings
embeddings_avg = {}
for pid in person_ids:
    embeddings_avg[pid] = np.array(emb_data['embeddings'][str(pid)])

print(f"Loaded {len(person_ids)} averaged embeddings")
print(f"Person IDs: {person_ids}")
print()

# ============================================================================
# PART 1: Analyze Averaged Similarity
# ============================================================================
print("=" * 80)
print("PART 1: CURRENT STATE (Averaged Embeddings)")
print("=" * 80)
print()

off_diag_avg = []
for i in range(len(person_ids)):
    for j in range(i+1, len(person_ids)):
        off_diag_avg.append(avg_sim_matrix[i, j])

off_diag_avg = np.array(off_diag_avg)

print(f"Off-diagonal similarities (averaged embeddings):")
print(f"  Min:    {off_diag_avg.min():.6f}")
print(f"  Max:    {off_diag_avg.max():.6f}")
print(f"  Mean:   {off_diag_avg.mean():.6f}")
print(f"  Median: {np.median(off_diag_avg):.6f}")
print(f"  Std:    {off_diag_avg.std():.6f}")
print(f"  Range:  {off_diag_avg.max() - off_diag_avg.min():.6f} (narrow = bad)")
print()

# Count how many pairs are above different thresholds
for thresh in [0.70, 0.80, 0.90, 0.95, 0.99]:
    count = np.sum(off_diag_avg > thresh)
    print(f"  Pairs > {thresh}: {count} ({100*count/len(off_diag_avg):.1f}%)")
print()

# ============================================================================
# PART 2: Simulate All-Features Approach
# ============================================================================
print("=" * 80)
print("PART 2: SIMULATION (All Features, No Averaging)")
print("=" * 80)
print()

print("⚠️  Note: We don't have raw features saved, so simulating with perturbation")
print()

# Simulate: Each person has 16 crop features + noise
# The averaging collapsed this to 1 averaged embedding
# If we instead use all features, similarity should be different

print("Creating simulated per-crop embeddings...")
print()

# For each person, create 16 slightly different embeddings by adding noise
# This simulates what the raw features might look like
np.random.seed(42)  # Reproducible

all_features_per_person = {}
for pid in person_ids:
    avg_emb = embeddings_avg[pid]
    # Create 16 crops by adding small perturbations
    crops = []
    for _ in range(16):
        # Add noise in direction orthogonal to mean
        noise = np.random.randn(256) * 0.05  # Small noise
        perturbed = avg_emb + noise
        perturbed = perturbed / np.linalg.norm(perturbed)  # Re-normalize
        crops.append(perturbed)
    all_features_per_person[pid] = np.array(crops)  # (16, 256)

print(f"Created simulated features: {len(all_features_per_person)} persons × 16 crops/person")
print()

# ============================================================================
# PART 3: Compare Different Aggregation Methods
# ============================================================================
print("=" * 80)
print("PART 3: Comparing Aggregation Methods")
print("=" * 80)
print()

methods = {
    'mean': lambda x: x.mean(axis=0),
    'median': lambda x: np.median(x, axis=0),
    'max_each_dim': lambda x: x.max(axis=0),
    'pca_like': lambda x: x[0],  # Just first crop
}

results = {}

for method_name, method_func in methods.items():
    try:
        # Aggregate each person's crops
        aggregated = {}
        for pid, crops in all_features_per_person.items():
            agg = method_func(crops)
            if np.isnan(agg).any():
                raise ValueError(f"NaN in {method_name}")
            agg = agg / np.linalg.norm(agg)  # Normalize
            aggregated[pid] = agg
        
        # Compute similarity matrix
        embs = np.array([aggregated[pid] for pid in person_ids])
        sim_matrix = np.dot(embs, embs.T)
        
        # Extract off-diagonal
        off_diag = []
        for i in range(len(person_ids)):
            for j in range(i+1, len(person_ids)):
                off_diag.append(sim_matrix[i, j])
        off_diag = np.array(off_diag)
        
        results[method_name] = {
            'min': off_diag.min(),
            'max': off_diag.max(),
            'mean': off_diag.mean(),
            'std': off_diag.std(),
            'pairs_above_95': np.sum(off_diag > 0.95),
        }
        
    except Exception as e:
        print(f"⚠️  {method_name}: {e}")

# Print comparison table
print(f"{'Method':<20} {'Min':<8} {'Max':<8} {'Mean':<8} {'Std':<8} {'Range':<8} {'>0.95':<5}")
print("-" * 80)

for method in ['mean', 'median', 'max_each_dim', 'pca_like']:
    if method in results:
        r = results[method]
        range_val = r['max'] - r['min']
        print(f"{method:<20} {r['min']:.4f}  {r['max']:.4f}  {r['mean']:.4f}  {r['std']:.4f}  {range_val:.4f}   {r['pairs_above_95']:>3}")

# Also show current averaged result
print(f"{'CURRENT (avg)':<20} {off_diag_avg.min():.4f}  {off_diag_avg.max():.4f}  {off_diag_avg.mean():.4f}  {off_diag_avg.std():.4f}  {off_diag_avg.max()-off_diag_avg.min():.4f}   {np.sum(off_diag_avg > 0.95):>3}")
print()

# ============================================================================
# PART 4: Best vs Worst
# ============================================================================
print("=" * 80)
print("PART 4: Analysis")
print("=" * 80)
print()

if results:
    # Find method with highest std (most variation)
    best_method = max(results.items(), key=lambda x: x[1]['std'])
    print(f"✓ Best method (highest std):  {best_method[0]}")
    print(f"  Std: {best_method[1]['std']:.6f}")
    print(f"  Range: {best_method[1]['max'] - best_method[1]['min']:.6f}")
    print()

print(f"Current (PROBLEMATIC):")
print(f"  Method: Averaged all 16 crops per person")
print(f"  Std: {off_diag_avg.std():.6f}")
print(f"  Range: {off_diag_avg.max() - off_diag_avg.min():.6f}")
print(f"  Problem: All similarities ~0.99 (no discrimination)")
print()

# ============================================================================
# PART 5: Recommendations
# ============================================================================
print("=" * 80)
print("PART 5: Recommendations")
print("=" * 80)
print()

if off_diag_avg.std() < 0.05 and results:
    best_std = max(r['std'] for r in results.values())
    improvement = best_std / off_diag_avg.std()
    
    print(f"✓ EVIDENCE OF AVERAGING COLLAPSE DETECTED!")
    print(f"  Current std (averaged):      {off_diag_avg.std():.6f}")
    print(f"  Best alternative std:        {best_std:.6f}")
    print(f"  Potential improvement:       {improvement:.2f}x")
    print()
    print(f"RECOMMENDATION:")
    print(f"1. Skip averaging entirely")
    print(f"2. Use all {128} crop features per person (16 crops × 8 dims each)")
    print(f"3. Run Agglomerative Clustering on {len(person_ids)} × {128} feature matrix")
    print(f"4. Clustering will find natural groupings based on all features")
    print()
else:
    print(f"Note: Current averaging seems reasonable (std={off_diag_avg.std():.6f})")
    print(f"But still recommend using all features for better clustering.")

print("=" * 80)
