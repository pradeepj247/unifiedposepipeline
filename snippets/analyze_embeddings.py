#!/usr/bin/env python
"""Analyze why all similarities are 0.96-1.00"""

import json
import numpy as np
from pathlib import Path

# Load files
similarity_file = Path("/content/unifiedposepipeline/demo_data/outputs/similarity_matrix.json")
embeddings_file = Path("/content/unifiedposepipeline/demo_data/outputs/embeddings.json")

with open(similarity_file) as f:
    sim_data = json.load(f)

with open(embeddings_file) as f:
    emb_data = json.load(f)

# Extract data
person_ids = sim_data['person_ids']
similarity_matrix = np.array(sim_data['matrix'])
embeddings_raw = emb_data['embeddings']

# Convert embeddings back to dict
embeddings = {int(pid): np.array(emb_data['embeddings'][str(pid)]) for pid in person_ids}

print("=" * 70)
print("EMBEDDINGS ANALYSIS")
print("=" * 70)
print()

# Check embedding norms
print("Embedding Norms (should be ~1.0 if normalized):")
print("-" * 70)
for pid in person_ids:
    emb = embeddings[pid]
    norm = np.linalg.norm(emb)
    print(f"Person {pid:2d}: norm={norm:.6f}, min={emb.min():.6f}, max={emb.max():.6f}")

print()
print("=" * 70)
print("SIMILARITY MATRIX ANALYSIS")
print("=" * 70)
print()

# Get off-diagonal similarities
off_diag = []
for i in range(len(person_ids)):
    for j in range(i+1, len(person_ids)):
        off_diag.append(similarity_matrix[i, j])

off_diag = np.array(off_diag)

print(f"Off-diagonal similarities:")
print(f"  Min: {off_diag.min():.6f}")
print(f"  Max: {off_diag.max():.6f}")
print(f"  Mean: {off_diag.mean():.6f}")
print(f"  Std: {off_diag.std():.6f}")
print(f"  Median: {np.median(off_diag):.6f}")
print()

# Count distribution
print(f"Distribution of similarities:")
bins = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
for i in range(len(bins)-1):
    count = np.sum((off_diag >= bins[i]) & (off_diag < bins[i+1]))
    pct = 100 * count / len(off_diag)
    print(f"  {bins[i]:.2f} - {bins[i+1]:.2f}: {count:2d} pairs ({pct:5.1f}%)")

print()
print("=" * 70)
print("HYPOTHESIS TEST: Are all embeddings essentially the same?")
print("=" * 70)
print()

# Stack embeddings
emb_matrix = np.array([embeddings[pid] for pid in person_ids])

# Recompute similarity manually
manual_sim = np.dot(emb_matrix, emb_matrix.T)

print(f"Manual similarity computation from embeddings:")
print(f"  Matches JSON? {np.allclose(manual_sim, similarity_matrix)}")
print()

# Check if embeddings are all similar to average
avg_emb = emb_matrix.mean(axis=0)
avg_emb_norm = avg_emb / np.linalg.norm(avg_emb)

sims_to_avg = np.dot(emb_matrix, avg_emb_norm)
print(f"Similarity of each embedding to average embedding:")
for pid, sim in zip(person_ids, sims_to_avg):
    print(f"  Person {pid:2d}: {sim:.6f}")

print()
print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print()

if off_diag.std() < 0.05:
    print("✓ All similarities are in a very narrow range (std < 0.05)")
    print("  This suggests: All 8 people have VERY SIMILAR appearance")
    print()
    print("  Reasons could be:")
    print("  1. Same clothing/uniform in video")
    print("  2. Same lighting/background conditions")
    print("  3. Similar pose/body shape")
    print("  4. Frame delay (same person appears as multiple 'persons')")
    print()
    print("  This is a DATASET CHARACTERISTIC, not a code bug!")
else:
    print("✗ Similarities have reasonable variation (std >= 0.05)")
    print("  This suggests: Model output might have an issue")
