"""
Deep diagnostic: Trace through exactly what's being stored.

We'll instrument the osnet_clustering code to see what's in all_features_dict
"""
import sys
sys.path.insert(0, r'D:\trials\unifiedpipeline\newrepo\det_track')

import numpy as np
from pathlib import Path
import json

# Let's manually check the similarity matrix to infer what happened
sim_json_path = Path(r"D:\trials\unifiedpipeline\similarity_matrix (1).json")

with open(sim_json_path) as f:
    sim_data = json.load(f)

print("Similarity Matrix Metadata:")
print(f"  Approach: {sim_data.get('approach', 'unknown')}")
print(f"  Model type: {sim_data.get('model_type', 'unknown')}")
print(f"  Person IDs: {sim_data['person_ids']}")
print()

# The key question: if per-crop features are working, 
# the similarity matrix should be DIFFERENT from the averaged approach

# Let's compute what the similarities should look like theoretically:
# If we use per-crop approach with mean of pairwise similarities,
# we need much MORE variation than averaging

# Let me check: were these similarities computed using averaged embeddings
# or per-crop features?

# CLUE: If std dev is still ~0.019 (same as before), 
# then we're STILL using averaged embeddings somehow!

# Let's verify by checking if the approach field changed
print(f"Is approach = 'per-crop'? {('per-crop' in sim_data.get('approach', '').lower())}")
print()
print(f"Similarity stats from JSON:")
m = np.array(sim_data['matrix'])
off_diag = [m[i,j] for i in range(len(m)) for j in range(i+1, len(m))]
print(f"  Std: {np.std(off_diag):.6f}")
print()

# CONCLUSION HYPOTHESES:
print("=" * 70)
print("HYPOTHESIS CHECK:")
print("=" * 70)
print()

if 'per-crop' in sim_data.get('approach', '').lower():
    print("JSON says: Using PER-CROP features")
    print()
    print("But the similarities look the same (std ~0.019)?")
    print("→ Either:")
    print("  1. Per-crop computation is mathematically equivalent to averaging")
    print("     (unlikely - pairwise should have MORE variation)")
    print("  2. The features themselves are all very similar")
    print("     (all 8 people genuinely look alike)")
    print("  3. The all_features_dict is NOT actually (16, 256) per person")
    print("     (maybe it's still (256,) - averaged)")
    print()
else:
    print("JSON says: NOT using per-crop (old approach)")
    print("→ Code changes may not have taken effect")
    print()

print("=" * 70)
print("ACTION: Need to verify extract_osnet_features output shape")
print("=" * 70)
