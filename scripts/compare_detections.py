#!/usr/bin/env python3
"""
Compare two detection NPZ files to verify format compatibility

Usage:
    python compare_detections.py
"""

import numpy as np
from pathlib import Path

print("=" * 80)
print("DETECTION FILE COMPARISON")
print("=" * 80)

# Paths
old_file = Path("demo_data/outputs/detections_old.npz")
new_file = Path("demo_data/outputs/detections.npz")

# Check files exist
if not old_file.exists():
    print(f"\n❌ File not found: {old_file}")
    exit(1)

if not new_file.exists():
    print(f"\n❌ File not found: {new_file}")
    exit(1)

# Load files
print(f"\n📂 Loading files...")
old_data = np.load(old_file)
new_data = np.load(new_file)

print(f"  ✓ {old_file.name}")
print(f"  ✓ {new_file.name}")

# Compare keys
print(f"\n{'='*80}")
print("KEY COMPARISON")
print(f"{'='*80}")

old_keys = set(old_data.keys())
new_keys = set(new_data.keys())

print(f"\nOld file keys: {sorted(old_keys)}")
print(f"New file keys: {sorted(new_keys)}")

if old_keys == new_keys:
    print(f"\n✓ Keys match perfectly!")
else:
    missing_in_new = old_keys - new_keys
    extra_in_new = new_keys - old_keys
    
    if missing_in_new:
        print(f"\n⚠️  Missing in new file: {missing_in_new}")
    if extra_in_new:
        print(f"\n⚠️  Extra in new file: {extra_in_new}")

# Compare shapes and dtypes
print(f"\n{'='*80}")
print("SHAPE AND DTYPE COMPARISON")
print(f"{'='*80}")

for key in sorted(old_keys & new_keys):
    old_shape = old_data[key].shape
    new_shape = new_data[key].shape
    old_dtype = old_data[key].dtype
    new_dtype = new_data[key].dtype
    
    shape_match = "✓" if old_shape == new_shape else "✗"
    dtype_match = "✓" if old_dtype == new_dtype else "✗"
    
    print(f"\nKey: '{key}'")
    print(f"  Shape: {old_shape} → {new_shape} {shape_match}")
    print(f"  Dtype: {old_dtype} → {new_dtype} {dtype_match}")

# Detailed comparison for common keys
print(f"\n{'='*80}")
print("VALUE COMPARISON (Frame 0)")
print(f"{'='*80}")

if 'bboxes' in old_keys and 'bboxes' in new_keys:
    print(f"\nBboxes (frame 0):")
    print(f"  Old: {old_data['bboxes'][0]}")
    print(f"  New: {new_data['bboxes'][0]}")
    
    if len(old_data['bboxes']) > 0 and len(new_data['bboxes']) > 0:
        diff = np.abs(old_data['bboxes'][0] - new_data['bboxes'][0])
        print(f"  Absolute difference: {diff}")
        print(f"  Max difference: {np.max(diff):.4f} pixels")

if 'scores' in old_keys and 'scores' in new_keys:
    print(f"\nScores (frame 0):")
    print(f"  Old: {old_data['scores'][0]:.4f}")
    print(f"  New: {new_data['scores'][0]:.4f}")
    print(f"  Difference: {abs(old_data['scores'][0] - new_data['scores'][0]):.4f}")

# Statistics
print(f"\n{'='*80}")
print("STATISTICS")
print(f"{'='*80}")

if 'scores' in old_keys and 'scores' in new_keys:
    old_valid = np.sum(old_data['scores'] > 0)
    new_valid = np.sum(new_data['scores'] > 0)
    
    print(f"\nDetection counts:")
    print(f"  Old file: {old_valid}/{len(old_data['scores'])} frames "
          f"({old_valid/len(old_data['scores'])*100:.1f}%)")
    print(f"  New file: {new_valid}/{len(new_data['scores'])} frames "
          f"({new_valid/len(new_data['scores'])*100:.1f}%)")
    
    if len(old_data['scores']) == len(new_data['scores']):
        agreement = np.sum((old_data['scores'] > 0) == (new_data['scores'] > 0))
        print(f"  Agreement: {agreement}/{len(old_data['scores'])} frames "
              f"({agreement/len(old_data['scores'])*100:.1f}%)")

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

format_compatible = (old_keys == new_keys and 
                    all(old_data[k].shape == new_data[k].shape for k in old_keys & new_keys) and
                    all(old_data[k].dtype == new_data[k].dtype for k in old_keys & new_keys))

if format_compatible:
    print(f"\n✅ FORMAT COMPATIBLE!")
    print(f"Both files have identical structure and can be used interchangeably.")
    print(f"\nThe new run_detector.py produces the same output format as udp_video.py Stage 1.")
else:
    print(f"\n⚠️  FORMAT DIFFERENCES DETECTED!")
    print(f"Files may not be fully compatible. Review differences above.")

print(f"\n{'='*80}\n")
