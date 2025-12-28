#!/usr/bin/env python3
"""
Compare Two 2D Keypoint NPZ Files

This script compares two 2D keypoint files (from udp_video.py or run_2d_posedet.py)
to verify format compatibility and output consistency.

Usage:
    python compare_2d_keypoints.py

Default paths:
    Old: demo_data/outputs/kps_2d_rtm_old.npz
    New: demo_data/outputs/kps_2d_rtm_new.npz
"""

import numpy as np
from pathlib import Path

print("=" * 80)
print("2D KEYPOINTS FILE COMPARISON")
print("=" * 80)

# Paths to files
old_path = Path("demo_data/outputs/kps_2d_rtm_old.npz")
new_path = Path("demo_data/outputs/kps_2d_rtm_new.npz")

# Load files
print(f"\n📂 Loading files...")
if not old_path.exists():
    print(f"❌ Old file not found: {old_path}")
    exit(1)
if not new_path.exists():
    print(f"❌ New file not found: {new_path}")
    exit(1)

old_data = np.load(old_path)
new_data = np.load(new_path)

print(f"  ✓ {old_path}")
print(f"  ✓ {new_path}")

# Compare keys
print(f"\n{'=' * 80}")
print("KEY COMPARISON")
print(f"{'=' * 80}")

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
print(f"\n{'=' * 80}")
print("SHAPE AND DTYPE COMPARISON")
print(f"{'=' * 80}")

all_keys = sorted(old_keys.union(new_keys))
shape_match = True
dtype_match = True

for key in all_keys:
    print(f"\nKey: '{key}'")
    
    if key in old_data and key in new_data:
        old_shape = old_data[key].shape
        new_shape = new_data[key].shape
        old_dtype = old_data[key].dtype
        new_dtype = new_data[key].dtype
        
        shape_ok = old_shape == new_shape
        dtype_ok = old_dtype == new_dtype
        
        print(f"  Shape: {old_shape} → {new_shape} {'✓' if shape_ok else '✗'}")
        print(f"  Dtype: {old_dtype} → {new_dtype} {'✓' if dtype_ok else '✗'}")
        
        if not shape_ok:
            shape_match = False
        if not dtype_ok:
            dtype_match = False
    elif key in old_data:
        print(f"  ⚠️  Only in old file")
    else:
        print(f"  ⚠️  Only in new file")

# Metadata comparison
print(f"\n{'=' * 80}")
print("METADATA COMPARISON")
print(f"{'=' * 80}")

if 'joint_format' in old_data and 'joint_format' in new_data:
    old_fmt = str(old_data['joint_format'])
    new_fmt = str(new_data['joint_format'])
    print(f"\nJoint format:")
    print(f"  Old: {old_fmt}")
    print(f"  New: {new_fmt}")
    print(f"  {'✓ Match' if old_fmt == new_fmt else '✗ Different'}")

if 'model_type' in old_data and 'model_type' in new_data:
    old_model = str(old_data['model_type'])
    new_model = str(new_data['model_type'])
    print(f"\nModel type:")
    print(f"  Old: {old_model}")
    print(f"  New: {new_model}")
    print(f"  {'✓ Match' if old_model == new_model else '✗ Different'}")

# Compare first 100 frames
print(f"\n{'=' * 80}")
print("VALUE COMPARISON (First 100 Frames)")
print(f"{'=' * 80}")

# Limit to first 100 frames
num_compare = min(100, len(old_data['frame_numbers']), len(new_data['frame_numbers']))

print(f"\nComparing first {num_compare} frames...")

# Frame numbers comparison
if 'frame_numbers' in old_data and 'frame_numbers' in new_data:
    old_frames = old_data['frame_numbers'][:num_compare]
    new_frames = new_data['frame_numbers'][:num_compare]
    
    frames_match = np.array_equal(old_frames, new_frames)
    print(f"\nFrame numbers: {'✓ Identical' if frames_match else '✗ Different'}")
    if not frames_match:
        diff_count = np.sum(old_frames != new_frames)
        print(f"  Different values: {diff_count}/{num_compare}")

# Keypoints comparison
if 'keypoints' in old_data and 'keypoints' in new_data:
    old_kpts = old_data['keypoints'][:num_compare]
    new_kpts = new_data['keypoints'][:num_compare]
    
    print(f"\nKeypoints shape (first {num_compare} frames):")
    print(f"  Old: {old_kpts.shape}")
    print(f"  New: {new_kpts.shape}")
    
    if old_kpts.shape == new_kpts.shape:
        # Calculate differences
        diff = np.abs(old_kpts - new_kpts)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # Count identical keypoints
        identical = np.allclose(old_kpts, new_kpts, atol=1e-6)
        
        print(f"\nKeypoint differences:")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        print(f"  Identical: {'✓ Yes (within 1e-6 tolerance)' if identical else '✗ No'}")
        
        # Show sample from first frame with valid detection
        for i in range(num_compare):
            if old_data['scores'][i, 0] > 0:  # Valid detection
                print(f"\nSample keypoints (Frame {i}, first 3 joints):")
                print(f"  Old:")
                for j in range(min(3, old_kpts.shape[1])):
                    print(f"    Joint {j}: [{old_kpts[i, j, 0]:.2f}, {old_kpts[i, j, 1]:.2f}]")
                print(f"  New:")
                for j in range(min(3, new_kpts.shape[1])):
                    print(f"    Joint {j}: [{new_kpts[i, j, 0]:.2f}, {new_kpts[i, j, 1]:.2f}]")
                print(f"  Difference:")
                for j in range(min(3, old_kpts.shape[1])):
                    print(f"    Joint {j}: [{diff[i, j, 0]:.2f}, {diff[i, j, 1]:.2f}]")
                break
    else:
        print(f"  ⚠️  Shape mismatch - cannot compare values")

# Scores comparison
if 'scores' in old_data and 'scores' in new_data:
    old_scores = old_data['scores'][:num_compare]
    new_scores = new_data['scores'][:num_compare]
    
    print(f"\nScores shape (first {num_compare} frames):")
    print(f"  Old: {old_scores.shape}")
    print(f"  New: {new_scores.shape}")
    
    if old_scores.shape == new_scores.shape:
        # Calculate differences
        diff_scores = np.abs(old_scores - new_scores)
        max_diff_score = np.max(diff_scores)
        mean_diff_score = np.mean(diff_scores)
        
        identical_scores = np.allclose(old_scores, new_scores, atol=1e-6)
        
        print(f"\nScore differences:")
        print(f"  Max difference: {max_diff_score:.6f}")
        print(f"  Mean difference: {mean_diff_score:.6f}")
        print(f"  Identical: {'✓ Yes (within 1e-6 tolerance)' if identical_scores else '✗ No'}")

# Detection statistics
print(f"\n{'=' * 80}")
print("DETECTION STATISTICS (First 100 Frames)")
print(f"{'=' * 80}")

if 'scores' in old_data and 'scores' in new_data:
    old_valid = np.sum(old_data['scores'][:num_compare, 0] > 0)
    new_valid = np.sum(new_data['scores'][:num_compare, 0] > 0)
    
    print(f"\nValid poses detected:")
    print(f"  Old: {old_valid}/{num_compare} ({old_valid/num_compare*100:.1f}%)")
    print(f"  New: {new_valid}/{num_compare} ({new_valid/num_compare*100:.1f}%)")
    print(f"  {'✓ Match' if old_valid == new_valid else '✗ Different'}")
    
    if old_valid > 0:
        old_valid_scores = old_data['scores'][:num_compare][old_data['scores'][:num_compare, 0] > 0]
        new_valid_scores = new_data['scores'][:num_compare][new_data['scores'][:num_compare, 0] > 0]
        
        print(f"\nAverage confidence (valid poses only):")
        print(f"  Old: {np.mean(old_valid_scores):.4f}")
        print(f"  New: {np.mean(new_valid_scores):.4f}")

# Final verdict
print(f"\n{'=' * 80}")
print("COMPATIBILITY VERDICT")
print(f"{'=' * 80}")

all_good = (
    old_keys == new_keys and
    shape_match and
    dtype_match
)

if all_good:
    print("\n✅ FILES ARE FULLY COMPATIBLE!")
    print("   Format, shapes, and dtypes match perfectly.")
    if 'keypoints' in old_data and 'keypoints' in new_data:
        if np.allclose(old_data['keypoints'][:num_compare], new_data['keypoints'][:num_compare], atol=1e-6):
            print("   Keypoint values are identical (within tolerance).")
        else:
            print("   ⚠️  Keypoint values differ slightly - may be due to:")
            print("      - Different random seeds")
            print("      - Model initialization differences")
            print("      - Floating-point precision")
else:
    print("\n⚠️  FILES HAVE DIFFERENCES!")
    if old_keys != new_keys:
        print("   - Keys don't match")
    if not shape_match:
        print("   - Shapes don't match")
    if not dtype_match:
        print("   - Data types don't match")

print()
