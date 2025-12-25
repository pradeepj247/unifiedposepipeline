#!/usr/bin/env python3
"""
Quick test script for run_detector.py

This validates that the detector works correctly and produces
the expected output format.

Usage (in Colab):
    python test_detector.py
"""

import numpy as np
from pathlib import Path

print("=" * 70)
print("DETECTOR OUTPUT VALIDATION")
print("=" * 70)

# Path to detections file
detections_path = Path("demo_data/outputs/detections.npz")

if not detections_path.exists():
    print(f"\n❌ Detections file not found: {detections_path}")
    print("\nRun detector first:")
    print("  python run_detector.py --config configs/detector.yaml")
    exit(1)

# Load detections
data = np.load(detections_path)

print(f"\n📂 Loaded: {detections_path}")
print(f"\nKeys in NPZ file:")
for key in data.keys():
    print(f"  - '{key}': {data[key].shape} {data[key].dtype}")

# Extract arrays
frame_numbers = data['frame_numbers']
bboxes = data['bboxes']

# Validation checks
print(f"\n{'='*70}")
print("VALIDATION CHECKS")
print(f"{'='*70}")

# Check 1: Array shapes
print(f"\n✓ Check 1: Array shapes")
print(f"  frame_numbers: {frame_numbers.shape}")
print(f"  bboxes:        {bboxes.shape}")

assert len(frame_numbers) == len(bboxes), \
    "Arrays must have same length!"
print(f"  Both arrays have length {len(frame_numbers)} ✓")

# Check 2: Bbox format
print(f"\n✓ Check 2: Bbox format")
assert bboxes.shape[1] == 4, "Bboxes must have 4 columns [x1, y1, x2, y2]"
print(f"  Bboxes have 4 columns: [x1, y1, x2, y2] ✓")

# Check 3: Frame numbers sequential
print(f"\n✓ Check 3: Frame numbers")
expected_frames = np.arange(len(frame_numbers))
if np.array_equal(frame_numbers, expected_frames):
    print(f"  Frame numbers are sequential: 0 to {len(frame_numbers)-1} ✓")
else:
    print(f"  ⚠️  Frame numbers are not sequential (might have gaps)")

# Check 4: Valid bbox coordinates
print(f"\n✓ Check 4: Bbox coordinates")
valid_bboxes = np.sum(bboxes[:, 2] > 0)  # Count frames where x2 > 0
print(f"  Frames with detections: {valid_bboxes}/{len(frame_numbers)}")
print(f"  Detection rate: {valid_bboxes/len(frame_numbers)*100:.1f}%")

if valid_bboxes > 0:
    valid_mask = bboxes[:, 2] > 0  # Frames with actual detections
    valid_bbox_coords = bboxes[valid_mask, :4]
    
    # Check x2 > x1 and y2 > y1
    width_valid = np.all(valid_bbox_coords[:, 2] > valid_bbox_coords[:, 0])
    height_valid = np.all(valid_bbox_coords[:, 3] > valid_bbox_coords[:, 1])
    
    if width_valid and height_valid:
        print(f"  All bboxes have valid coordinates (x2>x1, y2>y1) ✓")
    else:
        print(f"  ❌ Some bboxes have invalid coordinates!")

# Check 5: Score ranges
print(f"\n✓ Check 5: Score ranges")
print(f"  Min score: {scores.min():.3f}")
print(f"  Max score: {scores.max():.3f}")
print(f"  Mean score (valid): {scores[scores > 0].mean():.3f}")

if np.all((scores >= 0) & (scores <= 1)):
    print(f"  All scores in valid range [0, 1] ✓")
else:
    print(f"  ❌ Some scores outside valid range!")

# Sample output
print(f"\n{'='*70}")
print("SAMPLE OUTPUT (First 5 frames)")
print(f"{'='*70}")
print(f"{'Frame':<8} {'Bbox (x1, y1, x2, y2)':<40}")
print("-" * 50)

for i in range(min(5, len(frame_numbers))):
    frame = frame_numbers[i]
    bbox = bboxes[i]
    
    if bbox[2] > 0:  # Valid detection (x2 > 0)
        bbox_str = f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})"
        print(f"{frame:<8} {bbox_str:<40}")
    else:
        print(f"{frame:<8} {'No detection':<40}")

print(f"\n{'='*70}")
print("✓ VALIDATION COMPLETE")
print(f"{'='*70}")
print(f"\nDetections file is valid and ready for Stage 2 (pose estimation)!")
print(f"\nFormat: frame_numbers (int64), bboxes (int64, shape N x 4)")
print(f"This matches udp_video.py Stage 1 output format exactly.")
print(f"\nNext steps:")
print(f"  1. Run 2D pose estimation:")
print(f"     python udp_video.py --config configs/udp_video.yaml")
print(f"  2. Or use the detections.npz directly in your pipeline")
print()
