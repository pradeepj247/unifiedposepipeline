"""
Verify if wb3d 2D keypoints are just the X,Y from 3D keypoints.
"""

import numpy as np
import os

output_dir = '/content/unifiedposepipeline/demo_data/outputs'

# Load wb3d 2D keypoints
wb_2d_file = os.path.join(output_dir, 'keypoints_2D_wb.npz')
wb_2d_data = np.load(wb_2d_file)
kpts_2d = wb_2d_data['keypoints']  # (360, 133, 2)

# Load wb3d 3D keypoints
wb_3d_file = os.path.join(output_dir, 'keypoints_3D_wb.npz')
wb_3d_data = np.load(wb_3d_file)
kpts_3d = wb_3d_data['keypoints_3d']  # (360, 133, 3)

print("="*70)
print("WB3D: Verifying 2D vs 3D Relationship")
print("="*70)

print(f"\n2D keypoints shape: {kpts_2d.shape}")
print(f"3D keypoints shape: {kpts_3d.shape}")

# Extract X,Y from 3D
xy_from_3d = kpts_3d[:, :, :2]  # Take first 2 dimensions (X, Y)

print(f"\nX,Y extracted from 3D: {xy_from_3d.shape}")

# Compare
difference = kpts_2d - xy_from_3d
abs_diff = np.abs(difference)

print("\n--- Comparison Results ---")
print(f"Mean absolute difference: {np.mean(abs_diff):.10f}")
print(f"Max absolute difference: {np.max(abs_diff):.10f}")
print(f"Min absolute difference: {np.min(abs_diff):.10f}")
print(f"Std absolute difference: {np.std(abs_diff):.10f}")

# Check if they're essentially identical
if np.allclose(kpts_2d, xy_from_3d, atol=1e-6):
    print("\n✓ CONFIRMED: 2D keypoints ARE just X,Y from 3D (within numerical precision)")
else:
    print("\n✗ NOT IDENTICAL: 2D keypoints are computed differently!")
    
    # Sample some differences to see the pattern
    print("\n--- Sample Differences (first 5 frames, first 10 keypoints) ---")
    for frame_idx in range(min(5, kpts_2d.shape[0])):
        print(f"\nFrame {frame_idx}:")
        for kpt_idx in range(min(10, kpts_2d.shape[1])):
            diff_x = difference[frame_idx, kpt_idx, 0]
            diff_y = difference[frame_idx, kpt_idx, 1]
            if abs(diff_x) > 0.01 or abs(diff_y) > 0.01:
                print(f"  Keypoint {kpt_idx}: ΔX={diff_x:.4f}, ΔY={diff_y:.4f}")

print("\n--- Z Values (depth) from 3D keypoints ---")
z_values = kpts_3d[:, :, 2]
print(f"Z range: [{np.min(z_values):.4f}, {np.max(z_values):.4f}]")
print(f"Z mean: {np.mean(z_values):.4f}")
print(f"Z std: {np.std(z_values):.4f}")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)
