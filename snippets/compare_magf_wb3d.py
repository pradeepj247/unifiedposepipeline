"""
Compare MAGF vs WB3D 3D Keypoints - Frame 0 Analysis

This snippet compares the first frame of:
- keypoints_3D_magf.npz (MotionAGFormer output, H36M format, 17 joints)
- keypoints_3D_wb.npz (WholeBody3D output, COCO-WholeBody format, 133 joints)

Shows scale and magnitude differences between the two methods.

Usage in Colab:
    python compare_magf_wb3d.py
"""

import numpy as np
from pathlib import Path

# Load both files
print("Loading NPZ files...")
magf_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_3D_magf.npz')
wb3d_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_3D_wb.npz')

print(f"MAGF keys: {list(magf_data.keys())}")
print(f"WB3D keys: {list(wb3d_data.keys())}")

# Get first frame (MAGF uses 'poses_3d', WB3D uses 'keypoints_3d')
magf_frame0 = magf_data['poses_3d'][0]  # H36M format (17, 3)
wb3d_frame0 = wb3d_data['keypoints_3d'][0]  # COCO-WholeBody (133, 3)

print("=" * 80)
print("NPZ FILE COMPARISON - Frame 0")
print("=" * 80)
print(f"\nMAGF (MotionAGFormer):")
print(f"  Shape: {magf_frame0.shape}")
print(f"  Format: H36M (17 joints)")

print(f"\nWB3D (WholeBody3D):")
print(f"  Shape: {wb3d_frame0.shape}")
print(f"  Format: COCO-WholeBody (133 joints: 17 body + 6 feet + 42 hands + 68 face)")

# Joint naming conventions
h36m_names = [
    'Hip',           # 0  - Pelvis center
    'RHip',          # 1  - Right Hip
    'RKnee',         # 2  - Right Knee
    'RAnkle',        # 3  - Right Ankle
    'LHip',          # 4  - Left Hip
    'LKnee',         # 5  - Left Knee
    'LAnkle',        # 6  - Left Ankle
    'Spine',         # 7  - Spine
    'Thorax',        # 8  - Thorax/Chest
    'Neck/Nose',     # 9  - Neck/Nose
    'Head',          # 10 - Head top
    'LShoulder',     # 11 - Left Shoulder
    'LElbow',        # 12 - Left Elbow
    'LWrist',        # 13 - Left Wrist
    'RShoulder',     # 14 - Right Shoulder
    'RElbow',        # 15 - Right Elbow
    'RWrist'         # 16 - Right Wrist
]

coco_wb_names = [
    'Nose',          # 0
    'LEye',          # 1
    'REye',          # 2
    'LEar',          # 3
    'REar',          # 4
    'LShoulder',     # 5
    'RShoulder',     # 6
    'LElbow',        # 7
    'RElbow',        # 8
    'LWrist',        # 9
    'RWrist',        # 10
    'LHip',          # 11
    'RHip',          # 12
    'LKnee',         # 13
    'RKnee',         # 14
    'LAnkle',        # 15
    'RAnkle',        # 16
]

# Select 6 corresponding points for comparison
# Top 2: Head region
# Mid 2: Shoulders
# Bottom 2: Ankles
comparisons = [
    # (magf_idx, wb3d_idx, description)
    (10, 0, "HEAD - Top"),           # H36M Head vs COCO Nose
    (9, 0, "HEAD - Neck"),            # H36M Neck/Nose vs COCO Nose
    (14, 6, "SHOULDER - Right"),      # RShoulder
    (11, 5, "SHOULDER - Left"),       # LShoulder
    (3, 16, "ANKLE - Right"),         # RAnkle
    (6, 15, "ANKLE - Left"),          # LAnkle
]

print("\n" + "=" * 80)
print("KEYPOINT COMPARISON TABLE - Frame 0")
print("=" * 80)

for i, (m_idx, w_idx, desc) in enumerate(comparisons):
    magf_pt = magf_frame0[m_idx]
    wb3d_pt = wb3d_frame0[w_idx]
    
    print(f"\n{'─' * 80}")
    print(f"Point {i+1}: {desc}")
    print(f"{'─' * 80}")
    print(f"  MAGF: {h36m_names[m_idx]:12s} [#{m_idx:2d}]")
    print(f"        X={magf_pt[0]:9.4f}  Y={magf_pt[1]:9.4f}  Z={magf_pt[2]:9.4f}")
    print(f"\n  WB3D: {coco_wb_names[w_idx]:12s} [#{w_idx:2d}]")
    print(f"        X={wb3d_pt[0]:9.4f}  Y={wb3d_pt[1]:9.4f}  Z={wb3d_pt[2]:9.4f}")
    
    # Calculate absolute differences
    diff = np.abs(magf_pt - wb3d_pt)
    print(f"\n  DIFF: ΔX={diff[0]:9.4f}  ΔY={diff[1]:9.4f}  ΔZ={diff[2]:9.4f}")
    
    # Calculate Euclidean distance
    euclidean = np.linalg.norm(magf_pt - wb3d_pt)
    print(f"  Euclidean Distance: {euclidean:.4f}")

# Overall statistics
print("\n" + "=" * 80)
print("SCALE & MAGNITUDE ANALYSIS")
print("=" * 80)

# Use only body joints for WB3D (first 17)
wb3d_body = wb3d_frame0[:17]

print(f"\nMAGF Statistics (17 joints):")
print(f"  Value range:   [{np.min(magf_frame0):.4f}, {np.max(magf_frame0):.4f}]")
print(f"  Range span:    {np.max(magf_frame0) - np.min(magf_frame0):.4f}")
print(f"  Mean |value|:  {np.mean(np.abs(magf_frame0)):.4f}")
print(f"  X range:       [{np.min(magf_frame0[:, 0]):.4f}, {np.max(magf_frame0[:, 0]):.4f}]")
print(f"  Y range:       [{np.min(magf_frame0[:, 1]):.4f}, {np.max(magf_frame0[:, 1]):.4f}]")
print(f"  Z range:       [{np.min(magf_frame0[:, 2]):.4f}, {np.max(magf_frame0[:, 2]):.4f}]")

print(f"\nWB3D Statistics (17 body joints only):")
print(f"  Value range:   [{np.min(wb3d_body):.4f}, {np.max(wb3d_body):.4f}]")
print(f"  Range span:    {np.max(wb3d_body) - np.min(wb3d_body):.4f}")
print(f"  Mean |value|:  {np.mean(np.abs(wb3d_body)):.4f}")
print(f"  X range:       [{np.min(wb3d_body[:, 0]):.4f}, {np.max(wb3d_body[:, 0]):.4f}]")
print(f"  Y range:       [{np.min(wb3d_body[:, 1]):.4f}, {np.max(wb3d_body[:, 1]):.4f}]")
print(f"  Z range:       [{np.min(wb3d_body[:, 2]):.4f}, {np.max(wb3d_body[:, 2]):.4f}]")

# Scale comparison
magf_range = np.max(magf_frame0) - np.min(magf_frame0)
wb3d_range = np.max(wb3d_body) - np.min(wb3d_body)
magf_mean = np.mean(np.abs(magf_frame0))
wb3d_mean = np.mean(np.abs(wb3d_body))

print(f"\n{'─' * 80}")
print("SCALE RATIO (WB3D / MAGF):")
print(f"{'─' * 80}")
print(f"  Range ratio:   {wb3d_range / magf_range:.2f}x")
print(f"  Mean ratio:    {wb3d_mean / magf_mean:.2f}x")

# Per-axis scale ratios
x_ratio = (np.max(wb3d_body[:, 0]) - np.min(wb3d_body[:, 0])) / (np.max(magf_frame0[:, 0]) - np.min(magf_frame0[:, 0]))
y_ratio = (np.max(wb3d_body[:, 1]) - np.min(wb3d_body[:, 1])) / (np.max(magf_frame0[:, 1]) - np.min(magf_frame0[:, 1]))
z_ratio = (np.max(wb3d_body[:, 2]) - np.min(wb3d_body[:, 2])) / (np.max(magf_frame0[:, 2]) - np.min(magf_frame0[:, 2]))

print(f"\nPer-Axis Scale Ratios:")
print(f"  X-axis:        {x_ratio:.2f}x")
print(f"  Y-axis:        {y_ratio:.2f}x")
print(f"  Z-axis:        {z_ratio:.2f}x")

print("\n" + "=" * 80)
print("KEY OBSERVATIONS:")
print("=" * 80)
print("""
1. Both methods represent the SAME skeleton but at different scales
2. MAGF uses normalized coordinates (typical range: -0.5 to +0.5)
3. WB3D uses larger absolute values (typical range: -500 to +500)
4. To compare visually, normalize both to the same scale
5. Joint correspondence differs: H36M (17) vs COCO-WholeBody (17 body + extras)
""")

print("✅ Analysis complete!\n")
