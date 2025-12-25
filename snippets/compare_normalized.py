"""
Compare MAGF vs WB3D after normalizing WB3D by dividing by 1000

This script:
1. Loads both NPZ files
2. Divides WB3D values by 1000 to match MAGF scale
3. Compares corresponding joints side-by-side
4. Shows if they represent the same skeleton structure

Usage in Colab:
    python compare_normalized.py
"""

import numpy as np

# Load both files
print("Loading NPZ files...")
magf_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_3D_magf.npz')
wb3d_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_3D_wb.npz')

# Get first frame
magf_frame0 = magf_data['poses_3d'][0]  # H36M format (17, 3)
wb3d_frame0 = wb3d_data['keypoints_3d'][0]  # COCO-WholeBody (133, 3)

# Extract body joints only from WB3D (first 17)
wb3d_body = wb3d_frame0[:17].copy()

# Normalize WB3D by dividing by 1000
wb3d_normalized = wb3d_body / 1000.0

print("=" * 80)
print("NORMALIZED COMPARISON - Frame 0")
print("=" * 80)
print(f"\nMAGF Shape: {magf_frame0.shape} (H36M-17 format)")
print(f"WB3D Shape: {wb3d_body.shape} (COCO-17 body joints)")
print(f"\nWB3D normalized by dividing by 1000")

# Joint name mappings
h36m_names = [
    'Hip',           # 0  - Pelvis center
    'RHip',          # 1  
    'RKnee',         # 2  
    'RAnkle',        # 3  
    'LHip',          # 4  
    'LKnee',         # 5  
    'LAnkle',        # 6  
    'Spine',         # 7  
    'Thorax',        # 8  
    'Neck/Nose',     # 9  
    'Head',          # 10 
    'LShoulder',     # 11 
    'LElbow',        # 12 
    'LWrist',        # 13 
    'RShoulder',     # 14 
    'RElbow',        # 15 
    'RWrist'         # 16 
]

coco_names = [
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

# Define corresponding joints between H36M and COCO
# (magf_idx, wb3d_idx, description)
correspondences = [
    (10, 0,  "HEAD"),           # H36M Head <-> COCO Nose
    (9,  0,  "NECK/NOSE"),      # H36M Neck/Nose <-> COCO Nose
    (14, 6,  "RShoulder"),      # Right Shoulder
    (11, 5,  "LShoulder"),      # Left Shoulder
    (15, 8,  "RElbow"),         # Right Elbow
    (12, 7,  "LElbow"),         # Left Elbow
    (16, 10, "RWrist"),         # Right Wrist
    (13, 9,  "LWrist"),         # Left Wrist
    (1,  12, "RHip"),           # Right Hip
    (4,  11, "LHip"),           # Left Hip
    (2,  14, "RKnee"),          # Right Knee
    (5,  13, "LKnee"),          # Left Knee
    (3,  16, "RAnkle"),         # Right Ankle
    (6,  15, "LAnkle"),         # Left Ankle
]

print("\n" + "=" * 80)
print("JOINT-BY-JOINT COMPARISON (After Normalization)")
print("=" * 80)

for i, (m_idx, w_idx, desc) in enumerate(correspondences):
    magf_pt = magf_frame0[m_idx]
    wb3d_pt = wb3d_normalized[w_idx]
    
    print(f"\n{'─' * 80}")
    print(f"{i+1:2d}. {desc}")
    print(f"{'─' * 80}")
    print(f"  MAGF: {h36m_names[m_idx]:15s} [#{m_idx:2d}]   X={magf_pt[0]:9.4f}  Y={magf_pt[1]:9.4f}  Z={magf_pt[2]:9.4f}")
    print(f"  WB3D: {coco_names[w_idx]:15s} [#{w_idx:2d}]   X={wb3d_pt[0]:9.4f}  Y={wb3d_pt[1]:9.4f}  Z={wb3d_pt[2]:9.4f}")
    
    # Calculate differences
    diff = magf_pt - wb3d_pt
    abs_diff = np.abs(diff)
    euclidean = np.linalg.norm(diff)
    
    print(f"  DIFF:                         ΔX={diff[0]:+9.4f}  ΔY={diff[1]:+9.4f}  ΔZ={diff[2]:+9.4f}")
    print(f"  ABS:                          |ΔX|={abs_diff[0]:8.4f} |ΔY|={abs_diff[1]:8.4f} |ΔZ|={abs_diff[2]:8.4f}")
    print(f"  Euclidean Distance: {euclidean:.4f}")

# Statistics after normalization
print("\n" + "=" * 80)
print("STATISTICAL COMPARISON (After WB3D / 1000)")
print("=" * 80)

print(f"\nMAGF Statistics:")
print(f"  X: min={np.min(magf_frame0[:, 0]):8.4f}, max={np.max(magf_frame0[:, 0]):8.4f}, mean={np.mean(magf_frame0[:, 0]):8.4f}")
print(f"  Y: min={np.min(magf_frame0[:, 1]):8.4f}, max={np.max(magf_frame0[:, 1]):8.4f}, mean={np.mean(magf_frame0[:, 1]):8.4f}")
print(f"  Z: min={np.min(magf_frame0[:, 2]):8.4f}, max={np.max(magf_frame0[:, 2]):8.4f}, mean={np.mean(magf_frame0[:, 2]):8.4f}")

print(f"\nWB3D Statistics (normalized):")
print(f"  X: min={np.min(wb3d_normalized[:, 0]):8.4f}, max={np.max(wb3d_normalized[:, 0]):8.4f}, mean={np.mean(wb3d_normalized[:, 0]):8.4f}")
print(f"  Y: min={np.min(wb3d_normalized[:, 1]):8.4f}, max={np.max(wb3d_normalized[:, 1]):8.4f}, mean={np.mean(wb3d_normalized[:, 1]):8.4f}")
print(f"  Z: min={np.min(wb3d_normalized[:, 2]):8.4f}, max={np.max(wb3d_normalized[:, 2]):8.4f}, mean={np.mean(wb3d_normalized[:, 2]):8.4f}")

# Calculate mean absolute error per axis
mae_x = np.mean(np.abs(magf_frame0[:14, 0] - wb3d_normalized[:14, 0]))  # First 14 joints are most comparable
mae_y = np.mean(np.abs(magf_frame0[:14, 1] - wb3d_normalized[:14, 1]))
mae_z = np.mean(np.abs(magf_frame0[:14, 2] - wb3d_normalized[:14, 2]))
mae_total = np.mean(np.abs(magf_frame0[:14] - wb3d_normalized[:14]))

print(f"\nMean Absolute Error (first 14 joints):")
print(f"  MAE X-axis: {mae_x:.4f}")
print(f"  MAE Y-axis: {mae_y:.4f}")
print(f"  MAE Z-axis: {mae_z:.4f}")
print(f"  MAE Total:  {mae_total:.4f}")

# Calculate average Euclidean distance for corresponding joints
euclidean_distances = []
for m_idx, w_idx, _ in correspondences:
    dist = np.linalg.norm(magf_frame0[m_idx] - wb3d_normalized[w_idx])
    euclidean_distances.append(dist)

avg_euclidean = np.mean(euclidean_distances)
max_euclidean = np.max(euclidean_distances)
min_euclidean = np.min(euclidean_distances)

print(f"\nEuclidean Distance Between Corresponding Joints:")
print(f"  Average: {avg_euclidean:.4f}")
print(f"  Min:     {min_euclidean:.4f}")
print(f"  Max:     {max_euclidean:.4f}")

print("\n" + "=" * 80)
print("CONCLUSIONS:")
print("=" * 80)
print(f"""
After normalizing WB3D by dividing by 1000:
- Average distance between corresponding joints: {avg_euclidean:.4f}
- Mean Absolute Error across all axes: {mae_total:.4f}

If these values are SMALL (< 0.1): The skeletons are very similar!
If these values are LARGE (> 0.5): The skeletons represent different poses/formats.

Note: Some differences are expected due to:
1. Different joint definitions (H36M vs COCO)
2. Different estimation methods (temporal model vs single-frame)
3. Different coordinate systems/centering conventions
""")

print("✅ Analysis complete!\n")
