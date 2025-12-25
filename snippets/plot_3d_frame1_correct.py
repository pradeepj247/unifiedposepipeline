#!/usr/bin/env python3
"""
Plot Frame 1 from MAGF 3D data using CORRECT skeleton logic from udp_3d_lifting_fixed.py
This will verify we can now visualize the 3D pose correctly with proper joint semantics.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# CORRECT H36M-17 Joint Semantics (from udp_3d_lifting_fixed.py)
# ============================================================================

CORRECT_JOINT_NAMES = {
    0: 'Pelvis (Root)',
    1: 'RHip',
    2: 'RKnee',
    3: 'RAnkle',
    4: 'LHip',
    5: 'LKnee',
    6: 'LAnkle',
    7: 'Spine',
    8: 'Thorax',
    9: 'Neck/Nose',
    10: 'Head',
    11: 'LShoulder',
    12: 'LElbow',
    13: 'LWrist',
    14: 'RShoulder',
    15: 'RElbow',
    16: 'RWrist'
}

# CORRECT skeleton connections from MotionAGFormer show3Dpose() function
# Extracted from udp_3d_lifting_fixed.py lines 350-450
# I = np.array([0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
# J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

CORRECT_SKELETON = [
    (0, 1),   # Pelvis -> RHip
    (0, 4),   # Pelvis -> LHip
    (1, 2),   # RHip -> RKnee
    (4, 5),   # LHip -> LKnee
    (2, 3),   # RKnee -> RAnkle
    (5, 6),   # LKnee -> LAnkle
    (0, 7),   # Pelvis -> Spine
    (7, 8),   # Spine -> Thorax
    (8, 14),  # Thorax -> RShoulder
    (8, 11),  # Thorax -> LShoulder
    (14, 15), # RShoulder -> RElbow
    (15, 16), # RElbow -> RWrist
    (11, 12), # LShoulder -> LElbow
    (12, 13), # LElbow -> LWrist
    (8, 9),   # Thorax -> Neck/Nose ← This was MISSING!
    (9, 10),  # Neck/Nose -> Head ← This was MISSING!
]

# ============================================================================
# Load MAGF 3D Data
# ============================================================================

print("Loading MAGF 3D keypoints...")
# Google Colab path
data_path = "/content/unifiedposepipeline/demo_data/outputs/keypoints_3D_magf.npz"
data = np.load(data_path)
print(f"Available keys in NPZ file: {list(data.keys())}")

# Get the keypoints - adjust key name based on what's actually in the file
if 'keypoints_3d' in data:
    keypoints_3d = data['keypoints_3d']
elif 'reconstruction' in data:
    keypoints_3d = data['reconstruction']
elif 'poses_3d' in data:
    keypoints_3d = data['poses_3d']
else:
    # Print available keys and exit
    print(f"ERROR: Unexpected key in NPZ file")
    print(f"Available keys: {list(data.keys())}")
    exit(1)

print(f"Loaded 3D keypoints: {keypoints_3d.shape}")
print(f"Frame count: {keypoints_3d.shape[0]}")
print(f"Joint count: {keypoints_3d.shape[1]}")
print(f"Coordinates: {keypoints_3d.shape[2]} (X, Y, Z)")

# Extract Frame 1 (index 0)
frame_1 = keypoints_3d[0]  # Shape: (17, 3)

print("\n" + "=" * 80)
print("FRAME 1 - 3D JOINT COORDINATES (MAGF)")
print("=" * 80)

for joint_idx in range(17):
    x, y, z = frame_1[joint_idx]
    joint_name = CORRECT_JOINT_NAMES[joint_idx]
    print(f"Joint {joint_idx:2d} ({joint_name:20s}): X={x:7.3f}, Y={y:7.3f}, Z={z:7.3f}")

# ============================================================================
# Visualize 3D Skeleton using CORRECT connections
# ============================================================================

print("\n" + "=" * 80)
print("CREATING 3D VISUALIZATION WITH CORRECT SKELETON")
print("=" * 80)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Extract X, Y, Z coordinates
xs = frame_1[:, 0]
ys = frame_1[:, 1]
zs = frame_1[:, 2]

# Plot joints as scatter points
ax.scatter(xs, ys, zs, c='red', marker='o', s=100, label='Joints')

# Plot skeleton connections
for (i, j) in CORRECT_SKELETON:
    x_vals = [frame_1[i, 0], frame_1[j, 0]]
    y_vals = [frame_1[i, 1], frame_1[j, 1]]
    z_vals = [frame_1[i, 2], frame_1[j, 2]]
    ax.plot(x_vals, y_vals, z_vals, 'b-', linewidth=2)

# Label each joint with its number and name
for joint_idx in range(17):
    x, y, z = frame_1[joint_idx]
    label = f"{joint_idx}\n{CORRECT_JOINT_NAMES[joint_idx]}"
    ax.text(x, y, z, label, fontsize=8, color='darkblue', weight='bold')

# Set axis labels (MAGF uses body-relative coordinates)
ax.set_xlabel('X (Body-relative)', fontsize=12)
ax.set_ylabel('Y (Body-relative)', fontsize=12)
ax.set_zlabel('Z (Height)', fontsize=12)

# Set title
ax.set_title('Frame 1 - MAGF 3D Skeleton (CORRECTED from udp_3d_lifting_fixed.py)', 
             fontsize=14, weight='bold')

# Set viewing angle for better visualization
ax.view_init(elev=20, azim=45)

# Make axes equal aspect ratio
max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
mid_x = (xs.max()+xs.min()) * 0.5
mid_y = (ys.max()+ys.min()) * 0.5
mid_z = (zs.max()+zs.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Add legend
ax.legend()

# Add grid
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save the figure - Google Colab output path
output_path = "/content/unifiedposepipeline/demo_data/outputs/frame1_3d_skeleton_correct.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Saved 3D visualization: {output_path}")

plt.show()

print("\n" + "=" * 80)
print("VERIFICATION CHECKLIST")
print("=" * 80)
print("""
Please verify the following in the visualization:

✓ Pelvis (0) should be at the center/root
✓ Right leg: Pelvis → RHip(1) → RKnee(2) → RAnkle(3)
✓ Left leg: Pelvis → LHip(4) → LKnee(5) → LAnkle(6)
✓ Spine: Pelvis(0) → Spine(7) → Thorax(8)
✓ Right arm: Thorax(8) → RShoulder(14) → RElbow(15) → RWrist(16)
✓ Left arm: Thorax(8) → LShoulder(11) → LElbow(12) → LWrist(13)
✓ Head: Thorax(8) → Neck/Nose(9) → Head(10)  ← KEY: Was this missing before?

If the skeleton looks anatomically correct with all connections present,
then we've successfully corrected the joint semantic mapping!
""")

print("\n✅ Visualization complete!\n")
