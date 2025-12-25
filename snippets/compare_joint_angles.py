"""
Joint Angle Comparison - Direct Pose Configuration Analysis

Compare MAGF vs WB3D by measuring joint angles (e.g., elbow bend, knee bend).
This is the most direct way to compare skeletal poses because:
- Completely independent of scale, translation, rotation
- Anatomically meaningful (e.g., "elbow is bent 90 degrees")
- Shows if methods agree on pose configuration

Usage in Colab:
    python compare_joint_angles.py
"""

import numpy as np

def angle_between_vectors(v1, v2):
    """
    Calculate angle between two vectors in degrees
    
    Args:
        v1, v2: 3D vectors (np.array of shape (3,))
    
    Returns:
        angle in degrees [0, 180]
    """
    # Normalize vectors
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    
    # Calculate angle using dot product
    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def calculate_joint_angle(j1, j2, j3):
    """
    Calculate angle at joint j2 formed by j1-j2-j3
    
    Args:
        j1, j2, j3: Joint positions (3D coordinates)
    
    Returns:
        angle in degrees
    """
    # Vectors from j2 to j1 and j2 to j3
    v1 = j1 - j2
    v2 = j3 - j2
    
    return angle_between_vectors(v1, v2)


def calculate_all_joint_angles(joints_3d):
    """
    Calculate all relevant joint angles for H36M-17 skeleton
    
    Args:
        joints_3d: (17, 3) array in H36M format
    
    Returns:
        dict of angle_name: angle_degrees
    """
    angles = {}
    
    # Elbow angles (upper_arm to forearm)
    angles['LElbow'] = calculate_joint_angle(
        joints_3d[11],  # LShoulder
        joints_3d[12],  # LElbow
        joints_3d[13]   # LWrist
    )
    
    angles['RElbow'] = calculate_joint_angle(
        joints_3d[14],  # RShoulder
        joints_3d[15],  # RElbow
        joints_3d[16]   # RWrist
    )
    
    # Knee angles (thigh to shin)
    angles['LKnee'] = calculate_joint_angle(
        joints_3d[4],   # LHip
        joints_3d[5],   # LKnee
        joints_3d[6]    # LAnkle
    )
    
    angles['RKnee'] = calculate_joint_angle(
        joints_3d[1],   # RHip
        joints_3d[2],   # RKnee
        joints_3d[3]    # RAnkle
    )
    
    # Shoulder angles (how arm extends from body)
    # Using Hip-Shoulder-Elbow angle as proxy for arm position
    angles['LShoulder'] = calculate_joint_angle(
        joints_3d[4],   # LHip
        joints_3d[11],  # LShoulder
        joints_3d[12]   # LElbow
    )
    
    angles['RShoulder'] = calculate_joint_angle(
        joints_3d[1],   # RHip
        joints_3d[14],  # RShoulder
        joints_3d[15]   # RElbow
    )
    
    # Hip angles (torso to thigh)
    angles['LHip'] = calculate_joint_angle(
        joints_3d[11],  # LShoulder
        joints_3d[4],   # LHip
        joints_3d[5]    # LKnee
    )
    
    angles['RHip'] = calculate_joint_angle(
        joints_3d[14],  # RShoulder
        joints_3d[1],   # RHip
        joints_3d[2]    # RKnee
    )
    
    # Torso angles
    # Shoulder bridge angle (shoulder-to-shoulder alignment)
    angles['ShoulderBridge'] = calculate_joint_angle(
        joints_3d[11],  # LShoulder
        joints_3d[0],   # Hip (center)
        joints_3d[14]   # RShoulder
    )
    
    return angles


# ============================================================================
# Load data
# ============================================================================

print("=" * 80)
print("JOINT ANGLE COMPARISON")
print("=" * 80)

# Load both 3D poses
magf_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_3D_magf.npz')
wb3d_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_3D_wb.npz')

magf_poses = magf_data['poses_3d']  # (120, 17, 3) H36M
wb3d_poses_full = wb3d_data['keypoints_3d']  # (360, 133, 3) COCO-WholeBody

print(f"\n✅ MAGF: {magf_poses.shape} (H36M-17)")
print(f"✅ WB3D: {wb3d_poses_full.shape} (COCO-WholeBody-133)")

# Map WB3D COCO-17 to H36M-17 (using verified mapping)
coco_to_h36m_map = {
    0: 9,   # Nose
    5: 11,  # LShoulder
    6: 14,  # RShoulder
    7: 12,  # LElbow
    8: 15,  # RElbow
    9: 13,  # LWrist
    10: 16, # RWrist
    11: 4,  # LHip
    12: 1,  # RHip
    13: 5,  # LKnee
    14: 2,  # RKnee
    15: 6,  # LAnkle
    16: 3,  # RAnkle
}

# Extract WB3D body joints and remap to H36M order
wb3d_body = wb3d_poses_full[:, :17, :]  # (360, 17, 3)

n_frames = min(magf_poses.shape[0], wb3d_body.shape[0])
wb3d_h36m = np.zeros((n_frames, 17, 3), dtype=np.float32)

for t in range(n_frames):
    for coco_idx, h36m_idx in coco_to_h36m_map.items():
        wb3d_h36m[t, h36m_idx] = wb3d_body[t, coco_idx]
    
    # Compute Hip (H36M 0) as average of LHip and RHip
    wb3d_h36m[t, 0] = (wb3d_body[t, 11] + wb3d_body[t, 12]) / 2

print(f"\n✅ Comparing {n_frames} frames")

# ============================================================================
# Frame 0 comparison
# ============================================================================

print("\n" + "=" * 80)
print("FRAME 0: JOINT ANGLE COMPARISON")
print("=" * 80)

magf_frame0 = magf_poses[0]
wb3d_frame0 = wb3d_h36m[0]

magf_angles = calculate_all_joint_angles(magf_frame0)
wb3d_angles = calculate_all_joint_angles(wb3d_frame0)

print(f"\n{'Joint':<20s} {'MAGF (°)':>12s} {'WB3D (°)':>12s} {'Difference':>12s}")
print("-" * 80)

angle_names = [
    'LElbow',
    'RElbow',
    'LKnee',
    'RKnee',
    'LShoulder',
    'RShoulder',
    'LHip',
    'RHip',
    'ShoulderBridge',
]

frame0_diffs = []
for angle_name in angle_names:
    magf_angle = magf_angles[angle_name]
    wb3d_angle = wb3d_angles[angle_name]
    diff = abs(magf_angle - wb3d_angle)
    frame0_diffs.append(diff)
    
    print(f"{angle_name:<20s} {magf_angle:>12.2f} {wb3d_angle:>12.2f} {diff:>12.2f}")

# ============================================================================
# Multi-frame analysis
# ============================================================================

print("\n" + "=" * 80)
print(f"MULTI-FRAME ANALYSIS (first {min(n_frames, 30)} frames)")
print("=" * 80)

frame_angle_diffs = {angle: [] for angle in angle_names}

for t in range(min(n_frames, 30)):
    magf_angles_t = calculate_all_joint_angles(magf_poses[t])
    wb3d_angles_t = calculate_all_joint_angles(wb3d_h36m[t])
    
    for angle_name in angle_names:
        diff = abs(magf_angles_t[angle_name] - wb3d_angles_t[angle_name])
        frame_angle_diffs[angle_name].append(diff)

print(f"\n{'Joint':<20s} {'Mean Diff (°)':>15s} {'Std Dev (°)':>15s} {'Max Diff (°)':>15s}")
print("-" * 80)

mean_angle_diffs = []
for angle_name in angle_names:
    diffs = frame_angle_diffs[angle_name]
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    max_diff = np.max(diffs)
    mean_angle_diffs.append(mean_diff)
    
    print(f"{angle_name:<20s} {mean_diff:>15.2f} {std_diff:>15.2f} {max_diff:>15.2f}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

overall_mean_diff_f0 = np.mean(frame0_diffs)
overall_mean_diff = np.mean(mean_angle_diffs)

print(f"\nJoint Angle Differences:")
print(f"  Frame 0 - Mean difference:     {overall_mean_diff_f0:.2f}°")
print(f"  30 frames - Mean difference:   {overall_mean_diff:.2f}°")

print(f"\nMost similar joints (smallest angle difference):")
sorted_indices = np.argsort(mean_angle_diffs)
for i in range(3):
    idx = sorted_indices[i]
    print(f"  {angle_names[idx]:<20s} Mean diff: {mean_angle_diffs[idx]:>6.2f}°")

print(f"\nMost different joints (largest angle difference):")
for i in range(3):
    idx = sorted_indices[-(i+1)]
    print(f"  {angle_names[idx]:<20s} Mean diff: {mean_angle_diffs[idx]:>6.2f}°")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

print(f"""
Overall joint angle difference: {overall_mean_diff:.2f}°

Interpretation guide for joint angles:
  < 5°:  Excellent agreement - poses are nearly identical
  < 10°: Good agreement - minor pose differences
  < 20°: Moderate agreement - noticeable pose differences
  > 20°: Poor agreement - significantly different poses

Current result ({overall_mean_diff:.2f}°):
""")

if overall_mean_diff < 5:
    print("✅ EXCELLENT: Joint configurations are nearly identical!")
    print("   Both methods estimate the same pose.")
elif overall_mean_diff < 10:
    print("✅ GOOD: Joint configurations are similar with minor variations.")
    print("   Differences are typical for different 3D lifting methods.")
elif overall_mean_diff < 20:
    print("⚠️ MODERATE: Noticeable differences in joint angles.")
    print("   Methods have different estimates of limb bending/orientation.")
else:
    print("❌ POOR: Significantly different joint angles.")
    print("   Methods fundamentally disagree on pose configuration.")

print("""
Key insights:
- Joint angles are completely scale-invariant
- They directly measure pose configuration (e.g., how bent is the elbow?)
- Combined with bone ratios (0.03 diff) and joint angles (above), we get:
  * Same body proportions ✓
  * Different joint positions (PA-MPJPE 0.86) ?
  * Joint angle comparison shows if it's orientation or actual pose difference
""")

# Detailed interpretation
print("\n" + "=" * 80)
print("DETAILED FINDINGS")
print("=" * 80)

print(f"""
Previous findings recap:
  2D Input Difference:     1.9 pixels  → Same 2D starting point ✅
  Bone Length Ratios:      0.03 diff   → Same body proportions ✅
  PA-MPJPE (positions):    0.86        → Different joint positions ⚠️
  Joint Angles:            {overall_mean_diff:.2f}°       → {"Same pose ✅" if overall_mean_diff < 10 else "Different pose ⚠️"}

""")

if overall_mean_diff < 10:
    print("🎯 CONCLUSION:")
    print("   Both methods estimate the SAME POSE with same proportions.")
    print("   PA-MPJPE = 0.86 is due to different coordinate system choices,")
    print("   not actual differences in the estimated body configuration!")
    print("\n   → Both are equivalent representations of the same pose.")
else:
    print("🎯 CONCLUSION:")
    print("   Methods estimate DIFFERENT POSES (different joint angles).")
    print("   Even though they start from same 2D and preserve proportions,")
    print("   they make different choices about limb orientations in 3D space.")
    print("\n   → This is expected due to 2D→3D depth ambiguity.")

print("\n✅ Analysis complete!\n")
