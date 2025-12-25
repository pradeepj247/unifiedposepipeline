"""
Bone Length Ratio Comparison (Body-Relative Normalization)

Compare MAGF vs WB3D using anatomically meaningful bone length ratios.
This is more robust than global scale normalization because:
- Removes scale ambiguity completely
- Preserves anatomical proportions
- Easy to interpret (e.g., "forearm is 85% of upper arm length")

Usage in Colab:
    python compare_bone_ratios.py
"""

import numpy as np

def bone_length_3d(j1, j2):
    """Calculate 3D Euclidean distance between two joints"""
    return np.linalg.norm(j1 - j2)


def calculate_bone_lengths(joints_3d):
    """
    Calculate bone lengths for skeleton
    
    Args:
        joints_3d: (17, 3) array in H36M format
    
    Returns:
        dict of bone_name: length
    """
    bones = {}
    
    # Upper body bones
    bones['LShoulder-LElbow'] = bone_length_3d(joints_3d[11], joints_3d[12])
    bones['LElbow-LWrist'] = bone_length_3d(joints_3d[12], joints_3d[13])
    bones['RShoulder-RElbow'] = bone_length_3d(joints_3d[14], joints_3d[15])
    bones['RElbow-RWrist'] = bone_length_3d(joints_3d[15], joints_3d[16])
    bones['LShoulder-RShoulder'] = bone_length_3d(joints_3d[11], joints_3d[14])
    
    # Torso bones
    bones['LHip-LShoulder'] = bone_length_3d(joints_3d[4], joints_3d[11])
    bones['RHip-RShoulder'] = bone_length_3d(joints_3d[1], joints_3d[14])
    bones['LHip-RHip'] = bone_length_3d(joints_3d[4], joints_3d[1])
    
    # Lower body bones
    bones['LHip-LKnee'] = bone_length_3d(joints_3d[4], joints_3d[5])
    bones['LKnee-LAnkle'] = bone_length_3d(joints_3d[5], joints_3d[6])
    bones['RHip-RKnee'] = bone_length_3d(joints_3d[1], joints_3d[2])
    bones['RKnee-RAnkle'] = bone_length_3d(joints_3d[2], joints_3d[3])
    
    # Reference height: LShoulder to LAnkle (diagonal body height)
    bones['REFERENCE_HEIGHT'] = bone_length_3d(joints_3d[11], joints_3d[6])
    
    return bones


def normalize_bones(bones, reference_key='REFERENCE_HEIGHT'):
    """Normalize all bone lengths by reference length"""
    ref_length = bones[reference_key]
    normalized = {}
    for bone_name, length in bones.items():
        if bone_name != reference_key:
            normalized[bone_name] = length / ref_length if ref_length > 0 else 0
    return normalized


# ============================================================================
# Load data
# ============================================================================

print("=" * 80)
print("BONE LENGTH RATIO COMPARISON")
print("=" * 80)

# Load both 3D poses
magf_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_3D_magf.npz')
wb3d_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_3D_wb.npz')

magf_poses = magf_data['poses_3d']  # (120, 17, 3) H36M
wb3d_poses_full = wb3d_data['keypoints_3d']  # (360, 133, 3) COCO-WholeBody

print(f"\n✅ MAGF: {magf_poses.shape} (H36M-17)")
print(f"✅ WB3D: {wb3d_poses_full.shape} (COCO-WholeBody-133)")

# Map WB3D COCO-17 to H36M-17 (using verified mapping)
# For bone lengths, we can use a simpler direct mapping since we only need distances
# COCO to H36M index mapping for the 11 directly comparable joints
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

# Create H36M-ordered WB3D
n_frames = min(magf_poses.shape[0], wb3d_body.shape[0])
wb3d_h36m = np.zeros((n_frames, 17, 3), dtype=np.float32)

for t in range(n_frames):
    # Map the directly comparable joints
    for coco_idx, h36m_idx in coco_to_h36m_map.items():
        wb3d_h36m[t, h36m_idx] = wb3d_body[t, coco_idx]
    
    # Compute Hip (H36M 0) as average of LHip and RHip
    wb3d_h36m[t, 0] = (wb3d_body[t, 11] + wb3d_body[t, 12]) / 2

print(f"\n✅ Comparing {n_frames} frames")

# ============================================================================
# Frame 0 comparison
# ============================================================================

print("\n" + "=" * 80)
print("FRAME 0: BONE LENGTH COMPARISON")
print("=" * 80)

magf_frame0 = magf_poses[0]
wb3d_frame0 = wb3d_h36m[0]

# Calculate raw bone lengths
magf_bones = calculate_bone_lengths(magf_frame0)
wb3d_bones = calculate_bone_lengths(wb3d_frame0)

print(f"\nReference Height (LShoulder → LAnkle):")
print(f"  MAGF:  {magf_bones['REFERENCE_HEIGHT']:.4f}")
print(f"  WB3D:  {wb3d_bones['REFERENCE_HEIGHT']:.4f}")
print(f"  Ratio: {wb3d_bones['REFERENCE_HEIGHT'] / magf_bones['REFERENCE_HEIGHT']:.2f}x")

# Normalize bone lengths
magf_ratios = normalize_bones(magf_bones)
wb3d_ratios = normalize_bones(wb3d_bones)

print("\n" + "=" * 80)
print("NORMALIZED BONE LENGTH RATIOS (relative to body height)")
print("=" * 80)

print(f"\n{'Bone':<25s} {'MAGF Ratio':>12s} {'WB3D Ratio':>12s} {'Difference':>12s} {'% Diff':>10s}")
print("-" * 80)

bone_names = [
    'LShoulder-LElbow',
    'LElbow-LWrist',
    'RShoulder-RElbow',
    'RElbow-RWrist',
    'LShoulder-RShoulder',
    'LHip-LShoulder',
    'RHip-RShoulder',
    'LHip-RHip',
    'LHip-LKnee',
    'LKnee-LAnkle',
    'RHip-RKnee',
    'RKnee-RAnkle',
]

differences = []
for bone_name in bone_names:
    magf_ratio = magf_ratios[bone_name]
    wb3d_ratio = wb3d_ratios[bone_name]
    diff = abs(magf_ratio - wb3d_ratio)
    pct_diff = (diff / magf_ratio * 100) if magf_ratio > 0 else 0
    differences.append(diff)
    
    print(f"{bone_name:<25s} {magf_ratio:>12.4f} {wb3d_ratio:>12.4f} {diff:>12.4f} {pct_diff:>9.1f}%")

# ============================================================================
# Multi-frame analysis
# ============================================================================

print("\n" + "=" * 80)
print(f"MULTI-FRAME ANALYSIS (first {min(n_frames, 30)} frames)")
print("=" * 80)

frame_bone_diffs = {bone: [] for bone in bone_names}

for t in range(min(n_frames, 30)):
    magf_bones_t = calculate_bone_lengths(magf_poses[t])
    wb3d_bones_t = calculate_bone_lengths(wb3d_h36m[t])
    
    magf_ratios_t = normalize_bones(magf_bones_t)
    wb3d_ratios_t = normalize_bones(wb3d_bones_t)
    
    for bone_name in bone_names:
        diff = abs(magf_ratios_t[bone_name] - wb3d_ratios_t[bone_name])
        frame_bone_diffs[bone_name].append(diff)

print(f"\n{'Bone':<25s} {'Mean Diff':>12s} {'Std Dev':>12s} {'Max Diff':>12s}")
print("-" * 80)

mean_diffs = []
for bone_name in bone_names:
    diffs = frame_bone_diffs[bone_name]
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    max_diff = np.max(diffs)
    mean_diffs.append(mean_diff)
    
    print(f"{bone_name:<25s} {mean_diff:>12.4f} {std_diff:>12.4f} {max_diff:>12.4f}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

overall_mean_diff = np.mean(differences)
overall_mean_multi = np.mean(mean_diffs)

print(f"\nBone Length Ratio Differences:")
print(f"  Frame 0 - Mean difference:     {overall_mean_diff:.4f}")
print(f"  30 frames - Mean difference:   {overall_mean_multi:.4f}")

print(f"\nMost similar bones:")
sorted_indices = np.argsort(mean_diffs)
for i in range(3):
    idx = sorted_indices[i]
    print(f"  {bone_names[idx]:<25s} Diff: {mean_diffs[idx]:.4f}")

print(f"\nMost different bones:")
for i in range(3):
    idx = sorted_indices[-(i+1)]
    print(f"  {bone_names[idx]:<25s} Diff: {mean_diffs[idx]:.4f}")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

print(f"""
Overall bone ratio difference: {overall_mean_multi:.4f}

Interpretation:
  < 0.05: Excellent agreement - proportions nearly identical
  < 0.10: Good agreement - minor proportion differences
  < 0.20: Moderate agreement - noticeable differences in body proportions
  > 0.20: Poor agreement - significantly different body proportions

Current result ({overall_mean_multi:.4f}):
""")

if overall_mean_multi < 0.05:
    print("✅ EXCELLENT: Both methods produce very similar body proportions!")
    print("   The skeletons are essentially the same, just at different scales.")
elif overall_mean_multi < 0.10:
    print("✅ GOOD: Body proportions are similar with minor variations.")
    print("   Differences are likely due to different training biases.")
elif overall_mean_multi < 0.20:
    print("⚠️ MODERATE: Noticeable differences in estimated body proportions.")
    print("   The methods have different notions of limb length ratios.")
else:
    print("❌ POOR: Significantly different body proportions estimated.")
    print("   The methods fundamentally disagree on skeletal structure.")

print("""
Key insight: This bone ratio comparison is scale-invariant and tells us
if both methods agree on the SHAPE of the skeleton, independent of size.
""")

print("\n✅ Analysis complete!\n")
