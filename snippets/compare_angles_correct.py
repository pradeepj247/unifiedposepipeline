"""
CORRECTED 2D vs 3D Angle Comparison (No Fake Projection)

As user correctly identified: We should NOT drop Z and pretend it's a projection.

Instead: Compare angles directly in their native spaces.
- 2D angle from RTM 2D keypoints (image space)
- 3D angle from MAGF 3D keypoints (body space)

Both measure the SAME physical angle (e.g., elbow bend).
If they differ significantly, MAGF misinterpreted the pose.

Usage in Colab:
    python compare_angles_correct.py
"""

import numpy as np

def angle_between_vectors(v1, v2):
    """Calculate angle between two vectors in degrees [0, 180]"""
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    return np.degrees(angle_rad)


def calculate_joint_angle(j1, j2, j3, keypoints):
    """
    Calculate angle at joint j2 formed by j1-j2-j3
    Works for both 2D (N, 2) and 3D (N, 3) keypoints
    """
    p1 = keypoints[j1]
    p2 = keypoints[j2]
    p3 = keypoints[j3]
    
    v1 = p1 - p2
    v2 = p3 - p2
    
    return angle_between_vectors(v1, v2)


# Joint angle definitions (H36M-17 format)
JOINT_ANGLES = {
    'LElbow': (11, 12, 13),
    'RElbow': (14, 15, 16),
    'LKnee': (4, 5, 6),
    'RKnee': (1, 2, 3),
    'LShoulder': (0, 11, 12),
    'RShoulder': (0, 14, 15),
    'LHip': (11, 0, 4),
    'RHip': (14, 0, 1),
    'ShoulderBridge': (11, 0, 14),
}


print("=" * 80)
print("CORRECT 2D vs 3D ANGLE COMPARISON (Native Spaces)")
print("=" * 80)

print("""
✅ CORRECTED APPROACH:

- Calculate 2D angles from RTM 2D keypoints (in image space)
- Calculate 3D angles from MAGF 3D keypoints (in body space)
- Compare directly (no fake projection)

Both angles measure the SAME physical quantity (e.g., elbow bend).
The difference indicates how well MAGF interpreted the 2D pose.

Note: This assumes angles are invariant to coordinate system
(which is true - an elbow bent 90° is 90° in any coordinate system).
""")

# Load data
rtm_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_2D_rtm.npz')
magf_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_3D_magf.npz')

rtm_2d = rtm_data['keypoints']
magf_3d = magf_data['poses_3d']

print(f"\n✅ Loaded: RTM 2D {rtm_2d.shape}, MAGF 3D {magf_3d.shape}")

# Analyze frames
NUM_FRAMES = 30

print(f"\n📊 Analyzing {NUM_FRAMES} frames...")

# Collect statistics
all_deltas = []
joint_deltas = {name: [] for name in JOINT_ANGLES.keys()}

for frame_idx in range(NUM_FRAMES):
    frame_2d = rtm_2d[frame_idx]
    frame_3d = magf_3d[frame_idx]
    
    for joint_name, (j1, j2, j3) in JOINT_ANGLES.items():
        # Calculate angles in native spaces
        angle_2d = calculate_joint_angle(j1, j2, j3, frame_2d)
        angle_3d = calculate_joint_angle(j1, j2, j3, frame_3d)
        
        delta = abs(angle_2d - angle_3d)
        
        joint_deltas[joint_name].append(delta)
        all_deltas.append(delta)

# Calculate statistics
print("\n" + "=" * 80)
print("RESULTS: RTM 2D vs MAGF 3D (Native Space Comparison)")
print("=" * 80)

overall_mean = np.mean(all_deltas)
overall_max = np.max(all_deltas)

print(f"\n📊 Overall Statistics:")
print(f"   Mean angle difference:  {overall_mean:6.2f}°")
print(f"   Max angle difference:   {overall_max:6.2f}°")
print(f"   Frames analyzed:        {NUM_FRAMES}")
print(f"   Joint angles per frame: {len(JOINT_ANGLES)}")

print(f"\n📏 Per-Joint Statistics:")
print(f"{'Joint':<15s} {'Mean Δ':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
print("-" * 80)

sorted_joints = sorted(joint_deltas.items(), 
                      key=lambda x: np.mean(x[1]), 
                      reverse=True)

for joint_name, deltas in sorted_joints:
    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas)
    min_delta = np.min(deltas)
    max_delta = np.max(deltas)
    
    print(f"{joint_name:<15s} "
          f"{mean_delta:>9.2f}° "
          f"{std_delta:>9.2f}° "
          f"{min_delta:>9.2f}° "
          f"{max_delta:>9.2f}°")

# Threshold analysis
THRESHOLD = 5.0

print(f"\n🚨 Frames Exceeding {THRESHOLD}° Threshold:")
print(f"{'Joint':<15s} {'Count':>10s} {'Percentage':>12s}")
print("-" * 80)

for joint_name, deltas in sorted_joints:
    count = sum(1 for d in deltas if d > THRESHOLD)
    pct = (count / NUM_FRAMES) * 100
    print(f"{joint_name:<15s} {count:>10d} {pct:>11.1f}%")

# Quality assessment
print(f"\n✅ Quality Assessment:")

excellent = sum(1 for name, deltas in joint_deltas.items() if np.mean(deltas) < 3)
good = sum(1 for name, deltas in joint_deltas.items() if 3 <= np.mean(deltas) < 5)
moderate = sum(1 for name, deltas in joint_deltas.items() if 5 <= np.mean(deltas) < 10)
poor = sum(1 for name, deltas in joint_deltas.items() if 10 <= np.mean(deltas) < 20)
critical = sum(1 for name, deltas in joint_deltas.items() if np.mean(deltas) >= 20)

print(f"   Excellent (< 3°):   {excellent} joints")
print(f"   Good (3-5°):        {good} joints")
print(f"   Moderate (5-10°):   {moderate} joints")
print(f"   Poor (10-20°):      {poor} joints")
print(f"   Critical (≥ 20°):   {critical} joints")

# Detailed example (Frame 0, LElbow)
print("\n" + "=" * 80)
print("DETAILED EXAMPLE: Frame 0, Left Elbow")
print("=" * 80)

frame_0_2d = rtm_2d[0]
frame_0_3d = magf_3d[0]

j1, j2, j3 = JOINT_ANGLES['LElbow']  # Shoulder, Elbow, Wrist

# 2D calculation
shoulder_2d = frame_0_2d[j1]
elbow_2d = frame_0_2d[j2]
wrist_2d = frame_0_2d[j3]

v1_2d = shoulder_2d - elbow_2d
v2_2d = wrist_2d - elbow_2d
angle_2d = calculate_joint_angle(j1, j2, j3, frame_0_2d)

print(f"\n2D (RTM - Image Space):")
print(f"   Shoulder: {shoulder_2d}")
print(f"   Elbow:    {elbow_2d}")
print(f"   Wrist:    {wrist_2d}")
print(f"   Vector 1 (elbow→shoulder): {v1_2d}")
print(f"   Vector 2 (elbow→wrist):    {v2_2d}")
print(f"   **Angle: {angle_2d:.2f}°**")

# 3D calculation
shoulder_3d = frame_0_3d[j1]
elbow_3d = frame_0_3d[j2]
wrist_3d = frame_0_3d[j3]

v1_3d = shoulder_3d - elbow_3d
v2_3d = wrist_3d - elbow_3d
angle_3d = calculate_joint_angle(j1, j2, j3, frame_0_3d)

print(f"\n3D (MAGF - Body Space):")
print(f"   Shoulder: {shoulder_3d}")
print(f"   Elbow:    {elbow_3d}")
print(f"   Wrist:    {wrist_3d}")
print(f"   Vector 1 (elbow→shoulder): {v1_3d}")
print(f"   Vector 2 (elbow→wrist):    {v2_3d}")
print(f"   **Angle: {angle_3d:.2f}°**")

print(f"\n❗ Difference: {abs(angle_2d - angle_3d):.2f}°")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

if overall_mean < 5:
    assessment = "EXCELLENT"
    interpretation = """
    ✅ MAGF accurately interprets the 2D poses!
    The 3D skeletal angles match the 2D observations very well.
    """
elif overall_mean < 10:
    assessment = "GOOD"
    interpretation = """
    ✅ MAGF generally interprets 2D poses correctly.
    Small differences are expected due to depth ambiguity.
    """
elif overall_mean < 20:
    assessment = "MODERATE"
    interpretation = """
    ⚠️  MAGF has moderate disagreement with 2D observations.
    Some joints (likely legs) have significant depth ambiguity.
    """
else:
    assessment = "POOR"
    interpretation = """
    ❌ MAGF significantly misinterprets the 2D poses.
    Large angle differences indicate depth estimation problems.
    """

print(f"\n📊 Overall Assessment: {assessment}")
print(interpretation)

print(f"""
🔬 **Technical Note:**

This comparison is CORRECT because:

1. Angles are coordinate-system invariant
   - A 90° elbow is 90° in image space AND body space
   
2. Both angles measure the same physical quantity
   - RTM 2D: "How the elbow appears to bend in the image"
   - MAGF 3D: "How bent the elbow is in 3D space"
   
3. Large differences indicate MAGF's 3D doesn't match 2D observation
   - This happens due to depth ambiguity (multiple 3D poses → same 2D)
   
4. Previous "projection" approach was WRONG
   - Dropping Z doesn't give proper projection
   - MAGF uses body-relative coordinates, not camera coordinates

💡 **Bottom Line:**

If mean difference is > 10°, MAGF's 3D interpretation
differs significantly from what we see in 2D (RTM).

This is the DEPTH AMBIGUITY problem - given a 2D pose,
multiple 3D poses could produce it, and MAGF picks one
based on learned priors.
""")

print("\n✅ Analysis complete!\n")
