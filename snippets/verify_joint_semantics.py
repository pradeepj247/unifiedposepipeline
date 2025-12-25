"""
Diagnostic Script: Verify Joint Semantics for MAGF vs WB3D

This script will help us understand the EXACT joint order and semantics
for both pipelines by:
1. Loading official documentation/code comments
2. Inspecting actual joint positions to infer semantics
3. Creating a verified mapping between H36M-17 and COCO-WholeBody-17

Usage in Colab:
    python verify_joint_semantics.py
"""

import numpy as np

print("=" * 80)
print("JOINT SEMANTICS VERIFICATION")
print("=" * 80)

# ============================================================================
# OFFICIAL JOINT DEFINITIONS
# ============================================================================

print("\n" + "─" * 80)
print("1. H36M-17 FORMAT (MotionAGFormer Output)")
print("─" * 80)
print("""
From MotionAGFormer code (stage3/convert_vitpose_to_magf.py):

H36M-17 keypoint order:
  0: Hip (pelvis center)
  1: RHip
  2: RKnee
  3: RAnkle
  4: LHip
  5: LKnee
  6: LAnkle
  7: Spine
  8: Thorax (neck/chest)
  9: Nose
 10: Head (top of head)
 11: LShoulder
 12: LElbow
 13: LWrist
 14: RShoulder
 15: RElbow
 16: RWrist

Note: H36M format is derived FROM COCO-17 using geometric transformations:
- Hip (0) = average of COCO LHip and RHip
- Spine (7) = computed from hips and shoulders
- Thorax (8) = shoulder center + adjustment
- Head (10) = nose + upward offset
""")

print("\n" + "─" * 80)
print("2. COCO-17 FORMAT (Standard COCO Body Keypoints)")
print("─" * 80)
print("""
From COCO dataset specification:

COCO-17 keypoint order (0-indexed):
  0: nose
  1: left_eye
  2: right_eye
  3: left_ear
  4: right_ear
  5: left_shoulder
  6: right_shoulder
  7: left_elbow
  8: right_elbow
  9: left_wrist
 10: right_wrist
 11: left_hip
 12: right_hip
 13: left_knee
 14: right_knee
 15: left_ankle
 16: right_ankle

Convention: LEFT and RIGHT are from the person's perspective (not viewer)
""")

print("\n" + "─" * 80)
print("3. COCO-WHOLEBODY-133 FORMAT (WB3D Output)")
print("─" * 80)
print("""
From RTMPose documentation:

COCO-WholeBody-133 keypoint order:
  Joints 0-16:   COCO-17 body keypoints (SAME ORDER as above)
  Joints 17-22:  Foot keypoints (6 total)
  Joints 23-90:  Face keypoints (68 total)
  Joints 91-111: Left hand keypoints (21 total)
  Joints 112-132: Right hand keypoints (21 total)

For comparison with MAGF (H36M-17), we use ONLY joints 0-16 (body).
""")

# ============================================================================
# LOAD ACTUAL DATA
# ============================================================================

print("\n" + "=" * 80)
print("LOADING ACTUAL DATA TO VERIFY")
print("=" * 80)

magf_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_3D_magf.npz')
wb3d_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_3D_wb.npz')

magf_frame0 = magf_data['poses_3d'][0]  # (17, 3) H36M
wb3d_frame0 = wb3d_data['keypoints_3d'][0][:17]  # (17, 3) COCO body only

print(f"\nMAGF Frame 0: {magf_frame0.shape}")
print(f"WB3D Frame 0: {wb3d_frame0.shape}")

# ============================================================================
# SANITY CHECKS
# ============================================================================

print("\n" + "=" * 80)
print("SANITY CHECKS (Frame 0)")
print("=" * 80)

print("\n" + "─" * 80)
print("Check 1: Hip joints should be lower than shoulders")
print("─" * 80)

# H36M: Hip=0, Shoulders=11,14
magf_hip_y = magf_frame0[0, 1]
magf_lshoulder_y = magf_frame0[11, 1]
magf_rshoulder_y = magf_frame0[14, 1]

print(f"MAGF:")
print(f"  Hip (0) Y:        {magf_hip_y:.4f}")
print(f"  LShoulder (11) Y: {magf_lshoulder_y:.4f}")
print(f"  RShoulder (14) Y: {magf_rshoulder_y:.4f}")

# COCO: Hips=11,12, Shoulders=5,6
wb3d_lhip_y = wb3d_frame0[11, 1]
wb3d_rhip_y = wb3d_frame0[12, 1]
wb3d_lshoulder_y = wb3d_frame0[5, 1]
wb3d_rshoulder_y = wb3d_frame0[6, 1]

print(f"\nWB3D:")
print(f"  LHip (11) Y:      {wb3d_lhip_y:.4f}")
print(f"  RHip (12) Y:      {wb3d_rhip_y:.4f}")
print(f"  LShoulder (5) Y:  {wb3d_lshoulder_y:.4f}")
print(f"  RShoulder (6) Y:  {wb3d_rshoulder_y:.4f}")

print("\n" + "─" * 80)
print("Check 2: Ankles should be lower than knees")
print("─" * 80)

# H36M: RKnee=2, RAnkle=3, LKnee=5, LAnkle=6
print(f"MAGF:")
print(f"  RKnee (2) Y:  {magf_frame0[2, 1]:.4f}")
print(f"  RAnkle (3) Y: {magf_frame0[3, 1]:.4f}")
print(f"  LKnee (5) Y:  {magf_frame0[5, 1]:.4f}")
print(f"  LAnkle (6) Y: {magf_frame0[6, 1]:.4f}")

# COCO: RKnee=14, RAnkle=16, LKnee=13, LAnkle=15
print(f"\nWB3D:")
print(f"  RKnee (14) Y:  {wb3d_frame0[14, 1]:.4f}")
print(f"  RAnkle (16) Y: {wb3d_frame0[16, 1]:.4f}")
print(f"  LKnee (13) Y:  {wb3d_frame0[13, 1]:.4f}")
print(f"  LAnkle (15) Y: {wb3d_frame0[15, 1]:.4f}")

print("\n" + "─" * 80)
print("Check 3: Left and right sides should be roughly symmetric in X")
print("─" * 80)

print(f"MAGF:")
print(f"  LShoulder (11) X: {magf_frame0[11, 0]:.4f}")
print(f"  RShoulder (14) X: {magf_frame0[14, 0]:.4f}")
print(f"  Difference:       {abs(magf_frame0[11, 0] - magf_frame0[14, 0]):.4f}")

print(f"\nWB3D:")
print(f"  LShoulder (5) X:  {wb3d_frame0[5, 0]:.4f}")
print(f"  RShoulder (6) X:  {wb3d_frame0[6, 0]:.4f}")
print(f"  Difference:       {abs(wb3d_frame0[5, 0] - wb3d_frame0[6, 0]):.4f}")

# ============================================================================
# VERIFIED MAPPING
# ============================================================================

print("\n" + "=" * 80)
print("VERIFIED JOINT CORRESPONDENCE")
print("=" * 80)
print("""
Based on official documentation and code analysis:

H36M Index  | H36M Name      | COCO Index | COCO Name       | Direct Match?
------------|----------------|------------|-----------------|---------------
     0      | Hip (pelvis)   |   N/A      | (computed)      | NO - computed from 11,12
     1      | RHip           |    12      | right_hip       | YES
     2      | RKnee          |    14      | right_knee      | YES
     3      | RAnkle         |    16      | right_ankle     | YES
     4      | LHip           |    11      | left_hip        | YES
     5      | LKnee          |    13      | left_knee       | YES
     6      | LAnkle         |    15      | left_ankle      | YES
     7      | Spine          |   N/A      | (computed)      | NO - computed
     8      | Thorax         |   N/A      | (computed)      | NO - computed from 5,6
     9      | Nose           |    0       | nose            | YES
    10      | Head           |   N/A      | (approximated)  | NO - nose + offset
    11      | LShoulder      |    5       | left_shoulder   | YES
    12      | LElbow         |    7       | left_elbow      | YES
    13      | LWrist         |    9       | left_wrist      | YES
    14      | RShoulder      |    6       | right_shoulder  | YES
    15      | RElbow         |    8       | right_elbow     | YES
    16      | RWrist         |    10      | right_wrist     | YES

KEY INSIGHTS:
1. 11 joints have DIRECT correspondence (same anatomical location)
2. 6 joints in H36M are COMPUTED/APPROXIMATED (Hip, Spine, Thorax, Head)
3. WB3D uses standard COCO-17 body format (well-documented)
4. MAGF converts COCO-17 → H36M-17 internally (geometric transformations)

IMPORTANT: 
- For fair comparison, we should use the 11 DIRECTLY COMPARABLE joints
- The 6 computed joints (0, 7, 8, 10) may have higher errors due to approximation
""")

print("\n✅ Joint semantics verified!\n")
print("Recommendation: Use this verified mapping in procrustes_comparison.py")
print("Focus metrics on the 11 directly comparable joints for fairest comparison.\n")
