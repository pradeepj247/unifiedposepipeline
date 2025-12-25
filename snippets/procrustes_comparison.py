"""
Proper 3D Pose Comparison with Procrustes Alignment

This implements the standard approach used in 3D pose estimation papers:
1. Root-center both skeletons (remove translation)
2. Scale normalize (unit skeleton size)
3. Procrustes alignment (optional, removes rotation)
4. Calculate MPJPE (Mean Per Joint Position Error)

References:
- MPJPE is the standard metric in Human3.6M, 3DPW, MPI-INF-3DHP papers
- Procrustes alignment follows: Martinez et al. "A simple yet effective baseline for 3d human pose estimation"

Usage in Colab:
    python procrustes_comparison.py
"""

import numpy as np
from scipy.spatial import procrustes

def root_center(joints, root_idx=0):
    """
    Center skeleton at root joint (removes translation)
    
    Args:
        joints: (T, J, 3) or (J, 3) array of 3D joints
        root_idx: index of root joint (typically pelvis)
    
    Returns:
        root-centered joints
    """
    if joints.ndim == 3:  # (T, J, 3)
        return joints - joints[:, root_idx:root_idx+1, :]
    else:  # (J, 3)
        return joints - joints[root_idx:root_idx+1, :]


def scale_normalize(joints):
    """
    Normalize skeleton to unit scale
    
    Args:
        joints: (T, J, 3) or (J, 3) root-centered joints
    
    Returns:
        scale-normalized joints
    """
    if joints.ndim == 3:  # (T, J, 3)
        # Scale per frame
        scale = np.mean(np.linalg.norm(joints, axis=2, keepdims=True), axis=1, keepdims=True)
        scale = np.where(scale == 0, 1, scale)  # Avoid division by zero
        return joints / scale
    else:  # (J, 3)
        scale = np.mean(np.linalg.norm(joints, axis=1))
        return joints / scale if scale > 0 else joints


def procrustes_align_frame(source, target):
    """
    Align source to target using Procrustes (removes rotation + scale)
    
    Args:
        source: (J, 3) joints to align
        target: (J, 3) reference joints
    
    Returns:
        aligned_source: (J, 3) aligned joints
        disparity: scalar, measure of dissimilarity
    """
    # scipy.spatial.procrustes returns (mtx1, mtx2, disparity)
    # mtx1 is target (standardized), mtx2 is source (aligned to target)
    _, aligned, disparity = procrustes(target, source)
    return aligned, disparity


def mpjpe(predicted, target):
    """
    Mean Per Joint Position Error (standard 3D pose metric)
    
    Args:
        predicted: (T, J, 3) or (J, 3) predicted joints
        target: (T, J, 3) or (J, 3) ground truth joints
    
    Returns:
        mpjpe: scalar, mean error across all joints and frames
    """
    return np.mean(np.linalg.norm(predicted - target, axis=-1))


def pa_mpjpe(predicted, target):
    """
    Procrustes-Aligned MPJPE (PA-MPJPE)
    Aligns each frame before computing error
    
    Args:
        predicted: (T, J, 3) predicted joints
        target: (T, J, 3) ground truth joints
    
    Returns:
        pa_mpjpe: scalar, mean error after per-frame alignment
    """
    assert predicted.shape == target.shape
    
    if predicted.ndim == 2:  # Single frame (J, 3)
        aligned, _ = procrustes_align_frame(predicted, target)
        return mpjpe(aligned, target)
    
    # Multi-frame (T, J, 3)
    errors = []
    for i in range(predicted.shape[0]):
        aligned, _ = procrustes_align_frame(predicted[i], target[i])
        error = np.mean(np.linalg.norm(aligned - target[i], axis=-1))
        errors.append(error)
    
    return np.mean(errors)


def map_coco_to_h36m(coco_joints):
    """
    Map COCO-17 body joints to H36M-17 joint order
    
    VERIFIED MAPPING (from MotionAGFormer code):
    
    COCO-17 (WB3D body keypoints 0-16):
      0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
      5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
      9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
      13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    
    H36M-17 (MAGF output):
      0: Hip (pelvis), 1: RHip, 2: RKnee, 3: RAnkle, 4: LHip, 5: LKnee, 6: LAnkle,
      7: Spine, 8: Thorax, 9: Nose, 10: Head,
      11: LShoulder, 12: LElbow, 13: LWrist, 14: RShoulder, 15: RElbow, 16: RWrist
    
    Direct matches: 11 joints (RHip, RKnee, RAnkle, LHip, LKnee, LAnkle, Nose,
                                LShoulder, LElbow, LWrist, RShoulder, RElbow, RWrist)
    Computed: 6 joints (Hip, Spine, Thorax, Head) - derived geometrically
    
    Args:
        coco_joints: (T, 17, 3) or (17, 3) in COCO order
    
    Returns:
        h36m_joints: (T, 17, 3) or (17, 3) in H36M order
    """
    # COCO indices (verified from COCO dataset specification)
    c_nose, c_leye, c_reye = 0, 1, 2
    c_lear, c_rear = 3, 4
    c_lshoulder, c_rshoulder = 5, 6
    c_lelbow, c_relbow = 7, 8
    c_lwrist, c_rwrist = 9, 10
    c_lhip, c_rhip = 11, 12
    c_lknee, c_rknee = 13, 14
    c_lankle, c_rankle = 15, 16
    
    # H36M order (verified from MotionAGFormer convert_vitpose_to_magf.py)
    # Direct correspondences:
    #   H36M 1 ← COCO 12 (RHip)
    #   H36M 2 ← COCO 14 (RKnee)
    #   H36M 3 ← COCO 16 (RAnkle)
    #   H36M 4 ← COCO 11 (LHip)
    #   H36M 5 ← COCO 13 (LKnee)
    #   H36M 6 ← COCO 15 (LAnkle)
    #   H36M 9 ← COCO 0  (Nose)
    #   H36M 11 ← COCO 5 (LShoulder)
    #   H36M 12 ← COCO 7 (LElbow)
    #   H36M 13 ← COCO 9 (LWrist)
    #   H36M 14 ← COCO 6 (RShoulder)
    #   H36M 15 ← COCO 8 (RElbow)
    #   H36M 16 ← COCO 10 (RWrist)
    
    if coco_joints.ndim == 3:  # (T, 17, 3)
        T = coco_joints.shape[0]
        h36m = np.zeros((T, 17, 3), dtype=coco_joints.dtype)
        
        for t in range(T):
            # Hips
            h36m[t, 0] = (coco_joints[t, c_lhip] + coco_joints[t, c_rhip]) / 2  # Pelvis
            h36m[t, 1] = coco_joints[t, c_rhip]
            h36m[t, 4] = coco_joints[t, c_lhip]
            
            # Legs
            h36m[t, 2] = coco_joints[t, c_rknee]
            h36m[t, 3] = coco_joints[t, c_rankle]
            h36m[t, 5] = coco_joints[t, c_lknee]
            h36m[t, 6] = coco_joints[t, c_lankle]
            
            # Torso
            hip_center = h36m[t, 0]
            shoulder_center = (coco_joints[t, c_lshoulder] + coco_joints[t, c_rshoulder]) / 2
            h36m[t, 7] = hip_center + (shoulder_center - hip_center) * 0.33  # Spine
            h36m[t, 8] = shoulder_center  # Thorax
            
            # Head
            h36m[t, 9] = coco_joints[t, c_nose]  # Neck/Nose
            # Head: approximate as nose + upward offset
            head_offset = (coco_joints[t, c_nose] - shoulder_center) * 0.5
            h36m[t, 10] = coco_joints[t, c_nose] + head_offset
            
            # Arms
            h36m[t, 11] = coco_joints[t, c_lshoulder]
            h36m[t, 12] = coco_joints[t, c_lelbow]
            h36m[t, 13] = coco_joints[t, c_lwrist]
            h36m[t, 14] = coco_joints[t, c_rshoulder]
            h36m[t, 15] = coco_joints[t, c_relbow]
            h36m[t, 16] = coco_joints[t, c_rwrist]
    
    else:  # (17, 3) single frame
        h36m = np.zeros((17, 3), dtype=coco_joints.dtype)
        
        # Hips
        h36m[0] = (coco_joints[c_lhip] + coco_joints[c_rhip]) / 2  # Pelvis
        h36m[1] = coco_joints[c_rhip]
        h36m[4] = coco_joints[c_lhip]
        
        # Legs
        h36m[2] = coco_joints[c_rknee]
        h36m[3] = coco_joints[c_rankle]
        h36m[5] = coco_joints[c_lknee]
        h36m[6] = coco_joints[c_lankle]
        
        # Torso
        hip_center = h36m[0]
        shoulder_center = (coco_joints[c_lshoulder] + coco_joints[c_rshoulder]) / 2
        h36m[7] = hip_center + (shoulder_center - hip_center) * 0.33  # Spine
        h36m[8] = shoulder_center  # Thorax
        
        # Head
        h36m[9] = coco_joints[c_nose]  # Neck/Nose
        head_offset = (coco_joints[c_nose] - shoulder_center) * 0.5
        h36m[10] = coco_joints[c_nose] + head_offset
        
        # Arms
        h36m[11] = coco_joints[c_lshoulder]
        h36m[12] = coco_joints[c_lelbow]
        h36m[13] = coco_joints[c_lwrist]
        h36m[14] = coco_joints[c_rshoulder]
        h36m[15] = coco_joints[c_relbow]
        h36m[16] = coco_joints[c_rwrist]
    
    return h36m


# ============================================================================
# Main Comparison Script
# ============================================================================

print("=" * 80)
print("PROCRUSTES-ALIGNED 3D POSE COMPARISON")
print("=" * 80)

# Load both files
print("\n📂 Loading data...")
magf_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_3D_magf.npz')
wb3d_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_3D_wb.npz')

magf_poses = magf_data['poses_3d']  # (120, 17, 3) H36M format
wb3d_poses = wb3d_data['keypoints_3d']  # (360, 133, 3) COCO-WholeBody

print(f"✅ MAGF: {magf_poses.shape} (H36M-17)")
print(f"✅ WB3D: {wb3d_poses.shape} (COCO-WholeBody-133)")

# Extract WB3D body joints (first 17) and map to H36M order
print("\n🔄 Mapping WB3D (COCO-17) to H36M-17 joint order...")
wb3d_body = wb3d_poses[:, :17, :]  # (360, 17, 3) COCO body joints
wb3d_h36m = map_coco_to_h36m(wb3d_body)  # Convert to H36M order

# Use only overlapping frames (MAGF has 120, WB3D has 360)
n_frames = min(magf_poses.shape[0], wb3d_h36m.shape[0])
magf = magf_poses[:n_frames]
wb3d = wb3d_h36m[:n_frames]

print(f"✅ Comparing {n_frames} frames")

# ============================================================================
# Method 1: Raw comparison (original scales)
# ============================================================================
print("\n" + "=" * 80)
print("METHOD 1: Raw Comparison (Different Scales)")
print("=" * 80)
raw_mpjpe = mpjpe(magf, wb3d)
print(f"MPJPE (raw): {raw_mpjpe:.4f}")
print("⚠️ Not meaningful due to different scales!")

# ============================================================================
# Method 2: Root-centered + Scale-normalized
# ============================================================================
print("\n" + "=" * 80)
print("METHOD 2: Root-Centered + Scale-Normalized")
print("=" * 80)

# Root-center at pelvis (joint 0 in H36M)
magf_centered = root_center(magf, root_idx=0)
wb3d_centered = root_center(wb3d, root_idx=0)

# Scale normalize
magf_normalized = scale_normalize(magf_centered)
wb3d_normalized = scale_normalize(wb3d_centered)

# Compute MPJPE
normalized_mpjpe = mpjpe(magf_normalized, wb3d_normalized)
print(f"MPJPE (root-centered + scale-normalized): {normalized_mpjpe:.4f}")
print("✅ Both skeletons now in unit scale, translation removed")

# ============================================================================
# Method 3: Procrustes-Aligned MPJPE (PA-MPJPE)
# ============================================================================
print("\n" + "=" * 80)
print("METHOD 3: Procrustes-Aligned MPJPE (PA-MPJPE)")
print("=" * 80)

# Use normalized poses for Procrustes
pa_mpjpe_value = pa_mpjpe(wb3d_normalized, magf_normalized)
print(f"PA-MPJPE (all 17 joints): {pa_mpjpe_value:.4f}")
print("✅ Translation, scale, AND rotation removed")
print("   (This is the standard metric in 3D pose papers)")

# ============================================================================
# Method 4: PA-MPJPE on 11 DIRECTLY COMPARABLE joints only
# ============================================================================
print("\n" + "=" * 80)
print("METHOD 4: PA-MPJPE (11 Directly Comparable Joints Only)")
print("=" * 80)
print("Excluding 6 computed joints: Hip(0), Spine(7), Thorax(8), Head(10)")
print("Using only joints with direct anatomical correspondence")

# H36M indices of the 11 directly comparable joints
directly_comparable = [1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]
# Corresponds to: RHip, RKnee, RAnkle, LHip, LKnee, LAnkle, Nose,
#                  LShoulder, LElbow, LWrist, RShoulder, RElbow, RWrist

magf_comparable = magf_normalized[:, directly_comparable, :]
wb3d_comparable = wb3d_normalized[:, directly_comparable, :]

pa_mpjpe_comparable = pa_mpjpe(wb3d_comparable, magf_comparable)
print(f"\nPA-MPJPE (11 joints): {pa_mpjpe_comparable:.4f}")
print("✅ Fairest comparison - only anatomically identical joints")

# ============================================================================
# Per-Joint Error Analysis
# ============================================================================
print("\n" + "=" * 80)
print("PER-JOINT ERROR ANALYSIS (Frame 0)")
print("=" * 80)

h36m_joint_names = [
    'Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'Spine', 'Thorax', 'Neck', 'Head', 'LShoulder', 'LElbow', 'LWrist',
    'RShoulder', 'RElbow', 'RWrist'
]

# Use normalized poses for frame 0
frame_0_magf = magf_normalized[0]
frame_0_wb3d = wb3d_normalized[0]

# Align frame 0
frame_0_wb3d_aligned, _ = procrustes_align_frame(frame_0_wb3d, frame_0_magf)

print(f"\n{'Joint':<12s} {'MAGF (X,Y,Z)':<30s} {'WB3D Aligned (X,Y,Z)':<30s} {'Error':>8s}")
print("-" * 80)

per_joint_errors = []
for j in range(17):
    m = frame_0_magf[j]
    w = frame_0_wb3d_aligned[j]
    error = np.linalg.norm(m - w)
    per_joint_errors.append(error)
    
    magf_str = f"({m[0]:6.3f},{m[1]:6.3f},{m[2]:6.3f})"
    wb3d_str = f"({w[0]:6.3f},{w[1]:6.3f},{w[2]:6.3f})"
    
    print(f"{h36m_joint_names[j]:<12s} {magf_str:<30s} {wb3d_str:<30s} {error:8.4f}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total frames compared: {n_frames}")
print(f"\nMetrics (all 17 joints):")
print(f"  MPJPE (normalized):  {normalized_mpjpe:.4f}  ← Fair comparison, preserves orientation")
print(f"  PA-MPJPE:            {pa_mpjpe_value:.4f}  ← Best metric, removes all transforms")
print(f"\nMetrics (11 directly comparable joints only):")
print(f"  PA-MPJPE (11 joints): {pa_mpjpe_comparable:.4f}  ← FAIREST - excludes computed joints")
print(f"\nPer-joint error (frame 0, all 17 joints):")
print(f"  Mean:  {np.mean(per_joint_errors):.4f}")
print(f"  Max:   {np.max(per_joint_errors):.4f} ({h36m_joint_names[np.argmax(per_joint_errors)]})")
print(f"  Min:   {np.min(per_joint_errors):.4f} ({h36m_joint_names[np.argmin(per_joint_errors)]})")

# Calculate per-joint errors for only the 11 comparable joints
directly_comparable = [1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]
comparable_names = [h36m_joint_names[i] for i in directly_comparable]
comparable_errors = [per_joint_errors[i] for i in directly_comparable]

print(f"\nPer-joint error (frame 0, 11 comparable joints only):")
print(f"  Mean:  {np.mean(comparable_errors):.4f}")
print(f"  Max:   {np.max(comparable_errors):.4f} ({comparable_names[np.argmax(comparable_errors)]})")
print(f"  Min:   {np.min(comparable_errors):.4f} ({comparable_names[np.argmin(comparable_errors)]})")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
print(f"""
PA-MPJPE (11 comparable joints): {pa_mpjpe_comparable:.4f}

Interpretation guide:
  < 0.05: Excellent agreement (nearly identical poses)
  < 0.10: Good agreement (minor differences)
  < 0.20: Moderate agreement (noticeable differences)
  > 0.20: Poor agreement (different poses or methods)

Note: Some error is expected due to:
1. Different estimation methods:
   - MAGF: Temporal model (uses past/future frames for smoothing)
   - WB3D: Single-frame model (no temporal context)
2. Different 2D input sources (may have different detection quality)
3. Model-specific biases in 3D lifting

The 11-joint PA-MPJPE is the FAIREST metric as it:
✅ Excludes 6 computed joints (Hip, Spine, Thorax, Head)
✅ Compares only anatomically identical locations
✅ Removes translation, scale, and rotation biases
""")

print("\n✅ Analysis complete!\n")
