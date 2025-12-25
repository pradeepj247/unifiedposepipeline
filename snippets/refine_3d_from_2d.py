"""
2D-Constrained 3D Pose Refinement (Proof of Concept)

Theory: Use accurate 2D keypoints to refine ambiguous 3D estimates.
Since 3D→2D projection must match observed 2D, we can adjust depths
while preserving 2D reprojection accuracy.

This is a simplified implementation showing the concept.
Production version would use proper optimization (scipy.optimize, PyTorch).

Usage in Colab:
    python refine_3d_from_2d.py
"""

import numpy as np
from scipy.optimize import minimize

def weak_perspective_projection(points_3d, scale=1.0):
    """
    Simple weak perspective projection (ignores Z for scaling)
    
    Args:
        points_3d: (N, 3) array of 3D points
        scale: projection scale factor
    
    Returns:
        points_2d: (N, 2) projected 2D points
    """
    # Weak perspective: just take X, Y (ignore Z depth effects)
    # This is a simplification - real projection uses Z for scaling
    return points_3d[:, :2] * scale


def reprojection_error(points_3d, points_2d_target, scale=1.0):
    """
    Calculate error between projected 3D and target 2D
    
    Args:
        points_3d: (N, 3) 3D points
        points_2d_target: (N, 2) target 2D positions
        scale: projection scale
    
    Returns:
        error: sum of squared distances
    """
    projected = weak_perspective_projection(points_3d, scale)
    error = np.sum((projected - points_2d_target) ** 2)
    return error


def bone_length_preservation_loss(points_3d, bone_lengths_target, bone_pairs):
    """
    Penalize deviation from expected bone lengths
    
    Args:
        points_3d: (N, 3) current 3D points
        bone_lengths_target: list of target bone lengths
        bone_pairs: list of (j1, j2) bone connections
    
    Returns:
        loss: sum of squared bone length errors
    """
    loss = 0
    for (j1, j2), target_length in zip(bone_pairs, bone_lengths_target):
        current_length = np.linalg.norm(points_3d[j2] - points_3d[j1])
        loss += (current_length - target_length) ** 2
    return loss


def refine_3d_pose(pose_3d_initial, keypoints_2d, bone_pairs, 
                   lambda_reproj=1.0, lambda_bone=0.1):
    """
    Refine 3D pose using 2D constraints
    
    Args:
        pose_3d_initial: (17, 3) initial 3D pose estimate
        keypoints_2d: (17, 2) accurate 2D keypoints
        bone_pairs: list of (j1, j2) bone connections
        lambda_reproj: weight for reprojection loss
        lambda_bone: weight for bone length preservation
    
    Returns:
        pose_3d_refined: (17, 3) refined 3D pose
    """
    # Calculate target bone lengths from initial estimate
    bone_lengths_target = []
    for j1, j2 in bone_pairs:
        length = np.linalg.norm(pose_3d_initial[j2] - pose_3d_initial[j1])
        bone_lengths_target.append(length)
    
    # Estimate projection scale from initial pose
    # (ratio of 2D to 3D coordinates)
    scale = np.mean(keypoints_2d[:, :2]) / np.mean(pose_3d_initial[:, :2])
    
    def objective(pose_flat):
        """Objective function to minimize"""
        pose_3d = pose_flat.reshape(17, 3)
        
        # Reprojection loss (3D→2D must match observed 2D)
        reproj_loss = reprojection_error(pose_3d, keypoints_2d, scale)
        
        # Bone length preservation (maintain anatomical structure)
        bone_loss = bone_length_preservation_loss(
            pose_3d, bone_lengths_target, bone_pairs
        )
        
        total_loss = lambda_reproj * reproj_loss + lambda_bone * bone_loss
        return total_loss
    
    # Optimize
    x0 = pose_3d_initial.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', 
                     options={'maxiter': 100})
    
    pose_3d_refined = result.x.reshape(17, 3)
    
    return pose_3d_refined, result.fun


# ============================================================================
# Demonstration
# ============================================================================

print("=" * 80)
print("2D-CONSTRAINED 3D POSE REFINEMENT")
print("=" * 80)

# Load data
print("\n📂 Loading data...")
magf_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_3D_magf.npz')
rtm_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_2D_rtm.npz')

magf_poses = magf_data['poses_3d']  # (120, 17, 3)
rtm_2d = rtm_data['keypoints']      # (360, 17, 2)

print(f"✅ MAGF 3D: {magf_poses.shape}")
print(f"✅ RTM 2D:  {rtm_2d.shape}")

# Define skeleton bones (H36M format)
bone_pairs = [
    # Legs
    (1, 2), (2, 3),  # RHip-RKnee-RAnkle
    (4, 5), (5, 6),  # LHip-LKnee-LAnkle
    # Arms
    (14, 15), (15, 16),  # RShoulder-RElbow-RWrist
    (11, 12), (12, 13),  # LShoulder-LElbow-LWrist
    # Torso
    (0, 1), (0, 4),   # Hip-RHip, Hip-LHip
    (11, 14),         # LShoulder-RShoulder
]

# Test on frame 0
frame_idx = 0
pose_3d_initial = magf_poses[frame_idx]
keypoints_2d = rtm_2d[frame_idx]

print(f"\n🔧 Refining frame {frame_idx}...")

# Calculate initial reprojection error
scale = np.mean(keypoints_2d) / np.mean(pose_3d_initial[:, :2])
initial_error = reprojection_error(pose_3d_initial, keypoints_2d, scale)

print(f"   Initial reprojection error: {initial_error:.4f}")

# Refine
pose_3d_refined, final_loss = refine_3d_pose(
    pose_3d_initial, keypoints_2d, bone_pairs,
    lambda_reproj=1.0, lambda_bone=0.5
)

# Calculate final reprojection error
final_error = reprojection_error(pose_3d_refined, keypoints_2d, scale)

print(f"   Final reprojection error:   {final_error:.4f}")
print(f"   Improvement: {(initial_error - final_error) / initial_error * 100:.1f}%")

# Compare bone lengths before/after
print("\n📏 Bone length comparison (before → after):")
print(f"{'Bone':<25s} {'Before':>10s} {'After':>10s} {'Change':>10s}")
print("-" * 60)

for j1, j2 in bone_pairs[:6]:  # Show first 6 bones
    length_before = np.linalg.norm(pose_3d_initial[j2] - pose_3d_initial[j1])
    length_after = np.linalg.norm(pose_3d_refined[j2] - pose_3d_refined[j1])
    change = length_after - length_before
    
    bone_name = f"Joint{j1}-Joint{j2}"
    print(f"{bone_name:<25s} {length_before:>10.4f} {length_after:>10.4f} {change:>10.4f}")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

print(f"""
This proof-of-concept shows 2D-constrained refinement:

✅ What it does:
- Adjusts 3D joint positions to better match observed 2D
- Preserves anatomical bone lengths
- Reduces reprojection error

⚠️ Limitations of this simple version:
- Uses weak perspective (ignores depth effects in projection)
- Simple optimization (could use gradient descent, neural networks)
- No camera parameters (assumes orthographic projection)

🎯 Production implementation would need:
1. Proper camera model (perspective projection with focal length)
2. Better optimization (PyTorch/TensorFlow with gradients)
3. Additional constraints:
   - Joint angle limits (knees don't bend backwards)
   - Temporal smoothness (for video)
   - Symmetry constraints (left/right limbs similar)
4. Multiple view consistency (if multi-camera)

📚 Related methods in literature:
- SMPLify-X: Fits SMPL model to 2D keypoints
- SPIN: Uses 2D reprojection loss in training
- HybrIK: Combines 2D detection with inverse kinematics
- EFT: Fits 3D to 2D using optimization

💡 For your use case:
You could refine MAGF or WB3D outputs using this approach,
especially to fix depth ambiguities in legs (the 40-55° knee issue).
""")

print("\n✅ Demonstration complete!\n")
