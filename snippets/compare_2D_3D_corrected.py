"""
CORRECTED 2D-3D Comparison with Proper Coordinate Handling

Issue discovered: Previous script compared 2D angles (pixel space) with 
3D angles (world space) directly. This is incorrect!

Proper approach:
1. Normalize both 2D and 3D to same scale
2. Compare angles in consistent coordinate system
3. Account for depth projection effects

Usage in Colab:
    python compare_2D_3D_corrected.py
"""

import numpy as np

# ============================================================================
# Angle Calculation Functions
# ============================================================================

def angle_between_vectors(v1, v2):
    """Calculate angle between two vectors in degrees [0, 180]"""
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    return np.degrees(angle_rad)


def calculate_angle(j1, j2, j3, keypoints):
    """Calculate angle at joint j2 formed by j1-j2-j3"""
    p1 = keypoints[j1]
    p2 = keypoints[j2]
    p3 = keypoints[j3]
    
    v1 = p1 - p2
    v2 = p3 - p2
    
    return angle_between_vectors(v1, v2)


def normalize_pose(keypoints):
    """
    Normalize pose to unit scale (root-centered, scale-normalized)
    This makes 2D and 3D comparable
    """
    # Center at root (pelvis for H36M = joint 0, or mean of hips for COCO)
    if len(keypoints.shape) == 2:  # (N, 2) or (N, 3)
        centered = keypoints - keypoints.mean(axis=0)
    else:
        centered = keypoints
    
    # Scale normalize
    scale = np.sqrt(np.sum(centered ** 2)) + 1e-8
    normalized = centered / scale
    
    return normalized


def project_3d_to_2d_orthographic(keypoints_3d):
    """
    Simple orthographic projection: just take X,Y coordinates
    This is valid for weak perspective (distant camera)
    """
    return keypoints_3d[:, :2]


# ============================================================================
# Joint Angle Definitions
# ============================================================================

RTM_MAGF_ANGLES = {
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

WB3D_ANGLES = {
    'LElbow': (5, 7, 9),
    'RElbow': (6, 8, 10),
    'LKnee': (11, 13, 15),
    'RKnee': (12, 14, 16),
    'LShoulder': (11, 5, 7),
    'RShoulder': (12, 6, 8),
    'LHip': (5, 11, 13),
    'RHip': (6, 12, 14),
    'ShoulderBridge': (5, 0, 6),
}


# ============================================================================
# Comparison Functions
# ============================================================================

def compare_2d_vs_projected_3d(keypoints_2d, keypoints_3d, angle_defs):
    """
    Compare 2D angles vs 3D projected to 2D angles
    
    This is the CORRECT way: project 3D → 2D, then compare angles
    """
    # Normalize both for fair comparison
    kp_2d_norm = normalize_pose(keypoints_2d)
    kp_3d_norm = normalize_pose(keypoints_3d)
    
    # Project 3D to 2D
    kp_3d_projected = project_3d_to_2d_orthographic(kp_3d_norm)
    
    results = {}
    
    for joint_name, (j1, j2, j3) in angle_defs.items():
        angle_2d = calculate_angle(j1, j2, j3, kp_2d_norm)
        angle_3d_projected = calculate_angle(j1, j2, j3, kp_3d_projected)
        angle_3d_native = calculate_angle(j1, j2, j3, kp_3d_norm)
        
        delta_projected = abs(angle_2d - angle_3d_projected)
        delta_native = abs(angle_2d - angle_3d_native)
        
        results[joint_name] = {
            'angle_2d': angle_2d,
            'angle_3d_projected': angle_3d_projected,
            'angle_3d_native': angle_3d_native,
            'delta_projected': delta_projected,
            'delta_native': delta_native,
        }
    
    return results


def analyze_multi_frame(keypoints_2d_all, keypoints_3d_all, angle_defs, num_frames):
    """Analyze multiple frames"""
    deltas_projected = {name: [] for name in angle_defs.keys()}
    deltas_native = {name: [] for name in angle_defs.keys()}
    
    for frame_idx in range(num_frames):
        results = compare_2d_vs_projected_3d(
            keypoints_2d_all[frame_idx],
            keypoints_3d_all[frame_idx],
            angle_defs
        )
        
        for joint_name, data in results.items():
            deltas_projected[joint_name].append(data['delta_projected'])
            deltas_native[joint_name].append(data['delta_native'])
    
    # Calculate statistics
    stats_projected = {}
    stats_native = {}
    
    for joint_name in angle_defs.keys():
        arr_proj = np.array(deltas_projected[joint_name])
        arr_nat = np.array(deltas_native[joint_name])
        
        stats_projected[joint_name] = {
            'mean': np.mean(arr_proj),
            'std': np.std(arr_proj),
            'max': np.max(arr_proj),
        }
        
        stats_native[joint_name] = {
            'mean': np.mean(arr_nat),
            'std': np.std(arr_nat),
            'max': np.max(arr_nat),
        }
    
    return stats_projected, stats_native, deltas_projected, deltas_native


def print_corrected_report(pipeline_name, stats_proj, stats_nat, deltas_proj, num_frames, threshold=5.0):
    """Print corrected report"""
    
    print("\n" + "=" * 80)
    print(f"CORRECTED ANALYSIS: {pipeline_name}")
    print("=" * 80)
    
    # Overall statistics for projected comparison (the correct metric)
    all_deltas = [d for dlist in deltas_proj.values() for d in dlist]
    overall_mean = np.mean(all_deltas)
    overall_max = np.max(all_deltas)
    
    print(f"\n📊 2D vs 3D-Projected-to-2D Angle Comparison:")
    print(f"   Mean angle difference:    {overall_mean:6.2f}°")
    print(f"   Max angle difference:     {overall_max:6.2f}°")
    
    print(f"\n📏 Per-Joint Results (2D vs 3D projected to 2D):")
    print(f"{'Joint':<15s} {'Mean Δ':>10s} {'Std':>10s} {'Max Δ':>10s}")
    print("-" * 80)
    
    sorted_joints = sorted(stats_proj.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    for joint_name, data in sorted_joints:
        print(f"{joint_name:<15s} "
              f"{data['mean']:>9.2f}° "
              f"{data['std']:>9.2f}° "
              f"{data['max']:>9.2f}°")
    
    # Count frames above threshold
    frames_above = {}
    for joint_name, delta_list in deltas_proj.items():
        count = sum(1 for d in delta_list if d > threshold)
        frames_above[joint_name] = count
    
    total_refinements = sum(frames_above.values())
    total_possible = num_frames * len(stats_proj)
    
    print(f"\n🚨 Frames Exceeding {threshold}° Threshold:")
    print(f"{'Joint':<15s} {'Count':>10s} {'Percentage':>12s}")
    print("-" * 80)
    
    for joint_name in sorted(frames_above.keys(), key=lambda x: frames_above[x], reverse=True):
        count = frames_above[joint_name]
        pct = (count / num_frames) * 100
        print(f"{joint_name:<15s} {count:>10d} {pct:>11.1f}%")
    
    print(f"\n💡 Refinement Summary:")
    print(f"   Total joint-frames needing refinement: {total_refinements}/{total_possible} ({total_refinements/total_possible*100:.1f}%)")
    print(f"   Computation saved by selective:        {100 - total_refinements/total_possible*100:.1f}%")
    
    # Quality assessment
    excellent = sum(1 for s in stats_proj.values() if s['mean'] < 3)
    good = sum(1 for s in stats_proj.values() if 3 <= s['mean'] < 5)
    moderate = sum(1 for s in stats_proj.values() if 5 <= s['mean'] < 10)
    poor = sum(1 for s in stats_proj.values() if 10 <= s['mean'] < 20)
    critical = sum(1 for s in stats_proj.values() if s['mean'] >= 20)
    
    print(f"\n✅ Quality Distribution:")
    print(f"   Excellent (<3°):   {excellent}")
    print(f"   Good (3-5°):       {good}")
    print(f"   Moderate (5-10°):  {moderate}")
    print(f"   Poor (10-20°):     {poor}")
    print(f"   Critical (>20°):   {critical}")


# ============================================================================
# Main Analysis
# ============================================================================

print("=" * 80)
print("CORRECTED 2D-3D JOINT ANGLE COMPARISON")
print("=" * 80)

print("""
🔧 CORRECTION APPLIED:

Previous script compared:
  ❌ 2D angles (in pixel space) vs 3D angles (in 3D space)
  
This script compares:
  ✅ 2D angles vs 3D-projected-to-2D angles (same coordinate system)
  
Both are normalized to unit scale for fair comparison.
""")

NUM_FRAMES = 30
THRESHOLD = 5.0

# ============================================================================
# PIPELINE 1: RTM → MAGF
# ============================================================================

print("\n" + "=" * 80)
print("PIPELINE 1: RTM → MAGF")
print("=" * 80)

try:
    rtm_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_2D_rtm.npz')
    magf_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_3D_magf.npz')
    
    rtm_2d = rtm_data['keypoints']
    magf_3d = magf_data['poses_3d']
    
    print(f"✅ Loaded: RTM 2D {rtm_2d.shape}, MAGF 3D {magf_3d.shape}")
    
    num_frames_p1 = min(NUM_FRAMES, len(magf_3d))
    
    stats_proj_p1, stats_nat_p1, deltas_proj_p1, deltas_nat_p1 = analyze_multi_frame(
        rtm_2d[:num_frames_p1],
        magf_3d[:num_frames_p1],
        RTM_MAGF_ANGLES,
        num_frames_p1
    )
    
    print_corrected_report("RTM → MAGF", stats_proj_p1, stats_nat_p1, 
                          deltas_proj_p1, num_frames_p1, THRESHOLD)
    
    pipeline1_success = True

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    pipeline1_success = False


# ============================================================================
# PIPELINE 2: WB3D
# ============================================================================

print("\n" + "=" * 80)
print("PIPELINE 2: WB3D")
print("=" * 80)

try:
    wb_2d_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_2D_wb.npz')
    wb_3d_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_3D_wb.npz')
    
    wb_2d = wb_2d_data['keypoints'][:, :17, :]
    wb_3d = wb_3d_data['keypoints_3d'][:, :17, :]
    
    print(f"✅ Loaded: WB3D 2D {wb_2d.shape}, WB3D 3D {wb_3d.shape}")
    
    num_frames_p2 = min(NUM_FRAMES, len(wb_3d))
    
    stats_proj_p2, stats_nat_p2, deltas_proj_p2, deltas_nat_p2 = analyze_multi_frame(
        wb_2d[:num_frames_p2],
        wb_3d[:num_frames_p2],
        WB3D_ANGLES,
        num_frames_p2
    )
    
    print_corrected_report("WB3D", stats_proj_p2, stats_nat_p2, 
                          deltas_proj_p2, num_frames_p2, THRESHOLD)
    
    pipeline2_success = True

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    pipeline2_success = False


# ============================================================================
# COMPARATIVE SUMMARY
# ============================================================================

if pipeline1_success and pipeline2_success:
    print("\n" + "=" * 80)
    print("COMPARATIVE SUMMARY")
    print("=" * 80)
    
    all_p1 = [d for dlist in deltas_proj_p1.values() for d in dlist]
    all_p2 = [d for dlist in deltas_proj_p2.values() for d in dlist]
    
    mean_p1 = np.mean(all_p1)
    mean_p2 = np.mean(all_p2)
    
    print(f"\n📊 Overall 2D-3D Consistency (Corrected):")
    print(f"   RTM+MAGF:  {mean_p1:6.2f}° mean difference")
    print(f"   WB3D:      {mean_p2:6.2f}° mean difference")
    
    if mean_p1 < mean_p2:
        print(f"   ✅ RTM+MAGF is more 2D-consistent")
    elif mean_p2 < mean_p1:
        print(f"   ✅ WB3D is more 2D-consistent")
    else:
        print(f"   Both pipelines have equal consistency")
    
    print(f"\n📏 Per-Joint Winner:")
    for joint_name in sorted(set(stats_proj_p1.keys()) & set(stats_proj_p2.keys())):
        mean1 = stats_proj_p1[joint_name]['mean']
        mean2 = stats_proj_p2[joint_name]['mean']
        winner = "RTM+MAGF" if mean1 < mean2 else "WB3D"
        diff = abs(mean1 - mean2)
        print(f"   {joint_name:<15s}: {winner:<10s} (Δ = {diff:.2f}°)")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

print("""
✅ **This corrected analysis properly handles coordinate systems!**

Key differences from previous (incorrect) version:
1. Both 2D and 3D are normalized to unit scale
2. 3D is projected to 2D before angle comparison
3. Fair "apples-to-apples" comparison

If results still show large differences, it indicates:
- Depth ambiguity (Z-coordinate uncertainty)
- Different pose interpretations
- Need for refinement

If WB3D still shows 0.00°, it confirms that WB3D internally
enforces 2D-3D consistency (by design).
""")

print("\n✅ Analysis complete!\n")
