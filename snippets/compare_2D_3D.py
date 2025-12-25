"""
2D-3D Joint Angle Comparison for Both Pipelines

Compare joint angles between 2D keypoints and their corresponding 3D estimates
for both pipelines:
1. RTM (2D) → MAGF (3D lifting)
2. WB3D (2D + 3D in one model)

This diagnostic report identifies which pipeline and which joints need refinement.

Usage in Colab:
    python compare_2D_3D.py
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
    """
    Calculate angle at joint j2 formed by j1-j2-j3
    
    Args:
        j1, j2, j3: joint indices
        keypoints: (N, 2) or (N, 3) array of keypoints
    
    Returns:
        angle: angle at j2 in degrees
    """
    p1 = keypoints[j1]
    p2 = keypoints[j2]
    p3 = keypoints[j3]
    
    v1 = p1 - p2  # Vector from j2 to j1
    v2 = p3 - p2  # Vector from j2 to j3
    
    return angle_between_vectors(v1, v2)


# ============================================================================
# Joint Angle Definitions (H36M-17 format)
# ============================================================================

# For RTM → MAGF (both use H36M-17 format directly)
RTM_MAGF_ANGLES = {
    'LElbow': (11, 12, 13),     # LShoulder-LElbow-LWrist
    'RElbow': (14, 15, 16),     # RShoulder-RElbow-RWrist
    'LKnee': (4, 5, 6),          # LHip-LKnee-LAnkle
    'RKnee': (1, 2, 3),          # RHip-RKnee-RAnkle
    'LShoulder': (0, 11, 12),    # Hip-LShoulder-LElbow
    'RShoulder': (0, 14, 15),    # Hip-RShoulder-RElbow
    'LHip': (11, 0, 4),          # LShoulder-Hip-LHip
    'RHip': (14, 0, 1),          # RShoulder-Hip-RHip
    'ShoulderBridge': (11, 0, 14), # LShoulder-Hip-RShoulder
}

# For WB3D (2D uses COCO-133, 3D uses COCO-133, need to extract body joints)
# COCO body joints: 0-16 (first 17 of 133)
# Map COCO-17 body joints to H36M-17 equivalent angles

WB3D_ANGLES = {
    # COCO indices for body joints (0-16 of the 133)
    'LElbow': (5, 7, 9),        # left_shoulder-left_elbow-left_wrist
    'RElbow': (6, 8, 10),       # right_shoulder-right_elbow-right_wrist
    'LKnee': (11, 13, 15),      # left_hip-left_knee-left_ankle
    'RKnee': (12, 14, 16),      # right_hip-right_knee-right_ankle
    'LShoulder': (11, 5, 7),    # left_hip-left_shoulder-left_elbow
    'RShoulder': (12, 6, 8),    # right_hip-right_shoulder-right_elbow
    'LHip': (5, 11, 13),        # left_shoulder-left_hip-left_knee
    'RHip': (6, 12, 14),        # right_shoulder-right_hip-right_knee
    'ShoulderBridge': (5, 0, 6), # left_shoulder-nose-right_shoulder (approx)
}


# ============================================================================
# Comparison Functions
# ============================================================================

def compare_angles_single_frame(keypoints_2d, keypoints_3d, angle_defs):
    """
    Compare 2D vs 3D angles for a single frame
    
    Args:
        keypoints_2d: (N, 2) 2D keypoints
        keypoints_3d: (N, 3) 3D keypoints
        angle_defs: dict of angle definitions
    
    Returns:
        results: dict of {joint_name: (angle_2d, angle_3d, delta)}
    """
    results = {}
    
    for joint_name, (j1, j2, j3) in angle_defs.items():
        angle_2d = calculate_angle(j1, j2, j3, keypoints_2d)
        angle_3d = calculate_angle(j1, j2, j3, keypoints_3d)
        delta = abs(angle_2d - angle_3d)
        
        results[joint_name] = {
            'angle_2d': angle_2d,
            'angle_3d': angle_3d,
            'delta': delta
        }
    
    return results


def compare_angles_multi_frame(keypoints_2d_all, keypoints_3d_all, angle_defs, num_frames):
    """
    Compare 2D vs 3D angles across multiple frames
    
    Returns:
        stats: dict with per-joint statistics
    """
    joint_names = list(angle_defs.keys())
    
    # Collect deltas for all frames
    deltas = {name: [] for name in joint_names}
    
    for frame_idx in range(num_frames):
        frame_results = compare_angles_single_frame(
            keypoints_2d_all[frame_idx],
            keypoints_3d_all[frame_idx],
            angle_defs
        )
        
        for joint_name, data in frame_results.items():
            deltas[joint_name].append(data['delta'])
    
    # Calculate statistics
    stats = {}
    for joint_name in joint_names:
        delta_array = np.array(deltas[joint_name])
        stats[joint_name] = {
            'mean': np.mean(delta_array),
            'std': np.std(delta_array),
            'min': np.min(delta_array),
            'max': np.max(delta_array),
            'median': np.median(delta_array),
        }
    
    return stats, deltas


def count_frames_above_threshold(deltas, threshold):
    """Count how many frames have delta > threshold for each joint"""
    counts = {}
    for joint_name, delta_list in deltas.items():
        count = sum(1 for d in delta_list if d > threshold)
        counts[joint_name] = count
    return counts


# ============================================================================
# Report Generation
# ============================================================================

def print_pipeline_report(pipeline_name, stats, deltas, num_frames, threshold=5.0):
    """Print comprehensive report for one pipeline"""
    
    print("\n" + "=" * 80)
    print(f"PIPELINE: {pipeline_name}")
    print("=" * 80)
    
    # Overall summary
    all_deltas = []
    for delta_list in deltas.values():
        all_deltas.extend(delta_list)
    overall_mean = np.mean(all_deltas)
    overall_max = np.max(all_deltas)
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Mean angle difference:    {overall_mean:6.2f}°")
    print(f"   Max angle difference:     {overall_max:6.2f}°")
    print(f"   Frames analyzed:          {num_frames}")
    print(f"   Joint angles per frame:   {len(stats)}")
    print(f"   Total comparisons:        {num_frames * len(stats)}")
    
    # Per-joint statistics
    print(f"\n📏 Per-Joint Angle Differences (2D vs 3D):")
    print(f"{'Joint':<15s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'Median':>8s}")
    print("-" * 80)
    
    # Sort by mean difference (worst first)
    sorted_joints = sorted(stats.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    for joint_name, data in sorted_joints:
        print(f"{joint_name:<15s} "
              f"{data['mean']:>7.2f}° "
              f"{data['std']:>7.2f}° "
              f"{data['min']:>7.2f}° "
              f"{data['max']:>7.2f}° "
              f"{data['median']:>7.2f}°")
    
    # Threshold analysis
    counts = count_frames_above_threshold(deltas, threshold)
    
    print(f"\n🚨 Frames Exceeding {threshold}° Threshold:")
    print(f"{'Joint':<15s} {'Frames':>10s} {'Percentage':>12s}")
    print("-" * 80)
    
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    for joint_name, count in sorted_counts:
        percentage = (count / num_frames) * 100
        print(f"{joint_name:<15s} {count:>10d} {percentage:>11.1f}%")
    
    # Categorize joints
    print(f"\n✅ Joint Quality Assessment:")
    
    excellent = []  # < 3° mean
    good = []       # 3-5°
    moderate = []   # 5-10°
    poor = []       # 10-20°
    critical = []   # > 20°
    
    for joint_name, data in stats.items():
        mean_diff = data['mean']
        if mean_diff < 3:
            excellent.append(joint_name)
        elif mean_diff < 5:
            good.append(joint_name)
        elif mean_diff < 10:
            moderate.append(joint_name)
        elif mean_diff < 20:
            poor.append(joint_name)
        else:
            critical.append(joint_name)
    
    print(f"   Excellent (< 3°):   {len(excellent):2d} joints - {', '.join(excellent) if excellent else 'None'}")
    print(f"   Good (3-5°):        {len(good):2d} joints - {', '.join(good) if good else 'None'}")
    print(f"   Moderate (5-10°):   {len(moderate):2d} joints - {', '.join(moderate) if moderate else 'None'}")
    print(f"   Poor (10-20°):      {len(poor):2d} joints - {', '.join(poor) if poor else 'None'}")
    print(f"   Critical (> 20°):   {len(critical):2d} joints - {', '.join(critical) if critical else 'None'}")
    
    # Refinement recommendation
    print(f"\n💡 Refinement Recommendation:")
    
    needs_refinement = [name for name, count in counts.items() if count > 0]
    total_refinements_needed = sum(counts.values())
    total_possible = num_frames * len(stats)
    refinement_percentage = (total_refinements_needed / total_possible) * 100
    
    print(f"   Joints needing refinement:     {len(needs_refinement)}/{len(stats)}")
    print(f"   Total joint-frames to refine:  {total_refinements_needed}/{total_possible} ({refinement_percentage:.1f}%)")
    print(f"   Computation saved by selective: {100 - refinement_percentage:.1f}%")
    
    if critical or poor:
        print(f"   ⚠️  Priority: {', '.join(critical + poor)}")
    else:
        print(f"   ✅ All joints within acceptable range")


# ============================================================================
# Main Analysis
# ============================================================================

print("=" * 80)
print("2D-3D JOINT ANGLE COMPARISON: RTM+MAGF vs WB3D")
print("=" * 80)

# Configuration
NUM_FRAMES = 30
THRESHOLD = 5.0  # degrees

print(f"\n⚙️  Configuration:")
print(f"   Frames to analyze:     {NUM_FRAMES}")
print(f"   Refinement threshold:  {THRESHOLD}°")

# ============================================================================
# PIPELINE 1: RTM (2D) → MAGF (3D)
# ============================================================================

print("\n" + "=" * 80)
print("LOADING PIPELINE 1: RTM → MAGF")
print("=" * 80)

try:
    rtm_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_2D_rtm.npz')
    magf_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_3D_magf.npz')
    
    # Get data
    rtm_2d = rtm_data['keypoints']       # (360, 17, 2)
    magf_3d = magf_data['poses_3d']      # (120, 17, 3)
    
    print(f"✅ RTM 2D:  {rtm_2d.shape}")
    print(f"✅ MAGF 3D: {magf_3d.shape}")
    
    # Use first NUM_FRAMES (MAGF has fewer frames due to temporal window)
    num_frames_p1 = min(NUM_FRAMES, len(magf_3d))
    print(f"   Analyzing first {num_frames_p1} frames")
    
    # Compare angles
    stats_p1, deltas_p1 = compare_angles_multi_frame(
        rtm_2d[:num_frames_p1],
        magf_3d[:num_frames_p1],
        RTM_MAGF_ANGLES,
        num_frames_p1
    )
    
    # Print report
    print_pipeline_report("RTM → MAGF", stats_p1, deltas_p1, num_frames_p1, THRESHOLD)
    
    pipeline1_success = True

except Exception as e:
    print(f"❌ Error loading Pipeline 1: {e}")
    pipeline1_success = False


# ============================================================================
# PIPELINE 2: WB3D (2D + 3D)
# ============================================================================

print("\n" + "=" * 80)
print("LOADING PIPELINE 2: WB3D")
print("=" * 80)

try:
    wb_2d_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_2D_wb.npz')
    wb_3d_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_3D_wb.npz')
    
    # Get data (WB3D has 133 joints, we want body joints 0-16)
    wb_2d_all = wb_2d_data['keypoints']      # (360, 133, 2)
    wb_3d_all = wb_3d_data['keypoints_3d']   # (360, 133, 3)
    
    # Extract body joints (first 17 of 133)
    wb_2d = wb_2d_all[:, :17, :]  # (360, 17, 2)
    wb_3d = wb_3d_all[:, :17, :]  # (360, 17, 3)
    
    print(f"✅ WB3D 2D: {wb_2d.shape} (extracted body joints from {wb_2d_all.shape})")
    print(f"✅ WB3D 3D: {wb_3d.shape} (extracted body joints from {wb_3d_all.shape})")
    
    # Use first NUM_FRAMES
    num_frames_p2 = min(NUM_FRAMES, len(wb_3d))
    print(f"   Analyzing first {num_frames_p2} frames")
    
    # Compare angles
    stats_p2, deltas_p2 = compare_angles_multi_frame(
        wb_2d[:num_frames_p2],
        wb_3d[:num_frames_p2],
        WB3D_ANGLES,
        num_frames_p2
    )
    
    # Print report
    print_pipeline_report("WB3D", stats_p2, deltas_p2, num_frames_p2, THRESHOLD)
    
    pipeline2_success = True

except Exception as e:
    print(f"❌ Error loading Pipeline 2: {e}")
    pipeline2_success = False


# ============================================================================
# COMPARATIVE SUMMARY
# ============================================================================

if pipeline1_success and pipeline2_success:
    print("\n" + "=" * 80)
    print("COMPARATIVE SUMMARY: RTM+MAGF vs WB3D")
    print("=" * 80)
    
    # Overall comparison
    all_deltas_p1 = [d for dlist in deltas_p1.values() for d in dlist]
    all_deltas_p2 = [d for dlist in deltas_p2.values() for d in dlist]
    
    mean_p1 = np.mean(all_deltas_p1)
    mean_p2 = np.mean(all_deltas_p2)
    
    print(f"\n📊 Overall 2D-3D Consistency:")
    print(f"{'Pipeline':<20s} {'Mean Δ':>10s} {'Max Δ':>10s} {'Winner':>10s}")
    print("-" * 80)
    print(f"{'RTM → MAGF':<20s} {mean_p1:>9.2f}° {np.max(all_deltas_p1):>9.2f}° "
          f"{'✅' if mean_p1 < mean_p2 else ''}")
    print(f"{'WB3D':<20s} {mean_p2:>9.2f}° {np.max(all_deltas_p2):>9.2f}° "
          f"{'✅' if mean_p2 < mean_p1 else ''}")
    
    # Per-joint comparison
    print(f"\n📏 Per-Joint Comparison (Mean Angle Difference):")
    print(f"{'Joint':<15s} {'RTM+MAGF':>12s} {'WB3D':>12s} {'Better':>10s}")
    print("-" * 80)
    
    all_joints = set(stats_p1.keys()) | set(stats_p2.keys())
    
    for joint_name in sorted(all_joints):
        if joint_name in stats_p1 and joint_name in stats_p2:
            mean1 = stats_p1[joint_name]['mean']
            mean2 = stats_p2[joint_name]['mean']
            better = "RTM+MAGF" if mean1 < mean2 else "WB3D"
            print(f"{joint_name:<15s} {mean1:>11.2f}° {mean2:>11.2f}° {better:>10s}")
    
    # Refinement needs
    counts_p1 = count_frames_above_threshold(deltas_p1, THRESHOLD)
    counts_p2 = count_frames_above_threshold(deltas_p2, THRESHOLD)
    
    total_refine_p1 = sum(counts_p1.values())
    total_refine_p2 = sum(counts_p2.values())
    
    print(f"\n🔧 Refinement Workload:")
    print(f"   RTM+MAGF needs: {total_refine_p1} joint-frame refinements")
    print(f"   WB3D needs:     {total_refine_p2} joint-frame refinements")
    
    if total_refine_p1 < total_refine_p2:
        print(f"   ✅ RTM+MAGF is more 2D-consistent ({total_refine_p2 - total_refine_p1} fewer refinements)")
    elif total_refine_p2 < total_refine_p1:
        print(f"   ✅ WB3D is more 2D-consistent ({total_refine_p1 - total_refine_p2} fewer refinements)")
    else:
        print(f"   Both pipelines need equal refinement")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)

print(f"""
Based on this diagnostic report:

1️⃣  **Identify priority joints**: Focus on joints with mean Δ > {THRESHOLD}°

2️⃣  **Choose pipeline**: 
   - If RTM+MAGF is better → Refine RTM+MAGF output
   - If WB3D is better → Refine WB3D output
   - Or refine both for ensemble

3️⃣  **Apply selective refinement**: Use selective_refinement.py on problematic joints

4️⃣  **Validate improvement**: Re-run this script after refinement to verify

💡 **Key insight**: 
   2D-3D angle consistency measures how well the 3D estimate "agrees" with
   the observed 2D pose. Lower delta = more consistent = more trustworthy.
   
   This is different from our previous MAGF vs WB3D comparison, which
   compared two 3D methods against each other. Now we're comparing each
   method against its own 2D input (ground truth).
""")

print("\n✅ Analysis complete!\n")
