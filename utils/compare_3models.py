"""
Compare keypoints from RTMPose, RTMPose-Halpe26, and ViTPose NPZ files
- RTMPose: 17 keypoints (COCO)
- RTMPose-Halpe26: 26 keypoints (COCO + feet + extra body)
- ViTPose: 17 keypoints (COCO)

For RTM vs Halpe26 comparison, uses only first 17 keypoints for fair comparison

Usage:
    python compare_3models.py <rtm_npz> <halpe26_npz> <vit_npz>
"""

import numpy as np
from pathlib import Path
import sys

# COCO joint names (first 17 keypoints)
JOINT_NAMES = [
    "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hip", "R_Hip",
    "L_Knee", "R_Knee", "L_Ankle", "R_Ankle"
]

# Halpe26 extra keypoints (17-25)
HALPE26_EXTRA_NAMES = [
    "Head", "Neck", "Hip",
    "L_BigToe", "R_BigToe", "L_SmallToe",
    "R_SmallToe", "L_Heel", "R_Heel"
]


def compare_two_models(name1, kpts1, scores1, name2, kpts2, scores2, num_joints=17):
    """Compare two pose estimation models"""
    
    print("\n" + "=" * 70)
    print(f"🔍 COMPARISON: {name1} vs {name2}")
    print("=" * 70 + "\n")
    
    # Use only first num_joints for comparison
    kpts1_coco = kpts1[:, :num_joints, :]
    scores1_coco = scores1[:, :num_joints]
    kpts2_coco = kpts2[:, :num_joints, :]
    scores2_coco = scores2[:, :num_joints]
    
    # Confidence scores comparison
    print("📈 Confidence Scores (17 COCO keypoints):")
    avg_conf1 = np.mean(scores1_coco)
    avg_conf2 = np.mean(scores2_coco)
    
    print(f"   {name1:<20} Avg: {avg_conf1:.3f} | Min: {np.min(scores1_coco):.3f} | Max: {np.max(scores1_coco):.3f}")
    print(f"   {name2:<20} Avg: {avg_conf2:.3f} | Min: {np.min(scores2_coco):.3f} | Max: {np.max(scores2_coco):.3f}")
    print(f"   Difference: {avg_conf1 - avg_conf2:+.3f}")
    
    # Per-joint confidence
    print(f"\n🦴 Per-Joint Average Confidence (17 COCO keypoints):")
    joint_conf1 = np.mean(scores1_coco, axis=0)
    joint_conf2 = np.mean(scores2_coco, axis=0)
    
    print(f"   {'Joint':<15} {name1[:10]:<10} {name2[:10]:<10} {'Diff':<10}")
    print("   " + "-" * 50)
    for i, name in enumerate(JOINT_NAMES):
        diff = joint_conf1[i] - joint_conf2[i]
        symbol = "🟢" if abs(diff) < 0.05 else ("🔴" if abs(diff) > 0.1 else "🟡")
        print(f"   {name:<15} {joint_conf1[i]:.3f}      {joint_conf2[i]:.3f}      {diff:+.3f} {symbol}")
    
    # Spatial differences (Euclidean distance)
    print(f"\n📏 Spatial Differences (17 COCO keypoints):")
    distances = np.sqrt(np.sum((kpts1_coco - kpts2_coco) ** 2, axis=2))  # (frames, 17)
    avg_distances = np.mean(distances, axis=1)  # Average across joints per frame
    
    print(f"   Mean distance: {np.mean(avg_distances):.2f} pixels")
    print(f"   Median distance: {np.median(avg_distances):.2f} pixels")
    print(f"   Min distance: {np.min(avg_distances):.2f} pixels")
    print(f"   Max distance: {np.max(avg_distances):.2f} pixels")
    print(f"   Std deviation: {np.std(avg_distances):.2f} pixels")
    
    # Per-joint spatial differences
    print(f"\n🎯 Per-Joint Spatial Differences:")
    joint_distances = np.mean(distances, axis=0)  # Average across frames per joint
    
    print(f"   {'Joint':<15} {'Avg Distance':<15} {'Assessment'}")
    print("   " + "-" * 50)
    for i, name in enumerate(JOINT_NAMES):
        dist = joint_distances[i]
        if dist < 5:
            assessment = "✅ Very close"
        elif dist < 10:
            assessment = "🟢 Close"
        elif dist < 20:
            assessment = "🟡 Moderate"
        else:
            assessment = "🔴 Large"
        print(f"   {name:<15} {dist:<15.2f} {assessment}")
    
    # Overall assessment
    print(f"\n📋 Summary:")
    avg_spatial_diff = np.mean(avg_distances)
    conf_diff = abs(avg_conf1 - avg_conf2)
    
    print(f"   Confidence Agreement: ", end="")
    if conf_diff < 0.05:
        print("✅ Excellent (diff < 0.05)")
    elif conf_diff < 0.10:
        print("🟢 Good (diff < 0.10)")
    else:
        print("🟡 Moderate (diff >= 0.10)")
    
    print(f"   Spatial Agreement: ", end="")
    if avg_spatial_diff < 5:
        print("✅ Excellent (< 5 pixels)")
    elif avg_spatial_diff < 10:
        print("🟢 Good (< 10 pixels)")
    elif avg_spatial_diff < 20:
        print("🟡 Moderate (< 20 pixels)")
    else:
        print("🔴 Large differences (>= 20 pixels)")
    
    print(f"   {name1} confidence: {'🏆 Higher' if avg_conf1 > avg_conf2 else '   Lower'} ({avg_conf1:.3f})")
    print(f"   {name2} confidence: {'🏆 Higher' if avg_conf2 > avg_conf1 else '   Lower'} ({avg_conf2:.3f})")
    
    return {
        'avg_spatial_diff': avg_spatial_diff,
        'conf_diff': conf_diff,
        'avg_conf1': avg_conf1,
        'avg_conf2': avg_conf2
    }


def analyze_halpe26_extra_keypoints(halpe26_kpts, halpe26_scores):
    """Analyze the extra 9 keypoints in Halpe26"""
    
    print("\n" + "=" * 70)
    print("🦶 HALPE26 EXTRA KEYPOINTS ANALYSIS (Points 17-25)")
    print("=" * 70 + "\n")
    
    # Extract extra keypoints (indices 17-25)
    extra_kpts = halpe26_kpts[:, 17:26, :]
    extra_scores = halpe26_scores[:, 17:26]
    
    print("📊 Extra Keypoints Statistics:")
    avg_conf = np.mean(extra_scores)
    print(f"   Overall Avg Confidence: {avg_conf:.3f}")
    print(f"   Min Confidence: {np.min(extra_scores):.3f}")
    print(f"   Max Confidence: {np.max(extra_scores):.3f}")
    
    print(f"\n🎯 Per-Keypoint Confidence:")
    print(f"   {'Keypoint':<15} {'Avg Conf':<12} {'Min':<8} {'Max':<8} {'Assessment'}")
    print("   " + "-" * 60)
    
    for i, name in enumerate(HALPE26_EXTRA_NAMES):
        avg = np.mean(extra_scores[:, i])
        min_val = np.min(extra_scores[:, i])
        max_val = np.max(extra_scores[:, i])
        
        if avg > 0.8:
            assessment = "✅ Excellent"
        elif avg > 0.6:
            assessment = "🟢 Good"
        elif avg > 0.4:
            assessment = "🟡 Fair"
        else:
            assessment = "🔴 Poor"
        
        print(f"   {name:<15} {avg:<12.3f} {min_val:<8.3f} {max_val:<8.3f} {assessment}")
    
    # Feet vs Body extra keypoints
    feet_scores = extra_scores[:, 3:9]  # Indices 20-25 (feet)
    body_scores = extra_scores[:, 0:3]  # Indices 17-19 (head, neck, hip)
    
    print(f"\n📍 Group Statistics:")
    print(f"   Body Extra (Head, Neck, Hip):  {np.mean(body_scores):.3f}")
    print(f"   Feet (6 keypoints):             {np.mean(feet_scores):.3f}")


def main():
    if len(sys.argv) != 4:
        print("Usage: python compare_3models.py <rtm_npz> <halpe26_npz> <vit_npz>")
        print("\nExample:")
        print("  python compare_3models.py keypoints_rtm.npz keypoints_halpe26.npz keypoints_vit.npz")
        sys.exit(1)
    
    rtm_path = Path(sys.argv[1])
    halpe_path = Path(sys.argv[2])
    vit_path = Path(sys.argv[3])
    
    # Check files exist
    for path, name in [(rtm_path, "RTMPose"), (halpe_path, "Halpe26"), (vit_path, "ViTPose")]:
        if not path.exists():
            print(f"❌ Error: {name} NPZ not found: {path}")
            sys.exit(1)
    
    # Load NPZ files
    print("\n" + "🎬" * 35)
    print("3-MODEL POSE COMPARISON")
    print("🎬" * 35 + "\n")
    
    print("📦 Loading NPZ files...")
    rtm = np.load(rtm_path)
    halpe = np.load(halpe_path)
    vit = np.load(vit_path)
    
    rtm_kpts = rtm['keypoints']
    rtm_scores = rtm['scores']
    
    halpe_kpts = halpe['keypoints']
    halpe_scores = halpe['scores']
    
    vit_kpts = vit['keypoints']
    vit_scores = vit['scores']
    
    # Basic info
    print(f"\n📋 Dataset Info:")
    print(f"   RTMPose:        {len(rtm['frame_numbers'])} frames, {rtm_kpts.shape[1]} keypoints")
    print(f"   RTMPose-Halpe26: {len(halpe['frame_numbers'])} frames, {halpe_kpts.shape[1]} keypoints")
    print(f"   ViTPose:        {len(vit['frame_numbers'])} frames, {vit_kpts.shape[1]} keypoints")
    
    # Comparison 1: RTMPose vs Halpe26 (first 17 keypoints only)
    results_rtm_halpe = compare_two_models(
        "RTMPose", rtm_kpts, rtm_scores,
        "RTMPose-Halpe26", halpe_kpts, halpe_scores,
        num_joints=17
    )
    
    # Comparison 2: RTMPose vs ViTPose
    results_rtm_vit = compare_two_models(
        "RTMPose", rtm_kpts, rtm_scores,
        "ViTPose", vit_kpts, vit_scores,
        num_joints=17
    )
    
    # Comparison 3: Halpe26 vs ViTPose (first 17 keypoints)
    results_halpe_vit = compare_two_models(
        "RTMPose-Halpe26", halpe_kpts, halpe_scores,
        "ViTPose", vit_kpts, vit_scores,
        num_joints=17
    )
    
    # Analyze Halpe26 extra keypoints
    analyze_halpe26_extra_keypoints(halpe_kpts, halpe_scores)
    
    # Final summary
    print("\n" + "=" * 70)
    print("🏆 FINAL SUMMARY - OVERALL RANKINGS")
    print("=" * 70 + "\n")
    
    print("📊 Average Confidence (17 COCO keypoints):")
    confs = [
        ("RTMPose", results_rtm_halpe['avg_conf1']),
        ("RTMPose-Halpe26", results_rtm_halpe['avg_conf2']),
        ("ViTPose", results_rtm_vit['avg_conf2'])
    ]
    confs_sorted = sorted(confs, key=lambda x: x[1], reverse=True)
    for i, (name, conf) in enumerate(confs_sorted, 1):
        medal = "🥇" if i == 1 else ("🥈" if i == 2 else "🥉")
        print(f"   {medal} {i}. {name:<20} {conf:.3f}")
    
    print("\n🎯 Spatial Agreement (lower is better):")
    print(f"   RTMPose vs Halpe26:  {results_rtm_halpe['avg_spatial_diff']:.2f} pixels")
    print(f"   RTMPose vs ViTPose:  {results_rtm_vit['avg_spatial_diff']:.2f} pixels")
    print(f"   Halpe26 vs ViTPose:  {results_halpe_vit['avg_spatial_diff']:.2f} pixels")
    
    print("\n💡 Recommendations:")
    best_conf = confs_sorted[0][0]
    print(f"   • Best confidence: {best_conf}")
    
    if results_rtm_halpe['avg_spatial_diff'] < 5:
        print(f"   • RTMPose and Halpe26 are nearly identical on COCO keypoints (< 5px difference)")
        print(f"   • Halpe26 adds 9 extra keypoints (feet + head/neck/hip) with minimal overhead")
        print(f"   • ✅ Recommended: Use Halpe26 for richer pose data at same speed")
    elif results_rtm_halpe['avg_spatial_diff'] < 10:
        print(f"   • RTMPose and Halpe26 show good agreement (< 10px difference)")
        print(f"   • Halpe26 provides extra foot detail if needed")
    else:
        print(f"   • Notable differences between RTMPose and Halpe26")
        print(f"   • Choose based on specific use case requirements")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
