"""
Compare keypoints from RTMPose and ViTPose NPZ files
Shows differences in pose estimation between the two methods
"""

import numpy as np
from pathlib import Path
import sys

def compare_keypoints(rtm_path, vit_path):
    """Compare RTMPose and ViTPose keypoint outputs"""
    
    print("\n" + "=" * 70)
    print("🔍 KEYPOINT COMPARISON: RTMPose vs ViTPose")
    print("=" * 70 + "\n")
    
    # Load NPZ files
    rtm = np.load(rtm_path)
    vit = np.load(vit_path)
    
    rtm_frames = rtm['frame_numbers']
    rtm_kpts = rtm['keypoints']
    rtm_scores = rtm['scores']
    
    vit_frames = vit['frame_numbers']
    vit_kpts = vit['keypoints']
    vit_scores = vit['scores']
    
    # Basic statistics
    print("📊 Dataset Statistics:")
    print(f"   RTMPose frames: {len(rtm_frames)}")
    print(f"   ViTPose frames: {len(vit_frames)}")
    print(f"   RTMPose keypoints shape: {rtm_kpts.shape}")
    print(f"   ViTPose keypoints shape: {vit_kpts.shape}")
    
    # Confidence scores comparison
    print("\n📈 Confidence Scores:")
    rtm_avg_conf = np.mean(rtm_scores)
    vit_avg_conf = np.mean(vit_scores)
    rtm_min_conf = np.min(rtm_scores)
    vit_min_conf = np.min(vit_scores)
    rtm_max_conf = np.max(rtm_scores)
    vit_max_conf = np.max(vit_scores)
    
    print(f"   RTMPose - Avg: {rtm_avg_conf:.3f} | Min: {rtm_min_conf:.3f} | Max: {rtm_max_conf:.3f}")
    print(f"   ViTPose - Avg: {vit_avg_conf:.3f} | Min: {vit_min_conf:.3f} | Max: {vit_max_conf:.3f}")
    
    # Per-joint confidence
    print("\n🦴 Per-Joint Average Confidence:")
    joint_names = [
        "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
        "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
        "L_Wrist", "R_Wrist", "L_Hip", "R_Hip",
        "L_Knee", "R_Knee", "L_Ankle", "R_Ankle"
    ]
    
    rtm_joint_conf = np.mean(rtm_scores, axis=0)
    vit_joint_conf = np.mean(vit_scores, axis=0)
    
    print(f"   {'Joint':<15} {'RTMPose':<10} {'ViTPose':<10} {'Diff':<10}")
    print("   " + "-" * 50)
    for i, name in enumerate(joint_names):
        diff = rtm_joint_conf[i] - vit_joint_conf[i]
        symbol = "🟢" if abs(diff) < 0.05 else ("🔴" if abs(diff) > 0.1 else "🟡")
        print(f"   {name:<15} {rtm_joint_conf[i]:.3f}      {vit_joint_conf[i]:.3f}      {diff:+.3f} {symbol}")
    
    # Spatial differences (Euclidean distance)
    print("\n📏 Spatial Differences (Average per frame):")
    distances = np.sqrt(np.sum((rtm_kpts - vit_kpts) ** 2, axis=2))  # (frames, 17)
    avg_distances = np.mean(distances, axis=1)  # Average across joints per frame
    
    print(f"   Mean distance: {np.mean(avg_distances):.2f} pixels")
    print(f"   Median distance: {np.median(avg_distances):.2f} pixels")
    print(f"   Min distance: {np.min(avg_distances):.2f} pixels")
    print(f"   Max distance: {np.max(avg_distances):.2f} pixels")
    print(f"   Std deviation: {np.std(avg_distances):.2f} pixels")
    
    # Per-joint spatial differences
    print("\n🎯 Per-Joint Spatial Differences (Average):")
    joint_distances = np.mean(distances, axis=0)  # Average across frames per joint
    
    print(f"   {'Joint':<15} {'Avg Distance':<15} {'Assessment'}")
    print("   " + "-" * 50)
    for i, name in enumerate(joint_names):
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
    
    # Frame-by-frame analysis (first 10 frames)
    print("\n📹 Frame-by-Frame Analysis (First 10 frames):")
    print(f"   {'Frame':<8} {'Avg Distance':<15} {'Max Distance':<15}")
    print("   " + "-" * 50)
    for i in range(min(10, len(avg_distances))):
        max_dist = np.max(distances[i])
        print(f"   {i:<8} {avg_distances[i]:<15.2f} {max_dist:<15.2f}")
    
    # Worst frames (top 5)
    print("\n⚠️  Frames with Largest Differences (Top 5):")
    worst_frames = np.argsort(avg_distances)[-5:][::-1]
    print(f"   {'Frame':<8} {'Avg Distance':<15} {'Max Joint Distance':<20}")
    print("   " + "-" * 60)
    for frame_idx in worst_frames:
        max_dist = np.max(distances[frame_idx])
        print(f"   {frame_idx:<8} {avg_distances[frame_idx]:<15.2f} {max_dist:<20.2f}")
    
    # Best frames (top 5)
    print("\n✅ Frames with Smallest Differences (Top 5):")
    best_frames = np.argsort(avg_distances)[:5]
    print(f"   {'Frame':<8} {'Avg Distance':<15} {'Max Joint Distance':<20}")
    print("   " + "-" * 60)
    for frame_idx in best_frames:
        max_dist = np.max(distances[frame_idx])
        print(f"   {frame_idx:<8} {avg_distances[frame_idx]:<15.2f} {max_dist:<20.2f}")
    
    # Overall assessment
    print("\n" + "=" * 70)
    print("📋 OVERALL ASSESSMENT")
    print("=" * 70)
    
    avg_spatial_diff = np.mean(avg_distances)
    conf_diff = abs(rtm_avg_conf - vit_avg_conf)
    
    print(f"\n   Confidence Agreement: ", end="")
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
    
    print(f"\n   RTMPose confidence: {'🏆 Higher' if rtm_avg_conf > vit_avg_conf else '   Lower'}")
    print(f"   ViTPose confidence: {'🏆 Higher' if vit_avg_conf > rtm_avg_conf else '   Lower'}")
    
    print("\n" + "=" * 70 + "\n")
    
    return {
        'avg_spatial_diff': avg_spatial_diff,
        'conf_diff': conf_diff,
        'rtm_avg_conf': rtm_avg_conf,
        'vit_avg_conf': vit_avg_conf
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_keypoints.py <rtm_npz_path> <vit_npz_path>")
        sys.exit(1)
    
    rtm_path = Path(sys.argv[1])
    vit_path = Path(sys.argv[2])
    
    if not rtm_path.exists():
        print(f"Error: RTMPose NPZ not found: {rtm_path}")
        sys.exit(1)
    
    if not vit_path.exists():
        print(f"Error: ViTPose NPZ not found: {vit_path}")
        sys.exit(1)
    
    compare_keypoints(rtm_path, vit_path)
