"""
Compare joint angles between:
1. 2D keypoints: RTMPose vs Wholebody (frame 148)
2. 3D keypoints: Wholebody 3D vs MAGF 3D (frame 148)

Calculates elbow and knee bend angles for all models
"""

import numpy as np
import json

def load_json(json_path):
    """Load joint definition JSON"""
    with open(json_path, 'r') as f:
        return json.load(f)

def calculate_angle_2d(p1, p2, p3):
    """
    Calculate angle at p2 formed by points p1-p2-p3 in 2D
    Returns angle in degrees
    
    Args:
        p1, p2, p3: 2D points (x, y) as numpy arrays
    
    Returns:
        angle in degrees
    """
    # Vectors from p2 to p1 and p2 to p3
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Normalize vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    # Calculate angle using dot product
    cos_angle = np.dot(v1_norm, v2_norm)
    
    # Clamp to [-1, 1] to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Convert to degrees
    angle = np.arccos(cos_angle) * 180.0 / np.pi
    
    return angle

def calculate_angle_3d(p1, p2, p3):
    """
    Calculate angle at p2 formed by points p1-p2-p3 in 3D
    Returns angle in degrees
    
    Args:
        p1, p2, p3: 3D points (x, y, z) as numpy arrays
    
    Returns:
        angle in degrees
    """
    # Vectors from p2 to p1 and p2 to p3
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Normalize vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    # Calculate angle using dot product
    cos_angle = np.dot(v1_norm, v2_norm)
    
    # Clamp to [-1, 1] to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Convert to degrees
    angle = np.arccos(cos_angle) * 180.0 / np.pi
    
    return angle

def get_rtmpose_2d_angles(kps_2d, frame_idx):
    """
    Extract joint angles from RTMPose 2D data (COCO-17)
    
    COCO-17 indexing:
    5: left_shoulder, 6: right_shoulder
    7: left_elbow, 8: right_elbow
    9: left_wrist, 10: right_wrist
    11: left_hip, 12: right_hip
    13: left_knee, 14: right_knee
    15: left_ankle, 16: right_ankle
    """
    angles = {}
    
    # Right Elbow (shoulder -> elbow -> wrist)
    r_shoulder = kps_2d[frame_idx, 6, :2]  # right_shoulder (x, y)
    r_elbow = kps_2d[frame_idx, 8, :2]     # right_elbow
    r_wrist = kps_2d[frame_idx, 10, :2]    # right_wrist
    angles['R_Elbow'] = calculate_angle_2d(r_shoulder, r_elbow, r_wrist)
    
    # Left Elbow (shoulder -> elbow -> wrist)
    l_shoulder = kps_2d[frame_idx, 5, :2]  # left_shoulder
    l_elbow = kps_2d[frame_idx, 7, :2]     # left_elbow
    l_wrist = kps_2d[frame_idx, 9, :2]     # left_wrist
    angles['L_Elbow'] = calculate_angle_2d(l_shoulder, l_elbow, l_wrist)
    
    # Right Knee (hip -> knee -> ankle)
    r_hip = kps_2d[frame_idx, 12, :2]      # right_hip
    r_knee = kps_2d[frame_idx, 14, :2]     # right_knee
    r_ankle = kps_2d[frame_idx, 16, :2]    # right_ankle
    angles['R_Knee'] = calculate_angle_2d(r_hip, r_knee, r_ankle)
    
    # Left Knee (hip -> knee -> ankle)
    l_hip = kps_2d[frame_idx, 11, :2]      # left_hip
    l_knee = kps_2d[frame_idx, 13, :2]     # left_knee
    l_ankle = kps_2d[frame_idx, 15, :2]    # left_ankle
    angles['L_Knee'] = calculate_angle_2d(l_hip, l_knee, l_ankle)
    
    return angles

def get_wholebody_2d_angles(kps_2d, frame_idx):
    """
    Extract joint angles from Wholebody 2D data (COCO-133)
    Body joints use same indexing as COCO-17 for first 17 keypoints
    
    COCO-17 indexing (same for Wholebody body joints):
    5: left_shoulder, 6: right_shoulder
    7: left_elbow, 8: right_elbow
    9: left_wrist, 10: right_wrist
    11: left_hip, 12: right_hip
    13: left_knee, 14: right_knee
    15: left_ankle, 16: right_ankle
    """
    angles = {}
    
    # Right Elbow (shoulder -> elbow -> wrist)
    r_shoulder = kps_2d[frame_idx, 6, :2]  # right_shoulder (x, y)
    r_elbow = kps_2d[frame_idx, 8, :2]     # right_elbow
    r_wrist = kps_2d[frame_idx, 10, :2]    # right_wrist
    angles['R_Elbow'] = calculate_angle_2d(r_shoulder, r_elbow, r_wrist)
    
    # Left Elbow (shoulder -> elbow -> wrist)
    l_shoulder = kps_2d[frame_idx, 5, :2]  # left_shoulder
    l_elbow = kps_2d[frame_idx, 7, :2]     # left_elbow
    l_wrist = kps_2d[frame_idx, 9, :2]     # left_wrist
    angles['L_Elbow'] = calculate_angle_2d(l_shoulder, l_elbow, l_wrist)
    
    # Right Knee (hip -> knee -> ankle)
    r_hip = kps_2d[frame_idx, 12, :2]      # right_hip
    r_knee = kps_2d[frame_idx, 14, :2]     # right_knee
    r_ankle = kps_2d[frame_idx, 16, :2]    # right_ankle
    angles['R_Knee'] = calculate_angle_2d(r_hip, r_knee, r_ankle)
    
    # Left Knee (hip -> knee -> ankle)
    l_hip = kps_2d[frame_idx, 11, :2]      # left_hip
    l_knee = kps_2d[frame_idx, 13, :2]     # left_knee
    l_ankle = kps_2d[frame_idx, 15, :2]    # left_ankle
    angles['L_Knee'] = calculate_angle_2d(l_hip, l_knee, l_ankle)
    
    return angles

def get_wholebody_3d_angles(kps_3d, frame_idx):
    """
    Extract joint angles from Wholebody 3D data
    
    Wholebody uses COCO-17 indexing for body joints:
    5: left_shoulder, 6: right_shoulder
    7: left_elbow, 8: right_elbow
    9: left_wrist, 10: right_wrist
    11: left_hip, 12: right_hip
    13: left_knee, 14: right_knee
    15: left_ankle, 16: right_ankle
    """
    angles = {}
    
    # Right Elbow (shoulder -> elbow -> wrist)
    r_shoulder = kps_3d[frame_idx, 6]  # right_shoulder
    r_elbow = kps_3d[frame_idx, 8]     # right_elbow
    r_wrist = kps_3d[frame_idx, 10]    # right_wrist
    angles['R_Elbow'] = calculate_angle_3d(r_shoulder, r_elbow, r_wrist)
    
    # Left Elbow (shoulder -> elbow -> wrist)
    l_shoulder = kps_3d[frame_idx, 5]  # left_shoulder
    l_elbow = kps_3d[frame_idx, 7]     # left_elbow
    l_wrist = kps_3d[frame_idx, 9]     # left_wrist
    angles['L_Elbow'] = calculate_angle_3d(l_shoulder, l_elbow, l_wrist)
    
    # Right Knee (hip -> knee -> ankle)
    r_hip = kps_3d[frame_idx, 12]      # right_hip
    r_knee = kps_3d[frame_idx, 14]     # right_knee
    r_ankle = kps_3d[frame_idx, 16]    # right_ankle
    angles['R_Knee'] = calculate_angle_3d(r_hip, r_knee, r_ankle)
    
    # Left Knee (hip -> knee -> ankle)
    l_hip = kps_3d[frame_idx, 11]      # left_hip
    l_knee = kps_3d[frame_idx, 13]     # left_knee
    l_ankle = kps_3d[frame_idx, 15]    # left_ankle
    angles['L_Knee'] = calculate_angle_3d(l_hip, l_knee, l_ankle)
    
    return angles

def get_magf_3d_angles(poses_3d, frame_idx):
    """
    Extract joint angles from MAGF (H36M-17) 3D data
    
    H36M-17 indexing:
    1: right_hip, 2: right_knee, 3: right_ankle
    4: left_hip, 5: left_knee, 6: left_ankle
    11: left_shoulder, 12: left_elbow, 13: left_wrist
    14: right_shoulder, 15: right_elbow, 16: right_wrist
    """
    angles = {}
    
    # Right Elbow (shoulder -> elbow -> wrist)
    r_shoulder = poses_3d[frame_idx, 14]  # right_shoulder
    r_elbow = poses_3d[frame_idx, 15]     # right_elbow
    r_wrist = poses_3d[frame_idx, 16]     # right_wrist
    angles['R_Elbow'] = calculate_angle_3d(r_shoulder, r_elbow, r_wrist)
    
    # Left Elbow (shoulder -> elbow -> wrist)
    l_shoulder = poses_3d[frame_idx, 11]  # left_shoulder
    l_elbow = poses_3d[frame_idx, 12]     # left_elbow
    l_wrist = poses_3d[frame_idx, 13]     # left_wrist
    angles['L_Elbow'] = calculate_angle_3d(l_shoulder, l_elbow, l_wrist)
    
    # Right Knee (hip -> knee -> ankle)
    r_hip = poses_3d[frame_idx, 1]        # right_hip
    r_knee = poses_3d[frame_idx, 2]       # right_knee
    r_ankle = poses_3d[frame_idx, 3]      # right_ankle
    angles['R_Knee'] = calculate_angle_3d(r_hip, r_knee, r_ankle)
    
    # Left Knee (hip -> knee -> ankle)
    l_hip = poses_3d[frame_idx, 4]        # left_hip
    l_knee = poses_3d[frame_idx, 5]       # left_knee
    l_ankle = poses_3d[frame_idx, 6]      # left_ankle
    angles['L_Knee'] = calculate_angle_3d(l_hip, l_knee, l_ankle)
    
    return angles

def main():
    # File paths (update for Colab if needed)
    rtm_2d_path = "demo_data/outputs/kps_2d_rtm.npz"
    wb_2d_path = "demo_data/outputs/kps_2d_wholebody.npz"
    wb_3d_path = "demo_data/outputs/kps_3d_wholebody.npz"
    magf_3d_path = "demo_data/outputs/kps_3d_magf.npz"
    
    # Frame to analyze
    frame_idx = 148
    
    print("=" * 90)
    print(f"JOINT ANGLE COMPARISON - Frame {frame_idx}")
    print("=" * 90)
    print()
    
    # ========== 2D COMPARISON ==========
    print("╔" + "═" * 88 + "╗")
    print("║" + " " * 30 + "2D KEYPOINTS COMPARISON" + " " * 35 + "║")
    print("╚" + "═" * 88 + "╝")
    print()
    
    # Load 2D keypoints
    rtm_2d_data = np.load(rtm_2d_path)
    rtm_2d_kps = rtm_2d_data['keypoints']
    print(f"RTMPose 2D shape: {rtm_2d_kps.shape}")
    
    wb_2d_data = np.load(wb_2d_path)
    wb_2d_kps = wb_2d_data['keypoints']
    print(f"Wholebody 2D shape: {wb_2d_kps.shape}")
    print()
    
    # Calculate 2D angles
    rtm_2d_angles = get_rtmpose_2d_angles(rtm_2d_kps, frame_idx)
    wb_2d_angles = get_wholebody_2d_angles(wb_2d_kps, frame_idx)
    
    # Print 2D comparison table
    print("-" * 90)
    print(f"{'Joint':<15} {'RTMPose 2D':<20} {'Wholebody 2D':<20} {'Difference':<15}")
    print("-" * 90)
    
    for joint in ['R_Elbow', 'L_Elbow', 'R_Knee', 'L_Knee']:
        rtm_angle = rtm_2d_angles[joint]
        wb_angle = wb_2d_angles[joint]
        diff = abs(rtm_angle - wb_angle)
        
        print(f"{joint:<15} {rtm_angle:>6.2f}°{'':<13} {wb_angle:>6.2f}°{'':<13} {diff:>6.2f}°")
    
    print("-" * 90)
    
    # Calculate 2D average difference
    avg_diff_2d = np.mean([abs(rtm_2d_angles[j] - wb_2d_angles[j]) 
                           for j in ['R_Elbow', 'L_Elbow', 'R_Knee', 'L_Knee']])
    print(f"Average 2D angle difference: {avg_diff_2d:.2f}°")
    print()
    print()
    
    # ========== 3D COMPARISON ==========
    print("╔" + "═" * 88 + "╗")
    print("║" + " " * 30 + "3D KEYPOINTS COMPARISON" + " " * 35 + "║")
    print("╚" + "═" * 88 + "╝")
    print()
    
    # Load 3D keypoints
    wb_3d_data = np.load(wb_3d_path)
    wb_3d_kps = wb_3d_data['keypoints_3d']
    print(f"Wholebody 3D shape: {wb_3d_kps.shape}")
    
    magf_3d_data = np.load(magf_3d_path)
    magf_3d_poses = magf_3d_data['poses_3d']
    print(f"MAGF 3D shape: {magf_3d_poses.shape}")
    print()
    
    # Calculate 3D angles
    wb_3d_angles = get_wholebody_3d_angles(wb_3d_kps, frame_idx)
    magf_3d_angles = get_magf_3d_angles(magf_3d_poses, frame_idx)
    
    # Print 3D comparison table
    print("-" * 90)
    print(f"{'Joint':<15} {'Wholebody 3D':<20} {'MAGF 3D':<20} {'Difference':<15}")
    print("-" * 90)
    
    for joint in ['R_Elbow', 'L_Elbow', 'R_Knee', 'L_Knee']:
        wb_angle = wb_3d_angles[joint]
        magf_angle = magf_3d_angles[joint]
        diff = abs(wb_angle - magf_angle)
        
        print(f"{joint:<15} {wb_angle:>6.2f}°{'':<13} {magf_angle:>6.2f}°{'':<13} {diff:>6.2f}°")
    
    print("-" * 90)
    
    # Calculate 3D average difference
    avg_diff_3d = np.mean([abs(wb_3d_angles[j] - magf_3d_angles[j]) 
                           for j in ['R_Elbow', 'L_Elbow', 'R_Knee', 'L_Knee']])
    print(f"Average 3D angle difference: {avg_diff_3d:.2f}°")
    print()
    
    # ========== DETAILED COMPARISON ==========
    print()
    print("╔" + "═" * 88 + "╗")
    print("║" + " " * 34 + "DETAILED BREAKDOWN" + " " * 36 + "║")
    print("╚" + "═" * 88 + "╝")
    print()
    
    print("=" * 90)
    print("2D ANGLE DETAILS")
    print("=" * 90)
    
    for joint in ['R_Elbow', 'L_Elbow', 'R_Knee', 'L_Knee']:
        rtm_angle = rtm_2d_angles[joint]
        wb_angle = wb_2d_angles[joint]
        diff = rtm_angle - wb_angle
        diff_pct = (diff / rtm_angle) * 100 if rtm_angle != 0 else 0
        
        print(f"\n{joint}:")
        print(f"  RTMPose 2D:   {rtm_angle:6.2f}°")
        print(f"  Wholebody 2D: {wb_angle:6.2f}°")
        print(f"  Difference:   {diff:6.2f}° ({diff_pct:+.1f}%)")
    
    print()
    print("=" * 90)
    print("3D ANGLE DETAILS")
    print("=" * 90)
    
    for joint in ['R_Elbow', 'L_Elbow', 'R_Knee', 'L_Knee']:
        wb_angle = wb_3d_angles[joint]
        magf_angle = magf_3d_angles[joint]
        diff = wb_angle - magf_angle
        diff_pct = (diff / wb_angle) * 100 if wb_angle != 0 else 0
        
        print(f"\n{joint}:")
        print(f"  Wholebody 3D: {wb_angle:6.2f}°")
        print(f"  MAGF 3D:      {magf_angle:6.2f}°")
        print(f"  Difference:   {diff:6.2f}° ({diff_pct:+.1f}%)")
    
    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"Average 2D difference (RTMPose vs Wholebody):  {avg_diff_2d:.2f}°")
    print(f"Average 3D difference (Wholebody vs MAGF):     {avg_diff_3d:.2f}°")
    print("=" * 90)

if __name__ == "__main__":
    main()
