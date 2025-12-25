"""
Frame 148 Joint Angle Analysis with 4-Panel Visualization

Creates a comprehensive visualization showing:
- Panel 1: RTMPose 2D keypoints on original frame
- Panel 2: Wholebody 2D keypoints on original frame
- Panel 3: MAGF 3D skeleton
- Panel 4: Wholebody 3D skeleton

All panels have numbered joints for the analyzed joints (shoulders, elbows, wrists, hips, knees, ankles)

Usage:
    python visualize_frame148_analysis.py [--no-viz] [--no-joint-info]
    
    --no-viz: Skip visualization generation (only show tables)
    --no-joint-info: Skip detailed joint mapping information
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import argparse

# Joint names for logging
COCO17_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

H36M17_NAMES = [
    'pelvis_root', 'right_hip', 'right_knee', 'right_ankle',
    'left_hip', 'left_knee', 'left_ankle', 'spine_mid', 'spine_top_neck',
    'chin', 'head_top', 'left_shoulder', 'left_elbow', 'left_wrist',
    'right_shoulder', 'right_elbow', 'right_wrist'
]

# Joint indices for angle analysis
ANALYSIS_JOINTS_COCO = {
    'R_Elbow': [6, 8, 10],  # right_shoulder, right_elbow, right_wrist
    'L_Elbow': [5, 7, 9],   # left_shoulder, left_elbow, left_wrist
    'R_Knee': [12, 14, 16], # right_hip, right_knee, right_ankle
    'L_Knee': [11, 13, 15], # left_hip, left_knee, left_ankle
    'R_Shoulder': [8, 6, 12],  # right_elbow, right_shoulder, right_hip
    'L_Shoulder': [7, 5, 11],  # left_elbow, left_shoulder, left_hip
    'R_Hip': [6, 12, 14],   # right_shoulder, right_hip, right_knee
    'L_Hip': [5, 11, 13],   # left_shoulder, left_hip, left_knee
    'Spine_Upper': [-2, -1, 11],  # shoulder_mid (computed), spine_mid (computed), left_hip
    'Neck': [6, -2, 0]      # right_shoulder, shoulder_mid (computed), nose
}

ANALYSIS_JOINTS_H36M = {
    'R_Elbow': [14, 15, 16],  # right_shoulder, right_elbow, right_wrist
    'L_Elbow': [11, 12, 13],  # left_shoulder, left_elbow, left_wrist
    'R_Knee': [1, 2, 3],      # right_hip, right_knee, right_ankle
    'L_Knee': [4, 5, 6],      # left_hip, left_knee, left_ankle
    'R_Shoulder': [15, 14, 1],  # right_elbow, right_shoulder, right_hip
    'L_Shoulder': [12, 11, 4],  # left_elbow, left_shoulder, left_hip
    'R_Hip': [14, 1, 2],      # right_shoulder, right_hip, right_knee
    'L_Hip': [11, 4, 5],      # left_shoulder, left_hip, left_knee
    'Spine_Upper': [-2, 7, 4],   # shoulder_mid (computed), spine_mid, left_hip
    'Neck': [14, 8, 10]       # right_shoulder, spine_top_neck, head_top
}


def qrot(q, v):
    """Rotate vector(s) v about quaternion q"""
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    qvec = q[..., 1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2 * (q[..., :1] * uv + uuv)


def camera_to_world(X, R, t):
    """Apply camera rotation using quaternion"""
    R_tiled = np.tile(R, (X.shape[0], 1))
    return qrot(R_tiled, X) + t


def draw_2d_skeleton(frame, keypoints, joint_indices, title, color=(0, 255, 0)):
    """
    Draw 2D skeleton on frame with numbered joints for analysis
    
    Args:
        frame: Video frame (BGR)
        keypoints: (N, 2) array of keypoint positions
        joint_indices: Dictionary of joint groups for analysis
        title: Panel title
        color: Default line color (BGR)
    """
    img = frame.copy()
    h, w = img.shape[:2]
    
    # COCO skeleton connections
    connections = [
        # Head
        (0, 1), (0, 2), (1, 3), (2, 4),
        # Arms
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        # Torso
        (5, 11), (6, 12), (11, 12),
        # Legs
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    
    # Draw skeleton connections
    for i, j in connections:
        if i < len(keypoints) and j < len(keypoints):
            pt1 = tuple(keypoints[i].astype(int))
            pt2 = tuple(keypoints[j].astype(int))
            cv2.line(img, pt1, pt2, color, 2)
    
    # Draw all joints
    for idx, kp in enumerate(keypoints[:17]):  # Only body joints
        pt = tuple(kp.astype(int))
        cv2.circle(img, pt, 4, (255, 255, 0), -1)
        cv2.circle(img, pt, 4, (0, 0, 0), 1)
    
    # Highlight and number analysis joints
    all_analysis_joints = set()
    for joint_list in joint_indices.values():
        all_analysis_joints.update(joint_list)
    
    for idx in all_analysis_joints:
        if idx < len(keypoints):
            pt = tuple(keypoints[idx].astype(int))
            # Draw larger circle
            cv2.circle(img, pt, 8, (0, 0, 255), -1)
            cv2.circle(img, pt, 8, (255, 255, 255), 2)
            # Draw joint number
            cv2.putText(img, str(idx), (pt[0] + 10, pt[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, str(idx), (pt[0] + 10, pt[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    # Add title
    cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               1.0, (255, 255, 255), 3)
    cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               1.0, (0, 0, 255), 2)
    
    return img


def draw_3d_skeleton(pose_3d, joint_indices, title, is_h36m=False):
    """
    Draw 3D skeleton with numbered joints for analysis
    
    Args:
        pose_3d: (N, 3) array of 3D joint positions
        joint_indices: Dictionary of joint groups for analysis
        title: Panel title
        is_h36m: If True, use H36M format; else use COCO/Wholebody format
    """
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=15., azim=70)
    
    # Process skeleton
    if is_h36m:
        # H36M format (already centered at pelvis)
        body_pose = pose_3d.copy()
        # Apply camera rotation
        rot = np.array([0.1407056450843811, -0.1500701755285263, 
                       -0.755240797996521, 0.6223280429840088], dtype='float32')
        body_pose = camera_to_world(body_pose, R=rot, t=0)
        
        # Ground skeleton
        min_z = np.min(body_pose[:, 2])
        body_pose[:, 2] -= min_z
        
        # Normalize
        max_value = np.max(body_pose)
        if max_value > 0:
            body_pose /= max_value
        
        # H36M skeleton connections
        connections = [
            (0, 1), (1, 2), (2, 3),  # Right leg
            (0, 4), (4, 5), (5, 6),  # Left leg
            (0, 7), (7, 8), (8, 9), (9, 10),  # Spine to head
            (8, 11), (11, 12), (12, 13),  # Left arm
            (8, 14), (14, 15), (15, 16)   # Right arm
        ]
        
    else:
        # Wholebody format (COCO-17 body joints)
        body_pose = pose_3d[:17, :].copy()
        
        # Find pelvis
        left_hip = body_pose[11, :]
        right_hip = body_pose[12, :]
        pelvis = (left_hip + right_hip) / 2
        
        # Center at pelvis
        body_pose = body_pose - pelvis[None, :]
        
        # Apply camera rotation
        rot = np.array([0.1407056450843811, -0.1500701755285263, 
                       -0.755240797996521, 0.6223280429840088], dtype='float32')
        body_pose = camera_to_world(body_pose, R=rot, t=0)
        
        # Ground skeleton
        min_z = np.min(body_pose[:, 2])
        body_pose[:, 2] -= min_z
        
        # Normalize
        max_value = np.max(body_pose)
        if max_value > 0:
            body_pose /= max_value
        
        # COCO skeleton connections
        connections = [
            # Head
            (0, 1), (0, 2), (1, 3), (2, 4),
            # Arms
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            # Torso
            (5, 11), (6, 12), (11, 12),
            # Legs
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]
    
    # Draw skeleton connections
    for i, j in connections:
        if i < len(body_pose) and j < len(body_pose):
            x = [body_pose[i, 0], body_pose[j, 0]]
            y = [body_pose[i, 1], body_pose[j, 1]]
            z = [body_pose[i, 2], body_pose[j, 2]]
            ax.plot(x, y, z, lw=2, color='gray', alpha=0.6)
    
    # Draw all joints
    ax.scatter(body_pose[:, 0], body_pose[:, 1], body_pose[:, 2], 
              c='yellow', s=30, alpha=0.7, edgecolors='black', linewidth=1)
    
    # Highlight and number analysis joints with external labels
    all_analysis_joints = set()
    for joint_list in joint_indices.values():
        all_analysis_joints.update(joint_list)
    
    for idx in all_analysis_joints:
        if idx < len(body_pose):
            # Draw red sphere
            ax.scatter(body_pose[idx, 0], body_pose[idx, 1], body_pose[idx, 2],
                      c='red', s=100, alpha=1.0, edgecolors='white', linewidth=2)
            
            # Position label outside skeleton with offset
            offset_x = 0.15 if body_pose[idx, 0] > 0 else -0.15
            offset_y = 0.15 if body_pose[idx, 1] > 0 else -0.15
            label_x = body_pose[idx, 0] + offset_x
            label_y = body_pose[idx, 1] + offset_y
            label_z = body_pose[idx, 2]
            
            # Draw connection line from joint to label
            ax.plot([body_pose[idx, 0], label_x], 
                   [body_pose[idx, 1], label_y],
                   [body_pose[idx, 2], label_z],
                   'k--', linewidth=0.5, alpha=0.5)
            
            # Draw text label
            ax.text(label_x, label_y, label_z,
                   str(idx), fontsize=8, color='black', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.9, edgecolor='black'))
    
    # Set view limits
    RADIUS = 0.72
    RADIUS_Z = max(1.1, np.max(body_pose[:, 2]) * 1.1)
    xroot, yroot, zroot = 0, 0, 0
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([0, RADIUS_Z])
    ax.set_aspect('auto')
    
    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Convert to image
    fig.canvas.draw()
    # Use buffer_rgba() for newer matplotlib versions
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    plt.close(fig)
    
    return img


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


def print_angle_comparison_tables(rtm_2d_kps, wb_2d_kps, wb_3d_kps, magf_3d_poses, frame_idx, show_joint_info=True):
    """Print formatted comparison tables for 2D and 3D angles"""
    
    # Helper function to get spine midpoint for COCO format
    def get_spine_mid_coco(kps, is_3d=False):
        """Compute spine midpoint as average of left_hip (11) and right_hip (12)"""
        if is_3d:
            return (kps[11] + kps[12]) / 2.0
        else:
            return (kps[11] + kps[12]) / 2.0
    
    # Helper function to get shoulder midpoint for COCO format
    def get_shoulder_mid_coco(kps, is_3d=False):
        """Compute shoulder midpoint as average of left_shoulder (5) and right_shoulder (6)"""
        if is_3d:
            return (kps[5] + kps[6]) / 2.0
        else:
            return (kps[5] + kps[6]) / 2.0
    
    # Helper function to get shoulder midpoint for H36M format
    def get_shoulder_mid_h36m(kps):
        """Compute shoulder midpoint as average of left_shoulder (11) and right_shoulder (14)"""
        return (kps[11] + kps[14]) / 2.0
    
    # Calculate all angles
    angles_2d_rtm = {}
    angles_2d_wb = {}
    angles_3d_wb = {}
    angles_3d_magf = {}
    
    # 2D angles (RTMPose)
    for angle_name, joints in ANALYSIS_JOINTS_COCO.items():
        if angle_name == 'Spine_Upper':
            # Special handling: compute spine midpoint
            spine_mid = get_spine_mid_coco(rtm_2d_kps, is_3d=False)
            p1 = rtm_2d_kps[joints[0]]  # right_hip (12)
            p2 = spine_mid
            p3 = rtm_2d_kps[joints[2]]  # right_shoulder (6)
        elif angle_name == 'Neck':
            # Special handling: compute shoulder midpoint
            shoulder_mid = get_shoulder_mid_coco(rtm_2d_kps, is_3d=False)
            p1 = rtm_2d_kps[joints[0]]  # right_shoulder (6)
            p2 = shoulder_mid
            p3 = rtm_2d_kps[joints[2]]  # nose (0)
        else:
            p1 = rtm_2d_kps[joints[0]]
            p2 = rtm_2d_kps[joints[1]]
            p3 = rtm_2d_kps[joints[2]]
        angles_2d_rtm[angle_name] = calculate_angle_2d(p1, p2, p3)
    
    # 2D angles (Wholebody)
    for angle_name, joints in ANALYSIS_JOINTS_COCO.items():
        if angle_name == 'Spine_Upper':
            spine_mid = get_spine_mid_coco(wb_2d_kps, is_3d=False)
            p1 = wb_2d_kps[joints[0]]
            p2 = spine_mid
            p3 = wb_2d_kps[joints[2]]
        elif angle_name == 'Neck':
            shoulder_mid = get_shoulder_mid_coco(wb_2d_kps, is_3d=False)
            p1 = wb_2d_kps[joints[0]]
            p2 = shoulder_mid
            p3 = wb_2d_kps[joints[2]]
        else:
            p1 = wb_2d_kps[joints[0]]
            p2 = wb_2d_kps[joints[1]]
            p3 = wb_2d_kps[joints[2]]
        angles_2d_wb[angle_name] = calculate_angle_2d(p1, p2, p3)
    
    # 3D angles (Wholebody)
    for angle_name, joints in ANALYSIS_JOINTS_COCO.items():
        if angle_name == 'Spine_Upper':
            spine_mid = get_spine_mid_coco(wb_3d_kps, is_3d=True)
            p1 = wb_3d_kps[joints[0]]
            p2 = spine_mid
            p3 = wb_3d_kps[joints[2]]
        elif angle_name == 'Neck':
            shoulder_mid = get_shoulder_mid_coco(wb_3d_kps, is_3d=True)
            p1 = wb_3d_kps[joints[0]]
            p2 = shoulder_mid
            p3 = wb_3d_kps[joints[2]]
        else:
            p1 = wb_3d_kps[joints[0]]
            p2 = wb_3d_kps[joints[1]]
            p3 = wb_3d_kps[joints[2]]
        angles_3d_wb[angle_name] = calculate_angle_3d(p1, p2, p3)
    
    # 3D angles (MAGF)
    for angle_name, joints in ANALYSIS_JOINTS_H36M.items():
        # Special handling for computed joints (shoulder_mid for Spine_Upper)
        if angle_name == 'Spine_Upper':
            shoulder_mid = get_shoulder_mid_h36m(magf_3d_poses)
            p1 = shoulder_mid
            p2 = magf_3d_poses[joints[1]]  # spine_mid (7)
            p3 = magf_3d_poses[joints[2]]  # left_hip (4)
        else:
            p1 = magf_3d_poses[joints[0]]
            p2 = magf_3d_poses[joints[1]]
            p3 = magf_3d_poses[joints[2]]
        angles_3d_magf[angle_name] = calculate_angle_3d(p1, p2, p3)
    
    print("\n" + "="*100)
    print("3D JOINT ANGLE COMPARISON")
    print("="*100)
    print(f"{'Angle':<14} {'WB3D (j1,j2,j3)':<20} {'Value':<8} {'MAGF (j1,j2,j3)':<20} {'Value':<8} {'Diff':<8} {'Diff%':<8}")
    print("-"*100)
    
    angle_order = ['R_Elbow', 'L_Elbow', 'R_Shoulder', 'L_Shoulder', 'R_Hip', 'L_Hip', 'R_Knee', 'L_Knee', 'Spine_Upper', 'Neck']
    
    for angle_name in angle_order:
        wb_joints = ANALYSIS_JOINTS_COCO[angle_name]
        magf_joints = ANALYSIS_JOINTS_H36M[angle_name]
        
        # Special display for computed joints
        if angle_name == 'Spine_Upper':
            wb_joints_str = f"({wb_joints[0]},mid,{wb_joints[2]})"
        elif angle_name == 'Neck':
            wb_joints_str = f"({wb_joints[0]},mid,{wb_joints[2]})"
        else:
            wb_joints_str = f"({wb_joints[0]},{wb_joints[1]},{wb_joints[2]})"
        
        magf_joints_str = f"({magf_joints[0]},{magf_joints[1]},{magf_joints[2]})"
        
        wb_angle = angles_3d_wb[angle_name]
        magf_angle = angles_3d_magf[angle_name]
        diff = wb_angle - magf_angle
        diff_pct = (diff / wb_angle * 100) if wb_angle != 0 else 0
        
        print(f"{angle_name:<14} {wb_joints_str:<20} {wb_angle:>6.2f}°  {magf_joints_str:<20} {magf_angle:>6.2f}°  {diff:>6.2f}°  {diff_pct:>+6.1f}%")
    
    print("="*100)
    
    print("\n" + "="*100)
    print("2D & 3D COMBINED COMPARISON")
    print("="*100)
    print(f"{'Angle':<14} {'2D Diff':<10} {'2D Diff%':<10} {'3D Diff':<10} {'3D Diff%':<10}")
    print("-"*100)
    
    for angle_name in angle_order:
        # 2D differences
        diff_2d = angles_2d_rtm[angle_name] - angles_2d_wb[angle_name]
        diff_2d_pct = (diff_2d / angles_2d_rtm[angle_name] * 100) if angles_2d_rtm[angle_name] != 0 else 0
        
        # 3D differences
        diff_3d = angles_3d_wb[angle_name] - angles_3d_magf[angle_name]
        diff_3d_pct = (diff_3d / angles_3d_wb[angle_name] * 100) if angles_3d_wb[angle_name] != 0 else 0
        
        print(f"{angle_name:<14} {diff_2d:>+6.2f}°   {diff_2d_pct:>+6.1f}%    {diff_3d:>+6.2f}°   {diff_3d_pct:>+6.1f}%")
    
    print("="*100)
    
    # Self-comparison tables
    print("\n" + "="*100)
    print("RTMPOSE SELF-COMPARISON (Own 2D vs MAGF 3D from same 2D input)")
    print("="*100)
    print(f"{'Angle':<14} {'Own 2D':<10} {'MAGF 3D':<10} {'Diff':<12} {'Diff%':<10}")
    print("-"*100)
    
    for angle_name in angle_order:
        rtm_2d_angle = angles_2d_rtm[angle_name]
        magf_3d_angle = angles_3d_magf[angle_name]
        diff = rtm_2d_angle - magf_3d_angle
        diff_pct = (diff / rtm_2d_angle * 100) if rtm_2d_angle != 0 else 0
        
        print(f"{angle_name:<14} {rtm_2d_angle:>6.2f}°   {magf_3d_angle:>6.2f}°   {diff:>+6.2f}°   {diff_pct:>+6.1f}%")
    
    print("="*100)
    
    print("\n" + "="*100)
    print("WHOLEBODY SELF-COMPARISON (Own 2D vs Own 3D)")
    print("="*100)
    print(f"{'Angle':<14} {'Own 2D':<10} {'Own 3D':<10} {'Diff':<12} {'Diff%':<10}")
    print("-"*100)
    
    for angle_name in angle_order:
        wb_2d_angle = angles_2d_wb[angle_name]
        wb_3d_angle = angles_3d_wb[angle_name]
        diff = wb_2d_angle - wb_3d_angle
        diff_pct = (diff / wb_2d_angle * 100) if wb_2d_angle != 0 else 0
        
        print(f"{angle_name:<14} {wb_2d_angle:>6.2f}°   {wb_3d_angle:>6.2f}°   {diff:>+6.2f}°   {diff_pct:>+6.1f}%")
    
    print("="*100)
    
    if show_joint_info:
        print("\n" + "="*100)
        print("JOINT NAMES VERIFICATION")
        print("="*100)
        for angle_name in ['R_Elbow', 'L_Elbow', 'R_Knee', 'L_Knee']:
            coco_joints = ANALYSIS_JOINTS_COCO[angle_name]
            h36m_joints = ANALYSIS_JOINTS_H36M[angle_name]
            
            coco_names = [COCO17_NAMES[j] for j in coco_joints]
            h36m_names = [H36M17_NAMES[j] for j in h36m_joints]
            
            print(f"\n{angle_name}:")
            print(f"  COCO (WB3D, RTM, WB2D): {coco_joints} = {coco_names}")
            print(f"  H36M (MAGF):            {h36m_joints} = {h36m_names}")
        
        print("="*100 + "\n")


def log_joint_info(show_details=True):
    """Log joint names and indices for both formats"""
    if not show_details:
        return
    
    print("\n" + "="*80)
    print("JOINT MAPPING INFORMATION")
    print("="*80)
    
    print("\n--- COCO-17 Format (RTMPose, Wholebody 2D/3D) ---")
    for idx, name in enumerate(COCO17_NAMES):
        marker = " *" if any(idx in joints for joints in ANALYSIS_JOINTS_COCO.values()) else ""
        print(f"  {idx:2d}: {name:20s}{marker}")
    
    print("\n--- H36M-17 Format (MAGF 3D) ---")
    for idx, name in enumerate(H36M17_NAMES):
        marker = " *" if any(idx in joints for joints in ANALYSIS_JOINTS_H36M.values()) else ""
        print(f"  {idx:2d}: {name:20s}{marker}")
    
    print("\n--- Analysis Joints (COCO format) ---")
    for angle_name, joint_idxs in ANALYSIS_JOINTS_COCO.items():
        names = [COCO17_NAMES[i] for i in joint_idxs]
        print(f"  {angle_name}: {joint_idxs} -> {names}")
    
    print("\n--- Analysis Joints (H36M format) ---")
    for angle_name, joint_idxs in ANALYSIS_JOINTS_H36M.items():
        names = [H36M17_NAMES[i] for i in joint_idxs]
        print(f"  {angle_name}: {joint_idxs} -> {names}")
    
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Frame 148 Joint Angle Analysis')
    parser.add_argument('--no-viz', action='store_true', 
                       help='Skip visualization generation (only show tables)')
    parser.add_argument('--no-joint-info', action='store_true',
                       help='Skip detailed joint mapping information')
    args = parser.parse_args()
    
    # File paths
    video_path = "demo_data/videos/dance.mp4"
    rtm_2d_path = "demo_data/outputs/kps_2d_rtm.npz"
    wb_2d_path = "demo_data/outputs/kps_2d_wholebody.npz"
    wb_3d_path = "demo_data/outputs/kps_3d_wholebody.npz"
    magf_3d_path = "demo_data/outputs/kps_3d_magf.npz"
    output_path = "demo_data/outputs/frame148_analysis.png"
    
    frame_idx = 148
    
    # Log joint information (only if not disabled)
    log_joint_info(show_details=not args.no_joint_info)
    
    print(f"Loading data for frame {frame_idx}...")
    
    # Load video frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read frame {frame_idx} from video")
    
    # Resize frame for visualization
    target_height = 600
    aspect = frame.shape[1] / frame.shape[0]
    target_width = int(target_height * aspect)
    frame = cv2.resize(frame, (target_width, target_height))
    
    # Load 2D keypoints
    rtm_2d_data = np.load(rtm_2d_path)
    rtm_2d_kps = rtm_2d_data['keypoints'][frame_idx]
    
    # Scale keypoints to resized frame
    scale_x = target_width / rtm_2d_data['keypoints'].shape[2] if len(rtm_2d_data['keypoints'].shape) > 2 else target_width / 1920
    scale_y = target_height / rtm_2d_data['keypoints'].shape[1] if len(rtm_2d_data['keypoints'].shape) > 2 else target_height / 1080
    
    # Get original video dimensions for proper scaling
    cap_temp = cv2.VideoCapture(video_path)
    orig_width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_temp.release()
    
    scale_x = target_width / orig_width
    scale_y = target_height / orig_height
    
    rtm_2d_kps_scaled = rtm_2d_kps.copy()
    rtm_2d_kps_scaled[:, 0] *= scale_x
    rtm_2d_kps_scaled[:, 1] *= scale_y
    
    wb_2d_data = np.load(wb_2d_path)
    wb_2d_kps = wb_2d_data['keypoints'][frame_idx]
    wb_2d_kps_scaled = wb_2d_kps.copy()
    wb_2d_kps_scaled[:, 0] *= scale_x
    wb_2d_kps_scaled[:, 1] *= scale_y
    
    # Load 3D keypoints
    wb_3d_data = np.load(wb_3d_path)
    wb_3d_kps = wb_3d_data['keypoints_3d'][frame_idx]  # Body joints (first 17 of 133)
    
    magf_3d_data = np.load(magf_3d_path)
    magf_3d_poses = magf_3d_data['poses_3d'][frame_idx]  # H36M-17 format
    
    print(f"Loaded keypoints - RTM 2D: {rtm_2d_kps.shape}, WB 2D: {wb_2d_kps.shape}, WB 3D: {wb_3d_kps.shape}, MAGF 3D: {magf_3d_poses.shape}")
    
    # Print angle comparison tables
    print_angle_comparison_tables(rtm_2d_kps, wb_2d_kps, wb_3d_kps, magf_3d_poses, frame_idx, 
                                  show_joint_info=not args.no_joint_info)
    
    if args.no_viz:
        print("\n✓ Analysis complete (visualization skipped)")
        return
    
    # Create 4-panel visualization
    print("\nGenerating visualizations...")
    
    # Panel 1: RTMPose 2D
    panel1 = draw_2d_skeleton(frame, rtm_2d_kps_scaled, ANALYSIS_JOINTS_COCO,
                              "Panel 1: RTMPose 2D", color=(0, 255, 0))
    
    # Panel 2: Wholebody 2D
    panel2 = draw_2d_skeleton(frame, wb_2d_kps_scaled, ANALYSIS_JOINTS_COCO,
                              "Panel 2: Wholebody 2D", color=(255, 0, 255))
    
    # Panel 3: MAGF 3D
    panel3 = draw_3d_skeleton(magf_3d_poses, ANALYSIS_JOINTS_H36M,
                             "Panel 3: MAGF 3D", is_h36m=True)
    panel3 = cv2.cvtColor(panel3, cv2.COLOR_RGB2BGR)
    panel3 = cv2.resize(panel3, (target_width, target_height))
    
    # Panel 4: Wholebody 3D
    panel4 = draw_3d_skeleton(wb_3d_kps, ANALYSIS_JOINTS_COCO,
                             "Panel 4: Wholebody 3D", is_h36m=False)
    panel4 = cv2.cvtColor(panel4, cv2.COLOR_RGB2BGR)
    panel4 = cv2.resize(panel4, (target_width, target_height))
    
    # Combine into 2x2 grid
    top_row = np.hstack([panel1, panel2])
    bottom_row = np.hstack([panel3, panel4])
    final_img = np.vstack([top_row, bottom_row])
    
    # Add main title
    title_height = 60
    title_img = np.ones((title_height, final_img.shape[1], 3), dtype=np.uint8) * 255
    cv2.putText(title_img, f"Frame {frame_idx} - Joint Angle Analysis", 
               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    final_img = np.vstack([title_img, final_img])
    
    # Save output
    cv2.imwrite(output_path, final_img)
    print(f"\n✓ Visualization saved to: {output_path}")
    print(f"  Image size: {final_img.shape[1]}x{final_img.shape[0]}")
    print("\nRed circles with numbers = Analysis joints (elbows, knees, etc.)")


if __name__ == "__main__":
    main()
