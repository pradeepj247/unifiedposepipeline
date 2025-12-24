"""
WB3D Visualization Debugger

Standalone visualization for Whole-Body 3D (WB3D) skeleton with MotionAGFormer-style normalization.
Properly renders grounded skeleton with adaptive Z-axis limits to prevent head clipping.

Features:
- MotionAGFormer normalization (max of positive values, not max absolute)
- Adaptive Z-axis limits ensure full skeleton visibility
- Grounded skeleton (feet at z=0)
- Color-coded bones (blue=left, red=right, green=center)
- Left panel: 2D keypoints overlaid on video frame
- Right panel: 3D skeleton with proper grounding and normalization

Usage:
    python vis_wb3d.py \
        --video demo_data/videos/dance.mp4 \
        --kps2d demo_data/outputs/kps_2d_wb3d.npz \
        --kps3d demo_data/outputs/kps_3d_wb3d.npz \
        --output demo_data/outputs/debug_wb3d.mp4 \
        --max-frames 120
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects quaternion format [w, x, y, z] and vector [x, y, z].
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    
    qvec = q[..., 1:]  # [x, y, z] part of quaternion
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2 * (q[..., :1] * uv + uuv)


def camera_to_world(X, R, t):
    """
    Apply camera rotation to 3D keypoints using quaternion rotation.
    Args:
        X (Nx3): 3D points
        R (4,): Quaternion rotation [w, x, y, z]
        t: Translation (scalar or vector)
    Returns: Rotated points
    """
    R_tiled = np.tile(R, (X.shape[0], 1))
    return qrot(R_tiled, X) + t


def render_wb3d_skeleton(pose_3d, width, height, frame_idx=0, debug=True):
    """
    Render WB3D 3D skeleton (COCO-WholeBody format, 133 joints).
    Focus on body joints (first 17) with proper grounding and connections.
    
    Args:
        pose_3d: (133, 3) array of 3D joint positions
        width: Width of output image
        height: Height of output image
        frame_idx: Current frame index (for debug info)
        debug: If True, print debug information
    
    Returns:
        img: Rendered skeleton image (RGB)
    """
    # Extract body keypoints (first 23 joints: body + feet)
    body_pose = pose_3d[:23, :].copy()
    
    if debug and frame_idx == 0:
        print("\n" + "="*70)
        print("DEBUG INFO - WB3D Skeleton")
        print("="*70)
        print(f"Original body pose shape: {body_pose.shape}")
        print(f"Body joint positions (first 5):")
        for i in range(min(5, len(body_pose))):
            print(f"  Joint {i}: {body_pose[i]}")
    
    # COCO-17 Body Joint Order:
    # 0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar, 
    # 5:LShoulder, 6:RShoulder, 7:LElbow, 8:LWrist, 
    # 9:RShoulder, 10:RElbow, 11:RWrist, 
    # 12:LHip, 13:RHip, 14:LKnee, 15:RKnee, 16:LAnkle
    
    # Find pelvis (average of hips) for centering
    if len(body_pose) > 12:
        # User's convention (0-indexed): LHip=12, RHip=11
        left_hip = body_pose[12, :]
        right_hip = body_pose[11, :]
        pelvis = (left_hip + right_hip) / 2
    else:
        pelvis = body_pose[0, :]  # Fallback to nose
    
    if debug and frame_idx == 0:
        print(f"\nPelvis position: {pelvis}")
        print(f"Right Hip (12): {right_hip}")
        print(f"Left Hip (13): {left_hip}")
    
    # Center at pelvis
    body_pose = body_pose - pelvis[None, :]
    
    # Apply same rotation as MAGF for consistency
    rot = np.array([0.1407056450843811, -0.1500701755285263, 
                    -0.755240797996521, 0.6223280429840088], dtype='float32')
    body_pose = camera_to_world(body_pose, R=rot, t=0)
    
    # Ground the skeleton at z=0 (CRITICAL FIX for floating issue)
    min_z = np.min(body_pose[:, 2])
    body_pose[:, 2] -= min_z  # Shift so lowest point touches ground
    
    if debug and frame_idx == 0:
        print(f"\nAfter centering and rotation:")
        print(f"  Min Z before grounding: {min_z}")
        print(f"  Z range after grounding: [{np.min(body_pose[:, 2]):.3f}, {np.max(body_pose[:, 2]):.3f}]")
        print(f"  X range: [{np.min(body_pose[:, 0]):.3f}, {np.max(body_pose[:, 0]):.3f}]")
        print(f"  Y range: [{np.min(body_pose[:, 1]):.3f}, {np.max(body_pose[:, 1]):.3f}]")
    
    # Normalize using MotionAGFormer approach (max of positive values, not max absolute)
    max_value = np.max(body_pose)
    if max_value > 0:
        body_pose /= max_value
    
    if debug and frame_idx == 0:
        print(f"\nAfter normalization:")
        print(f"  Normalization factor: {max_value:.3f}")
        print(f"  X range: [{np.min(body_pose[:, 0]):.3f}, {np.max(body_pose[:, 0]):.3f}]")
        print(f"  Y range: [{np.min(body_pose[:, 1]):.3f}, {np.max(body_pose[:, 1]):.3f}]")
        print(f"  Z range: [{np.min(body_pose[:, 2]):.3f}, {np.max(body_pose[:, 2]):.3f}]")
    
    # Create figure
    fig = plt.figure(figsize=(height / 100, height / 100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=15., azim=70)
    
    # User's corrected connections (0-indexed, represents 1-23 user convention)
    # User says: 1→4,5 | 2→3,4 | 3→5 | 6↔7 | 6→8,12 | 7→9,13 | 8→10 | 9→11 | 
    #            12→13,14 | 14→16 | 13→15 | 15→17 | 16→18,19,20 | 17→21,22,23
    # Neck: 1→(midpoint of 6,7) approximated by connecting 1→6 and 1→7
    # Converting to 0-indexed: subtract 1
    user_connections = [
        # Head connections
        (0, 3),   # 1→4: Nose to REar
        (0, 4),   # 1→5: Nose to LEar
        (1, 2),   # 2→3: LEye to REye
        (1, 3),   # 2→4: LEye to REar
        (2, 4),   # 3→5: REye to LEar
        
        # Neck (approximation - nose to both shoulders)
        (0, 5),   # 1→6: Nose to RShoulder (neck approximation)
        (0, 6),   # 1→7: Nose to LShoulder (neck approximation)
        
        # Shoulders
        (5, 6),   # 6↔7: RShoulder to LShoulder
        
        # Torso
        (5, 7),   # 6→8: RShoulder to RElbow
        (5, 11),  # 6→12: RShoulder to RHip
        (6, 8),   # 7→9: LShoulder to LElbow
        (6, 12),  # 7→13: LShoulder to LHip
        
        # Arms
        (7, 9),   # 8→10: RElbow to RWrist
        (8, 10),  # 9→11: LElbow to LWrist
        
        # Hip connection
        (11, 12), # 12→13: RHip to LHip
        
        # Right Leg
        (11, 13), # 12→14: RHip to RKnee
        (13, 15), # 14→16: RKnee to RAnkle
        
        # Left Leg
        (12, 14), # 13→15: LHip to LKnee
        (14, 16), # 15→17: LKnee to LAnkle
        
        # Right Foot
        (15, 17), # 16→18: RAnkle to RBigToe
        (15, 18), # 16→19: RAnkle to RSmallToe
        (15, 19), # 16→20: RAnkle to RHeel
        
        # Left Foot
        (16, 20), # 17→21: LAnkle to LBigToe
        (16, 21), # 17→22: LAnkle to LSmallToe
        (16, 22), # 17→23: LAnkle to LHeel
    ]
    
    if debug and frame_idx == 0:
        print(f"\nSkeleton connections: {len(user_connections)} bones")
        print("Connection list (User's 1-23 convention):")
        joint_names = [
            'Nose', 'LEye', 'REye', 'LEar', 'REar',
            'RShoulder', 'LShoulder', 'RElbow', 'LElbow',
            'RWrist', 'LWrist', 'RHip', 'LHip',
            'RKnee', 'LKnee', 'RAnkle', 'LAnkle',
            'RBigToe', 'RSmallToe', 'RHeel',
            'LBigToe', 'LSmallToe', 'LHeel'
        ]
        for i, (a, b) in enumerate(user_connections):
            if a < len(joint_names) and b < len(joint_names):
                print(f"  {i}: {joint_names[a]}({a+1}) -> {joint_names[b]}({b+1})")
    
    # Draw skeleton connections
    # Color: Left side=blue, Right side=red, Center=green
    # Left: 2,4,6,8,10,12,14,16,20,21,22 (0-indexed)
    # Right: 1,3,5,7,9,11,13,15,17,18,19 (0-indexed)
    left_joints = [2, 4, 6, 8, 10, 12, 14, 16, 20, 21, 22]
    right_joints = [1, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19]
    
    for i, j in user_connections:
        if i < len(body_pose) and j < len(body_pose):
            x = [body_pose[i, 0], body_pose[j, 0]]
            y = [body_pose[i, 1], body_pose[j, 1]]
            z = [body_pose[i, 2], body_pose[j, 2]]
            
            if i in left_joints or j in left_joints:
                color = (0, 0, 1)  # Blue for left
            elif i in right_joints or j in right_joints:
                color = (1, 0, 0)  # Red for right
            else:
                color = (0, 0.8, 0)  # Green for center
            
            ax.plot(x, y, z, lw=2.5, color=color, alpha=0.8)
    
    # Draw joints
    colors = ['yellow' if i == 0 else 'orange' for i in range(len(body_pose))]
    ax.scatter(body_pose[:, 0], body_pose[:, 1], body_pose[:, 2], 
               c=colors, s=40, alpha=0.9, edgecolors='black', linewidth=1)
    
    # Add joint numbers for debugging (only first 3 frames, with 1-based numbering)
    if debug and frame_idx < 3:
        for i in range(len(body_pose)):
            ax.text(body_pose[i, 0], body_pose[i, 1], body_pose[i, 2], 
                   str(i+1), fontsize=14, color='black', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Set view limits (MotionAGFormer approach with adaptive Z-axis)
    RADIUS = 0.72
    RADIUS_Z = max(1.1, np.max(body_pose[:, 2]) * 1.1)  # Adaptive: ensure full skeleton visible
    xroot, yroot, zroot = 0, 0, 0
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([0, RADIUS_Z + zroot])  # From ground up, adaptive height
    ax.set_aspect('auto')
    
    if debug and frame_idx == 0:
        print(f"\nVisualization limits:")
        print(f"  X: [{-RADIUS:.2f}, {RADIUS:.2f}]")
        print(f"  Y: [{-RADIUS:.2f}, {RADIUS:.2f}]")
        print(f"  Z: [0.00, {RADIUS_Z:.2f}] (adaptive)")
    
    # White background
    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)
    
    # Labels
    ax.set_xlabel('X', fontsize=8)
    ax.set_ylabel('Y', fontsize=8)
    ax.set_zlabel('Z', fontsize=8)
    ax.tick_params(labelsize=6)
    
    # Add title
    ax.set_title('3D Skeleton', fontsize=11, fontweight='bold', pad=10)
    
    # Convert to image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    
    # Resize to target dimensions
    img = cv2.resize(img, (width, height))
    
    return img


def create_wb3d_visualization(video_path, kps2d_path, kps3d_path, output_path, max_frames=120):
    """
    Create visualization video for WB3D skeleton debugging.
    
    Args:
        video_path: Path to original video
        kps2d_path: Path to kps_2d_wb3d.npz (2D keypoints)
        kps3d_path: Path to kps_3d_wb3d.npz (3D keypoints)
        output_path: Path to save output video
        max_frames: Maximum frames to render (default: 120 for debugging)
    """
    print("\n" + "=" * 70)
    print("🐛 WB3D Skeleton Visualization Debugger")
    print("=" * 70)
    
    # Load 2D and 3D keypoints
    print(f"\n📂 Loading keypoints...")
    kps2d_data = np.load(kps2d_path)
    kps3d_data = np.load(kps3d_path)
    
    poses_2d = kps2d_data['keypoints']  # (N, 133, 2)
    poses_3d = kps3d_data['keypoints_3d']  # (N, 133, 3) - Note: key is 'keypoints_3d'
    
    print(f"   2D keypoints: {poses_2d.shape}")
    print(f"   3D keypoints: {poses_3d.shape}")
    print(f"   Total joints: 133 (Body: 23, Face: 68, Left hand: 21, Right hand: 21)")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n📹 Video: {video_width}x{video_height}, {fps} fps, {video_frames} frames")
    
    # Limit frames
    total_frames = min(poses_2d.shape[0], poses_3d.shape[0], video_frames, max_frames)
    print(f"   Processing: {total_frames} frames (debug mode)")
    
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Side-by-side layout
    panel_width = video_width
    panel_height = video_height
    output_width = panel_width * 2
    output_height = panel_height
    
    # Body skeleton connections (first 23 joints)
    BODY_EDGES = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
        (15, 17), (15, 18), (15, 19),  # Left foot
        (16, 20), (16, 21), (16, 22),  # Right foot
    ]
    
    print(f"\n🎨 Output: {output_width}x{output_height} (2D on Video | 3D Skeleton)")
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))
    
    print(f"\n⚙️  Rendering frames...")
    
    # Process frames
    for i in tqdm(range(total_frames), desc="Rendering"):
        # Read video frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize if needed
        if frame.shape[1] != panel_width or frame.shape[0] != panel_height:
            frame = cv2.resize(frame, (panel_width, panel_height))
        
        # LEFT PANEL: Draw 2D keypoints on video frame
        frame_2d = frame.copy()
        kpts_2d = poses_2d[i][:23]  # First 23 body joints
        
        # Draw skeleton connections
        for start, end in BODY_EDGES:
            x1, y1 = int(kpts_2d[start, 0]), int(kpts_2d[start, 1])
            x2, y2 = int(kpts_2d[end, 0]), int(kpts_2d[end, 1])
            cv2.line(frame_2d, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw keypoints
        for idx in range(23):
            x, y = int(kpts_2d[idx, 0]), int(kpts_2d[idx, 1])
            cv2.circle(frame_2d, (x, y), 4, (0, 0, 255), -1)
        
        # RIGHT PANEL: Render 3D skeleton
        wb3d_img = render_wb3d_skeleton(
            poses_3d[i], 
            panel_width, 
            panel_height, 
            frame_idx=i,
            debug=False  # No debug output during rendering
        )
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame_2d, f"2D Keypoints (Frame {i+1}/{total_frames})", (10, 30), font, 0.8, 
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(wb3d_img, f"3D Skeleton (Frame {i+1}/{total_frames})", (10, 30), 
                    font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Combine panels
        combined = np.hstack([frame_2d, wb3d_img])
        out.write(combined)
    
    cap.release()
    out.release()
    
    size_mb = output_path.stat().st_size / (1024 ** 2)
    
    print(f"\n✅ Debug video saved!")
    print(f"   Path: {output_path}")
    print(f"   Size: {size_mb:.2f} MB")
    print(f"   Frames: {total_frames}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="WB3D Skeleton Visualization Debugger"
    )
    parser.add_argument('--video', type=str, required=True,
                        help='Path to original video')
    parser.add_argument('--kps2d', type=str, required=True,
                        help='Path to kps_2d_wb3d.npz (2D keypoints)')
    parser.add_argument('--kps3d', type=str, required=True,
                        help='Path to kps_3d_wb3d.npz (3D keypoints)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output video (default: debug_wb3d.mp4)')
    parser.add_argument('--max-frames', type=int, default=120,
                        help='Maximum frames to process (default: 120 for debugging)')
    
    args = parser.parse_args()
    
    # Check inputs
    video_path = Path(args.video)
    kps2d_path = Path(args.kps2d)
    kps3d_path = Path(args.kps3d)
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return 1
    
    if not kps2d_path.exists():
        print(f"❌ 2D keypoints file not found: {kps2d_path}")
        return 1
    
    if not kps3d_path.exists():
        print(f"❌ 3D keypoints file not found: {kps3d_path}")
        return 1
    
    # Setup output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = video_path.parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "debug_wb3d.mp4"
    
    # Create debug visualization
    success = create_wb3d_visualization(
        video_path,
        kps2d_path,
        kps3d_path,
        output_path,
        max_frames=args.max_frames
    )
    
    if success:
        print("\n" + "=" * 70)
        print("✅ DEBUG VIDEO CREATED!")
        print("=" * 70)
    else:
        print("\n❌ Failed to create debug video")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
