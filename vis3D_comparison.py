"""
3D Pose Comparison Visualization

Creates side-by-side video comparison of:
- Original video
- MotionAGFormer (MAGF) 3D skeleton
- Whole-Body 3D (WB3D) 3D skeleton

Usage:
    python vis3D_comparison.py \
        --video demo_data/videos/dance.mp4 \
        --magf demo_data/outputs/keypoints_3D_magf.npz \
        --wb3d demo_data/outputs/keypoints_3D_wb.npz \
        --output demo_data/outputs/comparison_3d.mp4
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


def render_magf_skeleton(pose_3d, width, height):
    """
    Render MAGF 3D skeleton (H36M format, 17 joints).
    
    Args:
        pose_3d: (17, 3) array of 3D joint positions
        width: Width of output image
        height: Height of output image
    
    Returns:
        img: Rendered skeleton image (RGB)
    """
    # Rotation for better view (from MotionAGFormer demo)
    rot = np.array([0.1407056450843811, -0.1500701755285263, 
                    -0.755240797996521, 0.6223280429840088], dtype='float32')
    
    # Center Hip at origin BEFORE rotation
    pose_3d = pose_3d - pose_3d[0:1, :]
    
    # Apply rotation
    pose_3d = camera_to_world(pose_3d, R=rot, t=0)
    
    # Normalize for display
    pose_3d[:, 2] -= np.min(pose_3d[:, 2])  # Floor at z=0
    max_value = np.max(pose_3d)
    if max_value > 0:
        pose_3d /= max_value
    
    # Create figure
    fig = plt.figure(figsize=(height / 100, height / 100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=15., azim=70)
    
    # H36M skeleton connections
    I = np.array([0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])
    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], dtype=bool)
    
    lcolor = (0, 0, 1)  # Blue for left
    rcolor = (1, 0, 0)  # Red for right
    
    # Draw skeleton
    for i in range(len(I)):
        x, y, z = [np.array([pose_3d[I[i], j], pose_3d[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, color=lcolor if LR[i] else rcolor)
    
    # Set view limits
    RADIUS = 0.72
    RADIUS_Z = 0.7
    xroot, yroot, zroot = pose_3d[0, 0], pose_3d[0, 1], pose_3d[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS_Z + zroot, RADIUS_Z + zroot])
    ax.set_aspect('auto')
    
    # White background
    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)
    
    # Hide labels
    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)
    
    # Add title
    ax.set_title('MAGF (Temporal 3D)', fontsize=10, fontweight='bold', pad=10)
    
    # Convert to image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    
    # Resize to target dimensions
    img = cv2.resize(img, (width, height))
    
    return img


def render_wb3d_skeleton(pose_3d, width, height):
    """
    Render WB3D 3D skeleton (COCO-WholeBody format, 133 joints).
    Displays body joints (first 17) for comparison with MAGF.
    
    Args:
        pose_3d: (133, 3) array of 3D joint positions
        width: Width of output image
        height: Height of output image
    
    Returns:
        img: Rendered skeleton image (RGB)
    """
    # Extract body keypoints (first 17)
    body_pose = pose_3d[:17, :].copy()
    
    # Same rotation as MAGF for fair comparison
    rot = np.array([0.1407056450843811, -0.1500701755285263, 
                    -0.755240797996521, 0.6223280429840088], dtype='float32')
    
    # Center at Hip/Pelvis (average of joints 12 and 13 in COCO body)
    # COCO body: 0=Nose, 5=LShoulder, 6=RShoulder, 11=LHip, 12=RHip
    pelvis = (body_pose[12, :] + body_pose[13, :]) / 2 if len(body_pose) > 13 else body_pose[0, :]
    body_pose = body_pose - pelvis[None, :]
    
    # Apply rotation
    body_pose = camera_to_world(body_pose, R=rot, t=0)
    
    # Normalize for display
    body_pose[:, 2] -= np.min(body_pose[:, 2])  # Floor at z=0
    max_value = np.max(body_pose)
    if max_value > 0:
        body_pose /= max_value
    
    # Create figure
    fig = plt.figure(figsize=(height / 100, height / 100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=15., azim=70)
    
    # COCO body skeleton connections (first 17 joints)
    # 0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar, 5:LShoulder, 6:RShoulder,
    # 7:LElbow, 8:LWrist, 9:RShoulder, 10:RElbow, 11:RWrist,
    # 12:LHip, 13:RHip, 14:LKnee, 15:RKnee, 16:LAnkle
    # Note: COCO-WholeBody body subset follows COCO keypoint format
    coco_connections = [
        (0, 1), (0, 2),  # Nose to eyes
        (1, 3), (2, 4),  # Eyes to ears
        (0, 5), (0, 6),  # Nose to shoulders
        (5, 7), (7, 8),  # Left arm
        (6, 9), (9, 10), # Right arm (note: 9 should map correctly)
        (5, 12), (6, 13), # Shoulders to hips
        (12, 14), (14, 16), # Left leg (note: check indices)
        (13, 15),  # Right leg partial
    ]
    
    # Draw skeleton with green color for differentiation
    for i, j in coco_connections:
        if i < len(body_pose) and j < len(body_pose):
            x = [body_pose[i, 0], body_pose[j, 0]]
            y = [body_pose[i, 1], body_pose[j, 1]]
            z = [body_pose[i, 2], body_pose[j, 2]]
            ax.plot(x, y, z, lw=2, color=(0, 0.8, 0))  # Green
    
    # Draw joints
    ax.scatter(body_pose[:, 0], body_pose[:, 1], body_pose[:, 2], 
               c='orange', s=30, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Set view limits
    RADIUS = 0.72
    RADIUS_Z = 0.7
    xroot, yroot, zroot = 0, 0, 0
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS_Z + zroot, RADIUS_Z + zroot])
    ax.set_aspect('auto')
    
    # White background
    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)
    
    # Hide labels
    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)
    
    # Add title
    ax.set_title('WB3D (Single-Stage)', fontsize=10, fontweight='bold', pad=10)
    
    # Convert to image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    
    # Resize to target dimensions
    img = cv2.resize(img, (width, height))
    
    return img


def create_comparison_video(video_path, magf_path, wb3d_path, output_path, max_frames=None):
    """
    Create 3-panel comparison video: Original | MAGF | WB3D
    
    Args:
        video_path: Path to original video
        magf_path: Path to keypoints_3D_magf.npz
        wb3d_path: Path to keypoints_3D_wb.npz
        output_path: Path to save output video
        max_frames: Maximum frames to render (None = all)
    """
    print("\n" + "=" * 70)
    print("🎬 Creating 3D Pose Comparison Video")
    print("=" * 70)
    
    # Load 3D poses
    print(f"\n📂 Loading 3D pose data...")
    magf_data = np.load(magf_path)
    wb3d_data = np.load(wb3d_path)
    
    poses_magf = magf_data['poses_3d']  # (N, 17, 3)
    poses_wb3d = wb3d_data['keypoints_3d']  # (N, 133, 3)
    
    print(f"   MAGF: {poses_magf.shape} (H36M-17 joints)")
    print(f"   WB3D: {poses_wb3d.shape} (COCO-WholeBody-133 joints)")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n📹 Video Info:")
    print(f"   Resolution: {video_width}x{video_height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {video_frames}")
    
    # Determine total frames to process
    total_frames = min(poses_magf.shape[0], poses_wb3d.shape[0], video_frames)
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    
    print(f"   Processing: {total_frames} frames")
    
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate panel dimensions (3 panels side-by-side)
    panel_width = video_width
    panel_height = video_height
    output_width = panel_width * 3
    output_height = panel_height
    
    print(f"\n🎨 Output Configuration:")
    print(f"   Layout: 3 panels (Original | MAGF | WB3D)")
    print(f"   Panel size: {panel_width}x{panel_height}")
    print(f"   Total size: {output_width}x{output_height}")
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))
    
    print(f"\n⚙️  Rendering frames...")
    
    # Process each frame
    for i in tqdm(range(total_frames), desc="Processing"):
        # Read video frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize video frame if needed
        if frame.shape[1] != panel_width or frame.shape[0] != panel_height:
            frame = cv2.resize(frame, (panel_width, panel_height))
        
        # Render MAGF skeleton
        magf_img = render_magf_skeleton(poses_magf[i], panel_width, panel_height)
        
        # Render WB3D skeleton
        wb3d_img = render_wb3d_skeleton(poses_wb3d[i], panel_width, panel_height)
        
        # Add labels to each panel
        label_height = 40
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        
        # Label original frame
        cv2.putText(frame, "Original Video", (10, 30), font, font_scale, 
                    (255, 255, 255), font_thickness, cv2.LINE_AA)
        cv2.putText(frame, "Original Video", (10, 30), font, font_scale, 
                    (0, 0, 0), 1, cv2.LINE_AA)
        
        # Label MAGF
        cv2.putText(magf_img, "MotionAGFormer (MAGF)", (10, 30), font, font_scale, 
                    (255, 255, 255), font_thickness, cv2.LINE_AA)
        cv2.putText(magf_img, "MotionAGFormer (MAGF)", (10, 30), font, font_scale, 
                    (0, 0, 255), 1, cv2.LINE_AA)
        
        # Label WB3D
        cv2.putText(wb3d_img, "Whole-Body 3D (WB3D)", (10, 30), font, font_scale, 
                    (255, 255, 255), font_thickness, cv2.LINE_AA)
        cv2.putText(wb3d_img, "Whole-Body 3D (WB3D)", (10, 30), font, font_scale, 
                    (0, 255, 0), 1, cv2.LINE_AA)
        
        # Add frame counter
        frame_text = f"Frame: {i+1}/{total_frames}"
        cv2.putText(frame, frame_text, (10, panel_height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Combine panels horizontally
        combined = np.hstack([frame, magf_img, wb3d_img])
        
        # Write frame
        out.write(combined)
    
    cap.release()
    out.release()
    
    size_mb = output_path.stat().st_size / (1024 ** 2)
    
    print(f"\n✅ Comparison video saved!")
    print(f"   Path: {output_path}")
    print(f"   Size: {size_mb:.2f} MB")
    print(f"   Resolution: {output_width}x{output_height}")
    print(f"   Frames: {total_frames}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Create 3D pose comparison video (Original | MAGF | WB3D)"
    )
    parser.add_argument('--video', type=str, required=True,
                        help='Path to original video')
    parser.add_argument('--magf', type=str, required=True,
                        help='Path to keypoints_3D_magf.npz')
    parser.add_argument('--wb3d', type=str, required=True,
                        help='Path to keypoints_3D_wb.npz')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output video (default: comparison_3d.mp4)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum frames to process (default: all)')
    
    args = parser.parse_args()
    
    # Check inputs
    video_path = Path(args.video)
    magf_path = Path(args.magf)
    wb3d_path = Path(args.wb3d)
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return 1
    
    if not magf_path.exists():
        print(f"❌ MAGF file not found: {magf_path}")
        return 1
    
    if not wb3d_path.exists():
        print(f"❌ WB3D file not found: {wb3d_path}")
        return 1
    
    # Setup output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = video_path.parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "comparison_3d.mp4"
    
    # Create comparison video
    success = create_comparison_video(
        video_path,
        magf_path,
        wb3d_path,
        output_path,
        max_frames=args.max_frames
    )
    
    if success:
        print("\n" + "=" * 70)
        print("✅ SUCCESS!")
        print("=" * 70)
        print(f"\nComparison video created successfully!")
        print(f"Open: {output_path}")
    else:
        print("\n❌ Failed to create comparison video")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
