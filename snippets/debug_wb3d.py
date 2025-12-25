"""
Debug Wholebody3D - Single Frame 2D+3D Comparison

Tests Wholebody3D pose estimation on a single frame with 2-panel comparison:
- Panel 1: 2D keypoints on image (first 23 body joints)
- Panel 2: 3D skeleton with MotionAGFormer-style normalization (production-ready)

Usage:
    python debug_wb3d.py --video demo_data/videos/dance.mp4
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

REPO_ROOT = Path(__file__).parent
PARENT_DIR = REPO_ROOT.parent
MODELS_DIR = PARENT_DIR / "models"

# First 23 joints skeleton (body keypoints - COCO-17 + foot keypoints)
# These are the body joints from the 133-keypoint Wholebody format
BODY_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
    (15, 17), (15, 18), (15, 19),  # Left foot (ankle to toes/heel)
    (16, 20), (16, 21), (16, 22),  # Right foot (ankle to toes/heel)
]


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


def main():
    parser = argparse.ArgumentParser(description="Debug Wholebody3D - Single Frame 2D+3D Test")
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--output', type=str, default='demo_data/outputs/debug_wb3d_frame1.png',
                        help='Output image path')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("DEBUG: Wholebody3D Single Frame Test (2D + 3D)")
    print("=" * 70)
    
    # Initialize YOLO
    print("\n📦 Loading YOLO detector...")
    from ultralytics import YOLO
    yolo_path = MODELS_DIR / "yolo" / "yolov8s.pt"
    yolo = YOLO(str(yolo_path))
    print(f"   ✅ Loaded {yolo_path.name}")
    
    # Initialize Wholebody3D (RTMPose3d)
    print("\n📦 Loading Wholebody3D...")
    sys.path.insert(0, str(REPO_ROOT / "lib"))
    from rtmlib.tools import RTMPose3d
    
    model_path = "/content/models/wb3d/rtmw3d-l.onnx"
    wb_model = RTMPose3d(
        onnx_model=model_path,
        model_input_size=(288, 384),
        backend='onnxruntime',
        device='cuda'
    )
    print(f"   ✅ RTMPose3d loaded (133 keypoints, 2D+3D)")
    
    # Load video and get first frame
    print(f"\n🎬 Loading video: {args.video}")
    video_path = Path(args.video)
    if not video_path.exists():
        video_path = REPO_ROOT / args.video
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return 1
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Could not read first frame")
        return 1
    
    height, width = frame.shape[:2]
    print(f"   ✅ Frame loaded: {width}x{height}")
    
    # Stage 1: Detect person
    print("\n🎯 Stage 1: Detecting person...")
    results = yolo(frame, classes=[0], verbose=False)
    
    boxes = []
    for result in results:
        for box in result.boxes:
            if box.conf[0] >= 0.5:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
    
    if not boxes:
        print("❌ No person detected")
        return 1
    
    # Use largest box
    boxes_sorted = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    bbox = boxes_sorted[0]
    print(f"   ✅ Detected person: bbox {bbox}")
    
    # Stage 2: Estimate pose (2D + 3D)
    print("\n🎯 Stage 2: Estimating pose (Wholebody3D)...")
    # RTMPose3d returns: keypoints_3d, scores, keypoints_simcc, keypoints_2d
    keypoints_3d, scores, keypoints_simcc, keypoints_2d = wb_model(frame, bboxes=[bbox])
    
    if len(keypoints_2d) == 0:
        print("❌ No pose detected")
        return 1
    
    kpts_2d = keypoints_2d[0]  # (133, 2) - 2D keypoints
    kpts_3d = keypoints_3d[0]  # (133, 3) - 3D keypoints (X, Y, Z)
    
    print(f"   ✅ Pose detected:")
    print(f"      - 2D keypoints: {kpts_2d.shape}")
    print(f"      - 3D keypoints: {kpts_3d.shape}")
    
    # Extract first 23 keypoints (body only: 0-22)
    kpts_2d_body = kpts_2d[:23]  # (23, 2)
    kpts_3d_body = kpts_3d[:23]  # (23, 3)
    
    # Create figure with TWO panels
    fig = plt.figure(figsize=(16, 8))
    
    # ===== PANEL 1: 2D Image with keypoints =====
    ax1 = fig.add_subplot(1, 2, 1)
    
    # Convert BGR to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax1.imshow(frame_rgb)
    
    # Draw bbox
    bbox_rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                               fill=False, edgecolor='green', linewidth=2)
    ax1.add_patch(bbox_rect)
    
    # Draw keypoints with numbers (no skeleton lines to reduce clutter)
    # Joints that need special offset to avoid overlap
    offset_joints = {
        # Head cluster (0-4): offset upward and to sides
        0: (15, -30),   # nose
        1: (-25, -20),  # left_eye
        2: (25, -20),   # right_eye
        3: (-35, -10),  # left_ear
        4: (35, -10),   # right_ear
        # Foot cluster (17-22): offset downward
        17: (-25, 25),  # left_big_toe
        18: (-25, 35),  # left_small_toe
        19: (-25, 45),  # left_heel
        20: (25, 25),   # right_big_toe
        21: (25, 35),   # right_small_toe
        22: (25, 45),   # right_heel
    }
    
    for idx in range(23):
        x, y = kpts_2d_body[idx, 0], kpts_2d_body[idx, 1]
        
        # Draw point
        ax1.plot(x, y, 'ro', markersize=4)
        
        # Check if this joint needs special offset
        if idx in offset_joints:
            offset_x, offset_y = offset_joints[idx]
            text_x, text_y = x + offset_x, y + offset_y
            
            # Draw a thin line from point to text
            ax1.plot([x, text_x], [y, text_y], 'y-', linewidth=0.5, alpha=0.6)
            
            # Draw number at offset position
            ax1.text(text_x, text_y, str(idx), fontsize=4, color='white',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        else:
            # Normal offset for non-crowded joints
            ax1.text(x + 5, y - 5, str(idx), fontsize=4, color='white',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    ax1.set_title('Panel 1: 2D Keypoints (First 23 - Body)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # ===== PANEL 2: 3D Visualization (MotionAGFormer-style normalization) =====
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # Apply transformations with MotionAGFormer-style normalization
    kpts_3d_norm = kpts_3d_body.copy()
    
    # Center at pelvis
    left_hip = kpts_3d_norm[11, :]
    right_hip = kpts_3d_norm[12, :]
    pelvis = (left_hip + right_hip) / 2
    kpts_3d_norm = kpts_3d_norm - pelvis[None, :]
    
    # Apply rotation
    rot = np.array([0.1407056450843811, -0.1500701755285263, 
                    -0.755240797996521, 0.6223280429840088], dtype='float32')
    kpts_3d_norm = camera_to_world(kpts_3d_norm, R=rot, t=0)
    
    # Ground at z=0
    min_z = np.min(kpts_3d_norm[:, 2])
    kpts_3d_norm[:, 2] -= min_z
    
    # Normalize (MotionAGFormer approach: max of positive values, not max absolute)
    max_value = np.max(kpts_3d_norm)
    if max_value > 0:
        kpts_3d_norm /= max_value
    
    # Extract normalized coordinates
    xs = kpts_3d_norm[:, 0]
    ys = kpts_3d_norm[:, 1]
    zs = kpts_3d_norm[:, 2]
    
    # Draw skeleton edges with color coding
    # Left: blue, Right: red, Center: green
    left_joints = [1, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19]  # 0-indexed
    right_joints = [2, 4, 6, 8, 10, 12, 14, 16, 20, 21, 22]  # 0-indexed
    
    for start, end in BODY_EDGES:
        if start < 23 and end < 23:
            if start in left_joints or end in left_joints:
                color = (0, 0, 1)  # Blue for left
            elif start in right_joints or end in right_joints:
                color = (1, 0, 0)  # Red for right
            else:
                color = (0, 0.8, 0)  # Green for center
            
            ax2.plot([xs[start], xs[end]], 
                     [ys[start], ys[end]], 
                     [zs[start], zs[end]], 
                     color=color, linewidth=2.5, alpha=0.8)
    
    # Draw keypoints
    colors = ['yellow' if i == 0 else 'orange' for i in range(len(xs))]
    ax2.scatter(xs, ys, zs, c=colors, marker='o', s=40, alpha=0.9,
                edgecolors='black', linewidth=1)
    
    # Add numbers
    for idx in range(23):
        ax2.text(xs[idx], ys[idx], zs[idx], str(idx), fontsize=5,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('X', fontsize=10)
    ax2.set_ylabel('Y', fontsize=10)
    ax2.set_zlabel('Z', fontsize=10)
    ax2.set_title('Panel 2: 3D Skeleton (MotionAGFormer Normalization)', fontsize=12, fontweight='bold')
    
    # Set viewing angle
    ax2.view_init(elev=15., azim=70)
    
    # Set adaptive limits (MotionAGFormer approach)
    RADIUS = 0.72
    RADIUS_Z = max(1.1, np.max(zs) * 1.1)  # Adaptive: ensure full skeleton visible
    xroot, yroot, zroot = 0, 0, 0
    ax2.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax2.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax2.set_zlim3d([0, RADIUS_Z + zroot])  # From ground up, adaptive height
    ax2.set_aspect('auto')
    
    # White background
    white = (1.0, 1.0, 1.0, 0.0)
    ax2.xaxis.set_pane_color(white)
    ax2.yaxis.set_pane_color(white)
    ax2.zaxis.set_pane_color(white)
    
    plt.tight_layout()
    
    # Save output
    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved: {output_path}")
    
    # ===== COMPARISON TABLE: 2-Panel Joint Coordinates =====
    print("\n" + "=" * 90)
    print("2-PANEL COMPARISON TABLE: Joint Coordinates")
    print("=" * 90)
    
    # Joint names
    body_joint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
        "left_big_toe", "left_small_toe", "left_heel",
        "right_big_toe", "right_small_toe", "right_heel"
    ]
    
    # Header
    print(f"{'Joint':5s} {'Name':20s} | {'Panel 1: 2D (pixels)':30s} | {'Panel 2: 3D (normalized)':35s}")
    print("-" * 90)
    
    for idx in range(23):
        name = body_joint_names[idx] if idx < len(body_joint_names) else f"joint_{idx}"
        
        # Panel 1: 2D
        x_2d, y_2d = kpts_2d_body[idx]
        panel1 = f"x={x_2d:6.1f}, y={y_2d:6.1f}"
        
        # Panel 2: 3D normalized
        x_p2, y_p2, z_p2 = xs[idx], ys[idx], zs[idx]
        panel2 = f"x={x_p2:8.3f}, y={y_p2:8.3f}, z={z_p2:8.3f}"
        
        print(f"{idx:5d} {name:20s} | {panel1:30s} | {panel2:35s}")
    
    print("=" * 90)
    
    # Statistics comparison
    print("\n" + "=" * 80)
    print("STATISTICS - Panel 2 (3D Normalized)")
    print("=" * 80)
    
    print(f"  X range: [{np.min(xs):8.3f}, {np.max(xs):8.3f}]")
    print(f"  Y range: [{np.min(ys):8.3f}, {np.max(ys):8.3f}]")
    print(f"  Z range: [{np.min(zs):8.3f}, {np.max(zs):8.3f}]")
    print(f"  Normalization factor: {max_value:8.3f}")
    print(f"  Adaptive Z-limit: {RADIUS_Z:8.3f}")
    
    print("\n" + "=" * 80)
    print("📋 PANEL NOTES")
    print("=" * 80)
    print("""
Panel 1 (2D):
- Shows original 2D keypoints in pixel coordinates
- Overlaid on the video frame
- First 23 body joints only (COCO-17 + 6 foot keypoints)

Panel 2 (3D - MotionAGFormer-style normalization):
- Centered at pelvis (average of left and right hip)
- Rotated by quaternion for proper orientation
- Grounded at z=0 (feet touch floor)
- Normalized by MAX value (not max absolute) for consistent scale
- Adaptive Z-axis limit ensures full skeleton is visible
- Color-coded bones: Blue=left, Red=right, Green=center
- Production-ready visualization approach

VISUALIZATION PARAMETERS:
- X/Y limits: [-0.72, 0.72] (symmetric around pelvis)
- Z limit: [0, adaptive] (grounded with 10% padding above head)
- View angle: elevation=15°, azimuth=70°
""")
    print("=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
