"""
WB3D Single Frame Visualizer - User's Corrected Convention

Generates skeleton rendering using user's verified joint connections (1-23).
Includes body joints (1-17) and foot details (18-23).

Usage:
    python vis_wb3d_single_frame.py \
        --wb3d demo_data/outputs/keypoints_3D_wb.npz \
        --output demo_data/outputs
"""

import sys
import argparse
from pathlib import Path
import numpy as np
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


def render_skeleton(body_pose, output_path, title, joint_offset=0):
    """
    Render a single 3D skeleton with joint numbers.
    
    Args:
        body_pose: (23, 3) array of 3D joint positions (body + feet)
        output_path: Path to save PNG
        title: Title for the plot
        joint_offset: Offset to add to joint numbers (typically 1 for user's 1-based convention)
    """
    # Find pelvis (average of hips)
    # User's convention (0-indexed in array): LHip=11, RHip=12
    left_hip = body_pose[11, :]
    right_hip = body_pose[12, :]
    pelvis = (left_hip + right_hip) / 2
    
    print(f"\n{title}:")
    print(f"  Pelvis: {pelvis}")
    print(f"  LHip (11): {left_hip}")
    print(f"  RHip (12): {right_hip}")
    
    # Center at pelvis
    body_pose = body_pose - pelvis[None, :]
    
    # Apply rotation
    rot = np.array([0.1407056450843811, -0.1500701755285263, 
                    -0.755240797996521, 0.6223280429840088], dtype='float32')
    body_pose = camera_to_world(body_pose, R=rot, t=0)
    
    # Ground at z=0
    min_z = np.min(body_pose[:, 2])
    body_pose[:, 2] -= min_z
    
    # Print RAW VALUES before normalization
    print(f"\n  RAW VALUES (after centering and grounding):")
    
    x_min_idx = np.argmin(body_pose[:, 0])
    x_max_idx = np.argmax(body_pose[:, 0])
    y_min_idx = np.argmin(body_pose[:, 1])
    y_max_idx = np.argmax(body_pose[:, 1])
    z_min_idx = np.argmin(body_pose[:, 2])
    z_max_idx = np.argmax(body_pose[:, 2])
    
    print(f"    X: ranges from {body_pose[x_min_idx, 0]:.2f} for point {x_min_idx+1} to {body_pose[x_max_idx, 0]:.2f} for point {x_max_idx+1}")
    print(f"    Y: ranges from {body_pose[y_min_idx, 1]:.2f} for point {y_min_idx+1} to {body_pose[y_max_idx, 1]:.2f} for point {y_max_idx+1}")
    print(f"    Z: ranges from {body_pose[z_min_idx, 2]:.2f} for point {z_min_idx+1} to {body_pose[z_max_idx, 2]:.2f} for point {z_max_idx+1}")
    
    # Find absolute max value across all axes
    abs_values = np.abs(body_pose)
    max_idx = np.unravel_index(np.argmax(abs_values), abs_values.shape)
    max_value = abs_values[max_idx]
    axis_names = ['X', 'Y', 'Z']
    
    print(f"\n    MAX VALUE was: {max_value:.2f} in {axis_names[max_idx[1]]} axis for point {max_idx[0]+1}")
    
    # Normalize for display
    if max_value > 0:
        body_pose /= max_value
    
    # Print NORMALIZED VALUES after normalization
    print(f"\n  NORMALIZED VALUES:")
    
    nx_min_idx = np.argmin(body_pose[:, 0])
    nx_max_idx = np.argmax(body_pose[:, 0])
    ny_min_idx = np.argmin(body_pose[:, 1])
    ny_max_idx = np.argmax(body_pose[:, 1])
    nz_min_idx = np.argmin(body_pose[:, 2])
    nz_max_idx = np.argmax(body_pose[:, 2])
    
    print(f"    X range: {body_pose[nx_min_idx, 0]:+.3f} (point {nx_min_idx+1}) to {body_pose[nx_max_idx, 0]:+.3f} (point {nx_max_idx+1})")
    print(f"    Y range: {body_pose[ny_min_idx, 1]:+.3f} (point {ny_min_idx+1}) to {body_pose[ny_max_idx, 1]:+.3f} (point {ny_max_idx+1})")
    print(f"    Z range: {body_pose[nz_min_idx, 2]:+.3f} (point {nz_min_idx+1}) to {body_pose[nz_max_idx, 2]:+.3f} (point {nz_max_idx+1})")
    
    # Print coordinates AFTER normalization (matches plot values)
    print(f"\n  NORMALIZED coordinates (0-1 range, matching plot):")
    print(f"\n  TOP (Eyes):")
    print(f"    Joint  2 (LEye):   [{body_pose[1, 0]:+.3f}, {body_pose[1, 1]:+.3f}, {body_pose[1, 2]:+.3f}]")
    print(f"    Joint  3 (REye):   [{body_pose[2, 0]:+.3f}, {body_pose[2, 1]:+.3f}, {body_pose[2, 2]:+.3f}]")
    print(f"\n  MID (Hips):")
    print(f"    Joint 12 (RHip):   [{body_pose[11, 0]:+.3f}, {body_pose[11, 1]:+.3f}, {body_pose[11, 2]:+.3f}]")
    print(f"    Joint 13 (LHip):   [{body_pose[12, 0]:+.3f}, {body_pose[12, 1]:+.3f}, {body_pose[12, 2]:+.3f}]")
    print(f"\n  BOTTOM (Ankles):")
    print(f"    Joint 16 (RAnkle): [{body_pose[15, 0]:+.3f}, {body_pose[15, 1]:+.3f}, {body_pose[15, 2]:+.3f}]")
    print(f"    Joint 17 (LAnkle): [{body_pose[16, 0]:+.3f}, {body_pose[16, 1]:+.3f}, {body_pose[16, 2]:+.3f}]")
    print(f"\n  (All values divided by max_value={max_value:.2f})")
    
    # Create figure
    fig = plt.figure(figsize=(10, 10), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=15., azim=70)
    
    # User's corrected connections (0-indexed in array, but represents 1-based user convention)
    # User says: 1→4, 1→5, 2→3, 2→4, 3→5, 6→8, 6→12, 7→9, 7→13, 8→10, 9→11, 
    #            12→13, 12→14, 14→16, 13→15, 15→17, 17→21,22,23, 16→18,19,20
    # Converting to 0-indexed: subtract 1 from each
    user_connections = [
        # Head connections
        (0, 3),   # 1→4: Nose to REar
        (0, 4),   # 1→5: Nose to LEar
        (1, 2),   # 2→3: LEye to REye
        (1, 3),   # 2→4: LEye to REar
        (2, 4),   # 3→5: REye to LEar
        
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
        # Left Leg
        (12, 14), # 13→15: LHip to LKnee
        (14, 16), # 15→17: LKnee to LAnkle
        
        # Right Foot (ENABLED - joints 16→18,19,20)
        (15, 17), # 16→18: RAnkle to RBigToe
        (15, 18), # 16→19: RAnkle to RSmallToe
        (15, 19), # 16→20: RAnkle to RHeel
        
        # Left Foot (ENABLED - joints 17→21,22,23)
        (16, 20), # 17→21: LAnkle to LBigToe
        (16, 21), # 17→22: LAnkle to LSmallToe
        (16, 22), # 17→23: LAnkle to LHeel
    ]
    
    # Draw skeleton connections
    for i, j in user_connections:
        if i < len(body_pose) and j < len(body_pose):
            x = [body_pose[i, 0], body_pose[j, 0]]
            y = [body_pose[i, 1], body_pose[j, 1]]
            z = [body_pose[i, 2], body_pose[j, 2]]
            
            # Color: Left side=blue, Right side=red, Center=green
            # Left: 2,4,6,8,10,12,14,16,20,21,22 (0-indexed)
            # Right: 1,3,5,7,9,11,13,15,17,18,19 (0-indexed)
            left_joints = [2, 4, 6, 8, 10, 12, 14, 16, 20, 21, 22]
            right_joints = [1, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19]
            
            if i in left_joints or j in left_joints:
                color = (0, 0, 1)  # Blue for left
            elif i in right_joints or j in right_joints:
                color = (1, 0, 0)  # Red for right
            else:
                color = (0, 0.8, 0)  # Green for center
            
            ax.plot(x, y, z, lw=3, color=color, alpha=0.8)
    
    # Draw joints
    colors = ['yellow' if i == 0 else 'orange' for i in range(len(body_pose))]
    ax.scatter(body_pose[:, 0], body_pose[:, 1], body_pose[:, 2], 
               c=colors, s=80, alpha=0.9, edgecolors='black', linewidth=2)
    
    # Add joint numbers (with offset for different conventions)
    for i in range(len(body_pose)):
        ax.text(body_pose[i, 0], body_pose[i, 1], body_pose[i, 2], 
               str(i + joint_offset), fontsize=16, color='black', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8))
    
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
    
    # Labels
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.tick_params(labelsize=10)
    
    # Title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="WB3D Single Frame Visualizer - Compare Joint Conventions"
    )
    parser.add_argument('--wb3d', type=str, required=True,
                        help='Path to keypoints_3D_wb.npz')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: same as wb3d file)')
    
    args = parser.parse_args()
    
    # Check input
    wb3d_path = Path(args.wb3d)
    if not wb3d_path.exists():
        print(f"❌ WB3D file not found: {wb3d_path}")
        return 1
    
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = wb3d_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("🔬 WB3D Single Frame Comparison")
    print("=" * 70)
    
    # Load WB3D data
    print(f"\n📂 Loading: {wb3d_path}")
    wb3d_data = np.load(wb3d_path)
    poses_wb3d = wb3d_data['keypoints_3d']  # (N, 133, 3)
    print(f"   Shape: {poses_wb3d.shape}")
    
    # Extract first frame, first 23 joints (body + feet)
    frame_0 = poses_wb3d[0, :23, :].copy()
    print(f"   Using frame 0, joints 0-22 (23 total - body + feet)")
    
    # Generate skeleton with user's convention (1-23 for body + feet)
    print("\n🎨 Generating skeleton...")
    
    output_1to23 = output_dir / "wb3d_1to23.png"
    render_skeleton(
        frame_0.copy(), 
        output_1to23, 
        "WB3D Skeleton - User Convention (1-23)",
        joint_offset=1
    )
    
    print("\n" + "=" * 70)
    print("✅ SKELETON IMAGE GENERATED!")
    print("=" * 70)
    print(f"\nOutput: {output_1to23}")
    print(f"\nUsing corrected connections (user's convention):")
    print(f"  - Head: 1→4,5 | 2→3,4 | 3→5")
    print(f"  - Torso: 6→8,12 | 7→9,13 | 12↔13")
    print(f"  - Arms: 8→10 | 9→11")
    print(f"  - Legs: 12→14→16 | 13→15→17")
    print(f"  - Feet: 16→18,19,20 | 17→21,22,23")
    print(f"\nJoint numbering: 1-23 (user's convention, body + feet)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
