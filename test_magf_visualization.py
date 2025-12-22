"""
Minimal test script to verify MotionAGFormer 3D visualization is correct.
Reads keypoints_2D_3d.npy and displays a single frame with proper skeleton.
"""

import numpy as np
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
    R: quaternion [w, x, y, z]
    """
    # Tile quaternion to match number of joints
    R_tiled = np.tile(R, (X.shape[0], 1))
    return qrot(R_tiled, X) + t


def show3Dpose(vals, ax):
    """Render H36M-17 skeleton."""
    ax.view_init(elev=15., azim=70)
    
    lcolor = (0, 0, 1)  # Blue for left
    rcolor = (1, 0, 0)  # Red for right

    # H36M skeleton connections
    I = np.array([0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])
    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], dtype=bool)

    for i in range(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, color=lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
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


def main():
    import cv2
    from tqdm import tqdm
    
    # Load MotionAGFormer 3D predictions
    npy_file = '/content/unifiedposepipeline/demo_data/outputs/keypoints_2D_3d.npy'
    poses_3d = np.load(npy_file)
    
    print(f"Loaded poses: {poses_3d.shape}")
    print(f"Format: (frames={poses_3d.shape[0]}, joints={poses_3d.shape[1]}, coords={poses_3d.shape[2]})")
    print(f"H36M-17 joint order\n")
    
    # Video settings
    output_video = 'test_magf_skeleton_video.mp4'
    fps = 30
    width, height = 640, 640
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Rotation quaternion
    rot = np.array([0.1407056450843811, -0.1500701755285263, 
                    -0.755240797996521, 0.6223280429840088], dtype='float32')
    
    print(f"Generating {len(poses_3d)} frames...")
    for frame_idx in tqdm(range(len(poses_3d)), desc="Rendering"):
        pose_3d = poses_3d[frame_idx].copy()
        
        # CRITICAL STEP: Center Hip at origin BEFORE rotation
        pose_3d = pose_3d - pose_3d[0:1, :]
        
        # Apply camera_to_world rotation
        pose_3d = camera_to_world(pose_3d, R=rot, t=0)
        
        # Floor the skeleton (lowest Z point at 0)
        pose_3d[:, 2] -= np.min(pose_3d[:, 2])
        
        # Normalize by max value
        max_value = np.max(pose_3d)
        if max_value > 0:
            pose_3d /= max_value
        
        # Visualize
        fig = plt.figure(figsize=(6.4, 6.4), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        show3Dpose(pose_3d, ax)
        ax.set_title(f'MotionAGFormer 3D Pose\nFrame {frame_idx + 1}/{len(poses_3d)}', 
                     fontsize=12, pad=10)
        
        # Convert matplotlib to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        
        # Write frame
        out.write(img)
    
    out.release()
    
    print(f"\n✅ Saved video: {output_video}")
    print(f"Total frames: {len(poses_3d)}")
    print(f"FPS: {fps}")
    print(f"Please review the video to confirm skeleton looks correct!")


if __name__ == '__main__':
    main()
