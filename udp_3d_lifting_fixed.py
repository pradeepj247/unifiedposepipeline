"""
UDP 3D Lifting - 3D Pose Estimation from 2D Keypoints (FIXED - Matches MotionAGFormer vis.py)

Takes Stage 2 output (2D COCO keypoints) and lifts to 3D poses using MotionAGFormer.
This version EXACTLY matches the original MotionAGFormer demo/vis.py logic.

Pipeline:
    Stage 2 (COCO-17 keypoints) → Convert to H36M-17 → Clip Processing → 
    Test-Time Augmentation → MotionAGFormer → 3D poses → Visualization

Usage:
    python udp_3d_lifting_fixed.py \
        --keypoints demo_data/outputs/keypoints_2D.npz \
        --video demo_data/videos/sample.mp4 \
        --output demo_data/outputs/sample_3d.mp4 \
        --visualize
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import time
import cv2
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

# Use non-interactive backend
matplotlib.use('Agg')

REPO_ROOT = Path(__file__).parent
PARENT_DIR = REPO_ROOT.parent
MODELS_DIR = PARENT_DIR / "models"

# Add local libraries to path
sys.path.insert(0, str(REPO_ROOT / "lib"))


# ============================================================================
# COCO → H36M Conversion (from MotionAGFormer demo/lib/preprocess.py)
# ============================================================================

h36m_coco_order = [9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
coco_order = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
spple_keypoints = [10, 8, 0, 7]


def coco_h36m(keypoints):
    """
    Convert COCO keypoints to H36M format (from MotionAGFormer).
    Args:
        keypoints: (num_frames, 17, 2) COCO keypoints
    Returns:
        keypoints_h36m: (num_frames, 17, 2) H36M keypoints
        valid_frames: indices of valid frames
    """
    temporal = keypoints.shape[0]
    keypoints_h36m = np.zeros_like(keypoints, dtype=np.float32)
    htps_keypoints = np.zeros((temporal, 4, 2), dtype=np.float32)

    # htps_keypoints: head, thorax, pelvis, spine
    htps_keypoints[:, 0, 0] = np.mean(keypoints[:, 1:5, 0], axis=1, dtype=np.float32)
    htps_keypoints[:, 0, 1] = np.sum(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1]
    htps_keypoints[:, 1, :] = np.mean(keypoints[:, 5:7, :], axis=1, dtype=np.float32)
    htps_keypoints[:, 1, :] += (keypoints[:, 0, :] - htps_keypoints[:, 1, :]) / 3

    htps_keypoints[:, 2, :] = np.mean(keypoints[:, 11:13, :], axis=1, dtype=np.float32)
    htps_keypoints[:, 3, :] = np.mean(keypoints[:, [5, 6, 11, 12], :], axis=1, dtype=np.float32)

    keypoints_h36m[:, spple_keypoints, :] = htps_keypoints
    keypoints_h36m[:, h36m_coco_order, :] = keypoints[:, coco_order, :]

    keypoints_h36m[:, 9, :] -= (keypoints_h36m[:, 9, :] - np.mean(keypoints[:, 5:7, :], axis=1, dtype=np.float32)) / 4
    keypoints_h36m[:, 7, 0] += 2*(keypoints_h36m[:, 7, 0] - np.mean(keypoints_h36m[:, [0, 8], 0], axis=1, dtype=np.float32))
    keypoints_h36m[:, 8, 1] -= (np.mean(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1])*2/3

    valid_frames = np.where(np.sum(keypoints_h36m.reshape(-1, 34), axis=1) != 0)[0]
    
    return keypoints_h36m, valid_frames


def h36m_coco_format(keypoints, scores):
    """
    Wrapper to convert COCO to H36M with confidence scores (from MotionAGFormer).
    Args:
        keypoints: (batch, num_frames, 17, 2)
        scores: (batch, num_frames, 17)
    Returns:
        h36m_kpts: (batch, num_frames, 17, 2)
        h36m_scores: (batch, num_frames, 17)
        valid_frames: list of valid frame indices per batch
    """
    assert len(keypoints.shape) == 4 and len(scores.shape) == 3

    h36m_kpts = []
    h36m_scores = []
    valid_frames = []

    for i in range(keypoints.shape[0]):
        kpts = keypoints[i]
        score = scores[i]

        new_score = np.zeros_like(score, dtype=np.float32)

        if np.sum(kpts) != 0.:
            kpts, valid_frame = coco_h36m(kpts)
            h36m_kpts.append(kpts)
            valid_frames.append(valid_frame)

            new_score[:, h36m_coco_order] = score[:, coco_order]
            new_score[:, 0] = np.mean(score[:, [11, 12]], axis=1, dtype=np.float32)
            new_score[:, 8] = np.mean(score[:, [5, 6]], axis=1, dtype=np.float32)
            new_score[:, 7] = np.mean(new_score[:, [0, 8]], axis=1, dtype=np.float32)
            new_score[:, 10] = np.mean(score[:, [1, 2, 3, 4]], axis=1, dtype=np.float32)

            h36m_scores.append(new_score)

    h36m_kpts = np.asarray(h36m_kpts, dtype=np.float32)
    h36m_scores = np.asarray(h36m_scores, dtype=np.float32)

    return h36m_kpts, h36m_scores, valid_frames


# ============================================================================
# Clip Processing (from MotionAGFormer demo/vis.py)
# ============================================================================

def resample(n_frames):
    """Resample frames to 243 for model input."""
    even = np.linspace(0, n_frames, num=243, endpoint=False)
    result = np.floor(even)
    result = np.clip(result, a_min=0, a_max=n_frames - 1).astype(np.uint32)
    return result


def turn_into_clips(keypoints):
    """
    Split keypoints into 243-frame clips (from MotionAGFormer).
    Args:
        keypoints: (batch, num_frames, 17, 3) where last dim is [x, y, conf]
    Returns:
        clips: list of (batch, 243, 17, 3) arrays
        downsample: indices for last clip if it was resampled
    """
    clips = []
    n_frames = keypoints.shape[1]
    downsample = None
    
    if n_frames <= 243:
        new_indices = resample(n_frames)
        clips.append(keypoints[:, new_indices, ...])
        downsample = np.unique(new_indices, return_index=True)[1]
    else:
        for start_idx in range(0, n_frames, 243):
            keypoints_clip = keypoints[:, start_idx:start_idx + 243, ...]
            clip_length = keypoints_clip.shape[1]
            if clip_length != 243:
                new_indices = resample(clip_length)
                clips.append(keypoints_clip[:, new_indices, ...])
                downsample = np.unique(new_indices, return_index=True)[1]
            else:
                clips.append(keypoints_clip)
                
    return clips, downsample


# ============================================================================
# Test-Time Augmentation (from MotionAGFormer demo/vis.py)
# ============================================================================

def flip_data(data, left_joints=[1, 2, 3, 14, 15, 16], right_joints=[4, 5, 6, 11, 12, 13]):
    """
    Flip data for test-time augmentation (from MotionAGFormer).
    Args:
        data: [N, F, 17, D] or [F, 17, D]
    Returns:
        flipped_data: same shape as input
    """
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1  # flip x of all joints
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]
    return flipped_data


# ============================================================================
# Normalization (from MotionAGFormer demo/lib/utils.py)
# ============================================================================

def normalize_screen_coordinates(X, w, h):
    """
    Normalize screen coordinates (from MotionAGFormer).
    Args:
        X: (..., 2) or (..., 3) array
        w: width
        h: height
    Returns:
        normalized coordinates
    """
    assert X.shape[-1] == 2 or X.shape[-1] == 3
    result = np.copy(X)
    result[..., :2] = X[..., :2] / w * 2 - [1, h / w]
    return result


# ============================================================================
# Quaternion Rotation (from MotionAGFormer demo/lib/utils.py)
# ============================================================================

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v.
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]
    
    qvec = q[..., 1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2 * (q[..., :1] * uv + uuv)


def camera_to_world(X, R, t):
    """
    Apply camera rotation using quaternion (from MotionAGFormer).
    Args:
        X: (N, 3) 3D points
        R: (4,) quaternion [w, x, y, z]
        t: translation
    Returns:
        rotated points
    """
    R_tiled = np.tile(R, (X.shape[0], 1))
    return qrot(R_tiled, X) + t


# ============================================================================
# Model Loading
# ============================================================================

def load_motionagformer_model(checkpoint_path, device='cuda'):
    """Load MotionAGFormer model (matches vis.py config)."""
    from motionagformer.model import MotionAGFormer
    
    # Model configuration (MotionAGFormer-Base, from vis.py lines 209-220)
    args = {
        'n_layers': 16,
        'dim_in': 3,
        'dim_feat': 128,
        'dim_rep': 512,
        'dim_out': 3,
        'mlp_ratio': 4,
        'act_layer': nn.GELU,
        'attn_drop': 0.0,
        'drop': 0.0,
        'drop_path': 0.0,
        'use_layer_scale': True,
        'layer_scale_init_value': 0.00001,
        'use_adaptive_fusion': True,
        'num_heads': 8,
        'qkv_bias': False,
        'qkv_scale': None,
        'hierarchical': False,
        'use_temporal_similarity': True,
        'neighbour_num': 2,
        'temporal_connection_len': 1,
        'use_tcn': False,
        'graph_only': False,
        'n_frames': 243,
        'num_joints': 17,
    }
    
    model = nn.DataParallel(MotionAGFormer(**args)).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    
    return model


# ============================================================================
# 3D Inference (EXACTLY matches vis.py lines 272-295)
# ============================================================================

@torch.no_grad()
def get_pose3D(keypoints, model, img_width, img_height, device='cuda'):
    """
    Run 3D inference (EXACTLY matches MotionAGFormer vis.py).
    Args:
        keypoints: (1, num_frames, 17, 3) with [x, y, conf]
        model: MotionAGFormer model
        img_width: image width
        img_height: image height
    Returns:
        output_3D: (num_frames, 17, 3) 3D poses
    """
    print('\n🚀 Running 3D pose inference...')
    
    # Split into clips
    clips, downsample = turn_into_clips(keypoints)
    print(f'   Split into {len(clips)} clips of 243 frames each')
    
    all_output_3D = []
    
    for idx, clip in enumerate(clips):
        # Normalize (vis.py line 278)
        input_2D = normalize_screen_coordinates(clip, w=img_width, h=img_height)
        
        # Flip for test-time augmentation (vis.py line 279)
        input_2D_aug = flip_data(input_2D)
        
        # Convert to torch tensors
        input_2D = torch.from_numpy(input_2D.astype('float32')).to(device)
        input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32')).to(device)
        
        # Forward pass (vis.py lines 282-284)
        output_3D_non_flip = model(input_2D)
        output_3D_flip = flip_data(model(input_2D_aug).cpu().numpy())
        output_3D = (output_3D_non_flip.cpu().numpy() + output_3D_flip) / 2
        
        # Handle last clip resampling (vis.py lines 286-287)
        if idx == len(clips) - 1 and downsample is not None:
            output_3D = output_3D[:, downsample]
        
        # Set Hip to origin (vis.py line 289)
        output_3D[:, :, 0, :] = 0
        
        all_output_3D.append(output_3D)
    
    # Concatenate all clips
    final_output = np.concatenate(all_output_3D, axis=1)
    
    print(f'   ✅ Inference complete! Output shape: {final_output.shape}')
    
    return final_output[0]  # Remove batch dimension


# ============================================================================
# Visualization
# ============================================================================

def show3Dpose(vals, ax):
    """Render 3D skeleton (from MotionAGFormer)."""
    ax.view_init(elev=15., azim=70)
    
    lcolor = (0, 0, 1)
    rcolor = (1, 0, 0)

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

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)


def create_visualization(video_path, poses_3d, output_path, max_frames=None):
    """
    Create video with 3D pose overlay (matches MotionAGFormer vis.py).
    """
    print("\n🎨 Creating 3D Visualization")
    print("=" * 70)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    total_frames = min(poses_3d.shape[0], video_frames)
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    
    print(f"Rendering {total_frames} frames...")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width * 2, height))
    
    # Rotation quaternion (vis.py line 297)
    rot = np.array([0.1407056450843811, -0.1500701755285263, 
                    -0.755240797996521, 0.6223280429840088], dtype='float32')
    
    for i in tqdm(range(total_frames), desc="Rendering"):
        ret, frame = cap.read()
        if not ret:
            break
        
        post_out = poses_3d[i].copy()
        
        # Apply transformations (vis.py lines 298-301)
        post_out = camera_to_world(post_out, R=rot, t=0)
        post_out[:, 2] -= np.min(post_out[:, 2])
        max_value = np.max(post_out)
        post_out /= max_value
        
        # Render 3D
        fig = plt.figure(figsize=(height / 100, height / 100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        show3Dpose(post_out, ax)
        
        fig.canvas.draw()
        img_3d = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_3d = img_3d.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img_3d = cv2.cvtColor(img_3d, cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        
        img_3d = cv2.resize(img_3d, (width, height))
        combined = np.hstack([frame, img_3d])
        out.write(combined)
    
    cap.release()
    out.release()
    
    print(f"✅ Saved: {output_path}")
    return True


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="UDP 3D Lifting (FIXED - matches MotionAGFormer)")
    parser.add_argument('--keypoints', type=str, required=True,
                        help='Path to keypoints_2D.npz')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to original video')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output video')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to MotionAGFormer checkpoint')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualization video')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum frames to process')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    print("\n" + "🎬" * 35)
    print("UDP 3D LIFTING - FIXED (Matches MotionAGFormer vis.py)")
    print("🎬" * 35)
    
    # Paths
    keypoints_path = Path(args.keypoints)
    video_path = Path(args.video)
    
    if not keypoints_path.exists():
        print(f"❌ Keypoints not found: {keypoints_path}")
        return 1
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return 1
    
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = MODELS_DIR / "motionagformer" / "motionagformer-base-h36m.pth.tr"
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 1
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Device: {device}")
    
    # Load 2D keypoints
    print(f"\n📂 Loading 2D COCO keypoints...")
    data = np.load(keypoints_path)
    keypoints_2d = data['keypoints']  # (num_frames, 17, 2)
    scores = data['scores']           # (num_frames, 17)
    
    if args.max_frames:
        keypoints_2d = keypoints_2d[:args.max_frames]
        scores = scores[:args.max_frames]
    
    print(f"   Shape: {keypoints_2d.shape}")
    print(f"   Frames: {keypoints_2d.shape[0]}")
    
    # Add batch dimension (CRITICAL!)
    keypoints_2d = keypoints_2d[np.newaxis, ...]  # (1, num_frames, 17, 2)
    scores = scores[np.newaxis, ...]              # (1, num_frames, 17)
    
    print(f"   After adding batch dim: {keypoints_2d.shape}")
    
    # Convert COCO → H36M
    print(f"\n🔄 Converting COCO → H36M...")
    t_start = time.time()
    h36m_kpts, h36m_scores, valid_frames = h36m_coco_format(keypoints_2d, scores)
    elapsed = time.time() - t_start
    print(f"   ✅ Converted in {elapsed:.2f}s")
    print(f"   Shape: {h36m_kpts.shape}")
    
    # Add confidence scores to last dimension
    keypoints = np.concatenate((h36m_kpts, h36m_scores[..., None]), axis=-1)
    print(f"   Final shape with confidence: {keypoints.shape}")
    
    # Get video dimensions
    cap = cv2.VideoCapture(str(video_path))
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"   Video dimensions: {img_width}x{img_height}")
    
    # Load model
    print(f"\n📦 Loading MotionAGFormer model...")
    print(f"   Checkpoint: {checkpoint_path.name}")
    t_start = time.time()
    model = load_motionagformer_model(checkpoint_path, device=device)
    elapsed = time.time() - t_start
    print(f"   ✅ Loaded in {elapsed:.2f}s")
    
    # Run 3D inference
    t_start = time.time()
    poses_3d = get_pose3D(keypoints, model, img_width, img_height, device=device)
    elapsed = time.time() - t_start
    print(f"   Processing speed: {poses_3d.shape[0] / elapsed:.1f} fps")
    
    # Save 3D poses
    output_dir = keypoints_path.parent
    poses_3d_path = output_dir / f"{keypoints_path.stem}_3d.npy"
    np.save(poses_3d_path, poses_3d)
    print(f"\n💾 Saved 3D poses: {poses_3d_path}")
    
    # Visualization
    if args.visualize:
        if args.output:
            output_video_path = Path(args.output)
        else:
            output_video_path = output_dir / f"{video_path.stem}_3d.mp4"
        
        create_visualization(video_path, poses_3d, output_video_path, args.max_frames)
    
    print("\n" + "=" * 70)
    print("✅ COMPLETE!")
    print("=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
