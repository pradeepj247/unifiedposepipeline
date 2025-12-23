"""
UDP 3D Lifting - 3D Pose Estimation from 2D Keypoints

Takes Stage 2 output (2D COCO keypoints) from udp_video.py and lifts to 3D poses
using MotionAGFormer. Optionally creates visualization with 3D skeleton overlay.

Pipeline:
    Stage 2 (COCO-17 keypoints) → Convert to H36M-17 → MotionAGFormer → 3D poses

Usage:
    # Basic 3D lifting (no visualization)
    python udp_3d_lifting.py --keypoints demo_data/outputs/keypoints_2D.npz
    
    # With visualization overlay
    python udp_3d_lifting.py \
        --keypoints demo_data/outputs/stage2_keypoints.npz \
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

# Use non-interactive backend
matplotlib.use('Agg')

REPO_ROOT = Path(__file__).parent
PARENT_DIR = REPO_ROOT.parent
MODELS_DIR = PARENT_DIR / "models"

# Add local libraries to path
sys.path.insert(0, str(REPO_ROOT / "lib"))

import copy


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
    import copy
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


def load_motionagformer_model(checkpoint_path, chunk_size=243, device='cuda'):
    """
    Load MotionAGFormer model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pth.tr checkpoint
        chunk_size: Number of frames model processes (243 for Base)
        device: 'cuda' or 'cpu'
    
    Returns:
        model: Loaded MotionAGFormer model in eval mode
    """
    from motionagformer.model import MotionAGFormer
    
    # Model configuration (MotionAGFormer-Base)
    model = MotionAGFormer(
        n_layers=16,            # Number of MotionAGFormer blocks
        dim_in=3,              # Input dimension (x, y, z)
        dim_feat=128,          # Feature dimension
        dim_rep=512,           # Representation dimension
        dim_out=3,             # Output dimension (x, y, z)
        num_heads=8,           # Number of attention heads
        num_joints=17,         # H36M has 17 joints
        n_frames=chunk_size,   # Must match checkpoint (243)
        mlp_ratio=4,
        act_layer=nn.GELU,
        attn_drop=0.0,
        drop=0.0,
        drop_path=0.0,
        use_layer_scale=True,
        layer_scale_init_value=0.00001,
        use_adaptive_fusion=True,
        qkv_bias=False,
        qkv_scale=None,
        hierarchical=False,
        use_temporal_similarity=True,
        neighbour_num=2,
        temporal_connection_len=1,
        use_tcn=False,
        graph_only=False,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel training)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Wrap in DataParallel like MotionAGFormer
    model = nn.DataParallel(model).to(device)
    
    # Reconstruct state_dict with 'module.' prefix if needed
    if not list(state_dict.keys())[0].startswith('module.'):
        state_dict = {'module.' + k: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    return model


@torch.no_grad()
def run_3d_inference(keypoints, model, width, height, device='cuda'):
    """
    Run MotionAGFormer 3D pose inference (EXACTLY matches vis.py).
    
    Args:
        keypoints: (1, num_frames, 17, 3) H36M keypoints [x, y, confidence] with batch dim
        model: MotionAGFormer model
        width: Video width for normalization
        height: Video height for normalization
        device: 'cuda' or 'cpu'
    
    Returns:
        poses_3d: (num_frames, 17, 3) 3D poses [x, y, z]
    """
    # Split into clips
    clips, downsample = turn_into_clips(keypoints)
    
    all_output_3D = []
    
    for idx, clip in enumerate(clips):
        # Normalize (MotionAGFormer style)
        input_2D = normalize_screen_coordinates(clip, w=width, h=height)
        
        # Flip for test-time augmentation
        input_2D_aug = flip_data(input_2D)
        
        # Convert to torch tensors
        input_2D = torch.from_numpy(input_2D.astype('float32')).to(device)
        input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32')).to(device)
        
        # Forward pass with TTA
        output_3D_non_flip = model(input_2D)
        output_3D_flip = flip_data(model(input_2D_aug).cpu().numpy())
        output_3D = (output_3D_non_flip.cpu().numpy() + output_3D_flip) / 2
        
        # Handle last clip resampling
        if idx == len(clips) - 1 and downsample is not None:
            output_3D = output_3D[:, downsample]
        
        # Set Hip to origin (CRITICAL!)
        output_3D[:, :, 0, :] = 0
        
        all_output_3D.append(output_3D)
    
    # Concatenate all clips
    final_output = np.concatenate(all_output_3D, axis=1)
    
    # Remove batch dimension: (num_frames, 17, 3)
    poses_3d = final_output[0]
    
    return poses_3d


def show3Dpose(vals, ax):
    """Render 3D skeleton on matplotlib axis."""
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
    # Tile quaternion to match number of joints
    R_tiled = np.tile(R, (X.shape[0], 1))
    return qrot(R_tiled, X) + t


def create_visualization(video_path, poses_3d, output_path, max_frames=None):
    """
    Create video with 3D pose overlay.
    
    Args:
        video_path: Path to original video
        poses_3d: (num_frames, 17, 3) 3D poses
        output_path: Path to save output video
        max_frames: Maximum frames to render (None = all)
    """
    print("\n" + "=" * 70)
    print("🎨 Creating 3D Visualization")
    print("=" * 70)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {video_frames} frames, {fps} fps, {width}x{height}")
    print(f"3D poses: {poses_3d.shape[0]} frames")
    
    # Match frames
    total_frames = min(poses_3d.shape[0], video_frames)
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    
    print(f"Rendering {total_frames} frames...")
    
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Video writer (side-by-side: original | 3D)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_width = width * 2
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_width, height))
    
    # Rotation for better view (from MotionAGFormer demo)
    rot = np.array([0.1407056450843811, -0.1500701755285263, 
                    -0.755240797996521, 0.6223280429840088], dtype='float32')
    
    for i in tqdm(range(total_frames), desc="Rendering"):
        # Read video frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get 3D pose
        pose_3d = poses_3d[i].copy()
        
        # CRITICAL: Center Hip (joint 0) at origin BEFORE rotation
        # This ensures rotation happens around the skeleton's center (MotionAGFormer convention)
        pose_3d = pose_3d - pose_3d[0:1, :]  # Subtract Hip position from all joints
        
        # Apply camera_to_world rotation using quaternion (now rotating around centered Hip)
        pose_3d = camera_to_world(pose_3d, R=rot, t=0)
        
        # Normalize for display
        pose_3d[:, 2] -= np.min(pose_3d[:, 2])  # Floor at z=0
        max_value = np.max(pose_3d)
        if max_value > 0:
            pose_3d /= max_value
        
        # Render 3D pose
        fig = plt.figure(figsize=(height / 100, height / 100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        show3Dpose(pose_3d, ax)
        
        # Convert matplotlib to image
        fig.canvas.draw()
        img_3d = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_3d = img_3d.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img_3d = cv2.cvtColor(img_3d, cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        
        # Resize to match video height
        img_3d = cv2.resize(img_3d, (width, height))
        
        # Combine side by side
        combined = np.hstack([frame, img_3d])
        out.write(combined)
    
    cap.release()
    out.release()
    
    print(f"✅ Visualization saved: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="UDP 3D Lifting - Lift 2D poses to 3D")
    parser.add_argument('--keypoints', type=str, required=True,
                        help='Path to stage2_keypoints.npz from udp_video.py')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to original video (for visualization)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output 3D video (default: same dir as keypoints)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to MotionAGFormer checkpoint (default: models/motionagformer/)')
    parser.add_argument('--visualize', action='store_true',
                        help='Create 3D visualization video (requires --video)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum frames to process (default: all)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    print("\n" + "🎬" * 35)
    print("UDP 3D LIFTING - MotionAGFormer")
    print("🎬" * 35)
    
    # Check inputs
    keypoints_path = Path(args.keypoints)
    if not keypoints_path.exists():
        print(f"❌ Keypoints file not found: {keypoints_path}")
        return 1
    
    # Setup checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = MODELS_DIR / "motionagformer" / "motionagformer-base-h36m.pth.tr"
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"   Run setup_unified.py to download the checkpoint")
        return 1
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load Stage 2 keypoints (COCO format)
    print(f"\n📂 Loading 2D keypoints...")
    print(f"   {keypoints_path}")
    
    data = np.load(keypoints_path)
    keypoints_2d = data['keypoints']  # (num_frames, 17, 2)
    scores = data['scores']           # (num_frames, 17)
    
    if args.max_frames:
        keypoints_2d = keypoints_2d[:args.max_frames]
        scores = scores[:args.max_frames]
    
    num_frames = keypoints_2d.shape[0]
    print(f"   ✅ Loaded {num_frames} frames of COCO-17 keypoints")
    
    # Add batch dimension (CRITICAL!)
    keypoints_2d = keypoints_2d[np.newaxis, ...]  # (1, num_frames, 17, 2)
    scores = scores[np.newaxis, ...]              # (1, num_frames, 17)
    
    # Convert COCO → H36M
    print(f"\n🔄 Converting COCO-17 → H36M-17 format...")
    t_start = time.time()
    h36m_kpts, h36m_scores, valid_frames = h36m_coco_format(keypoints_2d, scores)
    elapsed = time.time() - t_start
    print(f"   ✅ Converted in {elapsed:.2f}s")
    
    # Add confidence scores to last dimension
    h36m_keypoints = np.concatenate((h36m_kpts, h36m_scores[..., None]), axis=-1)
    print(f"   Final shape with confidence: {h36m_keypoints.shape}")
    
    # Load MotionAGFormer model
    print(f"\n📦 Loading MotionAGFormer model...")
    print(f"   Checkpoint: {checkpoint_path.name}")
    t_start = time.time()
    model = load_motionagformer_model(checkpoint_path, chunk_size=243, device=device)
    elapsed = time.time() - t_start
    print(f"   ✅ Model loaded in {elapsed:.2f}s")
    print(f"   Config: 243-frame chunks, 16 layers, 128 dim")
    
    # Run 3D inference
    print(f"\n🚀 Running 3D pose inference...")
    print(f"   Processing {num_frames} frames...")
    
    # Get video dimensions from video if available, otherwise use defaults
    if args.video and Path(args.video).exists():
        cap = cv2.VideoCapture(args.video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    else:
        width, height = 1280, 720  # Default
    
    t_start = time.time()
    poses_3d = run_3d_inference(h36m_keypoints, model, width, height, device=device)
    elapsed = time.time() - t_start
    
    print(f"\n   ✅ Inference complete in {elapsed:.2f}s")
    print(f"   Processing speed: {poses_3d.shape[0] / elapsed:.1f} fps")
    print(f"   Output shape: {poses_3d.shape}")
    
    # Save 3D poses
    output_dir = keypoints_path.parent
    poses_3d_path = output_dir / "keypoints_3D_magf.npz"
    np.savez(poses_3d_path, poses_3d=poses_3d)
    
    size_mb = poses_3d_path.stat().st_size / (1024 ** 2)
    print(f"\n💾 Saved 3D poses: {poses_3d_path}")
    print(f"   Size: {size_mb:.2f} MB")
    
    # Visualization
    if args.visualize:
        if not args.video:
            print(f"\n⚠️  Visualization requires --video argument")
        elif not Path(args.video).exists():
            print(f"\n❌ Video not found: {args.video}")
        else:
            if args.output:
                output_video_path = Path(args.output)
            else:
                output_video_path = output_dir / f"{keypoints_path.stem}_3d.mp4"
            
            success = create_visualization(args.video, poses_3d, output_video_path, args.max_frames)
            
            if success:
                size_mb = output_video_path.stat().st_size / (1024 ** 2)
                print(f"\n💾 Saved visualization: {output_video_path}")
                print(f"   Size: {size_mb:.2f} MB")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"   Input: {num_frames} frames (COCO-17)")
    print(f"   Output: {poses_3d.shape[0]} frames (H36M-17 3D)")
    print(f"   3D poses: {poses_3d_path}")
    if args.visualize and args.video:
        print(f"   Visualization: {output_video_path}")
    print("=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
