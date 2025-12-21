"""
UDP 3D Lifting - 3D Pose Estimation from 2D Keypoints

Takes Stage 2 output (2D COCO keypoints) from udp_video.py and lifts to 3D poses
using MotionAGFormer. Optionally creates visualization with 3D skeleton overlay.

Pipeline:
    Stage 2 (COCO-17 keypoints) → Convert to H36M-17 → MotionAGFormer → 3D poses

Usage:
    # Basic 3D lifting (no visualization)
    python udp_3d_lifting.py --keypoints demo_data/outputs/stage2_keypoints.npz
    
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
from scipy.spatial.transform import Rotation as Rot

# Use non-interactive backend
matplotlib.use('Agg')

REPO_ROOT = Path(__file__).parent
PARENT_DIR = REPO_ROOT.parent
MODELS_DIR = PARENT_DIR / "models"

# Add local libraries to path
sys.path.insert(0, str(REPO_ROOT / "lib"))


def convert_coco_to_h36m(coco_kpts):
    """
    Convert COCO-17 keypoints to H36M-17 format.
    
    Args:
        coco_kpts: (num_frames, 17, 3) array with [x, y, confidence]
    
    Returns:
        h36m_kpts: (num_frames, 17, 3) array in H36M format
    """
    temporal = coco_kpts.shape[0]
    keypoints_h36m = np.zeros_like(coco_kpts, dtype=np.float32)
    
    # Extract coordinates (x, y only)
    keypoints = coco_kpts[:, :, :2]
    
    # COCO to H36M mapping
    h36m_coco_order = [9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
    coco_order = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    
    # Derived keypoints: head, thorax, pelvis, spine
    htps_keypoints = np.zeros((temporal, 4, 2), dtype=np.float32)
    
    # 0: Head (average of eyes and ears)
    htps_keypoints[:, 0, 0] = np.mean(keypoints[:, 1:5, 0], axis=1, dtype=np.float32)
    htps_keypoints[:, 0, 1] = np.sum(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1]
    
    # 1: Thorax/Neck (average of shoulders)
    htps_keypoints[:, 1, :] = np.mean(keypoints[:, 5:7, :], axis=1, dtype=np.float32)
    htps_keypoints[:, 1, :] += (keypoints[:, 0, :] - htps_keypoints[:, 1, :]) / 3

    # 2: Pelvis/Hip (average of hips)
    htps_keypoints[:, 2, :] = np.mean(keypoints[:, 11:13, :], axis=1, dtype=np.float32)
    
    # 3: Spine (average of shoulders and hips)
    htps_keypoints[:, 3, :] = np.mean(keypoints[:, [5, 6, 11, 12], :], axis=1, dtype=np.float32)

    # Assign derived keypoints to H36M format
    keypoints_h36m[:, 10, :2] = htps_keypoints[:, 0, :]  # Head
    keypoints_h36m[:, 8, :2] = htps_keypoints[:, 1, :]   # Thorax
    keypoints_h36m[:, 0, :2] = htps_keypoints[:, 2, :]   # Hip (pelvis)
    keypoints_h36m[:, 7, :2] = htps_keypoints[:, 3, :]   # Spine
    
    # Map COCO joints to H36M
    for i, coco_idx in enumerate(coco_order):
        h36m_idx = h36m_coco_order[i]
        keypoints_h36m[:, h36m_idx, :2] = keypoints[:, coco_idx, :]
    
    # Refinements from original MotionAGFormer preprocess
    keypoints_h36m[:, 9, :2] -= (keypoints_h36m[:, 9, :2] - np.mean(keypoints[:, 5:7, :], axis=1, dtype=np.float32)) / 4
    keypoints_h36m[:, 7, 0] += 2*(keypoints_h36m[:, 7, 0] - np.mean(keypoints_h36m[:, [0, 8], 0], axis=1, dtype=np.float32))
    keypoints_h36m[:, 8, 1] -= (np.mean(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1])*2/3
    
    # Copy confidence scores
    keypoints_h36m[:, 10, 2] = np.mean(coco_kpts[:, 1:5, 2], axis=1)  # Head
    keypoints_h36m[:, 8, 2] = np.mean(coco_kpts[:, 5:7, 2], axis=1)   # Thorax
    keypoints_h36m[:, 0, 2] = np.mean(coco_kpts[:, 11:13, 2], axis=1) # Hip
    keypoints_h36m[:, 7, 2] = np.mean(coco_kpts[:, [5,6,11,12], 2], axis=1) # Spine
    
    # Map confidence for COCO joints
    for i, coco_idx in enumerate(coco_order):
        h36m_idx = h36m_coco_order[i]
        keypoints_h36m[:, h36m_idx, 2] = coco_kpts[:, coco_idx, 2]
    
    return keypoints_h36m


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
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def run_3d_inference(keypoints_2d, model, width, height, chunk_size=243, device='cuda'):
    """
    Run MotionAGFormer 3D pose inference on 2D keypoints.
    
    Args:
        keypoints_2d: (num_frames, 17, 3) H36M keypoints [x, y, confidence]
        model: MotionAGFormer model
        width: Video width for normalization
        height: Video height for normalization
        chunk_size: Frames per chunk (243 for Base model)
        device: 'cuda' or 'cpu'
    
    Returns:
        poses_3d: (num_frames, 17, 3) 3D poses [x, y, z]
    """
    num_frames = keypoints_2d.shape[0]
    
    # Normalize keypoints using asymmetric normalization (matches training)
    keypoints_norm = keypoints_2d.copy()
    keypoints_norm[:, :, 0] = keypoints_2d[:, :, 0] / width * 2 - 1           # x: [-1, 1]
    keypoints_norm[:, :, 1] = keypoints_2d[:, :, 1] / width * 2 - (height / width)  # y: asymmetric
    
    # Add batch dimension: (1, num_frames, 17, 3)
    keypoints_norm = keypoints_norm[np.newaxis, ...]
    
    all_predictions = []
    
    if num_frames <= chunk_size:
        # Pad to chunk_size if needed
        pad_size = chunk_size - num_frames
        if pad_size > 0:
            keypoints_padded = np.pad(keypoints_norm, ((0, 0), (0, pad_size), (0, 0), (0, 0)), mode='edge')
        else:
            keypoints_padded = keypoints_norm
        
        # Prepare input: [x, y, 0] for model
        input_xyz = np.concatenate([keypoints_padded[..., :2], np.zeros_like(keypoints_padded[..., :1])], axis=-1)
        input_3d = torch.from_numpy(input_xyz).float().to(device)
        
        # Run model
        output_3d = model(input_3d)
        predictions_3d = output_3d.cpu().numpy()
        
        # Remove padding
        predictions_3d = predictions_3d[:, :num_frames, :, :]
    else:
        # Process in non-overlapping chunks
        num_chunks = (num_frames + chunk_size - 1) // chunk_size
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, num_frames)
            chunk_frames = end_idx - start_idx
            
            # Extract chunk
            chunk = keypoints_norm[:, start_idx:end_idx, :, :]
            
            # Pad if needed
            if chunk_frames < chunk_size:
                pad_size = chunk_size - chunk_frames
                chunk = np.pad(chunk, ((0, 0), (0, pad_size), (0, 0), (0, 0)), mode='edge')
            
            # Prepare input
            input_xyz = np.concatenate([chunk[..., :2], np.zeros_like(chunk[..., :1])], axis=-1)
            input_3d = torch.from_numpy(input_xyz).float().to(device)
            
            # Run model
            output_3d = model(input_3d)
            output_np = output_3d.cpu().numpy()
            
            # Remove padding
            output_np = output_np[:, :chunk_frames, :, :]
            all_predictions.append(output_np)
        
        # Concatenate chunks
        predictions_3d = np.concatenate(all_predictions, axis=1)
    
    # Remove batch dimension: (num_frames, 17, 3)
    poses_3d = predictions_3d[0]
    
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


def camera_to_world(X, R, t):
    """Convert camera coordinates to world coordinates."""
    if R.shape == (4,):  # Quaternion
        rot = Rot.from_quat(R)
        R = rot.as_matrix()
    
    return (R @ X.T).T + t


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
        
        # Apply camera_to_world rotation
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
    
    # Combine into (num_frames, 17, 3) with confidence
    coco_keypoints = np.concatenate([keypoints_2d, scores[..., np.newaxis]], axis=-1)
    
    if args.max_frames:
        coco_keypoints = coco_keypoints[:args.max_frames]
    
    num_frames = coco_keypoints.shape[0]
    print(f"   ✅ Loaded {num_frames} frames of COCO-17 keypoints")
    
    # Convert COCO → H36M
    print(f"\n🔄 Converting COCO-17 → H36M-17 format...")
    t_start = time.time()
    h36m_keypoints = convert_coco_to_h36m(coco_keypoints)
    elapsed = time.time() - t_start
    print(f"   ✅ Converted in {elapsed:.2f}s")
    print(f"   Shape: {h36m_keypoints.shape}")
    
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
    poses_3d = run_3d_inference(h36m_keypoints, model, width, height, chunk_size=243, device=device)
    elapsed = time.time() - t_start
    
    print(f"   ✅ Inference complete in {elapsed:.2f}s")
    print(f"   Processing speed: {num_frames / elapsed:.1f} fps")
    print(f"   Output shape: {poses_3d.shape}")
    
    # Save 3D poses
    output_dir = keypoints_path.parent
    poses_3d_path = output_dir / f"{keypoints_path.stem}_3d.npy"
    np.save(poses_3d_path, poses_3d)
    
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
