"""
Verify 3D Lifting - Single Frame Visualization with Joint Numbers

Shows Frame 1 side-by-side:
- Left: Original video frame with 2D keypoints (numbered)
- Right: 3D skeleton matplotlib plot (numbered)

Usage:
    python verify_3dlifting.py \
        --keypoints demo_data/outputs/keypoints_2D_rtm.npz \
        --video demo_data/videos/dance.mp4
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('Agg')

REPO_ROOT = Path(__file__).parent.parent  # Go up from snippets/
PARENT_DIR = REPO_ROOT.parent
MODELS_DIR = Path("/content/models")

# Add MotionAGFormer to path
sys.path.insert(0, str(REPO_ROOT / "lib"))
sys.path.insert(0, "/content/unifiedposepipeline/lib")

# ============================================================================
# COCO → H36M Conversion
# ============================================================================

h36m_coco_order = [9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
coco_order = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
spple_keypoints = [10, 8, 0, 7]


def coco_h36m(keypoints):
    """Convert COCO keypoints to H36M format."""
    temporal = keypoints.shape[0]
    keypoints_h36m = np.zeros_like(keypoints, dtype=np.float32)
    htps_keypoints = np.zeros((temporal, 4, 2), dtype=np.float32)

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
    """Wrapper to convert COCO to H36M with confidence scores."""
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
# Clip Processing
# ============================================================================

def resample(n_frames):
    """Resample frames to 243 for model input."""
    even = np.linspace(0, n_frames, num=243, endpoint=False)
    result = np.floor(even)
    result = np.clip(result, a_min=0, a_max=n_frames - 1).astype(np.uint32)
    return result


def turn_into_clips(keypoints):
    """Split keypoints into 243-frame clips."""
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
# Test-Time Augmentation
# ============================================================================

def flip_data(data, left_joints=[1, 2, 3, 14, 15, 16], right_joints=[4, 5, 6, 11, 12, 13]):
    """Flip data for test-time augmentation."""
    import copy
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]
    return flipped_data


# ============================================================================
# Normalization
# ============================================================================

def normalize_screen_coordinates(X, w, h):
    """Normalize screen coordinates."""
    assert X.shape[-1] == 2 or X.shape[-1] == 3
    result = np.copy(X)
    result[..., :2] = X[..., :2] / w * 2 - [1, h / w]
    return result


def load_motionagformer_model(checkpoint_path, chunk_size=243, device='cuda'):
    """Load MotionAGFormer model from checkpoint."""
    from motionagformer.model import MotionAGFormer
    
    model = MotionAGFormer(
        n_layers=16,
        dim_in=3,
        dim_feat=128,
        dim_rep=512,
        dim_out=3,
        num_heads=8,
        num_joints=17,
        n_frames=chunk_size,
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
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model = nn.DataParallel(model).to(device)
    
    if not list(state_dict.keys())[0].startswith('module.'):
        state_dict = {'module.' + k: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    return model


@torch.no_grad()
def run_3d_inference(keypoints, model, width, height, device='cuda'):
    """Run MotionAGFormer 3D pose inference."""
    clips, downsample = turn_into_clips(keypoints)
    
    all_output_3D = []
    
    for idx, clip in enumerate(clips):
        input_2D = normalize_screen_coordinates(clip, w=width, h=height)
        input_2D_aug = flip_data(input_2D)
        
        input_2D = torch.from_numpy(input_2D.astype('float32')).to(device)
        input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32')).to(device)
        
        output_3D_non_flip = model(input_2D)
        output_3D_flip = flip_data(model(input_2D_aug).cpu().numpy())
        output_3D = (output_3D_non_flip.cpu().numpy() + output_3D_flip) / 2
        
        if idx == len(clips) - 1 and downsample is not None:
            output_3D = output_3D[:, downsample]
        
        output_3D[:, :, 0, :] = 0
        
        all_output_3D.append(output_3D)
    
    final_output = np.concatenate(all_output_3D, axis=1)
    poses_3d = final_output[0]
    
    return poses_3d


def show3Dpose_with_numbers(vals, ax):
    """Render 3D skeleton with joint numbers."""
    ax.view_init(elev=15., azim=70)
    
    lcolor = (0, 0, 1)  # Blue for left
    rcolor = (1, 0, 0)  # Red for right

    # H36M skeleton connections
    I = np.array([0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])
    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], dtype=bool)

    # Draw skeleton
    for i in range(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, color=lcolor if LR[i] else rcolor)

    # Plot joints
    ax.scatter(vals[:, 0], vals[:, 1], vals[:, 2], c='green', marker='o', s=50)

    # Add joint numbers (small font)
    for joint_idx in range(17):
        x, y, z = vals[joint_idx]
        ax.text(x, y, z, f'{joint_idx}', fontsize=8, color='black', weight='bold')

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


def qrot(q, v):
    """Rotate vector(s) v about the rotation described by quaternion(s) q."""
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    
    qvec = q[..., 1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2 * (q[..., :1] * uv + uuv)


def camera_to_world(X, R, t):
    """Apply camera rotation to 3D keypoints using quaternion rotation."""
    R_tiled = np.tile(R, (X.shape[0], 1))
    return qrot(R_tiled, X) + t


def main():
    parser = argparse.ArgumentParser(description="Verify 3D Lifting - Frame 1 with joint numbers")
    parser.add_argument('--keypoints', type=str, required=True,
                        help='Path to keypoints_2D_rtm.npz')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to video file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to MotionAGFormer checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("VERIFY 3D LIFTING - Frame 1 with Joint Numbers")
    print("=" * 70)
    
    # Check inputs
    keypoints_path = Path(args.keypoints)
    video_path = Path(args.video)
    
    if not keypoints_path.exists():
        print(f"❌ Keypoints not found: {keypoints_path}")
        return 1
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return 1
    
    # Setup checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = MODELS_DIR / "motionagformer" / "motionagformer-base-h36m.pth.tr"
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 1
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Device: {device}")
    
    # Load 2D keypoints (only Frame 1)
    print(f"\n📂 Loading 2D keypoints (Frame 1)...")
    data = np.load(keypoints_path)
    keypoints_2d = data['keypoints'][0:1]  # Shape: (1, 17, 2) - Only frame 0
    scores = data['scores'][0:1]           # Shape: (1, 17)
    
    print(f"   ✅ Loaded Frame 1: {keypoints_2d.shape}")
    
    # Load video frame
    print(f"\n🎬 Loading video frame...")
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"❌ Could not read frame from video")
        return 1
    
    print(f"   ✅ Loaded frame: {width}x{height}")
    
    # Draw 2D keypoints with numbers on frame
    frame_2d = frame.copy()
    kpts_2d = keypoints_2d[0]  # (17, 2)
    
    # Draw skeleton connections (optional - can skip if you want only points)
    # For now, just plot points with numbers
    
    for joint_idx in range(17):
        x, y = kpts_2d[joint_idx]
        x, y = int(x), int(y)
        
        # Draw point
        cv2.circle(frame_2d, (x, y), 5, (0, 255, 0), -1)
        
        # Draw joint number (small font)
        cv2.putText(frame_2d, f'{joint_idx}', (x + 7, y - 7), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Add batch dimension for conversion
    keypoints_2d_batch = keypoints_2d[np.newaxis, ...]  # (1, 1, 17, 2)
    scores_batch = scores[np.newaxis, ...]              # (1, 1, 17)
    
    # Convert COCO → H36M
    print(f"\n🔄 Converting COCO → H36M...")
    h36m_kpts, h36m_scores, valid_frames = h36m_coco_format(keypoints_2d_batch, scores_batch)
    h36m_keypoints = np.concatenate((h36m_kpts, h36m_scores[..., None]), axis=-1)
    print(f"   ✅ Converted: {h36m_keypoints.shape}")
    
    # Load model
    print(f"\n📦 Loading MotionAGFormer...")
    model = load_motionagformer_model(checkpoint_path, chunk_size=243, device=device)
    print(f"   ✅ Model loaded")
    
    # Run 3D inference
    print(f"\n🚀 Running 3D inference...")
    poses_3d = run_3d_inference(h36m_keypoints, model, width, height, device=device)
    print(f"   ✅ 3D pose: {poses_3d.shape}")
    
    # Get Frame 1 3D pose
    pose_3d = poses_3d[0].copy()
    
    # Apply rotation
    rot = np.array([0.1407056450843811, -0.1500701755285263, 
                    -0.755240797996521, 0.6223280429840088], dtype='float32')
    
    pose_3d = pose_3d - pose_3d[0:1, :]  # Center at Hip
    pose_3d = camera_to_world(pose_3d, R=rot, t=0)
    
    # Normalize
    pose_3d[:, 2] -= np.min(pose_3d[:, 2])
    max_value = np.max(pose_3d)
    if max_value > 0:
        pose_3d /= max_value
    
    # Render 3D pose
    print(f"\n🎨 Creating visualization...")
    fig = plt.figure(figsize=(height / 100, height / 100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    show3Dpose_with_numbers(pose_3d, ax)
    
    # Convert to image
    fig.canvas.draw()
    img_3d = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_3d = img_3d.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img_3d = cv2.cvtColor(img_3d, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    
    # Resize to match
    img_3d = cv2.resize(img_3d, (width, height))
    
    # Combine side by side
    combined = np.hstack([frame_2d, img_3d])
    
    # Save
    output_path = keypoints_path.parent / "verify_frame1_with_numbers.png"
    cv2.imwrite(str(output_path), combined)
    
    print(f"✅ Saved: {output_path}")
    print("=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
