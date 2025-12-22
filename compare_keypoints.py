"""
Compare keypoints from different methods:
1. 2D keypoints: RTMPose vs wb3d
2. 3D keypoints: MotionAGFormer lifting vs wb3d

Usage:
    python compare_keypoints.py --output_dir /path/to/outputs
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_keypoints(output_dir):
    """Load all keypoint files from output directory."""
    data = {}
    
    # Load RTMPose 2D keypoints (COCO-17 format)
    rtmpose_file = os.path.join(output_dir, 'keypoints_2D.npz')
    if os.path.exists(rtmpose_file):
        rtmpose_data = np.load(rtmpose_file)
        data['rtmpose_2d'] = rtmpose_data['keypoints']
        data['rtmpose_2d_scores'] = rtmpose_data['scores']
        print(f"✓ Loaded RTMPose 2D: {data['rtmpose_2d'].shape}")
    else:
        print(f"✗ RTMPose 2D file not found: {rtmpose_file}")
    
    # Load wb3d 2D keypoints (133 keypoints)
    wb_2d_file = os.path.join(output_dir, 'keypoints_2D_wb.npz')
    if os.path.exists(wb_2d_file):
        wb_2d_data = np.load(wb_2d_file)
        data['wb_2d'] = wb_2d_data['keypoints']
        data['wb_2d_scores'] = wb_2d_data['scores']
        print(f"✓ Loaded wb3d 2D: {data['wb_2d'].shape}")
    else:
        print(f"✗ wb3d 2D file not found: {wb_2d_file}")
    
    # Load wb3d 3D keypoints (133 keypoints)
    wb_3d_file = os.path.join(output_dir, 'keypoints_3D_wb.npz')
    if os.path.exists(wb_3d_file):
        wb_3d_data = np.load(wb_3d_file)
        # Check available keys and load accordingly
        if 'keypoints_3d' in wb_3d_data.files:
            data['wb_3d'] = wb_3d_data['keypoints_3d']
            data['wb_3d_scores'] = wb_3d_data['scores']
        elif 'keypoints' in wb_3d_data.files:
            data['wb_3d'] = wb_3d_data['keypoints']
            data['wb_3d_scores'] = wb_3d_data['scores']
        else:
            print(f"⚠ Available keys in {wb_3d_file}: {wb_3d_data.files}")
            return data
        print(f"✓ Loaded wb3d 3D: {data['wb_3d'].shape}")
    else:
        print(f"✗ wb3d 3D file not found: {wb_3d_file}")
    
    # Load MotionAGFormer 3D keypoints (H36M-17 format)
    magf_file = os.path.join(output_dir, 'keypoints_2D_3d.npy')
    if os.path.exists(magf_file):
        data['magf_3d'] = np.load(magf_file)
        print(f"✓ Loaded MotionAGFormer 3D: {data['magf_3d'].shape}")
    else:
        print(f"✗ MotionAGFormer 3D file not found: {magf_file}")
    
    return data


def compare_2d_keypoints(rtmpose_kpts, wb_kpts, rtmpose_scores, wb_scores):
    """Compare 2D keypoints from RTMPose and wb3d."""
    print("\n" + "="*70)
    print("2D KEYPOINTS COMPARISON: RTMPose vs wb3d")
    print("="*70)
    
    n_frames = rtmpose_kpts.shape[0]
    print(f"\nNumber of frames: {n_frames}")
    print(f"RTMPose shape: {rtmpose_kpts.shape} (frames, 17, 2) - COCO-17")
    print(f"wb3d shape: {wb_kpts.shape} (frames, 133, 2) - RTMPose3d wholebody")
    
    # Extract body keypoints from wb3d (first 17 are COCO body)
    wb_body = wb_kpts[:, :17, :]
    wb_body_scores = wb_scores[:, :17]
    
    print(f"\nExtracting first 17 keypoints from wb3d for comparison")
    print(f"wb3d body shape: {wb_body.shape}")
    
    # Calculate differences
    diff = rtmpose_kpts - wb_body
    abs_diff = np.abs(diff)
    
    print(f"\n--- Statistics ---")
    print(f"Mean absolute difference: {np.mean(abs_diff):.4f} pixels")
    print(f"Max absolute difference: {np.max(abs_diff):.4f} pixels")
    print(f"Min absolute difference: {np.min(abs_diff):.4f} pixels")
    print(f"Std absolute difference: {np.std(abs_diff):.4f} pixels")
    
    # Per-keypoint analysis
    print(f"\n--- Per-Keypoint Analysis (averaged over frames) ---")
    keypoint_names = [
        "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
        "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
        "L_Wrist", "R_Wrist", "L_Hip", "R_Hip",
        "L_Knee", "R_Knee", "L_Ankle", "R_Ankle"
    ]
    
    mean_diff_per_kpt = np.mean(abs_diff, axis=0)
    for i, name in enumerate(keypoint_names):
        x_diff = mean_diff_per_kpt[i, 0]
        y_diff = mean_diff_per_kpt[i, 1]
        euclidean = np.sqrt(x_diff**2 + y_diff**2)
        print(f"{i:2d}. {name:12s}: X={x_diff:6.2f}, Y={y_diff:6.2f}, Euclidean={euclidean:6.2f} pixels")
    
    # Score comparison
    print(f"\n--- Confidence Scores ---")
    print(f"RTMPose mean score: {np.mean(rtmpose_scores):.4f}")
    print(f"wb3d mean score: {np.mean(wb_body_scores):.4f}")
    
    # Correlation
    rtmpose_flat = rtmpose_kpts.reshape(-1)
    wb_flat = wb_body.reshape(-1)
    correlation = np.corrcoef(rtmpose_flat, wb_flat)[0, 1]
    print(f"\nPearson correlation: {correlation:.6f}")
    
    return {
        'mean_diff': np.mean(abs_diff),
        'max_diff': np.max(abs_diff),
        'correlation': correlation,
        'per_keypoint_diff': mean_diff_per_kpt
    }


def convert_h36m_to_coco(h36m_kpts):
    """
    Convert H36M-17 joint ordering to COCO-17 ordering for comparison.
    
    H36M-17: [0-Hip, 1-RHip, 2-RKnee, 3-RAnkle, 4-LHip, 5-LKnee, 6-LAnkle,
              7-Spine, 8-Thorax, 9-Nose, 10-Head, 11-LShoulder, 12-LElbow, 
              13-LWrist, 14-RShoulder, 15-RElbow, 16-RWrist]
    
    COCO-17: [0-Nose, 1-LEye, 2-REye, 3-LEar, 4-REar, 5-LShoulder, 6-RShoulder,
              7-LElbow, 8-RElbow, 9-LWrist, 10-RWrist, 11-LHip, 12-RHip,
              13-LKnee, 14-RKnee, 15-LAnkle, 16-RAnkle]
    
    Note: H36M doesn't have eyes/ears, so we'll use Head and Nose to approximate
    """
    n_frames = h36m_kpts.shape[0]
    coco_kpts = np.zeros((n_frames, 17, 3))
    
    # Direct mappings (joints that exist in both)
    coco_kpts[:, 0] = h36m_kpts[:, 9]   # Nose
    coco_kpts[:, 5] = h36m_kpts[:, 11]  # L_Shoulder
    coco_kpts[:, 6] = h36m_kpts[:, 14]  # R_Shoulder
    coco_kpts[:, 7] = h36m_kpts[:, 12]  # L_Elbow
    coco_kpts[:, 8] = h36m_kpts[:, 15]  # R_Elbow
    coco_kpts[:, 9] = h36m_kpts[:, 13]  # L_Wrist
    coco_kpts[:, 10] = h36m_kpts[:, 16] # R_Wrist
    coco_kpts[:, 11] = h36m_kpts[:, 4]  # L_Hip
    coco_kpts[:, 12] = h36m_kpts[:, 1]  # R_Hip
    coco_kpts[:, 13] = h36m_kpts[:, 5]  # L_Knee
    coco_kpts[:, 14] = h36m_kpts[:, 2]  # R_Knee
    coco_kpts[:, 15] = h36m_kpts[:, 6]  # L_Ankle
    coco_kpts[:, 16] = h36m_kpts[:, 3]  # R_Ankle
    
    # Approximate eyes and ears using Head and Nose
    # Eyes: slightly left/right of head position
    head = h36m_kpts[:, 10]  # Head
    nose = h36m_kpts[:, 9]   # Nose
    thorax = h36m_kpts[:, 8] # Thorax
    
    # Vector from thorax to head
    head_dir = head - thorax
    # Perpendicular vector (approximate left-right)
    eye_offset = np.zeros_like(head_dir)
    eye_offset[:, 0] = 0.05 * np.linalg.norm(head_dir, axis=1)  # 5% of head height
    
    coco_kpts[:, 1] = head - eye_offset  # L_Eye (left of head)
    coco_kpts[:, 2] = head + eye_offset  # R_Eye (right of head)
    
    # Ears: further left/right from eyes
    ear_offset = eye_offset * 1.5
    coco_kpts[:, 3] = head - ear_offset  # L_Ear
    coco_kpts[:, 4] = head + ear_offset  # R_Ear
    
    return coco_kpts


def normalize_to_hip_center(kpts_3d):
    """Normalize 3D keypoints to hip center (pelvis) and scale."""
    n_frames = kpts_3d.shape[0]
    normalized = np.zeros_like(kpts_3d)
    
    for i in range(n_frames):
        frame_kpts = kpts_3d[i]
        # Use hip center as origin (average of left and right hip)
        # COCO: L_Hip=11, R_Hip=12
        hip_center = (frame_kpts[11] + frame_kpts[12]) / 2
        
        # Center the skeleton
        centered = frame_kpts - hip_center
        
        # Scale by torso height (distance from hip center to shoulders)
        # Shoulders: L=5, R=6
        shoulder_center = (centered[5] + centered[6]) / 2
        torso_height = np.linalg.norm(shoulder_center)
        
        if torso_height > 0:
            normalized[i] = centered / torso_height
        else:
            normalized[i] = centered
    
    return normalized


def compare_3d_keypoints(magf_kpts, wb_kpts, wb_scores):
    """Compare 3D keypoints from MotionAGFormer and wb3d."""
    print("\n" + "="*70)
    print("3D KEYPOINTS COMPARISON: MotionAGFormer vs wb3d")
    print("="*70)
    
    n_frames_magf = magf_kpts.shape[0]
    n_frames_wb = wb_kpts.shape[0]
    
    print(f"\nMotionAGFormer shape: {magf_kpts.shape} (frames, 17, 3) - H36M-17")
    print(f"wb3d shape: {wb_kpts.shape} (frames, 133, 3) - RTMPose3d wholebody")
    
    if n_frames_magf != n_frames_wb:
        print(f"\n⚠ Warning: Frame count mismatch! MAGF={n_frames_magf}, wb3d={n_frames_wb}")
        n_frames = min(n_frames_magf, n_frames_wb)
        print(f"Using first {n_frames} frames for comparison")
        magf_kpts = magf_kpts[:n_frames]
        wb_kpts = wb_kpts[:n_frames]
        wb_scores = wb_scores[:n_frames]
    else:
        n_frames = n_frames_magf
        print(f"Number of frames: {n_frames}")
    
    # Extract body keypoints from wb3d (first 17 are COCO body)
    wb_body = wb_kpts[:, :17, :]
    wb_body_scores = wb_scores[:, :17]
    
    print(f"\n--- Step 1: Convert H36M to COCO ordering ---")
    magf_as_coco = convert_h36m_to_coco(magf_kpts)
    print(f"Converted MotionAGFormer to COCO format: {magf_as_coco.shape}")
    print(f"Note: Eyes/ears approximated from Head position (H36M doesn't have these)")
    
    print(f"\n--- Step 2: Normalize both to common coordinate system ---")
    print(f"Normalizing to hip-centered, torso-scaled coordinates...")
    magf_normalized = normalize_to_hip_center(magf_as_coco)
    wb_normalized = normalize_to_hip_center(wb_body)
    print(f"✓ Both methods now in same coordinate system")
    
    print(f"\n--- Step 3: Compare normalized 3D poses ---")
    
    # Calculate differences on normalized coordinates
    diff_3d = magf_normalized - wb_normalized
    abs_diff_3d = np.abs(diff_3d)
    euclidean_diff = np.sqrt(np.sum(diff_3d**2, axis=2))
    
    print(f"\nOverall Statistics (normalized coordinates):")
    print(f"  Mean absolute difference: {np.mean(abs_diff_3d):.4f} units")
    print(f"  Mean Euclidean distance: {np.mean(euclidean_diff):.4f} units")
    print(f"  Max Euclidean distance: {np.max(euclidean_diff):.4f} units")
    print(f"  Std Euclidean distance: {np.std(euclidean_diff):.4f} units")
    
    # Per-keypoint analysis
    print(f"\n--- Per-Keypoint 3D Comparison (averaged over frames) ---")
    keypoint_names = [
        "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
        "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
        "L_Wrist", "R_Wrist", "L_Hip", "R_Hip",
        "L_Knee", "R_Knee", "L_Ankle", "R_Ankle"
    ]
    
    mean_euclidean_per_kpt = np.mean(euclidean_diff, axis=0)
    for i, name in enumerate(keypoint_names):
        dist = mean_euclidean_per_kpt[i]
        print(f"{i:2d}. {name:12s}: {dist:.4f} units")
    
    # Calculate correlation in 3D space
    magf_flat = magf_normalized.reshape(-1)
    wb_flat = wb_normalized.reshape(-1)
    correlation_3d = np.corrcoef(magf_flat, wb_flat)[0, 1]
    print(f"\nPearson correlation (3D normalized): {correlation_3d:.6f}")
    
    
    # Confidence scores
    print(f"\n--- wb3d Confidence Scores ---")
    print(f"Mean score (body keypoints): {np.mean(wb_body_scores):.4f}")
    print(f"Min score: {np.min(wb_body_scores):.4f}")
    print(f"Max score: {np.max(wb_body_scores):.4f}")
    
    # Summary
    print(f"\n--- COMPARISON SUMMARY ---")
    mean_diff = np.mean(euclidean_diff)
    if mean_diff < 0.1:
        verdict = "EXCELLENT - Very close agreement"
    elif mean_diff < 0.3:
        verdict = "GOOD - Reasonable agreement"
    elif mean_diff < 0.5:
        verdict = "MODERATE - Some differences"
    else:
        verdict = "POOR - Large differences"
    
    print(f"Mean 3D distance: {mean_diff:.4f} units")
    print(f"3D correlation: {correlation_3d:.6f}")
    print(f"Verdict: {verdict}")
    
    return {
        'mean_euclidean_distance': mean_diff,
        'max_euclidean_distance': np.max(euclidean_diff),
        'correlation_3d': correlation_3d,
        'per_keypoint_distances': mean_euclidean_per_kpt,
        'magf_normalized': magf_normalized,
        'wb_normalized': wb_normalized
    }


def visualize_comparison(data, output_dir):
    """Create visualization comparing keypoints."""
    print("\n" + "="*70)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("="*70)
    
    # Select a middle frame for visualization
    if 'rtmpose_2d' in data and 'wb_2d' in data:
        n_frames = data['rtmpose_2d'].shape[0]
        mid_frame = n_frames // 2
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        # RTMPose 2D
        ax = axes[0]
        kpts = data['rtmpose_2d'][mid_frame]
        ax.scatter(kpts[:, 0], kpts[:, 1], c='blue', s=50, alpha=0.6)
        for i, (x, y) in enumerate(kpts):
            ax.text(x, y, str(i), fontsize=8, color='white', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.5))
        ax.set_title(f'RTMPose 2D (frame {mid_frame})\nCOCO-17 format')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # wb3d 2D (first 17 keypoints)
        ax = axes[1]
        kpts = data['wb_2d'][mid_frame, :17, :]
        ax.scatter(kpts[:, 0], kpts[:, 1], c='red', s=50, alpha=0.6)
        for i, (x, y) in enumerate(kpts):
            ax.text(x, y, str(i), fontsize=8, color='white',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.5))
        ax.set_title(f'wb3d 2D (frame {mid_frame})\nFirst 17 of 133 keypoints')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, 'comparison_2d_keypoints.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved 2D comparison: {output_file}")
        plt.close()
    
    # 3D visualization
    if 'magf_3d' in data and 'wb_3d' in data:
        n_frames = min(data['magf_3d'].shape[0], data['wb_3d'].shape[0])
        mid_frame = n_frames // 2
        
        fig = plt.figure(figsize=(14, 7))
        
        # MotionAGFormer 3D
        ax = fig.add_subplot(121, projection='3d')
        kpts = data['magf_3d'][mid_frame]
        ax.scatter(kpts[:, 0], kpts[:, 1], kpts[:, 2], c='blue', s=50, alpha=0.6)
        for i, (x, y, z) in enumerate(kpts):
            ax.text(x, y, z, str(i), fontsize=8)
        ax.set_title(f'MotionAGFormer 3D (frame {mid_frame})\nH36M-17 format')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # wb3d 3D (first 17 keypoints)
        ax = fig.add_subplot(122, projection='3d')
        kpts = data['wb_3d'][mid_frame, :17, :]
        ax.scatter(kpts[:, 0], kpts[:, 1], kpts[:, 2], c='red', s=50, alpha=0.6)
        for i, (x, y, z) in enumerate(kpts):
            ax.text(x, y, z, str(i), fontsize=8)
        ax.set_title(f'wb3d 3D (frame {mid_frame})\nFirst 17 of 133 keypoints')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, 'comparison_3d_keypoints.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved 3D comparison: {output_file}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare keypoints from different methods')
    parser.add_argument('--output_dir', type=str, 
                       default='/content/unifiedposepipeline/demo_data/outputs',
                       help='Directory containing output files')
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory not found: {args.output_dir}")
        return
    
    print("="*70)
    print("KEYPOINT COMPARISON TOOL")
    print("="*70)
    print(f"Output directory: {args.output_dir}\n")
    
    # Load all keypoint files
    data = load_keypoints(args.output_dir)
    
    # Compare 2D keypoints
    if 'rtmpose_2d' in data and 'wb_2d' in data:
        compare_2d_keypoints(
            data['rtmpose_2d'], 
            data['wb_2d'],
            data['rtmpose_2d_scores'],
            data['wb_2d_scores']
        )
    else:
        print("\n⚠ Cannot compare 2D keypoints - missing data files")
    
    # Compare 3D keypoints
    if 'magf_3d' in data and 'wb_3d' in data:
        compare_3d_keypoints(
            data['magf_3d'],
            data['wb_3d'],
            data['wb_3d_scores']
        )
    else:
        print("\n⚠ Cannot compare 3D keypoints - missing data files")
    
    # Generate visualizations
    if len(data) >= 2:
        visualize_comparison(data, args.output_dir)
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
