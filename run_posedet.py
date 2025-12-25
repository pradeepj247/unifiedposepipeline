#!/usr/bin/env python3
"""
2D Pose Detection Script

This script performs 2D pose estimation on video frames using pre-computed detections.
It is a streamlined version of udp_video.py focused only on Stage 2 (2D pose estimation).

Supported models:
- RTMPose (COCO-17 keypoints)
- RTMPose Halpe26 (26 keypoints with feet)
- ViTPose (COCO-17 keypoints)

Usage:
    python run_2d_posedet.py --config configs/2d_posedet.yaml

Output:
    NPZ file with 2D keypoints: kps_2d_<model_type>.npz
    - rtmpose → kps_2d_rtm.npz
    - vitpose → kps_2d_vit.npz
    - rtmpose_h26 → kps_2d_h26.npz
"""

import sys
import argparse
from pathlib import Path
import yaml
import time
import cv2
import numpy as np

REPO_ROOT = Path(__file__).parent
PARENT_DIR = REPO_ROOT.parent
MODELS_DIR = PARENT_DIR / "models"


def estimate_poses_rtmpose(video_path, detections_path, config, max_frames, pose_model):
    """
    Estimate poses using RTMPose
    
    Args:
        video_path: Path to input video
        detections_path: Path to detections NPZ file
        config: Full config dict
        max_frames: Maximum frames to process (None for all)
        pose_model: Pre-initialized RTMPose model
    
    Returns:
        output_path: Path to saved NPZ file
        total_time: Processing time in seconds
        processing_fps: Processing FPS
    """
    print("\n" + "=" * 70)
    print("🎯 2D POSE ESTIMATION: RTMPose (COCO-17)")
    print("=" * 70)
    
    # Load detections
    detections = np.load(detections_path)
    frame_numbers = detections['frame_numbers']
    bboxes = detections['bboxes']
    
    # Apply max_frames limit if specified
    if max_frames and max_frames > 0:
        frame_numbers = frame_numbers[:max_frames]
        bboxes = bboxes[:max_frames]
    
    print(f"   Loaded detections: {len(frame_numbers)} frames")
    print(f"   Input video: {video_path.name}")
    print(f"   Model: RTMPose-L (17 keypoints)")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Storage for keypoints
    all_keypoints = []
    all_scores = []
    
    t_start = time.time()
    frames_processed = 0
    
    for frame_idx, bbox in zip(frame_numbers, bboxes):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if valid detection
        if bbox[2] > 0:  # Valid bbox (x2 > 0)
            # Run pose estimation
            keypoints, scores = pose_model(frame, bboxes=[bbox])
            if len(keypoints) > 0:
                all_keypoints.append(keypoints[0])  # Take first (only) detection
                all_scores.append(scores[0])
            else:
                all_keypoints.append(np.zeros((17, 2)))
                all_scores.append(np.zeros(17))
        else:
            # No detection, store empty
            all_keypoints.append(np.zeros((17, 2)))
            all_scores.append(np.zeros(17))
        
        frames_processed += 1
        
        # Progress indicator
        if frames_processed % 30 == 0:
            print(f"   Processed {frames_processed}/{len(frame_numbers)} frames", end='\r')
    
    cap.release()
    t_end = time.time()
    
    total_time = t_end - t_start
    processing_fps = frames_processed / total_time if total_time > 0 else 0
    
    # Save to NPZ
    output_dir = REPO_ROOT / config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "kps_2d_rtm.npz"
    
    np.savez_compressed(
        output_path,
        frame_numbers=frame_numbers,
        keypoints=np.array(all_keypoints),
        scores=np.array(all_scores),
        joint_format="coco17_2d.json",
        model_type="rtmpose"
    )
    
    valid_poses = np.sum(np.array(all_scores)[:, 0] > 0)
    
    print(f"\n   ✅ 2D Pose estimation complete!")
    print(f"   Processed: {frames_processed} frames")
    print(f"   Valid poses: {valid_poses}/{frames_processed}")
    print(f"   Processing FPS: {processing_fps:.1f}")
    print(f"   Time taken: {total_time:.2f}s")
    print(f"   Output: {output_path}")
    
    return output_path, total_time, processing_fps


def estimate_poses_rtmpose_h26(video_path, detections_path, config, max_frames, pose_model):
    """
    Estimate poses using RTMPose Halpe26 (26 keypoints)
    
    Args:
        video_path: Path to input video
        detections_path: Path to detections NPZ file
        config: Full config dict
        max_frames: Maximum frames to process (None for all)
        pose_model: Pre-initialized RTMPose Halpe26 model
    
    Returns:
        output_path: Path to saved NPZ file
        total_time: Processing time in seconds
        processing_fps: Processing FPS
    """
    print("\n" + "=" * 70)
    print("🎯 2D POSE ESTIMATION: RTMPose Halpe26 (26 keypoints)")
    print("=" * 70)
    
    # Load detections
    detections = np.load(detections_path)
    frame_numbers = detections['frame_numbers']
    bboxes = detections['bboxes']
    
    # Apply max_frames limit if specified
    if max_frames and max_frames > 0:
        frame_numbers = frame_numbers[:max_frames]
        bboxes = bboxes[:max_frames]
    
    print(f"   Loaded detections: {len(frame_numbers)} frames")
    print(f"   Input video: {video_path.name}")
    print(f"   Model: RTMPose-L Halpe26 (26 keypoints)")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Storage for keypoints
    all_keypoints = []
    all_scores = []
    
    t_start = time.time()
    frames_processed = 0
    
    for frame_idx, bbox in zip(frame_numbers, bboxes):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if valid detection
        if bbox[2] > 0:  # Valid bbox (x2 > 0)
            # Run pose estimation
            keypoints, scores = pose_model(frame, bboxes=[bbox])
            if len(keypoints) > 0:
                all_keypoints.append(keypoints[0])  # Take first (only) detection
                all_scores.append(scores[0])
            else:
                all_keypoints.append(np.zeros((26, 2)))
                all_scores.append(np.zeros(26))
        else:
            # No detection, store empty
            all_keypoints.append(np.zeros((26, 2)))
            all_scores.append(np.zeros(26))
        
        frames_processed += 1
        
        # Progress indicator
        if frames_processed % 30 == 0:
            print(f"   Processed {frames_processed}/{len(frame_numbers)} frames", end='\r')
    
    cap.release()
    t_end = time.time()
    
    total_time = t_end - t_start
    processing_fps = frames_processed / total_time if total_time > 0 else 0
    
    # Save to NPZ
    output_dir = REPO_ROOT / config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "kps_2d_h26.npz"
    
    np.savez_compressed(
        output_path,
        frame_numbers=frame_numbers,
        keypoints=np.array(all_keypoints),
        scores=np.array(all_scores),
        joint_format="halpe26_2d.json",
        model_type="rtmpose_h26"
    )
    
    valid_poses = np.sum(np.array(all_scores)[:, 0] > 0)
    
    print(f"\n   ✅ 2D Pose estimation complete!")
    print(f"   Processed: {frames_processed} frames")
    print(f"   Valid poses: {valid_poses}/{frames_processed}")
    print(f"   Processing FPS: {processing_fps:.1f}")
    print(f"   Time taken: {total_time:.2f}s")
    print(f"   Output: {output_path}")
    
    return output_path, total_time, processing_fps


def estimate_poses_vitpose(video_path, detections_path, config, max_frames, pose_model):
    """
    Estimate poses using ViTPose
    
    Args:
        video_path: Path to input video
        detections_path: Path to detections NPZ file
        config: Full config dict
        max_frames: Maximum frames to process (None for all)
        pose_model: Pre-initialized VitPoseOnly model
    
    Returns:
        output_path: Path to saved NPZ file
        total_time: Processing time in seconds
        processing_fps: Processing FPS
    """
    print("\n" + "=" * 70)
    print("🎯 2D POSE ESTIMATION: ViTPose (COCO-17)")
    print("=" * 70)
    
    # Load detections
    detections = np.load(detections_path)
    frame_numbers = detections['frame_numbers']
    bboxes = detections['bboxes']
    
    # Apply max_frames limit if specified
    if max_frames and max_frames > 0:
        frame_numbers = frame_numbers[:max_frames]
        bboxes = bboxes[:max_frames]
    
    print(f"   Loaded detections: {len(frame_numbers)} frames")
    print(f"   Input video: {video_path.name}")
    print(f"   Model: ViTPose-B (17 keypoints)")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Storage for keypoints
    all_keypoints = []
    all_scores = []
    
    t_start = time.time()
    frames_processed = 0
    
    for frame_idx, bbox in zip(frame_numbers, bboxes):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if valid detection
        if bbox[2] > 0:  # Valid bbox (x2 > 0)
            # Convert BGR to RGB for ViTPose
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run pose estimation
            kpts = pose_model.inference_bbox(frame_rgb, bbox)
            if len(kpts) > 0:
                # ViTPose returns (17, 3) with [y, x, conf]
                keypoints_xy = np.stack([kpts[:, 1], kpts[:, 0]], axis=1)  # [x, y]
                scores = kpts[:, 2]
                all_keypoints.append(keypoints_xy)
                all_scores.append(scores)
            else:
                all_keypoints.append(np.zeros((17, 2)))
                all_scores.append(np.zeros(17))
        else:
            # No detection, store empty
            all_keypoints.append(np.zeros((17, 2)))
            all_scores.append(np.zeros(17))
        
        frames_processed += 1
        
        # Progress indicator
        if frames_processed % 30 == 0:
            print(f"   Processed {frames_processed}/{len(frame_numbers)} frames", end='\r')
    
    cap.release()
    t_end = time.time()
    
    total_time = t_end - t_start
    processing_fps = frames_processed / total_time if total_time > 0 else 0
    
    # Save to NPZ
    output_dir = REPO_ROOT / config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "kps_2d_vit.npz"
    
    np.savez_compressed(
        output_path,
        frame_numbers=frame_numbers,
        keypoints=np.array(all_keypoints),
        scores=np.array(all_scores),
        joint_format="coco17_2d.json",
        model_type="vitpose"
    )
    
    valid_poses = np.sum(np.array(all_scores)[:, 0] > 0)
    
    print(f"\n   ✅ 2D Pose estimation complete!")
    print(f"   Processed: {frames_processed} frames")
    print(f"   Valid poses: {valid_poses}/{frames_processed}")
    print(f"   Processing FPS: {processing_fps:.1f}")
    print(f"   Time taken: {total_time:.2f}s")
    print(f"   Output: {output_path}")
    
    return output_path, total_time, processing_fps


def main():
    parser = argparse.ArgumentParser(
        description="2D Pose Detection - Standalone pose estimation with pre-computed detections"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    
    # Load config
    config_file = Path(args.config)
    if not config_file.is_absolute():
        config_file = REPO_ROOT / config_file
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    method = config.get("pose_estimation_model", "rtmpose").lower().strip()
    max_frames = config.get("max_frames", None)
    if max_frames == 0:
        max_frames = None
    
    print("\n" + "🎬" * 35)
    print(f"2D POSE DETECTION - {method.upper()} Mode")
    print("🎬" * 35)
    print(f"   Max frames: {max_frames if max_frames else 'All'}")
    print(f"   Using pre-computed detections: {config['detections_file']}")
    
    # Validate detections file exists
    detections_path = REPO_ROOT / config['detections_file']
    if not detections_path.exists():
        print(f"\n❌ Error: Detections file not found: {detections_path}")
        print("   Run detector first: python run_detector.py --config configs/detector.yaml")
        return 1
    
    # Initialize pose model
    print(f"\n📦 Loading {method.upper()} pose estimator...")
    sys.path.insert(0, str(REPO_ROOT / "lib"))
    
    if method == "rtmpose":
        from rtmlib.tools import RTMPose
        model_path = PARENT_DIR / config["rtmpose"]["pose_model_path"]
        pose_model = RTMPose(
            onnx_model=str(model_path),
            model_input_size=tuple(config["rtmpose"]["pose_input_size"]),
            backend=config["rtmpose"]["backend"],
            device=config["rtmpose"]["device"]
        )
        input_size = config["rtmpose"]["pose_input_size"]
        model_size = "L" if input_size[0] >= 288 else "M" if input_size[0] >= 256 else "S"
        print(f"   ✅ RTMPose-{model_size} loaded ({input_size[0]}×{input_size[1]}, 17 keypoints)")
    
    elif method == "rtmpose_h26":
        from rtmlib.tools import RTMPose
        model_path = PARENT_DIR / config["rtmpose_h26"]["pose_model_path"]
        pose_model = RTMPose(
            onnx_model=str(model_path),
            model_input_size=tuple(config["rtmpose_h26"]["pose_input_size"]),
            backend=config["rtmpose_h26"]["backend"],
            device=config["rtmpose_h26"]["device"]
        )
        input_size = config["rtmpose_h26"]["pose_input_size"]
        model_size = "L" if input_size[0] >= 288 else "M" if input_size[0] >= 256 else "S"
        print(f"   ✅ RTMPose-{model_size} Halpe26 loaded ({input_size[0]}×{input_size[1]}, 26 keypoints)")
    
    elif method == "vitpose":
        from vitpose.pose_only import VitPoseOnly
        model_path = PARENT_DIR / config["vitpose"]["model_path"]
        pose_model = VitPoseOnly(
            model=str(model_path),
            model_name=config["vitpose"]["model_name"],
            dataset=config["vitpose"]["dataset"],
            device=config["vitpose"]["device"]
        )
        print(f"   ✅ ViTPose-{config['vitpose']['model_name'].upper()} loaded (17 keypoints)")
    
    else:
        print(f"   ❌ Unknown method: {method}")
        print("   Supported methods: rtmpose, rtmpose_h26, vitpose")
        return 1
    
    video_path = REPO_ROOT / config["input_video"]
    if not video_path.exists():
        print(f"\n❌ Error: Video file not found: {video_path}")
        return 1
    
    # Run 2D pose estimation
    if method == "rtmpose":
        output_path, total_time, processing_fps = estimate_poses_rtmpose(
            video_path, detections_path, config, max_frames, pose_model
        )
    elif method == "rtmpose_h26":
        output_path, total_time, processing_fps = estimate_poses_rtmpose_h26(
            video_path, detections_path, config, max_frames, pose_model
        )
    elif method == "vitpose":
        output_path, total_time, processing_fps = estimate_poses_vitpose(
            video_path, detections_path, config, max_frames, pose_model
        )
    
    print("\n" + "🎉" * 35)
    print("2D POSE DETECTION COMPLETE!")
    print("🎉" * 35)
    print(f"\n   Output file: {output_path}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Processing FPS: {processing_fps:.1f}")
    print(f"\n   Next step: Run 3D lifting or visualization")
    print(f"   Example: python udp_3d_lifting.py --keypoints {output_path} --video {video_path}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
