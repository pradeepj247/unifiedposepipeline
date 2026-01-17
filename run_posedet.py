#!/usr/bin/env python3
"""
2D Pose Detection Script

This script performs 2D pose estimation on video frames using pre-computed detections.
It is a streamlined version of udp_video.py focused only on Stage 2 (2D pose estimation).

Supported models:
- RTMPose (COCO-17 keypoints)
- RTMPose Halpe26 (26 keypoints with feet)
- ViTPose (COCO-17 keypoints)
- Wholebody (133 keypoints with 3D)

Usage:
    python run_posedet.py --config configs/posedet.yaml

Output:
    NPZ file with 2D keypoints: kps_2d_<model_type>.npz
    - rtmpose → kps_2d_rtm.npz
    - vitpose → kps_2d_vit.npz
    - rtmpose_h26 → kps_2d_h26.npz
    - wholebody → kps_2d_wholebody.npz + kps_3d_wholebody.npz
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


def estimate_poses_wholebody(video_path, detections_path, config, max_frames, pose_model):
    """
    Estimate 3D whole-body poses using RTMPose3d (133 keypoints)
    Saves both 2D and 3D keypoints
    
    Args:
        video_path: Path to input video
        detections_path: Path to detections NPZ file
        config: Config dict
        max_frames: Maximum frames to process
        pose_model: Pre-initialized RTMPose3d model
    
    Returns:
        output_path_2d: Path to saved 2D keypoints NPZ file
        output_path_3d: Path to saved 3D keypoints NPZ file
        total_time: Processing time in seconds
        processing_fps: Processing FPS
    """
    print("\n📹 Processing video with Wholebody model (133 keypoints)...")
    
    # Load detections
    detections = np.load(detections_path)
    frame_numbers = detections['frame_numbers']
    bboxes = detections['bboxes']
    
    if max_frames is not None:
        frame_numbers = frame_numbers[:max_frames]
        bboxes = bboxes[:max_frames]
    
    print(f"   Loaded detections: {len(frame_numbers)} frames")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Storage for keypoints
    all_keypoints_2d = []
    all_keypoints_3d = []
    all_scores = []
    
    t_start = time.time()
    frames_processed = 0
    
    for frame_idx, bbox in zip(frame_numbers, bboxes):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if valid detection
        if bbox[2] > 0:  # Valid bbox
            # Run pose estimation (returns 3D coords, scores, simcc, 2D)
            keypoints_3d, scores, keypoints_simcc, keypoints_2d = pose_model(frame, bboxes=[bbox])
            if len(keypoints_2d) > 0:
                all_keypoints_2d.append(keypoints_2d[0])  # 2D projections
                all_keypoints_3d.append(keypoints_3d[0])  # 3D coordinates
                all_scores.append(scores[0])
            else:
                all_keypoints_2d.append(np.zeros((133, 2)))
                all_keypoints_3d.append(np.zeros((133, 3)))
                all_scores.append(np.zeros(133))
        else:
            # No detection, store empty
            all_keypoints_2d.append(np.zeros((133, 2)))
            all_keypoints_3d.append(np.zeros((133, 3)))
            all_scores.append(np.zeros(133))
        
        frames_processed += 1
        
        # Progress indicator
        if frames_processed % 30 == 0:
            print(f"   Processed {frames_processed}/{len(frame_numbers)} frames", end='\r')
    
    cap.release()
    t_end = time.time()
    
    total_time = t_end - t_start
    processing_fps = frames_processed / total_time if total_time > 0 else 0
    
    # Save 2D keypoints NPZ
    output_dir = REPO_ROOT / config.get('output_dir', 'outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path_2d = output_dir / "kps_2d_wholebody.npz"
    
    np.savez_compressed(
        output_path_2d,
        frame_numbers=frame_numbers,
        keypoints=np.array(all_keypoints_2d),
        scores=np.array(all_scores),
        joint_format="dwpose133_2d.json",
        model_type="wholebody"
    )
    
    # Save 3D keypoints NPZ
    output_path_3d = output_dir / "kps_3d_wholebody.npz"
    
    np.savez_compressed(
        output_path_3d,
        frame_numbers=frame_numbers,
        keypoints_3d=np.array(all_keypoints_3d),
        scores=np.array(all_scores),
        joint_format="dwpose133_3d.json",
        model_type="wholebody"
    )
    
    valid_poses = np.sum(np.array(all_scores)[:, 0] > 0)
    
    print(f"\n   ✅ Wholebody pose estimation complete!")
    print(f"   Processed: {frames_processed} frames")
    print(f"   Valid poses: {valid_poses}/{frames_processed}")
    print(f"   Processing FPS: {processing_fps:.1f}")
    print(f"   Time taken: {total_time:.2f}s")
    print(f"   Output 2D: {output_path_2d}")
    print(f"   Output 3D: {output_path_3d}")
    
    return output_path_2d, output_path_3d, total_time, processing_fps


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
    
    elif method == "wholebody":
        from rtmlib.tools import RTMPose3d
        model_path = PARENT_DIR / config["wholebody"]["pose_model_path"]
        pose_model = RTMPose3d(
            onnx_model=str(model_path),
            model_input_size=tuple(config["wholebody"]["pose_input_size"]),
            backend=config["wholebody"]["backend"],
            device=config["wholebody"]["device"]
        )
        input_size = config["wholebody"]["pose_input_size"]
        model_size = "L" if input_size[0] >= 288 else "M" if input_size[0] >= 256 else "S"
        print(f"   ✅ RTMPose3D-{model_size} Wholebody loaded ({input_size[0]}×{input_size[1]}, 133 keypoints)")
    
    else:
        print(f"   ❌ Unknown method: {method}")
        print("   Supported methods: rtmpose, rtmpose_h26, vitpose, wholebody")
        return 1
    
    video_path = REPO_ROOT / config["input_video"]
    
    # Auto-fallback: If video doesn't exist, try canonical_video.mp4 in output dir
    if not video_path.exists():
        # Try canonical_video.mp4 in the detections output directory
        detections_dir = detections_path.parent
        canonical_video = detections_dir / "canonical_video.mp4"
        
        if canonical_video.exists():
            video_path = canonical_video
            print(f"   📹 Using canonical video: {canonical_video}")
        else:
            print(f"\n❌ Error: Video file not found")
            print(f"   Tried:")
            print(f"   1. {REPO_ROOT / config['input_video']}")
            print(f"   2. {canonical_video}")
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
    elif method == "wholebody":
        output_path_2d, output_path_3d, total_time, processing_fps = estimate_poses_wholebody(
            video_path, detections_path, config, max_frames, pose_model
        )
        output_path = output_path_2d  # For display purposes
    
    print("\n" + "🎉" * 35)
    print("2D POSE DETECTION COMPLETE!")
    print("🎉" * 35)
    
    if method == "wholebody":
        print(f"\n   Output 2D file: {output_path_2d}")
        print(f"   Output 3D file: {output_path_3d}")
    else:
        print(f"\n   Output file: {output_path}")
    
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Processing FPS: {processing_fps:.1f}")
    print(f"\n   Next step: Run 3D lifting or visualization")
    print(f"   Example: python udp_3d_lifting.py --keypoints {output_path} --video {video_path}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
