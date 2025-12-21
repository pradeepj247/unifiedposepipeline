"""
UDP Video Demo - Three-stage video processing pipeline

Stage 1: Person detection (YOLOv8s) - saves largest bbox per frame
Stage 2: 2D pose estimation (RTMPose/ViTPose) - uses detected bboxes
Stage 3: Visualization (optional) - draws skeleton on video

Configurable pose estimation method and max frames processing.

Usage:
    python udp_video.py --config configs/udp_video.yaml
"""

import sys
import argparse
from pathlib import Path
import yaml
import time
import cv2
import numpy as np
import matplotlib.colors

REPO_ROOT = Path(__file__).parent
PARENT_DIR = REPO_ROOT.parent
MODELS_DIR = PARENT_DIR / "models"

# COCO skeleton edges (pre-defined for performance)
COCO_EDGES = [
    (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
    (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
    (12, 14), (14, 16), (5, 6), (11, 12)
]

# Pre-compute rainbow colors for skeleton edges
EDGE_COLORS = [
    tuple([int(c * 255) for c in matplotlib.colors.hsv_to_rgb([i/float(len(COCO_EDGES)), 1.0, 1.0])])
    for i in range(len(COCO_EDGES))
]


def stage1_detect_persons(video_path, yolo, config, max_frames):
    """
    Stage 1: Detect persons in video and save largest bbox per frame
    
    Args:
        video_path: Path to input video
        yolo: YOLO model instance
        config: Detection config dict
        max_frames: Maximum frames to process
    
    Returns:
        output_path: Path to saved NPZ file
        total_time: Processing time in seconds
        fps: Processing FPS
    """
    print("\n" + "=" * 70)
    print("🎯 STAGE 1: Person Detection (YOLOv8s)")
    print("=" * 70)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = min(max_frames, total_frames) if max_frames else total_frames
    
    print(f"   Video: {video_path.name}")
    print(f"   Total frames: {total_frames}")
    print(f"   Processing: {frames_to_process} frames")
    print(f"   Confidence threshold: {config['confidence_threshold']}")
    
    # Storage for detections
    frame_numbers = []
    bboxes = []
    
    t_start = time.time()
    frame_idx = 0
    
    while frame_idx < frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect persons (class 0)
        results = yolo(frame, classes=[0], verbose=False)
        
        # Find largest bbox
        largest_bbox = None
        largest_area = 0
        
        for result in results:
            for box in result.boxes:
                if box.conf[0] >= config['confidence_threshold']:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    area = (x2 - x1) * (y2 - y1)
                    if area > largest_area:
                        largest_area = area
                        largest_bbox = [int(x1), int(y1), int(x2), int(y2)]
        
        # Store result (even if no detection, store empty)
        frame_numbers.append(frame_idx)
        if largest_bbox is not None:
            bboxes.append(largest_bbox)
        else:
            bboxes.append([0, 0, 0, 0])  # No detection marker
        
        frame_idx += 1
        
        # Progress indicator
        if frame_idx % 30 == 0:
            print(f"   Processed {frame_idx}/{frames_to_process} frames", end='\r')
    
    cap.release()
    t_end = time.time()
    
    total_time = t_end - t_start
    processing_fps = frames_to_process / total_time
    
    # Save to NPZ
    output_path = REPO_ROOT / config['output']['stage1_detections']
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        frame_numbers=np.array(frame_numbers),
        bboxes=np.array(bboxes)
    )
    
    valid_detections = np.sum(np.array(bboxes)[:, 2] > 0)  # Count non-empty bboxes
    
    print(f"\n   ✅ Stage 1 complete!")
    print(f"   Processed: {frames_to_process} frames")
    print(f"   Valid detections: {valid_detections}/{frames_to_process}")
    print(f"   Time: {total_time:.2f}s")
    print(f"   FPS: {processing_fps:.1f}")
    print(f"   Output: {output_path}")
    
    return output_path, total_time, processing_fps


def stage2_estimate_poses_rtmpose(video_path, detections_path, config, max_frames, pose_model):
    """
    Stage 2: Estimate poses using RTMPose
    
    Args:
        video_path: Path to input video
        detections_path: Path to Stage 1 NPZ file
        config: Pose config dict
        max_frames: Maximum frames to process
        pose_model: Pre-initialized RTMPose model
    
    Returns:
        output_path: Path to saved NPZ file
        total_time: Processing time in seconds
        fps: Processing FPS
    """
    print("\n" + "=" * 70)
    print("🎯 STAGE 2: 2D Pose Estimation (RTMPose)")
    print("=" * 70)
    
    # Load detections
    detections = np.load(detections_path)
    frame_numbers = detections['frame_numbers']
    bboxes = detections['bboxes']
    
    print(f"   Loaded detections: {len(frame_numbers)} frames")
    
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
        if bbox[2] > 0:  # Valid bbox
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
    processing_fps = frames_processed / total_time
    
    # Save to NPZ
    output_path = REPO_ROOT / config['output']['stage2_keypoints']
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        frame_numbers=frame_numbers,
        keypoints=np.array(all_keypoints),
        scores=np.array(all_scores)
    )
    
    valid_poses = np.sum(np.array(all_scores)[:, 0] > 0)
    
    print(f"\n   ✅ Stage 2 complete!")
    print(f"   Processed: {frames_processed} frames")
    print(f"   Valid poses: {valid_poses}/{frames_processed}")
    print(f"   Time: {total_time:.2f}s")
    print(f"   FPS: {processing_fps:.1f}")
    print(f"   Output: {output_path}")
    
    return output_path, total_time, processing_fps


def stage2_estimate_poses_vitpose(video_path, detections_path, config, max_frames, pose_model):
    """
    Stage 2: Estimate poses using ViTPose
    
    Args:
        video_path: Path to input video
        detections_path: Path to Stage 1 NPZ file
        config: Pose config dict
        max_frames: Maximum frames to process
        pose_model: Pre-initialized VitPoseOnly model
    
    Returns:
        output_path: Path to saved NPZ file
        total_time: Processing time in seconds
        fps: Processing FPS
    """
    print("\n" + "=" * 70)
    print("🎯 STAGE 2: 2D Pose Estimation (ViTPose)")
    print("=" * 70)
    
    # Load detections
    detections = np.load(detections_path)
    frame_numbers = detections['frame_numbers']
    bboxes = detections['bboxes']
    
    print(f"   Loaded detections: {len(frame_numbers)} frames")
    
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
        if bbox[2] > 0:  # Valid bbox
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
    processing_fps = frames_processed / total_time
    
    # Save to NPZ
    output_path = REPO_ROOT / config['output']['stage2_keypoints']
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        frame_numbers=frame_numbers,
        keypoints=np.array(all_keypoints),
        scores=np.array(all_scores)
    )
    
    valid_poses = np.sum(np.array(all_scores)[:, 0] > 0)
    
    print(f"\n   ✅ Stage 2 complete!")
    print(f"   Processed: {frames_processed} frames")
    print(f"   Valid poses: {valid_poses}/{frames_processed}")
    print(f"   Time: {total_time:.2f}s")
    print(f"   FPS: {processing_fps:.1f}")
    print(f"   Output: {output_path}")
    
    return output_path, total_time, processing_fps


def draw_skeleton_unified(image, keypoints, scores, kpt_thr=0.5):
    """
    Unified colorful skeleton drawing
    
    Args:
        image: Input image (BGR)
        keypoints: Keypoints array (17 x 2) in [x, y] format
        scores: Confidence scores (17,)
        kpt_thr: Confidence threshold
    
    Returns:
        result_image: Annotated image
    """
    result_image = image.copy()
    
    # Draw skeleton lines
    for ie, (start_idx, end_idx) in enumerate(COCO_EDGES):
        if scores[start_idx] > kpt_thr and scores[end_idx] > kpt_thr:
            cv2.line(result_image, 
                   (int(keypoints[start_idx, 0]), int(keypoints[start_idx, 1])),
                   (int(keypoints[end_idx, 0]), int(keypoints[end_idx, 1])),
                   EDGE_COLORS[ie], 2, lineType=cv2.LINE_AA)
    
    # Draw keypoints
    for p in range(len(keypoints)):
        if scores[p] > kpt_thr:
            cv2.circle(result_image, 
                     (int(keypoints[p, 0]), int(keypoints[p, 1])), 
                     4, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    
    return result_image


def stage3_visualize(video_path, keypoints_path, config):
    """
    Stage 3: Visualize poses on video (optional)
    
    Args:
        video_path: Path to input video
        keypoints_path: Path to Stage 2 NPZ file
        config: Output config dict
    
    Returns:
        output_path: Path to saved video
        total_time: Processing time in seconds
    """
    print("\n" + "=" * 70)
    print("🎯 STAGE 3: Visualization")
    print("=" * 70)
    
    # Load keypoints
    data = np.load(keypoints_path)
    frame_numbers = data['frame_numbers']
    keypoints = data['keypoints']
    scores = data['scores']
    
    print(f"   Loaded keypoints: {len(frame_numbers)} frames")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    output_path = REPO_ROOT / config['output']['video_output']
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    t_start = time.time()
    frames_processed = 0
    
    for frame_idx, kpts, scrs in zip(frame_numbers, keypoints, scores):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw skeleton if valid pose
        if scrs[0] > 0:  # Check if valid pose
            frame = draw_skeleton_unified(frame, kpts, scrs, kpt_thr=0.5)
        
        out.write(frame)
        frames_processed += 1
        
        # Progress indicator
        if frames_processed % 30 == 0:
            print(f"   Processed {frames_processed}/{len(frame_numbers)} frames", end='\r')
    
    cap.release()
    out.release()
    t_end = time.time()
    
    total_time = t_end - t_start
    
    print(f"\n   ✅ Stage 3 complete!")
    print(f"   Processed: {frames_processed} frames")
    print(f"   Time: {total_time:.2f}s")
    print(f"   Output: {output_path}")
    
    return output_path, total_time


def main():
    parser = argparse.ArgumentParser(description="UDP Video Demo - 3-stage pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    
    # Load config
    config_file = Path(args.config)
    if not config_file.is_absolute():
        config_file = REPO_ROOT / config_file
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    method = config.get("pose_estimation", {}).get("method", "rtmpose")
    max_frames = config.get("video", {}).get("max_frames", None)
    plot_enabled = config.get("output", {}).get("plot", True)
    
    print("\n" + "🎬" * 35)
    print(f"UDP VIDEO DEMO - {method.upper()} Mode")
    print("🎬" * 35)
    print(f"   Max frames: {max_frames if max_frames else 'All'}")
    print(f"   Plot: {'Enabled' if plot_enabled else 'Disabled'}")
    
    # Initialize YOLO
    print("\n📦 Loading YOLO detector...")
    from ultralytics import YOLO
    
    # Handle both full paths and filenames for YOLO model
    yolo_config_path = config["detection"]["model_path"]
    if "/" in yolo_config_path or "\\" in yolo_config_path:
        # Full path provided - check multiple locations
        yolo_path = PARENT_DIR / yolo_config_path
        if not yolo_path.exists():
            yolo_path = REPO_ROOT / yolo_config_path
    else:
        # Just filename provided
        yolo_path = MODELS_DIR / "yolo" / yolo_config_path
        if not yolo_path.exists():
            yolo_path = REPO_ROOT / yolo_config_path
    
    yolo = YOLO(str(yolo_path))
    print(f"   ✅ Loaded {yolo_path.name}")
    
    # Initialize pose model
    print(f"\n📦 Loading {method.upper()} pose estimator...")
    sys.path.insert(0, str(REPO_ROOT / "lib"))
    
    if method == "rtmpose":
        from rtmlib.tools import RTMPose
        pose_model = RTMPose(
            onnx_model=config["pose_estimation"]["rtmpose"]["pose_model_url"],
            model_input_size=tuple(config["pose_estimation"]["rtmpose"]["pose_input_size"]),
            backend=config["pose_estimation"]["rtmpose"]["backend"],
            device=config["pose_estimation"]["rtmpose"]["device"]
        )
        print(f"   ✅ RTMPose-M loaded")
    elif method == "vitpose":
        from vitpose.pose_only import VitPoseOnly
        model_path = PARENT_DIR / config["pose_estimation"]["vitpose"]["model_path"]
        pose_model = VitPoseOnly(
            model=str(model_path),
            model_name=config["pose_estimation"]["vitpose"]["model_name"],
            dataset=config["pose_estimation"]["vitpose"]["dataset"],
            device=config["pose_estimation"]["vitpose"]["device"]
        )
        print(f"   ✅ ViTPose-{config['pose_estimation']['vitpose']['model_name'].upper()} loaded")
    else:
        print(f"   ❌ Unknown method: {method}")
        return 1
    
    video_path = REPO_ROOT / config["video"]["input_path"]
    
    # Stage 1: Detection
    detections_path, stage1_time, stage1_fps = stage1_detect_persons(
        video_path, yolo, config["detection"], max_frames
    )
    
    # Stage 2: Pose Estimation
    if method == "rtmpose":
        keypoints_path, stage2_time, stage2_fps = stage2_estimate_poses_rtmpose(
            video_path, detections_path, config, max_frames, pose_model
        )
    elif method == "vitpose":
        keypoints_path, stage2_time, stage2_fps = stage2_estimate_poses_vitpose(
            video_path, detections_path, config, max_frames, pose_model
        )
    
    # Stage 3: Visualization (optional)
    if plot_enabled:
        video_output_path, stage3_time = stage3_visualize(
            video_path, keypoints_path, config
        )
    else:
        print("\n⏭️  Stage 3 skipped (plot=false)")
        stage3_time = 0
    
    # Final summary
    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY")
    print("=" * 70)
    print(f"   Stage 1 (Detection): {stage1_time:.2f}s @ {stage1_fps:.1f} FPS")
    print(f"   Stage 2 (Pose): {stage2_time:.2f}s @ {stage2_fps:.1f} FPS")
    if plot_enabled:
        print(f"   Stage 3 (Visualization): {stage3_time:.2f}s")
    print(f"   Total time: {(stage1_time + stage2_time + stage3_time):.2f}s")
    print("=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
