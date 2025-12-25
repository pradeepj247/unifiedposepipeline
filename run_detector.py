#!/usr/bin/env python3
"""
Unified Detector and Tracker Script

This script handles object detection and tracking for the pose estimation pipeline.
Currently supports:
- Detection: YOLOv8, RTMDet
- Tracking: BoT-SORT, DeepOCSORT, ByteTrack, StrongSORT, OCSORT (planned)
- ReID: OSNet-based re-identification (planned)

Usage:
    python run_detector.py --config configs/detector.yaml

Output:
    detections.npz with keys:
        - frame_numbers: (N,) array of frame indices
        - bboxes: (N, 4) array of [x1, y1, x2, y2] (integer coordinates)
"""

import argparse
import yaml
import numpy as np
import cv2
import time
from pathlib import Path
from tqdm import tqdm
import sys


def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_yolo_detector(model_path, device='cuda', confidence=0.3):
    """
    Load YOLOv8 detector
    
    Args:
        model_path: Path to YOLO model (.pt file)
        device: Device to run on ('cuda' or 'cpu')
        confidence: Confidence threshold
    
    Returns:
        YOLO model object
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics package not found. Install with: pip install ultralytics")
    
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    model.to(device)
    
    return model


def detect_yolo(model, frame, confidence=0.3, detect_only_humans=True):
    """
    Run YOLO detection on a single frame
    
    Args:
        model: YOLO model object
        frame: Input frame (BGR)
        confidence: Confidence threshold
        detect_only_humans: If True, only return person detections (class 0)
    
    Returns:
        bboxes: (N, 5) array of [x1, y1, x2, y2, confidence]
    """
    # Run inference
    results = model(frame, conf=confidence, verbose=False)
    
    # Extract detections
    detections = []
    
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            # Get class ID
            cls_id = int(box.cls[0].item())
            
            # Filter for person class (COCO class 0) if requested
            if detect_only_humans and cls_id != 0:
                continue
            
            # Get bbox coordinates and confidence
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].item()
            
            detections.append([x1, y1, x2, y2, conf])
    
    if len(detections) == 0:
        return np.array([])
    
    return np.array(detections)


def select_largest_bbox(bboxes):
    """
    Select the largest bounding box by area
    
    Args:
        bboxes: (N, 5) array of [x1, y1, x2, y2, confidence] from detector
    
    Returns:
        bbox: (4,) array of [x1, y1, x2, y2] (largest bbox, integers), or empty array if no detections
    """
    if len(bboxes) == 0:
        return np.array([])
    
    # Calculate areas
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    
    # Get index of largest area
    largest_idx = np.argmax(areas)
    
    # Return only coordinates (no confidence)
    return bboxes[largest_idx, :4]


def process_video(config):
    """
    Process video with detection and optional tracking
    
    Args:
        config: Configuration dictionary
    
    Returns:
        detections_data: Dictionary with frame_numbers, bboxes, scores
    """
    # Extract config parameters
    video_path = config['input']['video_path']
    max_frames = config['input']['max_frames']
    
    detector_type = config['detector']['type']
    model_path = config['detector']['model_path']
    confidence = config['detector']['confidence']
    device = config['detector']['device']
    detect_only_humans = config['detector']['detect_only_humans']
    
    tracking_enabled = config['tracking']['enabled']
    largest_bbox_only = config['tracking']['largest_bbox_only']
    
    verbose = config['advanced']['verbose']
    
    # Load detector
    if detector_type.lower() == 'yolo':
        detector = load_yolo_detector(model_path, device, confidence)
    else:
        raise NotImplementedError(f"Detector type '{detector_type}' not yet supported")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Determine number of frames to process
    if max_frames > 0:
        num_frames = min(max_frames, total_frames)
    else:
        num_frames = total_frames
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Video: {video_path}")
        print(f"Resolution: {width}x{height} @ {fps:.2f} fps")
        print(f"Total frames: {total_frames}")
        print(f"Processing: {num_frames} frames")
        print(f"Detector: {detector_type.upper()}")
        print(f"Tracking: {'Enabled' if tracking_enabled else 'Disabled (largest bbox only)'}")
        print(f"{'='*70}\n")
    
    # Storage for detections
    frame_numbers = []
    bboxes_list = []
    
    # Start timing
    t_start = time.time()
    
    # Process frames
    pbar = tqdm(total=num_frames, desc="Detecting", disable=not verbose)
    
    frame_idx = 0
    
    while frame_idx < num_frames:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Run detection
        if detector_type.lower() == 'yolo':
            detections = detect_yolo(detector, frame, confidence, detect_only_humans)
        else:
            raise NotImplementedError(f"Detector type '{detector_type}' not supported")
        
        # Handle tracking vs largest bbox selection
        if tracking_enabled:
            # TODO: Implement tracking logic
            raise NotImplementedError("Tracking not yet implemented")
        else:
            # Select largest bbox only
            frame_numbers.append(frame_idx)
            if len(detections) > 0:
                largest_bbox = select_largest_bbox(detections)  # Returns (4,) array
                bboxes_list.append(largest_bbox)  # [x1, y1, x2, y2]
            else:
                # No detection - store empty bbox
                bboxes_list.append([0, 0, 0, 0])
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    # End timing
    t_end = time.time()
    total_time = t_end - t_start
    processing_fps = len(frame_numbers) / total_time if total_time > 0 else 0
    
    # Convert to numpy arrays (match udp_video.py format exactly)
    frame_numbers = np.array(frame_numbers, dtype=np.int64)  # Default numpy dtype
    bboxes = np.array(bboxes_list, dtype=np.int64)  # Integer coordinates like udp_video.py
    
    detections_data = {
        'frame_numbers': frame_numbers,
        'bboxes': bboxes
    }
    
    if verbose:
        # Count valid detections (bboxes where x2 > 0)
        valid_detections = np.sum(bboxes[:, 2] > 0)
        print(f"\n✓ Detection complete!")
        print(f"  Total frames processed: {len(frame_numbers)}")
        print(f"  Frames with detections: {valid_detections}")
        print(f"  Detection rate: {valid_detections / len(frame_numbers) * 100:.1f}%")
        print(f"  Processing FPS: {processing_fps:.1f}")
        print(f"  Time taken: {total_time:.2f}s")
    
    return detections_data


def save_detections(detections_data, output_path, verbose=True):
    """
    Save detections to NPZ file
    
    Args:
        detections_data: Dictionary with frame_numbers, bboxes
        output_path: Path to output NPZ file
        verbose: Print save confirmation
    """
    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save NPZ (use compressed to match udp_video.py)
    np.savez_compressed(
        output_path,
        frame_numbers=detections_data['frame_numbers'],
        bboxes=detections_data['bboxes']
    )
    
    if verbose:
        print(f"\n✓ Saved detections to: {output_path}")
        print(f"  Shape: bboxes={detections_data['bboxes'].shape}")
        print(f"  Format: frame_numbers (int64), bboxes (int64, 4 values per frame)")


def main():
    parser = argparse.ArgumentParser(
        description='Unified Detector and Tracker for Pose Estimation Pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to detector configuration YAML file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Process video
    detections_data = process_video(config)
    
    # Save detections
    output_path = config['output']['detections_file']
    save_detections(detections_data, output_path, verbose=config['advanced']['verbose'])
    
    print("\n✓ Detection pipeline complete!\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
