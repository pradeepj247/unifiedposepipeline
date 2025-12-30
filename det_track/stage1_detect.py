#!/usr/bin/env python3
"""
Stage 1: Detection

Runs YOLO detector on video and saves all person detections to NPZ format.
Supports downscaling for faster processing.

Usage:
    python stage1_detect.py --config configs/pipeline_config.yaml
"""

import argparse
import yaml
import numpy as np
import cv2
import time
import re
import os
import sys
from pathlib import Path
from tqdm import tqdm


def resolve_path_variables(config):
    """Recursively resolve ${variable} in config"""
    global_vars = config.get('global', {})
    
    def resolve_string(s):
        return re.sub(
            r'\$\{(\w+)\}',
            lambda m: str(global_vars.get(m.group(1), m.group(0))),
            s
        )
    
    def resolve_recursive(obj):
        if isinstance(obj, dict):
            return {k: resolve_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_recursive(v) for v in obj]
        elif isinstance(obj, str):
            return resolve_string(obj)
        return obj
    
    return resolve_recursive(config)


def load_config(config_path):
    """Load and resolve YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return resolve_path_variables(config)


def load_yolo_detector(model_path, device='cuda', verbose=False):
    """Load YOLO detector"""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not found. Install with: pip install ultralytics")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    
    if verbose:
        print(f"  ✅ Loading YOLO model: {model_path}")
    
    model = YOLO(model_path)
    model.to(device)
    
    return model


def detect_frame(model, frame, confidence=0.3, detect_only_humans=True):
    """
    Run YOLO detection on a single frame
    
    Returns:
        detections: (N, 5) array of [x1, y1, x2, y2, confidence]
        classes: (N,) array of class IDs
    """
    results = model(frame, conf=confidence, verbose=False)
    
    detections = []
    classes = []
    
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            cls_id = int(box.cls[0].item())
            
            # Filter for person class (COCO class 0)
            if detect_only_humans and cls_id != 0:
                continue
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].item()
            
            detections.append([x1, y1, x2, y2, conf])
            classes.append(cls_id)
    
    if len(detections) == 0:
        return np.array([]), np.array([])
    
    return np.array(detections, dtype=np.float32), np.array(classes, dtype=np.int64)


def filter_detections(detections, classes, method='hybrid', max_count=15, min_confidence=0.3):
    """
    Filter detections based on method
    
    Args:
        method: 'top_n', 'confidence', or 'hybrid'
        max_count: Maximum detections to keep
        min_confidence: Minimum confidence threshold
    """
    if len(detections) == 0:
        return detections, classes
    
    if method == 'top_n':
        # Keep top N by confidence
        if len(detections) > max_count:
            indices = np.argsort(detections[:, 4])[-max_count:]
            return detections[indices], classes[indices]
    
    elif method == 'confidence':
        # Keep all above threshold
        mask = detections[:, 4] >= min_confidence
        return detections[mask], classes[mask]
    
    elif method == 'hybrid':
        # Keep top N that are above threshold
        mask = detections[:, 4] >= min_confidence
        filtered_dets = detections[mask]
        filtered_cls = classes[mask]
        
        if len(filtered_dets) > max_count:
            indices = np.argsort(filtered_dets[:, 4])[-max_count:]
            return filtered_dets[indices], filtered_cls[indices]
        return filtered_dets, filtered_cls
    
    return detections, classes


def run_detection(config):
    """Run Stage 1: Detection"""
    
    stage_config = config['stage1_detect']
    verbose = stage_config.get('advanced', {}).get('verbose', False)
    
    # Extract configuration
    detector_config = stage_config['detector']
    detection_limit = stage_config['detection_limit']
    processing_config = stage_config['processing']
    input_config = stage_config['input']
    output_config = stage_config['output']
    
    model_path = detector_config['model_path']
    confidence = detector_config['confidence']
    device = detector_config['device']
    detect_only_humans = detector_config['detect_only_humans']
    
    video_path = input_config['video_path']
    max_frames = input_config['max_frames']
    
    detections_file = output_config['detections_file']
    
    processing_resolution = processing_config.get('processing_resolution')
    
    # Print header
    print(f"\n{'='*70}")
    print(f"📍 STAGE 1: DETECTION")
    print(f"{'='*70}\n")
    
    # Load detector
    print(f"🛠️  Loading detector...")
    detector = load_yolo_detector(model_path, device, verbose)
    print(f"  ✅ YOLO model loaded")
    
    # Open video
    print(f"\n📹 Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Determine processing resolution
    if processing_resolution is not None:
        proc_width, proc_height = processing_resolution
        scale_x = width / proc_width
        scale_y = height / proc_height
        use_downscaling = True
    else:
        proc_width, proc_height = width, height
        scale_x, scale_y = 1.0, 1.0
        use_downscaling = False
    
    num_frames = min(max_frames, total_frames) if max_frames > 0 else total_frames
    
    print(f"  Resolution: {width}x{height} @ {fps:.2f} fps")
    if use_downscaling:
        print(f"  Processing: {proc_width}x{proc_height} (downscaled {scale_x:.2f}x)")
    print(f"  Total frames: {total_frames}")
    print(f"  Processing: {num_frames} frames")
    
    # Storage
    all_frame_numbers = []
    all_bboxes = []
    all_confidences = []
    all_classes = []
    num_detections_per_frame = []
    
    # Process frames
    print(f"\n⚡ Running detection...")
    t_start = time.time()
    
    pbar = tqdm(total=num_frames, desc="Detecting")
    
    frame_idx = 0
    while frame_idx < num_frames:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Downscale if needed
        if use_downscaling:
            frame_proc = cv2.resize(frame, (proc_width, proc_height))
        else:
            frame_proc = frame
        
        # Detect
        detections, classes = detect_frame(detector, frame_proc, confidence, detect_only_humans)
        
        # Scale back to original resolution
        if use_downscaling and len(detections) > 0:
            detections[:, 0] *= scale_x  # x1
            detections[:, 1] *= scale_y  # y1
            detections[:, 2] *= scale_x  # x2
            detections[:, 3] *= scale_y  # y2
        
        # Filter detections
        detections, classes = filter_detections(
            detections, classes,
            method=detection_limit['method'],
            max_count=detection_limit['max_count'],
            min_confidence=detection_limit['min_confidence']
        )
        
        # Store detections for this frame
        num_dets = len(detections)
        num_detections_per_frame.append(num_dets)
        
        for i in range(num_dets):
            all_frame_numbers.append(frame_idx)
            all_bboxes.append(detections[i, :4])
            all_confidences.append(detections[i, 4])
            all_classes.append(classes[i])
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    t_end = time.time()
    total_time = t_end - t_start
    processing_fps = num_frames / total_time if total_time > 0 else 0
    
    # Convert to numpy arrays
    frame_numbers = np.array(all_frame_numbers, dtype=np.int64)
    bboxes = np.array(all_bboxes, dtype=np.float32)
    confidences = np.array(all_confidences, dtype=np.float32)
    classes_array = np.array(all_classes, dtype=np.int64)
    num_detections_per_frame = np.array(num_detections_per_frame, dtype=np.int64)
    
    # Summary
    total_detections = len(frame_numbers)
    avg_detections_per_frame = total_detections / num_frames if num_frames > 0 else 0
    
    print(f"\n✅ Detection complete!")
    print(f"  Frames processed: {num_frames}")
    print(f"  Total detections: {total_detections}")
    print(f"  Avg detections/frame: {avg_detections_per_frame:.1f}")
    print(f"  Processing FPS: {processing_fps:.1f}")
    print(f"  Time taken: {total_time:.2f}s")
    
    # Save NPZ
    print(f"\n💾 Saving detections...")
    output_path = Path(detections_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        frame_numbers=frame_numbers,
        bboxes=bboxes,
        confidences=confidences,
        classes=classes_array,
        num_detections_per_frame=num_detections_per_frame
    )
    
    print(f"  ✅ Saved: {output_path}")
    print(f"  Shape: {total_detections} detections across {num_frames} frames")
    
    return {
        'detections_file': str(output_path),
        'num_frames': num_frames,
        'total_detections': total_detections
    }


def main():
    parser = argparse.ArgumentParser(description='Stage 1: Detection')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Check if stage is enabled
    if not config['pipeline']['stages']['stage1_detect']:
        print("⏭️  Stage 1 is disabled in config")
        return
    
    # Run detection
    run_detection(config)
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
