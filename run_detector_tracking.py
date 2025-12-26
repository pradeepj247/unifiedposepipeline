#!/usr/bin/env python3
"""
Unified Detector and Tracker Script with Visualization

This script handles object detection and tracking for the pose estimation pipeline.
Currently supports:
- Detection: YOLOv8, RTMDet
- Tracking: BoT-SORT, DeepOCSORT, ByteTrack, StrongSORT, OCSORT, HybridSORT, BoostTrack
- ReID: OSNet-based re-identification (optional)
- Visualization: Annotated video with track IDs

Usage:
    python run_detector_tracking.py --config configs/detector.yaml
    
    # For benchmark with visualization:
    python run_detector_tracking.py --config configs/detector_tracking_benchmark.yaml

Output:
    detections.npz with keys:
        - frame_numbers: (N,) array of frame indices
        - bboxes: (N, 4) array of [x1, y1, x2, y2] (integer coordinates)
    
    Optional visualization video (if enabled in config):
        - Annotated video with colored bboxes and track IDs
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


def load_yolo_detector(model_path, device='cuda', confidence=0.3, verbose=False):
    """
    Load YOLOv8 detector
    
    Args:
        model_path: Path to YOLO model (.pt file)
        device: Device to run on ('cuda' or 'cpu')
        confidence: Confidence threshold
        verbose: Print loading messages
    
    Returns:
        YOLO model object
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics package not found. Install with: pip install ultralytics")
    
    import os
    
    # Multi-path resolution (similar to ReID model logic)
    # Check multiple locations for YOLO model
    candidate_paths = [
        model_path,  # As-is path (e.g., models/yolo/yolov8s.pt)
        os.path.join('..', model_path),  # Parent directory (e.g., ../models/yolo/yolov8s.pt)
        os.path.join('/content', model_path.lstrip('models/'))  # Absolute Colab path (e.g., /content/yolo/yolov8s.pt)
    ]
    
    resolved_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            resolved_path = path
            break
    
    if resolved_path is None:
        print(f"   ❌ YOLO model not found at any of these locations:")
        for path in candidate_paths:
            print(f"      - {path}")
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    
    if verbose:
        print(f"   ✅ YOLO model found: {resolved_path}")
        print(f"Loading YOLO model from: {resolved_path}")
    
    model = YOLO(resolved_path)
    model.to(device)
    
    return model


def load_tracker(tracker_name, reid_config, device='cuda', half=False, verbose=False):
    """
    Load BoxMOT tracker
    
    Args:
        tracker_name: Name of tracker (botsort, deepocsort, bytetrack, strongsort, ocsort, hybridsort, boosttrack)
        reid_config: ReID configuration dict
        device: Device to run on
        half: Use FP16 half precision
        verbose: Print loading messages
    
    Returns:
        Tracker object
    """
    try:
        # Import all 7 BoxMOT trackers (v16.0.4)
        # Note: Class names are case-sensitive! DeepOcSort, not DeepOCSORT!
        import boxmot
        from boxmot import (
            BotSort,      # Motion + Appearance (best accuracy)
            ByteTrack,    # Motion-only (fastest)
            BoostTrack,   # Motion + Appearance (best IDF1)
            DeepOcSort,   # Deep learning tracker (note: DeepOcSort!)
            HybridSort,   # Hybrid approach (note: HybridSort!)
            OcSort,       # Observation-centric (note: OcSort!)
            StrongSort    # Strong baseline (note: StrongSort!)
        )
    except ImportError as e:
        raise ImportError(f"boxmot package not found: {e}. Install with: pip install boxmot")
    
    # Tracker mapping (lowercase config names to actual classes)
    tracker_map = {
        'botsort': BotSort,
        'deepocsort': DeepOcSort,    # Fixed: DeepOcSort (not DeepOCSORT)
        'bytetrack': ByteTrack,
        'strongsort': StrongSort,    # Fixed: StrongSort (not StrongSORT)
        'ocsort': OcSort,            # Fixed: OcSort (not OCSORT)
        'hybridsort': HybridSort,    # Fixed: HybridSort (not HybridSORT)
        'boosttrack': BoostTrack
    }
    
    tracker_class = tracker_map.get(tracker_name.lower())
    if not tracker_class:
        available = list(tracker_map.keys())
        raise ValueError(f"Unknown tracker: '{tracker_name}'. Available: {available}")
    
    if verbose:
        print(f"Loading {tracker_name.upper()} tracker...")
    
    # Check if ReID is enabled
    if reid_config.get('enabled', False):
        reid_weights_str = reid_config['weights_path']
        reid_weights = Path(reid_weights_str)
        
        # Try multiple path resolution strategies (same as YOLO)
        # 1. Try as-is (absolute path)
        # 2. Try relative to current directory
        # 3. Try relative to parent directory (where models/ actually is)
        if not reid_weights.exists():
            # Try relative to parent directory
            reid_weights_parent = Path('..') / reid_weights_str
            if reid_weights_parent.exists():
                reid_weights = reid_weights_parent
            # Try as absolute path with /content/models
            elif Path('/content') / reid_weights_str:
                reid_weights_abs = Path('/content') / reid_weights_str
                if reid_weights_abs.exists():
                    reid_weights = reid_weights_abs
        
        if not reid_weights.exists():
            if verbose:
                print(f"   ⚠️  ReID weights not found: {reid_weights}")
                print(f"   Tried: {reid_weights_str}, ../{reid_weights_str}, /content/{reid_weights_str}")
                print(f"   Using motion-only tracking (no appearance features)")
            tracker = tracker_class(device=device, half=half)
        else:
            if verbose:
                print(f"   ✅ ReID weights found: {reid_weights}")
            tracker = tracker_class(
                reid_weights=reid_weights,
                device=device,
                half=half
            )
    else:
        # Motion-only tracking
        if verbose:
            print(f"   Using motion-only tracking (ReID disabled)")
        tracker = tracker_class(device=device, half=half)
    
    return tracker


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
        classes: (N,) array of class IDs
    """
    # Run inference
    results = model(frame, conf=confidence, verbose=False)
    
    # Extract detections
    detections = []
    classes = []
    
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
            classes.append(cls_id)
    
    if len(detections) == 0:
        return np.array([]), np.array([])
    
    return np.array(detections), np.array(classes)


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


def select_largest_tracked_bbox(tracked_bboxes):
    """
    Select the largest tracked bounding box by area
    
    Args:
        tracked_bboxes: (N, 8) array of [x1, y1, x2, y2, track_id, conf, cls, det_ind] from tracker
    
    Returns:
        bbox: (4,) array of [x1, y1, x2, y2] (largest bbox, integers), or empty array if no tracks
    """
    if len(tracked_bboxes) == 0:
        return np.array([])
    
    # Calculate areas
    areas = (tracked_bboxes[:, 2] - tracked_bboxes[:, 0]) * (tracked_bboxes[:, 3] - tracked_bboxes[:, 1])
    
    # Get index of largest area
    largest_idx = np.argmax(areas)
    
    # Return only coordinates [x1, y1, x2, y2]
    return tracked_bboxes[largest_idx, :4]


def draw_tracks(frame, tracked_bboxes, show_conf=False):
    """
    Draw bounding boxes with track IDs on frame
    
    Args:
        frame: Input frame (BGR)
        tracked_bboxes: (N, 8) array of [x1, y1, x2, y2, track_id, conf, cls, det_ind]
        show_conf: Show confidence scores
    
    Returns:
        frame: Annotated frame
    """
    frame_out = frame.copy()
    
    if len(tracked_bboxes) == 0:
        return frame_out
    
    # Define colors for different track IDs (cycling through palette)
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 255),  # Purple
        (255, 128, 0),  # Orange
    ]
    
    for track in tracked_bboxes:
        x1, y1, x2, y2, track_id, conf, cls, _ = track
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = int(track_id)
        
        # Select color based on track ID
        color = colors[track_id % len(colors)]
        
        # Draw bbox
        cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        if show_conf:
            label = f"ID:{track_id} ({conf:.2f})"
        else:
            label = f"ID:{track_id}"
        
        # Draw label inside bbox upper-right corner (prevents cutoff at frame edges)
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Position text inside bbox: right-aligned, 5 pixels from top-right corner
        text_x = x2 - label_w - 5  # 5 pixels from right edge
        text_y = y1 + label_h + 5  # 5 pixels from top edge
        
        # Draw label background (semi-transparent rectangle)
        cv2.rectangle(frame_out, (text_x - 2, y1 + 2), (x2 - 2, text_y + 2), color, -1)
        
        # Draw label text
        cv2.putText(frame_out, label, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame_out


def process_video(config):
    """
    Process video with detection and optional tracking
    
    Args:
        config: Configuration dictionary
    
    Returns:
        detections_data: Dictionary with frame_numbers, bboxes
    """
    # Extract config parameters
    import os
    video_dir = config['input'].get('video_path', 'demo_data/videos/')
    video_filename = config['input'].get('video_filename', 'dance.mp4')
    video_path = os.path.join(video_dir, video_filename)
    max_frames = config['input']['max_frames']
    
    detector_type = config['detector']['type']
    model_path = config['detector']['model_path']
    confidence = config['detector']['confidence']
    device = config['detector']['device']
    detect_only_humans = config['detector']['detect_only_humans']
    
    tracking_enabled = config['tracking']['enabled']
    tracker_name = config['tracking'].get('tracker', 'botsort')
    reid_config = config['tracking'].get('reid', {'enabled': False})
    largest_bbox_only = config['tracking']['largest_bbox_only']
    
    verbose = config['advanced']['verbose']
    
    # Load detector
    if detector_type.lower() == 'yolo':
        detector = load_yolo_detector(model_path, device, confidence, verbose=verbose)
    else:
        raise NotImplementedError(f"Detector type '{detector_type}' not yet supported")
    
    # Load tracker if enabled
    tracker = None
    if tracking_enabled:
        try:
            tracker = load_tracker(tracker_name, reid_config, device=device, half=False, verbose=verbose)
            if verbose:
                print(f"   ✅ {tracker_name.upper()} tracker loaded successfully")
        except Exception as e:
            print(f"   ❌ Failed to load tracker: {e}")
            print(f"   Falling back to detection-only mode")
            tracking_enabled = False
    
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
    
    # Print video info
    if verbose:
        print(f"\n{'='*70}")
        print(f"Video: {video_path}")
        print(f"Resolution: {width}x{height} @ {fps:.2f} fps")
        print(f"Total frames: {total_frames}")
        print(f"Processing: {num_frames} frames")
        print(f"Detector: {detector_type.upper()}")
        if tracking_enabled:
            print(f"Tracking: {tracker_name.upper()} (ReID: {'ON' if reid_config.get('enabled') else 'OFF'})")
        else:
            print(f"Tracking: Disabled (largest detection per frame)")
        print(f"{'='*70}\n")
    else:
        # Simple non-verbose header
        print(f"Video: {video_path}")
        print(f"Resolution: {width}x{height} @ {fps:.2f} fps")
        print(f"Total frames: {total_frames}")
        print(f"Processing: {num_frames} frames")
        print(f"Detector: {detector_type.upper()}")
        if tracking_enabled:
            print(f"Tracking: {tracker_name.upper()} (ReID: {'ON' if reid_config.get('enabled') else 'OFF'})")
    
    # Storage for detections and tracking data
    frame_numbers = []
    bboxes_list = []
    all_tracks_per_frame = []  # Store all tracks for visualization
    
    # Benchmarking
    detection_times = []
    tracking_times = []
    
    # Start timing
    t_start = time.time()
    
    # Process frames (always show progress bar - it's critical info)
    pbar = tqdm(total=num_frames, desc="Processing")
    
    frame_idx = 0
    
    while frame_idx < num_frames:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Run detection (timed)
        t_det_start = time.time()
        if detector_type.lower() == 'yolo':
            detections, classes = detect_yolo(detector, frame, confidence, detect_only_humans)
        else:
            raise NotImplementedError(f"Detector type '{detector_type}' not supported")
        t_det_end = time.time()
        detection_times.append(t_det_end - t_det_start)
        
        # Handle tracking vs largest bbox selection
        if tracking_enabled and tracker is not None:
            # Prepare detections for tracker
            # BoxMOT expects: (N, 6) array of [x1, y1, x2, y2, conf, cls]
            if len(detections) > 0:
                dets_for_tracker = np.column_stack([detections, classes])
            else:
                dets_for_tracker = np.empty((0, 6))
            
            # Update tracker (timed)
            # Returns: (N, 8) array of [x1, y1, x2, y2, track_id, conf, cls, det_ind]
            t_track_start = time.time()
            try:
                tracked_bboxes = tracker.update(dets_for_tracker, frame)
                t_track_end = time.time()
                tracking_times.append(t_track_end - t_track_start)
                
                # Store all tracks for this frame
                all_tracks_per_frame.append(tracked_bboxes.copy() if len(tracked_bboxes) > 0 else np.empty((0, 8)))
                
                # Select largest tracked bbox
                frame_numbers.append(frame_idx)
                if len(tracked_bboxes) > 0:
                    largest_bbox = select_largest_tracked_bbox(tracked_bboxes)
                    bboxes_list.append(largest_bbox)
                else:
                    # No tracks - store empty bbox
                    bboxes_list.append([0, 0, 0, 0])
            except Exception as e:
                # Tracker error - fall back to empty bbox
                print(f"\n⚠️  Tracker error at frame {frame_idx}: {e}")
                tracking_times.append(0)
                all_tracks_per_frame.append(np.empty((0, 8)))
                frame_numbers.append(frame_idx)
                bboxes_list.append([0, 0, 0, 0])
        else:
            # Detection-only mode: Select largest bbox
            tracking_times.append(0)  # No tracking time
            all_tracks_per_frame.append(np.empty((0, 8)))  # No tracks
            frame_numbers.append(frame_idx)
            if len(detections) > 0:
                largest_bbox = select_largest_bbox(detections)
                bboxes_list.append(largest_bbox)
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
    frame_numbers = np.array(frame_numbers, dtype=np.int64)
    bboxes = np.array(bboxes_list, dtype=np.int64)
    
    detections_data = {
        'frame_numbers': frame_numbers,
        'bboxes': bboxes,
        'all_tracks': all_tracks_per_frame  # For visualization
    }
    
    # Always show processing summary (critical info even in silent mode)
    # Count valid detections (bboxes where x2 > 0)
    valid_detections = np.sum(bboxes[:, 2] > 0)
    
    # Calculate average times
    avg_det_time = np.mean(detection_times) if detection_times else 0
    avg_track_time = np.mean(tracking_times) if tracking_times else 0
    avg_total_time = avg_det_time + avg_track_time
    
    # Calculate component FPS
    det_fps = 1.0 / avg_det_time if avg_det_time > 0 else 0
    track_fps = 1.0 / avg_track_time if avg_track_time > 0 else 0
    total_fps = 1.0 / avg_total_time if avg_total_time > 0 else 0
    
    print(f"\n✓ Processing complete!")
    print(f"  Frames processed: {len(frame_numbers)}")
    
    if verbose:
        print(f"  Frames with valid output: {valid_detections}")
        print(f"  Success rate: {valid_detections / len(frame_numbers) * 100:.1f}%")
    
    print(f"\n⏱️  Performance:")
    if verbose:
        # Detailed breakdown in verbose mode
        print(f"  Detection FPS: {det_fps:.1f} ({avg_det_time*1000:.1f}ms/frame)")
        if tracking_enabled:
            print(f"  Tracking FPS:  {track_fps:.1f} ({avg_track_time*1000:.1f}ms/frame)")
            print(f"  Combined FPS:  {total_fps:.1f} ({avg_total_time*1000:.1f}ms/frame)")
        print(f"  Overall FPS:   {processing_fps:.1f} (including I/O)")
    else:
        # Simple summary in silent mode
        print(f"  Overall FPS:   {processing_fps:.1f}")
    print(f"  Time taken: {total_time:.2f}s")
    
    # Track ID statistics (always show if tracking enabled - critical info)
    if tracking_enabled and len(all_tracks_per_frame) > 0:
        # Build per-ID statistics: frame appearances, start/end frames
        track_id_stats = {}  # {track_id: {'frames': [frame_nums], 'start': int, 'end': int, 'count': int}}
        
        for frame_num, tracks in enumerate(all_tracks_per_frame):
            if len(tracks) > 0:
                for track in tracks:
                    track_id = int(track[4])
                    if track_id not in track_id_stats:
                        track_id_stats[track_id] = {
                            'frames': [],
                            'start': frame_num,
                            'end': frame_num,
                            'count': 0
                        }
                    track_id_stats[track_id]['frames'].append(frame_num)
                    track_id_stats[track_id]['end'] = frame_num
                    track_id_stats[track_id]['count'] += 1
        
        # Always show basic tracking statistics
        print(f"\n📊 Tracking Statistics:")
        print(f"  Unique track IDs: {len(track_id_stats)}")
        
        # Detailed table only in verbose mode
        if verbose:
            print(f"  Track IDs seen: {sorted(track_id_stats.keys())}")
            
            if len(track_id_stats) > 0:
                print(f"\n  {'Person ID':<12} {'# Frames':<12} {'Start Frame':<12} {'End Frame':<12}")
                print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
                for track_id in sorted(track_id_stats.keys()):
                    stats = track_id_stats[track_id]
                    print(f"  {track_id:<12} {stats['count']:<12} {stats['start']:<12} {stats['end']:<12}")
    
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
    
    # Always show save confirmation (critical info)
    print(f"\n✓ Saved detections to: {output_path}")
    if verbose:
        print(f"  Shape: bboxes={detections_data['bboxes'].shape}")
        print(f"  Format: frame_numbers (int64), bboxes (int64, 4 values per frame)")


def save_raw_detections(detections_data, output_path, verbose=True):
    """
    Save raw multi-person tracking detections to NPZ file
    
    Args:
        detections_data: Dictionary with all_tracks data
        output_path: Path to output NPZ file
        verbose: Print save confirmation
    """
    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract all tracks from all_tracks_per_frame
    all_tracks = detections_data.get('all_tracks', [])
    
    if len(all_tracks) == 0:
        print(f"\n⚠️  No tracking data found. Skipping raw_detections.npz")
        return
    
    # Build multi-person arrays
    frame_numbers_list = []
    bboxes_list = []
    track_ids_list = []
    scores_list = []
    
    for frame_idx, tracks in enumerate(all_tracks):
        if len(tracks) > 0:
            for track in tracks:
                # track format: [x1, y1, x2, y2, track_id, conf, cls, det_ind]
                frame_numbers_list.append(frame_idx)
                bboxes_list.append(track[:4])  # [x1, y1, x2, y2]
                track_ids_list.append(int(track[4]))  # track_id
                scores_list.append(track[5])  # confidence
    
    # Convert to numpy arrays
    frame_numbers_raw = np.array(frame_numbers_list, dtype=np.int64)
    bboxes_raw = np.array(bboxes_list, dtype=np.int64)
    track_ids_raw = np.array(track_ids_list, dtype=np.int64)
    scores_raw = np.array(scores_list, dtype=np.float32)
    
    # Save NPZ with all tracking data
    np.savez_compressed(
        output_path,
        frame_numbers=frame_numbers_raw,
        bboxes=bboxes_raw,
        track_ids=track_ids_raw,
        scores=scores_raw
    )
    
    # Always show save confirmation with detection count (critical info)
    print(f"\n✓ Saved raw detections to: {output_path}")
    print(f"  Total detections: {len(frame_numbers_raw)}")
    if verbose:
        print(f"  Format: frame_numbers, bboxes, track_ids, scores")


def save_visualization(video_path, detections_data, output_path, max_frames=0, verbose=True):
    """
    Save video with tracking visualization (bboxes + track IDs)
    
    Args:
        video_path: Path to input video
        detections_data: Dictionary with all_tracks data
        output_path: Path to output video
        max_frames: Maximum frames to process (0 = all)
        verbose: Print progress
    """
    if 'all_tracks' not in detections_data:
        print("⚠️  No tracking data available for visualization")
        return
    
    all_tracks = detections_data['all_tracks']
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames > 0:
        num_frames = min(max_frames, total_frames)
    else:
        num_frames = total_frames
    
    # Reduce output resolution to 720p to save time and space
    # Maintain aspect ratio
    output_height = 720
    output_width = int(width * (output_height / height))
    # Ensure width is even (required for some codecs)
    if output_width % 2 != 0:
        output_width += 1
    
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create video writer (same FPS as input)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Creating tracking visualization...")
        print(f"Input: {video_path}")
        print(f"Output: {output_path}")
        print(f"Frames: {num_frames}")
        print(f"{'='*70}\n")
    else:
        # Silent mode: just show we're writing output
        print(f"\nWriting visualization video...")
    
    # Process frames (always show progress bar - critical info)
    pbar = tqdm(total=num_frames, desc="Writing output")
    
    frame_idx = 0
    while frame_idx < num_frames:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Get tracks for this frame
        if frame_idx < len(all_tracks):
            tracks = all_tracks[frame_idx]
            
            # Draw tracks
            frame_vis = draw_tracks(frame, tracks, show_conf=False)
            
            # Add frame info
            info_text = f"Frame: {frame_idx}  Tracks: {len(tracks)}"
            cv2.putText(frame_vis, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            frame_vis = frame
        
        # Resize to 720p before writing
        frame_vis = cv2.resize(frame_vis, (output_width, output_height))
        
        # Write frame
        out.write(frame_vis)
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    # Always show save confirmation (critical info)
    print(f"\n✓ Saved visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Unified Detector and Tracker for Pose Estimation Pipeline (EXPERIMENTAL)'
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
    
    # Determine if we should save detections.npz or raw_detections.npz
    tracking_enabled = config['tracking']['enabled']
    largest_bbox_only = config['tracking']['largest_bbox_only']
    
    # Only save detections.npz if:
    # 1. Tracking is disabled, OR
    # 2. Tracking is enabled BUT largest_bbox_only is true
    if not tracking_enabled or (tracking_enabled and largest_bbox_only):
        # Save detections (single bbox per frame - largest)
        output_path = config['output']['detections_file']
        save_detections(detections_data, output_path, verbose=config['advanced']['verbose'])
    
    # Save raw detections (all tracked persons) if tracking is enabled and largest_bbox_only is false
    if tracking_enabled and not largest_bbox_only:
        if 'raw_detections_file' in config['output']:
            raw_output_path = config['output']['raw_detections_file']
            save_raw_detections(detections_data, raw_output_path, verbose=config['advanced']['verbose'])
    
    # Save visualization if enabled
    if config['output'].get('save_visualization', False):
        import os
        # Reconstruct full video path
        video_dir = config['input'].get('video_path', 'demo_data/videos/')
        video_filename = config['input'].get('video_filename', 'dance.mp4')
        video_path = os.path.join(video_dir, video_filename)
        
        # Auto-generate output filename: <input_name>_tracking_reid.mp4
        vis_dir = config['output'].get('visualization_path', 'demo_data/outputs/')
        video_basename = os.path.splitext(video_filename)[0]  # Remove extension
        vis_filename = f"{video_basename}_tracking_reid.mp4"
        vis_path = os.path.join(vis_dir, vis_filename)
        
        max_frames = config['input']['max_frames']
        save_visualization(
            video_path, 
            detections_data, 
            vis_path, 
            max_frames=max_frames,
            verbose=config['advanced']['verbose']
        )
    
    print("\n✓ Detection/Tracking pipeline complete!\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
