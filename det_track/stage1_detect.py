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
import json
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm


def resolve_path_variables(config):
    """Recursively resolve ${variable} in config"""
    global_vars = config.get('global', {})
    
    # First pass: resolve variables within global section itself
    # This handles cases like: models_dir: ${repo_root}/models
    def resolve_string_once(s, vars_dict):
        """Resolve ${variable} references using provided dict"""
        if not isinstance(s, str):
            return s
        return re.sub(
            r'\$\{(\w+)\}',
            lambda m: str(vars_dict.get(m.group(1), m.group(0))),
            s
        )
    
    # Resolve global variables iteratively until no more changes
    max_iterations = 10  # Prevent infinite loops
    for _ in range(max_iterations):
        resolved_globals = {}
        changed = False
        for key, value in global_vars.items():
            if isinstance(value, str):
                resolved = resolve_string_once(value, global_vars)
                resolved_globals[key] = resolved
                if resolved != value:
                    changed = True
            else:
                resolved_globals[key] = value
        
        global_vars = resolved_globals
        if not changed:
            break
    
    # Now use the fully resolved global_vars for the rest of the config
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
    
    # Replace the global section with resolved values
    result = resolve_recursive(config)
    result['global'] = global_vars
    
    return result


def load_config(config_path):
    """Load and resolve YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if global section exists
    if 'global' not in config:
        raise ValueError("Config file missing 'global' section")
    
    # Auto-extract current_video from video_file
    video_file = config.get('global', {}).get('video_file', '')
    if video_file:
        import os
        video_name = os.path.splitext(video_file)[0]
        config['global']['current_video'] = video_name
    
    # Resolve path variables
    resolved_config = resolve_path_variables(config)
    
    # Debug: Print model_path after resolution (verbose only)
    if resolved_config['stage1'].get('advanced', {}).get('verbose', False):
        print(f"🔍 Debug - repo_root before resolution: {config['global'].get('repo_root', 'NOT FOUND')}")
        model_path = resolved_config['stage1']['detector']['model_path']
        print(f"🔍 Debug - model_path after resolution: {model_path}")
    
    return resolved_config


def load_yolo_detector(model_path, device='cuda', verbose=False):
    """Load YOLO detector - supports both PyTorch (.pt) and TensorRT (.engine) models"""
    try:
        from ultralytics import YOLO
        import torch
    except ImportError:
        raise ImportError("ultralytics or torch not found. Install with: pip install ultralytics torch")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    
    # CRITICAL: Initialize CUDA via PyTorch BEFORE any TensorRT operations
    # TensorRT cannot bootstrap CUDA by itself in Python
    # This MUST happen before YOLO() constructor for .engine files
    if model_path.endswith('.engine'):
        print(f"  🔧 Initializing CUDA via PyTorch (required for TensorRT)...")
        
        # Force CUDA initialization FIRST
        assert torch.cuda.is_available(), "CUDA not available"
        torch.cuda.set_device(0)
        _ = torch.zeros(1, device="cuda")  # Dummy tensor to ensure CUDA is fully initialized
        
        print(f"  ✅ CUDA initialized: {torch.cuda.get_device_name(0)}")
    
    if verbose:
        model_type = "TensorRT engine" if model_path.endswith('.engine') else "PyTorch model"
        print(f"  ✅ Loading {model_type}: {model_path}")
    
    # Load model with task="detect" to avoid warning
    model = YOLO(model_path, task="detect")
    
    # Only call .to(device) for PyTorch models
    # TensorRT engines don't support .to() - device is specified in predict()
    if model_path.endswith('.pt'):
        model.to(device)
        return {
            'type': 'pytorch',
            'model': model,
            'device': device
        }
    else:
        # TensorRT engine - device passed to predict() method
        # Check if TensorRT is available
        try:
            import tensorrt as trt
            trt_available = True
        except ImportError:
            trt_available = False
            print(f"  ⚠️  TensorRT not installed. Ultralytics will attempt auto-install.")
            print(f"  ⚠️  After installation, you MUST restart the runtime.")
            print(f"  💡 Alternatively, use PyTorch model (.pt) which works without TensorRT.")
        
        # Warm up the engine with a dummy prediction
        if verbose:
            print(f"  🔥 Warming up TensorRT engine...")
        
        import numpy as np
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        try:
            _ = model.predict(source=dummy_frame, conf=0.5, device=0, verbose=False)
            if verbose:
                print(f"  ✅ TensorRT engine ready")
        except ModuleNotFoundError as e:
            if 'tensorrt' in str(e).lower():
                print(f"\n❌ TensorRT module not found!")
                print(f"   Solution 1: Install TensorRT and restart runtime")
                print(f"      !pip install tensorrt")
                print(f"      Then: Runtime → Restart runtime")
                print(f"   Solution 2: Use PyTorch model instead")
                print(f"      Edit config: model_path: yolov8s.pt")
                raise RuntimeError("TensorRT not available. See solutions above.")
            else:
                raise
        except Exception as e:
            print(f"  ⚠️  TensorRT warmup warning: {e}")
        
        return {
            'type': 'tensorrt',
            'model': model,
            'device': 0 if device == 'cuda' else device  # TensorRT expects device ID
        }


def detect_frame(model, frame, confidence=0.3, detect_only_humans=True):
    """
    Run YOLO detection on a single frame
    
    Args:
        model: Model dict returned by load_yolo_detector or YOLO model object (for backward compatibility)
    
    Returns:
        detections: (N, 5) array of [x1, y1, x2, y2, confidence]
        classes: (N,) array of class IDs
    """
    # Handle both new dict format and old direct model format
    if isinstance(model, dict):
        model_type = model.get('type', 'pytorch')
        yolo_model = model['model']
        device = model.get('device', 'cuda')
        
        # Use different inference methods for PyTorch vs TensorRT
        if model_type == 'pytorch':
            # PyTorch: can call model directly
            results = yolo_model(frame, conf=confidence, verbose=False)
        elif model_type == 'tensorrt':
            # TensorRT: must use .predict() with device parameter
            results = yolo_model.predict(
                source=frame,
                conf=confidence,
                device=device,
                verbose=False
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    else:
        # Backward compatibility: assume it's a YOLO model object
        yolo_model = model
        results = yolo_model(frame, conf=confidence, verbose=False)
    
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
    input_config = stage_config['input']
    output_config = stage_config['output']
    
    model_path = detector_config['model_path']
    confidence = detector_config['confidence']
    device = detector_config['device']
    detect_only_humans = detector_config['detect_only_humans']
    
    # Get video path from global config (single source of truth)
    video_path = config['global']['video_dir'] + config['global']['video_file']
    max_frames = input_config['max_frames']
    
    detections_file = output_config['detections_file']
    
    # Print header
    print(f"\n{'='*70}")
    print(f"📍 STAGE 1: DETECTION")
    print(f"{'='*70}\n")
    
    # Load detector (measure model load time)
    t_model_load_start = time.time()
    if verbose:
        print(f"🛠️  Loading detector...")
    detector = load_yolo_detector(model_path, device, verbose)
    t_model_load_end = time.time()
    model_load_time = t_model_load_end - t_model_load_start
    print(f"  ✅ Detection model loaded\n")
    
    # Open video (measure open time)
    t_video_open_start = time.time()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    t_video_open_end = time.time()
    video_open_time = t_video_open_end - t_video_open_start
    
    # Use original resolution (downscaling adds 19% overhead with no benefit)
    proc_width, proc_height = width, height
    num_frames = min(max_frames, total_frames) if max_frames > 0 else total_frames
    
    print(f"     Opening video: {video_path}")
    print(f"     Resolution: {width}x{height} @ {fps:.2f} fps")
    print(f"     Total frames: {total_frames}\n")
    
    # Storage
    all_frame_numbers = []
    all_bboxes = []
    all_confidences = []
    all_classes = []
    num_detections_per_frame = []
    
    # Process frames
    if verbose:
        print(f"⚡ Running detection...")
    t_loop_start = time.time()
    
    pbar = tqdm(total=num_frames, desc="  🔍 Detecting", mininterval=1.0)
    
    frame_idx = 0
    while frame_idx < num_frames:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Detect (all coordinates in original 1920x1080 resolution)
        detections, classes = detect_frame(detector, frame, confidence, detect_only_humans)
        
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
    
    t_loop_end = time.time()
    t_loop_total = t_loop_end - t_loop_start
    processing_fps = num_frames / t_loop_total if t_loop_total > 0 else 0
    
    # Convert to numpy arrays
    frame_numbers = np.array(all_frame_numbers, dtype=np.int64)
    bboxes = np.array(all_bboxes, dtype=np.float32)
    confidences = np.array(all_confidences, dtype=np.float32)
    classes_array = np.array(all_classes, dtype=np.int64)
    num_detections_per_frame = np.array(num_detections_per_frame, dtype=np.int64)
    
    # Summary
    total_detections = len(frame_numbers)
    avg_detections_per_frame = total_detections / num_frames if num_frames > 0 else 0
    
    print(f"     Detection complete!")
    print(f"     Total detections: {total_detections}")
    print(f"     Avg detections/frame: {avg_detections_per_frame:.1f}")
    print(f"     Processing FPS: {processing_fps:.1f}")
    print(f"     Detection time: {t_loop_total:.2f}s\n")
    
    # Save NPZ
    output_path = Path(detections_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    t_npz_start = time.time()
    np.savez_compressed(
        output_path,
        frame_numbers=frame_numbers,
        bboxes=bboxes,
        confidences=confidences,
        classes=classes_array,
        num_detections_per_frame=num_detections_per_frame
    )
    t_npz_end = time.time()
    npz_save_time = t_npz_end - t_npz_start

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"     Saved: {output_path.name}")
    if verbose:
        print(f"     Size: {file_size_mb:.1f} MB")
        print(f"     Shape: {total_detections} detections across {num_frames} frames")
    print(f"     NPZ save time: {npz_save_time:.2f}s")

    # Print full timing breakdown
    sum_parts = model_load_time + video_open_time + t_loop_total + npz_save_time

    print(f"     Breakdown:")
    print(f"       model load: {model_load_time:.2f}s")
    print(f"       video open: {video_open_time:.2f}s")
    print(f"       detection loop: {t_loop_total:.2f}s")
    print(f"       npz save: {npz_save_time:.2f}s")
    print(f"     Sum of parts: {sum_parts:.2f}s")
    print()

    # Write a sidecar JSON with fine-grained timings for the orchestrator to read
    try:
        sidecar = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'model_load_time': float(model_load_time),
            'video_open_time': float(video_open_time),
            'detect_loop_time': float(t_loop_total),
            'npz_save_time': float(npz_save_time),
            'sum_parts': float(sum_parts),
            'num_frames': int(num_frames),
            'total_detections': int(total_detections),
            'detections_file': str(output_path)
        }

        sidecar_path = output_path.parent / (output_path.name + '.timings.json')
        with open(sidecar_path, 'w', encoding='utf-8') as sf:
            json.dump(sidecar, sf, indent=2)

        if verbose:
            print(f"     Wrote timings sidecar: {sidecar_path.name}")
    except Exception:
        # Non-fatal: sidecar write failure shouldn't stop the pipeline
        if verbose:
            print("     ⚠️  Failed to write timings sidecar")
    
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
    if not config['pipeline']['stages']['stage1']:
        print("⏭️  Stage 1 is disabled in config")
        return
    
    # Run detection
    run_detection(config)

    # Print trailing separator only in verbose mode to avoid extra delimiter
    verbose = config.get('stage1', {}).get('advanced', {}).get('verbose', False)
    if verbose:
        print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
