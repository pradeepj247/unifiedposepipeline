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
    if resolved_config['stage1_detect'].get('advanced', {}).get('verbose', False):
        print(f"🔍 Debug - repo_root before resolution: {config['global'].get('repo_root', 'NOT FOUND')}")
        model_path = resolved_config['stage1_detect']['detector']['model_path']
        print(f"🔍 Debug - model_path after resolution: {model_path}")
    
    return resolved_config


def auto_select_best_model(model_path, verbose=False):
    """
    Automatically select best available YOLO model:
    - Prefers yolov8n.engine (121 FPS) if exists and compatible
    - Falls back to specified model_path otherwise
    
    Returns: Path to best model to use
    """
    # If already specifying .engine, use it
    if model_path.endswith('.engine'):
        return model_path
    
    # Extract directory from model_path
    model_dir = os.path.dirname(model_path)
    yolov8n_engine = os.path.join(model_dir, 'yolov8n.engine')
    
    # Check if yolov8n.engine exists
    if not os.path.exists(yolov8n_engine):
        if verbose:
            print(f"  ℹ️  TensorRT engine not found: {yolov8n_engine}")
            print(f"  ℹ️  Using configured model: {model_path}")
        return model_path
    
    # Quick compatibility check: try to load engine
    try:
        if verbose:
            print(f"  🔍 Found TensorRT engine: {yolov8n_engine}")
            print(f"  🧪 Testing compatibility...")
        
        from ultralytics import YOLO
        import torch
        import numpy as np
        
        # Initialize CUDA
        if not torch.cuda.is_available():
            if verbose:
                print(f"  ⚠️  CUDA not available - using PyTorch model")
            return model_path
        
        torch.cuda.set_device(0)
        _ = torch.zeros(1, device="cuda")
        
        # Try to load engine (suppress output)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_model = YOLO(yolov8n_engine, task="detect")
            
            # Quick inference test
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = test_model.predict(source=dummy, conf=0.5, device=0, verbose=False)
        
        # Success! Engine is compatible
        if verbose:
            print(f"  ✅ TensorRT engine compatible - using yolov8n.engine (121 FPS)")
            print(f"  🚀 Performance boost: ~45% faster than PyTorch")
        else:
            print(f"  🚀 Auto-selected TensorRT engine: yolov8n.engine (121 FPS, 45% faster)")
        
        return yolov8n_engine
        
    except Exception as e:
        if verbose:
            print(f"  ⚠️  TensorRT engine incompatible: {str(e)[:100]}")
            print(f"  💡 Run: python det_track/debug/check_tensorrt_compatibility.py --auto-reexport")
            print(f"  ℹ️  Falling back to: {model_path}")
        else:
            print(f"  ⚠️  TensorRT engine incompatible - using PyTorch model")
            print(f"  💡 Fix: python det_track/debug/check_tensorrt_compatibility.py --auto-reexport")
        
        return model_path


def load_yolo_detector(model_path, device='cuda', verbose=False):
    """Load YOLO detector - supports both PyTorch (.pt) and TensorRT (.engine) models"""
    try:
        from ultralytics import YOLO
        import torch
    except ImportError:
        raise ImportError("ultralytics or torch not found. Install with: pip install ultralytics torch")
    
    # Suppress TensorRT verbose logging by redirecting stderr temporarily
    if model_path.endswith('.engine'):
        try:
            import tensorrt as trt
            import sys
            import os
            # Set TensorRT logger to ERROR level
            logger = trt.Logger(trt.Logger.ERROR)
            trt.init_libnvinfer_plugins(logger, "")
        except:
            pass  # TensorRT not available, will be handled by ultralytics
    
    # Auto-select best model (prefers TensorRT if available)
    original_model_path = model_path
    model_path = auto_select_best_model(model_path, verbose)
    
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
    
    # Suppress TensorRT verbose output during model loading
    import sys
    import os
    old_stderr = sys.stderr
    if not verbose and model_path.endswith('.engine'):
        sys.stderr = open(os.devnull, 'w')
    
    try:
        # Load model with task="detect" to avoid warning
        model = YOLO(model_path, task="detect")
    finally:
        if not verbose and model_path.endswith('.engine'):
            sys.stderr.close()
            sys.stderr = old_stderr
    
    if model_path.endswith('.engine'):
        print(f"  ✅ Detection model loaded")
    
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
    
    # Use canonical video from Stage 0 (normalized to 720p)
    # Read from stage0's output config to get exact path
    video_path = config['stage0_normalize']['output']['canonical_video_file']
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
    all_detection_indices = []  # NEW: Sequential detection IDs
    num_detections_per_frame = []
    
    # Crop storage for cache
    all_crops = []  # Will store all crops with metadata
    
    # Global detection counter for linking detections to crops
    detection_global_idx = 0
    
    # Process frames
    if verbose:
        print(f"⚡ Running detection and crop extraction...")
    t_loop_start = time.time()
    
    pbar = tqdm(total=num_frames, desc="  🔍 Detecting + extracting crops", mininterval=1.0)
    
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
            all_detection_indices.append(detection_global_idx)  # NEW: Store sequential ID
            
            # Extract crop immediately
            x1, y1, x2, y2 = detections[i, :4]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Clamp to frame boundaries
            x1 = max(0, min(x1, width))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height))
            y2 = max(0, min(y2, height))
            
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2].copy()
                
                # Resize maintaining aspect ratio (max dimension = 192px)
                h, w = crop.shape[:2]
                max_dim = 192
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    crop_resized = cv2.resize(crop, (new_w, new_h))
                else:
                    crop_resized = crop  # Don't upscale small crops
                
                all_crops.append({
                    'detection_idx': detection_global_idx,  # NEW: Link to detection
                    'frame_idx': frame_idx,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': detections[i, 4],
                    'class_id': classes[i],
                    'crop': crop_resized  # Store resized version (aspect ratio preserved)
                })
            
            # Increment global detection counter
            detection_global_idx += 1
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    t_loop_end = time.time()
    t_detect_extract_time = t_loop_end - t_loop_start  # t1: detection + crop extraction
    
    # Convert to numpy arrays
    frame_numbers = np.array(all_frame_numbers, dtype=np.int64)
    bboxes = np.array(all_bboxes, dtype=np.float32)
    confidences = np.array(all_confidences, dtype=np.float32)
    classes_array = np.array(all_classes, dtype=np.int64)
    detection_indices = np.array(all_detection_indices, dtype=np.int64)  # NEW
    num_detections_per_frame = np.array(num_detections_per_frame, dtype=np.int64)
    
    total_detections = len(frame_numbers)
    
    # Save detections NPZ and crops cache
    output_path = Path(detections_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    t_save_start = time.time()
    
    # Save detections NPZ
    np.savez_compressed(
        output_path,
        frame_numbers=frame_numbers,
        bboxes=bboxes,
        confidences=confidences,
        classes=classes_array,
        detection_indices=detection_indices,  # NEW: Links to crops_cache
        num_detections_per_frame=num_detections_per_frame
    )
    
    # Save crops cache
    import pickle
    crops_cache_path = output_path.parent / 'crops_cache.pkl'
    print(f"  💾 Saving crops_cache to: {crops_cache_path}")
    with open(crops_cache_path, 'wb') as f:
        pickle.dump(all_crops, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    t_save_end = time.time()
    t_save_time = t_save_end - t_save_start  # t2: saving time
    
    # Total time (t1 + t2)
    t_total = t_detect_extract_time + t_save_time
    
    # FPS calculation
    processing_fps = num_frames / t_total if t_total > 0 else 0
    
    crops_cache_size_mb = crops_cache_path.stat().st_size / (1024 * 1024)
    avg_detections_per_frame = total_detections / num_frames if num_frames > 0 else 0
    
    # Clean output: show only 4 key metrics
    print(f"\n  ✅ Detection + Crop Extraction: {t_detect_extract_time:.2f}s")
    print(f"  ✅ Saving (NPZ + crops cache): {t_save_time:.2f}s")
    print(f"  ✅ Total time: {t_total:.2f}s")
    print(f"  ✅ FPS: {processing_fps:.1f}")
    print(f"\n  📊 Total detections: {total_detections}")
    print(f"  📊 Avg detections/frame: {avg_detections_per_frame:.1f}")
    print(f"  📊 Total crops extracted: {len(all_crops)}")
    print(f"  💾 crops_cache.pkl: {crops_cache_size_mb:.1f} MB")
    print()

    # Write a sidecar JSON with timings
    try:
        sidecar = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'model_load_time': float(model_load_time),
            'detect_extract_time': float(t_detect_extract_time),  # t1
            'save_time': float(t_save_time),  # t2
            'total_time': float(t_total),  # t1 + t2
            'fps': float(processing_fps),
            'num_frames': int(num_frames),
            'total_detections': int(total_detections),
            'total_crops': len(all_crops),
            'crops_cache_size_mb': float(crops_cache_size_mb),
            'detections_file': str(output_path),
            'crops_cache_file': str(crops_cache_path)
        }

        sidecar_path = output_path.parent / (output_path.name + '.timings.json')
        with open(sidecar_path, 'w', encoding='utf-8') as sf:
            json.dump(sidecar, sf, indent=2)

        if verbose:
            print(f"  Wrote timings sidecar: {sidecar_path.name}")
    except Exception:
        # Non-fatal: sidecar write failure shouldn't stop the pipeline
        if verbose:
            print("  ⚠️  Failed to write timings sidecar")
    
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
