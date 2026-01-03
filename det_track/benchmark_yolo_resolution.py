#!/usr/bin/env python3
"""
Benchmark: YOLO Processing Speed at 1080p vs 720p

Tests if input resolution affects YOLO inference speed.
YOLO internally scales to 640p, so we expect minimal difference.

Usage:
    python benchmark_yolo_resolution.py --config configs/pipeline_config.yaml
"""

import argparse
import yaml
import numpy as np
import cv2
import time
import re
import os
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO


def resolve_path_variables(config):
    """Recursively resolve ${variable} in config"""
    global_vars = config.get('global', {})
    
    def resolve_string_once(s, vars_dict):
        if not isinstance(s, str):
            return s
        return re.sub(
            r'\$\{(\w+)\}',
            lambda m: str(vars_dict.get(m.group(1), m.group(0))),
            s
        )
    
    max_iterations = 10
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
    
    result = resolve_recursive(config)
    result['global'] = global_vars
    return result


def load_config(config_path):
    """Load and resolve YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return resolve_path_variables(config)


def load_yolo_model(model_path, device='cuda'):
    """Load YOLO model"""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not found. Install with: pip install ultralytics")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    
    print(f"  Loading YOLO: {model_path}")
    model = YOLO(model_path, task="detect")
    
    if model_path.endswith('.pt'):
        model.to(device)
    
    return model


def benchmark_resolution(video_path, model, resolution, num_frames=1000, confidence=0.3):
    """
    Benchmark YOLO at specific resolution
    
    Returns: processing_fps
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if resolution == 'original':
        target_width, target_height = original_width, original_height
    elif resolution == '720p':
        target_width, target_height = 1280, 720
    else:
        target_width, target_height = resolution
    
    print(f"\n  {'='*70}")
    print(f"  📹 Resolution: {target_width}x{target_height}")
    print(f"  📊 Processing {num_frames} frames")
    print(f"  {'='*70}\n")
    
    frame_count = 0
    t_start = time.time()
    
    pbar = tqdm(total=num_frames, desc="Processing", mininterval=1.0)
    
    while frame_count < num_frames:
        ret, frame = cap.read()
        
        if not ret:
            print(f"\n  ⚠️  Video ended at frame {frame_count}")
            break
        
        # Resize if needed
        if (target_width, target_height) != (original_width, original_height):
            frame = cv2.resize(frame, (target_width, target_height))
        
        # Run YOLO
        _ = model(frame, conf=confidence, verbose=False)
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    
    t_end = time.time()
    total_time = t_end - t_start
    processing_fps = frame_count / total_time if total_time > 0 else 0
    
    cap.release()
    
    print(f"\n  ✅ Completed {frame_count} frames")
    print(f"  ⏱️  Time: {total_time:.2f}s")
    print(f"  ⚡ FPS: {processing_fps:.1f}")
    
    return processing_fps, frame_count, total_time


def main():
    parser = argparse.ArgumentParser(description='Benchmark YOLO at different resolutions')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    parser.add_argument('--frames', type=int, default=1000,
                       help='Number of frames to process (default: 1000)')
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                       help='YOLO model path (default: yolov8s.pt)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Get paths
    video_path = config['global']['video_dir'] + config['global']['video_file']
    model_path = args.model
    
    print(f"\n{'='*70}")
    print(f"🔍 YOLO RESOLUTION BENCHMARK")
    print(f"{'='*70}\n")
    
    print(f"📹 Video: {Path(video_path).name}")
    print(f"🤖 Model: {Path(model_path).name}")
    print(f"📊 Frames: {args.frames}\n")
    
    # Load model
    print(f"🛠️  Loading model...")
    model = load_yolo_model(model_path, device='cuda')
    print(f"  ✅ Model loaded\n")
    
    # Benchmark at 1080p
    print(f"{'='*70}")
    print(f"TEST 1: 1080p (ORIGINAL RESOLUTION)")
    print(f"{'='*70}")
    fps_1080, frames_1080, time_1080 = benchmark_resolution(video_path, model, 'original', args.frames)
    
    # Benchmark at 720p
    print(f"\n{'='*70}")
    print(f"TEST 2: 720p (DOWNSCALED)")
    print(f"{'='*70}")
    fps_720, frames_720, time_720 = benchmark_resolution(video_path, model, '720p', args.frames)
    
    # Comparison
    print(f"\n{'='*70}")
    print(f"📊 RESULTS COMPARISON")
    print(f"{'='*70}\n")
    
    print(f"1080p:")
    print(f"  Frames: {frames_1080}")
    print(f"  Time: {time_1080:.2f}s")
    print(f"  FPS: {fps_1080:.1f}")
    
    print(f"\n720p:")
    print(f"  Frames: {frames_720}")
    print(f"  Time: {time_720:.2f}s")
    print(f"  FPS: {fps_720:.1f}")
    
    speedup = fps_720 / fps_1080 if fps_1080 > 0 else 0
    time_savings = ((time_1080 - time_720) / time_1080 * 100) if time_1080 > 0 else 0
    
    print(f"\n{'─'*70}")
    print(f"720p is {speedup:.2f}x faster than 1080p")
    print(f"Time savings: {time_savings:.1f}%")
    print(f"{'='*70}\n")
    
    # Interpretation
    print(f"💡 INTERPRETATION:")
    if abs(speedup - 1.0) < 0.1:
        print(f"   ✅ YOLO speed is INDEPENDENT of input resolution")
        print(f"   → Both 1080p and 720p run at similar speed")
        print(f"   → YOLO internally normalizes to 640p anyway")
        print(f"   → Recommendation: Use ORIGINAL 1080p for storage/accuracy")
    else:
        print(f"   ⚡ YOLO speed DOES vary with resolution")
        print(f"   → 720p is significantly faster")
        print(f"   → Tradeoff: Speed vs accuracy")
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
