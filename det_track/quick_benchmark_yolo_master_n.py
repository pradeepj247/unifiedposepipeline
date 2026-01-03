#!/usr/bin/env python3
"""
Benchmark: YOLO Master N speed test

Uses native ultralytics YOLO API to load yolo_master_n.pt and test inference speed.

Usage:
    python quick_benchmark_yolo_master_n.py --config configs/pipeline_config.yaml --frames 1000
"""

import argparse
import yaml
import cv2
import time
import re
import os
import sys
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO


def resolve_path_variables(config):
    """Recursively resolve ${variable} in config"""
    global_vars = config.get('global', {})
    
    # First pass: resolve variables within global section itself
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
    
    result = resolve_recursive(config)
    result['global'] = global_vars
    
    return result


def load_config(config_path):
    """Load and resolve YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'global' not in config:
        raise ValueError("Config file missing 'global' section")
    
    return resolve_path_variables(config)


def benchmark_yolo_master_n(model_path, video_path, num_frames=1000, verbose=True):
    """Benchmark YOLO Master N using native ultralytics API"""
    
    # Print header
    print(f"\n{'='*70}")
    print(f"⚡ YOLO MASTER N SPEED TEST")
    print(f"{'='*70}\n")
    
    print(f"📹 Video: {Path(video_path).name}")
    print(f"🤖 Model: YOLO Master N")
    print(f"📊 Frames: {num_frames}")
    
    # Load model using native ultralytics API
    print(f"\n🛠️  Loading YOLO Master N...")
    print(f"   Path: {model_path}")
    
    try:
        model = YOLO(model_path)
        print(f"  ✅ Model loaded")
    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
        return None
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"  ❌ Could not open video: {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  Resolution: {width}x{height}")
    
    # Limit frames
    num_frames = min(num_frames, total_frames)
    
    print(f"\n⚡ Processing {num_frames} frames...")
    
    # Process frames
    t_start = time.time()
    pbar = tqdm(total=num_frames, desc="Inference", mininterval=1.0)
    
    frame_count = 0
    while frame_count < num_frames:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Run inference using native ultralytics API
        results = model(frame, verbose=False, conf=0.45)
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    t_end = time.time()
    total_time = t_end - t_start
    fps_result = frame_count / total_time if total_time > 0 else 0
    
    # Print results
    print(f"\n{'='*70}")
    print(f"✅ RESULTS")
    print(f"{'='*70}\n")
    
    print(f"Frames processed: {frame_count}")
    print(f"Time: {total_time:.2f}s")
    print(f"FPS: {fps_result:.1f}\n")
    
    print(f"{'─'*70}")
    print(f"COMPARISON with YOLOv8s:")
    print(f"{'─'*70}\n")
    
    print(f"YOLOv8s @ 1080p: 55.4 FPS (18.06s for 1000 frames)")
    print(f"YOLO Master N @ 1080p: {fps_result:.1f} FPS ({total_time:.2f}s for {frame_count} frames)")
    
    if fps_result > 55.4:
        improvement = ((fps_result - 55.4) / 55.4) * 100
        print(f"\n✅ YOLO Master N is {improvement:.1f}% FASTER than YOLOv8s")
    elif fps_result < 55.4:
        slowdown = ((55.4 - fps_result) / 55.4) * 100
        print(f"\n⚠️  YOLO Master N is {slowdown:.1f}% SLOWER than YOLOv8s")
    else:
        print(f"\n≈ YOLO Master N matches YOLOv8s speed")
    
    print(f"\n{'='*70}\n")
    
    return fps_result


def main():
    parser = argparse.ArgumentParser(description='Benchmark YOLO Master N speed')
    parser.add_argument('--config', default='configs/pipeline_config.yaml',
                        help='Path to pipeline config')
    parser.add_argument('--frames', type=int, default=1000,
                        help='Number of frames to process')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Use model name directly - ultralytics will auto-download if needed
    model_path = 'yolo_master_n'
    
    # Get video path
    video_file = config['global'].get('video_file', '')
    
    if not video_file or not os.path.exists(video_file):
        print(f"❌ Could not find video: {video_file}")
        sys.exit(1)
    
    # Run benchmark (ultralytics will auto-download model if not found)
    benchmark_yolo_master_n(model_path, video_file, args.frames, args.verbose)


if __name__ == '__main__':
    main()
