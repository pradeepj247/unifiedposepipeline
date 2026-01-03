#!/usr/bin/env python3
"""
Quick Benchmark: YOLOv11n Speed Test

Tests YOLOv11n inference speed on 1000 frames at 1080p.

Usage:
    python quick_benchmark_yolov11n.py --config configs/pipeline_config.yaml --frames 1000
"""

import argparse
import yaml
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


def main():
    parser = argparse.ArgumentParser(description='Quick benchmark: YOLOv11n speed')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    parser.add_argument('--frames', type=int, default=1000,
                       help='Number of frames to process (default: 1000)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    video_path = config['global']['video_dir'] + config['global']['video_file']
    models_dir = config['global']['models_dir']
    
    # YOLOv11n model path
    yolov11n_path = os.path.join(models_dir, 'yolo', 'yolov11n.pt')
    
    print(f"\n{'='*70}")
    print(f"⚡ YOLOv11n SPEED TEST")
    print(f"{'='*70}\n")
    
    print(f"📹 Video: {Path(video_path).name}")
    print(f"🤖 Model: YOLOv11n")
    print(f"📊 Frames: {args.frames}")
    print(f"Resolution: 1920x1080\n")
    
    # Load YOLOv11n
    print(f"🛠️  Loading YOLOv11n...")
    print(f"   Path: {yolov11n_path}")
    try:
        model = YOLO(yolov11n_path, task="detect")
        model.to('cuda')
        print(f"  ✅ Model loaded\n")
    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
        print(f"  💡 Make sure YOLOv11n is at: {yolov11n_path}")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    frame_count = 0
    t_start = time.time()
    
    print(f"⚡ Processing {args.frames} frames...\n")
    pbar = tqdm(total=args.frames, desc="Inference", mininterval=1.0)
    
    while frame_count < args.frames:
        ret, frame = cap.read()
        
        if not ret:
            print(f"\n⚠️  Video ended at frame {frame_count}")
            break
        
        # Run YOLO
        _ = model(frame, conf=0.3, verbose=False)
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    
    t_end = time.time()
    total_time = t_end - t_start
    processing_fps = frame_count / total_time if total_time > 0 else 0
    
    cap.release()
    
    print(f"\n{'='*70}")
    print(f"✅ RESULTS")
    print(f"{'='*70}\n")
    
    print(f"Frames processed: {frame_count}")
    print(f"Time: {total_time:.2f}s")
    print(f"FPS: {processing_fps:.1f}\n")
    
    print(f"{'─'*70}")
    print(f"COMPARISON with YOLOv8s:")
    print(f"{'─'*70}\n")
    print(f"YOLOv8s @ 1080p: 55.4 FPS (18.06s for 1000 frames)")
    print(f"YOLOv11n @ 1080p: {processing_fps:.1f} FPS ({total_time:.2f}s for {frame_count} frames)")
    
    speedup = processing_fps / 55.4
    time_saved = ((18.06 - total_time) / 18.06 * 100)
    
    if speedup > 1.0:
        print(f"\n✅ YOLOv11n is {speedup:.2f}x FASTER than YOLOv8s")
        print(f"   Time saved: {time_saved:.1f}%")
    else:
        print(f"\n⚠️  YOLOv11n is {1/speedup:.2f}x SLOWER than YOLOv8s")
        print(f"   Time cost: {abs(time_saved):.1f}%")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
