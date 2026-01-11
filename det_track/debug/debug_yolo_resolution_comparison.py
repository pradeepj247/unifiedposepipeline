#!/usr/bin/env python3
"""
Debug: Compare YOLO Detection at Different Resolutions (1080p vs 720p)

Tests if resolution affects detection accuracy and bbox coordinates.

Usage:
    python debug_yolo_resolution_comparison.py --config configs/pipeline_config.yaml --frames 400

Outputs:
    p1080/frame_XXXX.png  - Detected bboxes at 1080p (original)
    p1080/frame_XXXX.txt  - Detection coordinates at 1080p
    p720/frame_XXXX.png   - Detected bboxes at 720p (resized)
    p720/frame_XXXX.txt   - Detection coordinates at 720p
"""

import argparse
import numpy as np
import cv2
import yaml
import re
from pathlib import Path
from ultralytics import YOLO


def resolve_path_variables(config):
    """Recursively resolve ${variable} in config with multi-pass resolution"""
    max_passes = 5
    
    for _ in range(max_passes):
        global_vars = config.get('global', {})
        changed = False
        
        def resolve_string(s):
            nonlocal changed
            if not isinstance(s, str):
                return s
            
            def replace_var(match):
                nonlocal changed
                var_name = match.group(1)
                replacement = str(global_vars.get(var_name, match.group(0)))
                if replacement != match.group(0):
                    changed = True
                return replacement
            
            return re.sub(r'\$\{(\w+)\}', replace_var, s)
        
        def resolve_recursive(obj):
            if isinstance(obj, dict):
                return {k: resolve_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve_recursive(v) for v in obj]
            elif isinstance(obj, str):
                return resolve_string(obj)
            return obj
        
        config = resolve_recursive(config)
        
        if not changed:
            break
    
    return config


def load_config(config_path):
    """Load and resolve YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Auto-extract current_video from video_file
    video_file = config.get('global', {}).get('video_file', '')
    if video_file:
        video_name = Path(video_file).stem
        config['global']['current_video'] = video_name
    
    return resolve_path_variables(config)


def load_yolo_model(model_path='yolov8s.pt'):
    """Load YOLO model"""
    try:
        model = YOLO(model_path)
        print(f"✅ Loaded YOLO: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load YOLO: {e}")
        raise


def run_yolo_detection(frame, model, conf_threshold=0.5):
    """Run YOLO on a frame and return detections"""
    results = model(frame, conf=conf_threshold, verbose=False)
    
    detections = []
    if len(results) > 0:
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': conf,
                'class': cls,
                'class_name': 'person' if cls == 0 else f'class_{cls}'
            })
    
    return detections


def draw_detections(frame, detections, color=(0, 255, 0), thickness=2):
    """Draw bboxes on frame"""
    frame_vis = frame.copy()
    
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = map(int, det['bbox'])
        conf = det['confidence']
        
        # Draw bbox
        cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        label = f"[{idx}] {det['class_name']} {conf:.2f}"
        cv2.putText(frame_vis, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame_vis


def write_detections_to_file(frame_num, detections, resolution, output_file):
    """Write detection coordinates to text file"""
    with open(output_file, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write(f"FRAME {frame_num} - YOLO DETECTIONS ({resolution})\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"Total detections: {len(detections)}\n\n")
        
        f.write("DETECTIONS:\n")
        f.write("-" * 80 + "\n")
        
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            w = x2 - x1
            h = y2 - y1
            conf = det['confidence']
            
            f.write(f"[{idx}] {det['class_name']}\n")
            f.write(f"    bbox: [x1:{x1:.1f}, y1:{y1:.1f}, x2:{x2:.1f}, y2:{y2:.1f}]\n")
            f.write(f"    size: {w:.1f}×{h:.1f}\n")
            f.write(f"    confidence: {conf:.2f}\n\n")
        
        f.write("="*80 + "\n")


def process_video(video_path, model, max_frames, resolution_name, output_dir, target_resolution=None):
    """Process video at specified resolution"""
    print(f"\n{'='*80}")
    print(f"📹 Processing at {resolution_name}")
    print(f"{'='*80}\n")
    
    cap = cv2.VideoCapture(str(video_path))
    
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📂 Video: {video_path.name}")
    print(f"📊 Original resolution: {original_width}×{original_height}")
    print(f"⏱️  FPS: {fps}")
    
    if target_resolution:
        print(f"🔄 Resizing to: {target_resolution[0]}×{target_resolution[1]}")
    
    print(f"📺 Processing first {max_frames} frames\n")
    
    frame_count = 0
    output_subdir = output_dir / resolution_name
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        
        if not ret:
            print(f"⚠️  Reached end of video at frame {frame_count}")
            break
        
        # Resize if needed
        if target_resolution:
            frame = cv2.resize(frame, target_resolution)
        
        # Run YOLO
        detections = run_yolo_detection(frame, model, conf_threshold=0.5)
        
        # Draw and save frame
        frame_vis = draw_detections(frame, detections, color=(0, 255, 0), thickness=2)
        
        frame_png = output_subdir / f'frame_{frame_count:04d}.png'
        cv2.imwrite(str(frame_png), frame_vis)
        
        # Save coordinates
        frame_txt = output_subdir / f'frame_{frame_count:04d}.txt'
        write_detections_to_file(frame_count, detections, resolution_name, frame_txt)
        
        # Progress
        if (frame_count + 1) % 50 == 0:
            print(f"   ✅ Processed frame {frame_count + 1}/{max_frames}")
        
        frame_count += 1
    
    cap.release()
    
    print(f"\n✅ Completed {resolution_name}: {frame_count} frames")
    print(f"📂 Output: {output_subdir}\n")
    
    return frame_count


def main():
    parser = argparse.ArgumentParser(description='Compare YOLO detection at 1080p vs 720p')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    parser.add_argument('--frames', type=int, default=400,
                       help='Number of frames to process (default: 400)')
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                       help='YOLO model path (default: yolov8s.pt)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Get paths
    video_path = Path(config['global']['video_dir'] + config['global']['video_file']).resolve().absolute()
    output_dir = Path(config['global']['outputs_dir']).resolve().absolute() / config['global']['current_video']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"🔍 YOLO RESOLUTION COMPARISON (1080p vs 720p)")
    print(f"{'='*80}\n")
    
    # Load model
    model = load_yolo_model(args.model)
    
    # Process at 1080p (original)
    count_1080 = process_video(video_path, model, args.frames, 'p1080', output_dir, target_resolution=None)
    
    # Process at 720p (resized)
    count_720 = process_video(video_path, model, args.frames, 'p720', output_dir, target_resolution=(1280, 720))
    
    print(f"\n{'='*80}")
    print(f"✅ ANALYSIS COMPLETE!")
    print(f"{'='*80}\n")
    print(f"📊 Results:")
    print(f"   p1080/: {count_1080} frames")
    print(f"   p720/:  {count_720} frames")
    print(f"\n📂 Output: {output_dir}")
    print(f"   • p1080/frame_XXXX.png  - Detections at 1080p")
    print(f"   • p1080/frame_XXXX.txt  - Coordinates at 1080p")
    print(f"   • p720/frame_XXXX.png   - Detections at 720p")
    print(f"   • p720/frame_XXXX.txt   - Coordinates at 720p")
    print(f"\n💡 Compare frame_XXXX.txt files to see bbox coordinate differences")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
