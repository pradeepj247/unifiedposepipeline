#!/usr/bin/env python3
"""
Stage 10: Create Person Selection Report (HTML with Embedded MP4 Videos)

Creates an interactive HTML report with:
- Top 10 persons sorted by appearance duration
- Embedded MP4 videos (50 frames each, ~3.3 seconds)
- Play/pause controls, timeline scrubbing
- Clean, responsive layout
- Lightweight HTML with embedded video data

Usage:
    python stage6b_create_selection_html.py --config configs/pipeline_config.yaml
"""

import argparse
import numpy as np
import pickle
import yaml
import re
import os
import cv2
import base64
import io
from pathlib import Path
import time
from datetime import timedelta


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
    
    video_file = config.get('global', {}).get('video_file', '')
    if video_file:
        video_name = os.path.splitext(video_file)[0]
        config['global']['current_video'] = video_name
    
    return resolve_path_variables(config)


def get_best_crop_for_person(person, crops_cache):
    """Get highest-confidence crop for person"""
    if person.get('frame_numbers') is None or len(person['frame_numbers']) == 0:
        return None
    
    confidences = person['confidences']
    best_idx = np.argmax(confidences)
    best_frame = int(person['frame_numbers'][best_idx])
    
    if best_frame in crops_cache:
        crops_in_frame = crops_cache[best_frame]
        for crop_image in crops_in_frame.values():
            if crop_image is not None and isinstance(crop_image, np.ndarray):
                return crop_image
    
    return None


def bbox_iou(bbox1, bbox2):
    """Calculate IoU between two bboxes [x1, y1, x2, y2]"""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Intersection
    inter_min_x = max(x1_min, x2_min)
    inter_min_y = max(y1_min, y2_min)
    inter_max_x = min(x1_max, x2_max)
    inter_max_y = min(y1_max, y2_max)
    
    if inter_max_x < inter_min_x or inter_max_y < inter_min_y:
        return 0.0
    
    inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
    
    # Union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def find_crop_for_person_in_frame(person_bbox, frame_to_detections, crops_cache, frame_idx):
    """
    Find the crop that matches a person's bbox in a specific frame.
    Uses IoU to find the best matching detection.
    frame_to_detections: {frame_idx: {local_idx: bbox, ...}}
    crops_cache: {frame_idx: {local_idx: crop_image, ...}}
    """
    if frame_idx not in frame_to_detections or frame_idx not in crops_cache:
        return None
    
    detections_in_frame = frame_to_detections[frame_idx]  # {local_idx: bbox, ...}
    crops_in_frame = crops_cache[frame_idx]  # {local_idx: crop_image, ...}
    
    # Find detection with highest IoU to person_bbox
    best_det_idx = None
    best_iou = 0.0
    
    for local_idx, det_bbox in detections_in_frame.items():
        iou = bbox_iou(person_bbox, det_bbox)
        if iou > best_iou:
            best_iou = iou
            best_det_idx = local_idx
    
    # Return crop if found and IoU is reasonable (>0.5)
    if best_det_idx is not None and best_iou > 0.5:
        if best_det_idx in crops_in_frame:
            crop = crops_in_frame[best_det_idx]
            if crop is not None and isinstance(crop, np.ndarray):
                return crop
    
    # Fallback: return first available crop (better than nothing)
    for crop_img in crops_in_frame.values():
        if crop_img is not None and isinstance(crop_img, np.ndarray):
            return crop_img
    
    return None


def create_selection_report(canonical_file, crops_cache_file, fps, video_duration_frames, output_html, videos_dir=None):
    """Create HTML selection report with embedded MP4 videos using file references"""
    
    # Load data
    print(f"📂 Loading canonical persons...")
    data = np.load(canonical_file, allow_pickle=True)
    persons = list(data['persons'])
    persons.sort(key=lambda p: len(p['frame_numbers']), reverse=True)
    
    # If video_duration_frames not provided, calculate from max frame in data
    if video_duration_frames is None or video_duration_frames == 0:
        max_frame = 0
        for person in persons:
            if len(person['frame_numbers']) > 0:
                max_frame = max(max_frame, int(person['frame_numbers'][-1]))
        video_duration_frames = max_frame + 1
        print(f"   Calculated video_duration_frames from data: {video_duration_frames}")
    
    # If videos_dir not provided, try to find it relative to canonical file
    if videos_dir is None:
        videos_dir = Path(canonical_file).parent / 'videos'
    else:
        videos_dir = Path(videos_dir)
    
    print(f"📂 Loading crops cache...")
    with open(crops_cache_file, 'rb') as f:
        crops_cache = pickle.load(f)
    
    # Load detections to map bboxes to detection indices
    print(f"📂 Loading detections (for bbox-to-crop mapping)...")
    detections_file = Path(crops_cache_file).parent / 'detections_raw.npz'
    detections_data = np.load(str(detections_file), allow_pickle=True)
    detection_frame_numbers = detections_data['frame_numbers']
    detection_bboxes = detections_data['bboxes']
    
    # Build frame->detection mapping using LOCAL frame indices
    frame_to_detections = {}
    frame_detection_counts = {}
    
    # First pass: count detections per frame
    for frame_idx in detection_frame_numbers:
        frame_idx = int(frame_idx)
        frame_detection_counts[frame_idx] = frame_detection_counts.get(frame_idx, 0) + 1
    
    # Second pass: build the mapping with local indices
    frame_local_indices = {}
    for global_det_idx in range(len(detection_frame_numbers)):
        frame_idx = int(detection_frame_numbers[global_det_idx])
        bbox = detection_bboxes[global_det_idx]
        
        if frame_idx not in frame_local_indices:
            frame_local_indices[frame_idx] = 0
        local_idx = frame_local_indices[frame_idx]
        frame_local_indices[frame_idx] += 1
        
        if frame_idx not in frame_to_detections:
            frame_to_detections[frame_idx] = {}
        frame_to_detections[frame_idx][local_idx] = bbox
    
    # Create HTML report
    print(f"📄 Creating HTML report with video file references...")
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate relative path from HTML to videos directory
    try:
        rel_videos_path = Path(videos_dir).relative_to(output_html.parent)
    except ValueError:
        # If on different drives, use absolute path
        rel_videos_path = Path(videos_dir).absolute()
    
    # Start HTML with video file references (not embedded)
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Person Selection Report with Videos</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, #1f4788 0%, #2c3e50 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .persons-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 25px;
            padding: 30px;
        }
        
        .person-card {
            background: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
            transition: transform 0.3s, box-shadow 0.3s;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .person-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }
        
        .person-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .person-rank {
            font-size: 1.8em;
            font-weight: bold;
        }
        
        .person-id {
            font-size: 2em;
            font-weight: bold;
        }
        
        .video-container {
            position: relative;
            width: 100%;
            aspect-ratio: 2 / 3;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        video {
            width: 100%;
            height: 100%;
            object-fit: contain;
            background: #000;
        }
        
        .person-stats {
            padding: 15px;
            background: white;
            font-size: 0.95em;
        }
        
        .stat-row {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            padding: 8px 0;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .stat-row:last-child {
            border-bottom: none;
        }
        
        .stat-label {
            color: #666;
            font-weight: 500;
        }
        
        .stat-value {
            color: #333;
            font-weight: bold;
        }
        
        .no-video {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 300px;
            background: #f0f0f0;
            color: #999;
            font-size: 1em;
            text-align: center;
            padding: 20px;
        }
        
        footer {
            text-align: center;
            padding: 20px;
            background: #f5f5f5;
            color: #666;
            font-size: 0.9em;
            border-top: 1px solid #e0e0e0;
        }
        
        .rank-badge {
            display: inline-block;
            background: gold;
            color: #333;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.85em;
        }
        
        .info-box {
            background: #fffbea;
            border: 1px solid #ffe8a3;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 30px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎯 Person Selection Report</h1>
            <p>Top 10 Persons with Embedded Video Preview (50 frames each)</p>
        </header>
        
        <div class="info-box">
            ℹ️ <strong>Note:</strong> Make sure the <code>videos</code> folder is in the same directory as this HTML file.
            Videos load from: <code>{rel_videos_path}</code>
        </div>
        
        <div class="persons-grid">
"""
    
    # Add top 10 persons with video file references
    for rank, person in enumerate(persons[:10], 1):
        person_id = person['person_id']
        frames = person['frame_numbers']
        num_frames = len(frames)
        
        # Get start and end frames
        start_frame = int(frames[0])
        end_frame = int(frames[-1])
        
        # Calculate % of video
        percent_video = (num_frames / video_duration_frames) * 100 if video_duration_frames > 0 else 0
        
        # Duration in seconds (50 frames at 15 fps)
        video_duration_sec = 50 / 15.0  # ~3.3 seconds
        
        # Create relative path to video file
        video_filename = f"person_{person_id:02d}.mp4"
        video_path = rel_videos_path / video_filename
        
        # Check if video exists
        video_exists = (Path(output_html.parent) / video_path).exists()
        
        if video_exists:
            video_html = f'''<video controls playsinline style="width: 100%; height: 100%;">
                <source src="{video_path}" type="video/mp4">
                Your browser does not support the video tag.
            </video>'''
        else:
            video_html = f'<div class="no-video">Video not found<br/>({video_filename})<br/><br/>Make sure videos folder exists</div>'
        
        html_content += f'''        <div class="person-card">
            <div class="person-header">
                <span class="person-rank">#{rank}</span>
                <span class="person-id">P{person_id}</span>
            </div>
            <div class="video-container">
                {video_html}
            </div>
            <div class="person-stats">
                <div class="stat-row">
                    <span class="stat-label">Frames Present:</span>
                    <span class="stat-value">{num_frames}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Frame Range:</span>
                    <span class="stat-value">{start_frame} - {end_frame}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Video Coverage:</span>
                    <span class="stat-value">{percent_video:.1f}%</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Video Duration:</span>
                    <span class="stat-value">~{video_duration_sec:.1f}s @ 15fps</span>
                </div>
            </div>
        </div>
'''
    
    # Close HTML
    html_content += """        </div>
        
        <footer>
            <p>Generated by Unified Pose Pipeline - Person Selection Report with Video Files</p>
            <p>Videos are in H.264 codec for wide browser compatibility</p>
        </footer>
    </div>
</body>
</html>
"""
    
    # Write HTML file
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Stage 10: Create Person Selection Report (HTML with Embedded MP4 Videos)'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    canonical_file = config['stage5']['output']['canonical_persons_file']
    crops_cache_file = config['stage4']['input']['crops_cache_file']
    
    output_dir = Path(canonical_file).parent
    output_html = output_dir / 'person_selection_report.html'
    
    # Try to find videos directory (from Stage 11 output)
    videos_dir = output_dir / 'videos'
    
    # Get video duration from config (or 0 to auto-calculate from data)
    video_duration_frames = config.get('global', {}).get('video_duration_frames', 0)
    
    print(f"\n{'='*70}")
    print(f"📄 STAGE 10: CREATE HTML SELECTION REPORT WITH EMBEDDED VIDEOS")
    print(f"{'='*70}\n")
    
    t_start = time.time()
    
    success = create_selection_report(
        canonical_file,
        crops_cache_file,
        fps=None,
        video_duration_frames=video_duration_frames,
        output_html=output_html,
        videos_dir=videos_dir
    )
    
    t_end = time.time()
    
    if success:
        html_size_mb = output_html.stat().st_size / (1024 * 1024) if output_html.exists() else 0
        
        print(f"\n✅ Report created!")
        print(f"   HTML file: {output_html.name} ({html_size_mb:.2f} MB)")
        print(f"   Open in browser: file://{output_html.absolute()}")
        print(f"⏱️  Time: {t_end - t_start:.2f}s")
        print(f"\n{'='*70}\n")
        return True
    else:
        print(f"\n❌ Failed to create report")
        print(f"{'='*70}\n")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
