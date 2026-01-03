#!/usr/bin/env python3
"""
Stage 6b: Create Person Selection Report (HTML with 3 Temporal Crops)

Creates an interactive HTML report with:
- Top 10 persons with temporal spread (25%, 50%, 75% of tracklet)
- 3 thumbnail images per person showing start, middle, end appearance
- Clean, sortable table format
- No external dependencies

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


def create_selection_report(canonical_file, crops_cache_file, fps, video_duration_frames, output_html):
    """Create HTML selection report with 3 temporal crops per person"""
    
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
    # {frame_idx: {local_frame_idx: bbox, ...}}
    # local_frame_idx is the 0-based index of detections in that frame
    frame_to_detections = {}
    frame_detection_counts = {}  # Count detections per frame
    
    # First pass: count detections per frame
    for frame_idx in detection_frame_numbers:
        frame_idx = int(frame_idx)
        frame_detection_counts[frame_idx] = frame_detection_counts.get(frame_idx, 0) + 1
    
    # Second pass: build the mapping with local indices
    frame_local_indices = {}  # {frame_idx: current_count}
    for global_det_idx in range(len(detection_frame_numbers)):
        frame_idx = int(detection_frame_numbers[global_det_idx])
        bbox = detection_bboxes[global_det_idx]
        
        # Get local index for this frame
        if frame_idx not in frame_local_indices:
            frame_local_indices[frame_idx] = 0
        local_idx = frame_local_indices[frame_idx]
        frame_local_indices[frame_idx] += 1
        
        # Store with local index (matches crops_cache indexing)
        if frame_idx not in frame_to_detections:
            frame_to_detections[frame_idx] = {}
        frame_to_detections[frame_idx][local_idx] = bbox
    
    # Create HTML report
    print(f"📄 Creating HTML report...")
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    
    # Start HTML
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Person Selection Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #1f4788;
            text-align: center;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th {
            background-color: #1f4788;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        td {
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }
        tr:hover {
            background-color: #f9f9f9;
        }
        .thumbnail {
            max-width: 100px;
            max-height: 120px;
            border: 1px solid #ccc;
            border-radius: 4px;
            margin: 4px;
        }
        .thumbnails-cell {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 4px;
        }
        .rank {
            font-weight: bold;
            color: #1f4788;
        }
        .person-id {
            background-color: #e8f0f7;
            font-weight: bold;
        }
        .stats {
            text-align: center;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <h1>🎯 Person Selection Report - Top 10 Persons</h1>
    <p style="text-align: center; color: #666;">
        Thumbnails show person at 25%, 50%, and 75% of their tracked appearance
    </p>
    <table>
        <thead>
            <tr>
                <th>Rank</th>
                <th>Person ID</th>
                <th>Frames Present</th>
                <th>% of Video (time)</th>
                <th>Thumbnails (25% / 50% / 75%)</th>
            </tr>
        </thead>
        <tbody>
"""
    
    # Add top 10 persons
    for rank, person in enumerate(persons[:10], 1):
        person_id = person['person_id']
        frames = person['frame_numbers']
        num_frames = len(frames)
        
        # Calculate % of video
        percent_video = (num_frames / video_duration_frames) * 100 if video_duration_frames > 0 else 0
        
        # Get 3 temporal crops: 25%, 50%, 75%
        indices = [
            int(num_frames * 0.25),  # 25%
            int(num_frames * 0.50),  # 50%
            int(num_frames * 0.75)   # 75%
        ]
        
        thumbnail_html = ""
        
        for i, idx in enumerate(indices):
            # Clamp to valid range
            idx = min(idx, num_frames - 1)
            frame_num = int(frames[idx])
            person_bbox = person['bboxes'][idx]  # Get person's bbox at this frame
            
            # Find crop matching this person's bbox in this frame
            crop = find_crop_for_person_in_frame(person_bbox, frame_to_detections, crops_cache, frame_num)
            
            if crop is not None:
                # Convert BGR to RGB
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                
                # Encode to PNG in memory
                success, png_array = cv2.imencode('.png', cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
                if success:
                    png_base64 = base64.b64encode(png_array.tobytes()).decode('utf-8')
                    percent_label = ['25%', '50%', '75%'][i]
                    thumbnail_html += f'<img src="data:image/png;base64,{png_base64}" class="thumbnail" title="{percent_label}" alt="{percent_label}">'
        
        # Add row
        html_content += f"""        <tr>
            <td class="rank">{rank}</td>
            <td class="person-id">P{person_id}</td>
            <td class="stats">{num_frames}</td>
            <td class="stats">{percent_video:.1f}%</td>
            <td class="thumbnails-cell">{thumbnail_html}</td>
        </tr>
"""
    
    # Close HTML
    html_content += """        </tbody>
    </table>
    <footer style="text-align: center; color: #666; margin-top: 30px;">
        <p>Generated by Unified Pose Pipeline - Person Selection Report</p>
    </footer>
</body>
</html>
"""
    
    # Write HTML file
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Stage 6b: Create Person Selection Report (HTML with 3 Temporal Crops)'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    canonical_file = config['stage5']['output']['canonical_persons_file']
    crops_cache_file = config['stage4']['input']['crops_cache_file']
    
    output_dir = Path(canonical_file).parent
    output_html = output_dir / 'person_selection_report.html'
    
    # Get video duration from config (or 0 to auto-calculate from data)
    video_duration_frames = config.get('global', {}).get('video_duration_frames', 0)
    
    print(f"\n{'='*70}")
    print(f"📄 STAGE 10: CREATE HTML SELECTION REPORT")
    print(f"{'='*70}\n")
    
    t_start = time.time()
    
    success = create_selection_report(
        canonical_file,
        crops_cache_file,
        fps=None,
        video_duration_frames=video_duration_frames,
        output_html=output_html
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
