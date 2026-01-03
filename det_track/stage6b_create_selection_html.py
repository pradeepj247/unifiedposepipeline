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
            
            # Get crop from cache
            if frame_num in crops_cache:
                crops_in_frame = crops_cache[frame_num]
                crop = None
                
                # Get first available crop from this frame
                for crop_img in crops_in_frame.values():
                    if crop_img is not None and isinstance(crop_img, np.ndarray):
                        crop = crop_img
                        break
                
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
    
    canonical_file = config['stage4b_group_canonical']['output']['canonical_persons_file']
    crops_cache_file = config['stage4a_reid_recovery']['input']['crops_cache_file']
    
    output_dir = Path(canonical_file).parent
    output_html = output_dir / 'person_selection_report.html'
    
    # Get video duration from config
    video_duration_frames = config.get('global', {}).get('video_duration_frames', 25200)
    
    print(f"\n{'='*70}")
    print(f"📄 STAGE 6b: CREATE PERSON SELECTION REPORT (3 TEMPORAL CROPS)")
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
