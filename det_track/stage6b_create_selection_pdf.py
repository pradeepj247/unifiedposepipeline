#!/usr/bin/env python3
"""
Stage 6b Alternative: Create Person Selection Report (HTML with Embedded Images)

Creates an interactive HTML report with:
- Top 10 persons displayed with statistics
- Thumbnail images embedded directly in the HTML
- Sortable, easy-to-read format
- No external dependencies

Usage:
    python stage6b_create_selection_pdf.py --config configs/pipeline_config.yaml
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


def create_selection_report(canonical_file, crops_cache_file, fps, output_html):
    """Create HTML selection report with embedded thumbnail images"""
    
    # Load data
    print(f"📂 Loading canonical persons...")
    data = np.load(canonical_file, allow_pickle=True)
    persons = list(data['persons'])
    persons.sort(key=lambda p: len(p['frame_numbers']), reverse=True)
    
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
            max-width: 128px;
            max-height: 150px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        .rank {
            font-weight: bold;
            color: #1f4788;
        }
        .person-id {
            background-color: #e8f0f7;
            font-weight: bold;
        }
        .duration {
            text-align: center;
        }
        .stats {
            text-align: center;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <h1>🎯 Person Selection Report - Top 10 Persons</h1>
    <table>
        <thead>
            <tr>
                <th>Rank</th>
                <th>Person ID</th>
                <th>Duration (sec)</th>
                <th>Frames</th>
                <th>Start Frame</th>
                <th>End Frame</th>
                <th>Avg Confidence</th>
                <th>Thumbnail</th>
            </tr>
        </thead>
        <tbody>
"""
    
    # Add top 10 persons
    for rank, person in enumerate(persons[:10], 1):
        person_id = person['person_id']
        frames = person['frame_numbers']
        durations = len(frames)
        start_frame = int(frames[0])
        end_frame = int(frames[-1])
        avg_conf = np.mean(person['confidences'])
        
        duration_seconds = durations / fps if fps > 0 else durations / 25
        
        # Get crop and encode as base64
        crop = get_best_crop_for_person(person, crops_cache)
        thumbnail_html = '(no crop)'
        
        if crop is not None:
            # Convert BGR to RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Encode to PNG in memory
            success, png_array = cv2.imencode('.png', cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
            if success:
                png_base64 = base64.b64encode(png_array.tobytes()).decode('utf-8')
                thumbnail_html = f'<img src="data:image/png;base64,{png_base64}" class="thumbnail" alt="P{person_id}">'
        
        # Add row
        html_content += f"""        <tr>
            <td class="rank">{rank}</td>
            <td class="person-id">P{person_id}</td>
            <td class="duration">{duration_seconds:.1f}</td>
            <td class="stats">{durations}</td>
            <td class="stats">{start_frame}</td>
            <td class="stats">{end_frame}</td>
            <td class="stats">{avg_conf:.3f}</td>
            <td>{thumbnail_html}</td>
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
    with open(output_html, 'w') as f:
        f.write(html_content)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Stage 6b: Create Person Selection Report (HTML with Embedded Images)'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    canonical_file = config['stage4b_group_canonical']['output']['canonical_persons_file']
    crops_cache_file = config['stage4a_reid_recovery']['input']['crops_cache_file']
    
    output_dir = Path(canonical_file).parent
    output_html = output_dir / 'person_selection_report.html'
    
    fps = config.get('global', {}).get('video_fps', 25)
    
    print(f"\n{'='*70}")
    print(f"📄 STAGE 6b: CREATE PERSON SELECTION REPORT")
    print(f"{'='*70}\n")
    
    t_start = time.time()
    
    success = create_selection_report(
        canonical_file,
        crops_cache_file,
        fps,
        output_html
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
