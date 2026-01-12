#!/usr/bin/env python3
"""
Stage 10: Create Person Selection Report - Horizontal Scrollable Tape Layout

Creates an interactive HTML report with:
- Horizontal scrollable tape layout (1 row × 10 columns)
- Auto-playing video thumbnails on hover (starts from first frame)
- Static poster frame on initial load
- Embedded base64 video and image data (fully self-contained)
- Person stats below each thumbnail
- Responsive design for any screen width
- ~1.2 MB total HTML file size

Features:
- Reduced video dimensions (128×192) for smaller file sizes
- Reduced bitrate (200 kbps) to minimize base64 bloat
- Extract first frame from each MP4 as poster image
- JavaScript hover events for auto-play/pause
- Click to select person with visual feedback

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
from tqdm import tqdm


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


def encode_image_to_base64(image_bgr):
    """Encode BGR image to base64 JPEG string"""
    if image_bgr is None:
        return None
    
    # Encode to JPEG with quality 80
    success, buffer = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not success:
        return None
    
    # Convert to base64
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_str}"


def encode_webp_to_base64(webp_path):
    """Encode WebP file to base64 data URI"""
    try:
        with open(webp_path, 'rb') as f:
            webp_data = f.read()
        
        base64_str = base64.b64encode(webp_data).decode('utf-8')
        return f"data:image/webp;base64,{base64_str}"
    except Exception as e:
        print(f"      ⚠️  Error encoding WebP: {str(e)[:100]}")
        return None


def create_selection_report_horizontal(canonical_file, crops_cache_file, output_html, webp_dir=None, video_duration_frames=None):
    """Create horizontal scrollable tape layout HTML with embedded animated WebPs"""
    
    # Load data
    print(f"📂 Loading canonical persons...")
    data = np.load(canonical_file, allow_pickle=True)
    persons = list(data['persons'])
    persons.sort(key=lambda p: len(p['frame_numbers']), reverse=True)
    
    # Calculate video duration if not provided
    if video_duration_frames is None or video_duration_frames == 0:
        max_frame = 0
        for person in persons:
            if len(person['frame_numbers']) > 0:
                max_frame = max(max_frame, int(person['frame_numbers'][-1]))
        video_duration_frames = max_frame + 1
        print(f"🛠️  Calculated video_duration_frames from data: {video_duration_frames}")
    
    # Locate WebP directory
    if webp_dir is None:
        webp_dir = Path(canonical_file).parent / 'webp'
    else:
        webp_dir = Path(webp_dir)
    
    print(f"📂 Loading crops cache...")
    with open(crops_cache_file, 'rb') as f:
        crops_cache = pickle.load(f)
    
    # Create output directory
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    
    # Start HTML with horizontal tape CSS
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Person Selection - Video Tape</title>
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
            max-width: 100%;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, #1f4788 0%, #2c3e50 100%);
            color: white;
            padding: 25px;
            text-align: center;
        }
        
        header h1 {
            font-size: 2.2em;
            margin-bottom: 8px;
        }
        
        header p {
            font-size: 1em;
            opacity: 0.9;
        }
        
        /* Horizontal scrollable tape */
        .tape-wrapper {
            position: relative;
            padding: 20px;
            background: #f5f5f5;
            overflow-x: auto;
            overflow-y: hidden;
        }
        
        .tape {
            display: flex;
            gap: 15px;
            min-width: min-content;
            padding: 10px 0;
        }
        
        /* Person card - optimized for tape layout */
        .person-card {
            flex: 0 0 auto;
            width: 140px;
            background: white;
            border: 2px solid #ddd;
            border-radius: 6px;
            overflow: hidden;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            cursor: pointer;
        }
        
        .person-card:hover {
            border-color: #667eea;
            box-shadow: 0 6px 16px rgba(102, 126, 234, 0.3);
            transform: scale(1.05);
        }
        
        .person-card.selected {
            border-color: #28a745;
            background: #f0f9ff;
            box-shadow: 0 0 12px rgba(40, 167, 69, 0.5);
        }
        
        /* Rank badge */
        .rank-badge {
            position: absolute;
            top: 5px;
            right: 5px;
            background: gold;
            color: #333;
            padding: 3px 8px;
            border-radius: 12px;
            font-weight: bold;
            font-size: 0.75em;
            z-index: 10;
        }
        
        /* Video thumbnail */
        .video-thumb {
            position: relative;
            width: 100%;
            aspect-ratio: 2 / 3;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        
        .video-thumb img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: opacity 0.3s;
        }
        
        .video-thumb video {
            width: 100%;
            height: 100%;
            object-fit: cover;
            opacity: 0;
            position: absolute;
            top: 0;
            left: 0;
        }
        
        .video-thumb video.playing {
            opacity: 1;
        }
        
        /* Play icon overlay on poster */
        .play-icon {
            position: absolute;
            width: 35px;
            height: 35px;
            background: rgba(255, 255, 255, 0.8);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 5;
        }
        
        .play-icon::after {
            content: '▶';
            color: #667eea;
            font-size: 16px;
            margin-left: 3px;
        }
        
        /* Person info below thumbnail */
        .person-info {
            padding: 10px;
            font-size: 0.85em;
            text-align: center;
        }
        
        .person-id {
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }
        
        .person-stats {
            font-size: 0.75em;
            color: #666;
            line-height: 1.3;
        }
        
        .stat {
            display: flex;
            justify-content: space-between;
            padding: 2px 0;
        }
        
        /* Footer */
        footer {
            text-align: center;
            padding: 20px;
            background: #f5f5f5;
            color: #666;
            font-size: 0.9em;
            border-top: 1px solid #ddd;
        }
        
        .info-section {
            background: #fffbea;
            border: 1px solid #ffe8a3;
            padding: 15px 20px;
            color: #666;
            font-size: 0.95em;
        }
        
        /* Scrollbar styling */
        .tape-wrapper::-webkit-scrollbar {
            height: 8px;
        }
        
        .tape-wrapper::-webkit-scrollbar-track {
            background: #e0e0e0;
        }
        
        .tape-wrapper::-webkit-scrollbar-thumb {
            background: #667eea;
            border-radius: 4px;
        }
        
        .tape-wrapper::-webkit-scrollbar-thumb:hover {
            background: #764ba2;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎬 Person Selection Tape</h1>
            <p>Hover to preview · Click to select</p>
        </header>
        
        <div class="info-section">
            <strong>📽️ How to use:</strong> Hover over any thumbnail to preview the video. 
            Videos are 128×192 resolution at 10 fps (reduced size for fast loading). 
            Click to select a person.
        </div>
        
        <div class="tape-wrapper">
            <div class="tape" id="tape">
"""
    
    print(f"🎬 Encoding WebP files and generating HTML...\n")
    
    # Process top 10 persons
    verbose = False  # Get from config if needed
    for rank, person in tqdm(enumerate(persons[:10], 1), total=min(10, len(persons)), desc="Encoding WebP files", disable=not verbose):
        person_id = person['person_id']
        frames = person['frame_numbers']
        num_frames = len(frames)
        
        # Get start and end frames
        start_frame = int(frames[0])
        end_frame = int(frames[-1])
        
        # Calculate % of video
        percent_video = (num_frames / video_duration_frames) * 100 if video_duration_frames > 0 else 0
        
        # Locate WebP file
        webp_filename = f"person_{person_id:02d}.webp"
        webp_path = webp_dir / webp_filename
        
        if not webp_path.exists():
            print(f"  ⚠️  Rank {rank}: P{person_id} - WebP not found ({webp_filename})")
            html_content += f"""            <div class="person-card" onclick="selectPerson(this, {person_id})">
                <div class="rank-badge">#{rank}</div>
                <div class="video-thumb">
                    <img src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='210'%3E%3Crect fill='%23f0f0f0'/%3E%3Ctext x='50%25' y='50%25' text-anchor='middle' dominant-baseline='middle' fill='%23999' font-family='Arial' font-size='12'%3EMISSING%3C/text%3E%3C/svg%3E"/>
                </div>
                <div class="person-info">
                    <div class="person-id">P{person_id}</div>
                    <div class="person-stats">
                        <div class="stat"><span>Frames:</span><span>{num_frames}</span></div>
                        <div class="stat"><span>Coverage:</span><span>{percent_video:.0f}%</span></div>
                    </div>
                </div>
            </div>
"""
            continue
        
        # Encode WebP to base64
        webp_data = encode_webp_to_base64(webp_path)
        
        if not webp_data:
            html_content += f"""            <div class="person-card" onclick="selectPerson(this, {person_id})">
                <div class="rank-badge">#{rank}</div>
                <div class="video-thumb">
                    <img src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='210'%3E%3Crect fill='%23f0f0f0'/%3E%3C/svg%3E"/>
                </div>
                <div class="person-info">
                    <div class="person-id">P{person_id}</div>
                    <div class="person-stats">
                        <div class="stat"><span>Frames:</span><span>{num_frames}</span></div>
                        <div class="stat"><span>Range:</span><span>{start_frame}-{end_frame}</span></div>
                        <div class="stat"><span>Coverage:</span><span>{percent_video:.0f}%</span></div>
                    </div>
                </div>
            </div>
"""
            continue
        
        # Create person card with animated WebP (no video tag needed)
        html_content += f"""            <div class="person-card" onclick="selectPerson(this, {person_id})">
                <div class="rank-badge">#{rank}</div>
                <div class="video-thumb">
                    <img src="{webp_data}" alt="Person {person_id}" class="webp-animation"/>
                </div>
                <div class="person-info">
                    <div class="person-id">P{person_id}</div>
                    <div class="person-stats">
                        <div class="stat"><span>Frames:</span><span>{num_frames}</span></div>
                        <div class="stat"><span>Range:</span><span>{start_frame}-{end_frame}</span></div>
                        <div class="stat"><span>Coverage:</span><span>{percent_video:.0f}%</span></div>
                    </div>
                </div>
            </div>
"""
    
    # Close HTML with JavaScript for hover auto-play
    html_content += """            </div>
        </div>
        
        <footer>
            <p>🎥 Person Selection Report · Horizontal Scrollable Tape Layout</p>
            <p>All videos embedded at 128×192 resolution, 10 fps, optimized for fast loading</p>
            <p id="selection-info" style="margin-top: 10px; color: #667eea; font-weight: bold;"></p>
        </footer>
    </div>
    
    <script>\n        function selectPerson(card, id) {\n            document.querySelectorAll(\".person-card\")\n                .forEach(c => c.classList.remove(\"selected\"));\n            card.classList.add(\"selected\");\n            console.log(\"Selected person:\", id);\n        }\n    </script>
    </script>
</body>
</html>
"""
    
    # Write HTML file
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    html_size_mb = output_html.stat().st_size / (1024 * 1024) if output_html.exists() else 0
    print(f"\n✅ HTML Selection Report Created")
    print(f"    Filename: {output_html.name}")
    print(f"    Size: {html_size_mb:.2f} MB")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Stage 10: Create Person Selection Report (Horizontal Tape Layout)'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    canonical_file = config['stage7']['input']['canonical_persons_file']
    crops_cache_file = config['stage4']['input']['crops_cache_file']
    
    output_dir = Path(canonical_file).parent
    output_html = output_dir / 'person_selection_report.html'
    
    # Try to find WebP directory
    webp_dir = output_dir / 'webp'
    
    # Get video duration from config
    video_duration_frames = config.get('global', {}).get('video_duration_frames', 0)
    
    print(f"\n{'='*70}")
    print(f"📄 STAGE 11: CREATE HORIZONTAL TAPE LAYOUT SELECTION REPORT")
    print(f"{'='*70}\n")
    
    t_start = time.time()
    
    success = create_selection_report_horizontal(
        canonical_file,
        crops_cache_file,
        output_html,
        webp_dir=webp_dir,
        video_duration_frames=video_duration_frames
    )
    
    t_end = time.time()
    
    if success:
        print(f"    Time taken: {t_end - t_start:.2f}s")
        print(f"\n✅ Stage 11: HTML Selection Report (Horizontal Tape) completed in {t_end - t_start:.2f}s")
        print(f"{'='*70}\n")
        return 0
    else:
        print(f"\n❌ Stage 11 failed")
        print(f"{'='*70}\n")
        return 1


if __name__ == '__main__':
    exit(main())
