#!/usr/bin/env python3
"""
Stage 4: Generate HTML Viewer (Visualization Only)

Simple visualization stage that:
1. Loads final_crops.pkl from Stage 3d (already merged)
2. Creates WebP animations for each person
3. Generates HTML viewer

NO clustering, NO OSNet - that's Stage 3d's job.

Usage:
    python stage4_generate_html.py --config configs/pipeline_config.yaml
"""

import argparse
import yaml
import numpy as np
import time
import re
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

from crop_utils import load_final_crops
from ondemand_crop_extraction import generate_webp_animations, cleanup_webp_files

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import PipelineLogger


def resolve_path_variables(config):
    """Recursively resolve ${variable} in config"""
    global_vars = config.get('global', {})
    
    # First pass: resolve variables within global section itself
    def resolve_string_once(s, vars_dict):
        if not isinstance(s, str):
            return s
        return re.sub(
            r'\$\{(\w+)\}',
            lambda m: str(vars_dict.get(m.group(1), m.group(0))),
            s
        )
    
    # Resolve global variables iteratively
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


def main():
    """Stage 4: Generate HTML Viewer"""
    parser = argparse.ArgumentParser(description='Stage 4: Generate HTML Viewer')
    parser.add_argument('--config', type=str, required=True, help='Path to pipeline config YAML')
    args = parser.parse_args()
    
    # Load and resolve config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Auto-extract current_video from video_file (needed for path resolution)
    video_file = config.get('global', {}).get('video_file', '')
    if video_file:
        import os
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        config['global']['current_video'] = video_name
    
    config = resolve_path_variables(config)
    
    stage_config = config.get('stage4_generate_html', {})
    if not stage_config:
        print("❌ stage4_generate_html config not found")
        return 1
    
    verbose = config.get('global', {}).get('verbose', False)
    
    logger = PipelineLogger("Stage 4: Generate HTML Viewer", verbose=verbose)
    logger.header()
    
    # Extract configuration (config is flat, not nested)
    final_crops_path = Path(stage_config['final_crops_file'])
    output_dir = Path(stage_config['output_dir'])
    
    resize_to = tuple(stage_config.get('resize_to', [256, 256]))
    webp_duration_ms = stage_config.get('webp_duration_ms', 100)
    webp_quality = stage_config.get('webp_quality', 80)
    
    # Print configuration
    if not verbose:
        print(f"   Loaded config: {args.config}")
        print(f"   Input: {final_crops_path.name}")
        print(f"   Output: {output_dir}")
        print(f"   WebP: {resize_to[0]}×{resize_to[1]}, {webp_duration_ms}ms per frame")
        print()
    
    # ==================== Load Crops ====================
    if verbose:
        logger.step("Loading crops from Stage 3d...")
    
    try:
        crops_data = load_final_crops(final_crops_path, verbose=verbose)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.error(f"Error loading final_crops.pkl: {e}")
        return 1
    
    # Load canonical persons for frame information
    canonical_file = final_crops_path.parent / 'canonical_persons_filtered.npz'
    if not canonical_file.exists():
        logger.error(f"Canonical persons file not found: {canonical_file}")
        return 1
    
    canonical_data = np.load(canonical_file, allow_pickle=True)
    canonical_persons = canonical_data['persons']
    
    # Build person info dict with frame data
    person_frame_info = {}
    total_frames = 0
    for person in canonical_persons:
        person_id = person['person_id']
        frame_numbers = person['frame_numbers']
        person_frame_info[person_id] = {
            'start_frame': int(frame_numbers[0]),
            'end_frame': int(frame_numbers[-1]),
            'num_frames': len(frame_numbers),
        }
        total_frames = max(total_frames, int(frame_numbers[-1]) + 1)
    
    # Convert pickle format to person_buckets
    person_buckets = {}
    person_metadata = {}
    
    for person_id in crops_data['person_ids']:
        crops = crops_data['crops'][person_id]
        metadata = crops_data['metadata'][person_id]
        
        person_buckets[person_id] = crops
        person_metadata[person_id] = metadata
    
    total_crops = sum(len(c) for c in person_buckets.values())
    if verbose:
        logger.info(f"Loaded {len(person_buckets)} persons, {total_crops} crops")
    else:
        print(f"   Loaded {len(person_buckets)} persons, {total_crops} crops from {final_crops_path.name}")
    
    # ==================== Generate WebP Animations ====================
    if verbose:
        print()
        logger.step("Generating WebP animations...")
    
    webp_start = time.time()
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate WebP files directly (limit to 50 crops per person)
        import imageio
        import cv2
        import base64
        
        webp_base64_dict = {}  # Store base64 encoded WebPs for HTML embedding
        
        for person_id, crops in sorted(person_buckets.items()):
            if crops is None or len(crops) == 0:
                continue
            
            # Limit to 50 crops (if merged persons have more)
            crops_to_use = crops[:50] if len(crops) > 50 else crops
            
            # Resize all crops to same size
            resized = []
            for crop in crops_to_use:
                resized_crop = cv2.resize(crop, resize_to)
                resized_crop = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)
                resized.append(resized_crop)
            
            # Save as WebP
            output_file = output_dir / f"person_{person_id}.webp"
            imageio.mimsave(
                output_file,
                resized,
                format='WEBP',
                duration=webp_duration_ms,
                loop=0
            )
            
            # Read and encode as base64 for embedding
            with open(output_file, 'rb') as f:
                webp_data = f.read()
            webp_base64_dict[person_id] = base64.b64encode(webp_data).decode('utf-8')
            
            if verbose:
                file_size_kb = output_file.stat().st_size / 1024
                logger.info(f"Person {person_id}: {len(crops_to_use)}/{len(crops)} crops → {output_file.name} ({file_size_kb:.0f} KB)")
        webp_time = time.time() - webp_start
        
        if verbose:
            logger.timing("WebP generation", webp_time)
        else:
            print(f"   ✅ Generated {len(person_buckets)} WebP animations in {webp_time:.2f}s")
    
    except Exception as e:
        logger.error(f"Error during WebP generation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ==================== Create HTML Viewer ====================
    if verbose:
        print()
        logger.step("Creating HTML viewer...")
    
    html_file = output_dir / "viewer.html"
    try:
        create_simple_html_viewer(html_file, person_buckets, person_metadata, person_frame_info, total_frames, webp_base64_dict, verbose)
        if verbose:
            logger.info(f"HTML viewer created: {html_file}")
        else:
            print(f"   ✅ HTML viewer: {html_file.name}")
    except Exception as e:
        logger.error(f"Error creating HTML viewer: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ==================== Cleanup WebP Files ====================
    # WebPs are embedded in HTML as base64, delete originals to save space
    if verbose:
        logger.step("Cleaning up WebP files (embedded in HTML)...")
    
    cleanup_start = time.time()
    deleted_count = 0
    for webp_file in output_dir.glob('person_*.webp'):
        try:
            webp_file.unlink()
            deleted_count += 1
        except Exception as e:
            if verbose:
                logger.warning(f"Could not delete {webp_file.name}: {e}")
    
    cleanup_time = time.time() - cleanup_start
    if verbose:
        logger.info(f"Deleted {deleted_count} WebP files ({cleanup_time:.2f}s)")
    
    # ==================== Summary ====================
    total_time = webp_time + cleanup_time
    
    if verbose:
        print()
        print("=" * 70)
        logger.info(f"Stage 4 Timing:")
        logger.info(f"  - WebP generation: {webp_time:.2f}s")
        logger.info(f"  - Cleanup: {cleanup_time:.2f}s")
        logger.info(f"  - Total: {total_time:.2f}s")
        print()
        logger.info(f"Stage 4 Timing:")
        logger.info(f"  - WebP generation: {webp_time:.2f}s")
        logger.info(f"  - Total: {total_time:.2f}s")
        print()
        logger.info(f"Output:")
        logger.info(f"  - HTML viewer: {html_file}")
        logger.info(f"  - WebP files: {len(person_buckets)} animations")
        logger.info(f"  - Persons: {len(person_buckets)}")
        print()
    
    logger.success()
    
    # Save timing info
    try:
        sidecar_path = output_dir / 'stage4.timings.json'
        sidecar_data = {
            'stage': 'stage4',
            'webp_time': float(webp_time),
            'total_time': float(total_time),
            'num_persons': len(person_buckets),
            'total_crops': total_crops,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        with open(sidecar_path, 'w') as f:
            json.dump(sidecar_data, f, indent=2)
    except Exception:
        pass
    
    return 0


def create_simple_html_viewer(html_file: Path, person_buckets: dict, person_metadata: dict, person_frame_info: dict, total_frames: int, webp_base64_dict: dict, verbose: bool = False):
    """
    Create a simple HTML viewer with WebP animations for each person.
    
    Args:
        html_file: Path to output HTML file
        person_buckets: Dict of person_id -> list of crops
        person_metadata: Dict of person_id -> list of metadata dicts
        person_frame_info: Dict of person_id -> {start_frame, end_frame, num_frames}
        total_frames: Total frames in video
        webp_base64_dict: Dict of person_id -> base64 encoded WebP data
        verbose: Enable verbose output
    """
    
    # Build person cards HTML
    person_cards = []
    
    for person_id in sorted(person_buckets.keys()):
        num_crops = len(person_buckets[person_id])
        
        # Get frame info
        frame_info = person_frame_info.get(person_id, {})
        start_frame = frame_info.get('start_frame', 0)
        end_frame = frame_info.get('end_frame', 0)
        num_frames = frame_info.get('num_frames', 0)
        presence_pct = (num_frames / total_frames * 100) if total_frames > 0 else 0
        
        # Get base64 encoded WebP
        webp_base64 = webp_base64_dict.get(person_id, '')
        data_uri = f"data:image/webp;base64,{webp_base64}"
        
        # Show how many crops were actually used for WebP (50 max)
        crops_used = min(num_crops, 50)
        
        card_html = f"""
        <div class="person-card" data-person-id="{person_id}">
            <div class="person-header">
                <h3>Person {person_id}</h3>
                <span class="person-info">{crops_used} crops | Frames {start_frame}-{end_frame}</span>
                <br>
                <span class="person-info">Presence: {num_frames} frames ({presence_pct:.1f}%)</span>
            </div>
            <div class="person-animation">
                <img src="{data_uri}" alt="Person {person_id}" class="webp-animation">
            </div>
            <button class="select-btn" onclick="selectPerson({person_id})">Select This Person</button>
        </div>
        """
        person_cards.append(card_html)
    
    # Full HTML template
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Person Selection Viewer</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            padding: 20px;
            min-height: 100vh;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding: 30px;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            backdrop-filter: blur(10px);
        }}
        
        .header h1 {{
            font-size: 2.5em;
            color: #4CAF50;
            margin-bottom: 10px;
            text-shadow: 0 2px 10px rgba(76, 175, 80, 0.3);
        }}
        
        .header p {{
            font-size: 1.1em;
            color: #aaa;
        }}
        
        .stats {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
        }}
        
        .stat-item {{
            background: rgba(76, 175, 80, 0.1);
            padding: 15px 30px;
            border-radius: 8px;
            border: 1px solid rgba(76, 175, 80, 0.3);
        }}
        
        .stat-item .label {{
            font-size: 0.9em;
            color: #888;
            margin-bottom: 5px;
        }}
        
        .stat-item .value {{
            font-size: 1.8em;
            color: #4CAF50;
            font-weight: bold;
        }}
        
        .gallery {{
            display: flex;
            gap: 20px;
            overflow-x: auto;
            padding: 20px;
            scroll-behavior: smooth;
        }}
        
        .person-card {{
            flex: 0 0 320px;
            min-width: 320px;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            overflow: hidden;
            transition: all 0.3s ease;
            border: 2px solid transparent;
            backdrop-filter: blur(10px);
        }}
        
        .person-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(76, 175, 80, 0.3);
            border-color: #4CAF50;
        }}
        
        .person-header {{
            padding: 20px;
            background: rgba(0,0,0,0.3);
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        
        .person-header h3 {{
            font-size: 1.5em;
            color: #4CAF50;
            margin-bottom: 8px;
        }}
        
        .person-info {{
            color: #888;
            font-size: 0.9em;
        }}
        
        .person-animation {{
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            background: rgba(0,0,0,0.2);
            min-height: 300px;
        }}
        
        .webp-animation {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        }}
        
        .select-btn {{
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            color: white;
            border: none;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .select-btn:hover {{
            background: linear-gradient(135deg, #45a049 0%, #3d8b40 100%);
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.4);
        }}
        
        .select-btn:active {{
            transform: scale(0.98);
        }}
        
        .selected {{
            border-color: #FFD700 !important;
            box-shadow: 0 0 20px rgba(255, 215, 0, 0.5) !important;
        }}
        
        #selection-info {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(76, 175, 80, 0.95);
            color: white;
            padding: 20px 30px;
            border-radius: 12px;
            font-size: 1.2em;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            display: none;
            animation: slideIn 0.3s ease;
        }}
        
        @keyframes slideIn {{
            from {{
                transform: translateX(400px);
                opacity: 0;
            }}
            to {{
                transform: translateX(0);
                opacity: 1;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Person Selection Viewer</h1>
        <p>Select a person from the gallery below to continue with pose estimation</p>
        <div class="stats">
            <div class="stat-item">
                <div class="label">Total Persons</div>
                <div class="value">{len(person_buckets)}</div>
            </div>
            <div class="stat-item">
                <div class="label">Total Crops</div>
                <div class="value">{sum(len(c) for c in person_buckets.values())}</div>
            </div>
        </div>
    </div>
    
    <div class="gallery">
        {''.join(person_cards)}
    </div>
    
    <div id="selection-info">
        ✓ Selected Person <span id="selected-person-id"></span>
    </div>
    
    <script>
        let selectedPersonId = null;
        
        function selectPerson(personId) {{
            // Remove previous selection
            document.querySelectorAll('.person-card').forEach(card => {{
                card.classList.remove('selected');
            }});
            
            // Mark new selection
            const card = document.querySelector(`[data-person-id="${{personId}}"]`);
            card.classList.add('selected');
            
            // Update selection info
            selectedPersonId = personId;
            document.getElementById('selected-person-id').textContent = personId;
            document.getElementById('selection-info').style.display = 'block';
            
            // Log selection (for automation/scripting)
            console.log(`SELECTED_PERSON: ${{personId}}`);
            
            // Save selection to localStorage
            localStorage.setItem('selected_person_id', personId);
        }}
        
        // Restore previous selection if exists
        window.addEventListener('load', () => {{
            const previousSelection = localStorage.getItem('selected_person_id');
            if (previousSelection) {{
                selectPerson(parseInt(previousSelection));
            }}
        }});
    </script>
</body>
</html>"""
    
    # Write HTML file
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    if verbose:
        print(f"   Created HTML viewer with {len(person_buckets)} persons")


if __name__ == '__main__':
    exit(main())
