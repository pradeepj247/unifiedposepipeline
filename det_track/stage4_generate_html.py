#!/usr/bin/env python3
"""
Stage 4: Generate HTML Viewer (Dual Comparison Mode)

Visualization stage for debugging Stage 3d merging logic:
1. Loads Stage 3c outputs (before merge) - typically 10 persons
2. Loads Stage 3d outputs (after merge) - typically 7 persons
3. Creates WebP animations for both sets
4. Generates HTML viewer with 2 rows and merge info panel

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
    parser.add_argument('--dual-row', type=lambda x: x.lower() in ['true', '1', 'yes'],
                       default=None,
                       help='Enable dual-row mode (shows Stage 3C and 3D comparison). Overrides config.')
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
    
    # Check if dual-row mode is enabled
    # Priority: CLI argument > config value > default (True)
    if args.dual_row is not None:
        dual_row_mode = args.dual_row
    else:
        dual_row_mode = config.get('stage4_html', {}).get('dual_row', True)
    
    logger = PipelineLogger("Stage 4: Generate HTML Viewer", verbose=verbose)
    logger.header()
    
    # Log mode
    if dual_row_mode:
        logger.info("Dual-row mode: ✅ Enabled (shows both Stage 3c and 3d)")
    else:
        logger.info("Dual-row mode: ❌ Disabled (shows only Stage 3d - single row)")
    
    # Extract configuration (both 3c and 3d inputs)
    crops_3c_path = Path(stage_config['final_crops_3c_file'])
    canonical_3c_path = Path(stage_config['canonical_persons_3c_file'])
    crops_3d_path = Path(stage_config['final_crops_3d_file'])
    canonical_3d_path = Path(stage_config['canonical_persons_3d_file'])
    merging_report_path = Path(stage_config['merging_report_file'])
    output_dir = Path(stage_config['output_dir'])
    
    resize_to = tuple(stage_config.get('resize_to', [256, 256]))
    webp_duration_ms = stage_config.get('webp_duration_ms', 100)
    webp_quality = stage_config.get('webp_quality', 80)
    
    # Print configuration
    if not verbose:
        print(f"   Input 3c: {crops_3c_path.name}, {canonical_3c_path.name}")
        print(f"   Input 3d: {crops_3d_path.name}, {canonical_3d_path.name}")
        print()
    
    # ==================== Load Stage 3c Data ====================
    person_buckets_3c = {}
    person_frame_info_3c = {}
    total_frames_3c = 0
    
    if dual_row_mode:
        if verbose:
            logger.step("Loading Stage 3c data (before merge)...")
        
        try:
            crops_3c_data = load_final_crops(crops_3c_path, verbose=verbose)
            canonical_3c_data = np.load(canonical_3c_path, allow_pickle=True)
            canonical_3c_persons = canonical_3c_data['persons']
            
            person_buckets_3c = {pid: crops_3c_data['crops'][pid] for pid in crops_3c_data['person_ids']}
            
            for person in canonical_3c_persons:
                person_id = person['person_id']
                frame_numbers = person['frame_numbers']
                person_frame_info_3c[person_id] = {
                    'start_frame': int(frame_numbers[0]),
                    'end_frame': int(frame_numbers[-1]),
                    'num_frames': len(frame_numbers),
                }
                total_frames_3c = max(total_frames_3c, int(frame_numbers[-1]) + 1)
            
            if not verbose:
                print(f"   ✅ Stage 3c: {len(person_buckets_3c)} persons")
                
        except FileNotFoundError as e:
            logger.error(str(e))
            return 1
        except Exception as e:
            logger.error(f"Error loading Stage 3c data: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        if verbose:
            logger.info("⏭️  Skipping Stage 3c data (single-row mode)")
        else:
            print(f"   ⏭️  Stage 3c: Skipped (single-row mode)")
    
    # ==================== Load Stage 3d Data ====================
    # Check if Stage 3d outputs exist (they won't if Stage 3d was disabled)
    stage3d_exists = crops_3d_path.exists() and canonical_3d_path.exists()
    
    if stage3d_exists:
        if verbose:
            logger.step("Loading Stage 3d data (after merge)...")
        
        try:
            crops_3d_data = load_final_crops(crops_3d_path, verbose=verbose)
            canonical_3d_data = np.load(canonical_3d_path, allow_pickle=True)
            canonical_3d_persons = canonical_3d_data['persons']
            
            person_buckets_3d = {pid: crops_3d_data['crops'][pid] for pid in crops_3d_data['person_ids']}
            person_frame_info_3d = {}
            total_frames_3d = 0
            
            for person in canonical_3d_persons:
                person_id = person['person_id']
                frame_numbers = person['frame_numbers']
                person_frame_info_3d[person_id] = {
                    'start_frame': int(frame_numbers[0]),
                    'end_frame': int(frame_numbers[-1]),
                    'num_frames': len(frame_numbers),
                }
                total_frames_3d = max(total_frames_3d, int(frame_numbers[-1]) + 1)
            
            if not verbose:
                print(f"   ✅ Stage 3d: {len(person_buckets_3d)} persons")
                
        except FileNotFoundError as e:
            logger.error(str(e))
            return 1
        except Exception as e:
            logger.error(f"Error loading Stage 3d data: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        # Stage 3d was skipped (e.g., fast mode) - use Stage 3c data as the main view
        if verbose:
            logger.info("⏭️  Stage 3d skipped - using Stage 3c data as main view")
        else:
            print(f"   ⏭️  Stage 3d: Skipped (using Stage 3c data)")
        
        # Load Stage 3c data if not already loaded
        if not person_buckets_3c:
            try:
                crops_3c_data = load_final_crops(crops_3c_path, verbose=verbose)
                canonical_3c_data = np.load(canonical_3c_path, allow_pickle=True)
                canonical_3c_persons = canonical_3c_data['persons']
                
                person_buckets_3c = {pid: crops_3c_data['crops'][pid] for pid in crops_3c_data['person_ids']}
                
                for person in canonical_3c_persons:
                    person_id = person['person_id']
                    frame_numbers = person['frame_numbers']
                    person_frame_info_3c[person_id] = {
                        'start_frame': int(frame_numbers[0]),
                        'end_frame': int(frame_numbers[-1]),
                        'num_frames': len(frame_numbers),
                    }
                    total_frames_3c = max(total_frames_3c, int(frame_numbers[-1]) + 1)
                
                if not verbose:
                    print(f"   ✅ Stage 3c: {len(person_buckets_3c)} persons")
            except Exception as e:
                logger.error(f"Error loading Stage 3c data: {e}")
                import traceback
                traceback.print_exc()
                return 1
        
        # Use 3c data as 3d data (no merging occurred)
        person_buckets_3d = person_buckets_3c
        person_frame_info_3d = person_frame_info_3c
        total_frames_3d = total_frames_3c
    
    # ==================== Load Merge Report ====================
    merge_info = []
    if merging_report_path.exists():
        try:
            with open(merging_report_path, 'r') as f:
                merge_report = json.load(f)
            merge_info = merge_report.get('merges', [])
        except Exception as e:
            if verbose:
                logger.warning(f"Could not load merge report: {e}")
    else:
        if verbose:
            logger.warning(f"Merge report not found: {merging_report_path}")
    
    # ==================== Generate WebP Animations ====================
    if verbose:
        print()
        logger.step("Generating WebP animations for both 3c and 3d...")
    
    webp_start = time.time()
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import imageio
        import cv2
        import base64
        
        webp_base64_dict_3c = {}  # Store base64 encoded WebPs for 3c
        webp_base64_dict_3d = {}  # Store base64 encoded WebPs for 3d
        
        # Generate WebPs for Stage 3c (only if dual-row mode enabled)
        if dual_row_mode:
            for person_id, crops in sorted(person_buckets_3c.items()):
                if crops is None or len(crops) == 0:
                    continue
                
                crops_to_use = crops[:50] if len(crops) > 50 else crops
                
                resized = []
                for crop in crops_to_use:
                    resized_crop = cv2.resize(crop, resize_to)
                    resized_crop = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)
                    resized.append(resized_crop)
                
                output_file = output_dir / f"person_3c_{person_id}.webp"
                imageio.mimsave(output_file, resized, format='WEBP', duration=webp_duration_ms, loop=0)
                
                with open(output_file, 'rb') as f:
                    webp_data = f.read()
                webp_base64_dict_3c[person_id] = base64.b64encode(webp_data).decode('utf-8')
                
                if verbose:
                    file_size_kb = output_file.stat().st_size / 1024
                    logger.info(f"3c Person {person_id}: {len(crops_to_use)}/{len(crops)} crops → {output_file.name} ({file_size_kb:.0f} KB)")
        else:
            if verbose:
                logger.info("⏭️  Skipping Stage 3c WebP generation (single-row mode)")
        
        # Generate WebPs for Stage 3d (limit to 50 crops per person)
        for person_id, crops in sorted(person_buckets_3d.items()):
            if crops is None or len(crops) == 0:
                continue
            
            crops_to_use = crops[:50] if len(crops) > 50 else crops
            
            resized = []
            for crop in crops_to_use:
                resized_crop = cv2.resize(crop, resize_to)
                resized_crop = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)
                resized.append(resized_crop)
            
            output_file = output_dir / f"person_3d_{person_id}.webp"
            imageio.mimsave(output_file, resized, format='WEBP', duration=webp_duration_ms, loop=0)
            
            with open(output_file, 'rb') as f:
                webp_data = f.read()
            webp_base64_dict_3d[person_id] = base64.b64encode(webp_data).decode('utf-8')
            
            if verbose:
                file_size_kb = output_file.stat().st_size / 1024
                logger.info(f"3d Person {person_id}: {len(crops_to_use)}/{len(crops)} crops → {output_file.name} ({file_size_kb:.0f} KB)")
        
        webp_time = time.time() - webp_start
        
        total_webps = len(webp_base64_dict_3c) + len(webp_base64_dict_3d)
        if verbose:
            logger.timing("WebP generation", webp_time)
        else:
            print(f"   ✅ Generated {total_webps} WebP animations ({len(webp_base64_dict_3c)} for 3c, {len(webp_base64_dict_3d)} for 3d) in {webp_time:.2f}s")
    
    except Exception as e:
        logger.error(f"Error during WebP generation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ==================== Create HTML Viewer ====================
    if verbose:
        print()
        logger.step("Creating dual-row HTML viewer...")
    
    html_file = output_dir / "viewer.html"
    try:
        create_dual_row_html_viewer(
            html_file, 
            person_buckets_3c, person_frame_info_3c, total_frames_3c, webp_base64_dict_3c,
            person_buckets_3d, person_frame_info_3d, total_frames_3d, webp_base64_dict_3d,
            merge_info,
            dual_row_mode,
            verbose
        )
        if verbose:
            logger.info(f"HTML viewer created: {html_file}")
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


def create_dual_row_html_viewer(
    html_file: Path,
    person_buckets_3c: dict, person_frame_info_3c: dict, total_frames_3c: int, webp_base64_dict_3c: dict,
    person_buckets_3d: dict, person_frame_info_3d: dict, total_frames_3d: int, webp_base64_dict_3d: dict,
    merge_info: list,
    dual_row_mode: bool,
    verbose: bool = False
):
    """
    Create dual-row HTML viewer comparing Stage 3c (before merge) vs Stage 3d (after merge).
    
    Args:
        html_file: Path to output HTML file
        person_buckets_3c: Dict of person_id -> list of crops (Stage 3c)
        person_frame_info_3c: Dict of person_id -> frame info (Stage 3c)
        total_frames_3c: Total frames (Stage 3c)
        webp_base64_dict_3c: Dict of person_id -> base64 WebP (Stage 3c)
        person_buckets_3d: Dict of person_id -> list of crops (Stage 3d)
        person_frame_info_3d: Dict of person_id -> frame info (Stage 3d)
        total_frames_3d: Total frames (Stage 3d)
        webp_base64_dict_3d: Dict of person_id -> base64 WebP (Stage 3d)
        merge_info: List of merge operations from merging_report.json
        verbose: Enable verbose output
    """
    
    # Create color mapping for merge groups
    merge_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']
    person_to_color = {}  # Maps person_id -> color
    
    if merge_info:
        for idx, merge in enumerate(merge_info):
            color = merge_colors[idx % len(merge_colors)]
            merged_ids = merge.get('merged_persons', [])
            result_id = merge.get('result_person_id')
            
            # Assign color to all persons in this merge group
            for pid in merged_ids:
                person_to_color[pid] = color
            if result_id:
                person_to_color[result_id] = color
    
    # Build 3c person cards (sorted chronologically) - Only if dual-row mode
    person_cards_3c = []
    if dual_row_mode:
        sorted_person_ids_3c = sorted(
            person_buckets_3c.keys(),
            key=lambda pid: (
                person_frame_info_3c.get(pid, {}).get('start_frame', 999999),  # Start frame ascending (chronological)
                -person_frame_info_3c.get(pid, {}).get('num_frames', 0)  # Duration descending (if same start)
            )
        )
        
        for person_id in sorted_person_ids_3c:
            # Build 3c cards (no radio buttons, for comparison only)
            num_crops = len(person_buckets_3c[person_id])
            frame_info = person_frame_info_3c.get(person_id, {})
            start_frame = frame_info.get('start_frame', 0)
            end_frame = frame_info.get('end_frame', 0)
            num_frames = frame_info.get('num_frames', 0)
            presence_pct = (num_frames / total_frames_3c * 100) if total_frames_3c > 0 else 0
            
            webp_base64 = webp_base64_dict_3c.get(person_id, '')
            data_uri = f"data:image/webp;base64,{webp_base64}"
            
            # Add merge color styling if this person is in a merge group
            merge_color = person_to_color.get(person_id, '')
            border_style = f"border-left: 5px solid {merge_color};" if merge_color else ""
            badge_html = f'<div class="merge-badge" style="background: {merge_color};"></div>' if merge_color else ""
            
            card_html = f"""
        <div class="person-card person-card-3c" data-person-id="{person_id}" style="{border_style}">
            {badge_html}
            <div class="person-header">
                <h3>ID #{person_id}</h3>
                <span class="person-info">Frames: {start_frame}-{end_frame} ({presence_pct:.1f}%)</span>
            </div>
            <div class="person-animation">
                <img src="{data_uri}" alt="Person {person_id}" class="webp-animation">
            </div>
        </div>
        """
            person_cards_3c.append(card_html)
    
    # Build 3d person cards (sorted by presence, with radio buttons)
    person_cards_3d = []
    sorted_person_ids_3d = sorted(
        person_buckets_3d.keys(),
        key=lambda pid: (
            -person_frame_info_3d.get(pid, {}).get('num_frames', 0) / total_frames_3d if total_frames_3d > 0 else 0,  # Presence % descending
            person_frame_info_3d.get(pid, {}).get('start_frame', 999999),  # Start frame ascending
            -person_frame_info_3d.get(pid, {}).get('num_frames', 0)  # Total frames descending
        )
    )
    
    for rank, person_id in enumerate(sorted_person_ids_3d, 1):
        num_crops = len(person_buckets_3d[person_id])
        
        # Get frame info
        frame_info = person_frame_info_3d.get(person_id, {})
        start_frame = frame_info.get('start_frame', 0)
        end_frame = frame_info.get('end_frame', 0)
        num_frames = frame_info.get('num_frames', 0)
        presence_pct = (num_frames / total_frames_3d * 100) if total_frames_3d > 0 else 0
        
        # Get base64 encoded WebP
        webp_base64 = webp_base64_dict_3d.get(person_id, '')
        data_uri = f"data:image/webp;base64,{webp_base64}"
        
        # Add merge color styling if this person is in a merge group
        merge_color = person_to_color.get(person_id, '')
        border_style = f"border-left: 5px solid {merge_color};" if merge_color else ""
        badge_html = f'<div class="merge-badge" style="background: {merge_color};"></div>' if merge_color else ""
        
        card_html = f"""
        <div class="person-card person-card-3d" data-person-id="{person_id}" data-rank="{rank}" onclick="selectPerson({person_id})" style="{border_style}">
            {badge_html}
            <div class="person-header">
                <h3>Person {rank} (ID #{person_id})</h3>
                <span class="person-info">Frames: {start_frame}-{end_frame} ({presence_pct:.1f}%)</span>
            </div>
            <div class="person-animation">
                <img src="{data_uri}" alt="Person {person_id}" class="webp-animation">
            </div>
            <div class="select-radio">
                <input type="radio" name="person-select" id="radio-{person_id}" value="{person_id}">
            </div>
        </div>
        """
        person_cards_3d.append(card_html)
    
    # Build merge info panel
    merge_panel_html = ""
    if merge_info:
        merge_lines = []
        for merge in merge_info:
            merged_ids = merge.get('merged_persons', [])
            result_id = merge.get('result_person_id', '?')
            if len(merged_ids) > 0:
                merge_lines.append(f"<li>3D Person {result_id} ← merged 3C Persons {merged_ids}</li>")
        
        if merge_lines:
            merge_panel_html = f"""
    <div class="merge-info-panel">
        <h2>🔗 Merge Operations ({len(merge_info)})</h2>
        <ul>
            {''.join(merge_lines)}
        </ul>
    </div>
    """
    
    # Full HTML template with dual rows
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Person Selection Viewer - Debug Mode</title>
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
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            backdrop-filter: blur(10px);
        }}
        
        .header h1 {{
            font-size: 2em;
            color: #4CAF50;
            margin-bottom: 10px;
            text-shadow: 0 2px 10px rgba(76, 175, 80, 0.3);
        }}
        
        .header p {{
            font-size: 1em;
            color: #aaa;
        }}
        
        .section-title {{
            font-size: 1.5em;
            color: #4CAF50;
            margin: 30px 20px 10px 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(76, 175, 80, 0.3);
        }}
        
        .gallery {{
            display: flex;
            gap: 15px;
            overflow-x: auto;
            padding: 20px;
            scroll-behavior: smooth;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        
        .person-card {{
            flex: 0 0 180px;
            min-width: 180px;
            background: rgba(255,255,255,0.05);
            border: 2px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            position: relative;
        }}
        
        .merge-badge {{
            position: absolute;
            top: 8px;
            right: 8px;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            border: 2px solid rgba(255,255,255,0.8);
            z-index: 10;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }}
        
        .person-card-3d {{
            cursor: pointer;
        }}
        
        .person-card-3d:hover {{
            border-color: #4CAF50;
            box-shadow: 0 6px 16px rgba(76, 175, 80, 0.3);
            transform: scale(1.05);
        }}
        
        .person-header {{
            padding: 10px;
            text-align: center;
        }}
        
        .person-header h3 {{
            font-size: 18px;
            font-weight: bold;
            color: #4CAF50;
            margin: 0 0 10px 0;
        }}
        
        .person-info {{
            font-size: 12px;
            color: #aaa;
            display: block;
            margin-bottom: 5px;
        }}
        
        .person-animation {{
            display: flex;
            justify-content: center;
            background: #1a1a1a;
            border-radius: 4px;
            padding: 10px;
            margin: 0 10px 10px 10px;
        }}
        
        .webp-animation {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        
        .select-radio {{
            padding: 15px;
            text-align: center;
            background: rgba(76, 175, 80, 0.2);
            border-top: 1px solid rgba(255,255,255,0.1);
            display: flex;
            justify-content: center;
            align-items: center;
        }}
        
        .select-radio input[type="radio"] {{
            appearance: none;
            -webkit-appearance: none;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: rgba(100, 100, 100, 0.3);
            border: 2px solid #2196F3;
            cursor: pointer;
            position: relative;
            transition: all 0.3s ease;
        }}
        
        .select-radio input[type="radio"]:hover {{
            background: rgba(33, 150, 243, 0.3);
            box-shadow: 0 0 15px rgba(33, 150, 243, 0.6);
            transform: scale(1.1);
        }}
        
        .select-radio input[type="radio"]:checked {{
            background: #4CAF50;
            border-color: #2196F3;
            box-shadow: 0 0 20px rgba(76, 175, 80, 0.8);
            animation: pulse 1.5s infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{
                box-shadow: 0 0 20px rgba(76, 175, 80, 0.8);
            }}
            50% {{
                box-shadow: 0 0 30px rgba(76, 175, 80, 1), 0 0 40px rgba(76, 175, 80, 0.6);
            }}
        }}
        
        .selected {{
            border-color: #FFD700 !important;
            border-width: 3px !important;
            box-shadow: 0 0 20px rgba(255, 215, 0, 0.5) !important;
        }}
        
        .merge-info-panel {{
            background: rgba(255, 152, 0, 0.1);
            border: 2px solid rgba(255, 152, 0, 0.5);
            border-radius: 12px;
            padding: 20px;
            margin: 30px 20px;
        }}
        
        .merge-info-panel h2 {{
            color: #FF9800;
            font-size: 1.3em;
            margin-bottom: 15px;
        }}
        
        .merge-info-panel ul {{
            list-style: none;
            padding: 0;
        }}
        
        .merge-info-panel li {{
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 152, 0, 0.2);
            color: #FFA726;
            font-size: 1.1em;
        }}
        
        .merge-info-panel li:last-child {{
            border-bottom: none;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Person Selection Viewer{' - Debug Mode' if dual_row_mode else ''}</h1>
        <p>{'Comparing Stage 3c (before merge) vs Stage 3d (after merge)' if dual_row_mode else 'Select person for pose estimation'}</p>
    </div>
    
    {f'''<h2 class="section-title">Stage 3C Outputs ({len(person_buckets_3c)} persons) - Before Merge</h2>
    <div class="gallery">
        {''.join(person_cards_3c)}
    </div>
    
    {merge_panel_html}
    ''' if dual_row_mode else ''}
    
    <h2 class="section-title">{'Stage 3D Outputs' if dual_row_mode else 'Stage 3C Outputs'} ({len(person_buckets_3d)} persons){' - After Merge' if dual_row_mode else ''}</h2>
    <div class="gallery">
        {''.join(person_cards_3d)}
    </div>
    
    <script>
        let selectedPersonId = null;
        
        function selectPerson(personId) {{
            // Remove previous selection
            document.querySelectorAll('.person-card-3d').forEach(card => {{
                card.classList.remove('selected');
            }});
            
            // Mark new selection
            const card = document.querySelector(`.person-card-3d[data-person-id="${{personId}}"]`);
            if (card) {{
                card.classList.add('selected');
            }}
            
            // Check radio button
            const radio = document.getElementById(`radio-${{personId}}`);
            if (radio) radio.checked = true;
            
            // Update selection
            selectedPersonId = personId;
            
            // Log selection (for automation/scripting)
            console.log(`SELECTED_PERSON: ${{personId}}`);
            
            // Save selection to localStorage
            localStorage.setItem('selected_person_id', personId);
        }}
        
        // Default select first 3d person (highest ranked)
        window.addEventListener('load', () => {{
            const firstCard = document.querySelector('.person-card-3d');
            if (firstCard) {{
                const firstPersonId = parseInt(firstCard.getAttribute('data-person-id'));
                selectPerson(firstPersonId);
            }}
        }});
    </script>
</body>
</html>"""
    
    # Write HTML file
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    if verbose:
        print(f"   Created dual-row HTML viewer: {len(person_buckets_3c)} 3c persons, {len(person_buckets_3d)} 3d persons")


if __name__ == '__main__':
    exit(main())
