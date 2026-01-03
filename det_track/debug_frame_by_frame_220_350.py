#!/usr/bin/env python3
"""
Debug: Detailed frame-by-frame comparison for frames 220-350

Compare which persons are visible in:
1. Canonical persons NPZ (ground truth)
2. Stage 6 filtering (top 16 only)

Identify which person disappears.
"""

import argparse
import numpy as np
import cv2
import yaml
import re
from pathlib import Path


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
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    video_file = config.get('global', {}).get('video_file', '')
    if video_file:
        import os
        video_name = os.path.splitext(video_file)[0]
        config['global']['current_video'] = video_name
    
    return resolve_path_variables(config)


def main():
    parser = argparse.ArgumentParser(description='Frame-by-frame comparison 220-350')
    parser.add_argument('--config', required=True, help='Pipeline config file')
    parser.add_argument('--start', type=int, default=220, help='Start frame')
    parser.add_argument('--end', type=int, default=350, help='End frame')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    canonical_persons_file = config['stage6_create_output_video']['input']['canonical_persons_file']
    video_path = config['stage6_create_output_video']['input']['video_file']
    
    # Get video FPS
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Load canonical persons
    persons_data = np.load(canonical_persons_file, allow_pickle=True)
    persons = persons_data['persons'].tolist()
    
    # Get top 16 (Stage 6 filtering)
    min_frames = int(5.0 * video_fps)
    top_16_ids = set()
    all_persons = []
    for p in persons:
        frames = p['frame_numbers']
        if len(frames) >= min_frames:
            top_16_ids.add(p['person_id'])
        all_persons.append({
            'id': p['person_id'],
            'frames': p['frame_numbers'],
            'count': len(p['frame_numbers']),
            'included': len(p['frame_numbers']) >= min_frames
        })
    
    print(f"Frame range: {args.start}-{args.end}")
    print(f"Video FPS: {video_fps}")
    print(f"Min frames threshold: {min_frames}")
    print(f"Top 16 person IDs: {sorted(top_16_ids)}\n")
    
    print(f"{'Frame':<8} {'ALL Persons':<40} {'Stage 6 (Top 16)':<40}")
    print("=" * 88)
    
    for frame_idx in range(args.start, args.end + 1):
        # Find all persons at this frame
        all_at_frame = []
        filtered_at_frame = []
        
        for p in all_persons:
            if frame_idx in p['frames']:
                all_at_frame.append(p['id'])
                if p['id'] in top_16_ids:
                    filtered_at_frame.append(p['id'])
        
        # Find disappeared persons
        disappeared = set(all_at_frame) - set(filtered_at_frame)
        
        all_str = f"{len(all_at_frame):2d}: {sorted(all_at_frame)}"
        filt_str = f"{len(filtered_at_frame):2d}: {sorted(filtered_at_frame)}"
        
        if disappeared:
            print(f"{frame_idx:<8} {all_str:<40} {filt_str:<40} ⚠️ Missing: {sorted(disappeared)}")
        else:
            print(f"{frame_idx:<8} {all_str:<40} {filt_str:<40}")


if __name__ == '__main__':
    main()
