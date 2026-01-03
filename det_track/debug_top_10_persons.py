#!/usr/bin/env python3
"""
Debug: Show top 10 persons loaded by Stage 6
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
    parser = argparse.ArgumentParser(description='Show what Stage 6 loads as top 10')
    parser.add_argument('--config', required=True, help='Pipeline config file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    canonical_persons_file = config['stage6_create_output_video']['input']['canonical_persons_file']
    video_path = config['stage6_create_output_video']['input']['video_file']
    
    # Get video FPS
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    min_duration = 5.0
    min_frames = int(min_duration * video_fps)
    
    print(f"Video FPS: {video_fps}")
    print(f"Min duration: {min_duration}s = {min_frames} frames\n")
    
    # Load canonical persons
    persons_data = np.load(canonical_persons_file, allow_pickle=True)
    persons = persons_data['persons'].tolist()
    
    # Filter and sort like Stage 6 does
    persons_with_duration = []
    for p in persons:
        frames = p['frame_numbers']
        frame_count = len(frames)
        if frame_count >= min_frames:
            persons_with_duration.append((p, frame_count))
    
    persons_with_duration.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Persons meeting min duration criteria: {len(persons_with_duration)}")
    print(f"Top 10 selected:\n")
    
    print(f"{'Rank':<6} {'Person ID':<12} {'Frames':<10} {'Duration(s)':<12} {'Tracklets':<20}")
    print("=" * 70)
    
    top_10 = persons_with_duration[:10]
    top_10_ids = set()
    
    for rank, (person, duration) in enumerate(top_10, 1):
        p_id = person['person_id']
        tracklet_ids = person.get('original_tracklet_ids', [])
        dur_sec = duration / video_fps
        top_10_ids.add(p_id)
        print(f"{rank:<6} P{p_id:<11} {duration:<10} {dur_sec:<12.2f} {str(tracklet_ids):<20}")
    
    # Find persons 11+
    if len(persons_with_duration) > 10:
        print(f"\n\nPersons 11+ (NOT in top 10):\n")
        print(f"{'Rank':<6} {'Person ID':<12} {'Frames':<10} {'Duration(s)':<12} {'Tracklets':<20}")
        print("=" * 70)
        
        for rank, (person, duration) in enumerate(persons_with_duration[10:20], 11):
            p_id = person['person_id']
            tracklet_ids = person.get('original_tracklet_ids', [])
            dur_sec = duration / video_fps
            print(f"{rank:<6} P{p_id:<11} {duration:<10} {dur_sec:<12.2f} {str(tracklet_ids):<20}")


if __name__ == '__main__':
    main()
