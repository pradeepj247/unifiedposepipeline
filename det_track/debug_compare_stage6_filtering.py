#!/usr/bin/env python3
"""
Debug: Compare Stage 6 filtering vs raw canonical persons

Shows which persons are being filtered out and why.
"""

import argparse
import numpy as np
import cv2
import json
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
    """Load and resolve YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    video_file = config.get('global', {}).get('video_file', '')
    if video_file:
        import os
        video_name = os.path.splitext(video_file)[0]
        config['global']['current_video'] = video_name
    
    return resolve_path_variables(config)


def main():
    parser = argparse.ArgumentParser(description='Debug Stage 6 filtering')
    parser.add_argument('--config', required=True, help='Pipeline config file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Get paths
    video_path = config['stage6_create_output_video']['input']['video_file']
    canonical_persons_file = config['stage6_create_output_video']['input']['canonical_persons_file']
    ranking_report_file = config['stage5_rank']['output']['ranking_report_file']
    
    print(f"Video: {video_path}")
    print(f"Canonical persons: {canonical_persons_file}")
    print(f"Ranking report: {ranking_report_file}\n")
    
    # Get video FPS
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Video FPS: {video_fps}")
    
    # Get min duration
    min_seconds = config['stage6_create_output_video']['visualization']['min_duration_seconds']
    min_frames = int(min_seconds * video_fps)
    print(f"Min duration: {min_seconds}s = {min_frames} frames @ {video_fps} fps\n")
    
    # Load canonical persons
    persons_data = np.load(canonical_persons_file, allow_pickle=True)
    persons = persons_data['persons'].tolist()
    
    print(f"Total canonical persons: {len(persons)}\n")
    print(f"{'Person ID':<12} {'Tracklets':<30} {'Appearances':<12} {'Duration(s)':<12} {'Status':<12}")
    print("=" * 80)
    
    included = []
    excluded = []
    
    for p in persons:
        person_id = p['person_id']
        tracklet_ids = p.get('original_tracklet_ids', [])
        frames = p['frame_numbers']
        frame_count = len(frames)
        duration_sec = frame_count / video_fps
        
        status = "✓ INCLUDED" if frame_count >= min_frames else "✗ FILTERED"
        
        if frame_count >= min_frames:
            included.append((person_id, frame_count, duration_sec, tracklet_ids))
        else:
            excluded.append((person_id, frame_count, duration_sec, tracklet_ids))
        
        print(f"P{person_id:<11} {str(tracklet_ids):<30} {frame_count:<12} {duration_sec:<12.2f} {status:<12}")
    
    print(f"\n{'='*80}")
    print(f"SUMMARY:")
    print(f"  ✓ Included: {len(included)} persons")
    if included:
        print(f"    IDs: {sorted([p[0] for p in included])}")
    
    print(f"  ✗ Filtered: {len(excluded)} persons")
    if excluded:
        print(f"    IDs: {sorted([p[0] for p in excluded])}")
        print(f"    Details:")
        for p_id, count, dur, tids in sorted(excluded, key=lambda x: x[1], reverse=True):
            print(f"      Person {p_id}: {count} frames ({dur:.2f}s), tracklets {tids}")
    
    # Load ranking report to compare
    print(f"\n{'='*80}")
    print(f"RANKING REPORT (from Stage 5):")
    if Path(ranking_report_file).exists():
        with open(ranking_report_file, 'r') as f:
            ranking = json.load(f)
        for rank_idx, entry in enumerate(ranking[:15], 1):
            p_id = entry['person_id']
            frames = entry['frame_count']
            dur = frames / video_fps
            included_status = "✓" if p_id in [p[0] for p in included] else "✗"
            print(f"  {included_status} Rank {rank_idx}: Person {p_id}, "
                  f"{frames} frames ({dur:.2f}s), tracklets {entry['tracklet_ids']}")


if __name__ == '__main__':
    main()
