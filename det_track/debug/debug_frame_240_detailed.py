#!/usr/bin/env python3
"""
Debug: Extract and compare exact plotting data for frame 240

Compare:
1. What Stage 6 is drawing (top_persons_visualization.mp4)
2. What tracklets data shows (all raw tracklets)
3. What canonical persons show (all 46 persons)
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
    parser = argparse.ArgumentParser(description='Compare frame 240 data across all sources')
    parser.add_argument('--config', required=True, help='Pipeline config file')
    parser.add_argument('--frame', type=int, default=240, help='Frame to analyze')
    args = parser.parse_args()
    
    config = load_config(args.config)
    frame_idx = args.frame
    
    # Get file paths
    tracklets_file = config['stage2_track']['output']['tracklets_file']
    canonical_persons_file = config['stage6_create_output_video']['input']['canonical_persons_file']
    video_path = config['stage6_create_output_video']['input']['video_file']
    
    # Get video FPS
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    min_frames = int(5.0 * video_fps)
    
    print(f"{'='*100}")
    print(f"FRAME {frame_idx} ANALYSIS")
    print(f"{'='*100}")
    print(f"Video FPS: {video_fps}")
    print(f"Min frames for Stage 6 filter: {min_frames}\n")
    
    # Load tracklets
    print(f"{'='*100}")
    print(f"1. RAW TRACKLETS (from tracklets_raw.npz)")
    print(f"{'='*100}")
    tracklets_data = np.load(tracklets_file, allow_pickle=True)
    tracklets = tracklets_data['tracklets'].tolist()
    
    tracklets_at_frame = []
    for t in tracklets:
        if frame_idx in t['frame_numbers']:
            frame_pos = np.where(t['frame_numbers'] == frame_idx)[0][0]
            bbox = t['bboxes'][frame_pos]
            conf = t['confidences'][frame_pos]
            tracklets_at_frame.append({
                'tracklet_id': t['tracklet_id'],
                'bbox': bbox,
                'confidence': conf
            })
    
    print(f"Total tracklets at frame {frame_idx}: {len(tracklets_at_frame)}\n")
    for t in sorted(tracklets_at_frame, key=lambda x: x['tracklet_id']):
        print(f"  Tracklet {t['tracklet_id']:3d}: bbox={t['bbox']}, conf={t['confidence']:.3f}")
    
    # Load canonical persons
    print(f"\n{'='*100}")
    print(f"2. CANONICAL PERSONS (from canonical_persons.npz)")
    print(f"{'='*100}")
    persons_data = np.load(canonical_persons_file, allow_pickle=True)
    persons = persons_data['persons'].tolist()
    
    persons_at_frame = []
    filtered_at_frame = []
    for p in persons:
        if frame_idx in p['frame_numbers']:
            frame_pos = np.where(p['frame_numbers'] == frame_idx)[0][0]
            bbox = p['bboxes'][frame_pos]
            conf = p['confidences'][frame_pos]
            person_info = {
                'person_id': p['person_id'],
                'bbox': bbox,
                'confidence': conf,
                'tracklet_ids': p.get('original_tracklet_ids', []),
                'frame_count': len(p['frame_numbers']),
                'included_in_stage6': len(p['frame_numbers']) >= min_frames
            }
            persons_at_frame.append(person_info)
            if person_info['included_in_stage6']:
                filtered_at_frame.append(person_info)
    
    print(f"Total canonical persons at frame {frame_idx}: {len(persons_at_frame)}")
    print(f"Stage 6 filtered (>={min_frames} frames): {len(filtered_at_frame)}\n")
    
    print(f"ALL CANONICAL PERSONS AT FRAME {frame_idx}:")
    for p in sorted(persons_at_frame, key=lambda x: x['person_id']):
        status = "✓ INCLUDED" if p['included_in_stage6'] else "✗ FILTERED"
        print(f"  Person {p['person_id']:3d} {status:<15} bbox={p['bbox']}, conf={p['confidence']:.3f}, "
              f"tracklets={p['tracklet_ids']}, total_frames={p['frame_count']}")
    
    # Comparison
    print(f"\n{'='*100}")
    print(f"3. COMPARISON: Which persons are at frame {frame_idx}?")
    print(f"{'='*100}")
    
    all_person_ids = set(p['person_id'] for p in persons_at_frame)
    included_person_ids = set(p['person_id'] for p in filtered_at_frame)
    filtered_out_ids = all_person_ids - included_person_ids
    
    if filtered_out_ids:
        print(f"\n⚠️  FILTERED OUT at frame {frame_idx}:")
        for p in sorted(persons_at_frame):
            if p['person_id'] in filtered_out_ids:
                print(f"  Person {p['person_id']:3d}: {p['frame_count']} total frames (< {min_frames}), "
                      f"tracklets {p['tracklet_ids']}, bbox={p['bbox']}")
    else:
        print(f"\n✓ All persons at this frame are included in Stage 6")
    
    print(f"\n✓ INCLUDED at frame {frame_idx}:")
    for p in sorted(filtered_at_frame, key=lambda x: x['person_id']):
        print(f"  Person {p['person_id']:3d}: {p['frame_count']} total frames (>= {min_frames}), "
              f"tracklets {p['tracklet_ids']}, bbox={p['bbox']}")
    
    # Find untracked people
    tracklet_ids_in_frame = set(t['tracklet_id'] for t in tracklets_at_frame)
    person_tracklet_ids = set()
    for p in persons_at_frame:
        for tid in p['tracklet_ids']:
            person_tracklet_ids.add(tid)
    
    untracked_tracklets = tracklet_ids_in_frame - person_tracklet_ids
    if untracked_tracklets:
        print(f"\n⚠️  UNTRACKED TRACKLETS (in tracklets but not in persons):")
        for t in tracklets_at_frame:
            if t['tracklet_id'] in untracked_tracklets:
                print(f"  Tracklet {t['tracklet_id']:3d}: bbox={t['bbox']}, conf={t['confidence']:.3f}")


if __name__ == '__main__':
    main()
