#!/usr/bin/env python3
"""
Debug: Trace P14 and P29 - why weren't they merged?

P14 (tracklet 14): 250 frames
P29 (tracklet 29): 425 frames

Around frame 365-370, they appear to be the same person.
Why did Stage 4b not merge them?
"""

import argparse
import numpy as np
import yaml
import re
import json
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
    parser = argparse.ArgumentParser(description='Trace P14/P29 merge issue')
    parser.add_argument('--config', required=True, help='Pipeline config file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    tracklets_file = config['stage2_track']['output']['tracklets_file']
    canonical_persons_file = config['stage4b_group_canonical']['output']['canonical_persons_file']
    grouping_log_file = config['stage4b_group_canonical']['output']['grouping_log_file']
    reid_candidates_file = config['stage3_analyze']['output']['candidates_file']
    
    print(f"{'='*100}")
    print(f"TRACING P14 and P29: Why weren't they merged?")
    print(f"{'='*100}\n")
    
    # Load tracklets
    print(f"1. LOAD RAW TRACKLETS")
    print(f"{'='*100}")
    tracklets_data = np.load(tracklets_file, allow_pickle=True)
    tracklets = tracklets_data['tracklets'].tolist()
    
    tracklet_14 = None
    tracklet_29 = None
    
    for t in tracklets:
        if t['tracklet_id'] == 14:
            tracklet_14 = t
        elif t['tracklet_id'] == 29:
            tracklet_29 = t
    
    if tracklet_14:
        frames_14 = tracklet_14['frame_numbers']
        print(f"Tracklet 14: {len(frames_14)} frames, range [{frames_14[0]}, {frames_14[-1]}]")
        print(f"  First bbox: {tracklet_14['bboxes'][0]}")
        print(f"  Last bbox: {tracklet_14['bboxes'][-1]}")
    
    if tracklet_29:
        frames_29 = tracklet_29['frame_numbers']
        print(f"Tracklet 29: {len(frames_29)} frames, range [{frames_29[0]}, {frames_29[-1]}]")
        print(f"  First bbox: {tracklet_29['bboxes'][0]}")
        print(f"  Last bbox: {tracklet_29['bboxes'][-1]}")
    
    # Check overlap
    if tracklet_14 and tracklet_29:
        overlap = np.intersect1d(frames_14, frames_29)
        print(f"\nFrame overlap: {len(overlap)} frames")
        if len(overlap) > 0:
            print(f"  Overlapping frames: {overlap}")
    
    # Load grouping log
    print(f"\n{'='*100}")
    print(f"2. CHECK GROUPING LOG")
    print(f"{'='*100}")
    
    with open(grouping_log_file, 'r') as f:
        grouping_log = json.load(f)
    
    p14_person = None
    p29_person = None
    
    for entry in grouping_log:
        if 14 in entry['original_tracklet_ids']:
            p14_person = entry
        if 29 in entry['original_tracklet_ids']:
            p29_person = entry
    
    if p14_person:
        print(f"P{p14_person['canonical_id']}: tracklets {p14_person['original_tracklet_ids']}, "
              f"{p14_person['num_merged']} merged")
    
    if p29_person:
        print(f"P{p29_person['canonical_id']}: tracklets {p29_person['original_tracklet_ids']}, "
              f"{p29_person['num_merged']} merged")
    
    # Check if they were candidates for merging
    print(f"\n{'='*100}")
    print(f"3. CHECK REID MERGE CANDIDATES")
    print(f"{'='*100}")
    
    if Path(reid_candidates_file).exists():
        with open(reid_candidates_file, 'r') as f:
            candidates = json.load(f)
        
        found_pair = False
        for c in candidates:
            t1, t2 = c['tracklet_1'], c['tracklet_2']
            if (t1 == 14 and t2 == 29) or (t1 == 29 and t2 == 14):
                found_pair = True
                print(f"Candidate pair: Tracklet {t1} + Tracklet {t2}")
                print(f"  Temporal gap: {c['temporal_gap']} frames")
                print(f"  Spatial distance: {c['spatial_distance']:.2f}")
                print(f"  Area ratio: {c['area_ratio']:.2f}")
                print(f"  Likely reason for rejection: {c.get('reason', 'not specified')}")
        
        if not found_pair:
            print(f"⚠️  Tracklets 14 and 29 were NOT even considered as merge candidates!")
            print(f"This means Stage 3 analysis didn't identify them as potential matches.")
            print(f"\nSearching for any candidates involving these tracklets:")
            for c in candidates:
                if c['tracklet_1'] == 14 or c['tracklet_2'] == 14:
                    print(f"  Tracklet 14: {c['tracklet_1']} + {c['tracklet_2']}")
                if c['tracklet_1'] == 29 or c['tracklet_2'] == 29:
                    print(f"  Tracklet 29: {c['tracklet_1']} + {c['tracklet_2']}")
    
    # Load canonical persons to see final result
    print(f"\n{'='*100}")
    print(f"4. FINAL CANONICAL PERSONS")
    print(f"{'='*100}")
    
    persons_data = np.load(canonical_persons_file, allow_pickle=True)
    persons = persons_data['persons'].tolist()
    
    for p in persons:
        if 14 in p.get('original_tracklet_ids', []):
            frames_p = p['frame_numbers']
            print(f"P{p['person_id']}: tracklets {p['original_tracklet_ids']}, "
                  f"{len(frames_p)} frames [{frames_p[0]}, {frames_p[-1]}]")
        if 29 in p.get('original_tracklet_ids', []):
            frames_p = p['frame_numbers']
            print(f"P{p['person_id']}: tracklets {p['original_tracklet_ids']}, "
                  f"{len(frames_p)} frames [{frames_p[0]}, {frames_p[-1]}]")
    
    # Summary
    print(f"\n{'='*100}")
    print(f"ANALYSIS")
    print(f"{'='*100}")
    print(f"✓ Tracklets 14 and 29 are SEPARATE persons in the output")
    print(f"✓ They should have been merged if they're temporally close and spatially similar")
    print(f"⚠️  The fact that P14 'becomes' P29 suggests they should be the SAME person")
    print(f"\nPossible causes:")
    print(f"  1. No temporal overlap (one ends before the other starts)")
    print(f"  2. Too far apart spatially (> area/distance threshold)")
    print(f"  3. Not identified as merge candidates by Stage 3")


if __name__ == '__main__':
    main()
