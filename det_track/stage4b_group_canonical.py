#!/usr/bin/env python3
"""
Stage 4b: Canonical Grouping (Optional)

Groups tracklets into canonical persons using geometric heuristics.
Alternative to ReID recovery for faster processing.

Usage:
    python stage4b_group_canonical.py --config configs/pipeline_config.yaml
"""

import argparse
import yaml
import numpy as np
import json
import time
import re
from pathlib import Path


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


def load_config(config_path):
    """Load and resolve YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return resolve_path_variables(config)


def compute_tracklet_features(tracklet):
    """Compute geometric features for grouping"""
    frames = tracklet['frame_numbers']
    bboxes = tracklet['bboxes']
    
    # Centers
    centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2
    mean_center = centers.mean(axis=0)
    
    # Areas
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    mean_area = areas.mean()
    
    # Temporal
    start_frame = int(frames[0])
    end_frame = int(frames[-1])
    
    # First/last bbox
    first_bbox = bboxes[0]
    last_bbox = bboxes[-1]
    
    return {
        'start_frame': start_frame,
        'end_frame': end_frame,
        'mean_center': mean_center,
        'mean_area': mean_area,
        'first_bbox': first_bbox,
        'last_bbox': last_bbox
    }


def can_merge_heuristic(feat1, feat2, criteria):
    """Check if two tracklets can be merged based on heuristics"""
    max_temporal_gap = criteria['max_temporal_gap']
    max_spatial_distance = criteria['max_spatial_distance']
    area_ratio_range = criteria['area_ratio_range']
    
    # Temporal order (feat1 should end before feat2 starts)
    if feat1['end_frame'] >= feat2['start_frame']:
        return False
    
    gap = feat2['start_frame'] - feat1['end_frame']
    if gap > max_temporal_gap:
        return False
    
    # Spatial proximity
    last_center_1 = (feat1['last_bbox'][:2] + feat1['last_bbox'][2:]) / 2
    first_center_2 = (feat2['first_bbox'][:2] + feat2['first_bbox'][2:]) / 2
    distance = np.linalg.norm(last_center_1 - first_center_2)
    
    if distance > max_spatial_distance:
        return False
    
    # Size consistency
    area_ratio = feat2['mean_area'] / (feat1['mean_area'] + 1e-8)
    if area_ratio < area_ratio_range[0] or area_ratio > area_ratio_range[1]:
        return False
    
    return True


def group_tracklets_heuristic(tracklets, criteria):
    """
    Group tracklets using greedy heuristic merging.
    Merges tracklets that satisfy temporal/spatial/size constraints.
    """
    # Compute features
    features = [compute_tracklet_features(t) for t in tracklets]
    
    # Sort by start frame
    sorted_indices = sorted(range(len(tracklets)), key=lambda i: features[i]['start_frame'])
    
    # Build groups greedily
    groups = []
    assigned = set()
    
    for i in sorted_indices:
        if i in assigned:
            continue
        
        # Start new group
        current_group = [i]
        assigned.add(i)
        
        # Try to extend group with later tracklets
        for j in sorted_indices:
            if j in assigned:
                continue
            
            # Check if j can merge with any member of current group
            can_add = False
            for member_idx in current_group:
                if can_merge_heuristic(features[member_idx], features[j], criteria):
                    can_add = True
                    break
            
            if can_add:
                current_group.append(j)
                assigned.add(j)
        
        groups.append(current_group)
    
    return groups


def merge_group(tracklets, group_indices):
    """Merge tracklets in a group into one canonical tracklet"""
    member_tracklets = [tracklets[i] for i in group_indices]
    
    # Sort by start frame
    member_tracklets.sort(key=lambda t: t['frame_numbers'][0])
    
    # Concatenate
    merged_frames = np.concatenate([t['frame_numbers'] for t in member_tracklets])
    merged_bboxes = np.concatenate([t['bboxes'] for t in member_tracklets])
    merged_confs = np.concatenate([t['confidences'] for t in member_tracklets])
    
    # Canonical ID is the first tracklet ID in the group
    canonical_id = member_tracklets[0]['tracklet_id']
    original_ids = [tracklets[i]['tracklet_id'] for i in group_indices]
    
    return {
        'person_id': canonical_id,
        'frame_numbers': merged_frames,
        'bboxes': merged_bboxes,
        'confidences': merged_confs,
        'original_tracklet_ids': original_ids,
        'num_tracklets_merged': len(original_ids)
    }


def run_canonical_grouping(config):
    """Run Stage 4b: Canonical Grouping"""
    
    stage_config = config['stage4b_group_canonical']
    verbose = stage_config.get('advanced', {}).get('verbose', False)
    
    # Extract configuration
    input_config = stage_config['input']
    output_config = stage_config['output']
    grouping_config = stage_config['grouping']
    
    # Determine input source
    stage4a_enabled = config['pipeline']['stages']['stage4a_reid_recovery']
    
    if stage4a_enabled:
        input_file = input_config['recovered_tracklets_file']
        print(f"📂 Using ReID-recovered tracklets")
    else:
        input_file = input_config['tracklets_raw_file']
        print(f"📂 Using raw tracklets (Stage 4a disabled)")
    
    canonical_file = output_config['canonical_persons_file']
    grouping_log_file = output_config['grouping_log_file']
    
    # Print header
    print(f"\n{'='*70}")
    print(f"📍 STAGE 4b: CANONICAL GROUPING")
    print(f"{'='*70}\n")
    
    # Load tracklets
    print(f"📂 Loading tracklets: {input_file}")
    data = np.load(input_file, allow_pickle=True)
    tracklets = list(data['tracklets'])
    print(f"  ✅ Loaded {len(tracklets)} tracklets")
    
    # Group tracklets
    print(f"\n🔗 Grouping tracklets (method: {grouping_config['method']})...")
    t_start = time.time()
    
    if grouping_config['method'] == 'heuristic':
        criteria = grouping_config['heuristic_criteria']
        groups = group_tracklets_heuristic(tracklets, criteria)
    else:
        raise ValueError(f"Unknown grouping method: {grouping_config['method']}")
    
    t_end = time.time()
    print(f"  ✅ Created {len(groups)} canonical persons ({t_end - t_start:.2f}s)")
    
    # Merge groups into canonical persons
    print(f"\n🔀 Merging tracklets into canonical persons...")
    canonical_persons = []
    grouping_log = []
    
    for group_idx, group_indices in enumerate(groups):
        canonical = merge_group(tracklets, group_indices)
        canonical_persons.append(canonical)
        
        grouping_log.append({
            'canonical_id': int(canonical['person_id']),
            'original_tracklet_ids': [int(tid) for tid in canonical['original_tracklet_ids']],
            'num_merged': canonical['num_tracklets_merged'],
            'start_frame': int(canonical['frame_numbers'][0]),
            'end_frame': int(canonical['frame_numbers'][-1]),
            'total_frames': len(canonical['frame_numbers'])
        })
    
    print(f"  ✅ Created {len(canonical_persons)} canonical persons")
    
    if verbose:
        print(f"\n  Sample canonical persons:")
        for log in grouping_log[:5]:
            print(f"    Person {log['canonical_id']}: "
                  f"{log['num_merged']} tracklets, "
                  f"{log['total_frames']} frames, "
                  f"range [{log['start_frame']}, {log['end_frame']}]")
        if len(grouping_log) > 5:
            print(f"    ... and {len(grouping_log) - 5} more")
    
    # Save canonical persons
    output_path = Path(canonical_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, persons=np.array(canonical_persons, dtype=object))
    print(f"  ✅ Saved: {output_path}")
    
    # Save log
    log_path = Path(grouping_log_file)
    with open(log_path, 'w') as f:
        json.dump(grouping_log, f, indent=2)
    print(f"  ✅ Saved grouping log: {log_path}")
    
    print(f"\n✅ Canonical grouping complete!")
    print(f"  Input tracklets: {len(tracklets)}")
    print(f"  Canonical persons: {len(canonical_persons)}")


def main():
    parser = argparse.ArgumentParser(description='Stage 4b: Canonical Grouping')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Check if stage is enabled
    if not config['pipeline']['stages']['stage4b_group_canonical']:
        print("⏭️  Stage 4b is disabled in config")
        return
    
    # Run grouping
    run_canonical_grouping(config)
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
