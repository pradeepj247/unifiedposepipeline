#!/usr/bin/env python3
"""
Stage 3b: Enhanced Canonical Grouping

Groups tracklets into canonical persons using 5 merge checks:
  1. Temporal gap (existing)
  2. Spatial proximity (existing)
  3. Area ratio (existing)
  4. Motion direction alignment (NEW)
  5. Movement smoothness/jitter (NEW)

Loads pre-computed statistics from Stage 3a (NO recomputation).

Usage:
    python stage3b_group_canonical.py --config configs/pipeline_config.yaml
"""

import argparse
import yaml
import numpy as np
import json
import time
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import PipelineLogger
from datetime import datetime, timezone


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
    
    # Auto-extract current_video from video_file
    video_file = config.get('global', {}).get('video_file', '')
    if video_file:
        import os
        video_name = os.path.splitext(video_file)[0]
        config['global']['current_video'] = video_name
    
    return resolve_path_variables(config)


def can_merge_enhanced(stat1, stat2, criteria):
    """
    Enhanced merge check with 5 criteria:
      1. Temporal gap
      2. Spatial proximity
      3. Area ratio
      4. Motion direction alignment (NEW)
      5. Movement smoothness/jitter (NEW)
    """
    # Unpack criteria
    max_temporal_gap = criteria['max_temporal_gap']
    max_spatial_distance = criteria['max_spatial_distance']
    area_ratio_range = criteria['area_ratio_range']
    min_motion_alignment = criteria.get('min_motion_alignment', 0.6)
    max_jitter_difference = criteria.get('max_jitter_difference', 40.0)
    
    # Check 1: Temporal order (stat1 should end before stat2 starts)
    if stat1['end_frame'] >= stat2['start_frame']:
        return False
    
    gap = stat2['start_frame'] - stat1['end_frame']
    if gap > max_temporal_gap:
        return False
    
    # Check 2: Spatial proximity (last bbox of stat1 vs first bbox of stat2)
    last_bbox_1 = np.array(stat1['last_bbox'])
    first_bbox_2 = np.array(stat2['first_bbox'])
    
    last_center_1 = (last_bbox_1[:2] + last_bbox_1[2:]) / 2
    first_center_2 = (first_bbox_2[:2] + first_bbox_2[2:]) / 2
    distance = np.linalg.norm(last_center_1 - first_center_2)
    
    if distance > max_spatial_distance:
        return False
    
    # Check 3: Size consistency
    area_ratio = stat2['mean_area'] / (stat1['mean_area'] + 1e-8)
    if area_ratio < area_ratio_range[0] or area_ratio > area_ratio_range[1]:
        return False
    
    # Check 4: Motion direction alignment (NEW)
    # Compute cosine similarity between velocity vectors
    vel1 = np.array(stat1['mean_velocity'])
    vel2 = np.array(stat2['mean_velocity'])
    
    # Only check if both tracklets have sufficient motion
    vel1_mag = np.linalg.norm(vel1)
    vel2_mag = np.linalg.norm(vel2)
    
    if vel1_mag > 1.0 and vel2_mag > 1.0:  # Both moving
        cosine_sim = np.dot(vel1, vel2) / (vel1_mag * vel2_mag + 1e-8)
        if cosine_sim < min_motion_alignment:
            return False
    
    # Check 5: Movement smoothness/jitter (NEW)
    # Tracklets of same person should have similar jitter levels
    jitter_diff = abs(stat1['center_jitter'] - stat2['center_jitter'])
    if jitter_diff > max_jitter_difference:
        return False
    
    # Passed all checks
    return True


def group_tracklets_enhanced(tracklets, stats, criteria):
    """
    Group tracklets using enhanced heuristic merging.
    Loads pre-computed stats instead of recomputing.
    """
    # Sort by start frame
    sorted_indices = sorted(range(len(tracklets)), key=lambda i: stats[i]['start_frame'])
    
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
                if can_merge_enhanced(stats[member_idx], stats[j], criteria):
                    can_add = True
                    break
            
            if can_add:
                current_group.append(j)
                assigned.add(j)
        
        groups.append(current_group)
    
    return groups


def merge_group(tracklets, group_indices):
    """
    Merge tracklets in a group into one canonical person.
    
    For overlapping frames, compute union bbox.
    For non-overlapping frames, use the tracklet's bbox.
    """
    member_tracklets = [tracklets[i] for i in group_indices]
    
    # Sort by start frame
    member_tracklets.sort(key=lambda t: t['frame_numbers'][0])
    
    # Collect all unique frames and their data
    frame_to_data = {}  # frame_num -> (bboxes, confs, det_inds from all tracklets for that frame)
    
    for tracklet in member_tracklets:
        for frame_idx, frame_num in enumerate(tracklet['frame_numbers']):
            frame_num = int(frame_num)
            bbox = tracklet['bboxes'][frame_idx]
            conf = tracklet['confidences'][frame_idx]
            det_ind = tracklet['detection_indices'][frame_idx] if 'detection_indices' in tracklet else -1
            
            if frame_num not in frame_to_data:
                frame_to_data[frame_num] = {'bboxes': [], 'confs': [], 'det_inds': []}
            
            frame_to_data[frame_num]['bboxes'].append(bbox)
            frame_to_data[frame_num]['confs'].append(conf)
            frame_to_data[frame_num]['det_inds'].append(det_ind)
    
    # Build merged sequence
    sorted_frames = sorted(frame_to_data.keys())
    merged_frames = []
    merged_bboxes = []
    merged_confs = []
    merged_det_inds = []
    
    for frame_num in sorted_frames:
        bboxes_in_frame = frame_to_data[frame_num]['bboxes']
        confs_in_frame = frame_to_data[frame_num]['confs']
        det_inds_in_frame = frame_to_data[frame_num]['det_inds']
        
        # Compute union bbox (encompasses all tracklets in this frame)
        bboxes_array = np.array(bboxes_in_frame)
        x1 = np.min(bboxes_array[:, 0])
        y1 = np.min(bboxes_array[:, 1])
        x2 = np.max(bboxes_array[:, 2])
        y2 = np.max(bboxes_array[:, 3])
        union_bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        
        # Use maximum confidence in frame
        max_conf = np.max(confs_in_frame)
        
        # Use detection index from highest-confidence detection in frame
        max_conf_idx = np.argmax(confs_in_frame)
        det_ind = det_inds_in_frame[max_conf_idx]
        
        merged_frames.append(frame_num)
        merged_bboxes.append(union_bbox)
        merged_confs.append(max_conf)
        merged_det_inds.append(det_ind)
    
    # Convert to arrays
    merged_frames = np.array(merged_frames, dtype=np.int64)
    merged_bboxes = np.array(merged_bboxes, dtype=np.float32)
    merged_confs = np.array(merged_confs, dtype=np.float32)
    merged_det_inds = np.array(merged_det_inds, dtype=np.int64)
    
    # Canonical ID is the first tracklet ID in the group
    canonical_id = member_tracklets[0]['tracklet_id']
    original_ids = [tracklets[i]['tracklet_id'] for i in group_indices]
    
    # Collect ALL unique detection indices from all merged tracklets
    # This is for Stage 3c to extract all crops for quality scoring
    all_detection_indices = []
    for tracklet in member_tracklets:
        if 'detection_indices' in tracklet:
            all_detection_indices.extend(tracklet['detection_indices'].tolist())
    
    # Remove duplicates and -1 (invalid indices), keep unique valid indices
    unique_detection_indices = np.array(sorted(set(idx for idx in all_detection_indices if idx >= 0)), dtype=np.int64)
    
    return {
        'person_id': canonical_id,
        'frame_numbers': merged_frames,
        'bboxes': merged_bboxes,
        'confidences': merged_confs,
        'detection_indices': merged_det_inds,  # Per-frame indices (for bbox alignment)
        'all_detection_indices': unique_detection_indices,  # ALL unique indices (for Stage 3c crop extraction)
        'original_tracklet_ids': original_ids,
        'num_tracklets_merged': len(original_ids)
    }


def run_enhanced_grouping(config):
    """Run Stage 3b: Enhanced Canonical Grouping"""
    
    stage_config = config['stage3b_group']
    verbose = stage_config.get('advanced', {}).get('verbose', False) or config.get('global', {}).get('verbose', False)
    
    logger = PipelineLogger("Stage 3b: Enhanced Canonical Grouping", verbose=verbose)
    logger.header()
    
    # Extract configuration
    input_config = stage_config['input']
    output_config = stage_config['output']
    grouping_config = stage_config['grouping']
    
    # Load tracklet statistics from Stage 3a
    tracklet_stats_file = input_config['tracklet_stats_file']
    canonical_file = output_config['canonical_persons_file']
    grouping_log_file = output_config['grouping_log_file']
    
    logger.info(f"Loading tracklet stats: {Path(tracklet_stats_file).name}")
    data = np.load(tracklet_stats_file, allow_pickle=True)
    tracklets = list(data['tracklets'])
    stats = list(data['statistics'])
    
    # Group tracklets with enhanced criteria
    t_start = time.time()
    
    criteria = grouping_config['enhanced_criteria']
    groups = group_tracklets_enhanced(tracklets, stats, criteria)
    
    t_end = time.time()
    grouping_time = t_end - t_start
    logger.info(f"Created {len(groups)} canonical persons by merging tracklets")
    
    if verbose:
        # Count how many groups merged multiple tracklets
        merged_count = sum(1 for g in groups if len(g) > 1)
        logger.verbose_info(f"Groups with merges: {merged_count}/{len(groups)}")
        logger.verbose_info(f"Merge criteria used:")
        logger.verbose_info(f"  - Temporal gap: <{criteria['max_temporal_gap']} frames")
        logger.verbose_info(f"  - Spatial distance: <{criteria['max_spatial_distance']} px")
        logger.verbose_info(f"  - Area ratio: {criteria['area_ratio_range']}")
        logger.verbose_info(f"  - Motion alignment: >{criteria.get('min_motion_alignment', 0.6)}")
        logger.verbose_info(f"  - Jitter difference: <{criteria.get('max_jitter_difference', 40)} px")
    
    # Merge groups into canonical persons
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
    
    if verbose:
        logger.verbose_info(f"Sample canonical persons:")
        for log in grouping_log[:5]:
            logger.verbose_info(f"  Person {log['canonical_id']}: "
                  f"{log['num_merged']} tracklets, "
                  f"{log['total_frames']} frames, "
                  f"range [{log['start_frame']}, {log['end_frame']}]")
        if len(grouping_log) > 5:
            logger.verbose_info(f"  ... and {len(grouping_log) - 5} more")
    
    # Save canonical persons
    output_path = Path(canonical_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t_save_start = time.time()
    np.savez_compressed(output_path, persons=np.array(canonical_persons, dtype=object))
    npz_save_time = time.time() - t_save_start
    
    logger.info(f"Saved canonical persons: {output_path.name}")
    
    # Save log
    log_path = Path(grouping_log_file)
    t_log_save_start = time.time()
    with open(log_path, 'w') as f:
        json.dump(grouping_log, f, indent=2)
    log_save_time = time.time() - t_log_save_start

    # Write timings sidecar
    try:
        sidecar_path = Path(canonical_file).parent / (Path(canonical_file).name + '.timings.json')
        sidecar = {
            'canonical_file': str(canonical_file),
            'grouping_time': float(grouping_time),
            'npz_save_time': float(npz_save_time),
            'grouping_log_save_time': float(log_save_time),
            'num_persons': int(len(canonical_persons)),
            'num_merged_groups': int(sum(1 for g in groups if len(g) > 1)),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        with open(sidecar_path, 'w', encoding='utf-8') as sf:
            json.dump(sidecar, sf, indent=2)
        if verbose:
            logger.verbose_info(f"Wrote timings sidecar: {sidecar_path.name}")
    except Exception:
        if verbose:
            logger.verbose_info("Failed to write timings sidecar")


def main():
    parser = argparse.ArgumentParser(description='Stage 3b: Enhanced Canonical Grouping')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Check if stage is enabled
    if not config['pipeline']['stages'].get('stage3b', False):
        logger = PipelineLogger("Stage 3b: Enhanced Canonical Grouping", verbose=False)
        logger.info("Stage 3b is disabled in config")
        return
    
    # Run grouping
    run_enhanced_grouping(config)


if __name__ == '__main__':
    main()
