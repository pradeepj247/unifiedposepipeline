#!/usr/bin/env python3
"""
Stage 3: Tracklet Analysis

Computes statistics for each tracklet and identifies candidate pairs
for ReID recovery.

Usage:
    python stage3_analyze_tracklets.py --config configs/pipeline_config.yaml
"""

import argparse
import yaml
import numpy as np
import json
import time
from datetime import datetime, timezone
import re
import sys
from pathlib import Path

# Add parent directory to path for imports
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


def load_tracklets(tracklets_file):
    """Load tracklets from NPZ file"""
    data = np.load(tracklets_file, allow_pickle=True)
    return data['tracklets']


def compute_tracklet_statistics(tracklet):
    """Compute geometric statistics for a tracklet"""
    frames = tracklet['frame_numbers']
    bboxes = tracklet['bboxes']
    confidences = tracklet['confidences']
    
    # Temporal
    start_frame = int(frames[0])
    end_frame = int(frames[-1])
    duration = len(frames)
    
    # Spatial
    centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2  # (x_center, y_center)
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    
    mean_center = centers.mean(axis=0)
    center_jitter = np.std(centers, axis=0).mean()
    
    mean_area = areas.mean()
    area_variance = areas.std()
    
    # Motion
    if len(centers) > 1:
        velocities = np.diff(centers, axis=0)
        mean_velocity = velocities.mean(axis=0)
        velocity_magnitude = np.linalg.norm(velocities, axis=1).mean()
    else:
        mean_velocity = np.array([0.0, 0.0])
        velocity_magnitude = 0.0
    
    # First and last bbox
    first_bbox = bboxes[0]
    last_bbox = bboxes[-1]
    
    return {
        'start_frame': start_frame,
        'end_frame': end_frame,
        'duration': duration,
        'mean_center': mean_center.tolist(),
        'center_jitter': float(center_jitter),
        'mean_area': float(mean_area),
        'area_variance': float(area_variance),
        'mean_velocity': mean_velocity.tolist(),
        'velocity_magnitude': float(velocity_magnitude),
        'first_bbox': first_bbox.tolist(),
        'last_bbox': last_bbox.tolist(),
        'mean_confidence': float(confidences.mean())
    }


def identify_reid_candidates(tracklets, stats, criteria):
    """Identify tracklet pairs that might be the same person"""
    max_temporal_gap = criteria['max_temporal_gap']
    max_spatial_distance = criteria['max_spatial_distance']
    area_ratio_range = criteria['area_ratio_range']
    
    candidates = []
    
    for i in range(len(tracklets)):
        for j in range(i + 1, len(tracklets)):
            stat_i = stats[i]
            stat_j = stats[j]
            
            # Rule 1: Temporal proximity (tracklets close in time or overlapping)
            gap = stat_j['start_frame'] - stat_i['end_frame']
            # Allow overlapping tracklets (gap < 0) and tracklets with small gaps
            if gap < -100 or gap > max_temporal_gap:  # Allow up to 100 frames overlap
                continue
            
            # Rule 2: Spatial proximity
            # For overlapping tracklets: compare bboxes at overlapping frames
            # For non-overlapping: compare last of i with first of j
            if gap < 0:  # Overlapping case
                # Find overlapping frames
                overlap_start = stat_j['start_frame']
                overlap_end = stat_i['end_frame']
                
                # Compare bboxes at overlap_start frame
                i_frames = tracklets[i]['frame_numbers']
                j_frames = tracklets[j]['frame_numbers']
                
                # Use np.isin for fast membership checking
                if np.isin(overlap_start, i_frames) and np.isin(overlap_start, j_frames):
                    i_idx = np.where(i_frames == overlap_start)[0][0]
                    j_idx = np.where(j_frames == overlap_start)[0][0]
                    
                    i_bbox = tracklets[i]['bboxes'][i_idx]
                    j_bbox = tracklets[j]['bboxes'][j_idx]
                    
                    i_center = np.array([(i_bbox[0] + i_bbox[2]) / 2, (i_bbox[1] + i_bbox[3]) / 2])
                    j_center = np.array([(j_bbox[0] + j_bbox[2]) / 2, (j_bbox[1] + j_bbox[3]) / 2])
                    distance = np.linalg.norm(i_center - j_center)
                else:
                    # Can't find overlap, skip
                    continue
            else:  # Non-overlapping case
                # Use last bbox of i and first bbox of j
                last_center_i = np.array([
                    (stat_i['last_bbox'][0] + stat_i['last_bbox'][2]) / 2,
                    (stat_i['last_bbox'][1] + stat_i['last_bbox'][3]) / 2
                ])
                first_center_j = np.array([
                    (stat_j['first_bbox'][0] + stat_j['first_bbox'][2]) / 2,
                    (stat_j['first_bbox'][1] + stat_j['first_bbox'][3]) / 2
                ])
                distance = np.linalg.norm(last_center_i - first_center_j)
            
            if distance > max_spatial_distance:
                continue
            
            # Rule 3: Size consistency
            area_i = stat_i['mean_area']
            area_j = stat_j['mean_area']
            area_ratio = area_j / area_i if area_i > 0 else 0
            
            if area_ratio < area_ratio_range[0] or area_ratio > area_ratio_range[1]:
                continue
            
            # Passed all checks - candidate for ReID
            candidates.append({
                'tracklet_1': int(tracklets[i]['tracklet_id']),
                'tracklet_2': int(tracklets[j]['tracklet_id']),
                'gap': int(gap),
                'distance': float(distance),
                'area_ratio': float(area_ratio),
                'transition_frame_1': int(stat_i['end_frame']),
                'transition_frame_2': int(stat_j['start_frame'])
            })
    
    return candidates


def run_analysis(config):
    """Run Stage 3: Analysis"""
    
    stage_config = config['stage3']
    verbose = stage_config.get('advanced', {}).get('verbose', False) or config.get('global', {}).get('verbose', False)
    
    # Initialize logger
    logger = PipelineLogger("Stage 3: Tracklet Analysis", verbose=verbose)
    logger.header()
    
    # Extract configuration
    analysis_config = stage_config['analysis']
    candidate_criteria = stage_config['candidate_criteria']
    input_config = stage_config['input']
    output_config = stage_config['output']
    
    tracklets_file = input_config['tracklets_file']
    tracklet_stats_file = output_config['tracklet_stats_file']
    candidates_file = output_config['candidates_file']
    
    compute_statistics = analysis_config['compute_statistics']
    identify_candidates = analysis_config['identify_candidates']
    
    # Load tracklets
    tracklets = load_tracklets(tracklets_file)
    num_tracklets = len(tracklets)
    logger.info(f"Loading tracklets: {Path(tracklets_file).name}")
    
    # Compute statistics
    if compute_statistics:
        t_start = time.time()
        
        stats = []
        for tracklet in tracklets:
            stat = compute_tracklet_statistics(tracklet)
            stats.append(stat)
        
        t_end = time.time()
        stats_time = t_end - t_start
        logger.info(f"Computing stats for {num_tracklets} tracklets")
        
        # Save statistics
        output_path = Path(tracklet_stats_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        t_save_start = time.time()
        np.savez_compressed(
            output_path,
            tracklets=tracklets,
            statistics=np.array(stats, dtype=object)
        )
        npz_save_time = time.time() - t_save_start
    else:
        stats = None
        stats_time = 0.0
        npz_save_time = 0.0
    
    # Identify ReID candidates
    if identify_candidates and stats is not None:
        t_start = time.time()
        
        candidates = identify_reid_candidates(tracklets, stats, candidate_criteria)
        
        t_end = time.time()
        candidate_id_time = t_end - t_start
        num_candidates = len(candidates)
        logger.info(f"Found {num_candidates} candidate pairs")
        
        if verbose and num_candidates > 0:
            logger.verbose_info("Sample candidates:")
            for cand in candidates[:5]:
                logger.verbose_info(f"  T{cand['tracklet_1']} → T{cand['tracklet_2']}: "
                      f"gap={cand['gap']}f, dist={cand['distance']:.1f}px, "
                      f"area_ratio={cand['area_ratio']:.2f}")
            if num_candidates > 5:
                logger.verbose_info(f"  ... and {num_candidates - 5} more")
        
        # Save candidates
        output_path = Path(candidates_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        t_cand_save_start = time.time()
        with open(output_path, 'w') as f:
            json.dump(candidates, f, indent=2)
        candidate_save_time = time.time() - t_cand_save_start
    else:
        candidates = []
        candidate_id_time = 0.0
        candidate_save_time = 0.0
    
    # Write timings sidecar
    try:
        sidecar_path = Path(tracklet_stats_file).parent / (Path(tracklet_stats_file).name + '.timings.json')
        sidecar = {
            'tracklet_stats_file': str(tracklet_stats_file),
            'candidates_file': str(candidates_file),
            'stats_time': float(stats_time) if 'stats_time' in locals() else 0.0,
            'npz_save_time': float(npz_save_time) if 'npz_save_time' in locals() else 0.0,
            'candidate_id_time': float(candidate_id_time) if 'candidate_id_time' in locals() else 0.0,
            'candidate_save_time': float(candidate_save_time) if 'candidate_save_time' in locals() else 0.0,
            'num_tracklets': int(num_tracklets),
            'num_candidates': int(len(candidates)),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        with open(sidecar_path, 'w', encoding='utf-8') as sf:
            json.dump(sidecar, sf, indent=2)
        if verbose:
            logger.info(f"Wrote timings sidecar: {sidecar_path.name}")
    except Exception:
        if verbose:
            logger.info("Failed to write timings sidecar")

    return {
        'tracklet_stats_file': tracklet_stats_file,
        'candidates_file': candidates_file,
        'num_tracklets': num_tracklets,
        'num_candidates': len(candidates)
    }


def main():
    parser = argparse.ArgumentParser(description='Stage 3: Tracklet Analysis')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Check if stage is enabled
    if not config['pipeline']['stages']['stage3']:
        logger = PipelineLogger("Stage 3: Tracklet Analysis", verbose=False)
        logger.step("Stage 3 is disabled in config")
        return
    
    # Run analysis
    run_analysis(config)


if __name__ == '__main__':
    main()
