#!/usr/bin/env python3
"""
Stage 3a: Tracklet Analysis

Computes statistics for each tracklet including motion features (velocity, jitter).
This is part of the reorganized pipeline where analysis happens once and 
results are reused by subsequent stages.

Usage:
    python stage3a_analyze_tracklets.py --config configs/pipeline_config.yaml
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
    """Compute geometric and motion statistics for a tracklet"""
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


def run_analysis(config):
    """Run Stage 3a: Tracklet Analysis"""
    
    stage_config = config['stage3a']
    verbose = stage_config.get('advanced', {}).get('verbose', False) or config.get('global', {}).get('verbose', False)
    
    # Initialize logger
    logger = PipelineLogger("Stage 3a: Tracklet Analysis", verbose=verbose)
    logger.header()
    
    # Extract configuration
    input_config = stage_config['input']
    output_config = stage_config['output']
    
    tracklets_file = input_config['tracklets_file']
    tracklet_stats_file = output_config['tracklet_stats_file']
    
    # Load tracklets
    tracklets = load_tracklets(tracklets_file)
    num_tracklets = len(tracklets)
    logger.info(f"Loading tracklets: {Path(tracklets_file).name}")
    
    # Compute statistics
    t_start = time.time()
    
    stats = []
    for tracklet in tracklets:
        stat = compute_tracklet_statistics(tracklet)
        stats.append(stat)
    
    t_end = time.time()
    stats_time = t_end - t_start
    logger.info(f"Computing stats for {num_tracklets} tracklets")
    
    if verbose:
        logger.verbose_info("Sample tracklet statistics:")
        for i in range(min(3, len(stats))):
            stat = stats[i]
            logger.verbose_info(f"  T{tracklets[i]['tracklet_id']}: "
                  f"frames [{stat['start_frame']}, {stat['end_frame']}], "
                  f"duration={stat['duration']}, "
                  f"velocity_mag={stat['velocity_magnitude']:.2f}px/frame, "
                  f"jitter={stat['center_jitter']:.2f}px")
        if len(stats) > 3:
            logger.verbose_info(f"  ... and {len(stats) - 3} more")
    
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
    
    logger.info(f"Saved tracklet stats: {output_path.name}")
    
    # Write timings sidecar
    try:
        sidecar_path = Path(tracklet_stats_file).parent / (Path(tracklet_stats_file).name + '.timings.json')
        sidecar = {
            'tracklet_stats_file': str(tracklet_stats_file),
            'stats_time': float(stats_time),
            'npz_save_time': float(npz_save_time),
            'num_tracklets': int(num_tracklets),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        with open(sidecar_path, 'w', encoding='utf-8') as sf:
            json.dump(sidecar, sf, indent=2)
        if verbose:
            logger.verbose_info(f"Wrote timings sidecar: {sidecar_path.name}")
    except Exception:
        if verbose:
            logger.verbose_info("Failed to write timings sidecar")

    return {
        'tracklet_stats_file': tracklet_stats_file,
        'num_tracklets': num_tracklets
    }


def main():
    parser = argparse.ArgumentParser(description='Stage 3a: Tracklet Analysis')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Check if stage is enabled
    if not config['pipeline']['stages'].get('stage3a', False):
        logger = PipelineLogger("Stage 3a: Tracklet Analysis", verbose=False)
        logger.info("Stage 3a is disabled in config")
        return
    
    # Run analysis
    run_analysis(config)


if __name__ == '__main__':
    main()
