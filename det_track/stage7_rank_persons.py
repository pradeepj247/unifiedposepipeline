#!/usr/bin/env python3
"""
Stage 5: Primary Person Ranking

Ranks canonical persons and selects the primary person for pose estimation.

Usage:
    python stage5_rank_persons.py --config configs/pipeline_config.yaml
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


def compute_ranking_scores(person, video_width, video_height, weights):
    """Compute ranking scores for a person"""
    frames = person['frame_numbers']
    bboxes = person['bboxes']
    
    # 1. Duration score (longer presence = higher score)
    duration = len(frames)
    duration_score = duration
    
    # 2. Coverage score (frame range coverage)
    start_frame = int(frames[0])
    end_frame = int(frames[-1])
    frame_range = end_frame - start_frame + 1
    coverage_score = duration / frame_range if frame_range > 0 else 0
    
    # 3. Center bias (closer to center = higher score)
    centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2
    frame_center = np.array([video_width / 2, video_height / 2])
    distances = np.linalg.norm(centers - frame_center, axis=1)
    center_score = 1.0 / (distances.mean() + 1)
    
    # 4. Smoothness (consistent motion = higher score)
    if len(centers) > 1:
        velocities = np.diff(centers, axis=0)
        velocity_variance = np.var(np.linalg.norm(velocities, axis=1))
        smoothness_score = 1.0 / (velocity_variance + 1)
    else:
        smoothness_score = 0.0
    
    # Normalize scores
    max_duration = 10000  # Assume max video length
    duration_normalized = min(duration / max_duration, 1.0)
    coverage_normalized = coverage_score
    center_normalized = min(center_score / 10, 1.0)
    smoothness_normalized = min(smoothness_score / 100, 1.0)
    
    # Weighted combination
    final_score = (
        weights['duration'] * duration_normalized +
        weights['coverage'] * coverage_normalized +
        weights['center'] * center_normalized +
        weights['smoothness'] * smoothness_normalized
    )
    
    return {
        'duration': duration,
        'coverage': float(coverage_score),
        'center_distance': float(distances.mean()),
        'smoothness': float(smoothness_score),
        'duration_normalized': float(duration_normalized),
        'coverage_normalized': float(coverage_normalized),
        'center_normalized': float(center_normalized),
        'smoothness_normalized': float(smoothness_normalized),
        'final_score': float(final_score)
    }


def rank_persons_auto(persons, video_width, video_height, weights):
    """Automatic ranking based on heuristics"""
    scores = []
    
    for person in persons:
        score_dict = compute_ranking_scores(person, video_width, video_height, weights)
        scores.append(score_dict)
    
    # Rank by final score (descending)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i]['final_score'], reverse=True)
    
    return ranked_indices, scores


def run_ranking(config):
    """Run Stage 5: Ranking"""
    
    stage_config = config['stage7']
    verbose = stage_config.get('advanced', {}).get('verbose', False) or config.get('global', {}).get('verbose', False)
    
    logger = PipelineLogger("Stage 7: Rank Persons", verbose=verbose)
    
    # Extract configuration
    input_config = stage_config['input']
    output_config = stage_config['output']
    ranking_config = stage_config['ranking']
    
    canonical_file = input_config['canonical_persons_file']
    primary_file = output_config['primary_person_file']
    ranking_report_file = output_config['ranking_report_file']
    
    # Video dimensions (for center bias)
    video_width = config.get('global', {}).get('video_width', 1920)
    video_height = config.get('global', {}).get('video_height', 1080)
    
    logger.header()
    
    # Load canonical persons
    logger.step(f"Loading canonical persons: {canonical_file}")
    data = np.load(canonical_file, allow_pickle=True)
    persons = list(data['persons'])
    logger.info(f"Loaded {len(persons)} persons")
    
    if len(persons) == 0:
        logger.error(f"No persons to rank")
        return
    
    # Rank persons
    logger.step(f"Ranking persons (method: {ranking_config['method']})...")
    t_start = time.time()
    
    if ranking_config['method'] == 'auto':
        weights = ranking_config['weights']
        ranked_indices, scores = rank_persons_auto(persons, video_width, video_height, weights)
    else:
        raise ValueError(f"Unknown ranking method: {ranking_config['method']}")
    
    t_end = time.time()
    logger.timing("Ranking", t_end - t_start)
    logger.info(f"Ranked {len(persons)} persons")
    
    # Select primary
    primary_idx = ranked_indices[0]
    primary_person = persons[primary_idx]
    
    logger.found(f"Primary person selected: Person {primary_person['person_id']}")
    logger.stat("Duration (frames)", len(primary_person['frame_numbers']))
    logger.stat("Primary score", f"{scores[primary_idx]['final_score']:.4f}")
    
    if verbose:
        logger.verbose_info(f"Top 5 persons:")
        for rank, idx in enumerate(ranked_indices[:5], 1):
            person = persons[idx]
            score = scores[idx]
            logger.verbose_info(f"  {rank}. Person {person['person_id']}: "
                  f"score={score['final_score']:.4f}, "
                  f"duration={score['duration']}, "
                  f"coverage={score['coverage']:.2f}")
        if len(ranked_indices) > 5:
            logger.verbose_info(f"  ... and {len(ranked_indices) - 5} more")
    
    # Save primary person
    output_path = Path(primary_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        primary_person=primary_person,
        person_id=primary_person['person_id'],
        frame_numbers=primary_person['frame_numbers'],
        bboxes=primary_person['bboxes'],
        confidences=primary_person['confidences']
    )
    logger.file_size(f"Primary person file", output_path.stat().st_size / (1024 * 1024))
    
    # Save ranking report
    ranking_report = []
    for rank, idx in enumerate(ranked_indices, 1):
        person = persons[idx]
        score = scores[idx]
        ranking_report.append({
            'rank': rank,
            'person_id': int(person['person_id']),
            'scores': score,
            'is_primary': (rank == 1)
        })
    
    report_path = Path(ranking_report_file)
    with open(report_path, 'w') as f:
        json.dump(ranking_report, f, indent=2)
    logger.file_size(f"Ranking report", report_path.stat().st_size / (1024 * 1024))
    
    logger.stat("Primary person ID", primary_person['person_id'])
    logger.stat("Total candidates", len(persons))


def main():
    parser = argparse.ArgumentParser(description='Stage 5: Primary Person Ranking')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Check if stage is enabled
    if not config['pipeline']['stages']['stage7']:
        logger = PipelineLogger("Stage 7: Rank Persons", verbose=False)
        logger.info("Stage 7 is disabled in config")
        return
    
    # Run ranking
    run_ranking(config)


if __name__ == '__main__':
    main()
