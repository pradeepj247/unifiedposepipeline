#!/usr/bin/env python3
"""
Stage 3c: Person Filtering & Crop Extraction

Filters canonical persons to TOP 10 based on ranking scores.
Then applies late-appearance penalty within the top 10 (may reduce to ~8).
Extracts crops for each selected person.

Input: canonical_persons.npz (40+ persons from Stage 3b)
Output: canonical_persons_filtered.npz (8-10 persons), final_crops.pkl (8-10 persons with crops)

Ranking criteria (with weights):
- Duration (40%): How long person appears in video
- Coverage (30%): Percentage of appearance timespan covered by detections
- Center (20%): Proximity to frame center
- Smoothness (10%): Motion stability (less jitter)
- Late-appearance penalty: Applied to top 10, persons starting after 50% get penalized (up to 30%)

Output files:
- canonical_persons_filtered.npz: Filtered persons (input for Stage 3d)
- final_crops.pkl: Crops for filtered persons (input for Stage 3d)

Usage:
    python stage3c_filter_persons.py --config configs/pipeline_config.yaml
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


def compute_ranking_scores(person, video_width, video_height, total_frames, weights):
    """Compute ranking scores for a person with late-appearance penalty"""
    frames = person['frame_numbers']
    bboxes = person['bboxes']
    
    # 1. Duration score (longer presence = higher score)
    duration = len(frames)
    
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
    
    # 5. Late-appearance penalty
    # Persons starting after 50% of video get penalized
    appearance_ratio = start_frame / total_frames if total_frames > 0 else 0
    max_appearance_ratio = weights.get('max_appearance_ratio', 0.5)
    
    if appearance_ratio > max_appearance_ratio:
        # Linear penalty: 50% → no penalty, 100% → full penalty
        penalty_factor = (appearance_ratio - max_appearance_ratio) / (1.0 - max_appearance_ratio)
        late_appearance_penalty = 1.0 - (penalty_factor * 0.3)  # Up to 30% penalty
    else:
        late_appearance_penalty = 1.0
    
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
    ) * late_appearance_penalty
    
    return {
        'duration': duration,
        'coverage': float(coverage_score),
        'center_distance': float(distances.mean()),
        'smoothness': float(smoothness_score),
        'start_frame': start_frame,
        'appearance_ratio': float(appearance_ratio),
        'late_appearance_penalty': float(late_appearance_penalty),
        'duration_normalized': float(duration_normalized),
        'coverage_normalized': float(coverage_normalized),
        'center_normalized': float(center_normalized),
        'smoothness_normalized': float(smoothness_normalized),
        'final_score': float(final_score)
    }


def rank_persons(persons, video_width, video_height, total_frames, weights):
    """Rank persons based on heuristics with late-appearance penalty"""
    scores = []
    
    for person in persons:
        score_dict = compute_ranking_scores(person, video_width, video_height, total_frames, weights)
        scores.append(score_dict)
    
    # Rank by final score (descending)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i]['final_score'], reverse=True)
    
    return ranked_indices, scores


def run_filter(config):
    """Run Stage 3c: Person Filtering & Crop Extraction"""
    
    stage_config = config['stage3c_filter']
    verbose = stage_config.get('advanced', {}).get('verbose', False) or config.get('global', {}).get('verbose', False)
    
    logger = PipelineLogger("Stage 3c: Filter Persons", verbose=verbose)
    logger.header()
    
    # Extract configuration
    input_config = stage_config['input']
    output_config = stage_config['output']
    filter_config = stage_config['filtering']
    
    canonical_file = input_config['canonical_persons_file']
    filtered_file = output_config['canonical_persons_filtered_file']
    crops_file = output_config['final_crops_file']
    
    # Video dimensions (for center bias)
    video_width = config.get('global', {}).get('video_width', 1920)
    video_height = config.get('global', {}).get('video_height', 1080)
    video_fps = config.get('global', {}).get('video_fps', 30)
    video_duration = config.get('global', {}).get('video_duration_seconds', 0)
    
    # Calculate total frames from video metadata if available
    total_frames = int(video_duration * video_fps) if video_duration > 0 else 10000
    
    # Load canonical persons
    logger.info(f"Loading canonical persons: {Path(canonical_file).name}")
    data = np.load(canonical_file, allow_pickle=True)
    all_persons = list(data['persons'])
    
    if len(all_persons) == 0:
        logger.error(f"No persons to filter")
        return
    
    logger.stat("Total canonical persons", len(all_persons))
    
    # Step 1: Select TOP 10 persons by score
    logger.info(f"Step 1: Ranking all persons...")
    t_start = time.time()
    
    weights = filter_config.get('weights', {
        'duration': 0.4,
        'coverage': 0.3,
        'center': 0.2,
        'smoothness': 0.1,
        'max_appearance_ratio': 0.5
    })
    
    ranked_indices, scores = rank_persons(all_persons, video_width, video_height, total_frames, weights)
    t_ranking = time.time() - t_start
    
    # Select top N (default 10)
    top_n = filter_config.get('top_n', 10)
    top_indices = ranked_indices[:top_n]
    top_persons = [all_persons[i] for i in top_indices]
    
    logger.found(f"Selected TOP {len(top_persons)} persons (from {len(all_persons)} candidates)")
    
    if verbose:
        logger.verbose_info(f"Top {len(top_persons)} persons (before penalty):")
        for rank, idx in enumerate(top_indices, 1):
            person = all_persons[idx]
            score = scores[idx]
            logger.verbose_info(f"  {rank}. Person {person['person_id']}: "
                  f"score={score['final_score']:.4f}, duration={score['duration']}")
    
    # Step 2: Apply late-appearance penalty to top N
    logger.info(f"Step 2: Applying late-appearance penalty to top {len(top_persons)}...")
    
    penalized_persons = []
    penalized_scores = []
    
    for person in top_persons:
        frames = person['frame_numbers']
        start_frame = int(frames[0])
        
        # Check late-appearance penalty
        appearance_ratio = start_frame / total_frames if total_frames > 0 else 0
        max_appearance_ratio = weights.get('max_appearance_ratio', 0.5)
        
        if appearance_ratio > max_appearance_ratio:
            # Mark for potential removal (late appearance)
            penalty_factor = (appearance_ratio - max_appearance_ratio) / (1.0 - max_appearance_ratio)
            penalty = 1.0 - (penalty_factor * 0.3)
            
            penalized_persons.append((person, penalty))
            if verbose:
                logger.verbose_info(f"  Person {person['person_id']}: appearance_ratio={appearance_ratio:.2f}, penalty={penalty:.2f}")
        else:
            # No penalty
            penalized_persons.append((person, 1.0))
    
    # Filter: keep persons with penalty > threshold (e.g., 0.8 = only 20% penalty)
    penalty_threshold = filter_config.get('penalty_threshold', 0.7)
    selected_persons = [p for p, penalty in penalized_persons if penalty >= penalty_threshold]
    
    logger.found(f"After penalty filtering: {len(selected_persons)} persons (threshold: {penalty_threshold:.1f})")
    
    if verbose:
        logger.verbose_info(f"Selected persons after penalty:")
        for person in selected_persons:
            logger.verbose_info(f"  - Person {person['person_id']}")
    
    # Save filtered canonical persons
    output_path = Path(filtered_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t_save_start = time.time()
    
    np.savez_compressed(
        output_path,
        persons=selected_persons
    )
    npz_save_time = time.time() - t_save_start
    
    logger.info(f"Saved filtered persons: {output_path.name}")
    logger.stat("Output file", str(output_path))
    
    # ==================== Extract Crops ====================
    logger.step("Extracting crops from video...")
    
    # Get video path
    video_name = config.get('global', {}).get('current_video', 'canonical_video')
    video_path = output_path.parent.parent / video_name / 'canonical_video.mp4'
    
    # Fallback: use original video_file if canonical not found
    if not video_path.exists():
        video_path = config.get('global', {}).get('video_file', '')
    
    crops_per_person = filter_config.get('crops_per_person', 50)
    
    if not video_path or not str(video_path).strip():
        logger.error("Cannot extract crops: video_path not configured or found")
        return
    
    try:
        from crop_utils import extract_crops_with_quality, save_final_crops
        
        t_crop_start = time.time()
        
        # Extract crops with quality metrics
        logger.info(f"Extracting {len(selected_persons)} persons, {crops_per_person} crops each...")
        crops_with_quality = extract_crops_with_quality(
            video_path=str(video_path),
            persons=selected_persons,
            target_crops_per_person=crops_per_person,
            top_n=len(selected_persons),  # Extract crops for all selected persons
            max_first_appearance_ratio=1.0,  # No additional filtering
            verbose=verbose
        )
        
        # Save to final_crops.pkl
        final_crops_path = Path(crops_file)
        save_final_crops(
            crops_with_quality=crops_with_quality,
            output_path=final_crops_path,
            video_source=str(video_path),
            verbose=verbose
        )
        
        t_crop_elapsed = time.time() - t_crop_start
        
        logger.info(f"✓ Extracted crops for {len(crops_with_quality)} persons")
        logger.stat("Output crops file", str(final_crops_path))
        logger.timing("Crop extraction & save", t_crop_elapsed)
        
    except ImportError as e:
        logger.error(f"crop_utils module not found: {e}")
    except Exception as e:
        logger.error(f"Error during crop extraction: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
    
    # ==================== Write Timings ====================
    try:
        sidecar_path = Path(filtered_file).parent / (Path(filtered_file).name + '.timings.json')
        sidecar = {
            'filtered_persons_file': str(filtered_file),
            'final_crops_file': str(crops_file),
            'input_total_persons': int(len(all_persons)),
            'output_top_n': int(top_n),
            'output_final_persons': int(len(selected_persons)),
            'penalty_threshold': float(penalty_threshold),
            'ranking_time': float(t_ranking),
            'npz_save_time': float(npz_save_time),
            'crops_per_person': int(crops_per_person),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        with open(sidecar_path, 'w', encoding='utf-8') as sf:
            json.dump(sidecar, sf, indent=2)
        if verbose:
            logger.verbose_info(f"Wrote timings sidecar: {sidecar_path.name}")
    except Exception as e:
        if verbose:
            logger.verbose_info(f"Failed to write timings sidecar: {e}")
    
    logger.success()


def main():
    parser = argparse.ArgumentParser(description='Stage 3c: Filter Persons')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Check if stage is enabled
    if not config['pipeline']['stages'].get('stage3c', False):
        logger = PipelineLogger("Stage 3c: Filter Persons", verbose=False)
        logger.info("Stage 3c is disabled in config")
        return
    
    # Run filtering
    run_filter(config)


if __name__ == '__main__':
    main()
