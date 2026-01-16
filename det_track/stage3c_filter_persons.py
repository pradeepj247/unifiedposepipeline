#!/usr/bin/env python3
"""
Stage 3c: Person Filtering & Crop Extraction

Filters canonical persons to TOP 10 based on ranking scores.
Then applies late-appearance penalty within the top 10 (may reduce to ~8).
Extracts crops for each selected person.

Input: canonical_persons.npz (40+ persons from Stage 3b)
Output: canonical_persons_3c.npz (8-10 persons), final_crops_3c.pkl (8-10 persons with crops)

Ranking criteria (with weights):
- Duration (40%): How long person appears in video
- Coverage (30%): Percentage of appearance timespan covered by detections
- Center (20%): Proximity to frame center
- Smoothness (10%): Motion stability (less jitter)
- Late-appearance penalty: Applied to top 10, persons starting after 50% get penalized (up to 30%)

Output files:
- canonical_persons_3c.npz: Filtered persons (input for Stage 3d)
- final_crops_3c.pkl: Crops for filtered persons (input for Stage 3d)

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
import cv2
import pickle
import os
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
    
    # Get video path - try canonical video first (from Stage 0), then fallback to original
    output_dir = Path(filtered_file).parent
    video_name = config.get('global', {}).get('current_video', 'canonical_video')
    canonical_video_path = output_dir.parent / video_name / 'canonical_video.mp4'
    
    if canonical_video_path.exists():
        video_path = canonical_video_path
    else:
        # Fallback to original video_file from config (construct full path)
        video_dir = config.get('global', {}).get('video_dir', '')
        video_file = config.get('global', {}).get('video_file', '')
        if video_dir and video_file:
            video_path = Path(video_dir) / video_file
        elif video_file:
            # video_file might already be a full path
            video_path = Path(video_file)
        else:
            logger.error("video_file not configured and canonical_video.mp4 not found")
            return
    
    # Read ACTUAL frame count from video (not config default)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if total_frames <= 0:
        logger.error(f"Invalid frame count from video: {total_frames}")
        return
    
    print(f"   VIDEO METADATA: {total_frames} frames, {video_fps} fps, max_appearance_ratio=0.5 (threshold: frame {int(total_frames*0.5)})")
    
    # Load canonical persons
    logger.info(f"Loading canonical persons: {Path(canonical_file).name}")
    data = np.load(canonical_file, allow_pickle=True)
    all_persons = list(data['persons'])
    
    if len(all_persons) == 0:
        logger.error(f"No persons to filter")
        return
    
    logger.stat("Total canonical persons", len(all_persons))
    
    # STEP 0: Filter by minimum duration (NEW - catches short persons early)
    min_duration_frames = filter_config.get('min_duration_frames', 150)
    filtered_by_duration = [p for p in all_persons if len(p['frame_numbers']) >= min_duration_frames]
    removed_by_duration = [p for p in all_persons if len(p['frame_numbers']) < min_duration_frames]
    
    if removed_by_duration:
        logger.info(f"Step 0: Filtering by minimum duration ({min_duration_frames} frames)...")
        # Summary only - details in sidecar JSON later
    
    logger.stat("After min_duration filter", f"{len(filtered_by_duration)} persons (threshold: {min_duration_frames} frames)")
    
    # Step 1: Select TOP 10 persons by score (from filtered candidates)
    logger.info(f"Step 1: Ranking all persons...")
    t_start = time.time()
    
    weights = filter_config.get('weights', {
        'duration': 0.4,
        'coverage': 0.3,
        'center': 0.2,
        'smoothness': 0.1,
        'max_appearance_ratio': 0.5
    })
    
    ranked_indices, scores = rank_persons(filtered_by_duration, video_width, video_height, total_frames, weights)
    t_ranking = time.time() - t_start
    
    # Select top N (default 10)
    top_n = filter_config.get('top_n', 10)
    top_indices = ranked_indices[:top_n]
    top_persons = [filtered_by_duration[i] for i in top_indices]  # From filtered_by_duration, not all_persons
    
    logger.found(f"Selected TOP {len(top_persons)} persons (from {len(filtered_by_duration)} filtered candidates)")
    
    if verbose:
        logger.verbose_info(f"Top {len(top_persons)} persons by composite score:")
        for rank, idx in enumerate(top_indices, 1):
            person = filtered_by_duration[idx]
            score = scores[idx]
            logger.verbose_info(f"  {rank}. Person {person['person_id']}: "
                  f"score={score['final_score']:.4f}, duration={score['duration']}, "
                  f"coverage={score['coverage']:.2f}, appearance_ratio={score['appearance_ratio']:.2f}")
    
    # Step 2: Apply late-appearance penalty to top N
    logger.info(f"Step 2: Applying late-appearance penalty to top {len(top_persons)}...")
    
    penalized_persons = []
    penalized_scores = []
    removed_persons = []
    
    for rank, person in enumerate(top_persons, 1):
        frames = person['frame_numbers']
        start_frame = int(frames[0])
        
        # Get the precomputed scores
        person_idx = filtered_by_duration.index(person)
        score_dict = scores[person_idx]
        
        # Check late-appearance penalty
        appearance_ratio = score_dict['appearance_ratio']
        max_appearance_ratio = weights.get('max_appearance_ratio', 0.5)
        penalty = score_dict['late_appearance_penalty']
        
        # No verbose printing - evaluate silently
        if appearance_ratio > max_appearance_ratio:
            # Person appeared late
            penalized_persons.append((person, penalty))
            if penalty < filter_config.get('penalty_threshold', 0.75):
                removed_persons.append((person, penalty))
            else:
                penalized_persons[-1] = (person, penalty)  # Mark as kept
        else:
            # Person appeared early (no penalty)
            penalized_persons.append((person, 1.0))
    
    # Filter: keep persons with penalty >= threshold
    penalty_threshold = filter_config.get('penalty_threshold', 0.75)
    selected_persons = [p for p, penalty in penalized_persons if penalty >= penalty_threshold]
    
    # Print concise summary with 💡 emoji
    total_input = len(filtered_by_duration)
    total_removed = total_input - len(selected_persons)
    top_person = selected_persons[0] if selected_persons else None
    if top_person:
        top_idx = filtered_by_duration.index(top_person)
        top_score = scores[top_idx]
        print(f"   💡 Filtered: {total_input} → {len(selected_persons)} persons (removed {total_removed} below threshold)")
        print(f"   💡 Top ranked: person_{top_person['person_id']} ({top_score['coverage']:.1%} coverage, score {top_score['final_score']:.2f})")
    else:
        print(f"   💡 Filtered: {total_input} → {len(selected_persons)} persons (removed {total_removed} below threshold)")
    
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
    
    # ==================== Save Filtering Details to JSON Sidecar ====================
    filtering_details = {
        "stage": "stage3c",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input": {
            "total_persons": len(all_persons),
            "min_duration_threshold": min_duration_frames
        },
        "removed_by_duration": [
            {
                "person_id": int(p['person_id']),
                "frames": len(p['frame_numbers']),
                "reason": "below_min_duration"
            }
            for p in removed_by_duration
        ],
        "top_10_scoring": [
            {
                "rank": rank,
                "person_id": int(person['person_id']),
                "duration_frames": int(scores[filtered_by_duration.index(person)]['duration']),
                "coverage": float(scores[filtered_by_duration.index(person)]['coverage']),
                "center_distance": float(scores[filtered_by_duration.index(person)]['center_distance']),
                "appearance_frame": int(person['frame_numbers'][0]),
                "appearance_ratio": float(scores[filtered_by_duration.index(person)]['appearance_ratio']),
                "late_appearance_penalty": float(scores[filtered_by_duration.index(person)]['late_appearance_penalty']),
                "final_score": float(scores[filtered_by_duration.index(person)]['final_score'])
            }
            for rank, person in enumerate(top_persons, 1)
        ],
        "filtering_result": {
            "penalty_threshold": float(penalty_threshold),
            "kept": len(selected_persons),
            "removed_by_penalty": len(removed_persons),
            "removed_person_ids": [int(p['person_id']) for p, _ in removed_persons]
        },
        "output": {
            "selected_persons": len(selected_persons),
            "npz_file": output_path.name
        },
        "timing": {
            "ranking_time": float(t_ranking),
            "npz_save_time": float(npz_save_time)
        }
    }
    
    sidecar_path = output_path.parent / 'stage3c_sidecar.json'
    with open(sidecar_path, 'w', encoding='utf-8') as f:
        json.dump(filtering_details, f, indent=2)
    
    if verbose:
        logger.verbose_info(f"Saved filtering details: {sidecar_path.name}")
    
    # ==================== Extract Crops ====================
    logger.step("Extracting crops from video...")
    
    # Get video path
    video_name = config.get('global', {}).get('current_video', 'canonical_video')
    video_path = output_path.parent.parent / video_name / 'canonical_video.mp4'
    
    # Fallback: use original video_file if canonical not found
    if not video_path.exists():
        video_path = config.get('global', {}).get('video_file', '')
    
    crops_per_person = filter_config.get('crops_per_person', 50)
    
    # ========== PHASE 4: LOAD CROPS_CACHE & O(1) LOOKUP ==========
    logger.info(f"Loading crops cache from Stage 1...")
    t_crop_start = time.time()
    
    # Load crops_cache.pkl from Stage 1 (same directory as detections_raw.npz)
    stage1_detections_file = config['stage1_detect']['output']['detections_file']
    crops_cache_path = Path(stage1_detections_file).parent / 'crops_cache.pkl'
    
    if verbose:
        logger.verbose_info(f"Stage 1 detections: {stage1_detections_file}")
        logger.verbose_info(f"Looking for crops_cache at: {crops_cache_path}")
    
    if not crops_cache_path.exists():
        logger.error(f"crops_cache.pkl not found at {crops_cache_path}")
        logger.error("Run Stage 1 first to generate crops_cache.pkl")
        return
    
    try:
        with open(crops_cache_path, 'rb') as f:
            all_crops = pickle.load(f)
        
        # Build lookup dict: {detection_idx: crop_dict}
        crops_by_idx = {crop['detection_idx']: crop for crop in all_crops}
        logger.info(f"✓ Loaded {len(crops_by_idx)} crops from cache (O(1) lookup ready)")
        
        # Extract and score crops for each selected person
        from crop_utils import compute_crop_quality_metrics
        
        crops_with_quality = []
        
        for person in selected_persons:
            person_id = person['person_id']
            
            # Get ALL detection indices for this person (from Stage 3b merge)
            if 'all_detection_indices' not in person:
                logger.error(f"Person {person_id} missing all_detection_indices (Stage 3b issue)")
                continue
            
            detection_indices = person['all_detection_indices']
            
            # O(1) lookup for each detection
            person_crops = []
            for det_idx in detection_indices:
                det_idx = int(det_idx)
                if det_idx in crops_by_idx:
                    crop_dict = crops_by_idx[det_idx]
                    
                    # Compute quality metrics
                    quality = compute_crop_quality_metrics(
                        bbox=crop_dict['bbox'],
                        frame_shape=(video_height, video_width),
                        confidence=crop_dict['confidence'],
                        frame_number=crop_dict['frame_idx']  # Stage 1 uses 'frame_idx' not 'frame_number'
                    )
                    
                    # Combined quality score (confidence + visibility)
                    combined_score = quality['confidence'] * 0.6 + quality['visibility_score'] * 0.4
                    
                    person_crops.append({
                        'crop': crop_dict['crop'],
                        'bbox': crop_dict['bbox'],
                        'frame_number': crop_dict['frame_idx'],  # Normalize to 'frame_number' for output
                        'confidence': crop_dict['confidence'],
                        'quality': quality,
                        'combined_score': combined_score
                    })
            
            if len(person_crops) == 0:
                logger.warning(f"Person {person_id}: No crops found in cache")
                continue
            
            # ========== 3-BIN TEMPORAL SELECTION: BEGINNING, MIDDLE, END ==========
            # Sort by frame_number for temporal ordering
            person_crops.sort(key=lambda x: x['frame_number'])
            
            # Divide into 3 equal bins: beginning, middle, end
            total_crops = len(person_crops)
            bin_size = total_crops // 3
            crops_per_bin = crops_per_person // 3  # Equal distribution across bins
            
            # Define bins
            bins = [
                person_crops[:bin_size],                    # Beginning
                person_crops[bin_size:2*bin_size],          # Middle
                person_crops[2*bin_size:]                   # End
            ]
            
            selected_crops = []
            for bin_idx, bin_crops in enumerate(bins):
                if len(bin_crops) == 0:
                    continue
                
                # Within each bin: pick best by quality
                bin_crops_sorted = sorted(bin_crops, key=lambda x: x['combined_score'], reverse=True)
                selected_crops.extend(bin_crops_sorted[:crops_per_bin])
            
            # Trim to exact target count and re-sort by quality for final selection
            best_crops = sorted(selected_crops, key=lambda x: x['combined_score'], reverse=True)[:crops_per_person]
            
            crops_with_quality.append({
                'person_id': person_id,
                'crops': best_crops,
                'total_available': len(person_crops),
                'selected_count': len(best_crops)
            })
            
            if verbose:
                logger.verbose_info(f"Person {person_id}: {len(best_crops)}/{len(person_crops)} best crops selected (3-bin: beginning/middle/end)")
        
        # Save to final_crops_3c.pkl
        final_crops_path = Path(crops_file)
        final_crops_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            'crops_with_quality': crops_with_quality,
            'video_source': str(video_path),
            'crops_per_person': crops_per_person,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        with open(final_crops_path, 'wb') as f:
            pickle.dump(output_data, f)
        
        t_crop_elapsed = time.time() - t_crop_start
        
        logger.info(f"✓ Extracted best crops for {len(crops_with_quality)} persons (quality-based selection)")
        logger.stat("Output crops file", str(final_crops_path))
        logger.timing("Crop lookup, scoring & save", t_crop_elapsed)
        
        # ========== CLEANUP: DELETE EPHEMERAL CROPS_CACHE ==========
        try:
            if crops_cache_path.exists():
                cache_size_mb = crops_cache_path.stat().st_size / 1024**2
                os.remove(crops_cache_path)
                logger.info(f"✓ Cleaned up ephemeral crops_cache.pkl ({cache_size_mb:.1f} MB freed)")
            else:
                logger.warning(f"crops_cache.pkl already deleted or not found at {crops_cache_path}")
        except Exception as e:
            logger.warning(f"Failed to delete crops_cache.pkl: {e}")
        
    except Exception as e:
        logger.error(f"Error during crop extraction: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
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
    parser.add_argument('--crops-per-person', type=int, default=None,
                       help='Number of crops to extract per person. Overrides config.')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply CLI override for crops_per_person if provided
    if args.crops_per_person is not None:
        if 'stage3c_filter' not in config:
            config['stage3c_filter'] = {}
        if 'filtering' not in config['stage3c_filter']:
            config['stage3c_filter']['filtering'] = {}
        config['stage3c_filter']['filtering']['crops_per_person'] = args.crops_per_person
    
    # Check if stage is enabled
    if not config['pipeline']['stages'].get('stage3c', False):
        logger = PipelineLogger("Stage 3c: Filter Persons", verbose=False)
        logger.info("Stage 3c is disabled in config")
        return
    
    # Run filtering
    run_filter(config)


if __name__ == '__main__':
    main()
