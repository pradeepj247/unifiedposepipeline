#!/usr/bin/env python3
"""
Stage 3c NEW: Person Filtering & On-Demand Crop Extraction (EXPERIMENTAL)

EXPERIMENT: Extract crops on-demand from canonical_video.mp4 instead of using crops_cache.pkl

Hypothesis: canonical_video.mp4 (720p, all I-frames) seeking is fast enough to offset
            the Stage 1 savings from skipping crop extraction (89 FPS → 121 FPS).

This variant:
- Uses canonical_persons.npz (same as original)
- Extracts crops directly from canonical_video.mp4 (instead of crops_cache.pkl)
- Only extracts needed crops (60 per person × 8-10 persons = 480-600 crops)
- Leverages I-frame optimization for fast seeking

Comparison vs Original:
- Original: Stage 1 extracts ALL 8642 crops (slow), Stage 3c does O(1) lookup (fast)
- This: Stage 1 skips crops (fast), Stage 3c seeks+extracts 600 crops (unknown speed)

Usage:
    python stage3c_new.py --config configs/pipeline_config.yaml
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


def extract_crop_from_video(cap, frame_idx, bbox):
    """Extract a single crop from video at specific frame"""
    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    
    if not ret or frame is None:
        return None
    
    # Extract crop using bbox
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    
    # Clamp bbox to frame boundaries
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    crop = frame[y1:y2, x1:x2]
    
    # Resize to 256x192 (standard pose estimation size)
    if crop.size > 0:
        crop_resized = cv2.resize(crop, (192, 256))
        return crop_resized
    
    return None


def run_filter(config):
    """Run Stage 3c NEW: Person Filtering & On-Demand Crop Extraction"""
    
    stage_config = config['stage3c_filter']
    verbose = stage_config.get('advanced', {}).get('verbose', False) or config.get('global', {}).get('verbose', False)
    
    logger = PipelineLogger("Stage 3c NEW: Filter Persons (On-Demand Crops)", verbose=verbose)
    logger.header()
    logger.info("🧪 EXPERIMENTAL MODE: Extracting crops on-demand from canonical_video.mp4")
    
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
    
    # Get canonical video path from Stage 0
    canonical_video_path = Path(config['stage0_normalize']['output']['canonical_video_file'])
    
    if not canonical_video_path.exists():
        logger.error(f"Canonical video not found: {canonical_video_path}")
        logger.error("Stage 0 must run first to create canonical_video.mp4")
        return
    
    video_path = canonical_video_path
    logger.info(f"📹 Using canonical video: {video_path.name}")
    
    # Read ACTUAL frame count from video
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
    
    # STEP 0: Filter by minimum duration
    min_duration_frames = filter_config.get('min_duration_frames', 150)
    filtered_by_duration = [p for p in all_persons if len(p['frame_numbers']) >= min_duration_frames]
    removed_by_duration = [p for p in all_persons if len(p['frame_numbers']) < min_duration_frames]
    
    if removed_by_duration:
        logger.info(f"Step 0: Filtering by minimum duration ({min_duration_frames} frames)...")
    
    logger.stat("After min_duration filter", f"{len(filtered_by_duration)} persons (threshold: {min_duration_frames} frames)")
    
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
    
    ranked_indices, scores = rank_persons(filtered_by_duration, video_width, video_height, total_frames, weights)
    t_ranking = time.time() - t_start
    
    # Select top N (default 10)
    top_n = filter_config.get('top_n', 10)
    top_indices = ranked_indices[:top_n]
    top_persons = [filtered_by_duration[i] for i in top_indices]
    
    logger.found(f"Selected TOP {len(top_persons)} persons (from {len(filtered_by_duration)} filtered candidates)")
    
    # Step 2: Apply late-appearance penalty
    logger.info(f"Step 2: Applying late-appearance penalty to top {len(top_persons)}...")
    
    penalized_persons = []
    removed_persons = []
    
    for rank, person in enumerate(top_persons, 1):
        frames = person['frame_numbers']
        person_idx = filtered_by_duration.index(person)
        score_dict = scores[person_idx]
        
        appearance_ratio = score_dict['appearance_ratio']
        max_appearance_ratio = weights.get('max_appearance_ratio', 0.5)
        penalty = score_dict['late_appearance_penalty']
        
        if appearance_ratio > max_appearance_ratio:
            penalized_persons.append((person, penalty))
            if penalty < filter_config.get('penalty_threshold', 0.75):
                removed_persons.append((person, penalty))
            else:
                penalized_persons[-1] = (person, penalty)
        else:
            penalized_persons.append((person, 1.0))
    
    # Filter: keep persons with penalty >= threshold
    penalty_threshold = filter_config.get('penalty_threshold', 0.75)
    selected_persons = [p for p, penalty in penalized_persons if penalty >= penalty_threshold]
    
    # Print summary
    total_input = len(filtered_by_duration)
    total_removed = total_input - len(selected_persons)
    top_person = selected_persons[0] if selected_persons else None
    if top_person:
        top_idx = filtered_by_duration.index(top_person)
        top_score = scores[top_idx]
        print(f"   💡 Filtered: {total_input} → {len(selected_persons)} persons (removed {total_removed} below threshold)")
        print(f"   💡 Top ranked: person_{top_person['person_id']} ({top_score['coverage']:.1%} coverage, score {top_score['final_score']:.2f})")
    
    logger.found(f"After penalty filtering: {len(selected_persons)} persons (threshold: {penalty_threshold:.1f})")
    
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
    
    # ==================== ON-DEMAND CROP EXTRACTION ====================
    logger.step("🧪 Extracting crops ON-DEMAND from canonical video...")
    logger.info(f"   Canonical video (720p, I-frames only): {video_path}")
    
    crops_per_person = filter_config.get('crops_per_person', 60)
    
    # Open video for extraction
    t_crop_start = time.time()
    t_video_open_start = time.time()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video for crop extraction: {video_path}")
        return
    t_video_open = time.time() - t_video_open_start
    
    logger.info(f"✓ Video opened ({t_video_open:.3f}s)")
    
    # Import quality metrics
    from crop_utils import compute_crop_quality_metrics
    
    crops_with_quality = []
    total_seeks = 0
    total_extractions = 0
    
    t_extraction_start = time.time()
    
    for person in selected_persons:
        person_id = person['person_id']
        
        # Get frame numbers and bboxes for this person
        frame_numbers = person['frame_numbers']
        bboxes = person['bboxes']
        confidences = person.get('confidences', np.ones(len(frame_numbers)))
        
        # Extract all crops for quality scoring
        person_crops = []
        for i, (frame_idx, bbox, conf) in enumerate(zip(frame_numbers, bboxes, confidences)):
            total_seeks += 1
            crop = extract_crop_from_video(cap, int(frame_idx), bbox)
            
            if crop is not None:
                total_extractions += 1
                
                # Compute quality metrics
                quality = compute_crop_quality_metrics(
                    bbox=bbox,
                    frame_shape=(video_height, video_width),
                    confidence=float(conf),
                    frame_number=int(frame_idx)
                )
                
                # Combined quality score
                combined_score = quality['confidence'] * 0.6 + quality['visibility_score'] * 0.4
                
                person_crops.append({
                    'crop': crop,
                    'bbox': bbox,
                    'frame_number': int(frame_idx),
                    'confidence': float(conf),
                    'quality': quality,
                    'combined_score': combined_score
                })
        
        if len(person_crops) == 0:
            logger.warning(f"Person {person_id}: No crops extracted")
            continue
        
        # ========== 3-BIN CONTIGUOUS SELECTION ==========
        person_crops.sort(key=lambda x: x['frame_number'])
        
        total_crops = len(person_crops)
        bin_size = total_crops // 3
        crops_per_bin = crops_per_person // 3  # 20 crops per bin
        
        bins = [
            person_crops[:bin_size],
            person_crops[bin_size:2*bin_size],
            person_crops[2*bin_size:]
        ]
        
        selected_crops = []
        for bin_crops in bins:
            if len(bin_crops) == 0:
                continue
            
            bin_len = len(bin_crops)
            if bin_len <= crops_per_bin:
                selected_crops.extend(bin_crops)
            else:
                mid_point = bin_len // 2
                start_idx = max(0, mid_point - crops_per_bin // 2)
                end_idx = min(bin_len, start_idx + crops_per_bin)
                selected_crops.extend(bin_crops[start_idx:end_idx])
        
        crops_with_quality.append({
            'person_id': person_id,
            'crops': selected_crops,
            'total_available': len(person_crops),
            'selected_count': len(selected_crops)
        })
        
        if verbose:
            logger.verbose_info(f"Person {person_id}: {len(selected_crops)}/{len(person_crops)} crops selected")
    
    cap.release()
    t_extraction_elapsed = time.time() - t_extraction_start
    
    # Save to final_crops.pkl
    final_crops_path = Path(crops_file)
    final_crops_path.parent.mkdir(parents=True, exist_ok=True)
    
    t_save_crops_start = time.time()
    output_data = {
        'crops_with_quality': crops_with_quality,
        'video_source': str(video_path),
        'crops_per_person': crops_per_person,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    
    with open(final_crops_path, 'wb') as f:
        pickle.dump(output_data, f)
    t_save_crops = time.time() - t_save_crops_start
    
    t_crop_total = time.time() - t_crop_start
    
    # Calculate extraction rate
    extraction_fps = total_extractions / t_extraction_elapsed if t_extraction_elapsed > 0 else 0
    
    logger.info(f"✓ Extracted crops for {len(crops_with_quality)} persons")
    logger.stat("Output crops file", str(final_crops_path))
    
    # Detailed timing breakdown
    print(f"\n   ⏱️  TIMING BREAKDOWN:")
    print(f"       Video open:        {t_video_open:.3f}s")
    print(f"       Extraction loop:   {t_extraction_elapsed:.3f}s")
    print(f"       Saving PKL:        {t_save_crops:.3f}s")
    print(f"       Total crop stage:  {t_crop_total:.3f}s")
    print(f"\n   📊 EXTRACTION STATS:")
    print(f"       Total seeks:       {total_seeks}")
    print(f"       Successful crops:  {total_extractions}")
    print(f"       Extraction FPS:    {extraction_fps:.1f}")
    
    logger.timing("Total crop extraction", t_crop_total)
    
    # ==================== Write Timings with Experiment Details ====================
    sidecar_path = Path(filtered_file).parent / (Path(filtered_file).name + '.timings.json')
    sidecar = {
        'experiment': 'on_demand_extraction',
        'filtered_persons_file': str(filtered_file),
        'final_crops_file': str(crops_file),
        'input_total_persons': int(len(all_persons)),
        'output_final_persons': int(len(selected_persons)),
        'ranking_time': float(t_ranking),
        'npz_save_time': float(npz_save_time),
        'crops_per_person': int(crops_per_person),
        'crop_extraction_breakdown': {
            'video_open_time': float(t_video_open),
            'extraction_loop_time': float(t_extraction_elapsed),
            'save_pkl_time': float(t_save_crops),
            'total_crop_time': float(t_crop_total),
            'total_seeks': int(total_seeks),
            'successful_extractions': int(total_extractions),
            'extraction_fps': float(extraction_fps)
        },
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    with open(sidecar_path, 'w', encoding='utf-8') as sf:
        json.dump(sidecar, sf, indent=2)
    
    logger.success()


def main():
    parser = argparse.ArgumentParser(description='Stage 3c NEW: Filter Persons (On-Demand Crops)')
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
        logger = PipelineLogger("Stage 3c NEW: Filter Persons", verbose=False)
        logger.info("Stage 3c is disabled in config")
        return
    
    # Run filtering
    run_filter(config)


if __name__ == '__main__':
    main()
