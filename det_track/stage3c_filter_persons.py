#!/usr/bin/env python3
"""
Stage 3c: Person Filtering & Fast Crop Extraction

Optimized approach with on-demand crop extraction:
- Filters canonical persons to TOP 10 based on duration
- Processes only 60% of video frames sequentially (no seeking)
- Collects up to 120 crops per person (max)
- Simplified selection - no complex quality scoring
- Stops when all buckets have 120 crops OR 60% of video reached

Input: canonical_persons.npz (from Stage 3b)
Output: canonical_persons_3c.npz (filtered persons), final_crops_3c.pkl (crops for Stage 4)

This replaces the old stage3c with a much faster on-demand extraction approach.
"""

import argparse
import yaml
import numpy as np
import time
import sys
import cv2
import pickle
import re
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def resolve_path_variables(config):
    """Recursively resolve ${variable} in config"""
    global_vars = config.get('global', {})
    
    def resolve_string_once(s, vars_dict):
        if not isinstance(s, str):
            return s
        pattern = re.compile(r'\$\{([^}]+)\}')
        matches = pattern.findall(s)
        for var in matches:
            if var in vars_dict:
                s = s.replace(f'${{{var}}}', str(vars_dict[var]))
        return s
    
    max_iterations = 10
    for _ in range(max_iterations):
        old_vars = global_vars.copy()
        for key, value in global_vars.items():
            global_vars[key] = resolve_string_once(value, global_vars)
        if old_vars == global_vars:
            break
    
    def resolve_recursive(obj, vars_dict):
        if isinstance(obj, dict):
            return {k: resolve_recursive(v, vars_dict) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_recursive(item, vars_dict) for item in obj]
        elif isinstance(obj, str):
            return resolve_string_once(obj, vars_dict)
        else:
            return obj
    
    return resolve_recursive(config, global_vars)


def build_frame_to_persons_map(persons):
    """
    Build mapping: frame_num -> [(person_id, bbox_idx), ...]
    
    Returns:
        dict: {frame_num: [(person_id, bbox_idx), ...]}
    """
    frame_map = defaultdict(list)
    
    for person in persons:
        person_id = person['person_id']
        for bbox_idx, frame_num in enumerate(person['frame_numbers']):
            frame_map[frame_num].append((person_id, bbox_idx))
    
    return frame_map


def extract_crops_sequential(persons, video_path, max_crops_per_person=120, video_percent=0.6, verbose=True):
    """
    Extract crops sequentially from video (0% to 60% only).
    
    Strategy:
    - Process frames 0 to 60% of video
    - For each frame, check if any person appears
    - Extract crops and add to buckets
    - Stop filling a bucket when it reaches 120 crops
    - Resize crops like Stage 1 (max dimension = 192px)
    
    Args:
        persons: List of person dicts with frame_numbers, bboxes, person_id
        video_path: Path to canonical video
        max_crops_per_person: Stop filling bucket when reached (default 120)
        video_percent: Process only this fraction of video (default 0.6 = 60%)
        verbose: Print progress
    
    Returns:
        dict: {person_id: [{'crop': img, 'frame_idx': int, 'bbox': [...], ...}, ...]}
    """
    t_start = time.time()
    
    # Build frame → persons mapping
    if verbose:
        print(f"   ⚙️  Building frame map...")
    frame_map = build_frame_to_persons_map(persons)
    
    # Initialize crop buckets
    crop_buckets = {p['person_id']: [] for p in persons}
    bucket_full = {p['person_id']: False for p in persons}
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frame = int(total_frames * video_percent)
    
    if verbose:
        print(f"   🎬 Video: {width}x{height}, {total_frames} frames")
        print(f"   📏 Processing: frames 0-{max_frame} ({video_percent*100:.0f}% of video)")
        print(f"   🪣 Target: up to {max_crops_per_person} crops per person")
    
    # Sequential extraction
    t_extract_start = time.time()
    
    pbar = tqdm(
        total=max_frame,
        desc="   🔍 Extracting crops",
        disable=not verbose,
        mininterval=1.0
    )
    
    frame_idx = 0
    crops_extracted = 0
    frames_with_crops = 0
    
    while frame_idx < max_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if this frame has any persons of interest
        if frame_idx in frame_map:
            frames_with_crops += 1
            
            # Extract crops for all persons in this frame
            for person_id, bbox_idx in frame_map[frame_idx]:
                # Skip if bucket already full
                if bucket_full[person_id]:
                    continue
                
                # Get bbox for this person at this frame
                person = next(p for p in persons if p['person_id'] == person_id)
                bbox = person['bboxes'][bbox_idx]
                confidence = person['confidences'][bbox_idx]
                
                # Extract crop with boundary clamping
                x1, y1, x2, y2 = map(int, bbox)
                x1 = max(0, min(x1, width))
                x2 = max(0, min(x2, width))
                y1 = max(0, min(y1, height))
                y2 = max(0, min(y2, height))
                
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2].copy()
                    
                    # Resize maintaining aspect ratio (max dimension = 192px)
                    h, w = crop.shape[:2]
                    max_dim = 192
                    if max(h, w) > max_dim:
                        scale = max_dim / max(h, w)
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        crop_resized = cv2.resize(crop, (new_w, new_h))
                    else:
                        crop_resized = crop  # Don't upscale small crops
                    
                    # Add to bucket
                    crop_buckets[person_id].append({
                        'crop': crop_resized,
                        'frame_idx': frame_idx,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(confidence)
                    })
                    crops_extracted += 1
                    
                    # Mark bucket as full if threshold reached
                    if len(crop_buckets[person_id]) >= max_crops_per_person:
                        bucket_full[person_id] = True
        
        frame_idx += 1
        pbar.update(1)
        
        # Early exit if all buckets full
        if all(bucket_full.values()):
            if verbose:
                print(f"\n   ✅ All buckets full at frame {frame_idx}/{max_frame}")
            break
    
    pbar.close()
    cap.release()
    
    t_extract_end = time.time()
    t_extract = t_extract_end - t_extract_start
    
    # Report statistics
    if verbose:
        print(f"\n   📊 Extraction stats:")
        print(f"      Frames processed: {frame_idx}/{max_frame}")
        print(f"      Frames with crops: {frames_with_crops}")
        print(f"      Total crops extracted: {crops_extracted}")
        print(f"      Extraction time: {t_extract:.2f}s")
        print(f"\n   🪣 Bucket sizes:")
        for person_id in sorted(crop_buckets.keys()):
            count = len(crop_buckets[person_id])
            status = "✅ FULL" if bucket_full[person_id] else "⚠️ partial"
            print(f"      Person {person_id}: {count} crops {status}")
    
    t_total = time.time() - t_start
    
    return crop_buckets, t_total


def filter_top_persons(all_persons, top_n=10, min_duration_frames=150, verbose=True):
    """
    Filter to top N persons based on duration.
    Simplified version - just picks longest-appearing persons.
    
    Args:
        all_persons: List of person dicts from canonical_persons.npz
        top_n: Number of top persons to keep (default 10)
        min_duration_frames: Minimum frames to be considered (default 150)
        verbose: Print filtering info
    
    Returns:
        List of filtered person dicts (top N)
    """
    # Filter by minimum duration
    candidates = [p for p in all_persons if len(p['frame_numbers']) >= min_duration_frames]
    
    if verbose:
        print(f"   🔍 Filtering: {len(all_persons)} → {len(candidates)} persons (min {min_duration_frames} frames)")
    
    # Sort by duration (number of frames)
    candidates_sorted = sorted(candidates, key=lambda p: len(p['frame_numbers']), reverse=True)
    
    # Take top N
    top_persons = candidates_sorted[:top_n]
    
    if verbose:
        print(f"   ✅ Selected top {len(top_persons)} persons by duration")
        for i, p in enumerate(top_persons[:5]):  # Show top 5
            duration = len(p['frame_numbers'])
            print(f"      #{i+1}: Person {p['person_id']} - {duration} frames")
        if len(top_persons) > 5:
            print(f"      ... and {len(top_persons)-5} more")
    
    return top_persons


def run_filter(config_path, verbose=True):
    """
    Main entry point for Stage 3c (filter + fast crop extraction)
    """
    print("\n" + "="*70)
    print("📍 STAGE 3C: FILTER & FAST CROP EXTRACTION")
    print("="*70)
    
    t_start = time.time()
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Auto-extract current_video from video_file (needed for path resolution)
    video_file = config.get('global', {}).get('video_file', '')
    if video_file:
        import os
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        config['global']['current_video'] = video_name
    
    config = resolve_path_variables(config)
    
    # Get paths - use canonical_persons.npz from stage3b (unfiltered)
    canonical_video = config['stage0_normalize']['output']['canonical_video_file']
    canonical_persons_file = config['stage3b_group']['output']['canonical_persons_file']
    
    # Output directory
    output_dir = Path(canonical_persons_file).parent
    
    # Output files (standard names for Stage 4 compatibility)
    crops_output = output_dir / 'final_crops_3c.pkl'
    persons_output = output_dir / 'canonical_persons_3c.npz'
    
    # Load all canonical persons (from stage3b)
    if verbose:
        print(f"\n   📂 Loading canonical persons: {Path(canonical_persons_file).name}")
    
    data = np.load(canonical_persons_file, allow_pickle=True)
    all_persons = list(data['persons'])
    
    if verbose:
        print(f"   ✅ Loaded {len(all_persons)} persons")
    
    # Filter to top 10
    if verbose:
        print(f"\n   🔍 Filtering to top 10 persons...")
    
    persons = filter_top_persons(all_persons, top_n=10, min_duration_frames=150, verbose=verbose)
    
    # Extract crops sequentially
    if verbose:
        print(f"\n   🎬 Opening video: {Path(canonical_video).name}")
    
    crop_buckets, extraction_time = extract_crops_sequential(
        persons=persons,
        video_path=canonical_video,
        max_crops_per_person=120,
        video_percent=0.6,
        verbose=verbose
    )
    
    # Save crops
    t_save_start = time.time()
    
    if verbose:
        print(f"\n   💾 Saving crops to: {crops_output.name}")
    
    # Convert crop_buckets to Stage 4 compatible format (crops_with_quality)
    crops_with_quality = []
    for person_id, crops_list in crop_buckets.items():
        if len(crops_list) > 0:  # Only include persons with crops
            crops_with_quality.append({
                'person_id': person_id,
                'crops': crops_list  # List of dicts with 'crop', 'frame_idx', 'bbox', 'confidence'
            })
    
    # Save in format compatible with Stage 4
    from datetime import datetime, timezone
    output_data = {
        'crops_with_quality': crops_with_quality,
        'video_source': str(canonical_video),
        'crops_per_person': 120,  # Max crops collected
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    
    with open(crops_output, 'wb') as f:
        pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    t_save_crops = time.time() - t_save_start
    crops_size_mb = crops_output.stat().st_size / (1024 * 1024)
    
    if verbose:
        print(f"   ✅ Saved {crops_size_mb:.1f} MB in {t_save_crops:.2f}s")
    
    # Save filtered persons NPZ
    if verbose:
        print(f"\n   💾 Saving filtered persons to: {persons_output.name}")
    
    np.savez_compressed(persons_output, persons=persons)
    
    t_save_npz = time.time() - t_save_start - t_save_crops
    
    if verbose:
        print(f"   ✅ Saved filtered persons ({len(persons)} persons)")
    
    t_total = time.time() - t_start
    
    # Summary
    print(f"\n   ⏱️  TIMING:")
    print(f"      Filtering: <0.01s")
    print(f"      Extraction: {extraction_time:.2f}s")
    print(f"      Saving: {t_save_crops + t_save_npz:.2f}s")
    print(f"      Total: {t_total:.2f}s")
    
    print(f"\n   ✅ Stage 3c completed in {t_total:.2f}s")
    print("="*70)
    
    return t_total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 3c: Filter Persons & Fast Crop Extraction')
    parser.add_argument('--config', type=str, required=True, help='Path to pipeline config YAML')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    parser.add_argument('--crops-per-person', type=int, default=120, help='Max crops per person (legacy arg, ignored - always 120)')
    
    args = parser.parse_args()
    
    # Note: --crops-per-person is ignored in the new implementation (always collects up to 120 crops)
    # It's kept for backward compatibility with run_pipeline.py
    run_filter(args.config, verbose=args.verbose)
