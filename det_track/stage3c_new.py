#!/usr/bin/env python3
"""
Stage 3c NEW: Fast Crop Extraction (Sequential, No Quality Scoring)

Simplified approach:
- Processes only 60% of video frames sequentially (no seeking)
- Collects up to 120 crops per person (max)
- No quality scoring - just collect crops as we encounter them
- Stops when all buckets have 120 crops OR 60% of video reached

Input: canonical_persons_3c.npz (10 filtered persons from original stage3c)
Output: final_crops_3c_new.pkl (crops for visualization)

This is a TEST version to benchmark on-demand extraction speed.
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


def run_stage3c_new(config_path, verbose=True):
    """
    Main entry point for Stage 3c NEW (fast crop extraction)
    """
    print("\n" + "="*70)
    print("📍 STAGE 3C NEW: FAST CROP EXTRACTION (SEQUENTIAL)")
    print("="*70)
    
    t_start = time.time()
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    config = resolve_path_variables(config)
    
    # Get paths
    canonical_video = config['stage0_normalize']['output']['canonical_video_file']
    canonical_persons_filtered_file = config['stage3c_filter']['output']['canonical_persons_filtered_file']
    output_dir = Path(config['global']['output_dir'])
    
    # NEW output file (don't overwrite existing)
    crops_output = output_dir / 'final_crops_3c_new.pkl'
    
    # Load filtered persons (10 persons from original stage3c)
    if verbose:
        print(f"\n   📂 Loading filtered persons: {Path(canonical_persons_filtered_file).name}")
    
    data = np.load(canonical_persons_filtered_file, allow_pickle=True)
    persons = data['persons']
    
    if verbose:
        print(f"   ✅ Loaded {len(persons)} persons")
    
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
    
    with open(crops_output, 'wb') as f:
        pickle.dump(crop_buckets, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    t_save = time.time() - t_save_start
    crops_size_mb = crops_output.stat().st_size / (1024 * 1024)
    
    if verbose:
        print(f"   ✅ Saved {crops_size_mb:.1f} MB in {t_save:.2f}s")
    
    t_total = time.time() - t_start
    
    # Summary
    print(f"\n   ⏱️  TIMING:")
    print(f"      Extraction: {extraction_time:.2f}s")
    print(f"      Saving: {t_save:.2f}s")
    print(f"      Total: {t_total:.2f}s")
    
    print(f"\n   ✅ Stage 3c NEW completed in {t_total:.2f}s")
    print("="*70)
    
    return t_total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 3c NEW: Fast Crop Extraction')
    parser.add_argument('--config', type=str, required=True, help='Path to pipeline config YAML')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    
    args = parser.parse_args()
    
    run_stage3c_new(args.config, verbose=args.verbose)
