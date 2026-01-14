#!/usr/bin/env python3
"""
Proof-of-Concept: On-Demand Crop Extraction

Tests the linear-pass algorithm for extracting crops directly from video
instead of loading from crops_by_person.pkl.

This script:
1. Loads canonical_persons.npz (has bboxes for top 10 persons)
2. Does a single linear pass through the video
3. Extracts crops on-demand for the top 10 persons
4. Generates WebP animations
5. Compares timing with the old approach

Usage:
    # Local Windows (download canonical_persons.npz from Drive first)
    python test_ondemand_extraction.py --video kohli_nets.mp4 --data canonical_persons.npz

    # Or use full paths
    python test_ondemand_extraction.py \
        --video D:/trials/unifiedpipeline/newrepo/demo_data/videos/kohli_nets.mp4 \
        --data D:/path/to/canonical_persons.npz \
        --output D:/trials/unifiedpipeline/newrepo/det_track/test_output/
"""

import argparse
import numpy as np
import cv2
import time
from pathlib import Path
import json


def load_canonical_persons(data_file):
    """Load canonical persons data from npz file"""
    print(f"\n Loading canonical persons data...")
    start = time.time()
    
    data = np.load(data_file, allow_pickle=True)
    persons = data['persons']
    
    elapsed = time.time() - start
    print(f"    Loaded {len(persons)} persons in {elapsed:.3f}s")
    
    return persons


def prepare_extraction_plan(persons, target_crops_per_person=50):
    """
    Prepare the extraction plan: which frames to visit and what to extract.
    
    Returns:
        frame_to_persons: dict mapping frame_num  list of (person_id, bbox)
        person_buckets: dict mapping person_id  empty list (to fill)
    """
    print(f"\n  Preparing extraction plan...")
    
    frame_to_persons = {}  # frame_num  [(person_id, bbox), ...]
    person_buckets = {}    # person_id  []
    person_targets = {}    # person_id  target_count
    
    # Sort persons by frame count (descending) to prioritize main persons
    persons_sorted = sorted(persons, key=lambda p: len(p['frame_numbers']), reverse=True)
    top_10 = persons_sorted[:10]
    
    print(f"   Top 10 persons:")
    for i, person in enumerate(top_10):
        person_id = int(person['person_id'])
        frame_count = len(person['frame_numbers'])
        
        # Determine target count (min of available frames and target)
        target_count = min(frame_count, target_crops_per_person)
        
        print(f"   {i+1}. Person {person_id}: {frame_count} frames, will extract {target_count} crops")
        
        person_buckets[person_id] = []
        person_targets[person_id] = target_count
        
        # Map frames to person_id and bbox
        for frame_num, bbox in zip(person['frame_numbers'], person['bboxes']):
            frame_num = int(frame_num)
            if frame_num not in frame_to_persons:
                frame_to_persons[frame_num] = []
            frame_to_persons[frame_num].append((person_id, bbox))
    
    # Calculate frame range
    all_frames = sorted(frame_to_persons.keys())
    frame_range = (all_frames[0], all_frames[-1]) if all_frames else (0, 0)
    
    print(f"\n    Extraction plan:")
    print(f"      Frames with targets: {len(frame_to_persons)}")
    print(f"      Frame range: {frame_range[0]} - {frame_range[1]}")
    print(f"      Total potential crops: {sum(len(v) for v in frame_to_persons.values())}")
    
    return frame_to_persons, person_buckets, person_targets


def create_optimal_seeking_plan(persons, target_crops_per_person=50):
    """
    PHASE 1: Analyze NPZ to determine MINIMAL frames to seek.
    
    This plans extraction WITHOUT touching the video.
    Returns the optimal set of frames to seek for maximum efficiency.
    """
    print(f"\n  Creating optimal seeking plan...")
    
    # Sort persons by frame count
    persons_sorted = sorted(persons, key=lambda p: len(p['frame_numbers']), reverse=True)
    top_10 = persons_sorted[:10]
    
    frames_to_seek = set()  # Deduplicated set of frames
    person_frame_selections = {}  # person_id -> list of selected frames
    person_bboxes_map = {}  # (person_id, frame_num) -> bbox
    
    for i, person in enumerate(top_10):
        person_id = int(person['person_id'])
        available_frames = sorted(zip(person['frame_numbers'], person['bboxes']))
        
        # Select frames for this person
        if len(available_frames) >= target_crops_per_person:
            # Strategy: Take first 50 chronologically (simple, deterministic)
            selected = available_frames[:target_crops_per_person]
        else:
            # Need all available frames
            selected = available_frames
        
        selected_frames = [int(f) for f, b in selected]
        person_frame_selections[person_id] = set(selected_frames)  # Use SET for O(1) lookup!
        
        # Store bbox mapping
        for frame_num, bbox in selected:
            person_bboxes_map[(person_id, int(frame_num))] = bbox
        
        # Add to set (auto-deduplicates)
        frames_to_seek.update(selected_frames)
        
        print(f"   {i+1}. Person {person_id}: selected {len(selected_frames)} frames "
              f"(range: {selected_frames[0]}-{selected_frames[-1]})")
    
    # Sort for sequential seeking (decoder-friendly)
    optimal_seek_order = sorted(frames_to_seek)
    
    print(f"\n    Optimal seeking plan:")
    print(f"      Total unique frames to seek: {len(optimal_seek_order)}")
    print(f"      Frame range: {optimal_seek_order[0]} - {optimal_seek_order[-1]}")
    print(f"      Deduplication saved: {sum(len(v) for v in person_frame_selections.values()) - len(optimal_seek_order)} seeks")
    
    return optimal_seek_order, person_frame_selections, person_bboxes_map


def extract_crops_optimal_seeking(video_path, optimal_seek_order, person_frame_selections, person_bboxes_map, person_targets):
    """
    PHASE 2: Extract crops by seeking only to planned frames.
    
    Much faster than linear pass when we know exact frames needed.
    """
    print(f"\n  Starting optimal seeking extraction...")
    
    start_time = time.time()
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"   Video: {total_frames} frames @ {fps:.2f} FPS")
    print(f"   Will seek to {len(optimal_seek_order)} frames")
    
    # Initialize buckets
    person_buckets = {pid: [] for pid in person_frame_selections}
    
    # Stats
    frames_seeked = 0
    crops_extracted = 0
    last_progress_time = time.time()
    
    # Seek to each planned frame
    for frame_num in optimal_seek_order:
        # Seek to specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            print(f"   Warning: Failed to read frame {frame_num}")
            continue
        
        frames_seeked += 1
        
        # Extract crops for all persons that need this frame
        for person_id, needed_frames in person_frame_selections.items():
            if frame_num in needed_frames:
                # Get bbox for this person at this frame
                bbox = person_bboxes_map.get((person_id, frame_num))
                if bbox is None:
                    continue
                
                # Extract crop
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Clamp to frame boundaries
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2].copy()
                    person_buckets[person_id].append(crop)
                    crops_extracted += 1
        
        # Progress reporting (every 2 seconds)
        now = time.time()
        if now - last_progress_time > 2.0:
            progress_pct = (frames_seeked / len(optimal_seek_order)) * 100
            filled_buckets = sum(1 for bucket in person_buckets.values() 
                               if len(bucket) >= person_targets[list(person_targets.keys())[0]])
            print(f"   Progress: {frames_seeked}/{len(optimal_seek_order)} frames ({progress_pct:.1f}%), "
                  f"{crops_extracted} crops, {filled_buckets}/10 buckets filled")
            last_progress_time = now
    
    cap.release()
    
    elapsed = time.time() - start_time
    fps_processing = frames_seeked / elapsed if elapsed > 0 else 0
    
    print(f"\n    Optimal seeking complete:")
    print(f"      Frames seeked: {frames_seeked}/{len(optimal_seek_order)}")
    print(f"      Crops extracted: {crops_extracted}")
    print(f"      Time: {elapsed:.2f}s")
    print(f"      Effective speed: {fps_processing:.1f} seeks/sec")
    
    # Show bucket status
    print(f"\n    Bucket fill status:")
    for person_id, crops in person_buckets.items():
        target = person_targets[person_id]
        status = "" if len(crops) >= target else ""
        print(f"      {status} Person {person_id}: {len(crops)}/{target} crops")
    
    return person_buckets, {
        'frames_seeked': frames_seeked,
        'total_planned_seeks': len(optimal_seek_order),
        'crops_extracted': crops_extracted,
        'elapsed_seconds': elapsed,
        'seeks_per_second': fps_processing
    }


def extract_crops_linear_pass(video_path, frame_to_persons, person_buckets, person_targets):
    """
    Linear pass through video extracting crops on-demand.
    
    This is the core algorithm:
    - Open video once
    - Read frames sequentially (fast!)
    - Extract crops for multiple persons per frame
    - Early termination when all buckets filled
    """
    print(f"\n Starting linear pass extraction...")
    
    start_time = time.time()
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"   Video: {total_frames} frames @ {fps:.2f} FPS")
    
    # Stats
    frames_processed = 0
    crops_extracted = 0
    frames_with_targets = sorted(frame_to_persons.keys())
    next_target_idx = 0
    
    # Read frames sequentially
    frame_num = 0
    last_progress_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames_processed += 1
        
        # Check if this frame has targets
        if frame_num in frame_to_persons:
            persons_in_frame = frame_to_persons[frame_num]
            
            for person_id, bbox in persons_in_frame:
                # Check if this person's bucket is already full
                if len(person_buckets[person_id]) >= person_targets[person_id]:
                    continue
                
                # Extract crop
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Clamp to frame boundaries
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2].copy()
                    person_buckets[person_id].append(crop)
                    crops_extracted += 1
        
        frame_num += 1
        
        # Progress reporting (every 2 seconds)
        now = time.time()
        if now - last_progress_time > 2.0:
            filled_buckets = sum(1 for bucket in person_buckets.values() 
                               if len(bucket) >= person_targets[list(person_targets.keys())[0]])
            progress_pct = (frames_processed / total_frames) * 100
            print(f"   Progress: {frames_processed}/{total_frames} frames ({progress_pct:.1f}%), "
                  f"{crops_extracted} crops, {filled_buckets}/10 buckets filled")
            last_progress_time = now
        
        # Early termination check (every 100 frames)
        if frames_processed % 100 == 0:
            all_filled = all(len(person_buckets[pid]) >= person_targets[pid] 
                           for pid in person_targets)
            if all_filled:
                print(f"\n    Early termination! All buckets filled at frame {frame_num}")
                break
    
    cap.release()
    
    elapsed = time.time() - start_time
    fps_processing = frames_processed / elapsed if elapsed > 0 else 0
    
    print(f"\n    Linear pass complete:")
    print(f"      Frames processed: {frames_processed}/{total_frames} ({frames_processed/total_frames*100:.1f}%)")
    print(f"      Crops extracted: {crops_extracted}")
    print(f"      Time: {elapsed:.2f}s")
    print(f"      Processing speed: {fps_processing:.1f} FPS")
    
    # Show bucket status
    print(f"\n    Bucket fill status:")
    for person_id, crops in person_buckets.items():
        target = person_targets[person_id]
        status = "" if len(crops) >= target else ""
        print(f"      {status} Person {person_id}: {len(crops)}/{target} crops")
    
    return person_buckets, {
        'frames_processed': frames_processed,
        'total_frames': total_frames,
        'crops_extracted': crops_extracted,
        'elapsed_seconds': elapsed,
        'processing_fps': fps_processing
    }


def generate_webps(person_buckets, output_dir, duration_per_frame=100):
    """Generate WebP animations from extracted crops"""
    print(f"\n Generating WebP animations...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    webp_count = 0
    
    for person_id, crops in person_buckets.items():
        if len(crops) == 0:
            continue
        
        # Generate WebP
        output_path = output_dir / f"person_{person_id:03d}.webp"
        
        # Resize all crops to a consistent size (256x256 for speed)
        target_size = (256, 256)
        
        try:
            import imageio
            # Resize crops to consistent dimensions and convert BGR to RGB
            crops_resized = []
            for crop in crops:
                resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                crops_resized.append(rgb)
            
            # Create WebP animation
            imageio.mimsave(str(output_path), crops_resized, 
                          format='WEBP', duration=duration_per_frame, loop=0)
            webp_count += 1
            print(f"    Person {person_id}: {len(crops)} frames  {output_path.name}")
        except Exception as e:
            print(f"     Person {person_id}: Failed to create WebP ({e})")
    
    elapsed = time.time() - start_time
    print(f"\n    Generated {webp_count} WebP animations in {elapsed:.2f}s")
    
    return {
        'webp_count': webp_count,
        'elapsed_seconds': elapsed
    }


def compare_with_old_approach(timing_data):
    """
    Compare with the old approach timings.
    
    Old approach (from Phase 1 results):
    - Stage 4b save: 4.55s
    - Stage 10b load: 6.15s
    - Total overhead: 10.7s
    
    Note: We should also account for crop extraction during Stage 1
    """
    print(f"\n Performance Comparison:")
    print(f"=" * 70)
    
    # Old approach
    old_stage4b_save = 4.55
    old_stage10b_load = 6.15
    old_total = old_stage4b_save + old_stage10b_load
    old_storage = 812  # MB
    
    print(f"\n   OLD APPROACH (crops_by_person.pkl):")
    print(f"      Stage 4b save:    {old_stage4b_save:.2f}s")
    print(f"      Stage 10b load:   {old_stage10b_load:.2f}s")
    print(f"      Total overhead:   {old_total:.2f}s")
    print(f"      Storage:          {old_storage} MB")
    
    # New approach
    new_extraction = timing_data['extraction']['elapsed_seconds']
    new_webp_gen = timing_data['webp_generation']['elapsed_seconds']
    new_total = new_extraction + new_webp_gen
    new_storage = 1  # MB (just bboxes)
    
    print(f"\n   NEW APPROACH (on-demand extraction):")
    print(f"      Linear pass:      {new_extraction:.2f}s")
    print(f"      WebP generation:  {new_webp_gen:.2f}s")
    print(f"      Total time:       {new_total:.2f}s")
    print(f"      Storage:          {new_storage} MB")
    
    # Comparison
    time_diff = old_total - new_total
    time_pct = (time_diff / old_total) * 100 if old_total > 0 else 0
    storage_diff = old_storage - new_storage
    storage_pct = (storage_diff / old_storage) * 100 if old_storage > 0 else 0
    
    print(f"\n   IMPROVEMENT:")
    print(f"      Time:    {time_diff:+.2f}s ({time_pct:+.1f}%)")
    print(f"      Storage: {storage_diff:+.0f} MB ({storage_pct:+.1f}%)")
    
    print(f"\n" + "=" * 70)
    
    return {
        'old_time': old_total,
        'new_time': new_total,
        'time_saved': time_diff,
        'old_storage_mb': old_storage,
        'new_storage_mb': new_storage,
        'storage_saved_mb': storage_diff
    }


def main():
    parser = argparse.ArgumentParser(description='Test on-demand crop extraction')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--data', required=True, help='Path to canonical_persons.npz')
    parser.add_argument('--output', default='./test_output/', help='Output directory for WebPs')
    parser.add_argument('--crops-per-person', type=int, default=50, 
                       help='Target crops per person (default: 50)')
    parser.add_argument('--method', choices=['linear', 'optimal', 'both'], default='both',
                       help='Extraction method: linear pass, optimal seeking, or both (default: both)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("[TEST] PROOF-OF-CONCEPT: On-Demand Crop Extraction")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"   Video: {args.video}")
    print(f"   Data:  {args.data}")
    print(f"   Output: {args.output}")
    print(f"   Target crops/person: {args.crops_per_person}")
    print(f"   Method: {args.method}")
    
    # Check files exist
    if not Path(args.video).exists():
        print(f"\n ERROR: Video not found: {args.video}")
        return 1
    
    if not Path(args.data).exists():
        print(f"\n ERROR: Data file not found: {args.data}")
        return 1
    
    # Load canonical persons
    persons = load_canonical_persons(args.data)
    
    results = {}
    
    # ===== LINEAR PASS APPROACH =====
    if args.method in ['linear', 'both']:
        print("\n" + "=" * 70)
        print("[APPROACH 1] LINEAR PASS")
        print("=" * 70)
        
        # Prepare extraction plan
        frame_to_persons, person_buckets, person_targets = prepare_extraction_plan(
            persons, args.crops_per_person
        )
        
        # Extract crops via linear pass
        person_buckets_linear, extraction_timing_linear = extract_crops_linear_pass(
            args.video, frame_to_persons, person_buckets, person_targets
        )
        
        # Generate WebPs
        output_linear = Path(args.output) / 'linear'
        webp_timing_linear = generate_webps(person_buckets_linear, output_linear)
        
        results['linear'] = {
            'extraction': extraction_timing_linear,
            'webp_generation': webp_timing_linear,
            'total_time': extraction_timing_linear['elapsed_seconds'] + webp_timing_linear['elapsed_seconds']
        }
        
        print(f"\n   LINEAR PASS SUMMARY:")
        print(f"      Frames processed: {extraction_timing_linear['frames_processed']}/{extraction_timing_linear['total_frames']}")
        print(f"      Extraction time: {extraction_timing_linear['elapsed_seconds']:.2f}s @ {extraction_timing_linear['processing_fps']:.1f} FPS")
        print(f"      WebP generation: {webp_timing_linear['elapsed_seconds']:.2f}s")
        print(f"      TOTAL TIME: {results['linear']['total_time']:.2f}s")
    
    # ===== OPTIMAL SEEKING APPROACH =====
    if args.method in ['optimal', 'both']:
        print("\n" + "=" * 70)
        print("[APPROACH 2] OPTIMAL SEEKING")
        print("=" * 70)
        
        # Create optimal seeking plan
        optimal_seek_order, person_frame_selections, person_bboxes_map = create_optimal_seeking_plan(
            persons, args.crops_per_person
        )
        
        # Prepare person targets (same as linear)
        _, _, person_targets = prepare_extraction_plan(persons, args.crops_per_person)
        
        # Extract crops via optimal seeking
        person_buckets_optimal, extraction_timing_optimal = extract_crops_optimal_seeking(
            args.video, optimal_seek_order, person_frame_selections, person_bboxes_map, person_targets
        )
        
        # Generate WebPs
        output_optimal = Path(args.output) / 'optimal'
        webp_timing_optimal = generate_webps(person_buckets_optimal, output_optimal)
        
        results['optimal'] = {
            'extraction': extraction_timing_optimal,
            'webp_generation': webp_timing_optimal,
            'total_time': extraction_timing_optimal['elapsed_seconds'] + webp_timing_optimal['elapsed_seconds']
        }
        
        print(f"\n   OPTIMAL SEEKING SUMMARY:")
        print(f"      Frames seeked: {extraction_timing_optimal['frames_seeked']}/{extraction_timing_optimal['total_planned_seeks']}")
        print(f"      Extraction time: {extraction_timing_optimal['elapsed_seconds']:.2f}s @ {extraction_timing_optimal['seeks_per_second']:.1f} seeks/sec")
        print(f"      WebP generation: {webp_timing_optimal['elapsed_seconds']:.2f}s")
        print(f"      TOTAL TIME: {results['optimal']['total_time']:.2f}s")
    
    # ===== COMPARISON =====
    if args.method == 'both':
        print("\n" + "=" * 70)
        print("[COMPARISON] LINEAR vs OPTIMAL SEEKING")
        print("=" * 70)
        
        linear_time = results['linear']['total_time']
        optimal_time = results['optimal']['total_time']
        time_saved = linear_time - optimal_time
        speedup = linear_time / optimal_time if optimal_time > 0 else 0
        
        linear_frames = results['linear']['extraction']['frames_processed']
        optimal_frames = results['optimal']['extraction']['frames_seeked']
        frames_saved = linear_frames - optimal_frames
        
        print(f"\n   LINEAR PASS:")
        print(f"      Frames processed: {linear_frames}")
        print(f"      Total time: {linear_time:.2f}s")
        
        print(f"\n   OPTIMAL SEEKING:")
        print(f"      Frames seeked: {optimal_frames}")
        print(f"      Total time: {optimal_time:.2f}s")
        
        print(f"\n   WINNER: {'OPTIMAL SEEKING' if optimal_time < linear_time else 'LINEAR PASS'}")
        print(f"      Time saved: {abs(time_saved):.2f}s ({abs(time_saved/linear_time*100):.1f}%)")
        print(f"      Speedup: {speedup:.2f}x")
        print(f"      Frames saved: {frames_saved} ({frames_saved/linear_frames*100:.1f}%)")
    
    # ===== COMPARE WITH OLD APPROACH =====
    print("\n" + "=" * 70)
    print("[COMPARISON] NEW vs OLD APPROACH")
    print("=" * 70)
    
    # Use best method for comparison
    if args.method == 'both':
        best_method = 'optimal' if results['optimal']['total_time'] < results['linear']['total_time'] else 'linear'
        timing_data = results[best_method]
        print(f"\n   Using {best_method.upper()} for comparison (best performer)")
    elif args.method == 'linear':
        timing_data = results['linear']
    else:
        timing_data = results['optimal']
    
    comparison = compare_with_old_approach(timing_data)
    
    # Save all results
    results_file = Path(args.output) / 'timing_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'results': results,
            'comparison_with_old': comparison
        }, f, indent=2)
    
    print(f"\n Test complete! Results saved to: {results_file}")
    if args.method == 'both':
        print(f"   Linear WebPs: {output_linear}")
        print(f"   Optimal WebPs: {output_optimal}")
    elif args.method == 'linear':
        print(f"   WebP animations: {output_linear}")
    else:
        print(f"   WebP animations: {output_optimal}")
    
    return 0


if __name__ == '__main__':
    exit(main())
