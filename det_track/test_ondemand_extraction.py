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
    print(f"\n📦 Loading canonical persons data...")
    start = time.time()
    
    data = np.load(data_file, allow_pickle=True)
    persons = data['persons']
    
    elapsed = time.time() - start
    print(f"   ✅ Loaded {len(persons)} persons in {elapsed:.3f}s")
    
    return persons


def prepare_extraction_plan(persons, target_crops_per_person=50):
    """
    Prepare the extraction plan: which frames to visit and what to extract.
    
    Returns:
        frame_to_persons: dict mapping frame_num → list of (person_id, bbox)
        person_buckets: dict mapping person_id → empty list (to fill)
    """
    print(f"\n🗺️  Preparing extraction plan...")
    
    frame_to_persons = {}  # frame_num → [(person_id, bbox), ...]
    person_buckets = {}    # person_id → []
    person_targets = {}    # person_id → target_count
    
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
    
    print(f"\n   📊 Extraction plan:")
    print(f"      Frames with targets: {len(frame_to_persons)}")
    print(f"      Frame range: {frame_range[0]} - {frame_range[1]}")
    print(f"      Total potential crops: {sum(len(v) for v in frame_to_persons.values())}")
    
    return frame_to_persons, person_buckets, person_targets


def extract_crops_linear_pass(video_path, frame_to_persons, person_buckets, person_targets):
    """
    Linear pass through video extracting crops on-demand.
    
    This is the core algorithm:
    - Open video once
    - Read frames sequentially (fast!)
    - Extract crops for multiple persons per frame
    - Early termination when all buckets filled
    """
    print(f"\n🎬 Starting linear pass extraction...")
    
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
                print(f"\n   🎯 Early termination! All buckets filled at frame {frame_num}")
                break
    
    cap.release()
    
    elapsed = time.time() - start_time
    fps_processing = frames_processed / elapsed if elapsed > 0 else 0
    
    print(f"\n   ✅ Linear pass complete:")
    print(f"      Frames processed: {frames_processed}/{total_frames} ({frames_processed/total_frames*100:.1f}%)")
    print(f"      Crops extracted: {crops_extracted}")
    print(f"      Time: {elapsed:.2f}s")
    print(f"      Processing speed: {fps_processing:.1f} FPS")
    
    # Show bucket status
    print(f"\n   📊 Bucket fill status:")
    for person_id, crops in person_buckets.items():
        target = person_targets[person_id]
        status = "✅" if len(crops) >= target else "⚠️"
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
    print(f"\n🎨 Generating WebP animations...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    webp_count = 0
    
    for person_id, crops in person_buckets.items():
        if len(crops) == 0:
            continue
        
        # Generate WebP
        output_path = output_dir / f"person_{person_id:03d}.webp"
        
        # Use first crop to determine size
        h, w = crops[0].shape[:2]
        
        # Write WebP (using cv2)
        # Note: OpenCV might not support WebP animation directly
        # For now, let's save as individual images or use imageio
        try:
            import imageio
            # Convert BGR to RGB
            crops_rgb = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in crops]
            imageio.mimsave(str(output_path), crops_rgb, 
                          format='WEBP', duration=duration_per_frame, loop=0)
            webp_count += 1
            print(f"   ✅ Person {person_id}: {len(crops)} frames → {output_path.name}")
        except Exception as e:
            print(f"   ⚠️  Person {person_id}: Failed to create WebP ({e})")
    
    elapsed = time.time() - start_time
    print(f"\n   ✅ Generated {webp_count} WebP animations in {elapsed:.2f}s")
    
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
    print(f"\n📊 Performance Comparison:")
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
    args = parser.parse_args()
    
    print("=" * 70)
    print("🧪 PROOF-OF-CONCEPT: On-Demand Crop Extraction")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"   Video: {args.video}")
    print(f"   Data:  {args.data}")
    print(f"   Output: {args.output}")
    print(f"   Target crops/person: {args.crops_per_person}")
    
    # Check files exist
    if not Path(args.video).exists():
        print(f"\n❌ ERROR: Video not found: {args.video}")
        return 1
    
    if not Path(args.data).exists():
        print(f"\n❌ ERROR: Data file not found: {args.data}")
        return 1
    
    # Load canonical persons
    persons = load_canonical_persons(args.data)
    
    # Prepare extraction plan
    frame_to_persons, person_buckets, person_targets = prepare_extraction_plan(
        persons, args.crops_per_person
    )
    
    # Extract crops via linear pass
    person_buckets, extraction_timing = extract_crops_linear_pass(
        args.video, frame_to_persons, person_buckets, person_targets
    )
    
    # Generate WebPs
    webp_timing = generate_webps(person_buckets, args.output)
    
    # Compare with old approach
    timing_data = {
        'extraction': extraction_timing,
        'webp_generation': webp_timing
    }
    comparison = compare_with_old_approach(timing_data)
    
    # Save timing results
    results_file = Path(args.output) / 'timing_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'extraction': extraction_timing,
            'webp_generation': webp_timing,
            'comparison': comparison
        }, f, indent=2)
    
    print(f"\n✅ Test complete! Results saved to: {results_file}")
    print(f"   WebP animations: {args.output}")
    
    return 0


if __name__ == '__main__':
    exit(main())
