#!/usr/bin/env python3
"""
On-Demand Crop Extraction - Linear Pass Algorithm

Extracts person crops directly from video using a single sequential pass.
No intermediate crop storage needed - saves 811 MB and maintains speed.

Algorithm:
1. Open video once
2. Read frames sequentially (decoder-friendly!)
3. Extract crops for multiple persons per frame
4. Early termination when all target quotas filled

Performance: ~225 FPS, 9s for 1700 frames, 811 MB storage savings

Usage:
    from ondemand_crop_extraction import extract_crops_from_video
    
    person_buckets = extract_crops_from_video(
        video_path='video.mp4',
        persons=canonical_persons,  # List of dicts with 'person_id', 'frame_numbers', 'bboxes'
        target_crops_per_person=50
    )
"""

import numpy as np
import cv2
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any


def extract_crops_from_video(
    video_path: str,
    persons: List[Dict[str, Any]],
    target_crops_per_person: int = 50,
    top_n: int = 10,
    max_first_appearance_ratio: float = 0.5
) -> Dict[int, List[np.ndarray]]:
    """
    Extract person crops via single linear pass through video.
    
    Args:
        video_path: Path to video file
        persons: List of person dicts from canonical_persons.npz
                 Each dict has: 'person_id', 'frame_numbers', 'bboxes'
        target_crops_per_person: How many crops to extract per person
        top_n: How many persons to extract (default: 10)
        max_first_appearance_ratio: Maximum ratio (0-1) of video length for first appearance
                                     Persons appearing after this are excluded (default: 0.5)
    
    Returns:
        Dict mapping person_id -> list of crop images (np.ndarray)
        
    Performance:
        - ~225 FPS processing speed
        - Early termination when all quotas filled
        - Multi-person batching (extract 3-4 persons per frame)
    """
    
    print(f"\n[On-Demand Extraction] Starting linear pass...")
    start_time = time.time()
    
    # Get video info to determine appearance threshold
    cap_temp = cv2.VideoCapture(str(video_path))
    total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_temp.release()
    
    max_first_frame = int(total_frames * max_first_appearance_ratio)
    
    # Sort persons by frame count and take top N candidates
    persons_sorted = sorted(persons, key=lambda p: len(p['frame_numbers']), reverse=True)
    top_n_candidates = persons_sorted[:top_n]
    
    # Filter: only keep those who appear early enough (DON'T backfill)
    top_persons = [
        p for p in top_n_candidates
        if p['frame_numbers'][0] <= max_first_frame
    ]
    
    # Report filtering
    excluded = [p for p in top_n_candidates if p['frame_numbers'][0] > max_first_frame]
    print(f"   Target: top {top_n} persons by frame count, {target_crops_per_person} crops each")
    print(f"   Early appearance filter: first frame ≤ {max_first_frame} ({max_first_appearance_ratio*100:.0f}% of video)")
    if excluded:
        print(f"   Excluded {len(excluded)} late-appearing person(s) from top {top_n}:")
        for p in excluded:
            pid = int(p['person_id'])
            first_frame = p['frame_numbers'][0]
            frame_count = len(p['frame_numbers'])
            print(f"     Person {pid}: starts at frame {first_frame} ({first_frame/total_frames*100:.1f}%), {frame_count} frames")
    print(f"   Selected {len(top_persons)} persons for extraction (no backfill)")
    
    # Phase 1: Build extraction plan
    # Maps frame_number -> [(person_id, bbox), ...]
    frame_to_persons = {}
    person_buckets = {}
    person_targets = {}
    person_frames_used = {}  # Track which frames were actually used
    person_total_frames = {}  # Track total available frames per person
    
    for person in top_persons:
        person_id = int(person['person_id'])
        available_frames = person['frame_numbers']
        bboxes = person['bboxes']
        
        # Determine target count
        target = min(len(available_frames), target_crops_per_person)
        person_targets[person_id] = target
        person_buckets[person_id] = []
        person_frames_used[person_id] = []  # Track actual frames
        person_total_frames[person_id] = len(available_frames)  # Total frames available
        
        # Select which frames to use (take first N chronologically)
        selected_indices = list(range(min(target, len(available_frames))))
        
        # Map each frame to this person
        for idx in selected_indices:
            frame_num = int(available_frames[idx])
            bbox = bboxes[idx]
            
            if frame_num not in frame_to_persons:
                frame_to_persons[frame_num] = []
            
            frame_to_persons[frame_num].append((person_id, bbox))
        
        print(f"     Person {person_id}: target={target}, first_frame={available_frames[0]}, last_frame={available_frames[-1]}")
    
    # Calculate maximum frame needed (we can stop here!)
    max_frame_needed = max(frame_to_persons.keys())
    print(f"\n   Maximum frame needed: {max_frame_needed} (no need to read beyond this)")
    
    # Phase 2: Linear pass through video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"   Video: {total_frames} frames @ {fps:.2f} FPS")
    print(f"   Will read: 0-{max_frame_needed} ({max_frame_needed+1} frames, {(max_frame_needed+1)/total_frames*100:.1f}% of video)")
    
    frames_with_targets = sorted(frame_to_persons.keys())
    next_target_idx = 0
    frames_processed = 0
    crops_extracted = 0
    
    # Sequential decode
    frame_idx = 0
    while frame_idx <= max_frame_needed:  # Hard stop at pre-calculated max
        ret, frame = cap.read()
        if not ret:
            break
        
        frames_processed += 1
        
        # Check if this frame has targets
        if next_target_idx < len(frames_with_targets) and frame_idx == frames_with_targets[next_target_idx]:
            persons_at_frame = frame_to_persons[frame_idx]
            
            # Extract crops for all persons at this frame
            for person_id, bbox in persons_at_frame:
                # Check if bucket already full
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
                    person_frames_used[person_id].append(frame_idx)  # Track frame
                    crops_extracted += 1
            
            next_target_idx += 1
        
        # Early termination: all buckets full (may happen before max_frame_needed)
        if all(len(person_buckets[pid]) >= person_targets[pid] for pid in person_targets):
            print(f"   Early termination: All buckets full at frame {frame_idx}/{max_frame_needed} ({frame_idx/max_frame_needed*100:.1f}% of needed range)")
            break
        
        frame_idx += 1
    
    cap.release()
    
    elapsed = time.time() - start_time
    processing_fps = frames_processed / elapsed if elapsed > 0 else 0
    
    # Report results
    print(f"\n   Extraction complete:")
    print(f"     Frames processed: {frames_processed}/{total_frames}")
    print(f"     Crops extracted: {crops_extracted}")
    print(f"     Time: {elapsed:.2f}s @ {processing_fps:.1f} FPS")
    
    buckets_filled = sum(1 for pid in person_targets if len(person_buckets[pid]) >= person_targets[pid])
    print(f"     Buckets filled: {buckets_filled}/{len(person_targets)}")
    
    # Print debug table: frame ranges used per person
    print(f"\n   Frame ranges used per bucket:")
    print(f"   {'Person ID':<12} {'Extracted':<10} {'Available':<10} {'Start':<8} {'End':<8} {'Span':<8}")
    print(f"   {'-'*68}")
    for person_id in sorted(person_buckets.keys()):
        count = len(person_buckets[person_id])
        total_available = person_total_frames[person_id]
        frames_used = person_frames_used[person_id]
        if frames_used:
            start_frame = frames_used[0]
            end_frame = frames_used[-1]
            span = end_frame - start_frame
            print(f"   {person_id:<12} {count:<10} {total_available:<10} {start_frame:<8} {end_frame:<8} {span:<8}")
        else:
            print(f"   {person_id:<12} {count:<10} {total_available:<10} {'N/A':<8} {'N/A':<8} {'N/A':<8}")
    
    # Return buckets + metadata for HTML generation
    metadata = {
        'total_frames': total_frames,
        'person_info': {}
    }
    for person_id in person_buckets.keys():
        frames_used = person_frames_used[person_id]
        metadata['person_info'][person_id] = {
            'num_frames': person_total_frames[person_id],
            'start_frame': frames_used[0] if frames_used else 0,
            'end_frame': frames_used[-1] if frames_used else 0
        }
    
    return person_buckets, metadata


def generate_webp_animations(
    person_buckets: Dict[int, List[np.ndarray]],
    output_dir: Path,
    metadata: Dict[str, Any] = None,
    resize_to: Tuple[int, int] = (256, 256),
    duration_ms: int = 100
) -> None:
    """
    Generate WebP animations from extracted crops.
    
    Args:
        person_buckets: Dict mapping person_id -> list of crops
        output_dir: Where to save WebP files
        metadata: Optional dict with 'total_frames' and 'person_info' for HTML generation
        resize_to: Target size for crops (width, height)
        duration_ms: Frame duration in milliseconds
    """
    import imageio
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[WebP Generation] Creating animations...")
    start_time = time.time()
    
    for person_id, crops in sorted(person_buckets.items()):
        if not crops:
            continue
        
        # Resize all crops to same size
        resized = []
        for crop in crops:
            resized_crop = cv2.resize(crop, resize_to)
            resized_crop = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)
            resized.append(resized_crop)
        
        # Save as WebP
        output_file = output_dir / f"person_{person_id:03d}.webp"
        imageio.mimsave(
            output_file,
            resized,
            format='WEBP',
            duration=duration_ms,
            loop=0
        )
        
        file_size_kb = output_file.stat().st_size / 1024
        print(f"   Person {person_id}: {len(crops)} frames → {output_file.name} ({file_size_kb:.0f} KB)")
    
    elapsed = time.time() - start_time
    print(f"   Generated {len(person_buckets)} WebP animations in {elapsed:.2f}s")
    
    # Generate HTML viewer
    html_file = output_dir / "viewer.html"
    _generate_html_viewer(person_buckets, output_dir, html_file, metadata)
    print(f"   HTML viewer: {html_file}")


def _generate_html_viewer(person_buckets: Dict[int, List[np.ndarray]], output_dir: Path, html_file: Path, metadata: Dict[str, Any] = None) -> None:
    """Generate HTML file to view all WebP animations with embedded base64 images (horizontal tape layout)"""
    import base64
    
    # Get video metadata
    total_frames = metadata.get('total_frames', 0) if metadata else 0
    person_info = metadata.get('person_info', {}) if metadata else {}
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Person Crops Viewer</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            margin: 0;
            padding: 20px;
        }
        h1 {
            text-align: center;
            color: #4CAF50;
            margin-bottom: 30px;
        }
        .grid {
            display: flex;
            gap: 15px;
            overflow-x: auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .person-card {
            flex: 0 0 auto;
            width: 180px;
            background: white;
            border: 2px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            cursor: pointer;
        }
        .person-card:hover {
            border-color: #4CAF50;
            box-shadow: 0 6px 16px rgba(76, 175, 80, 0.3);
            transform: scale(1.05);
        }
        .person-title {
            font-size: 18px;
            font-weight: bold;
            color: #4CAF50;
            margin-bottom: 10px;
            text-align: center;
        }
        .person-info {
            font-size: 12px;
            color: #aaa;
            margin-bottom: 15px;
            text-align: center;
        }
        .webp-container {
            display: flex;
            justify-content: center;
            background: #1a1a1a;
            border-radius: 4px;
            padding: 10px;
        }
        img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }
        .stats {
            background: #333;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 30px;
            max-width: 1400px;
            margin: 0 auto 30px auto;
        }
        .stats h2 {
            margin-top: 0;
            color: #4CAF50;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .stat-item {
            background: #2a2a2a;
            padding: 10px;
            border-radius: 4px;
        }
        .stat-label {
            font-size: 12px;
            color: #aaa;
            margin-bottom: 5px;
        }
        .stat-value {
            font-size: 20px;
            font-weight: bold;
            color: #4CAF50;
        }
    </style>
</head>
<body>
    <h1>Person Crops Viewer</h1>
    
    <div class="stats">
        <h2>Extraction Summary</h2>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-label">Total Persons</div>
                <div class="stat-value">__TOTAL_PERSONS__</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Total Crops</div>
                <div class="stat-value">__TOTAL_CROPS__</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Crops per Person</div>
                <div class="stat-value">__CROPS_PER_PERSON__</div>
            </div>
        </div>
    </div>
    
    <div class="grid">
__PERSON_CARDS__
    </div>
</body>
</html>"""
    
    # Generate person cards with base64-embedded WebP images
    cards = []
    total_crops = 0
    for rank, person_id in enumerate(sorted(person_buckets.keys()), 1):
        crops = person_buckets[person_id]
        webp_file = output_dir / f"person_{person_id:03d}.webp"
        num_crops = len(crops)
        total_crops += num_crops
        
        # Get metadata for this person
        info = person_info.get(person_id, {})
        num_frames = info.get('num_frames', 0)
        start_frame = info.get('start_frame', 0)
        end_frame = info.get('end_frame', 0)
        coverage_pct = (num_frames / total_frames * 100) if total_frames > 0 else 0
        
        # Read WebP file and encode as base64
        with open(webp_file, 'rb') as f:
            webp_data = f.read()
        base64_webp = base64.b64encode(webp_data).decode('utf-8')
        data_uri = f"data:image/webp;base64,{base64_webp}"
        
        card = f"""        <div class="person-card">
            <div class="person-title">#{rank} Person {person_id}</div>
            <div class="person-info">{num_frames} frames | {start_frame}-{end_frame} | {coverage_pct:.0f}% coverage</div>
            <div class="webp-container">
                <img src="{data_uri}" alt="Person {person_id}">
            </div>
        </div>"""
        cards.append(card)
    
    # Replace placeholders
    html_content = html_content.replace('__TOTAL_PERSONS__', str(len(person_buckets)))
    html_content = html_content.replace('__TOTAL_CROPS__', str(total_crops))
    html_content = html_content.replace('__CROPS_PER_PERSON__', str(total_crops // len(person_buckets) if person_buckets else 0))
    html_content = html_content.replace('__PERSON_CARDS__', '\n'.join(cards))
    
    # Write HTML file
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)


# ===== Example Usage =====
if __name__ == '__main__':
    """
    Example: Extract crops from canonical_persons.npz and generate WebPs
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract person crops on-demand from video')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--data', required=True, help='Path to canonical_persons.npz')
    parser.add_argument('--output', default='output/', help='Output directory')
    parser.add_argument('--crops-per-person', type=int, default=50, help='Crops per person')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top persons')
    
    args = parser.parse_args()
    
    # Load canonical persons
    print(f"\n[Loading Data] {args.data}")
    data = np.load(args.data, allow_pickle=True)
    persons = data['persons']
    print(f"   Loaded {len(persons)} persons")
    
    # Extract crops
    person_buckets = extract_crops_from_video(
        video_path=args.video,
        persons=persons,
        target_crops_per_person=args.crops_per_person,
        top_n=args.top_n
    )
    
    # Generate WebPs
    generate_webp_animations(person_buckets, Path(args.output))
    
    print(f"\n✓ Complete! WebPs saved to: {args.output}")
