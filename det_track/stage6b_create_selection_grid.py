#!/usr/bin/env python3
"""
Stage 6b: Create Person Selection Grid

Creates a single grid image showing all top persons efficiently:
- Smart frame selection: minimal video seeks by finding frames with multiple persons
- One crop per person at high-confidence frame
- Sorted by duration (longest appearance first)

Usage:
    python stage6b_create_selection_grid.py --config configs/pipeline_config.yaml
"""

import argparse
import numpy as np
import cv2
import json
import yaml
import re
import os
from pathlib import Path
import time

# Suppress OpenCV/FFmpeg warnings
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
cv2.setLogLevel(0)


def resolve_path_variables(config):
    """Recursively resolve ${variable} in config with multi-pass resolution"""
    max_passes = 5
    
    for _ in range(max_passes):
        global_vars = config.get('global', {})
        changed = False
        
        def resolve_string(s):
            nonlocal changed
            if not isinstance(s, str):
                return s
            
            def replace_var(match):
                nonlocal changed
                var_name = match.group(1)
                replacement = str(global_vars.get(var_name, match.group(0)))
                if replacement != match.group(0):
                    changed = True
                return replacement
            
            return re.sub(r'\$\{(\w+)\}', replace_var, s)
        
        def resolve_recursive(obj):
            if isinstance(obj, dict):
                return {k: resolve_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve_recursive(v) for v in obj]
            elif isinstance(obj, str):
                return resolve_string(obj)
            return obj
        
        config = resolve_recursive(config)
        
        if not changed:
            break
    
    return config


def load_config(config_path):
    """Load and resolve YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Auto-extract current_video from video_file
    video_file = config.get('global', {}).get('video_file', '')
    if video_file:
        video_name = os.path.splitext(video_file)[0]
        config['global']['current_video'] = video_name
    
    return resolve_path_variables(config)


def load_top_persons(canonical_persons_file, min_duration_seconds, video_fps, max_persons=10):
    """Load and filter top persons, sorted by duration (descending)"""
    data = np.load(canonical_persons_file, allow_pickle=True)
    persons = data['persons']
    
    min_frames = int(min_duration_seconds * video_fps)
    
    # Filter by frame count and sort by duration (descending)
    persons_filtered = [p for p in persons if len(p['frame_numbers']) >= min_frames]
    persons_filtered.sort(key=lambda p: len(p['frame_numbers']), reverse=True)
    
    return persons_filtered[:max_persons]


def select_frames_smartly(persons, min_confidence=0.6, frame_offset=5):
    """
    Smart frame selection: find minimal frames that cover all persons
    
    Strategy:
    1. Sort persons by start frame (chronological)
    2. For each uncovered person, select frame = start_frame + offset
    3. Check which other uncovered persons appear in that frame
    4. Mark all as covered
    
    Returns:
        frame_plan: List of (frame_num, [person_ids])
    """
    # Create working list sorted by start frame
    persons_by_start = sorted(persons, key=lambda p: p['frame_numbers'][0])
    
    covered_persons = set()
    frame_plan = []
    
    for person in persons_by_start:
        person_id = person['person_id']
        
        # Skip if already covered
        if person_id in covered_persons:
            continue
        
        # Find a good frame for this person
        start_frame = person['frame_numbers'][0]
        candidate_frame = start_frame + frame_offset
        
        # Find index of this frame in person's timeline
        frame_idx = None
        for idx, frame_num in enumerate(person['frame_numbers']):
            if frame_num >= candidate_frame:
                frame_idx = idx
                break
        
        if frame_idx is None:
            frame_idx = 0  # Fallback to first frame
        
        # Check confidence
        if person['confidences'][frame_idx] < min_confidence:
            # Find first frame with good confidence
            for idx, conf in enumerate(person['confidences']):
                if conf >= min_confidence:
                    frame_idx = idx
                    break
        
        selected_frame = person['frame_numbers'][frame_idx]
        
        # Find which other uncovered persons appear in this frame
        persons_in_frame = [person_id]
        
        for other_person in persons:
            other_id = other_person['person_id']
            if other_id in covered_persons or other_id == person_id:
                continue
            
            # Check if other person exists in this frame
            if selected_frame in other_person['frame_numbers']:
                other_idx = np.where(other_person['frame_numbers'] == selected_frame)[0][0]
                if other_person['confidences'][other_idx] >= min_confidence:
                    persons_in_frame.append(other_id)
        
        # Mark all as covered
        covered_persons.update(persons_in_frame)
        frame_plan.append((int(selected_frame), persons_in_frame))
        
        # Stop if all covered
        if len(covered_persons) >= len(persons):
            break
    
    return frame_plan


def extract_crop(frame, bbox, padding_percent=10):
    """Extract crop with padding"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    
    pad_x = bbox_w * (padding_percent / 100)
    pad_y = bbox_h * (padding_percent / 100)
    
    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(w, int(x2 + pad_x))
    y2 = min(h, int(y2 + pad_y))
    
    return frame[y1:y2, x1:x2].copy()


def extract_crops_from_frames(video_path, frame_plan, persons_dict):
    """
    Extract crops from selected frames
    
    Args:
        video_path: Path to video
        frame_plan: List of (frame_num, [person_ids])
        persons_dict: Dict of {person_id: person_data}
    
    Returns:
        crops: Dict of {person_id: crop_image}
    """
    crops = {}
    cap = cv2.VideoCapture(str(video_path))
    
    for frame_num, person_ids in frame_plan:
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            print(f"   ⚠️  Could not read frame {frame_num}")
            continue
        
        # Extract crops for all persons in this frame
        for person_id in person_ids:
            person = persons_dict[person_id]
            
            # Find bbox at this frame
            frame_idx = np.where(person['frame_numbers'] == frame_num)[0][0]
            bbox = person['bboxes'][frame_idx]
            
            crop = extract_crop(frame, bbox, padding_percent=10)
            crops[person_id] = crop
    
    cap.release()
    return crops


def create_grid_image(crops, persons, fixed_height=600):
    """
    Create single grid image with all person crops
    
    Args:
        crops: Dict of {person_id: crop_image}
        persons: List of person dicts (sorted by duration)
        fixed_height: Target height for all crops
    
    Returns:
        grid_image: Single image with all crops side-by-side
    """
    # Resize crops to fixed height, calculate widths
    resized_crops = []
    person_ids = []
    
    for person in persons:
        person_id = person['person_id']
        if person_id not in crops:
            continue
        
        crop = crops[person_id]
        h, w = crop.shape[:2]
        
        # Maintain aspect ratio
        new_h = fixed_height
        new_w = int(w * (fixed_height / h))
        
        resized = cv2.resize(crop, (new_w, new_h))
        resized_crops.append(resized)
        person_ids.append(person_id)
    
    if not resized_crops:
        return None
    
    # Find max width for padding
    max_width = max(c.shape[1] for c in resized_crops)
    
    # Pad all crops to same width
    padded_crops = []
    for crop in resized_crops:
        h, w = crop.shape[:2]
        if w < max_width:
            # Pad on right
            pad_width = max_width - w
            padded = cv2.copyMakeBorder(crop, 0, 0, 0, pad_width, 
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            padded = crop
        padded_crops.append(padded)
    
    # Stack horizontally
    grid = np.hstack(padded_crops)
    
    # Add label bar below
    label_height = 60
    label_bar = np.ones((label_height, grid.shape[1], 3), dtype=np.uint8) * 255
    
    # Add person IDs
    x_offset = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    for person_id in person_ids:
        text = f"Person {person_id}"
        text_x = x_offset + max_width // 2 - 50
        cv2.putText(label_bar, text, (text_x, 40),
                   font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        x_offset += max_width
    
    # Stack label below grid
    final_image = np.vstack([grid, label_bar])
    
    return final_image


def main():
    parser = argparse.ArgumentParser(description='Stage 6b: Create person selection grid')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Load config
    config = load_config(args.config)
    stage_config = config.get('stage6b_create_selection_grid', {})
    
    # Get paths (use absolute paths to avoid creating nested directories)
    video_path = Path(config['global']['video_dir'] + config['global']['video_file']).resolve().absolute()
    canonical_persons_file = Path(config['stage4b_group_canonical']['output']['canonical_persons_file']).resolve().absolute()
    
    # Output paths
    output_dir = Path(config['global']['outputs_dir']).resolve().absolute() / config['global']['current_video']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    grid_output = output_dir / 'person_selection_grid.png'
    
    # Settings
    min_duration = stage_config.get('filters', {}).get('min_duration_seconds', 5)
    max_persons = stage_config.get('filters', {}).get('max_persons_shown', 10)
    
    print(f"\n{'='*70}")
    print(f"🎬 STAGE 6b: CREATE PERSON SELECTION GRID")
    print(f"{'='*70}\n")
    print(f"📂 Video: {video_path.name}")
    print(f"📊 Input: {canonical_persons_file.name}")
    print(f"🎯 Output: {grid_output.name}")
    
    # Check files
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    if not canonical_persons_file.exists():
        print(f"❌ Canonical persons not found: {canonical_persons_file}")
        return
    
    # Get video properties
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Load top persons
    print(f"\n📊 Loading persons...")
    persons = load_top_persons(canonical_persons_file, min_duration, video_fps, max_persons)
    
    if not persons:
        print(f"\n❌ No persons meet the criteria (min {min_duration}s)!")
        return
    
    print(f"   ✅ Selected {len(persons)} persons (sorted by duration)")
    for i, p in enumerate(persons[:5], 1):
        duration = len(p['frame_numbers']) / video_fps
        print(f"      {i}. Person {p['person_id']}: {len(p['frame_numbers'])} frames ({duration:.1f}s)")
    if len(persons) > 5:
        print(f"      ... and {len(persons)-5} more")
    
    # Smart frame selection (NPZ-only phase)
    print(f"\n🧠 Smart frame selection...")
    planning_start = time.time()
    frame_plan = select_frames_smartly(persons, min_confidence=0.6, frame_offset=5)
    planning_time = time.time() - planning_start
    
    print(f"   ✅ Optimized to {len(frame_plan)} frames (from potential {len(persons)*3} seeks)")
    for frame_num, person_ids in frame_plan:
        print(f"      Frame {frame_num}: {len(person_ids)} persons - {person_ids}")
    
    # Extract crops (minimal video access)
    print(f"\n🎨 Extracting crops from video...")
    extraction_start = time.time()
    persons_dict = {p['person_id']: p for p in persons}
    crops = extract_crops_from_frames(video_path, frame_plan, persons_dict)
    extraction_time = time.time() - extraction_start
    
    print(f"   ✅ Extracted {len(crops)} crops in {extraction_time:.2f}s")
    
    # Create grid image
    print(f"\n🖼️  Creating grid image...")
    grid_start = time.time()
    grid_image = create_grid_image(crops, persons, fixed_height=600)
    grid_time = time.time() - grid_start
    
    if grid_image is None:
        print(f"❌ Failed to create grid image!")
        return
    
    # Save image
    cv2.imwrite(str(grid_output), grid_image)
    file_size = grid_output.stat().st_size / (1024 * 1024)
    
    print(f"   ✅ Grid created: {grid_image.shape[1]}×{grid_image.shape[0]} pixels")
    print(f"   ✅ Saved: {grid_output.name} ({file_size:.2f} MB)")
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"✅ GRID COMPLETE!")
    print(f"{'='*70}\n")
    print(f"⏱️  Total Time: {elapsed:.2f}s")
    print(f"   • Planning: {planning_time:.2f}s ({planning_time/elapsed*100:.1f}%)")
    print(f"   • Extraction: {extraction_time:.2f}s ({extraction_time/elapsed*100:.1f}%)")
    print(f"   • Grid creation: {grid_time:.2f}s ({grid_time/elapsed*100:.1f}%)")
    print(f"\n📦 Output:")
    print(f"   • {grid_output.name} ({len(persons)} persons in {len(frame_plan)} frames)")
    print(f"\n💡 Review the grid and select your preferred person!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
