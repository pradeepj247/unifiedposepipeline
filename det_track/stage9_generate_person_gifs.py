#!/usr/bin/env python3
"""
Stage 9: Generate Animated GIFs for Top 10 Persons (Experimental)

Creates compact animated GIFs for each of the top 10 persons, showing their
first 75 frames with smart padding to center smaller crops within a
consistent frame size per person.

Features:
- Dynamic frame sizing per person (based on largest bbox in first 75 frames)
- Smart centering and padding of crops
- 20 fps playback (3.75 seconds per GIF)
- Organized output in dedicated 'gifs' subfolder

Usage:
    python stage9_generate_person_gifs.py --config configs/pipeline_config.yaml
"""

import argparse
import numpy as np
import pickle
import yaml
import re
import os
import cv2
from pathlib import Path
import time
import imageio
from tqdm import tqdm


def resolve_path_variables(config):
    """Recursively resolve ${variable} in config"""
    global_vars = config.get('global', {})
    
    def resolve_string_once(s, vars_dict):
        if not isinstance(s, str):
            return s
        return re.sub(
            r'\$\{(\w+)\}',
            lambda m: str(vars_dict.get(m.group(1), m.group(0))),
            s
        )
    
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


def compute_frame_size_for_person(person, crops_cache, padding=15):
    """
    Compute the frame size needed for this person.
    
    Scans first 75 frames and finds the largest bbox.
    Adds padding around it to create the final frame size.
    
    Returns: (width, height) for this person's GIF frames
    """
    frames = person['frame_numbers']
    bboxes = person['bboxes']
    
    # Look at first 75 frames (or all if less than 75)
    num_frames_to_check = min(75, len(frames))
    
    max_width = 0
    max_height = 0
    
    for i in range(num_frames_to_check):
        frame_idx = int(frames[i])
        bbox = bboxes[i]
        
        # Bbox format: [x1, y1, x2, y2]
        width = int(bbox[2] - bbox[0])
        height = int(bbox[3] - bbox[1])
        
        max_width = max(max_width, width)
        max_height = max(max_height, height)
    
    # Add padding on all sides
    final_width = max_width + (2 * padding)
    final_height = max_height + (2 * padding)
    
    return int(final_width), int(final_height)


def pad_crop_to_frame(crop, frame_width, frame_height, padding_color=(0, 0, 0)):
    """
    Pad a crop to fit within frame_width x frame_height.
    Center the crop within the frame.
    
    padding_color: RGB or BGR tuple (default black)
    """
    if crop is None:
        return np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    
    crop_height, crop_width = crop.shape[:2]
    
    # If crop is already the right size, return it
    if crop_width == frame_width and crop_height == frame_height:
        return crop
    
    # Create frame with padding color
    frame = np.full((frame_height, frame_width, 3), padding_color, dtype=np.uint8)
    
    # Calculate position to center crop
    start_x = max(0, (frame_width - crop_width) // 2)
    start_y = max(0, (frame_height - crop_height) // 2)
    
    # Handle case where crop is larger than frame (shouldn't happen, but just in case)
    crop_start_x = max(0, (crop_width - frame_width) // 2)
    crop_start_y = max(0, (crop_height - frame_height) // 2)
    crop_end_x = crop_start_x + min(crop_width, frame_width)
    crop_end_y = crop_start_y + min(crop_height, frame_height)
    
    frame_end_x = start_x + (crop_end_x - crop_start_x)
    frame_end_y = start_y + (crop_end_y - crop_start_y)
    
    # Place crop in frame
    frame[start_y:frame_end_y, start_x:frame_end_x] = crop[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
    
    return frame


def create_gif_for_person(person, crops_cache, gifs_dir, fps=20, num_frames=75):
    """
    Create an animated GIF for a single person.
    
    person: dict with 'person_id', 'frame_numbers', 'bboxes'
    crops_cache: {frame_idx: {local_idx: crop_image, ...}}
    gifs_dir: output directory for GIFs
    fps: frames per second (20 for 3.75 second GIFs)
    num_frames: number of frames to include (75)
    """
    person_id = person['person_id']
    frames = person['frame_numbers']
    
    # Determine frame size for this person
    frame_width, frame_height = compute_frame_size_for_person(person, crops_cache)
    
    # Collect frames
    gif_frames = []
    frames_collected = 0
    
    for frame_idx in frames:
        if frames_collected >= num_frames:
            break
        
        frame_idx = int(frame_idx)
        
        # Try to get crop from cache
        if frame_idx not in crops_cache:
            continue
        
        crops_in_frame = crops_cache[frame_idx]
        
        # Get first available crop (we don't have person-to-crop mapping in this stage)
        crop = None
        for crop_img in crops_in_frame.values():
            if crop_img is not None and isinstance(crop_img, np.ndarray):
                crop = crop_img
                break
        
        if crop is None:
            continue
        
        # Pad crop to frame size
        padded_frame = pad_crop_to_frame(crop, frame_width, frame_height)
        gif_frames.append(padded_frame)
        frames_collected += 1
    
    if not gif_frames:
        return False, f"No frames found for person {person_id}"
    
    # Save GIF
    gif_filename = gifs_dir / f"person_{person_id:02d}.gif"
    
    try:
        # imageio.mimsave expects frames in sequence
        # duration is in seconds per frame (1/fps)
        imageio.mimsave(
            str(gif_filename),
            gif_frames,
            fps=fps,
            loop=0  # Loop indefinitely
        )
        return True, f"person_{person_id:02d}.gif ({len(gif_frames)} frames, {frame_width}x{frame_height})"
    except Exception as e:
        return False, f"Failed to save GIF for person {person_id}: {str(e)}"


def create_gifs_for_top_persons(canonical_file, crops_cache_file, output_gifs_dir, fps=20, num_frames=75):
    """Create GIFs for top 10 persons"""
    
    # Load canonical persons
    print(f"📂 Loading canonical persons...")
    data = np.load(canonical_file, allow_pickle=True)
    persons = list(data['persons'])
    persons.sort(key=lambda p: len(p['frame_numbers']), reverse=True)
    
    # Load crops cache
    print(f"📂 Loading crops cache...")
    with open(crops_cache_file, 'rb') as f:
        crops_cache = pickle.load(f)
    
    # Create output directory
    gifs_dir = Path(output_gifs_dir) / 'gifs'
    gifs_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {gifs_dir}")
    
    # Generate GIFs for top 10 persons
    print(f"\n🎬 Generating GIFs for top 10 persons...\n")
    
    success_count = 0
    failed_count = 0
    
    for rank, person in enumerate(persons[:10], 1):
        person_id = person['person_id']
        num_person_frames = len(person['frame_numbers'])
        
        success, message = create_gif_for_person(
            person,
            crops_cache,
            gifs_dir,
            fps=fps,
            num_frames=num_frames
        )
        
        if success:
            print(f"  ✅ Rank {rank}: P{person_id} - {message}")
            success_count += 1
        else:
            print(f"  ❌ Rank {rank}: P{person_id} - {message}")
            failed_count += 1
    
    print(f"\n{'='*70}")
    print(f"📊 GIF Generation Summary:")
    print(f"  ✅ Successful: {success_count}/10")
    print(f"  ❌ Failed: {failed_count}/10")
    print(f"  📁 Output: {gifs_dir}")
    print(f"{'='*70}\n")
    
    return success_count > 0


def main():
    parser = argparse.ArgumentParser(
        description='Stage 9: Generate Animated GIFs for Top 10 Persons (Experimental)'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    canonical_file = config['stage4b_group_canonical']['output']['canonical_persons_file']
    crops_cache_file = config['stage4a_reid_recovery']['input']['crops_cache_file']
    
    output_dir = Path(canonical_file).parent
    
    print(f"\n{'='*70}")
    print(f"🎬 STAGE 9: GENERATE PERSON GIFS (EXPERIMENTAL)")
    print(f"{'='*70}\n")
    
    t_start = time.time()
    
    success = create_gifs_for_top_persons(
        canonical_file,
        crops_cache_file,
        output_dir,
        fps=20,
        num_frames=75
    )
    
    t_end = time.time()
    
    if success:
        print(f"⏱️  Time: {t_end - t_start:.2f}s")
        print(f"{'='*70}\n")
        return True
    else:
        print(f"❌ GIF generation encountered issues")
        print(f"{'='*70}\n")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
