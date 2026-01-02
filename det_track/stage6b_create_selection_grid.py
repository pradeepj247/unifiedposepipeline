#!/usr/bin/env python3
"""
Stage 6b: Create Person Selection Slideshow

Creates a slideshow video (MP4 + GIF) showing top persons with 3 frames each:
- Beginning, Middle, End of their appearance
- 1 FPS for easy review
- Label bar with person info

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
    """Load and filter top persons"""
    data = np.load(canonical_persons_file, allow_pickle=True)
    persons = data['persons']
    
    min_frames = int(min_duration_seconds * video_fps)
    
    # Filter by frame count and sort by duration
    persons_filtered = [p for p in persons if len(p['frame_numbers']) >= min_frames]
    persons_filtered.sort(key=lambda p: len(p['frame_numbers']), reverse=True)
    
    return persons_filtered[:max_persons]


def get_three_frames(person):
    """Get beginning, middle, and end frame indices for a person"""
    frame_numbers = person['frame_numbers']
    n = len(frame_numbers)
    
    # Get indices: beginning (10%), middle (50%), end (90%)
    begin_idx = min(int(n * 0.1), n - 1)
    mid_idx = n // 2
    end_idx = max(int(n * 0.9), 0)
    
    return [
        (int(frame_numbers[begin_idx]), begin_idx),
        (int(frame_numbers[mid_idx]), mid_idx),
        (int(frame_numbers[end_idx]), end_idx)
    ]


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


def create_slideshow_frame(crops, person_info, crop_height=600, fixed_width=400):
    """Create one slideshow frame with 3 crops + label bar"""
    
    # Resize crops to uniform height AND width to ensure consistent dimensions
    resized_crops = []
    for crop in crops:
        # Resize to fixed dimensions
        resized = cv2.resize(crop, (fixed_width, crop_height))
        resized_crops.append(resized)
    
    # Stack horizontally
    grid = np.hstack(resized_crops)
    
    # Create label bar
    label_height = 80
    label_bar = np.ones((label_height, grid.shape[1], 3), dtype=np.uint8) * 255  # White
    
    # Add text to label bar
    person_id = person_info['person_id']
    frame_count = person_info['frame_count']
    duration = person_info['duration_seconds']
    rank = person_info['rank']
    
    label_text = f"Person {person_id} ({frame_count} frames, {duration:.1f}s)"
    progress_text = f"{rank}/{person_info['total']}"
    
    # Main label (centered)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
    text_x = (grid.shape[1] - text_w) // 2
    text_y = label_height // 2 + text_h // 2
    
    cv2.putText(label_bar, label_text, (text_x, text_y),
               font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    # Progress indicator (top right)
    cv2.putText(label_bar, progress_text, (grid.shape[1] - 100, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2, cv2.LINE_AA)
    
    # Add labels "Beginning", "Middle", "End" below each crop
    labels = ["Beginning", "Middle", "End"]
    crop_widths = [c.shape[1] for c in resized_crops]
    x_offset = 0
    
    for label, crop_w in zip(labels, crop_widths):
        text_x = x_offset + crop_w // 2 - 40
        cv2.putText(label_bar, label, (text_x, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1, cv2.LINE_AA)
        x_offset += crop_w
    
    # Stack label bar above grid
    final_frame = np.vstack([label_bar, grid])
    
    return final_frame


def write_video(frames, output_path, fps=1):
    """Write frames to video"""
    if not frames:
        return False
    
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    return True


def write_gif(frames, output_path):
    """Write frames to GIF"""
    try:
        import imageio
        # Convert BGR to RGB
        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        imageio.mimsave(str(output_path), frames_rgb, duration=1.0, loop=0)
        return True
    except ImportError:
        print("   ⚠️  imageio not available, skipping GIF")
        return False


def main():
    parser = argparse.ArgumentParser(description='Stage 6b: Create person selection slideshow')
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
    
    mp4_output = output_dir / 'person_selection_slideshow.mp4'
    gif_output = output_dir / 'person_selection_slideshow.gif'
    info_output = output_dir / 'selection_slideshow_info.json'
    
    # Settings
    min_duration = stage_config.get('filters', {}).get('min_duration_seconds', 5)
    max_persons = stage_config.get('filters', {}).get('max_persons_shown', 10)
    
    print(f"\n{'='*70}")
    print(f"🎬 STAGE 6b: CREATE PERSON SELECTION SLIDESHOW")
    print(f"{'='*70}\n")
    print(f"📂 Video: {video_path.name}")
    print(f"📊 Input: {canonical_persons_file.name}")
    print(f"🎯 Output: {mp4_output.name} + {gif_output.name}")
    
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
    
    # Load top persons
    print(f"\n📊 Loading persons...")
    persons = load_top_persons(canonical_persons_file, min_duration, video_fps, max_persons)
    
    if not persons:
        cap.release()
        print(f"\n❌ No persons meet the criteria (min {min_duration}s)!")
        return
    
    print(f"   ✅ Selected {len(persons)} persons")
    
    # Create slideshow frames
    print(f"\n🎨 Creating slideshow frames (1 FPS)...")
    frame_extraction_time = 0
    frame_creation_time = 0
    slideshow_frames = []
    metadata = []
    
    for rank, person in enumerate(persons, 1):
        person_id = person['person_id']
        frame_count = len(person['frame_numbers'])
        duration = frame_count / video_fps
        
        # Get 3 frame numbers
        three_frames = get_three_frames(person)
        
        # Load and crop (timing extraction)
        extract_start = time.time()
        crops = []
        for frame_num, bbox_idx in three_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                print(f"   ⚠️  Could not read frame {frame_num}")
                continue
            
            bbox = person['bboxes'][bbox_idx]
            crop = extract_crop(frame, bbox, padding_percent=10)
            crops.append(crop)
        frame_extraction_time += time.time() - extract_start
        
        if len(crops) == 3:
            # Create slideshow frame (timing creation)
            create_start = time.time()
            person_info = {
                'person_id': int(person_id),
                'frame_count': frame_count,
                'duration_seconds': duration,
                'rank': rank,
                'total': len(persons)
            }
            
            slideshow_frame = create_slideshow_frame(crops, person_info)
            frame_creation_time += time.time() - create_start
            
            slideshow_frames.append(slideshow_frame)
            
            metadata.append({
                'person_id': int(person_id),
                'rank': rank,
                'frame_count': frame_count,
                'duration_seconds': round(duration, 2),
                'frames_shown': [int(f) for f, _ in three_frames]
            })
            
            extract_ms = (time.time() - extract_start) * 1000
            print(f"   ✅ Person {person_id} ({rank}/{len(persons)}) - {frame_count} frames [{extract_ms:.0f}ms extract]")
    
    cap.release()
    
    if not slideshow_frames:
        print(f"\n❌ No slideshow frames created!")
        return
    
    # Write MP4
    print(f"\n⏱️  Frame extraction: {frame_extraction_time:.2f}s")
    print(f"⏱️  Frame creation: {frame_creation_time:.2f}s")
    
    write_start = time.time()
    print(f"\n💾 Writing MP4 (1 FPS, {len(slideshow_frames)} frames = {len(slideshow_frames)}s)...")
    success_mp4 = write_video(slideshow_frames, mp4_output, fps=1)
    write_mp4_time = time.time() - write_start
    
    if success_mp4:
        mp4_size = mp4_output.stat().st_size / (1024 * 1024)
        print(f"   ✅ Saved: {mp4_output.name} ({mp4_size:.2f} MB) [{write_mp4_time:.2f}s]")
    
    # Write GIF
    gif_start = time.time()
    print(f"\n💾 Writing GIF...")
    success_gif = write_gif(slideshow_frames, gif_output)
    write_gif_time = time.time() - gif_start
    
    if success_gif:
        gif_size = gif_output.stat().st_size / (1024 * 1024)
        print(f"   ✅ Saved: {gif_output.name} ({gif_size:.2f} MB) [{write_gif_time:.2f}s]")
    
    # Save metadata
    with open(info_output, 'w') as f:
        json.dump({
            'persons': metadata,
            'video_info': {
                'fps': 1,
                'frame_count': len(slideshow_frames),
                'duration_seconds': len(slideshow_frames)
            }
        }, f, indent=2)
    
    print(f"   ✅ Saved: {info_output.name}")
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"✅ SLIDESHOW COMPLETE!")
    print(f"{'='*70}\n")
    print(f"⏱️  Total Time: {elapsed:.2f}s")
    print(f"   • Frame extraction: {frame_extraction_time:.2f}s ({frame_extraction_time/elapsed*100:.1f}%)")
    print(f"   • Frame creation: {frame_creation_time:.2f}s ({frame_creation_time/elapsed*100:.1f}%)")
    print(f"   • MP4 write: {write_mp4_time:.2f}s ({write_mp4_time/elapsed*100:.1f}%)")
    print(f"   • GIF write: {write_gif_time:.2f}s ({write_gif_time/elapsed*100:.1f}%)")
    print(f"📦 Outputs:")
    print(f"   • MP4: {mp4_output.name} ({len(slideshow_frames)} seconds @ 1 FPS)")
    print(f"   • GIF: {gif_output.name}")
    print(f"   • Info: {info_output.name}")
    print(f"\n💡 Review the slideshow and select your preferred person!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
