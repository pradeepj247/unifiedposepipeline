#!/usr/bin/env python3
"""
Stage 11: Generate Animated WebP Videos for Top 10 Persons

Creates compact animated WebP files for each of the top 10 persons, showing their
first 50 frames at reduced size (128x192) for fast loading and embedding.

Features:
- Uses crops_enriched.h5 from Stage 6 for correct person-crop association
- Fixed frame sizing (128x192) for consistent playback
- Smart centering and padding of crops
- 10 fps playback (~5 seconds per WebP)
- Animated WebP (modern format, ~60% of GIF size, ~80% of MP4 speed)
- ~50-100 KB per WebP (~0.5-1 MB total for 10 persons)
- Organized output in dedicated 'webp' subfolder

Usage:
    python stage9_generate_person_gifs.py --config configs/pipeline_config.yaml
"""

import argparse
import numpy as np
import yaml
import re
import os
import h5py
import cv2
try:
    from PIL import Image
except ImportError:
    print("Installing Pillow for WebP support...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'Pillow'])
    from PIL import Image
from pathlib import Path
import time


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


def compute_frame_size_for_person_hdf5(person, h5_person_group, target_width=256, target_height=384):
    """
    Use fixed frame size for all persons (optimized for faster GIF generation).
    
    All crops will be resized to fit within target_width x target_height
    while maintaining aspect ratio.
    
    Returns: (width, height) for this person's GIF frames (always fixed size)
    """
    # Return fixed size for all persons - much faster than computing per-person
    return target_width, target_height


def resize_crop_to_frame(crop, frame_width, frame_height, padding_color=(0, 0, 0)):
    """
    Resize a crop to fit within frame_width x frame_height while maintaining aspect ratio.
    Centers the resized crop within the frame with padding.
    
    padding_color: BGR tuple (default black)
    """
    if crop is None:
        return np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    
    crop_height, crop_width = crop.shape[:2]
    
    # Calculate scaling to fit crop within target frame while maintaining aspect ratio
    scale = min(frame_width / crop_width, frame_height / crop_height)
    new_width = int(crop_width * scale)
    new_height = int(crop_height * scale)
    
    # Resize crop (faster than padding)
    resized = cv2.resize(crop, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Create frame with padding color
    frame = np.full((frame_height, frame_width, 3), padding_color, dtype=np.uint8)
    
    # Center the resized crop in the frame
    start_x = (frame_width - new_width) // 2
    start_y = (frame_height - new_height) // 2
    
    # Place resized crop in frame
    frame[start_y:start_y + new_height, start_x:start_x + new_width] = resized
    
    return frame


def create_webp_for_person(person, h5_person_group, webp_dir, frame_width=128, frame_height=192, fps=10, num_frames=50):
    """
    Create an animated WebP for a single person using HDF5 data.
    
    person: dict with 'person_id', 'frame_numbers'
    h5_person_group: HDF5 group for this person (e.g., h5f['person_03'])
    webp_dir: output directory for WebPs
    frame_width: fixed frame width (128)
    frame_height: fixed frame height (192)
    fps: frames per second (10)
    num_frames: number of frames to include (50)
    """
    person_id = person['person_id']
    frames = person['frame_numbers']
    
    # Prepare WebP filename
    webp_filename = webp_dir / f"person_{person_id:02d}.webp"
    
    frames_list = []
    frames_written = 0
    frames_skipped = 0
    
    for frame_idx in frames:
        if frames_written >= num_frames:
            break
        
        frame_idx = int(frame_idx)
        frame_key = f'frame_{frame_idx:06d}'
        
        # Check if frame exists in HDF5
        if frame_key not in h5_person_group:
            frames_skipped += 1
            continue
        
        frame_group = h5_person_group[frame_key]
        
        # Load crop image from HDF5
        try:
            crop_bgr = frame_group['image_bgr'][()]
            
            if crop_bgr is None or crop_bgr.size == 0:
                frames_skipped += 1
                continue
            
            # Convert BGR to RGB for PIL (WebP expects RGB)
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            
            # Resize crop to frame size (maintains aspect ratio)
            resized_frame = resize_crop_to_frame(crop_rgb, frame_width, frame_height)
            
            # Convert to PIL Image
            frames_list.append(Image.fromarray(resized_frame.astype('uint8')))
            frames_written += 1
            
        except Exception as e:
            frames_skipped += 1
            if frames_skipped <= 3:  # Log first few errors only
                print(f"      [DEBUG] Frame {frame_idx} error: {str(e)[:100]}")
            continue
    
    if frames_written == 0:
        return False, f"No frames found for person {person_id}"
    
    # Write animated WebP
    duration = int(1000 / fps)  # Duration per frame in milliseconds
    try:
        frames_list[0].save(
            str(webp_filename),
            format='WEBP',
            save_all=True,
            append_images=frames_list[1:],
            duration=duration,
            loop=0,
            quality=80  # Quality setting for WebP (0-100)
        )
    except Exception as e:
        return False, f"Failed to save WebP for person {person_id}: {str(e)[:100]}"
    
    # Get file size in MB
    file_size_mb = webp_filename.stat().st_size / (1024 * 1024)
    
    return True, f"person_{person_id:02d}.webp ({frames_written} frames, {frame_width}x{frame_height}, {file_size_mb:.2f} MB)"


def create_webp_for_top_persons(canonical_file, crops_enriched_file, output_webp_dir, 
                                 frame_width=128, frame_height=192, fps=10, num_frames=50):
    """Create animated WebP files for top 10 persons using crops_enriched.h5"""
    
    # Load canonical persons
    print(f"📂 Loading canonical persons...")
    data = np.load(canonical_file, allow_pickle=True)
    persons = list(data['persons'])
    persons.sort(key=lambda p: len(p['frame_numbers']), reverse=True)
    
    # Create output directory
    webp_dir = Path(output_webp_dir) / 'webp'
    webp_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {webp_dir}")
    
    # Generate WebP files for top 10 persons using HDF5
    print(f"\n🎬 Generating animated WebP files for top 10 persons...\n")
    
    success_count = 0
    failed_count = 0
    
    try:
        with h5py.File(crops_enriched_file, 'r') as h5f:
            for rank, person in enumerate(persons[:10], 1):
                person_id = person['person_id']
                num_person_frames = len(person['frame_numbers'])
                
                # Check if person exists in HDF5
                person_key = f'person_{person_id:02d}'
                if person_key not in h5f:
                    print(f"  ⚠️  Rank {rank}: P{person_id} - Not found in crops_enriched.h5")
                    failed_count += 1
                    continue
                
                h5_person_group = h5f[person_key]
                
                success, message = create_webp_for_person(
                    person,
                    h5_person_group,
                    webp_dir,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    fps=fps,
                    num_frames=num_frames
                )
                
                if success:
                    print(f"  ✅ Rank {rank}: P{person_id} - {message}")
                    success_count += 1
                else:
                    print(f"  ❌ Rank {rank}: P{person_id} - {message}")
                    failed_count += 1
    
    except FileNotFoundError:
        print(f"❌ crops_enriched.h5 not found: {crops_enriched_file}")
        print(f"   Make sure Stage 6 has run successfully")
        return False
    except Exception as e:
        print(f"❌ Error reading HDF5 file: {str(e)}")
        return False
    
    print(f"\n{'='*70}")
    print(f"📊 WebP Generation Summary:")
    print(f"  ✅ Successful: {success_count}/10")
    print(f"  📁 Output: {webp_dir}")
    print(f"{'='*70}\n")
    
    return success_count > 0


def main():
    parser = argparse.ArgumentParser(
        description='Stage 11: Generate Animated WebP Files for Top 10 Persons'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Get paths from config
    canonical_file = config['stage7']['input']['canonical_persons_file']
    crops_enriched_file = config['stage6']['output']['crops_enriched_file']
    # Use the parent directory of canonical_persons.npz (video-specific outputs folder)
    output_dir = str(Path(canonical_file).parent)
    
    # Get WebP generation settings from config
    webp_config = config.get('stage11', {}).get('video_generation', {})
    frame_width = webp_config.get('frame_width', 128)
    frame_height = webp_config.get('frame_height', 192)
    fps = webp_config.get('fps', 10)
    num_frames = webp_config.get('max_frames', 50)
    
    print(f"\n{'='*70}")
    print(f"🎬 STAGE 11: GENERATE PERSON ANIMATED WEBP FILES")
    print(f"{'='*70}\n")
    
    t_start = time.time()
    
    success = create_webp_for_top_persons(
        canonical_file,
        crops_enriched_file,
        output_dir,
        frame_width=frame_width,
        frame_height=frame_height,
        fps=fps,
        num_frames=num_frames
    )
    
    t_end = time.time()
    
    if success:
        print(f"✅ Stage 11 completed in {t_end - t_start:.2f}s\n")
        return 0
    else:
        print(f"❌ Stage 11 failed\n")
        return 1


if __name__ == '__main__':
    exit(main())
