#!/usr/bin/env python3
"""
Stage 11: Generate Animated WebP Videos for Top 10 Persons (FIXED - Position to Global Index Conversion)

CRITICAL FIX: Canonical persons store (frame_number, position_in_frame) pairs, NOT global detection indices.
This stage converts them to global indices using num_detections_per_frame for correct crop lookup.

This approach:
- Stores position_indices in tracklets from ByteTrack
- Propagates through canonical persons grouping
- Stage 11 converts (frame_num, position) → global_detection_index
- Then uses global index to look up crop

Benefits:
- Clean data flow from ByteTrack through all stages
- Position indices are correct (not affected by frame-level merging)
- Global index conversion is deterministic
- Direct crop cache access

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
from tqdm import tqdm
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


def resize_crop_to_frame(crop, frame_width, frame_height, padding_color=(0, 0, 0)):
    """
    Resize a crop to fit within frame_width x frame_height while maintaining aspect ratio.
    Centers the resized crop within the frame with padding.
    """
    if crop is None:
        return np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    
    crop_height, crop_width = crop.shape[:2]
    
    scale = min(frame_width / crop_width, frame_height / crop_height)
    new_width = int(crop_width * scale)
    new_height = int(crop_height * scale)
    
    resized = cv2.resize(crop, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    frame = np.full((frame_height, frame_width, 3), padding_color, dtype=np.uint8)
    start_x = (frame_width - new_width) // 2
    start_y = (frame_height - new_height) // 2
    frame[start_y:start_y + new_height, start_x:start_x + new_width] = resized
    
    return frame


def create_webp_for_person(person, crops_cache, detection_idx_to_frame_pos, num_detections_per_frame,
                          webp_dir, frame_width=128, frame_height=192, fps=10, num_frames=60):
    """
    Create an animated WebP for a single person.
    
    CRITICAL: Canonical persons store (frame_number, position_in_frame) pairs.
    We must convert them to global detection indices using num_detections_per_frame.
    """
    person_id = person['person_id']
    
    frame_numbers = person.get('frame_numbers', np.array([]))
    position_indices = person.get('detection_indices', np.array([]))  # Positions, not global!
    
    if len(frame_numbers) == 0 or len(position_indices) == 0:
        return False, f"No frame data for person {person_id}"
    
    # Convert (frame_num, position) pairs to global detection indices
    global_detection_indices = []
    for frame_num, pos_in_frame in zip(frame_numbers, position_indices):
        frame_idx = int(frame_num)
        pos = int(pos_in_frame)
        
        # Global index = sum of detections in all previous frames + position in this frame
        global_idx = int(np.sum(num_detections_per_frame[:frame_idx])) + pos
        global_detection_indices.append(global_idx)
    
    # Filter valid indices
    max_global_idx = len(detection_idx_to_frame_pos) - 1
    valid_indices = [idx for idx in global_detection_indices if 0 <= idx <= max_global_idx]
    
    if len(valid_indices) == 0:
        return False, f"No valid detection indices for person {person_id}"
    
    # Apply adaptive offset
    offset = min(int(len(valid_indices) * 0.2), 50)
    start_idx = offset
    end_idx = min(start_idx + num_frames, len(valid_indices))
    
    indices_to_use = valid_indices[start_idx:end_idx]
    
    # Generate WebP
    webp_filename = webp_dir / f"person_{person_id:02d}.webp"
    
    frames_list = []
    frames_written = 0
    frames_skipped = 0
    
    for detection_idx in indices_to_use:
        if detection_idx not in detection_idx_to_frame_pos:
            frames_skipped += 1
            continue
        
        frame_idx, pos_in_frame = detection_idx_to_frame_pos[detection_idx]
        
        # Get crop from cache
        crop = None
        if frame_idx in crops_cache and pos_in_frame in crops_cache[frame_idx]:
            crop = crops_cache[frame_idx][pos_in_frame]
        
        if crop is None or (hasattr(crop, 'size') and crop.size == 0):
            frames_skipped += 1
            continue
        
        try:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized_frame = resize_crop_to_frame(crop_rgb, frame_width, frame_height)
            frames_list.append(Image.fromarray(resized_frame.astype('uint8')))
            frames_written += 1
        except Exception as e:
            frames_skipped += 1
            continue
    
    if frames_written == 0:
        return False, f"No valid frames found for person {person_id}"
    
    # Save WebP
    duration = int(1000 / fps)
    try:
        frames_list[0].save(
            str(webp_filename),
            format='WEBP',
            save_all=True,
            append_images=frames_list[1:],
            duration=duration,
            loop=0,
            quality=80
        )
    except Exception as e:
        return False, f"Failed to save WebP for person {person_id}: {str(e)[:100]}"
    
    file_size_mb = webp_filename.stat().st_size / (1024 * 1024)
    return True, f"person_{person_id:02d}.webp ({frames_written} frames, {frame_width}x{frame_height}, {file_size_mb:.2f} MB)"


def create_webp_for_top_persons(canonical_file, crops_cache_file, detections_file, output_webp_dir,
                                 frame_width=128, frame_height=192, fps=10, num_frames=60):
    """Create animated WebP files for top 10 persons"""
    
    # Load canonical persons
    data = np.load(canonical_file, allow_pickle=True)
    persons = list(data['persons'])
    persons.sort(key=lambda p: len(p['frame_numbers']), reverse=True)
    
    # Check detection_indices
    if 'detection_indices' not in persons[0]:
        print(f"   ⚠️  WARNING: detection_indices not in canonical persons!")
        return False
    
    # Load detections
    det_data = np.load(detections_file, allow_pickle=True)
    
    # Load crops cache
    if not Path(crops_cache_file).exists():
        print(f"❌ Crops cache not found: {crops_cache_file}")
        return False
    
    try:
        with open(crops_cache_file, 'rb') as f:
            crops_cache = pickle.load(f)
    except Exception as e:
        print(f"❌ Error reading crops cache: {str(e)}")
        return False
    
    # Build detection index mapping
    num_detections_per_frame = det_data.get('num_detections_per_frame', np.array([]))
    
    detection_idx_to_frame_pos = {}
    detection_idx = 0
    for frame_idx, num_dets_in_frame in enumerate(num_detections_per_frame):
        for pos_in_frame in range(int(num_dets_in_frame)):
            detection_idx_to_frame_pos[detection_idx] = (frame_idx, pos_in_frame)
            detection_idx += 1
    
    # Create output directory
    webp_dir = Path(output_webp_dir) / 'webp'
    webp_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate WebPs
    print(f"\n🎬 Generating animated WebP files for top 10 persons...\n")
    
    success_count = 0
    
    for rank, person in tqdm(enumerate(persons[:10], 1), total=min(10, len(persons)), desc="Generating WebP files"):
        person_id = person['person_id']
        
        success, message = create_webp_for_person(
            person,
            crops_cache,
            detection_idx_to_frame_pos,
            num_detections_per_frame,
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
    
    print(f"\n{'='*70}")
    print(f"📊 WebP Generation Summary:")
    print(f"  ✅ Successful: {success_count}/10")
    print(f"  📁 Output: {webp_dir}")
    print(f"{'='*70}\n")
    
    return success_count > 0


def main():
    parser = argparse.ArgumentParser(
        description='Stage 11: Generate Animated WebP Files for Top 10 Persons (Position to Global Index Conversion)'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    canonical_file = config['stage5']['output']['canonical_persons_file']
    crops_cache_file = config['stage1']['output']['crops_cache_file']
    detections_file = config['stage1']['output']['detections_file']
    output_dir = str(Path(canonical_file).parent)
    
    webp_config = config.get('stage11', {}).get('video_generation', {})
    frame_width = webp_config.get('frame_width', 128)
    frame_height = webp_config.get('frame_height', 192)
    fps = webp_config.get('fps', 10)
    num_frames = webp_config.get('max_frames', 60)
    
    print(f"\n{'='*70}")
    print(f"🎬 STAGE 11: GENERATE PERSON ANIMATED WEBP FILES (POSITION→GLOBAL CONVERSION)")
    print(f"{'='*70}\n")
    
    t_start = time.time()
    
    success = create_webp_for_top_persons(
        canonical_file,
        crops_cache_file,
        detections_file,
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
