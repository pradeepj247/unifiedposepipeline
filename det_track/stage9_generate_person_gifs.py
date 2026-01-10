#!/usr/bin/env python3
"""
Stage 11: Generate Animated WebP Videos for Top 10 Persons (OPTIMIZED - Direct Detection Indices)

OPTIMIZATION: Uses detection_indices from canonical persons to directly access crops.
This approach:
- Stores detection_indices in tracklets (from ByteTrack output)
- Propagates detection_indices through canonical persons grouping
- Stage 11 uses indices directly - no complex frame/position lookups needed

Benefits:
- Eliminates complex frame-number-based detection matching
- Detection selection is provably correct (from tracking stage)
- Much simpler code, no IOU matching or spatial heuristics
- Direct index-to-crop mapping for clean data flow

Features:
- Direct detection index lookup via crops_cache
- Adaptive frame offset: Skips first 20% of appearance to avoid intro flicker
- Fixed frame sizing (128x192) for consistent playback
- Smart centering and padding of crops
- 10 fps playback (6 seconds per WebP)
- Animated WebP (modern format, ~50-100 KB per person)
- Organized output in dedicated 'webp' subfolder

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


def create_webp_for_person(person, crops_cache, detection_idx_to_frame_pos, webp_dir, 
                          frame_width=128, frame_height=192, fps=10, num_frames=60):
    """
    Create an animated WebP for a single person using direct detection indices.
    
    Args:
        person: dict with 'person_id', 'detection_indices' (from canonical persons)
        crops_cache: dict from Stage 1 with structure {frame_idx: {position_in_frame: crop_image}}
        detection_idx_to_frame_pos: mapping from global detection index to (frame_idx, position_in_frame)
        webp_dir: output directory for WebPs
        frame_width: fixed frame width (128)
        frame_height: fixed frame height (192)
        fps: frames per second (10)
        num_frames: number of frames to include (60)
    """
    person_id = person['person_id']
    
    # Get detection indices for this person (from canonical grouping)
    # These are the DIRECT indices of detections that belong to this person
    detection_indices = person.get('detection_indices', np.array([]))
    if not hasattr(detection_indices, '__len__'):
        return False, f"No detection indices found for person {person_id}"
    
    if len(detection_indices) == 0:
        return False, f"No detection indices found for person {person_id}"
    
    # Remove invalid detection indices (e.g., -1 from merging)
    valid_indices = [int(idx) for idx in detection_indices if int(idx) >= 0]
    if len(valid_indices) == 0:
        return False, f"All detection indices invalid for person {person_id}"
    
    # DEBUG
    if person_id in [3, 65]:  # Only for top 2 persons
        print(f"\n      [DEBUG-P{person_id}] Total detection_indices: {len(detection_indices)}")
        print(f"      [DEBUG-P{person_id}] Valid indices: {len(valid_indices)}")
        print(f"      [DEBUG-P{person_id}] First 20 indices: {valid_indices[:20]}")
        print(f"      [DEBUG-P{person_id}] Last 20 indices: {valid_indices[-20:]}")
    
    # Apply adaptive offset to skip intro flicker
    # Skip first 20% of appearance, but at most 50 frames
    offset = min(int(len(valid_indices) * 0.2), 50)
    start_idx = offset
    end_idx = min(start_idx + num_frames, len(valid_indices))
    
    indices_to_use = valid_indices[start_idx:end_idx]
    
    # Prepare WebP filename
    webp_filename = webp_dir / f"person_{person_id:02d}.webp"
    
    frames_list = []
    frames_written = 0
    frames_skipped = 0
    
    for idx_in_sequence, detection_idx in enumerate(indices_to_use):
        # Convert global detection index to (frame_idx, position_in_frame)
        if detection_idx not in detection_idx_to_frame_pos:
            frames_skipped += 1
            # Debug first few misses
            if frames_skipped <= 3 and person_id in [3, 65]:
                print(f"      [DEBUG-P{person_id}] Detection {detection_idx} not in mapping!")
            continue
        
        frame_idx, pos_in_frame = detection_idx_to_frame_pos[detection_idx]
        
        # Debug first few fetches
        if idx_in_sequence < 5 and person_id in [3, 65]:
            print(f"      [DEBUG-P{person_id}] Frame {idx_in_sequence}: det_idx={detection_idx} -> frame_idx={frame_idx}, pos={pos_in_frame}")
        
        # Get crop from crops_cache using (frame_idx, position_in_frame)
        # crops_cache structure: {frame_idx: {position_in_frame: crop_image}}
        crop = None
        if frame_idx in crops_cache and pos_in_frame in crops_cache[frame_idx]:
            crop = crops_cache[frame_idx][pos_in_frame]
        else:
            # Debug first few misses
            if idx_in_sequence < 5 and person_id in [3, 65]:
                print(f"      [DEBUG-P{person_id}] Frame {idx_in_sequence}: CACHE MISS - frame_idx {frame_idx} in cache: {frame_idx in crops_cache}, pos {pos_in_frame} in frame: {pos_in_frame in crops_cache.get(frame_idx, {})}")
        
        if crop is None or (hasattr(crop, 'size') and crop.size == 0):
            frames_skipped += 1
            continue
        
        try:
            # crop is already BGR from extraction
            # Convert BGR to RGB for PIL
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Resize crop to frame size (maintains aspect ratio)
            resized_frame = resize_crop_to_frame(crop_rgb, frame_width, frame_height)
            
            # Convert to PIL Image
            frames_list.append(Image.fromarray(resized_frame.astype('uint8')))
            frames_written += 1
            
        except Exception as e:
            frames_skipped += 1
            continue
    
    if frames_written == 0:
        return False, f"No valid frames found for person {person_id}"
    
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


def create_webp_for_top_persons(canonical_file, crops_cache_file, detections_file, output_webp_dir, 
                                 frame_width=128, frame_height=192, fps=10, num_frames=60):
    """Create animated WebP files for top 10 persons using direct detection indices"""
    
    # Load canonical persons
    print(f"📂 Loading canonical persons...")
    data = np.load(canonical_file, allow_pickle=True)
    persons = list(data['persons'])
    persons.sort(key=lambda p: len(p['frame_numbers']), reverse=True)
    print(f"   ✅ Loaded {len(persons)} canonical persons")
    
    # Check if detection_indices are present in canonical persons
    has_detection_indices = 'detection_indices' in persons[0]
    if not has_detection_indices:
        print(f"   ⚠️  WARNING: detection_indices not found in canonical persons!")
        print(f"   Available keys: {persons[0].keys()}")
        print(f"   This may indicate Stage 2/4b have not been updated to store detection indices")
        return False
    
    # DEBUG: Check detection_indices values
    print(f"\n   [DEBUG] Checking detection_indices in canonical persons:")
    for rank, person in enumerate(persons[:3], 1):
        person_id = person['person_id']
        det_inds = person.get('detection_indices', np.array([]))
        print(f"   Person {person_id}: {len(det_inds)} detection indices")
        print(f"      Type: {type(det_inds)}, dtype: {det_inds.dtype}")
        print(f"      First 10: {det_inds[:10]}")
        print(f"      Last 10: {det_inds[-10:]}")
        print(f"      Min: {np.min(det_inds)}, Max: {np.max(det_inds)}")
        print(f"      -1 count: {np.sum(det_inds == -1)}")
        print(f"      Valid: {np.sum(det_inds >= 0)}")
        
        # Check if indices are in valid range
        valid_range = np.sum((det_inds >= 0) & (det_inds < 8782))
        print(f"      In range [0, 8782): {valid_range}")
        if valid_range < len(det_inds):
            print(f"      ⚠️  WARNING: Some indices out of range!")
    
    # Load detections for mapping
    print(f"📂 Loading detections...")
    det_data = np.load(detections_file, allow_pickle=True)
    print(f"   ✅ Loaded {len(det_data['frame_numbers'])} detections")
    
    # Load crops cache from Stage 1 output
    print(f"📂 Loading crops cache...")
    if not Path(crops_cache_file).exists():
        print(f"❌ Crops cache not found: {crops_cache_file}")
        return False
    
    try:
        with open(crops_cache_file, 'rb') as f:
            crops_cache = pickle.load(f)
        print(f"   ✅ Loaded crops cache")
    except Exception as e:
        print(f"❌ Error reading crops cache: {str(e)}")
        return False
    
    # Build mapping from global detection_idx to (frame_idx, position_in_frame)
    print(f"📊 Building detection index mapping...")
    num_detections_per_frame = det_data.get('num_detections_per_frame', np.array([]))
    
    print(f"   num_detections_per_frame shape: {num_detections_per_frame.shape}")
    print(f"   First 20 values: {num_detections_per_frame[:20]}")
    print(f"   Sum: {np.sum(num_detections_per_frame)} (should equal {len(det_data['frame_numbers'])})")
    
    detection_idx_to_frame_pos = {}
    detection_idx = 0
    for frame_idx, num_dets_in_frame in enumerate(num_detections_per_frame):
        for pos_in_frame in range(int(num_dets_in_frame)):
            detection_idx_to_frame_pos[detection_idx] = (frame_idx, pos_in_frame)
            detection_idx += 1
    
    print(f"   ✅ Built mapping for {len(detection_idx_to_frame_pos)} detections")
    print(f"   [DEBUG] Sample mappings:")
    print(f"      detection 0 -> {detection_idx_to_frame_pos.get(0)}")
    print(f"      detection 1 -> {detection_idx_to_frame_pos.get(1)}")
    print(f"      detection 2 -> {detection_idx_to_frame_pos.get(2)}")
    print(f"      detection 100 -> {detection_idx_to_frame_pos.get(100)}")
    print(f"      detection 1000 -> {detection_idx_to_frame_pos.get(1000)}")
    
    # Create output directory
    webp_dir = Path(output_webp_dir) / 'webp'
    webp_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {webp_dir}")
    
    # Generate WebP files for top 10 persons
    print(f"\n🎬 Generating animated WebP files for top 10 persons...\n")
    
    success_count = 0
    failed_count = 0
    
    for rank, person in enumerate(persons[:10], 1):
        person_id = person['person_id']
        
        success, message = create_webp_for_person(
            person,
            crops_cache,
            detection_idx_to_frame_pos,
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
    
    print(f"\n{'='*70}")
    print(f"📊 WebP Generation Summary:")
    print(f"  ✅ Successful: {success_count}/10")
    print(f"  📁 Output: {webp_dir}")
    print(f"{'='*70}\n")
    
    return success_count > 0


def main():
    parser = argparse.ArgumentParser(
        description='Stage 11: Generate Animated WebP Files for Top 10 Persons (Using Direct Detection Indices)'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Get paths from config
    canonical_file = config['stage5']['output']['canonical_persons_file']
    crops_cache_file = config['stage1']['output']['crops_cache_file']  
    detections_file = config['stage1']['output']['detections_file']
    
    # Use the parent directory of canonical_persons.npz (video-specific outputs folder)
    output_dir = str(Path(canonical_file).parent)
    
    # Get WebP generation settings from config
    webp_config = config.get('stage11', {}).get('video_generation', {})
    frame_width = webp_config.get('frame_width', 128)
    frame_height = webp_config.get('frame_height', 192)
    fps = webp_config.get('fps', 10)
    num_frames = webp_config.get('max_frames', 60)
    
    print(f"\n{'='*70}")
    print(f"🎬 STAGE 11: GENERATE PERSON ANIMATED WEBP FILES (DIRECT DETECTION INDICES)")
    print(f"{'='*70}\n")
    print(f"📊 Settings: {num_frames} frames @ {fps} fps = {num_frames/fps:.1f}s per person\n")
    
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
