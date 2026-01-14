#!/usr/bin/env python3
"""
Stage 10b: Generate WebP Animations (SIMPLIFIED)

Pure visualization - NO association logic!
Uses pre-organized crops_by_person.pkl from Stage 4b.

DRAMATICALLY SIMPLIFIED compared to old Stage 10:
- Old: 50+ lines of complex index conversion (detection_idx → frame+pos → crop)
- New: 20 lines - just process crops list directly!

Input:
    - crops_by_person.pkl (pre-organized from Stage 4b)

Output:
    - person_XX.webp files (top 10 persons by duration)

Usage:
    python stage10b_generate_webps.py --config configs/pipeline_config.yaml
"""

import argparse
import yaml
import numpy as np
import pickle
import time
import re
import sys
import cv2
from pathlib import Path
from tqdm import tqdm

try:
    from PIL import Image
except ImportError:
    print("Installing Pillow for WebP support...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'Pillow'])
    from PIL import Image

try:
    import h5py
except ImportError:
    print("Installing h5py for HDF5 support...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'h5py'])
    import h5py


def resolve_path_variables(config):
    """Recursively resolve ${variable} in config"""
    global_vars = config.get('global', {})
    
    # First pass: resolve variables within global section itself
    def resolve_string_once(s, vars_dict):
        if not isinstance(s, str):
            return s
        return re.sub(
            r'\$\{(\w+)\}',
            lambda m: str(vars_dict.get(m.group(1), m.group(0))),
            s
        )
    
    # Resolve global variables iteratively
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
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        config['global']['current_video'] = video_name
    
    return resolve_path_variables(config)


def create_webp_for_person_simple(person_id, person_data, output_dir,
                                   frame_width=128, frame_height=192,
                                   fps=10, num_frames=60):
    """
    Create WebP from pre-organized crops.
    
    DRAMATICALLY SIMPLIFIED: No index conversion, no detection mapping!
    Just process crops directly.
    
    Args:
        person_id: Integer person ID
        person_data: {
            'frame_numbers': np.array([5, 6, 7, ...]),
            'crops': [crop1, crop2, crop3, ...]
        }
        output_dir: Directory to save WebP file
        frame_width: Target width for WebP frames
        frame_height: Target height for WebP frames
        fps: Frames per second for animation
        num_frames: Maximum number of frames to include
    
    Returns:
        (success: bool, message: str)
    """
    crops = person_data['crops']
    frame_numbers = person_data['frame_numbers']
    
    if len(crops) == 0:
        return False, f"No crops for person {person_id}"
    
    # Frame selection: Skip first 20%, take next num_frames
    offset = min(int(len(crops) * 0.2), 50)
    end_idx = min(offset + num_frames, len(crops))
    crops_to_use = crops[offset:end_idx]
    
    if len(crops_to_use) == 0:
        return False, f"No valid frames after offset for person {person_id}"
    
    # Generate WebP frames
    frames_list = []
    for crop in crops_to_use:
        # Validate crop
        if crop is None or crop.size == 0:
            continue
        
        # Convert BGR → RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Resize to fixed dimensions
        resized = cv2.resize(crop_rgb, (frame_width, frame_height))
        
        # Convert to PIL Image
        frames_list.append(Image.fromarray(resized.astype('uint8')))
    
    if len(frames_list) == 0:
        return False, f"No valid frames for person {person_id}"
    
    # Save WebP
    webp_path = output_dir / f"person_{person_id:02d}.webp"
    duration = int(1000 / fps)  # milliseconds per frame
    
    frames_list[0].save(
        str(webp_path),
        format='WEBP',
        save_all=True,
        append_images=frames_list[1:],
        duration=duration,
        loop=0,
        quality=80
    )
    
    file_size_mb = webp_path.stat().st_size / (1024 * 1024)
    return True, f"person_{person_id:02d}.webp ({len(frames_list)} frames, {file_size_mb:.2f} MB)"


def load_top_n_persons_from_hdf5(hdf5_file, max_persons=10):
    """
    Load only top N persons from HDF5 file.
    
    KEY OPTIMIZATION: Only reads needed persons, not all 48!
    This saves ~2s vs loading entire file.
    
    Args:
        hdf5_file: Path to crops_by_person.h5
        max_persons: Number of top persons to load (default: 10)
    
    Returns:
        crops_by_person: Dict mapping person_id → person_data
    """
    crops_by_person = {}
    
    with h5py.File(hdf5_file, 'r') as f:
        # Get all person groups
        person_ids = []
        person_sizes = []
        
        for key in f.keys():
            if key.startswith('person_') and key != 'metadata':
                person_id = int(key.split('_')[1])
                person_group = f[key]
                num_crops = len(person_group['frame_numbers'])
                person_ids.append(person_id)
                person_sizes.append(num_crops)
        
        # Sort by size (duration) and take top N
        sorted_indices = np.argsort(person_sizes)[::-1][:max_persons]
        top_person_ids = [person_ids[i] for i in sorted_indices]
        
        # Load only top persons
        print(f"Loading top {len(top_person_ids)} persons from HDF5 (out of {len(person_ids)} total)...")
        for person_id in tqdm(top_person_ids, desc="Loading HDF5"):
            person_key = f'person_{person_id:03d}'
            person_group = f[person_key]
            
            # Load crops
            crops_group = person_group['crops']
            crops = []
            for i in range(len(crops_group)):
                crop = crops_group[str(i)][:]
                crops.append(crop)
            
            # Load metadata
            frame_numbers = person_group['frame_numbers'][:]
            bboxes = person_group['bboxes'][:]
            confidences = person_group['confidences'][:]
            
            crops_by_person[person_id] = {
                'frame_numbers': frame_numbers,
                'crops': crops,
                'bboxes': bboxes,
                'confidences': confidences
            }
    
    return crops_by_person


def load_crops_by_person(file_path, max_persons=10):
    """
    Load crops with auto-format detection (HDF5 or pickle).
    
    Args:
        file_path: Path to crops_by_person file (.h5 or .pkl)
        max_persons: Number of top persons to load (default: 10)
    
    Returns:
        crops_by_person: Dict mapping person_id → person_data
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.h5':
        print("Detected HDF5 format")
        return load_top_n_persons_from_hdf5(file_path, max_persons)
    else:
        print("Detected pickle format (loading all persons)")
        with open(file_path, 'rb') as f:
            crops_by_person = pickle.load(f)
        
        # Sort and limit to top N
        sorted_persons = sorted(
            crops_by_person.items(),
            key=lambda x: len(x[1]['crops']),
            reverse=True
        )
        top_persons = dict(sorted_persons[:max_persons])
        return top_persons


def create_webps_for_top_persons(crops_by_person_file, output_dir, config):
    """
    Generate WebPs for top 10 persons (by duration).
    
    Args:
        crops_by_person_file: Path to crops_by_person file (.h5 or .pkl)
        output_dir: Directory to save WebP files
        config: Stage configuration dict
    
    Returns:
        success_count: Number of WebPs successfully created
    """
    # Load pre-organized crops (auto-detects format)
    max_persons = config.get('max_persons', 10)
    print(f"Loading pre-organized crops (top {max_persons} persons)...")
    crops_by_person = load_crops_by_person(crops_by_person_file, max_persons)
    
    print(f"  Loaded {len(crops_by_person)} persons")
    
    # Sort by number of crops (duration) - already sorted if HDF5
    sorted_persons = sorted(
        crops_by_person.items(),
        key=lambda x: len(x[1]['crops']),
        reverse=True
    )
    
    print(f"\nGenerating WebPs for {len(sorted_persons)} persons...")
    
    # Extract parameters
    video_config = config.get('video_generation', {})
    frame_width = video_config.get('frame_width', 128)
    frame_height = video_config.get('frame_height', 192)
    fps = video_config.get('fps', 10)
    num_frames = video_config.get('max_frames', 60)
    
    # Create output directory
    webp_dir = Path(output_dir)
    webp_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate WebPs
    success_count = 0
    failed_persons = []
    
    for person_id, person_data in tqdm(sorted_persons, desc="Generating WebPs"):
        success, message = create_webp_for_person_simple(
            person_id,
            person_data,
            webp_dir,
            frame_width=frame_width,
            frame_height=frame_height,
            fps=fps,
            num_frames=num_frames
        )
        
        if success:
            success_count += 1
            print(f"  ✅ Person {person_id}: {message}")
        else:
            failed_persons.append((person_id, message))
            print(f"  ❌ Person {person_id}: {message}")
    
    # Report failures
    if failed_persons:
        print(f"\n⚠️  Failed to create WebPs for {len(failed_persons)} persons:")
        for person_id, message in failed_persons:
            print(f"    Person {person_id}: {message}")
    
    return success_count


def run_generate_webps(config):
    """Main function for Stage 10b"""
    
    # Extract config parameters
    stage_config = config.get('stage10b', {})
    crops_by_person_file = stage_config['input']['crops_by_person_file']
    webp_dir = stage_config['output']['webp_dir']
    
    # Timing sidecar
    timing = {
        'stage': 'stage10b_generate_webps',
        'start_time': time.time()
    }
    
    print("\n" + "="*60)
    print("STAGE 10b: GENERATE WEBP ANIMATIONS (SIMPLIFIED)")
    print("="*60)
    print(f"Input:  {crops_by_person_file}")
    print(f"Output: {webp_dir}")
    print("-"*60)
    
    # Check if input exists
    if not Path(crops_by_person_file).exists():
        print(f"❌ ERROR: Crops by person file not found: {crops_by_person_file}")
        sys.exit(1)
    
    # Generate WebPs
    gen_start = time.time()
    success_count = create_webps_for_top_persons(
        crops_by_person_file,
        webp_dir,
        stage_config
    )
    gen_time = time.time() - gen_start
    timing['generation_time'] = gen_time
    timing['num_webps_created'] = success_count
    
    print(f"\n✅ Created {success_count} WebP animations ({gen_time:.3f}s)")
    
    # Timing summary
    timing['end_time'] = time.time()
    timing['total_time'] = timing['end_time'] - timing['start_time']
    
    # Save timing sidecar
    timing_file = Path(webp_dir) / 'webp_generation_timing.json'
    with open(timing_file, 'w') as f:
        import json
        json.dump(timing, f, indent=2)
    
    print("\n" + "="*60)
    print("STAGE 10b COMPLETE")
    print("="*60)
    print(f"Total time: {timing['total_time']:.3f}s")
    print(f"  Generation:    {timing['generation_time']:.3f}s")
    print(f"\nOutput: {webp_dir}")
    print(f"Timing: {timing_file}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Stage 10b: Generate WebP Animations (Simplified)')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to pipeline config YAML')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Run WebP generation
    run_generate_webps(config)


if __name__ == '__main__':
    main()
