#!/usr/bin/env python3
"""
Stage 4b: Reorganize Crops by Person

Pre-organizes crops by person_id for efficient downstream access.
Uses detection_indices from canonical_persons to map frame+position → crop.

Input:
    - crops_cache.pkl (from Stage 1): {frame_idx: {position_in_frame: crop_image_bgr}}
    - canonical_persons.npz (from Stage 3b): List of person dicts with detection_indices

Output:
    - crops_by_person.pkl: {person_id: {'frame_numbers': [...], 'crops': [...], ...}}

Usage:
    python stage4b_reorganize_crops.py --config configs/pipeline_config.yaml
"""

import argparse
import yaml
import numpy as np
import pickle
import time
import re
import sys
from pathlib import Path
from tqdm import tqdm


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


def reorganize_crops_by_person(crops_cache, canonical_persons):
    """
    Reorganize raw crops into person-indexed structure.
    
    Uses detection_indices to map frame+position → crop.
    
    Args:
        crops_cache: {frame_idx: {position_in_frame: crop_image_bgr}}
        canonical_persons: List of person dicts with detection_indices
    
    Returns:
        crops_by_person: {
            person_id: {
                'frame_numbers': np.array([...]),
                'crops': [crop1, crop2, ...],
                'bboxes': np.array([...]),
                'confidences': np.array([...])
            }
        }
    """
    crops_by_person = {}
    
    for person in tqdm(canonical_persons, desc="Reorganizing crops by person"):
        person_id = person['person_id']
        frame_numbers = person['frame_numbers']
        detection_indices = person['detection_indices']  # Position in each frame
        bboxes = person.get('bboxes', None)
        confidences = person.get('confidences', None)
        
        crops_list = []
        valid_frames = []
        valid_bboxes = []
        valid_confs = []
        
        # Extract crops for this person using detection_indices
        for i, (frame_num, pos_in_frame) in enumerate(zip(frame_numbers, detection_indices)):
            frame_idx = int(frame_num)
            pos = int(pos_in_frame)
            
            # Look up crop: crops_cache[frame][position]
            if frame_idx in crops_cache and pos in crops_cache[frame_idx]:
                crop = crops_cache[frame_idx][pos]
                
                # Validate crop
                if crop is not None and crop.size > 0:
                    crops_list.append(crop)
                    valid_frames.append(frame_num)
                    
                    if bboxes is not None:
                        valid_bboxes.append(bboxes[i])
                    if confidences is not None:
                        valid_confs.append(confidences[i])
        
        # Store organized data for this person
        if len(crops_list) > 0:
            crops_by_person[person_id] = {
                'frame_numbers': np.array(valid_frames, dtype=np.int64),
                'crops': crops_list,  # List of numpy arrays (varying sizes OK)
                'bboxes': np.array(valid_bboxes) if valid_bboxes else None,
                'confidences': np.array(valid_confs) if valid_confs else None
            }
    
    return crops_by_person


def run_reorganize_crops(config):
    """Main function for Stage 4b"""
    
    # Extract config parameters
    stage_config = config.get('stage4b', {})
    crops_cache_file = stage_config['input']['crops_cache_file']
    canonical_persons_file = stage_config['input']['canonical_persons_file']
    output_file = stage_config['output']['crops_by_person_file']
    
    # Timing sidecar
    timing = {
        'stage': 'stage4b_reorganize_crops',
        'start_time': time.time()
    }
    
    print("\n" + "="*60)
    print("STAGE 4b: REORGANIZE CROPS BY PERSON")
    print("="*60)
    print(f"Input (crops cache):      {crops_cache_file}")
    print(f"Input (canonical persons): {canonical_persons_file}")
    print(f"Output:                   {output_file}")
    print("-"*60)
    
    # Check if inputs exist
    if not Path(crops_cache_file).exists():
        print(f"❌ ERROR: Crops cache file not found: {crops_cache_file}")
        sys.exit(1)
    
    if not Path(canonical_persons_file).exists():
        print(f"❌ ERROR: Canonical persons file not found: {canonical_persons_file}")
        sys.exit(1)
    
    # Load crops cache
    print("Loading crops cache...")
    load_start = time.time()
    with open(crops_cache_file, 'rb') as f:
        crops_cache = pickle.load(f)
    load_time = time.time() - load_start
    timing['load_crops_time'] = load_time
    
    num_frames = len(crops_cache)
    total_crops = sum(len(frame_crops) for frame_crops in crops_cache.values())
    print(f"  Loaded {total_crops} crops across {num_frames} frames ({load_time:.3f}s)")
    
    # Load canonical persons
    print("Loading canonical persons...")
    persons_data = np.load(canonical_persons_file, allow_pickle=True)
    canonical_persons = persons_data['persons']
    print(f"  Loaded {len(canonical_persons)} canonical persons")
    
    # Reorganize
    print("\nReorganizing crops by person...")
    reorg_start = time.time()
    crops_by_person = reorganize_crops_by_person(crops_cache, canonical_persons)
    reorg_time = time.time() - reorg_start
    timing['reorganize_time'] = reorg_time
    
    # Statistics
    total_persons = len(crops_by_person)
    total_crops_reorg = sum(len(data['crops']) for data in crops_by_person.values())
    timing['num_persons'] = total_persons
    timing['num_crops'] = total_crops_reorg
    
    print(f"\n✅ Reorganized {total_crops_reorg} crops for {total_persons} persons ({reorg_time:.3f}s)")
    
    # Save
    print(f"\nSaving to {output_file}...")
    save_start = time.time()
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(crops_by_person, f)
    save_time = time.time() - save_start
    timing['save_time'] = save_time
    
    file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"  Saved: {file_size_mb:.2f} MB ({save_time:.3f}s)")
    
    # Timing summary
    timing['end_time'] = time.time()
    timing['total_time'] = timing['end_time'] - timing['start_time']
    
    # Save timing sidecar
    timing_file = output_file.replace('.pkl', '_timing.json')
    with open(timing_file, 'w') as f:
        import json
        json.dump(timing, f, indent=2)
    
    print("\n" + "="*60)
    print("STAGE 4b COMPLETE")
    print("="*60)
    print(f"Total time: {timing['total_time']:.3f}s")
    print(f"  Load crops:    {timing['load_crops_time']:.3f}s")
    print(f"  Reorganize:    {timing['reorganize_time']:.3f}s")
    print(f"  Save:          {timing['save_time']:.3f}s")
    print(f"\nOutput: {output_file}")
    print(f"Timing: {timing_file}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Stage 4b: Reorganize Crops by Person')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to pipeline config YAML')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Run reorganization
    run_reorganize_crops(config)


if __name__ == '__main__':
    main()
