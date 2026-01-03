#!/usr/bin/env python3
"""
Stage 6b: Create Person Selection Grid (FIXED - Uses Crops Cache, NOT Video Seeking)

Creates a single grid image showing all top persons using PRE-CACHED CROPS:
- No video seeking at all!
- Crops loaded from crops_cache.pkl (created in Stage 1)
- One high-confidence crop per person
- Sorted by duration (longest appearance first)

This is the CORRECT implementation that respects the crop caching design.

Usage:
    python stage6b_create_selection_grid_fixed.py --config configs/pipeline_config.yaml
"""

import argparse
import numpy as np
import pickle
import yaml
import re
import os
from pathlib import Path
import time
from PIL import Image, ImageDraw, ImageFont


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
        video_name = os.path.splitext(video_file)[0]
        config['global']['current_video'] = video_name
    
    return resolve_path_variables(config)


def get_best_crop_for_person(person, crops_cache):
    """
    Get the best crop for a person (highest confidence frame).
    
    Args:
        person: dict with 'frame_numbers', 'confidences'
        crops_cache: dict {frame_idx: {det_idx: crop_image}}
    
    Returns:
        crop_image (numpy array) or None
    """
    if not person.get('frame_numbers') is not None or len(person['frame_numbers']) == 0:
        return None
    
    # Find the highest confidence frame
    confidences = person['confidences']
    best_idx = np.argmax(confidences)
    best_frame = int(person['frame_numbers'][best_idx])
    
    # Get crop from cache
    if best_frame in crops_cache:
        crops_in_frame = crops_cache[best_frame]
        # Get first available crop (any detection index)
        for crop_image in crops_in_frame.values():
            if crop_image is not None and isinstance(crop_image, np.ndarray):
                return crop_image
    
    return None


def create_grid_from_crops(crops_dict, persons_list, grid_shape=(2, 5), cell_size=(384, 216)):
    """
    Create a grid image from crops (no video seeking needed!)
    
    Args:
        crops_dict: {person_id: crop_image}
        persons_list: List of persons sorted by duration
        grid_shape: (rows, cols)
        cell_size: (width, height) per cell
    
    Returns:
        PIL Image or None
    """
    rows, cols = grid_shape
    cell_w, cell_h = cell_size
    
    # Create blank grid
    grid_w = cols * cell_w
    grid_h = rows * cell_h
    grid = Image.new('RGB', (grid_w, grid_h), color='white')
    
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(grid)
    
    # Place crops in grid
    for idx, person in enumerate(persons_list[:rows * cols]):
        row = idx // cols
        col = idx % cols
        
        x_start = col * cell_w
        y_start = row * cell_h
        
        # Get crop
        crop = crops_dict.get(person['person_id'])
        
        if crop is not None:
            # Resize crop to fit cell
            crop_pil = Image.fromarray(crop)
            crop_pil.thumbnail((cell_w - 10, cell_h - 40), Image.Resampling.LANCZOS)
            
            # Center crop in cell
            crop_x = x_start + (cell_w - crop_pil.width) // 2
            crop_y = y_start + 10 + (cell_h - 50 - crop_pil.height) // 2
            
            grid.paste(crop_pil, (crop_x, crop_y))
        
        # Add label
        label = f"P{person['person_id']}\n{len(person['frame_numbers'])} frames"
        draw.text((x_start + 5, y_start + cell_h - 30), label, fill=(0, 0, 0), font=font)
        
        # Draw cell border
        draw.rectangle(
            [(x_start, y_start), (x_start + cell_w, y_start + cell_h)],
            outline=(200, 200, 200)
        )
    
    return grid


def main():
    parser = argparse.ArgumentParser(
        description='Stage 6b: Create Person Selection Grid (Using Cached Crops)'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Get paths
    canonical_file = config['stage4b_group_canonical']['output']['canonical_persons_file']
    crops_cache_file = config['stage4a_reid_recovery']['input']['crops_cache_file']
    grid_output = Path(config['stage6b_create_selection_grid']['output']['cropped_grid'])
    
    print(f"\n{'='*70}")
    print(f"🎬 STAGE 6b: CREATE SELECTION GRID (FROM CACHED CROPS)")
    print(f"{'='*70}\n")
    
    # Load canonical persons
    print(f"📂 Loading canonical persons...")
    t_start = time.time()
    
    data = np.load(canonical_file, allow_pickle=True)
    persons = list(data['persons'])
    
    # Sort by duration (descending)
    persons.sort(key=lambda p: len(p['frame_numbers']), reverse=True)
    
    print(f"   ✅ Loaded {len(persons)} persons")
    
    # Load crops cache
    print(f"\n📂 Loading crops cache...")
    if not Path(crops_cache_file).exists():
        print(f"❌ Crops cache not found: {crops_cache_file}")
        return False
    
    with open(crops_cache_file, 'rb') as f:
        crops_cache = pickle.load(f)
    
    print(f"   ✅ Loaded crops from {len(crops_cache)} frames")
    
    # Extract best crop for each person (from cache, NO seeking!)
    print(f"\n🎨 Extracting crops from cache (instant, no video seeking)...")
    t_extract = time.time()
    
    crops_dict = {}
    missing_count = 0
    
    for person in persons:
        crop = get_best_crop_for_person(person, crops_cache)
        if crop is not None:
            crops_dict[person['person_id']] = crop
        else:
            missing_count += 1
    
    t_extract_end = time.time()
    
    print(f"   ✅ Extracted {len(crops_dict)} crops in {t_extract_end - t_extract:.3f}s")
    if missing_count > 0:
        print(f"   ⚠️  {missing_count} persons have no crops in cache")
    
    # Show top 10
    print(f"\n📊 Top 10 Persons (sorted by duration):")
    for idx, person in enumerate(persons[:10], 1):
        frames = len(person['frame_numbers'])
        has_crop = "✅" if person['person_id'] in crops_dict else "❌"
        print(f"   {idx:2d}. Person {person['person_id']:3d}: {frames:4d} frames {has_crop}")
    
    # Create grid
    print(f"\n🖼️  Creating grid image (2×5 layout)...")
    t_grid = time.time()
    
    grid = create_grid_from_crops(crops_dict, persons[:10], grid_shape=(2, 5), cell_size=(384, 216))
    
    if grid is None:
        print(f"❌ Failed to create grid!")
        return False
    
    # Save
    grid_output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(grid_output)
    
    file_size = grid_output.stat().st_size / (1024 * 1024)
    t_grid_end = time.time()
    
    print(f"   ✅ Saved: {grid_output.name} ({file_size:.2f} MB)")
    
    t_total = time.time() - t_start
    
    print(f"\n{'='*70}")
    print(f"✅ GRID CREATED (NO VIDEO SEEKING!)")
    print(f"{'='*70}\n")
    print(f"⏱️  Total Time: {t_total:.2f}s")
    print(f"   • Cache load: {t_extract - t_start:.3f}s")
    print(f"   • Crop extraction: {t_extract_end - t_extract:.3f}s (instant from cache)")
    print(f"   • Grid creation: {t_grid_end - t_grid:.3f}s")
    print(f"\n📦 Output:")
    print(f"   • {grid_output.name}")
    print(f"   • {len(crops_dict)} persons with crops")
    print(f"\n💡 This is the correct implementation - NO VIDEO SEEKING!")
    print(f"{'='*70}\n")
    
    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
