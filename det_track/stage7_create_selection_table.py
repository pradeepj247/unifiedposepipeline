#!/usr/bin/env python3
"""
Stage 7b: Create Visual Selection Table

Creates a visual PNG table showing all canonical persons with their high-confidence crops
for easy manual selection and comparison.

Input:
  - canonical_persons.npz (from Stage 4b)
  - crops_cache.pkl (from Stage 1)
  - tracklet_stats.npz (from Stage 3)

Output:
  - selection_table.png (visual table with crops)

This table helps users visually compare persons before selecting one with --person-id.

Usage:
    python stage7_create_selection_table.py --config configs/pipeline_config.yaml
    python stage7_create_selection_table.py --config configs/pipeline_config.yaml --verbose
"""

import argparse
import yaml
import numpy as np
import pickle
import re
import time
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None


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
        video_name = os.path.splitext(video_file)[0]
        config['global']['current_video'] = video_name
    
    return resolve_path_variables(config)


def find_best_crop(person, crops_cache, stats):
    """
    Find the best crop for a person (highest confidence frame).
    
    Args:
        person: dict with 'frame_numbers', 'confidences'
        crops_cache: dict {frame_idx: {det_idx: crop_image}}
        stats: dict with frame_to_detection_idx mapping
    
    Returns:
        crop_image (numpy array) or None
    """
    if not person.get('frame_numbers') is not None or len(person['frame_numbers']) == 0:
        return None
    
    # Find the highest confidence frame
    confidences = person['confidences']
    best_idx = np.argmax(confidences)
    best_frame = int(person['frame_numbers'][best_idx])
    
    # Try to get crop from cache
    if best_frame in crops_cache:
        crops_in_frame = crops_cache[best_frame]
        # Get first available crop (any detection index)
        for det_idx, crop_image in crops_in_frame.items():
            if crop_image is not None and crop_image.size > 0:
                return crop_image
    
    return None


def create_selection_table(config, verbose=False):
    """Create visual selection table with crops"""
    
    t_start = time.time()
    
    print(f"\n{'='*70}")
    print(f"📋 STAGE 7b: CREATE VISUAL SELECTION TABLE")
    print(f"{'='*70}\n")
    
    # Check PIL availability
    if Image is None:
        print("❌ Pillow library required: pip install pillow")
        return False
    
    # Get file paths
    canonical_file = config['stage4b_group_canonical']['output']['canonical_persons_file']
    stats_file = config['stage3_analyze']['output']['tracklet_stats_file']
    crops_cache_file = config['stage4a_reid_recovery']['input']['crops_cache_file']
    
    output_dir = Path(canonical_file).parent
    output_file = output_dir / 'selection_table.png'
    
    # Load data
    print(f"📂 Loading data...")
    t_load = time.time()
    
    # Load canonical persons
    canonical_path = Path(canonical_file)
    if not canonical_path.exists():
        print(f"❌ Canonical persons file not found: {canonical_path}")
        return False
    
    data = np.load(canonical_path, allow_pickle=True)
    persons = data['persons']
    print(f"  ✅ Loaded {len(persons)} canonical persons")
    
    # Load stats
    stats_path = Path(stats_file)
    if stats_path.exists():
        stats_data = np.load(stats_path, allow_pickle=True)
        stats = stats_data['stats']
        print(f"  ✅ Loaded stats for {len(stats)} tracklets")
    else:
        stats = None
        print(f"  ⚠️  Stats file not found (optional)")
    
    # Load crops cache
    crops_cache_path = Path(crops_cache_file)
    if crops_cache_path.exists():
        with open(crops_cache_path, 'rb') as f:
            crops_cache = pickle.load(f)
        print(f"  ✅ Loaded crops cache ({len(crops_cache)} frames)")
    else:
        crops_cache = {}
        print(f"  ⚠️  Crops cache not found - will create table without images")
    
    t_load_end = time.time()
    print(f"  ⏱️  Load time: {t_load_end - t_load:.2f}s")
    
    # Table parameters
    CROP_WIDTH = 128
    CROP_HEIGHT = 256
    MARGIN = 10
    HEADER_HEIGHT = 50
    ROW_HEIGHT = CROP_HEIGHT + 40
    TEXT_SIZE = 14
    COLUMNS = ['#', 'Person ID', 'Crop', 'Start Frame', 'End Frame', 'Appearances']
    COL_WIDTHS = [40, 80, CROP_WIDTH + 20, 100, 100, 100]
    
    TABLE_WIDTH = sum(COL_WIDTHS) + MARGIN * 2
    TABLE_HEIGHT = HEADER_HEIGHT + ROW_HEIGHT * len(persons) + MARGIN * 2
    
    print(f"\n🎨 Creating table ({TABLE_WIDTH}x{TABLE_HEIGHT} pixels)...")
    t_table = time.time()
    
    # Create image
    img = Image.new('RGB', (TABLE_WIDTH, TABLE_HEIGHT), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", TEXT_SIZE)
        font_bold = ImageFont.truetype("arialbd.ttf", TEXT_SIZE)
    except (IOError, OSError):
        # Fallback to default font
        font = ImageFont.load_default()
        font_bold = font
    
    # Draw header
    y = MARGIN
    x_positions = [MARGIN]
    for width in COL_WIDTHS[:-1]:
        x_positions.append(x_positions[-1] + width)
    
    for col_idx, (col_name, col_width, x_start) in enumerate(zip(COLUMNS, COL_WIDTHS, x_positions)):
        # Header background
        draw.rectangle(
            [(x_start, y), (x_start + col_width, y + HEADER_HEIGHT)],
            fill=(200, 200, 200),
            outline=(100, 100, 100)
        )
        # Header text
        draw.text((x_start + 5, y + 15), col_name, fill=(0, 0, 0), font=font_bold)
    
    # Draw rows
    persons_sorted = sorted(persons, key=lambda p: len(p['frame_numbers']), reverse=True)
    
    for row_idx, person in enumerate(persons_sorted):
        y = MARGIN + HEADER_HEIGHT + row_idx * ROW_HEIGHT
        
        # Get person info
        person_id = person['person_id']
        start_frame = int(person['frame_numbers'][0])
        end_frame = int(person['frame_numbers'][-1])
        appearances = len(person['frame_numbers'])
        
        # Column 0: #
        x = x_positions[0]
        draw.rectangle([(x, y), (x + COL_WIDTHS[0], y + ROW_HEIGHT)], outline=(100, 100, 100))
        draw.text((x + 5, y + 10), str(row_idx + 1), fill=(0, 0, 0), font=font)
        
        # Column 1: Person ID
        x = x_positions[1]
        draw.rectangle([(x, y), (x + COL_WIDTHS[1], y + ROW_HEIGHT)], outline=(100, 100, 100))
        draw.text((x + 5, y + 10), str(person_id), fill=(0, 0, 0), font=font)
        
        # Column 2: Crop
        x = x_positions[2]
        draw.rectangle([(x, y), (x + COL_WIDTHS[2], y + ROW_HEIGHT)], outline=(100, 100, 100))
        
        crop_img = find_best_crop(person, crops_cache, stats)
        if crop_img is not None:
            # Resize crop to fit in table cell
            crop_pil = Image.fromarray(crop_img)
            crop_pil.thumbnail((CROP_WIDTH, CROP_HEIGHT), Image.Resampling.LANCZOS)
            
            # Paste crop in center of cell
            crop_x = x + (COL_WIDTHS[2] - crop_pil.width) // 2
            crop_y = y + (ROW_HEIGHT - crop_pil.height) // 2
            img.paste(crop_pil, (crop_x, crop_y))
        
        # Column 3: Start Frame
        x = x_positions[3]
        draw.rectangle([(x, y), (x + COL_WIDTHS[3], y + ROW_HEIGHT)], outline=(100, 100, 100))
        draw.text((x + 5, y + 10), str(start_frame), fill=(0, 0, 0), font=font)
        
        # Column 4: End Frame
        x = x_positions[4]
        draw.rectangle([(x, y), (x + COL_WIDTHS[4], y + ROW_HEIGHT)], outline=(100, 100, 100))
        draw.text((x + 5, y + 10), str(end_frame), fill=(0, 0, 0), font=font)
        
        # Column 5: Appearances
        x = x_positions[5]
        draw.rectangle([(x, y), (x + COL_WIDTHS[5], y + ROW_HEIGHT)], outline=(100, 100, 100))
        draw.text((x + 5, y + 10), str(appearances), fill=(0, 0, 0), font=font)
    
    t_table_end = time.time()
    print(f"  ⏱️  Table creation time: {t_table_end - t_table:.2f}s")
    
    # Save image
    print(f"\n💾 Saving table...")
    t_save = time.time()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    img.save(output_file, 'PNG')
    
    t_save_end = time.time()
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  ✅ Saved: {output_file}")
    print(f"  📊 Size: {file_size_mb:.1f} MB")
    print(f"  ⏱️  Save time: {t_save_end - t_save:.2f}s")
    
    # Summary
    t_total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"✅ SELECTION TABLE CREATED!")
    print(f"{'='*70}\n")
    print(f"📋 Table details:")
    print(f"  - Persons: {len(persons)}")
    print(f"  - Dimensions: {TABLE_WIDTH}x{TABLE_HEIGHT} pixels")
    print(f"  - Output: {output_file.name}")
    print(f"\n⏱️  Total time: {t_total:.2f}s")
    print(f"{'='*70}\n")
    
    if verbose:
        print(f"📊 Persons in table (sorted by appearances):")
        for idx, person in enumerate(persons_sorted, 1):
            frames = len(person['frame_numbers'])
            start = int(person['frame_numbers'][0])
            end = int(person['frame_numbers'][-1])
            print(f"  {idx:2d}. Person {person['person_id']:2d}: {frames:4d} frames "
                  f"({start:4d}-{end:4d})")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Stage 7b: Create Visual Selection Table',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create selection table with default settings
  python stage7_create_selection_table.py --config configs/pipeline_config.yaml
  
  # Create table with detailed output
  python stage7_create_selection_table.py --config configs/pipeline_config.yaml --verbose
        """
    )
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create table
    success = create_selection_table(config, verbose=args.verbose)
    
    if not success:
        exit(1)


if __name__ == '__main__':
    main()
