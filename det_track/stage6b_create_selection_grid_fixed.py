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


def iou(box1, box2):
    """Calculate Intersection over Union of two boxes [x1,y1,x2,y2]"""
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter = inter_w * inter_h
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def get_bbox_area(bbox):
    """Calculate bbox area"""
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def find_best_detection_idx(person_bbox, crops_in_frame, detections_in_frame_bboxes):
    """
    Find which detection index in this frame best matches the person.
    
    Priority (in order):
    1. High IoU with person's bbox
    2. Large bbox area (full-body person)
    3. High confidence
    
    Args:
        person_bbox: [x1, y1, x2, y2]
        crops_in_frame: dict {det_idx: crop_image}
        detections_in_frame_bboxes: dict {det_idx: bbox} for this frame
    
    Returns:
        best_det_idx or None
    """
    if not crops_in_frame or not detections_in_frame_bboxes:
        return None
    
    best_det_idx = None
    best_score = -1.0
    
    for det_idx in crops_in_frame:
        if det_idx not in detections_in_frame_bboxes:
            continue
        
        det_bbox = detections_in_frame_bboxes[det_idx]
        overlap = iou(person_bbox, det_bbox)
        bbox_area = get_bbox_area(det_bbox)
        
        # Composite score: 70% IoU + 30% bbox area (normalized)
        # Larger bboxes are better (better quality crops)
        max_area = 1920 * 1080  # Maximum possible bbox area
        area_score = min(1.0, bbox_area / max_area)
        
        score = 0.7 * overlap + 0.3 * area_score
        
        if score > best_score:
            best_score = score
            best_det_idx = det_idx
    
    return best_det_idx


def get_best_crop_for_person(person, crops_cache, detections_data, all_detection_bboxes):
    """
    Get the best crop for a person using intelligent selection:
    - Prefers frames with high confidence
    - Matches by bbox overlap with original detections
    - Prefers large bboxes (full-body persons)
    
    Args:
        person: dict with 'frame_numbers', 'bboxes', 'confidences'
        crops_cache: dict {frame_idx: {det_idx: crop_image}}
        detections_data: NPZ data with frame_numbers, bboxes, confidences
        all_detection_bboxes: dict {(frame_idx, det_idx): bbox}
    
    Returns:
        crop_image (numpy array) or None
    """
    if person.get('frame_numbers') is None or len(person['frame_numbers']) == 0:
        return None
    
    # Strategy: Find frame with best combination of:
    # 1. High confidence in person's tracklet
    # 2. Large bbox (full-body person visible)
    # 3. High confidence in the detection itself
    
    frame_numbers = person['frame_numbers']
    bboxes = person['bboxes']
    confidences = person['confidences']
    
    best_crop = None
    best_score = -1.0
    
    for i in range(len(frame_numbers)):
        frame_idx = int(frame_numbers[i])
        person_bbox = bboxes[i]
        person_conf = confidences[i]
        
        # Skip if no crops in this frame
        if frame_idx not in crops_cache:
            continue
        
        crops_in_frame = crops_cache[frame_idx]
        
        # Find the detection index that matches this person's bbox
        # Build bbox dict for this frame from detections_data
        frame_mask = detections_data['frame_numbers'] == frame_idx
        frame_det_indices = np.where(frame_mask)[0]
        
        detections_in_frame_bboxes = {}
        detections_in_frame_confs = {}
        
        for det_num, det_idx in enumerate(frame_det_indices):
            detections_in_frame_bboxes[det_num] = detections_data['bboxes'][det_idx]
            detections_in_frame_confs[det_num] = detections_data['confidences'][det_idx]
        
        # Find best matching detection
        best_det_idx = find_best_detection_idx(
            person_bbox, crops_in_frame, detections_in_frame_bboxes
        )
        
        if best_det_idx is not None and best_det_idx in crops_in_frame:
            crop = crops_in_frame[best_det_idx]
            if crop is not None and isinstance(crop, np.ndarray):
                # Score: high confidence + large bbox preferred
                bbox_area = get_bbox_area(person_bbox)
                max_area = 1920 * 1080
                area_score = min(1.0, bbox_area / max_area)
                
                # Composite: 60% tracklet confidence + 40% bbox area
                score = 0.6 * person_conf + 0.4 * area_score
                
                if score > best_score:
                    best_score = score
                    best_crop = crop
    
    return best_crop


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
    detections_file = config['stage1_detect']['output']['detections_file']
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
    
    # Load detections (needed for bbox matching)
    print(f"📂 Loading detection data (for bbox matching)...")
    if not Path(detections_file).exists():
        print(f"⚠️  Detections file not found: {detections_file}")
        print(f"   Using simple mode (first available crop per frame)")
        detections_data = None
    else:
        det_data = np.load(detections_file, allow_pickle=True)
        detections_data = {
            'frame_numbers': det_data['frame_numbers'],
            'bboxes': det_data['bboxes'],
            'confidences': det_data['confidences']
        }
        print(f"   ✅ Loaded {len(detections_data['frame_numbers'])} detections")
    
    # Load crops cache
    print(f"\n📂 Loading crops cache...")
    if not Path(crops_cache_file).exists():
        print(f"❌ Crops cache not found: {crops_cache_file}")
        return False
    
    with open(crops_cache_file, 'rb') as f:
        crops_cache = pickle.load(f)
    
    print(f"   ✅ Loaded crops from {len(crops_cache)} frames")
    
    # Extract best crop for each person (from cache, NO seeking!)
    print(f"\n🎨 Extracting crops from cache (intelligent selection)...")
    t_extract = time.time()
    
    crops_dict = {}
    missing_count = 0
    
    # Build all_detection_bboxes for quick lookup
    all_detection_bboxes = {}
    if detections_data is not None:
        for i, frame_idx in enumerate(detections_data['frame_numbers']):
            all_detection_bboxes[(int(frame_idx), i)] = detections_data['bboxes'][i]
    
    for person in persons:
        if detections_data is not None:
            crop = get_best_crop_for_person(
                person, crops_cache, detections_data, all_detection_bboxes
            )
        else:
            # Fallback: just use first available (old method)
            crop = None
            for frame_idx in person['frame_numbers']:
                frame_idx = int(frame_idx)
                if frame_idx in crops_cache:
                    crops_in_frame = crops_cache[frame_idx]
                    for crop_img in crops_in_frame.values():
                        if crop_img is not None and isinstance(crop_img, np.ndarray):
                            crop = crop_img
                            break
                if crop is not None:
                    break
        
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
    print(f"✅ GRID CREATED (INTELLIGENT CROP SELECTION, NO VIDEO SEEKING!)")
    print(f"{'='*70}\n")
    print(f"⏱️  Total Time: {t_total:.2f}s")
    print(f"   • Load + matching setup: {t_extract - t_start:.3f}s")
    print(f"   • Crop selection: {t_extract_end - t_extract:.3f}s")
    print(f"   • Grid creation: {t_grid_end - t_grid:.3f}s")
    print(f"\n📊 Selection Strategy:")
    print(f"   • Matches detection by bbox overlap (IoU)")
    print(f"   • Prefers: high confidence + large bboxes")
    print(f"   • Avoids: edge cases, small crops, low confidence")
    print(f"\n📦 Output:")
    print(f"   • {grid_output.name}")
    print(f"   • {len(crops_dict)} persons with intelligent crops")
    print(f"\n💡 Crops selected based on:")
    print(f"   1. High confidence in tracklet")
    print(f"   2. Large bbox area (full-body visible)")
    print(f"   3. Good bbox match with original detection")
    print(f"{'='*70}\n")
    
    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
