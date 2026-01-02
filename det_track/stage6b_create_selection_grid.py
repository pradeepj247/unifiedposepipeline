#!/usr/bin/env python3
"""
Stage 6b: Create Selection Grid Images

Creates two selection grid images for manual person selection:
1. Full Frame Grid: Shows frames with all visible top persons marked
2. Cropped Grid: Shows individual person crops from their best frames

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
from collections import defaultdict
import math

# Suppress OpenCV/FFmpeg h264 warnings
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
cv2.setLogLevel(0)


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


# Color palette (BGR format) - same as Stage 6
COLORS = [
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 0, 128),    # Purple
    (0, 128, 128),    # Olive
    (128, 128, 0),    # Teal
    (255, 128, 0),    # Orange
    (0, 128, 255),    # Light Blue
    (128, 255, 0),    # Lime
    (255, 0, 128),    # Pink
    (128, 0, 255),    # Violet
    (0, 255, 128),    # Spring Green
]


def get_person_color(person_id, person_id_list):
    """Get consistent color for person based on their rank in list"""
    try:
        rank = person_id_list.index(person_id)
        return COLORS[rank % len(COLORS)]
    except ValueError:
        return COLORS[0]


def load_top_persons(canonical_persons_file, min_duration_seconds, video_fps, max_persons=10):
    """Load and filter top persons"""
    data = np.load(canonical_persons_file, allow_pickle=True)
    persons = data['persons']
    
    min_frames = int(min_duration_seconds * video_fps)
    print(f"\n📏 Filtering persons: min {min_duration_seconds}s = {min_frames} frames @ {video_fps:.2f} fps")
    
    # Filter by frame count
    persons_filtered = []
    for p in persons:
        frame_count = len(p['frame_numbers'])
        if frame_count >= min_frames:
            persons_filtered.append(p)
    
    # Sort by frame count descending
    persons_filtered.sort(key=lambda p: len(p['frame_numbers']), reverse=True)
    
    # Limit to top N
    top_persons = persons_filtered[:max_persons]
    
    print(f"   Found {len(persons_filtered)} persons with >={min_frames} frames")
    print(f"   Selected top {len(top_persons)} persons")
    
    return top_persons


def select_frames_greedy_coverage(persons, target_count=10):
    """
    Select frames using greedy coverage algorithm
    Returns list of (frame_num, persons_in_frame) tuples
    """
    # Build frame -> persons mapping
    frame_persons_map = defaultdict(set)
    for person in persons:
        person_id = person['person_id']
        for frame_num in person['frame_numbers']:
            frame_persons_map[int(frame_num)].add(person_id)
    
    covered_persons = set()
    selected_frames = []
    all_person_ids = {p['person_id'] for p in persons}
    
    # Greedy selection: pick frames until all persons covered
    while covered_persons != all_person_ids:
        # Find frame with most uncovered persons
        best_frame = None
        best_uncovered_count = 0
        best_persons = set()
        
        for frame_num, persons_in_frame in frame_persons_map.items():
            uncovered = persons_in_frame - covered_persons
            if len(uncovered) > best_uncovered_count:
                best_uncovered_count = len(uncovered)
                best_frame = frame_num
                best_persons = persons_in_frame
        
        if best_frame is None:
            break
        
        selected_frames.append((best_frame, best_persons))
        covered_persons.update(best_persons)
        del frame_persons_map[best_frame]  # Don't reuse same frame
    
    print(f"\n   Greedy coverage: {len(selected_frames)} frames cover all {len(all_person_ids)} persons")
    
    # If we have less than target_count frames, add more frames
    if len(selected_frames) < target_count and frame_persons_map:
        # Add frames with most persons (showing variety)
        remaining_frames = sorted(
            frame_persons_map.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        for frame_num, persons_in_frame in remaining_frames[:target_count - len(selected_frames)]:
            selected_frames.append((frame_num, persons_in_frame))
    
    # Sort by frame number
    selected_frames.sort(key=lambda x: x[0])
    
    print(f"   Total frames selected: {len(selected_frames)}")
    
    return selected_frames


def find_best_frame_for_person(person, video_width, video_height):
    """
    Find best frame for a person based on scoring:
    - Confidence: 40%
    - Bbox size: 30%
    - Centrality: 20%
    - Aspect ratio: 10%
    """
    frame_numbers = person['frame_numbers']
    bboxes = person['bboxes']
    confidences = person['confidences']
    
    best_score = -1
    best_frame_idx = 0
    
    video_center_x = video_width / 2
    video_center_y = video_height / 2
    
    for idx in range(len(frame_numbers)):
        bbox = bboxes[idx]
        conf = confidences[idx]
        
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Confidence score (normalized)
        conf_score = float(conf)
        
        # Size score (normalized by video dimensions)
        size = width * height
        max_size = video_width * video_height
        size_score = min(size / (max_size * 0.5), 1.0)  # Cap at 50% of video size
        
        # Centrality score
        dist_from_center = math.sqrt((center_x - video_center_x)**2 + (center_y - video_center_y)**2)
        max_dist = math.sqrt(video_center_x**2 + video_center_y**2)
        centrality_score = 1.0 - (dist_from_center / max_dist)
        
        # Aspect ratio score (ideal standing person ~1:2)
        aspect_ratio = width / height if height > 0 else 0
        ideal_ratio = 0.5
        aspect_score = 1.0 - min(abs(aspect_ratio - ideal_ratio) / ideal_ratio, 1.0)
        
        # Combined score
        score = (0.4 * conf_score + 
                0.3 * size_score + 
                0.2 * centrality_score + 
                0.1 * aspect_score)
        
        if score > best_score:
            best_score = score
            best_frame_idx = idx
    
    return int(frame_numbers[best_frame_idx]), best_score


def draw_person_bbox(frame, bbox, person_id, color, thickness=2):
    """Draw bounding box and label on frame"""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label with background
    label = f"P{person_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    
    # Background rectangle
    cv2.rectangle(frame, 
                 (x1, y1 - text_height - baseline - 5), 
                 (x1 + text_width + 5, y1), 
                 color, -1)
    
    # Text with outline for visibility
    cv2.putText(frame, label, (x1 + 2, y1 - baseline - 2), 
               font, font_scale, (255, 255, 255), font_thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, label, (x1 + 2, y1 - baseline - 2), 
               font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
    
    return frame


def create_fullframe_grid(video_path, persons, selected_frames, output_path, cell_size=(384, 216)):
    """Create grid of full frames with all visible persons marked"""
    
    print(f"\n🎨 Creating full frame grid...")
    
    # Build person_id -> person mapping
    person_dict = {p['person_id']: p for p in persons}
    person_id_list = [p['person_id'] for p in persons]
    
    # Build frame -> person bboxes mapping
    frame_person_bboxes = defaultdict(list)
    for person in persons:
        person_id = person['person_id']
        frames = person['frame_numbers']
        bboxes = person['bboxes']
        for frame_num, bbox in zip(frames, bboxes):
            frame_person_bboxes[int(frame_num)].append((person_id, bbox))
    
    # Load video
    cap = cv2.VideoCapture(str(video_path))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate grid dimensions
    num_frames = len(selected_frames)
    cols = 5
    rows = math.ceil(num_frames / cols)
    
    grid_width = cell_size[0] * cols
    grid_height = cell_size[1] * rows
    
    # Create blank grid
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Process each selected frame
    for idx, (frame_num, persons_in_frame) in enumerate(selected_frames):
        row = idx // cols
        col = idx % cols
        
        # Read frame from video
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            print(f"   ⚠️  Could not read frame {frame_num}")
            continue
        
        # Draw all persons in this frame
        for person_id, bbox in frame_person_bboxes[frame_num]:
            if person_id in person_id_list:  # Only draw top persons
                color = get_person_color(person_id, person_id_list)
                frame = draw_person_bbox(frame, bbox, person_id, color, thickness=3)
        
        # Add frame number annotation
        cv2.putText(frame, f"Frame {frame_num}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Frame {frame_num}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Resize to cell size
        frame_resized = cv2.resize(frame, cell_size)
        
        # Place in grid
        y_start = row * cell_size[1]
        y_end = y_start + cell_size[1]
        x_start = col * cell_size[0]
        x_end = x_start + cell_size[0]
        
        grid[y_start:y_end, x_start:x_end] = frame_resized
    
    cap.release()
    
    # Save grid
    cv2.imwrite(str(output_path), grid)
    print(f"   ✅ Saved: {output_path.name} ({grid_width}x{grid_height})")
    
    return grid_width, grid_height


def create_cropped_grid(video_path, persons, output_path, padding_percent=10, max_cell_size=(384, 600)):
    """Create grid of person crops from their best frames"""
    
    print(f"\n🎨 Creating cropped person grid...")
    
    person_id_list = [p['person_id'] for p in persons]
    
    # Get video dimensions
    cap = cv2.VideoCapture(str(video_path))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Find best frame for each person
    person_best_frames = {}
    for person in persons:
        best_frame, score = find_best_frame_for_person(person, video_width, video_height)
        person_best_frames[person['person_id']] = {
            'frame': best_frame,
            'score': score
        }
    
    print(f"   Selected best frames for {len(persons)} persons")
    
    # Load frames and create crops
    crops = []
    for person in persons:
        person_id = person['person_id']
        best_frame_num = person_best_frames[person_id]['frame']
        
        # Find bbox in that frame
        frame_idx = np.where(person['frame_numbers'] == best_frame_num)[0][0]
        bbox = person['bboxes'][frame_idx]
        
        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_num)
        ret, frame = cap.read()
        
        if not ret:
            print(f"   ⚠️  Could not read frame {best_frame_num} for person {person_id}")
            continue
        
        # Calculate padded bbox
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        pad_x = width * (padding_percent / 100)
        pad_y = height * (padding_percent / 100)
        
        x1_pad = max(0, int(x1 - pad_x))
        y1_pad = max(0, int(y1 - pad_y))
        x2_pad = min(video_width, int(x2 + pad_x))
        y2_pad = min(video_height, int(y2 + pad_y))
        
        # Crop
        crop = frame[y1_pad:y2_pad, x1_pad:x2_pad].copy()
        
        # Draw bbox on crop (adjust coordinates)
        bbox_in_crop = [
            int(x1 - x1_pad),
            int(y1 - y1_pad),
            int(x2 - x1_pad),
            int(y2 - y1_pad)
        ]
        
        color = get_person_color(person_id, person_id_list)
        crop = draw_person_bbox(crop, bbox_in_crop, person_id, color, thickness=2)
        
        # Add person info label at top
        frame_count = len(person['frame_numbers'])
        label = f"Person {person_id} ({frame_count} frames)"
        
        # Add label bar at top
        label_height = 30
        crop_with_label = np.zeros((crop.shape[0] + label_height, crop.shape[1], 3), dtype=np.uint8)
        crop_with_label[:] = (40, 40, 40)  # Dark gray background
        crop_with_label[label_height:, :] = crop
        
        # Draw label text
        cv2.putText(crop_with_label, label, (5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        crops.append({
            'person_id': person_id,
            'image': crop_with_label,
            'frame_count': frame_count
        })
    
    cap.release()
    
    # Calculate grid dimensions
    num_crops = len(crops)
    cols = 5
    rows = math.ceil(num_crops / cols)
    
    # Find max dimensions for uniform cells
    max_h = max([c['image'].shape[0] for c in crops])
    max_w = max([c['image'].shape[1] for c in crops])
    
    # Scale to fit max_cell_size
    scale = min(max_cell_size[0] / max_w, max_cell_size[1] / max_h, 1.0)
    cell_w = int(max_w * scale)
    cell_h = int(max_h * scale)
    
    grid_width = cell_w * cols
    grid_height = cell_h * rows
    
    # Create blank grid
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Place crops in grid
    for idx, crop_data in enumerate(crops):
        row = idx // cols
        col = idx % cols
        
        # Resize crop to cell size
        crop_resized = cv2.resize(crop_data['image'], (cell_w, cell_h))
        
        # Place in grid
        y_start = row * cell_h
        y_end = y_start + cell_h
        x_start = col * cell_w
        x_end = x_start + cell_w
        
        grid[y_start:y_end, x_start:x_end] = crop_resized
    
    # Save grid
    cv2.imwrite(str(output_path), grid)
    print(f"   ✅ Saved: {output_path.name} ({grid_width}x{grid_height})")
    
    return person_best_frames


def main():
    parser = argparse.ArgumentParser(description='Stage 6b: Create selection grid images')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    stage_config = config.get('stage6b_create_selection_grid', {})
    
    # Get paths
    video_path = Path(config['global']['video_dir'] + config['global']['video_file'])
    canonical_persons_file = Path(config['stage4b_group_canonical']['output']['canonical_persons_file'])
    
    # Output paths
    output_dir = Path(config['global']['outputs_dir']) / config['global']['current_video']
    fullframe_output = output_dir / 'top10_persons_fullframe_grid.png'
    cropped_output = output_dir / 'top10_persons_cropped_grid.png'
    info_output = output_dir / 'selection_grid_info.json'
    
    # Settings
    min_duration = stage_config.get('filters', {}).get('min_duration_seconds', 5)
    max_persons = stage_config.get('filters', {}).get('max_persons_shown', 10)
    target_frames = stage_config.get('full_frame_grid', {}).get('target_frame_count', 10)
    
    print(f"\n{'='*70}")
    print(f"🖼️  STAGE 6b: CREATE SELECTION GRIDS")
    print(f"{'='*70}\n")
    print(f"📂 Video: {video_path.name}")
    print(f"📊 Canonical Persons: {canonical_persons_file.name}")
    
    # Check files
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    if not canonical_persons_file.exists():
        print(f"❌ Canonical persons not found: {canonical_persons_file}")
        return
    
    # Get video FPS
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Load top persons
    persons = load_top_persons(canonical_persons_file, min_duration, video_fps, max_persons)
    
    if not persons:
        print(f"\n❌ No persons meet the criteria!")
        return
    
    # Select frames for full frame grid
    selected_frames = select_frames_greedy_coverage(persons, target_frames)
    
    # Create full frame grid
    fullframe_output.parent.mkdir(parents=True, exist_ok=True)
    grid_w, grid_h = create_fullframe_grid(video_path, persons, selected_frames, fullframe_output)
    
    # Create cropped grid
    person_best_frames = create_cropped_grid(video_path, persons, cropped_output)
    
    # Save info JSON
    info_data = {
        'fullframe_grid': {
            'frames_selected': [f[0] for f in selected_frames],
            'persons_per_frame': {
                str(frame_num): sorted(list(persons_in_frame)) 
                for frame_num, persons_in_frame in selected_frames
            },
            'dimensions': [grid_w, grid_h]
        },
        'cropped_grid': {
            'person_frames': {
                str(pid): {
                    'frame': int(data['frame']),
                    'score': float(data['score'])
                }
                for pid, data in person_best_frames.items()
            }
        },
        'persons': [
            {
                'person_id': int(p['person_id']),
                'rank': idx + 1,
                'frame_count': int(len(p['frame_numbers'])),
                'duration_seconds': round(len(p['frame_numbers']) / video_fps, 2)
            }
            for idx, p in enumerate(persons)
        ]
    }
    
    with open(info_output, 'w') as f:
        json.dump(info_data, f, indent=2)
    
    print(f"\n   ✅ Saved info: {info_output.name}")
    
    print(f"\n{'='*70}")
    print(f"✅ SELECTION GRIDS COMPLETE!")
    print(f"{'='*70}\n")
    print(f"📦 Outputs:")
    print(f"   1. Full Frame Grid: {fullframe_output.name}")
    print(f"   2. Cropped Grid: {cropped_output.name}")
    print(f"   3. Info JSON: {info_output.name}")
    print(f"\n💡 Review both grids and select your preferred person!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
