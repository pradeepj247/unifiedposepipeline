#!/usr/bin/env python3
"""
Two-panel visualization: Raw Tracklets vs Canonical Persons
Left panel: Tracklet IDs with bboxes
Right panel: Canonical person IDs with bboxes (showing grouping result)

This helps debug the merging logic by showing which tracklets merged into which persons.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict
import colorsys
import yaml
import re


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
    
    # Auto-extract current_video from video_file
    video_file = global_vars.get('video_file', '')
    if video_file:
        import os
        video_name = os.path.splitext(video_file)[0]
        global_vars['current_video'] = video_name
    
    # Second pass: resolve entire config using resolved global vars
    def resolve_string(s, vars_dict):
        if not isinstance(s, str):
            return s
        result = s
        for _ in range(max_iterations):
            new_result = re.sub(
                r'\$\{(\w+)\}',
                lambda m: str(vars_dict.get(m.group(1), m.group(0))),
                result
            )
            if new_result == result:
                break
            result = new_result
        return result
    
    def resolve_recursive(obj):
        if isinstance(obj, dict):
            return {k: resolve_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_recursive(v) for v in obj]
        elif isinstance(obj, str):
            return resolve_string(obj, global_vars)
        return obj
    
    result = resolve_recursive(config)
    result['global'] = global_vars
    return result


def load_npz_data(npz_path):
    """Load NPZ file and return data."""
    data = np.load(npz_path, allow_pickle=True)
    return data


def get_color_for_id(id_num, total_ids=200):
    """Generate consistent color for each ID using HSV color space."""
    hue = (id_num % total_ids) / total_ids
    saturation = 0.8
    value = 0.9
    
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    bgr = tuple(int(c * 255) for c in (rgb[2], rgb[1], rgb[0]))  # RGB to BGR
    return bgr


def extract_tracklets_for_frame(tracklets_data, frame_num):
    """Extract all tracklets visible in a given frame."""
    tracklets = tracklets_data['tracklets'].tolist()
    
    frame_tracklets = []
    for tracklet_id, tracklet in enumerate(tracklets):
        frame_numbers = tracklet['frame_numbers']
        
        # Check if this frame is in this tracklet
        if frame_num in frame_numbers:
            frame_idx = np.where(frame_numbers == frame_num)[0][0]
            bbox = tracklet['bboxes'][frame_idx]
            confidence = tracklet['confidences'][frame_idx]
            
            frame_tracklets.append({
                'tracklet_id': tracklet_id,
                'bbox': bbox,  # [x1, y1, x2, y2]
                'confidence': confidence
            })
    
    return frame_tracklets


def extract_persons_for_frame(persons_data, frame_num):
    """Extract all canonical persons visible in a given frame."""
    persons = persons_data['persons'].tolist()
    
    frame_persons = []
    for person_id, person in enumerate(persons):
        frame_numbers = person['frame_numbers']
        
        # Check if this frame is in this person
        if frame_num in frame_numbers:
            frame_idx = np.where(frame_numbers == frame_num)[0][0]
            bbox = person['bboxes'][frame_idx]
            confidence = person['confidences'][frame_idx]
            
            frame_persons.append({
                'person_id': person_id,
                'bbox': bbox,  # [x1, y1, x2, y2]
                'confidence': confidence
            })
    
    return frame_persons


def draw_bboxes(image, detections, panel_name="Panel"):
    """Draw bboxes with IDs on image."""
    for det in detections:
        if 'tracklet_id' in det:
            id_num = det['tracklet_id']
            id_label = f"T{id_num}"
        else:
            id_num = det['person_id']
            id_label = f"P{id_num}"
        
        bbox = det['bbox']
        confidence = det['confidence']
        
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Get color for this ID
        color = get_color_for_id(id_num, total_ids=150)
        
        # Draw bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with confidence
        label = f"{id_label}:{confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x = x1
        text_y = max(y1 - 5, 20)
        
        # Draw background for text
        cv2.rectangle(image, 
                      (text_x, text_y - text_size[1] - 4),
                      (text_x + text_size[0] + 4, text_y + 4),
                      color, -1)
        
        # Draw text
        cv2.putText(image, label, (text_x + 2, text_y - 2),
                   font, font_scale, (255, 255, 255), thickness)
    
    return image


def main():
    parser = argparse.ArgumentParser(description="Two-panel tracklet vs canonical visualization")
    parser.add_argument('--config', required=True, help='Pipeline config file')
    parser.add_argument('--max-frames', type=int, default=600, 
                        help='Maximum frames to process (default: 600)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve paths
    config = resolve_path_variables(config)
    
    # Get file paths from config sections (like stage5b does)
    tracklets_path = Path(config['stage2_track']['output']['tracklets_file'])
    persons_path = Path(config['stage4b_group_canonical']['output']['canonical_persons_file'])
    
    # Get video path
    video_dir = config['global']['video_dir']
    video_file = config['global']['video_file']
    video_path = f"{video_dir}{video_file}" if video_dir.endswith('/') else f"{video_dir}/{video_file}"
    
    output_dir = Path(config['global']['outputs_dir'])
    
    output_video = Path(output_dir) / 'debug_tracklets_vs_canonical.mp4'
    
    print(f"Video: {video_path}")
    print(f"Tracklets: {tracklets_path}")
    print(f"Persons: {persons_path}")
    print(f"Output: {output_video}")
    print(f"Max frames: {args.max_frames}")
    
    # Verify files exist
    if not Path(tracklets_path).exists():
        print(f"ERROR: {tracklets_path} not found")
        return 1
    if not Path(persons_path).exists():
        print(f"ERROR: {persons_path} not found")
        return 1
    if not Path(video_path).exists():
        print(f"ERROR: {video_path} not found")
        return 1
    
    # Load data
    print("\nLoading NPZ files...")
    tracklets_data = load_npz_data(tracklets_path)
    persons_data = load_npz_data(persons_path)
    
    # Also try to load grouping log to understand merging
    grouping_log_path = Path(config['stage4b_group_canonical']['output']['grouping_log_file'])
    grouping_info = {}
    if grouping_log_path.exists():
        import json
        with open(grouping_log_path, 'r') as f:
            grouping_log = json.load(f)
        print(f"\nGrouping Log: {len(grouping_log)} persons")
        for entry in grouping_log[:5]:
            grouping_info[entry['canonical_id']] = entry
            print(f"  Person {entry['canonical_id']}: "
                  f"merged from {entry['num_merged']} tracklets, "
                  f"tracklet_ids={entry['original_tracklet_ids']}")
    
    # Debug: Print data structure
    print(f"\nTracklets NPZ keys: {tracklets_data.files}")
    tracklets_list = tracklets_data['tracklets'].tolist()
    print(f"  Number of tracklets: {len(tracklets_list)}")
    if len(tracklets_list) > 0:
        print(f"  First tracklet keys: {tracklets_list[0].dtype.names if hasattr(tracklets_list[0], 'dtype') else tracklets_list[0].keys()}")
        print(f"  Sample tracklet 0: ID={tracklets_list[0]['tracklet_id']}, "
              f"frames={len(tracklets_list[0]['frame_numbers'])}, "
              f"frame_range=[{tracklets_list[0]['frame_numbers'][0]}, {tracklets_list[0]['frame_numbers'][-1]}]")
    
    print(f"\nPersons NPZ keys: {persons_data.files}")
    persons_list = persons_data['persons'].tolist()
    print(f"  Number of persons: {len(persons_list)}")
    if len(persons_list) > 0:
        print(f"  First person keys: {persons_list[0].dtype.names if hasattr(persons_list[0], 'dtype') else persons_list[0].keys()}")
        print(f"  Sample person 0: ID={persons_list[0]['person_id']}, "
              f"tracklets_merged={persons_list[0].get('num_tracklets_merged', '?')}, "
              f"frames={len(persons_list[0]['frame_numbers'])}, "
              f"frame_range=[{persons_list[0]['frame_numbers'][0]}, {persons_list[0]['frame_numbers'][-1]}]")
    
    # Open video
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_path}")
        return 1
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties:")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Resolution: {frame_width}×{frame_height}")
    
    # Calculate processing
    max_frames = min(args.max_frames, total_frames)
    print(f"  Processing: {max_frames} frames")
    
    # Setup output video writer
    output_width = 3840  # Two 1920-wide panels
    output_height = frame_height
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, 
                         (output_width, output_height))
    
    if not out.isOpened():
        print(f"ERROR: Cannot create output video writer")
        return 1
    
    print(f"\nOutput video: {output_width}×{output_height} @ {fps} FPS")
    print(f"Writing to: {output_video}\n")
    
    # Process frames
    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get detections for this frame
        tracklets_in_frame = extract_tracklets_for_frame(tracklets_data, frame_count)
        persons_in_frame = extract_persons_for_frame(persons_data, frame_count)
        
        # Debug output every 50 frames
        if frame_count % 50 == 0:
            print(f"Frame {frame_count}: {len(tracklets_in_frame)} tracklets, {len(persons_in_frame)} persons")
        
        # Create left panel (tracklets)
        left_panel = frame.copy()
        left_panel = draw_bboxes(left_panel, tracklets_in_frame, "Tracklets")
        
        # Create right panel (canonical persons)
        right_panel = frame.copy()
        right_panel = draw_bboxes(right_panel, persons_in_frame, "Persons")
        
        # Add panel titles
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(left_panel, f"Raw Tracklets (Frame {frame_count})", 
                   (20, 40), font, 1.2, (0, 255, 0), 2)
        cv2.putText(right_panel, f"Canonical Persons (Frame {frame_count})", 
                   (20, 40), font, 1.2, (0, 255, 0), 2)
        
        # Add detection counts
        cv2.putText(left_panel, f"Count: {len(tracklets_in_frame)}", 
                   (20, 80), font, 0.8, (200, 200, 200), 1)
        cv2.putText(right_panel, f"Count: {len(persons_in_frame)}", 
                   (20, 80), font, 0.8, (200, 200, 200), 1)
        
        # Combine panels side-by-side
        combined = np.hstack([left_panel, right_panel])
        
        # Write frame
        out.write(combined)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{max_frames} frames...")
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"\n✅ Complete!")
    print(f"Processed: {frame_count} frames")
    print(f"Output saved: {output_video}")
    
    return 0


if __name__ == '__main__':
    exit(main())
