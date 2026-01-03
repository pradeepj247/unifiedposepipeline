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
    import yaml
    import sys
    from pathlib import Path
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve paths - add parent to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from det_track.utils.path_utils import resolve_config_paths
    config = resolve_config_paths(config)
    
    # Get file paths
    video_path = config['global']['video_file']
    output_dir = config['global']['output_dir']
    
    tracklets_path = Path(output_dir) / 'tracklets_raw.npz'
    persons_path = Path(output_dir) / 'canonical_persons.npz'
    
    output_video = Path(output_dir) / 'debug_tracklets_vs_canonical.mp4'
    
    print(f"Video: {video_path}")
    print(f"Tracklets: {tracklets_path}")
    print(f"Persons: {persons_path}")
    print(f"Output: {output_video}")
    print(f"Max frames: {args.max_frames}")
    
    # Load data
    print("\nLoading NPZ files...")
    tracklets_data = load_npz_data(tracklets_path)
    persons_data = load_npz_data(persons_path)
    
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
