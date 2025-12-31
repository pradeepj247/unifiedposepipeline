#!/usr/bin/env python3
"""
Create Visualization Video - Canonical Persons with >=5s Appearance

Draws bounding boxes for all persons who appear for at least 5 seconds.
Each person gets a unique color and is labeled with their person ID.
Output video is at 90% fps and downscaled to 720p for faster processing.

Usage:
    python create_visualization_video.py --output-dir /path/to/outputs/video_name
    python create_visualization_video.py --output-dir /path/to/outputs/video_name --min-seconds 3.0
"""

import argparse
import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm


# Color palette for persons (BGR format for OpenCV)
# Extended palette to support more than 5 persons
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


def load_top_persons(canonical_persons_file, min_duration_seconds=5.0, video_fps=30.0):
    """Load persons who appear for at least min_duration_seconds
    
    Args:
        canonical_persons_file: Path to canonical_persons.npz
        min_duration_seconds: Minimum appearance duration in seconds (default: 5.0)
        video_fps: Original video frame rate (used to compute frame threshold)
    """
    data = np.load(canonical_persons_file, allow_pickle=True)
    persons = data['persons']
    
    # Calculate minimum frame threshold
    min_frames = int(min_duration_seconds * video_fps)
    print(f"\n📏 Filtering persons: min {min_duration_seconds}s = {min_frames} frames @ {video_fps:.2f} fps")
    
    # Filter persons by total frame count (not temporal duration)
    persons_with_duration = []
    for p in persons:
        frames = p['frame_numbers']
        frame_count = len(frames)  # Number of frames person appears in
        if frame_count >= min_frames:
            persons_with_duration.append((p, frame_count))
    
    # Sort descending by frame count
    persons_with_duration.sort(key=lambda x: x[1], reverse=True)
    
    print(f"   Found {len(persons_with_duration)} persons with >={min_frames} frames")
    
    # Use all filtered persons (no top_n limit)
    top_persons = persons_with_duration
    
    # Create person_id -> data mapping
    persons_dict = {}
    for idx, (person, duration) in enumerate(top_persons):
        person_id = person['person_id']
        persons_dict[person_id] = {
            'rank': idx + 1,
            'color': COLORS[idx % len(COLORS)],
            'frame_numbers': person['frame_numbers'],
            'bboxes': person['bboxes'],
            'confidences': person['confidences'],
            'tracklet_ids': person['original_tracklet_ids'],
            'duration': duration
        }
    
    return persons_dict


def get_person_at_frame(persons_dict, frame_idx):
    """Get all persons visible at this frame"""
    visible_persons = []
    
    for person_id, person_data in persons_dict.items():
        # Find if this frame exists in person's frame_numbers
        frame_mask = person_data['frame_numbers'] == frame_idx
        if np.any(frame_mask):
            # Get bbox at this frame
            bbox_idx = np.where(frame_mask)[0][0]
            bbox = person_data['bboxes'][bbox_idx]
            conf = person_data['confidences'][bbox_idx]
            
            visible_persons.append({
                'person_id': person_id,
                'rank': person_data['rank'],
                'bbox': bbox,
                'confidence': conf,
                'color': person_data['color']
            })
    
    return visible_persons


def draw_bbox(frame, person_info):
    """Draw bounding box and label for a person"""
    bbox = person_info['bbox']
    x1, y1, x2, y2 = map(int, bbox)
    color = person_info['color']
    person_id = person_info['person_id']
    rank = person_info['rank']
    conf = person_info['confidence']
    
    # Draw box
    thickness = 3
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label background
    label = f"P{person_id} (#{rank})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    
    (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    
    # Position label above box
    label_y1 = max(y1 - label_h - 10, 0)
    label_y2 = label_y1 + label_h + 10
    label_x1 = x1
    label_x2 = x1 + label_w + 10
    
    # Draw label background
    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, -1)
    
    # Draw label text
    cv2.putText(frame, label, (label_x1 + 5, label_y2 - 5), 
                font, font_scale, (255, 255, 255), font_thickness)
    
    return frame


def create_visualization_video(video_path, persons_dict, output_path):
    """Create visualization video with bboxes"""
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output at 90% of original fps for smoother playback
    output_fps = orig_fps * 0.9
    
    # Downscale to 720p max width (or 640p if already small)
    max_width = 720 if orig_width > 800 else 640
    if orig_width > max_width:
        scale_factor = max_width / orig_width
        output_width = max_width
        output_height = int(orig_height * scale_factor)
    else:
        output_width = orig_width
        output_height = orig_height
    
    print(f"\n📹 Video Input: {orig_width}x{orig_height} @ {orig_fps:.2f} fps")
    print(f"   Video Output: {output_width}x{output_height} @ {output_fps:.2f} fps")
    print(f"   Total frames: {total_frames}")
    print(f"   Drawing {len(persons_dict)} persons\n")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (output_width, output_height))
    
    # Process frames
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Creating video")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame if needed
        if output_width != orig_width or output_height != orig_height:
            frame = cv2.resize(frame, (output_width, output_height))
            scale_factor = output_width / orig_width
        else:
            scale_factor = 1.0
        
        # Get persons visible at this frame
        visible_persons = get_person_at_frame(persons_dict, frame_idx)
        
        # Draw each person (scale bboxes if needed)
        for person_info in visible_persons:
            if scale_factor != 1.0:
                # Scale bbox coordinates
                bbox = person_info['bbox']
                scaled_bbox = bbox * scale_factor
                person_info = person_info.copy()
                person_info['bbox'] = scaled_bbox
            frame = draw_bbox(frame, person_info)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f"\n✅ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Create visualization video with persons appearing >=5 seconds')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory containing pipeline outputs')
    parser.add_argument('--min-seconds', type=float, default=5.0,
                        help='Minimum appearance duration in seconds (default: 5.0)')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Find video file (from config)
    # For now, assume it's in ../../demo_data/videos/{video_name}.mp4
    video_name = output_dir.name
    video_path = output_dir.parent.parent / 'videos' / f'{video_name}.mp4'
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        print(f"   Please specify video path manually")
        return
    
    # Get video fps for frame threshold calculation
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # File paths
    canonical_persons_file = output_dir / 'canonical_persons.npz'
    output_video_path = output_dir / f'{video_name}_5sec_visualization.mp4'
    
    # Check files exist
    if not canonical_persons_file.exists():
        print(f"❌ Canonical persons file not found: {canonical_persons_file}")
        return
    
    print(f"📂 Loading data from: {output_dir}")
    print(f"📹 Video: {video_path}")
    
    # Load persons with min duration filter
    persons_dict = load_top_persons(canonical_persons_file, 
                                     min_duration_seconds=args.min_seconds,
                                     video_fps=video_fps)
    
    print(f"\n📊 Top {len(persons_dict)} Persons:")
    for person_id, data in sorted(persons_dict.items(), key=lambda x: x[1]['rank']):
        print(f"  Rank {data['rank']}: Person {person_id} "
              f"(Tracklets: {data['tracklet_ids']}, "
              f"Duration: {data['duration']} frames)")
    
    # Create visualization
    print(f"\n🎨 Creating visualization video...")
    create_visualization_video(video_path, persons_dict, output_video_path)
    
    print(f"\n{'='*70}")
    print(f"✅ Visualization complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
