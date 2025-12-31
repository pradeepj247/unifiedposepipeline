#!/usr/bin/env python3
"""
Create Visualization Video - Top 5 Canonical Persons

Draws bounding boxes for the top 5 persons (by duration) on the video.
Each person gets a unique color and is labeled with their person ID.

Usage:
    python create_visualization_video.py --output-dir /path/to/outputs/video_name
"""

import argparse
import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm


# Color palette for top 5 persons (BGR format for OpenCV)
COLORS = [
    (0, 255, 0),      # Green - Person 1
    (255, 0, 0),      # Blue - Person 2
    (0, 0, 255),      # Red - Person 3
    (255, 255, 0),    # Cyan - Person 4
    (255, 0, 255),    # Magenta - Person 5
]


def load_top_persons(canonical_persons_file, top_n=5):
    """Load top N persons by duration"""
    data = np.load(canonical_persons_file, allow_pickle=True)
    persons = data['persons']
    
    # Sort by duration (end - start)
    persons_with_duration = []
    for p in persons:
        frames = p['frame_numbers']
        duration = frames[-1] - frames[0] + 1
        persons_with_duration.append((p, duration))
    
    # Sort descending by duration
    persons_with_duration.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N
    top_persons = persons_with_duration[:top_n]
    
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
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Video: {width}x{height} @ {fps:.2f} fps")
    print(f"   Total frames: {total_frames}")
    print(f"   Drawing top {len(persons_dict)} persons\n")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Process frames
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Creating video")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get persons visible at this frame
        visible_persons = get_person_at_frame(persons_dict, frame_idx)
        
        # Draw each person
        for person_info in visible_persons:
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
    parser = argparse.ArgumentParser(description='Create visualization video with top persons')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory containing pipeline outputs')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Number of top persons to visualize (default: 5)')
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
    
    # File paths
    canonical_persons_file = output_dir / 'canonical_persons.npz'
    output_video_path = output_dir / f'{video_name}_top{args.top_n}_visualization.mp4'
    
    # Check files exist
    if not canonical_persons_file.exists():
        print(f"❌ Canonical persons file not found: {canonical_persons_file}")
        return
    
    print(f"📂 Loading data from: {output_dir}")
    print(f"📹 Video: {video_path}")
    
    # Load top persons
    persons_dict = load_top_persons(canonical_persons_file, top_n=args.top_n)
    
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
