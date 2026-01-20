"""
Visualize canonical persons from Stage 3c (canonical_persons.npz)
Creates a video showing all merged persons with their IDs
"""

import numpy as np
import cv2
import yaml
import os
from pathlib import Path
from tqdm import tqdm

# Rainbow colors for persons
def get_rainbow_colors(n):
    """Generate n distinct colors using HSV rainbow"""
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        colors.append(tuple(map(int, rgb[0, 0])))
    return colors

def resolve_paths(config, max_iterations=5):
    """Resolve ${variable} references in config"""
    global_vars = config.get('global', {})
    
    # Add current_video if not present
    if 'current_video' not in global_vars:
        video_file = global_vars.get('video_file', 'kohli_nets.mp4')
        global_vars['current_video'] = Path(video_file).stem
    
    for _ in range(max_iterations):
        updated = False
        for key, value in global_vars.items():
            if isinstance(value, str) and '${' in value:
                for var_name, var_value in global_vars.items():
                    if var_name != key and isinstance(var_value, str) and '${' not in var_value:
                        placeholder = '${' + var_name + '}'
                        if placeholder in value:
                            global_vars[key] = value.replace(placeholder, var_value)
                            updated = True
                            break
        if not updated:
            break
    
    config['global'] = global_vars
    return config

def main():
    # Load config
    config_path = Path(__file__).parent / 'configs' / 'pipeline_config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    config = resolve_paths(config)
    
    # Get paths
    video_dir = config['global'].get('video_dir', '')
    video_file = config['global']['video_file']
    video_path = str(Path(video_dir) / video_file) if video_dir else video_file
    
    outputs_dir = config['global']['outputs_dir']
    current_video = config['global']['current_video']
    output_dir = Path(outputs_dir) / current_video
    
    persons_file = output_dir / 'canonical_persons.npz'
    output_video = output_dir / 'stage3c_persons_visualization.mp4'
    
    print(f"📹 Loading video: {video_path}")
    print(f"📊 Loading persons: {persons_file}")
    
    # Load persons
    data = np.load(persons_file, allow_pickle=True)
    persons = data['persons']
    
    print(f"\n👥 Found {len(persons)} canonical persons")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    
    # Generate colors for persons
    colors = get_rainbow_colors(len(persons))
    
    # Build frame-to-persons mapping
    print("\n🔄 Building frame index...")
    frame_persons = {}
    for i, person in enumerate(persons):
        person_id = person['person_id']
        frame_nums = person['frame_numbers']
        bboxes = person['bboxes']
        tracklet_ids = person['tracklet_ids']
        
        for frame_num, bbox in zip(frame_nums, bboxes):
            if frame_num not in frame_persons:
                frame_persons[frame_num] = []
            frame_persons[frame_num].append({
                'person_id': person_id,
                'bbox': bbox,
                'color': colors[i % len(colors)],
                'tracklet_ids': tracklet_ids
            })
    
    # Process video
    print(f"\n🎬 Creating visualization video...")
    pbar = tqdm(total=total_frames, desc="Processing frames")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw persons for this frame
        if frame_idx in frame_persons:
            for person_info in frame_persons[frame_idx]:
                person_id = person_info['person_id']
                bbox = person_info['bbox']
                color = person_info['color']
                tracklet_ids = person_info['tracklet_ids']
                
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw bounding box (thicker for persons)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw person ID
                label = f"P{person_id}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0] + 10, y1), color, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Draw tracklet composition (smaller text)
                tracklet_text = f"T{tracklet_ids}"
                cv2.putText(frame, tracklet_text, (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw frame number
        frame_text = f"Frame: {frame_idx}"
        cv2.putText(frame, frame_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Draw person count
        count_text = f"Persons: {len(frame_persons.get(frame_idx, []))}"
        cv2.putText(frame, count_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        out.write(frame)
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f"\n✅ Visualization saved to: {output_video}")
    print(f"👥 Total persons visualized: {len(persons)}")

if __name__ == '__main__':
    main()
