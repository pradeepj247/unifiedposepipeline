"""
Visualize raw ByteTrack tracklets from Stage 2 (tracklets_raw.npz)
Creates a video showing all tracklets with their IDs
"""

import numpy as np
import cv2
import yaml
import os
from pathlib import Path
from tqdm import tqdm

# Rainbow colors for tracklets
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
    video_path = config['global']['video_file']
    output_dir = Path(config['global']['output_dir'])
    tracklets_file = output_dir / 'tracklets_raw.npz'
    output_video = output_dir / 'stage2_tracklets_visualization.mp4'
    
    print(f"📹 Loading video: {video_path}")
    print(f"📊 Loading tracklets: {tracklets_file}")
    
    # Load tracklets
    data = np.load(tracklets_file, allow_pickle=True)
    tracklets = data['tracklets']
    
    print(f"\n📦 Found {len(tracklets)} tracklets")
    
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
    
    # Generate colors for tracklets
    colors = get_rainbow_colors(len(tracklets))
    
    # Build frame-to-tracklets mapping
    print("\n🔄 Building frame index...")
    frame_tracklets = {}
    for i, tracklet in enumerate(tracklets):
        tracklet_id = tracklet['tracklet_id']
        frame_nums = tracklet['frame_numbers']
        bboxes = tracklet['bboxes']
        
        for frame_num, bbox in zip(frame_nums, bboxes):
            if frame_num not in frame_tracklets:
                frame_tracklets[frame_num] = []
            frame_tracklets[frame_num].append({
                'tracklet_id': tracklet_id,
                'bbox': bbox,
                'color': colors[i % len(colors)]
            })
    
    # Process video
    print(f"\n🎬 Creating visualization video...")
    pbar = tqdm(total=total_frames, desc="Processing frames")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw tracklets for this frame
        if frame_idx in frame_tracklets:
            for track_info in frame_tracklets[frame_idx]:
                tracklet_id = track_info['tracklet_id']
                bbox = track_info['bbox']
                color = track_info['color']
                
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw tracklet ID
                label = f"T{tracklet_id}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw frame number
        frame_text = f"Frame: {frame_idx}"
        cv2.putText(frame, frame_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Draw tracklet count
        count_text = f"Tracklets: {len(frame_tracklets.get(frame_idx, []))}"
        cv2.putText(frame, count_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        out.write(frame)
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f"\n✅ Visualization saved to: {output_video}")
    print(f"📊 Total tracklets visualized: {len(tracklets)}")

if __name__ == '__main__':
    main()
