"""
Extract crops with incremental saving - more robust for Colab

This version saves crops every N frames to avoid memory/timeout issues
"""

import argparse
import pickle
from pathlib import Path
import time
import cv2
import numpy as np
import sys


def extract_crops_incremental(selected_person_path, video_path, output_path, expand_bbox=0.15, save_interval=500):
    """Extract crops and save incrementally"""
    
    print("\n" + "=" * 70)
    print("🎬 EXTRACTING PERSON CROPS (INCREMENTAL MODE)")
    print("=" * 70)
    print(f"   Bbox expansion: {expand_bbox * 100:.0f}%")
    print(f"   Save interval: every {save_interval} frames")
    sys.stdout.flush()
    
    # Load selected person data
    print(f"\n📦 Loading selected person data...")
    sys.stdout.flush()
    data = np.load(selected_person_path)
    frame_numbers = data['frame_numbers']
    bboxes = data['bboxes']
    
    print(f"   Frames: {len(frame_numbers)}")
    print(f"   Person ID: {data.get('person_id', 'N/A')}")
    sys.stdout.flush()
    
    # Open video
    print(f"\n📹 Opening video: {Path(video_path).name}")
    sys.stdout.flush()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   Resolution: {width}×{height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total frames: {total_frames}")
    sys.stdout.flush()
    
    # Create frame lookup
    frame_set = set(frame_numbers)
    frame_to_idx = {frame_num: idx for idx, frame_num in enumerate(frame_numbers)}
    
    # Initialize
    crops = [None] * len(frame_numbers)
    valid_crops = 0
    current_frame = 0
    t_start = time.time()
    
    print(f"\n🔪 Extracting crops...")
    sys.stdout.flush()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract if in selection
            if current_frame in frame_set:
                idx = frame_to_idx[current_frame]
                bbox = bboxes[idx]
                x1, y1, x2, y2 = map(int, bbox)
                
                if x2 > x1 and y2 > y1:
                    # Expand bbox
                    bbox_w = x2 - x1
                    bbox_h = y2 - y1
                    expand_w = int(bbox_w * expand_bbox)
                    expand_h = int(bbox_h * expand_bbox)
                    
                    x1_exp = max(0, x1 - expand_w)
                    y1_exp = max(0, y1 - expand_h)
                    x2_exp = min(width, x2 + expand_w)
                    y2_exp = min(height, y2 + expand_h)
                    
                    crop = frame[y1_exp:y2_exp, x1_exp:x2_exp]
                    if crop.size > 0:
                        crops[idx] = crop
                        valid_crops += 1
            
            current_frame += 1
            
            # Progress
            if current_frame % 100 == 0:
                elapsed = time.time() - t_start
                fps_proc = current_frame / elapsed
                print(f"   ✅ {current_frame}/{total_frames} frames | {fps_proc:.1f} FPS | {valid_crops} crops")
                sys.stdout.flush()
            
            # Incremental save
            if current_frame % save_interval == 0:
                print(f"\n   💾 Saving checkpoint at frame {current_frame}...")
                sys.stdout.flush()
                checkpoint_path = Path(str(output_path).replace('.pkl', f'_checkpoint_{current_frame}.pkl'))
                checkpoint_data = {
                    'crops': crops[:],
                    'frame_numbers': frame_numbers,
                    'bboxes': bboxes,
                    'person_id': data.get('person_id', None),
                    'expand_bbox': expand_bbox,
                    'checkpoint_frame': current_frame,
                    'video_metadata': {'width': width, 'height': height, 'fps': fps, 'total_frames': total_frames}
                }
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"   ✅ Checkpoint saved: {checkpoint_path.name}")
                sys.stdout.flush()
        
        cap.release()
        print(f"\n\n   📹 Video released")
        sys.stdout.flush()
        
        t_end = time.time()
        total_time = t_end - t_start
        
        print(f"   ⏱️  Time calculated: {total_time:.2f}s")
        sys.stdout.flush()
        
        print(f"\n\n   ✅ Extraction complete!")
        print(f"   Frames: {current_frame}, Crops: {valid_crops}")
        print(f"   Time: {total_time:.2f}s ({current_frame / total_time:.1f} FPS)")
        sys.stdout.flush()
        
        # Final save
        print(f"\n📦 Starting final save...")
        sys.stdout.flush()
        
        output_path = Path(output_path)
        print(f"   Creating directory: {output_path.parent}")
        sys.stdout.flush()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"   Building crops_data dict...")
        sys.stdout.flush()
        
        crops_data = {
            'crops': crops,
            'frame_numbers': frame_numbers,
            'bboxes': bboxes,
            'person_id': data.get('person_id', None),
            'expand_bbox': expand_bbox,
            'video_metadata': {'width': width, 'height': height, 'fps': fps, 'total_frames': total_frames}
        }
        
        print(f"   Opening file for write: {output_path}")
        sys.stdout.flush()
        
        with open(output_path, 'wb') as f:
            print(f"   Writing pickle...")
            sys.stdout.flush()
            pickle.dump(crops_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"   Pickle written!")
            sys.stdout.flush()
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ Saved: {output_path}")
        print(f"   Size: {file_size_mb:.1f} MB")
        print(f"\n   🎉 COMPLETE!")
        sys.stdout.flush()
        
        return output_path, total_time
        
    except Exception as e:
        print(f"\n   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        cap.release()
        return None, 0


def main():
    parser = argparse.ArgumentParser(description='Extract crops incrementally')
    parser.add_argument('--selected_person', type=str, required=True)
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--expand_bbox', type=float, default=0.15)
    parser.add_argument('--save_interval', type=int, default=500, help='Save checkpoint every N frames')
    
    args = parser.parse_args()
    
    if args.output is None:
        selected_path = Path(args.selected_person)
        args.output = selected_path.parent / "selected_crops.pkl"
    
    extract_crops_incremental(args.selected_person, args.video, args.output, args.expand_bbox, args.save_interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
