"""
Extract crops from video using selected_person.npz bboxes

Reads selected_person.npz and extracts person crops from video,
saving them as a PKL file for fast downstream pose estimation.

Usage:
    python extract_crops.py --selected_person demo_data/outputs/kohli_nets/selected_person.npz \
                            --video demo_data/outputs/kohli_nets/canonical_video.mp4 \
                            --output demo_data/outputs/kohli_nets/selected_crops.pkl
"""

import argparse
import pickle
from pathlib import Path
import time
import cv2
import numpy as np


def extract_crops(selected_person_path, video_path, output_path, expand_bbox=0.1):
    """
    Extract person crops from video using bboxes from selected_person.npz
    
    Args:
        selected_person_path: Path to selected_person.npz
        video_path: Path to video file
        output_path: Path to save crops PKL
        expand_bbox: Percentage to expand bbox (e.g., 0.1 = 10% expansion)
    
    Returns:
        output_path: Path to saved PKL file
        total_time: Processing time in seconds
    """
    print("\n" + "=" * 70)
    print("🎬 EXTRACTING PERSON CROPS FROM VIDEO")
    print("=" * 70)
    print(f"   Bbox expansion: {expand_bbox * 100:.0f}%")
    
    # Load selected person data
    print(f"📦 Loading selected person data...")
    data = np.load(selected_person_path)
    frame_numbers = data['frame_numbers']
    bboxes = data['bboxes']
    
    print(f"   Frames: {len(frame_numbers)}")
    print(f"   Person ID: {data.get('person_id', 'N/A')}")
    
    # Open video
    print(f"\n📹 Opening video: {Path(video_path).name}")
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
    
    # Extract crops
    print(f"\n🔪 Extracting crops...")
    crops = []
    valid_crops = 0
    t_start = time.time()
    
    for idx, (frame_num, bbox) in enumerate(zip(frame_numbers, bboxes)):
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print(f"\n   ⚠️  Warning: Could not read frame {frame_num}")
            crops.append(None)
            continue
        
        # Extract crop with expansion
        x1, y1, x2, y2 = map(int, bbox)
        
        # Validate bbox
        if x2 <= x1 or y2 <= y1:
            # Invalid bbox (likely no detection)
            crops.append(None)
        else:
            # Expand bbox
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            expand_w = int(bbox_w * expand_bbox)
            expand_h = int(bbox_h * expand_bbox)
            
            x1_exp = x1 - expand_w
            y1_exp = y1 - expand_h
            x2_exp = x2 + expand_w
            y2_exp = y2 + expand_h
            
            # Clip to frame bounds
            x1_exp = max(0, x1_exp)
            y1_exp = max(0, y1_exp)
            x2_exp = min(width, x2_exp)
            y2_exp = min(height, y2_exp)
            
            # Extract crop
            crop = frame[y1_exp:y2_exp, x1_exp:x2_exp]
            crops.append(crop)
            valid_crops += 1
        
        # Progress
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t_start
            fps_proc = (idx + 1) / elapsed
            print(f"   Processed {idx + 1}/{len(frame_numbers)} frames ({fps_proc:.1f} FPS)", end='\r')
    
    cap.release()
    t_end = time.time()
    total_time = t_end - t_start
    
    print(f"\n   ✅ Extraction complete!")
    print(f"   Valid crops: {valid_crops}/{len(frame_numbers)}")
    print(f"   Time: {total_time:.2f}s ({total_time / 60:.2f} min)")
    print(f"   Processing FPS: {len(frame_numbers) / total_time:.1f}")
    print(f"   Time per frame: {(total_time / len(frame_numbers)) * 1000:.1f} ms")
    
    # Calculate storage size
    crop_sizes = [crop.nbytes for crop in crops if crop is not None]
    avg_crop_size = np.mean(crop_sizes) if crop_sizes else 0
    total_size_mb = sum(crop_sizes) / (1024 * 1024)
    
    # Calculate average crop dimensions
    crop_shapes = [crop.shape for crop in crops if crop is not None]
    avg_height = np.mean([s[0] for s in crop_shapes]) if crop_shapes else 0
    avg_width = np.mean([s[1] for s in crop_shapes]) if crop_shapes else 0
    
    print(f"\n💾 Storage info:")
    print(f"   Avg crop size: {avg_crop_size / 1024:.1f} KB")
    print(f"   Avg dimensions: {avg_width:.0f}×{avg_height:.0f} px")
    print(f"   Total size: {total_size_mb:.1f} MB")
    
    # Save to PKL
    print(f"\n📦 Saving crops to PKL...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    t_save_start = time.time()
    
    # Save with metadata
    crops_data = {
        'crops': crops,
        'frame_numbers': frame_numbers,
        'bboxes': bboxes,
        'person_id': data.get('person_id', None),
        'expand_bbox': expand_bbox,
        'video_metadata': {
            'width': width,
            'height': height,
            'fps': fps,
            'total_frames': total_frames
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(crops_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    t_save_end = time.time()
    save_time = t_save_end - t_save_start
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   ✅ Saved: {output_path}")
    print(f"   File size: {file_size_mb:.1f} MB")
    print(f"   Save time: {save_time:.2f}s")
    
    print("\n" + "=" * 70)
    print("🎉 CROP EXTRACTION COMPLETE!")
    print("=" * 70)
    print(f"   Output: {output_path}")
    print(f"   Crops: {valid_crops}/{len(frame_numbers)} ({valid_crops / len(frame_numbers) * 100:.1f}%)")
    print(f"   File size: {file_size_mb:.1f} MB")
    print(f"   ")
    print(f"   ⏱️  TIMING BREAKDOWN:")
    print(f"   - Extraction: {total_time:.2f}s ({len(frame_numbers) / total_time:.1f} FPS)")
    print(f"   - Save PKL: {save_time:.2f}s")
    print(f"   - Total: {(total_time + save_time):.2f}s")
    print(f"   ")
    print(f"   📊 SPEEDUP ESTIMATE:")
    print(f"   - Video decode/seek: ~{(total_time / len(frame_numbers)) * 1000:.1f}ms per frame")
    print(f"   - PKL load estimate: ~2-5ms per frame")
    print(f"   - Expected speedup: 5-10x faster for pose detection")
    print(f"\n   Next: Use these crops for fast pose detection")
    print(f"   Example: python run_posedet_fast.py --crops {output_path}")
    print("=" * 70 + "\n")
    
    return output_path, total_time + save_time


def main():
    parser = argparse.ArgumentParser(description='Extract person crops from video')
    parser.add_argument('--selected_person', type=str, required=True,
                        help='Path to selected_person.npz')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to video file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save crops PKL (default: same dir as selected_person)')
    parser.add_argument('--expand_bbox', type=float, default=0.15,
                        help='Percentage to expand bbox (default: 0.15 = 15%% expansion)')
    
    args = parser.parse_args()
    
    # Default output path
    if args.output is None:
        selected_path = Path(args.selected_person)
        args.output = selected_path.parent / "selected_crops.pkl"
    
    # Extract crops
    extract_crops(args.selected_person, args.video, args.output, args.expand_bbox)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
