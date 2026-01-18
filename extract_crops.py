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
    import sys
    sys.stdout.flush()
    
    # Load selected person data
    print(f"📦 Loading selected person data...")
    import sys
    sys.stdout.flush()
    data = np.load(selected_person_path)
    frame_numbers = data['frame_numbers']
    bboxes = data['bboxes']
    
    print(f"   Frames: {len(frame_numbers)}")
    print(f"   Person ID: {data.get('person_id', 'N/A')}")
    sys.stdout.flush()
    
    # Open video
    print(f"\n📹 Opening video: {Path(video_path).name}")
    import sys
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
    
    # Extract crops
    print(f"\n🔪 Extracting crops...")
    print(f"   Starting frame-by-frame extraction...")
    import sys
    sys.stdout.flush()
    
    crops = []
    valid_crops = 0
    t_start = time.time()
    
    # Create a set of frame numbers for fast lookup
    frame_set = set(frame_numbers)
    frame_to_idx = {frame_num: idx for idx, frame_num in enumerate(frame_numbers)}
    
    # Initialize crops list with None
    crops = [None] * len(frame_numbers)
    
    # Read all frames and extract crops for frames we need
    current_frame = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if this frame is in our selection
            if current_frame in frame_set:
                idx = frame_to_idx[current_frame]
                bbox = bboxes[idx]
                
                # Extract crop with expansion
                x1, y1, x2, y2 = map(int, bbox)
                
                # Validate bbox
                if x2 > x1 and y2 > y1:
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
                    try:
                        crop = frame[y1_exp:y2_exp, x1_exp:x2_exp]
                        if crop.size > 0:
                            crops[idx] = crop
                            valid_crops += 1
                    except Exception as e:
                        print(f"\n   ⚠️  Error extracting crop at frame {current_frame}: {e}")
                        sys.stdout.flush()
            
            current_frame += 1
            
            # Progress every 100 frames
            if current_frame % 100 == 0:
                elapsed = time.time() - t_start
                fps_proc = current_frame / elapsed
                print(f"\n   ✅ Progress: {current_frame}/{total_frames} frames | {fps_proc:.1f} FPS | {valid_crops} crops extracted")
                sys.stdout.flush()
    
    except KeyboardInterrupt:
        print(f"\n\n   ⚠️  Process interrupted by user at frame {current_frame}")
        print(f"   Extracted {valid_crops} crops before interruption")
        sys.stdout.flush()
        cap.release()
        return None, 0
    except Exception as e:
        print(f"\n\n   ❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        cap.release()
        return None, 0
    
    # Close video
    cap.release()
    t_end = time.time()
    total_time = t_end - t_start
    
    print(f"\n\n   ✅ Extraction complete!")
    print(f"   Video frames read: {current_frame}/{total_frames}")
    print(f"   Valid crops: {valid_crops}/{len(frame_numbers)}")
    print(f"   Time: {total_time:.2f}s ({total_time / 60:.2f} min)")
    print(f"   Processing FPS: {current_frame / total_time:.1f}")
    print(f"   Time per frame: {(total_time / current_frame) * 1000:.1f} ms")
    sys.stdout.flush()
    
    # Calculate storage size
    print(f"\n💾 Calculating storage info...")
    sys.stdout.flush()
    
    crop_sizes = [crop.nbytes for crop in crops if crop is not None]
    avg_crop_size = np.mean(crop_sizes) if crop_sizes else 0
    total_size_mb = sum(crop_sizes) / (1024 * 1024)
    
    # Calculate average crop dimensions
    crop_shapes = [crop.shape for crop in crops if crop is not None]
    avg_height = np.mean([s[0] for s in crop_shapes]) if crop_shapes else 0
    avg_width = np.mean([s[1] for s in crop_shapes]) if crop_shapes else 0
    
    print(f"   Avg crop size: {avg_crop_size / 1024:.1f} KB")
    print(f"   Avg dimensions: {avg_width:.0f}×{avg_height:.0f} px")
    print(f"   Total size: {total_size_mb:.1f} MB")
    sys.stdout.flush()
    
    # Save to PKL
    print(f"\n📦 Saving crops to PKL...")
    sys.stdout.flush()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"   Output path: {output_path}")
    sys.stdout.flush()
    
    t_save_start = time.time()
    
    try:
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
        
        print(f"   Writing to disk...")
        sys.stdout.flush()
        
        with open(output_path, 'wb') as f:
            pickle.dump(crops_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"   ✅ Write complete!")
        sys.stdout.flush()
        
    except Exception as e:
        print(f"\n   ❌ Error saving PKL: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return None, 0
    
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
    print(f"   - Extraction: {total_time:.2f}s ({current_frame / total_time:.1f} FPS)")
    print(f"   - Save PKL: {save_time:.2f}s")
    print(f"   - Total: {(total_time + save_time):.2f}s")
    print(f"   ")
    print(f"   📊 SPEEDUP ESTIMATE:")
    print(f"   - Video decode/seek: ~{(total_time / current_frame) * 1000:.1f}ms per frame")
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
