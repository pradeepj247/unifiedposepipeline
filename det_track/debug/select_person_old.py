#!/usr/bin/env python3
"""
Person Track Selection Script

Extracts a single person's track from multi-person tracking data.

Usage:
    python select_person.py \
        --input demo_data/outputs/raw_detections.npz \
        --output demo_data/outputs/selections.npz \
        --person-id 5

Input Format (raw_detections.npz):
    - frame_numbers: (N,) array of frame indices (repeated per person)
    - bboxes: (N, 4) array of [x1, y1, x2, y2]
    - track_ids: (N,) array of person IDs
    - scores: (N,) array of confidence scores

Output Format (selections.npz):
    - frame_numbers: (M,) array of frames where person appears
    - bboxes: (M, 4) array of person's bboxes [x1, y1, x2, y2]
    
    where M = number of frames person appears in (sparse format)
"""

import argparse
import numpy as np
from pathlib import Path
import sys


def load_raw_detections(input_path):
    """
    Load raw multi-person tracking data
    
    Args:
        input_path: Path to raw_detections.npz
        
    Returns:
        Dictionary with frame_numbers, bboxes, track_ids, scores
    """
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    data = np.load(input_path)
    
    required_keys = ['frame_numbers', 'bboxes', 'track_ids', 'scores']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in {input_path}")
    
    return {
        'frame_numbers': data['frame_numbers'],
        'bboxes': data['bboxes'],
        'track_ids': data['track_ids'],
        'scores': data['scores']
    }


def select_person_track(raw_data, person_id):
    """
    Extract single person's track from multi-person data
    
    Args:
        raw_data: Dictionary with frame_numbers, bboxes, track_ids, scores
        person_id: Person ID to extract
        
    Returns:
        Dictionary with frame_numbers, bboxes for selected person
    """
    # Filter by person ID
    mask = raw_data['track_ids'] == person_id
    
    if not np.any(mask):
        raise ValueError(f"Person ID {person_id} not found in tracking data")
    
    # Extract person's data
    person_frames = raw_data['frame_numbers'][mask]
    person_bboxes = raw_data['bboxes'][mask]
    person_scores = raw_data['scores'][mask]
    
    # Handle duplicate frames (same person ID appears twice in one frame - rare but possible)
    # Strategy: Keep largest bbox per frame
    unique_frames = np.unique(person_frames)
    final_frames = []
    final_bboxes = []
    
    for frame in unique_frames:
        frame_mask = person_frames == frame
        frame_bboxes = person_bboxes[frame_mask]
        
        if len(frame_bboxes) == 1:
            # Single bbox - use it
            final_frames.append(frame)
            final_bboxes.append(frame_bboxes[0])
        else:
            # Multiple bboxes for same person in same frame
            # Select largest bbox (by area)
            areas = (frame_bboxes[:, 2] - frame_bboxes[:, 0]) * (frame_bboxes[:, 3] - frame_bboxes[:, 1])
            largest_idx = np.argmax(areas)
            final_frames.append(frame)
            final_bboxes.append(frame_bboxes[largest_idx])
    
    return {
        'frame_numbers': np.array(final_frames, dtype=np.int64),
        'bboxes': np.array(final_bboxes, dtype=np.int64)
    }


def save_selections(selections_data, output_path):
    """
    Save person selections to NPZ file
    
    Args:
        selections_data: Dictionary with frame_numbers, bboxes
        output_path: Path to output selections.npz
    """
    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save NPZ (compressed, matches standard detection format)
    np.savez_compressed(
        output_path,
        frame_numbers=selections_data['frame_numbers'],
        bboxes=selections_data['bboxes']
    )


def print_selection_stats(raw_data, person_id, selections_data):
    """
    Print selection statistics
    
    Args:
        raw_data: Original multi-person data
        person_id: Selected person ID
        selections_data: Extracted person data
    """
    total_frames = int(np.max(raw_data['frame_numbers'])) + 1
    selected_frames = len(selections_data['frame_numbers'])
    
    print(f"\n{'='*70}")
    print(f"Person Track Selection")
    print(f"{'='*70}")
    print(f"\nSelected Person ID: {person_id}")
    print(f"Frames found: {selected_frames} out of {total_frames} total frames ({selected_frames/total_frames*100:.1f}%)")
    print(f"Frame range: {int(selections_data['frame_numbers'][0])} - {int(selections_data['frame_numbers'][-1])}")
    print(f"Bbox count: {len(selections_data['bboxes'])}")
    
    # Check for frame gaps (person disappeared temporarily)
    frame_diffs = np.diff(selections_data['frame_numbers'])
    max_gap = int(np.max(frame_diffs)) if len(frame_diffs) > 0 else 0
    num_gaps = int(np.sum(frame_diffs > 1))
    
    if num_gaps > 0:
        print(f"\n⚠️  Tracking Gaps:")
        print(f"  Number of gaps: {num_gaps}")
        print(f"  Largest gap: {max_gap} frames")
        print(f"  Note: Person disappeared/reappeared {num_gaps} times")
    else:
        print(f"\n✓ Continuous tracking (no gaps)")
    
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Extract single person track from multi-person tracking data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python select_person.py --input demo_data/outputs/raw_detections.npz \\
                          --output demo_data/outputs/selections.npz \\
                          --person-id 5
  
  # After selection, use with pose estimation pipeline
  python udp_video.py --config configs/udp_video.yaml \\
                      --detections demo_data/outputs/selections.npz
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to raw_detections.npz (multi-person tracking data)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output selections.npz (single-person track)'
    )
    
    parser.add_argument(
        '--person-id',
        type=int,
        required=True,
        help='Person ID to extract (from tracking visualization)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load raw detections
        print(f"\n[1/3] Loading raw detections from: {args.input}")
        raw_data = load_raw_detections(args.input)
        print(f"   ✅ Loaded {len(raw_data['frame_numbers'])} detections")
        print(f"   ✅ Unique person IDs: {sorted(np.unique(raw_data['track_ids']))}")
        
        # Select person track
        print(f"\n[2/3] Extracting person ID {args.person_id}...")
        selections_data = select_person_track(raw_data, args.person_id)
        print(f"   ✅ Extracted {len(selections_data['frame_numbers'])} frames")
        
        # Save selections
        print(f"\n[3/3] Saving selections to: {args.output}")
        save_selections(selections_data, args.output)
        print(f"   ✅ Saved selections.npz")
        
        # Print statistics
        print_selection_stats(raw_data, args.person_id, selections_data)
        
        print("✓ Person selection complete!")
        print(f"\nNext steps:")
        print(f"  1. Use selections.npz as input to pose estimation pipeline")
        print(f"  2. python udp_video.py --config configs/udp_video.yaml --detections {args.output}\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
