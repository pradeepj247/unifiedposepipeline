#!/usr/bin/env python3
"""
Person Selector - Extract selected persons' tracklet data
=========================================================

Extracts frame-by-frame bounding box data for selected persons from canonical_persons.npz
and saves as final_tracklet.npz.

Usage:
    python stage12_keyperson_selector.py --persons p3
    python stage12_keyperson_selector.py --persons p3,p4,p40
    python stage12_keyperson_selector.py --persons p3,p4,p40 --video kohli_nets.mp4
"""

import argparse
import numpy as np
from pathlib import Path
import json
import sys
from datetime import datetime


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract selected persons' tracklet data from canonical_persons.npz",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stage12_keyperson_selector.py --persons p3
  python stage12_keyperson_selector.py --persons p3,p4,p40
  python stage12_keyperson_selector.py --persons p3,p4 --video kohli_nets.mp4
        """
    )
    
    parser.add_argument(
        '--persons',
        type=str,
        required=True,
        help='Comma-separated person IDs (e.g., p3 or p3,p4,p40)'
    )
    
    parser.add_argument(
        '--video',
        type=str,
        default='kohli_nets.mp4',
        help='Video file name (default: kohli_nets.mp4)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: auto-detected from video name)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def parse_person_ids(persons_str):
    """
    Parse comma-separated person IDs.
    
    Examples:
        "p3" → [3]
        "p3,p4,p40" → [3, 4, 40]
        "3,4,40" → [3, 4, 40]
    """
    person_ids = []
    
    for person_str in persons_str.split(','):
        person_str = person_str.strip().lower()
        
        # Remove 'p' prefix if present
        if person_str.startswith('p'):
            person_str = person_str[1:]
        
        try:
            person_id = int(person_str)
            person_ids.append(person_id)
        except ValueError:
            raise ValueError(f"Invalid person ID: {person_str}")
    
    return sorted(set(person_ids))  # Remove duplicates and sort


def get_output_dir(video_file):
    """Auto-detect output directory based on video file name."""
    # Remove extension from video file
    video_base = Path(video_file).stem
    
    # Standard location: demo_data/outputs/{video_base}/
    repo_root = Path(__file__).parent.parent
    output_dir = repo_root / 'demo_data' / 'outputs' / video_base
    
    return output_dir


def load_canonical_persons(npz_path):
    """Load canonical_persons.npz and return persons list."""
    if not npz_path.exists():
        raise FileNotFoundError(f"File not found: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    persons = data['persons']
    
    return persons


def extract_selected_persons(persons, selected_ids):
    """
    Extract data for selected person IDs with metadata about start frames.
    
    Returns:
        Dict mapping person_id → {frame_numbers, bboxes, start_frame}
    """
    selected_data = {}
    
    for person in persons:
        person_id = person['person_id']
        
        if person_id in selected_ids:
            frame_numbers = person['frame_numbers'].copy()
            selected_data[person_id] = {
                'frame_numbers': frame_numbers,
                'bboxes': person['bboxes'].copy(),
                'start_frame': int(frame_numbers[0]),  # Track when this person starts
            }
    
    return selected_data


def create_detector_format_output(selected_data):
    """
    Convert selected persons data to detector format (same as run_detector.py output).
    
    Handles overlapping frames by using the person with the LATER start time.
    This ensures continuous data flow with one bbox per frame.
    
    Returns:
        Dict with keys:
        - frame_numbers: (N,) int64 array of all frame indices
        - bboxes: (N, 4) int64 array of bboxes [x1, y1, x2, y2]
        - person_mapping: (N,) int64 array of which person contributed each frame (for reference)
    """
    # Build a dict mapping frame_number → best_person_id
    frame_to_person = {}
    
    # Sort persons by start_frame (descending) - later starters get priority
    sorted_persons = sorted(
        selected_data.items(),
        key=lambda x: x[1]['start_frame'],
        reverse=True  # Higher start_frame first = gets priority in overlaps
    )
    
    # Assign frames to persons (later starters override earlier ones)
    for person_id, person_data in sorted_persons:
        for frame_num in person_data['frame_numbers']:
            frame_num_int = int(frame_num)
            frame_to_person[frame_num_int] = person_id
    
    # Now build contiguous arrays in frame order
    sorted_frames = sorted(frame_to_person.keys())
    
    frame_numbers = np.array(sorted_frames, dtype=np.int64)
    bboxes_list = []
    person_mapping = []
    
    for frame_num in sorted_frames:
        person_id = frame_to_person[frame_num]
        person_data = selected_data[person_id]
        
        # Find bbox for this frame in this person's data
        frame_idx = np.where(person_data['frame_numbers'] == frame_num)[0][0]
        bbox = person_data['bboxes'][frame_idx]
        
        bboxes_list.append(bbox)
        person_mapping.append(person_id)
    
    bboxes = np.array(bboxes_list, dtype=np.int64)
    person_mapping_array = np.array(person_mapping, dtype=np.int64)
    
    output_data = {
        'frame_numbers': frame_numbers,
        'bboxes': bboxes,
        'person_mapping': person_mapping_array,  # Which person contributed each frame
    }
    
    return output_data


def save_detector_format(output_data, output_path):
    """
    Save output in detector format (same as run_detector.py).
    
    Args:
        output_data: Dict with frame_numbers, bboxes, person_mapping
        output_path: Path to output NPZ file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save in compressed format (matches run_detector.py output)
    np.savez_compressed(
        output_path,
        frame_numbers=output_data['frame_numbers'],
        bboxes=output_data['bboxes'],
        person_mapping=output_data['person_mapping']  # Extra metadata
    )


def print_summary_detector_format(output_data, selected_data):
    """Print summary of extracted data in detector format."""
    print("\n" + "="*70)
    print("PERSON SELECTOR - DETECTOR FORMAT OUTPUT")
    print("="*70)
    
    frame_numbers = output_data['frame_numbers']
    bboxes = output_data['bboxes']
    person_mapping = output_data['person_mapping']
    
    print(f"\n✅ Selected persons:")
    for person_id in sorted(selected_data.keys()):
        data = selected_data[person_id]
        num_frames_orig = len(data['frame_numbers'])
        start_frame = data['start_frame']
        end_frame = int(data['frame_numbers'][-1])
        
        # Count how many frames this person actually contributed (after priority resolution)
        contributed = np.sum(person_mapping == person_id)
        
        print(f"   • P{person_id:2d}: Original {num_frames_orig:4d} frames | Range: {start_frame:5d} - {end_frame:5d} | Contributed: {contributed:4d} frames")
    
    print(f"\n📊 Detector Format Output:")
    print(f"   • Total frames in output: {len(frame_numbers)}")
    print(f"   • Bboxes shape: {bboxes.shape}")
    print(f"   • Frame range: {int(frame_numbers[0])} - {int(frame_numbers[-1])}")
    
    # Check for overlaps
    total_original_frames = sum(len(data['frame_numbers']) for data in selected_data.values())
    overlaps = total_original_frames - len(frame_numbers)
    if overlaps > 0:
        print(f"   • Overlapping frames (resolved by priority): {overlaps}")
    
    print(f"\n   This output is compatible with run_posedet.py")
    print(f"   Usage: python run_posedet.py --config configs/posedet.yaml")
    print(f"          (with detections_file pointing to final_tracklet.npz)")
    
    print("\n" + "="*70 + "\n")


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Parse person IDs
    try:
        selected_ids = parse_person_ids(args.persons)
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    print(f"\n🔍 Extracting persons: {', '.join([f'P{p}' for p in selected_ids])}")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = get_output_dir(args.video)
    
    if args.verbose:
        print(f"   Output directory: {output_dir}")
    
    # Load canonical persons
    canonical_persons_path = output_dir / 'canonical_persons.npz'
    if args.verbose:
        print(f"   Loading: {canonical_persons_path}")
    
    try:
        persons = load_canonical_persons(canonical_persons_path)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Extract selected persons
    selected_data = extract_selected_persons(persons, selected_ids)
    
    # Check if all requested persons were found
    found_ids = set(selected_data.keys())
    missing_ids = set(selected_ids) - found_ids
    
    if missing_ids:
        print(f"⚠️  Warning: Persons not found: {', '.join([f'P{p}' for p in sorted(missing_ids)])}")
        if not found_ids:
            print(f"❌ No valid persons found!")
            sys.exit(1)
    
    # Convert to detector format (with priority handling for overlaps)
    output_data = create_detector_format_output(selected_data)
    
    # Save to file in detector format
    output_path = output_dir / 'final_tracklet.npz'
    save_detector_format(output_data, output_path)
    
    # Print summary
    print_summary_detector_format(output_data, selected_data)
    print(f"✅ Saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"   Format: Detector compatible (frame_numbers, bboxes)")
    print(f"\n   Ready for: python run_posedet.py --config configs/posedet.yaml")


if __name__ == '__main__':
    main()
