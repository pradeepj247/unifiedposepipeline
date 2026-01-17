#!/usr/bin/env python3
"""
Stage 5: Person Selection & Bbox Extraction
============================================

Extracts frame-by-frame bounding box data for a selected person from canonical_persons_3c.npz
and saves as selected_person.npz in a format compatible with pose estimation pipelines.

This is a MANUAL step - run AFTER viewing the HTML report from Stage 4 to visually identify
which person you want to extract for pose estimation.

Usage:
    python stage5_select_person.py --config configs/pipeline_config.yaml --person_id 5
    python stage5_select_person.py --config configs/pipeline_config.yaml --person_id 3 --verbose

Output:
    selected_person.npz with keys:
        - frame_numbers: (N,) Frame indices where person appears
        - bboxes: (N, 4) Bounding boxes [x1, y1, x2, y2]
        - confidences: (N,) YOLO detection confidences (optional metadata)
        - person_id: Selected person ID (optional metadata)
        - video_metadata: Video info dict (optional metadata)
        - person_metadata: Person info dict (optional metadata)
"""

import argparse
import numpy as np
from pathlib import Path
import sys
import yaml
import re
import cv2
import time


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 5: Extract selected person bbox data from canonical_persons_3c.npz",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # After viewing Stage 4 HTML report, select person by ID:
  python stage5_select_person.py --config configs/pipeline_config.yaml --person_id 5
  python stage5_select_person.py --config configs/pipeline_config.yaml --person_id 3 --verbose
        """
    )
    
    parser.add_argument(
        '--person_id',
        type=int,
        required=True,
        help='Person ID to extract (e.g., 5 for person_3c_005 from HTML viewer)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to pipeline config YAML'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def resolve_path_variables(config):
    """Recursively resolve ${variable} in config"""
    global_vars = config.get('global', {})
    
    # First pass: resolve variables within global section itself
    def resolve_string_once(s, vars_dict):
        if not isinstance(s, str):
            return s
        return re.sub(
            r'\$\{(\w+)\}',
            lambda m: str(vars_dict.get(m.group(1), m.group(0))),
            s
        )
    
    # Resolve global variables iteratively
    max_iterations = 10
    for _ in range(max_iterations):
        resolved_globals = {}
        changed = False
        for key, value in global_vars.items():
            if isinstance(value, str):
                resolved = resolve_string_once(value, global_vars)
                resolved_globals[key] = resolved
                if resolved != value:
                    changed = True
            else:
                resolved_globals[key] = value
        global_vars = resolved_globals
        if not changed:
            break
    
    def resolve_string(s):
        return re.sub(
            r'\$\{(\w+)\}',
            lambda m: str(global_vars.get(m.group(1), m.group(0))),
            s
        )
    
    def resolve_recursive(obj):
        if isinstance(obj, dict):
            return {k: resolve_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_recursive(v) for v in obj]
        elif isinstance(obj, str):
            return resolve_string(obj)
        return obj
    
    result = resolve_recursive(config)
    result['global'] = global_vars
    return result


def load_config(config_path):
    """Load and resolve YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Auto-extract current_video from video_file
    video_file = config.get('global', {}).get('video_file', '')
    if video_file:
        video_name = Path(video_file).stem
        config['global']['current_video'] = video_name
    
    # Resolve path variables
    config = resolve_path_variables(config)
    
    return config


def load_canonical_persons_3c(npz_path):
    """Load canonical_persons_3c.npz and return persons array."""
    if not npz_path.exists():
        raise FileNotFoundError(f"File not found: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    persons = data['persons']
    
    return persons


def find_person_by_id(persons, person_id):
    """
    Find person by ID in persons array.
    
    Returns:
        person dict if found, None otherwise
    """
    for person in persons:
        if person['person_id'] == person_id:
            return person
    return None


def get_video_metadata(video_path):
    """
    Extract metadata from video file.
    
    Returns:
        dict with video_path, total_frames, fps, resolution
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    return {
        'video_path': str(video_path),
        'total_frames': total_frames,
        'fps': fps,
        'resolution': (width, height)
    }


def create_selected_person_output(person, video_metadata):
    """
    Create output dict in format compatible with pose detection pipelines.
    
    Args:
        person: Person dict from canonical_persons_3c.npz
        video_metadata: Video metadata dict
    
    Returns:
        Dict with all required and optional keys for selected_person.npz
    """
    frame_numbers = person['frame_numbers']
    bboxes = person['bboxes']
    confidences = person['confidences']
    person_id = person['person_id']
    
    # Calculate person metadata
    duration_frames = len(frame_numbers)
    first_frame = int(frame_numbers[0])
    last_frame = int(frame_numbers[-1])
    span_frames = last_frame - first_frame + 1
    coverage_ratio = duration_frames / span_frames if span_frames > 0 else 0.0
    
    person_metadata = {
        'source': '3c',
        'tracklet_ids': person.get('original_tracklet_ids', []),
        'duration_frames': duration_frames,
        'first_frame': first_frame,
        'last_frame': last_frame,
        'coverage_ratio': coverage_ratio,
        'num_tracklets_merged': person.get('num_tracklets_merged', 1)
    }
    
    output_data = {
        # REQUIRED keys for pose detection (backward compatible)
        'frame_numbers': frame_numbers.astype(np.int64),
        'bboxes': bboxes.astype(np.int64),
        
        # OPTIONAL keys (rich metadata)
        'person_id': int(person_id),
        'confidences': confidences.astype(np.float32),
        'video_metadata': video_metadata,
        'person_metadata': person_metadata
    }
    
    return output_data


def save_selected_person(output_data, output_path):
    """
    Save selected person data to NPZ file.
    
    Args:
        output_data: Dict with all keys
        output_path: Path to output NPZ file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        **output_data
    )


def print_summary(output_data, output_path, verbose=False):
    """Print summary of extraction."""
    frame_numbers = output_data['frame_numbers']
    bboxes = output_data['bboxes']
    confidences = output_data['confidences']
    person_id = output_data['person_id']
    person_meta = output_data['person_metadata']
    video_meta = output_data['video_metadata']
    
    print(f"\n{'='*70}")
    print(f"✅ STAGE 5: PERSON EXTRACTION COMPLETE")
    print(f"{'='*70}\n")
    
    print(f"📋 Selected Person:")
    print(f"   • Person ID: {person_id}")
    print(f"   • Merged from {person_meta['num_tracklets_merged']} tracklet(s): {person_meta['tracklet_ids']}")
    print(f"   • Frame range: {person_meta['first_frame']} - {person_meta['last_frame']}")
    print(f"   • Duration: {person_meta['duration_frames']} frames ({person_meta['coverage_ratio']:.1%} coverage)")
    print(f"   • Mean confidence: {confidences.mean():.3f}")
    
    print(f"\n📊 Bounding Boxes:")
    bbox_min = bboxes.min(axis=0)
    bbox_max = bboxes.max(axis=0)
    print(f"   • Total bboxes: {len(bboxes)}")
    print(f"   • Min bbox: ({bbox_min[0]}, {bbox_min[1]}, {bbox_max[2]}, {bbox_max[3]})")
    print(f"   • Max bbox: ({bbox_max[0]}, {bbox_max[1]}, {bbox_max[2]}, {bbox_max[3]})")
    
    print(f"\n🎥 Video Info:")
    print(f"   • Video: {Path(video_meta['video_path']).name}")
    print(f"   • Resolution: {video_meta['resolution'][0]}x{video_meta['resolution'][1]}")
    print(f"   • FPS: {video_meta['fps']:.1f}")
    print(f"   • Total frames: {video_meta['total_frames']}")
    
    print(f"\n💾 Output:")
    print(f"   • File: {output_path}")
    print(f"   • Size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"   • Format: Pose detection compatible (frame_numbers, bboxes + metadata)")
    
    if verbose:
        print(f"\n🔍 Detailed Info:")
        print(f"   • First 5 frames: {frame_numbers[:5].tolist()}")
        print(f"   • First 2 bboxes: {bboxes[:2].tolist()}")
        print(f"   • Confidence range: [{confidences.min():.3f}, {confidences.max():.3f}]")
    
    print(f"\n✅ Next Step:")
    print(f"   Run pose estimation: python run_posedet.py --config configs/posedet.yaml")
    print(f"   (Ensure detections_file in config points to: {output_path.name})")
    
    print(f"\n{'='*70}\n")


def main():
    """Main entry point."""
    args = parse_arguments()
    
    person_id = args.person_id
    verbose = args.verbose
    
    print(f"\n{'='*70}")
    print(f"🎯 STAGE 5: PERSON SELECTION & BBOX EXTRACTION")
    print(f"{'='*70}\n")
    print(f"🔍 Extracting person ID: {person_id}")
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"❌ Error: Config file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)
    
    if verbose:
        print(f"   ✓ Loaded config: {args.config}")
    
    # Get paths from config
    stage_config = config.get('stage3c_filter', {})
    output_config = stage_config.get('output', {})
    
    # Input file: canonical_persons_3c.npz
    canonical_persons_path = Path(output_config.get('canonical_persons_filtered_file', ''))
    if not canonical_persons_path.exists():
        print(f"❌ Error: Input file not found: {canonical_persons_path}")
        print(f"   Run Stage 3c first: python run_pipeline.py --stages 3c")
        sys.exit(1)
    
    if verbose:
        print(f"   ✓ Found canonical_persons_3c.npz: {canonical_persons_path}")
    
    # Output file: selected_person.npz
    output_dir = canonical_persons_path.parent
    output_path = output_dir / 'selected_person.npz'
    
    # Video file for metadata - try multiple locations
    # 1. First try canonical_video.mp4 (created by Stage 0 in output dir)
    canonical_video_path = output_dir / 'canonical_video.mp4'
    if canonical_video_path.exists():
        video_path = canonical_video_path
        if verbose:
            print(f"   ✓ Using canonical video: {canonical_video_path}")
    else:
        # 2. Try constructing from video_dir + video_file
        video_dir = Path(config['global'].get('video_dir', ''))
        video_file = config['global'].get('video_file', '')
        video_path = video_dir / video_file
        
        if not video_path.exists():
            print(f"❌ Error: Video file not found")
            print(f"   Tried:")
            print(f"   1. {canonical_video_path}")
            print(f"   2. {video_path}")
            print(f"\n   💡 Tip: Stage 0 creates canonical_video.mp4 in the output directory")
            print(f"   Run Stage 0 first if missing: python run_pipeline.py --stages 0")
            sys.exit(1)
        
        if verbose:
            print(f"   ✓ Found video: {video_path}")
    
    # Load canonical persons from Stage 3c
    print(f"\n📂 Loading canonical persons from Stage 3c...")
    try:
        persons = load_canonical_persons_3c(canonical_persons_path)
        print(f"   ✓ Loaded {len(persons)} persons")
    except Exception as e:
        print(f"❌ Error loading canonical persons: {e}")
        sys.exit(1)
    
    # Find selected person by ID
    print(f"\n🔎 Searching for person {person_id}...")
    person = find_person_by_id(persons, person_id)
    
    if person is None:
        available_ids = sorted([p['person_id'] for p in persons])
        print(f"❌ Error: Person ID {person_id} not found in canonical_persons_3c.npz")
        print(f"   Available person IDs: {available_ids}")
        print(f"   View the HTML report from Stage 4 to see all persons")
        sys.exit(1)
    
    print(f"   ✓ Found person {person_id}")
    print(f"   • Appears in {len(person['frame_numbers'])} frames")
    print(f"   • Merged from {person.get('num_tracklets_merged', 1)} tracklet(s)")
    
    # Extract video metadata
    print(f"\n📹 Extracting video metadata...")
    try:
        video_metadata = get_video_metadata(video_path)
        print(f"   ✓ Video: {video_metadata['resolution'][0]}x{video_metadata['resolution'][1]} @ {video_metadata['fps']:.1f} fps")
        print(f"   ✓ Total frames: {video_metadata['total_frames']}")
    except Exception as e:
        print(f"❌ Error reading video metadata: {e}")
        sys.exit(1)
    
    # Create output data
    print(f"\n📦 Creating output NPZ...")
    t_start = time.time()
    output_data = create_selected_person_output(person, video_metadata)
    t_create = time.time() - t_start
    
    if verbose:
        print(f"   ✓ Created output data in {t_create*1000:.1f}ms")
    
    # Save to file
    print(f"\n💾 Saving to: {output_path.name}...")
    t_start = time.time()
    save_selected_person(output_data, output_path)
    t_save = time.time() - t_start
    
    if verbose:
        print(f"   ✓ Saved in {t_save*1000:.1f}ms")
    
    # Print summary
    print_summary(output_data, output_path, verbose=verbose)


if __name__ == '__main__':
    main()
