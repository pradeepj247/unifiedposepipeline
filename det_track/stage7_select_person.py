#!/usr/bin/env python3
"""
Stage 7: Manual Person Selection

Manually select a specific person from canonical_persons.npz and create primary_person.npz.
Use this as an alternative to Stage 5 auto-ranking when you want to manually choose 
which person to use for pose estimation.

Workflow:
  1. Run Stages 1-6 to generate visualization video
  2. Watch the visualization to see all persons with their IDs
  3. Run this script with --person-id to select your chosen person
  4. Continue with pose estimation using the generated primary_person.npz

Usage:
    python stage7_select_person.py --config configs/pipeline_config.yaml --person-id 3
    python stage7_select_person.py --config configs/pipeline_config.yaml --person-id 7 --verbose
"""

import argparse
import yaml
import numpy as np
import json
import re
from pathlib import Path


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
    """Load and resolve YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Auto-extract current_video from video_file
    video_file = config.get('global', {}).get('video_file', '')
    if video_file:
        import os
        video_name = os.path.splitext(video_file)[0]
        config['global']['current_video'] = video_name
    
    return resolve_path_variables(config)


def select_person(config, person_id, verbose=False):
    """Select a specific person and create primary_person.npz"""
    
    # Get file paths from config
    canonical_file = config['stage4b_group_canonical']['output']['canonical_persons_file']
    primary_file = config['stage5_rank']['output']['primary_person_file']
    
    print(f"\n{'='*70}")
    print(f"📌 STAGE 7: MANUAL PERSON SELECTION")
    print(f"{'='*70}\n")
    
    # Load canonical persons
    print(f"📂 Loading canonical persons...")
    canonical_path = Path(canonical_file)
    if not canonical_path.exists():
        print(f"❌ Canonical persons file not found: {canonical_path}")
        print(f"   Please run Stages 1-4b first to generate canonical persons.")
        return False
    
    data = np.load(canonical_path, allow_pickle=True)
    persons = data['persons']
    
    print(f"  ✅ Loaded {len(persons)} canonical persons")
    
    # Find the requested person
    selected_person = None
    for person in persons:
        if person['person_id'] == person_id:
            selected_person = person
            break
    
    if selected_person is None:
        print(f"\n❌ Person ID {person_id} not found in canonical persons!")
        print(f"\n📋 Available Person IDs:")
        
        # Show all available persons sorted by duration
        persons_sorted = sorted(persons, key=lambda p: len(p['frame_numbers']), reverse=True)
        for idx, p in enumerate(persons_sorted, 1):
            frames = len(p['frame_numbers'])
            tracklets = p['tracklet_ids']
            print(f"  {idx:2d}. Person {p['person_id']:2d}: {frames:4d} frames, "
                  f"tracklets {tracklets}")
        
        return False
    
    # Display person information
    print(f"\n🎯 Selected Person: {person_id}")
    print(f"  Frame count: {len(selected_person['frame_numbers'])} frames")
    print(f"  Frame range: {int(selected_person['frame_numbers'][0])} - {int(selected_person['frame_numbers'][-1])}")
    print(f"  Tracklets: {selected_person['tracklet_ids']}")
    
    if verbose:
        print(f"\n📊 Detailed Statistics:")
        frames = selected_person['frame_numbers']
        bboxes = selected_person['bboxes']
        confidences = selected_person['confidences']
        
        # Compute statistics
        start_frame = int(frames[0])
        end_frame = int(frames[-1])
        frame_range = end_frame - start_frame + 1
        coverage = len(frames) / frame_range if frame_range > 0 else 0
        avg_confidence = float(np.mean(confidences))
        
        # Bbox statistics
        bbox_widths = bboxes[:, 2] - bboxes[:, 0]
        bbox_heights = bboxes[:, 3] - bboxes[:, 1]
        avg_width = float(np.mean(bbox_widths))
        avg_height = float(np.mean(bbox_heights))
        
        print(f"  Duration: {len(frames)} frames over {frame_range} frame range")
        print(f"  Coverage: {coverage*100:.1f}%")
        print(f"  Avg confidence: {avg_confidence:.3f}")
        print(f"  Avg bbox size: {avg_width:.1f} x {avg_height:.1f} pixels")
    
    # Save primary person
    print(f"\n💾 Saving primary person...")
    output_path = Path(primary_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        primary_person=selected_person,
        person_id=selected_person['person_id'],
        frame_numbers=selected_person['frame_numbers'],
        bboxes=selected_person['bboxes'],
        confidences=selected_person['confidences']
    )
    
    print(f"  ✅ Saved: {output_path}")
    
    # Create selection report
    report_data = {
        'selection_method': 'manual',
        'selected_person_id': int(person_id),
        'frame_count': int(len(selected_person['frame_numbers'])),
        'frame_range': [int(selected_person['frame_numbers'][0]), 
                       int(selected_person['frame_numbers'][-1])],
        'tracklet_ids': [int(tid) for tid in selected_person['tracklet_ids']],
        'total_canonical_persons': int(len(persons))
    }
    
    # Save alongside ranking report (overwrite if exists)
    ranking_report_file = config['stage5_rank']['output']['ranking_report_file']
    report_path = Path(ranking_report_file).parent / 'selection_report.json'
    
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"  ✅ Saved selection report: {report_path}")
    
    print(f"\n{'='*70}")
    print(f"✅ PERSON SELECTION COMPLETE!")
    print(f"{'='*70}\n")
    print(f"📦 Output: {output_path.name}")
    print(f"🎯 Selected: Person {person_id} ({len(selected_person['frame_numbers'])} frames)")
    print(f"\n💡 Next: Use this person for pose estimation")
    print(f"{'='*70}\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Stage 7: Manual Person Selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Select person 3
  python stage7_select_person.py --config configs/pipeline_config.yaml --person-id 3
  
  # Select person 7 with verbose output
  python stage7_select_person.py --config configs/pipeline_config.yaml --person-id 7 --verbose
  
  # List all available persons (will fail but show list)
  python stage7_select_person.py --config configs/pipeline_config.yaml --person-id 999
        """
    )
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    parser.add_argument('--person-id', type=int, required=True,
                       help='Person ID to select (as shown in visualization video)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed statistics')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Select person
    success = select_person(config, args.person_id, args.verbose)
    
    if not success:
        exit(1)


if __name__ == '__main__':
    main()
