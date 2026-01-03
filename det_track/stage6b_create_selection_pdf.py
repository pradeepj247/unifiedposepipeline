#!/usr/bin/env python3
"""
Stage 6b Alternative: Create Person Selection Report (CSV + Images)

Creates a simple report with:
- CSV table of all persons with statistics
- Thumbnail images saved separately for easy review

Usage:
    python stage6b_create_selection_pdf.py --config configs/pipeline_config.yaml
"""

import argparse
import numpy as np
import pickle
import yaml
import re
import os
import cv2
import csv
from pathlib import Path
import time
from datetime import timedelta


def resolve_path_variables(config):
    """Recursively resolve ${variable} in config"""
    global_vars = config.get('global', {})
    
    def resolve_string_once(s, vars_dict):
        if not isinstance(s, str):
            return s
        return re.sub(
            r'\$\{(\w+)\}',
            lambda m: str(vars_dict.get(m.group(1), m.group(0))),
            s
        )
    
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
    
    video_file = config.get('global', {}).get('video_file', '')
    if video_file:
        video_name = os.path.splitext(video_file)[0]
        config['global']['current_video'] = video_name
    
    return resolve_path_variables(config)


def get_best_crop_for_person(person, crops_cache):
    """Get highest-confidence crop for person"""
    if person.get('frame_numbers') is None or len(person['frame_numbers']) == 0:
        return None
    
    confidences = person['confidences']
    best_idx = np.argmax(confidences)
    best_frame = int(person['frame_numbers'][best_idx])
    
    if best_frame in crops_cache:
        crops_in_frame = crops_cache[best_frame]
        for crop_image in crops_in_frame.values():
            if crop_image is not None and isinstance(crop_image, np.ndarray):
                return crop_image
    
    return None


def create_selection_report(canonical_file, crops_cache_file, fps, output_csv, thumbnails_dir):
    """Create selection report with CSV table and thumbnail images"""
    
    # Load data
    print(f"📂 Loading canonical persons...")
    data = np.load(canonical_file, allow_pickle=True)
    persons = list(data['persons'])
    persons.sort(key=lambda p: len(p['frame_numbers']), reverse=True)
    
    print(f"📂 Loading crops cache...")
    with open(crops_cache_file, 'rb') as f:
        crops_cache = pickle.load(f)
    
    # Create thumbnails directory
    print(f"🎨 Extracting thumbnails...")
    thumbnails_dir = Path(thumbnails_dir)
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    
    # Create CSV file
    print(f"📄 Creating CSV report...")
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Rank', 'Person_ID', 'Duration_Sec', 'Frames', 'Start_Frame', 'End_Frame', 'Avg_Confidence', 'Thumbnail_File'])
        
        for rank, person in enumerate(persons[:50], 1):  # Top 50 persons
            person_id = person['person_id']
            frames = person['frame_numbers']
            durations = len(frames)
            start_frame = int(frames[0])
            end_frame = int(frames[-1])
            avg_conf = np.mean(person['confidences'])
            
            duration_seconds = durations / fps if fps > 0 else durations / 25
            
            # Get crop and save
            crop = get_best_crop_for_person(person, crops_cache)
            thumbnail_file = ''
            
            if crop is not None:
                # Convert BGR to RGB and save
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                thumb_path = thumbnails_dir / f"P{person_id:03d}_rank{rank:02d}.png"
                cv2.imwrite(str(thumb_path), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
                thumbnail_file = thumb_path.name
            
            # Write row
            writer.writerow([
                rank,
                person_id,
                f'{duration_seconds:.1f}',
                durations,
                start_frame,
                end_frame,
                f'{avg_conf:.3f}',
                thumbnail_file
            ])
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Stage 6b: Create Person Selection Report (CSV + Thumbnails)'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    canonical_file = config['stage4b_group_canonical']['output']['canonical_persons_file']
    crops_cache_file = config['stage4a_reid_recovery']['input']['crops_cache_file']
    
    output_dir = Path(canonical_file).parent
    output_csv = output_dir / 'person_selection_report.csv'
    thumbnails_dir = output_dir / 'person_thumbnails'
    
    fps = config.get('global', {}).get('video_fps', 25)
    
    print(f"\n{'='*70}")
    print(f"📄 STAGE 6b: CREATE PERSON SELECTION REPORT")
    print(f"{'='*70}\n")
    
    t_start = time.time()
    
    success = create_selection_report(
        canonical_file,
        crops_cache_file,
        fps,
        output_csv,
        thumbnails_dir
    )
    
    t_end = time.time()
    
    if success:
        csv_size_mb = output_csv.stat().st_size / (1024 * 1024) if output_csv.exists() else 0
        thumb_count = len(list(thumbnails_dir.glob('*.png'))) if thumbnails_dir.exists() else 0
        
        print(f"\n✅ Report created!")
        print(f"   CSV file: {output_csv.name} ({csv_size_mb:.2f} MB)")
        print(f"   Thumbnails: {thumb_count} images in {thumbnails_dir.name}/")
        print(f"⏱️  Time: {t_end - t_start:.2f}s")
        print(f"\n{'='*70}\n")
        return True
    else:
        print(f"\n❌ Failed to create report")
        print(f"{'='*70}\n")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
