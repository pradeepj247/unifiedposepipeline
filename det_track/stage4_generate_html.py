#!/usr/bin/env python3
"""
Stage 4: Generate HTML Viewer with On-Demand Crop Extraction

Extracts person crops on-demand from video and generates WebP animations with HTML viewer.
Replaces the old multi-stage approach (Stage 4→10→11).

Key Improvements (Phase 3):
- No intermediate crop storage (saves ~812 MB)
- Faster execution (~13s total vs ~10.8s for old 3-stage approach)
- Better quality control (early appearance filter)
- Simpler pipeline (no Stage 4a/4b/10/11 needed)

Algorithm:
1. Load canonical_persons.npz (persons with bboxes)
2. Apply early appearance filter (exclude late-appearing persons)
3. Extract crops via single linear pass through video
4. Generate WebP animations
5. Create HTML viewer

Usage:
    python stage4_generate_html.py --config configs/pipeline_config.yaml
"""

import argparse
import yaml
import numpy as np
import time
import re
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

# Import the on-demand extraction module
from ondemand_crop_extraction import extract_crops_from_video, generate_webp_animations

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import PipelineLogger


def resolve_path_variables(config):
    """Recursively resolve ${variable} in config"""
    global_vars = config.get('global', {})
    
    def resolve_string_once(s, vars_dict):
        if not isinstance(s, str):
            return s
        pattern = re.compile(r'\$\{(\w+)\}')
        return pattern.sub(lambda m: str(vars_dict.get(m.group(1), m.group(0))), s)
    
    def resolve_dict(d, vars_dict):
        for key, value in d.items():
            if isinstance(value, dict):
                resolve_dict(value, vars_dict)
            elif isinstance(value, list):
                d[key] = [resolve_string_once(item, vars_dict) if isinstance(item, str) else item for item in value]
            elif isinstance(value, str):
                d[key] = resolve_string_once(value, vars_dict)
    
    # Multi-pass resolution
    for _ in range(5):
        old_config = str(config)
        resolve_dict(config, global_vars)
        resolve_dict(config, config.get('global', {}))
        if str(config) == old_config:
            break
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Stage 4: Generate HTML Viewer')
    parser.add_argument('--config', type=str, required=True, help='Path to pipeline config YAML')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Auto-extract current_video from video_file (needed for path resolution)
    video_file = config.get('global', {}).get('video_file', '')
    if video_file:
        import os
        video_name = os.path.splitext(video_file)[0]
        config['global']['current_video'] = video_name
    
    config = resolve_path_variables(config)
    
    # Extract configuration
    global_config = config.get('global', {})
    stage_config = config.get('stage4_generate_html', {})
    
    # Input/output paths
    video_path = stage_config.get('video_file')  # Use canonical video from stage config
    canonical_persons_file = stage_config.get('canonical_persons_file')
    output_dir = stage_config.get('output_dir')
    
    # Parameters
    crops_per_person = stage_config.get('crops_per_person', 50)
    top_n_persons = stage_config.get('top_n_persons', 10)
    max_first_appearance_ratio = stage_config.get('max_first_appearance_ratio', 0.5)
    resize_to = tuple(stage_config.get('resize_to', [256, 256]))
    webp_duration_ms = stage_config.get('webp_duration_ms', 100)
    
    # Logging
    log_file = stage_config.get('log_file')
    verbose = stage_config.get('advanced', {}).get('verbose', False) or config.get('global', {}).get('verbose', False)
    logger = PipelineLogger("Stage 4: Generate HTML Viewer", verbose=verbose)
    
    logger.header()
    if verbose:
        logger.info(f"Video: {video_path}")
        logger.info(f"Canonical persons: {canonical_persons_file}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Configuration:")
        logger.info(f"  - Crops per person: {crops_per_person}")
        logger.info(f"  - Top N persons: {top_n_persons}")
        logger.info(f"  - Early appearance threshold: {max_first_appearance_ratio*100:.0f}% of video")
        logger.info(f"  - Resize to: {resize_to}")
        logger.info(f"  - WebP duration: {webp_duration_ms}ms")
        print()
    
    # Load canonical persons
    if verbose:
        logger.step("Loading canonical persons...")
    try:
        data = np.load(canonical_persons_file, allow_pickle=True)
        persons = data['persons']
        if verbose:
            logger.info(f"Loaded {len(persons)} persons from {Path(canonical_persons_file).name}")
    except Exception as e:
        logger.error(f"Error loading canonical persons: {e}")
        return 1
    
    # Extract crops on-demand
    if verbose:
        print()
        logger.step("Extracting crops on-demand from video...")
    extraction_start = time.time()
    
    try:
        person_buckets, metadata = extract_crops_from_video(
            video_path=video_path,
            persons=persons,
            target_crops_per_person=crops_per_person,
            top_n=top_n_persons,
            max_first_appearance_ratio=max_first_appearance_ratio,
            verbose=verbose
        )
        extraction_time = time.time() - extraction_start
        if verbose:
            logger.timing("Extraction", extraction_time)
            logger.info(f"Extracted {sum(len(crops) for crops in person_buckets.values())} total crops")
            logger.info(f"Selected {len(person_buckets)} persons")
    except Exception as e:
        logger.error(f"Error during crop extraction: {e}")
        return 1
    
    # Generate WebPs
    if verbose:
        print()
        logger.step("Generating WebP animations...")
    webp_start = time.time()
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generate_webp_animations(
            person_buckets=person_buckets,
            output_dir=output_path,
            metadata=metadata,
            resize_to=resize_to,
            duration_ms=webp_duration_ms,
            verbose=verbose
        )
        webp_time = time.time() - webp_start
        if verbose:
            logger.timing("WebP generation", webp_time)
    except Exception as e:
        logger.error(f"Error during WebP generation: {e}")
        return 1
    
    # Summary
    total_time = extraction_time + webp_time
    if verbose:
        print()
        print("=" * 70)
        logger.info(f"Timing breakdown:")
        logger.info(f"  - Crop extraction: {extraction_time:.2f}s")
        logger.info(f"  - WebP generation: {webp_time:.2f}s")
        logger.info(f"  - Total: {total_time:.2f}s")
        print()
        logger.info(f"Output:")
        logger.info(f"  - WebP files: {output_path}")
        logger.info(f"  - HTML viewer: {output_path / 'viewer.html'}")
        print()
        logger.verbose_info(f"Storage savings vs old approach:")
        logger.verbose_info(f"  - Old: crops_by_person.pkl (~812 MB)")
        logger.verbose_info(f"  - New: Direct extraction (0 MB intermediate)")
        logger.verbose_info(f"  - Savings: ~812 MB")
        print()
    
    logger.success()
    
    # Save timing sidecar for run_pipeline.py
    try:
        sidecar_data = {
            'extraction_time': extraction_time,
            'webp_generation_time': webp_time,
            'total_time': total_time,
            'num_webps_created': len(person_buckets),
            'storage_saved_mb': 812
        }
        sidecar_path = output_path / 'ondemand_webp_timing.json'
        with open(sidecar_path, 'w') as f:
            import json
            json.dump(sidecar_data, f, indent=2)
    except Exception:
        pass  # Non-fatal
    
    return 0


if __name__ == '__main__':
    exit(main())
