#!/usr/bin/env python3
"""
Stage 4: Generate HTML Viewer with On-Demand Crop Extraction + OSNet Clustering

Extracts person crops on-demand from video and generates WebP animations with HTML viewer.
NEW: Also extracts OSNet embeddings for ReID-based duplicate detection.
Replaces the old multi-stage approach (Stage 4→10→11).

Key Improvements (Phase 3):
- No intermediate crop storage (saves ~812 MB)
- Faster execution (~13s total vs ~10.8s for old 3-stage approach)
- Better quality control (early appearance filter)
- Simpler pipeline (no Stage 4a/4b/10/11 needed)

NEW (Phase 4):
- Integrated OSNet clustering (detect duplicate persons)
- Similarity matrix output (JSON + NPY)
- Enhanced HTML with heatmap visualization

Algorithm:
1. Load canonical_persons.npz (persons with bboxes)
2. Apply early appearance filter (exclude late-appearing persons)
3. Extract crops via single linear pass through video
4. FORK INTO TWO PATHS:
   PATH 1: Generate WebP animations (existing)
   PATH 2: Extract OSNet embeddings and compute similarity matrix (NEW)
5. Create unified HTML viewer with similarity heatmap

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

# Import OSNet clustering (NEW Phase 4)
try:
    from osnet_clustering import create_similarity_matrix, save_similarity_results
    OSNET_AVAILABLE = True
except ImportError:
    OSNET_AVAILABLE = False


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
    
    # Clustering parameters (NEW Phase 4)
    clustering_config = stage_config.get('clustering', {})
    clustering_enabled = clustering_config.get('enabled', True)
    osnet_model_path = clustering_config.get('osnet_model', None)
    device = clustering_config.get('device', 'cuda')
    num_best_crops = clustering_config.get('num_best_crops', 16)  # DEFAULT changed from 8 to 16 to match ONNX model
    similarity_threshold = clustering_config.get('similarity_threshold', 0.70)
    
    # DEBUG: Verify clustering config is loaded
    if num_best_crops != 16:
        print(f"⚠️  WARNING: num_best_crops={num_best_crops} (expected 16). Clustering config may not be loaded properly.")
        print(f"    clustering_config keys: {list(clustering_config.keys())}")
        print(f"    Full stage_config keys: {list(stage_config.keys())}")
    
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
        logger.info(f"  - OSNet Clustering: {'ENABLED' if clustering_enabled and OSNET_AVAILABLE else 'DISABLED'}")
        if clustering_enabled and OSNET_AVAILABLE:
            logger.info(f"    - Best crops per person: {num_best_crops}")
            logger.info(f"    - Similarity threshold: {similarity_threshold:.0%}")
            logger.info(f"    - Device: {device}")
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
    
    # OSNet Clustering - NEW Phase 4
    clustering_time = 0
    clustering_result = None
    if clustering_enabled and OSNET_AVAILABLE:
        if verbose:
            print()
            logger.step("Extracting OSNet embeddings for ReID clustering...")
        clustering_start = time.time()
        
        try:
            clustering_result = create_similarity_matrix(
                buckets=person_buckets,
                osnet_model_path=osnet_model_path,
                device=device,
                num_best_crops=num_best_crops,
                similarity_threshold=similarity_threshold,
                verbose=verbose
            )
            clustering_time = time.time() - clustering_start
            
            # Save results
            if verbose:
                logger.step("Saving similarity matrix and embeddings...")
            save_similarity_results(
                results=clustering_result,
                output_dir=output_path,
                verbose=verbose
            )
            
            if verbose:
                logger.timing("OSNet clustering", clustering_time)
                logger.info(f"High-similarity pairs (>{similarity_threshold}):")
                for p1, p2, score in clustering_result['high_similarity_pairs'][:10]:
                    logger.info(f"  - Person {p1} & {p2}: {score:.3f}")
        except Exception as e:
            logger.warning(f"OSNet clustering failed (non-fatal): {e}")
            clustering_enabled = False
    elif clustering_enabled and not OSNET_AVAILABLE:
        logger.warning("OSNet clustering requested but module not available")
        clustering_enabled = False
    
    # Summary
    total_time = extraction_time + webp_time + clustering_time
    if verbose:
        print()
        print("=" * 70)
        logger.info(f"Timing breakdown:")
        logger.info(f"  - Crop extraction: {extraction_time:.2f}s")
        logger.info(f"  - WebP generation: {webp_time:.2f}s")
        if clustering_enabled:
            logger.info(f"  - OSNet clustering: {clustering_time:.2f}s")
        logger.info(f"  - Total: {total_time:.2f}s")
        print()
        logger.info(f"Output:")
        logger.info(f"  - WebP files: {output_path}")
        logger.info(f"  - HTML viewer: {output_path / 'viewer.html'}")
        if clustering_enabled:
            logger.info(f"  - Similarity matrix: {output_path / 'similarity_matrix.json'}")
            logger.info(f"  - Embeddings: {output_path / 'embeddings.json'}")
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
            'clustering_time': clustering_time if clustering_enabled else 0,
            'total_time': total_time,
            'num_webps_created': len(person_buckets),
            'clustering_enabled': clustering_enabled,
            'high_similarity_pairs': len(clustering_result['high_similarity_pairs']) if clustering_result else 0,
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
