#!/usr/bin/env python3
"""
Stage 4a: Lightweight Crops Cache Loader

Simply loads the crops cache from disk for use in Stage 7.
No ReID processing - just cache management.

Usage:
    python stage4_load_crops_cache.py --config configs/pipeline_config.yaml
"""

import argparse
import yaml
import pickle
import time
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import PipelineLogger


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
        import os
        video_name = os.path.splitext(video_file)[0]
        config['global']['current_video'] = video_name
    
    return resolve_path_variables(config)


def run_load_crops_cache(config):
    """Run Stage 4a: Load Crops Cache"""
    
    stage_config = config['stage4']
    verbose = stage_config.get('advanced', {}).get('verbose', False) or config.get('global', {}).get('verbose', False)
    
    logger = PipelineLogger("Stage 4: Load Crops Cache", verbose=verbose)
    
    input_config = stage_config['input']
    crops_cache_file = input_config.get('crops_cache_file')
    
    logger.header()
    
    if not crops_cache_file:
        logger.warning("No crops_cache_file specified in config")
        logger.info("This is OK if Stage 1 just ran and created it.")
        return {
            'crops_cache_file': None,
            'cache_size_mb': 0,
            'num_frames': 0
        }
    logger.step(f"Loading crops cache: {crops_cache_file}")
    
    if not Path(crops_cache_file).exists():
        logger.error(f"Crops cache file not found: {crops_cache_file}")
        logger.info("rops cache file not found: {crops_cache_file}")
        print(f"   Please run Stage 1 first to generate crops_cache.pkl")
        return {
            'crops_cache_file': None,
            'cache_size_mb': 0,
            'num_frames': 0
        }
    
    # Load crops cache
    t_start = time.time()
    with open(crops_cache_file, 'rb') as f:
        crops_cache = pickle.load(f)
    t_load = time.time() - t_start
    
    # Analyze cache
    num_frames = len(crops_cache)
    total_crops = sum(len(frame_crops) for frame_crops in crops_cache.values())
    cache_size_mb = Path(crops_cache_file).stat().st_size / (1024 * 1024)
    
    logger.info(f"Crops cache loaded!")
    logger.stat("Frames", num_frames)
    logger.stat("Total crops", total_crops)
    logger.file_size("Cache size", cache_size_mb)
    logger.timing("Load time", t_load)
    
    logger.success()
    
    return {
        'crops_cache_file': crops_cache_file,
        'crops_cache': crops_cache,
        'cache_size_mb': cache_size_mb,
        'num_frames': num_frames,
        'total_crops': total_crops
    }


def main():
    parser = argparse.ArgumentParser(description='Stage 4a: Load Crops Cache')
    parser.add_argument('--config', required=True, help='Config file path')
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_load_crops_cache(config)


if __name__ == '__main__':
    main()
