#!/usr/bin/env python3
"""Debug script to verify config loading and path resolution."""

import yaml
import json
import sys
import re
from pathlib import Path

def resolve_path_variables(config):
    """Resolve ${variable} references in config (from stage4_generate_html.py)"""
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
    config_path = '/content/unifiedposepipeline/det_track/configs/pipeline_config.yaml'
    
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*70)
    print("BEFORE path resolution:")
    print("="*70)
    
    stage4 = config.get('stage4_generate_html', {})
    clustering = stage4.get('clustering', {})
    print(f"stage4_generate_html keys: {list(stage4.keys())}")
    print(f"clustering keys: {list(clustering.keys())}")
    print(f"clustering.num_best_crops: {clustering.get('num_best_crops', 'NOT FOUND')}")
    
    print("\n" + "="*70)
    print("AFTER path resolution:")
    print("="*70)
    
    config = resolve_path_variables(config)
    stage4 = config.get('stage4_generate_html', {})
    clustering = stage4.get('clustering', {})
    
    print(f"stage4_generate_html keys: {list(stage4.keys())}")
    print(f"clustering keys: {list(clustering.keys())}")
    print(f"clustering.num_best_crops: {clustering.get('num_best_crops', 'NOT FOUND')}")
    
    print("\n" + "="*70)
    print("FULL STAGE 4 CONFIG (pretty-printed):")
    print("="*70)
    print(json.dumps(config.get('stage4_generate_html', {}), indent=2))

if __name__ == '__main__':
    main()
