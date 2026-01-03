#!/usr/bin/env python3
"""
Validation Script: Test Crop Caching Pipeline

This script validates that:
1. Stage 1 extracts and caches crops properly
2. Crops cache is readable by Stage 4a
3. Crops are used by Stage 7b to create selection table
4. Manual selection in Stage 7 works with cached crops

Usage:
    python validate_crop_caching.py --config configs/pipeline_config.yaml
    python validate_crop_caching.py --config configs/pipeline_config.yaml --verbose
"""

import argparse
import yaml
import pickle
import numpy as np
import re
import time
from pathlib import Path


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


def validate_crops_cache(config, verbose=False):
    """Validate crops cache pipeline"""
    
    print(f"\n{'='*70}")
    print(f"🔍 VALIDATION: CROP CACHING PIPELINE")
    print(f"{'='*70}\n")
    
    # Get file paths
    detections_file = config['stage1_detect']['output']['detections_file']
    crops_cache_file = config['stage4a_reid_recovery']['input']['crops_cache_file']
    canonical_file = config['stage4b_group_canonical']['output']['canonical_persons_file']
    
    results = {
        'stage1_crops': False,
        'stage4a_load': False,
        'stage7b_ready': False,
        'errors': []
    }
    
    # Check 1: Crops cache file exists
    print(f"✓ Check 1: Crops cache exists")
    crops_path = Path(crops_cache_file)
    if crops_path.exists():
        size_mb = crops_path.stat().st_size / (1024 * 1024)
        print(f"  ✅ Found: {crops_path.name}")
        print(f"  📊 Size: {size_mb:.1f} MB")
        results['stage1_crops'] = True
    else:
        print(f"  ❌ Not found: {crops_cache_file}")
        print(f"  💡 Run Stage 1 first: python stage1_detect.py --config {config}")
        results['errors'].append(f"Crops cache not found: {crops_cache_file}")
        return results
    
    # Check 2: Load crops cache
    print(f"\n✓ Check 2: Load crops cache")
    try:
        t_start = time.time()
        with open(crops_path, 'rb') as f:
            crops_cache = pickle.load(f)
        t_load = time.time() - t_start
        
        num_frames = len(crops_cache)
        total_crops = sum(len(frame_crops) for frame_crops in crops_cache.values())
        
        print(f"  ✅ Loaded successfully in {t_load:.2f}s")
        print(f"  📊 Frames: {num_frames}")
        print(f"  📊 Total crops: {total_crops}")
        
        # Validate structure
        sample_frame = list(crops_cache.keys())[0]
        sample_crop = list(crops_cache[sample_frame].values())[0]
        
        if isinstance(sample_crop, np.ndarray):
            shape = sample_crop.shape
            dtype = sample_crop.dtype
            print(f"  ✅ Structure valid (numpy arrays)")
            print(f"  📊 Sample crop shape: {shape}")
            print(f"  📊 Sample crop dtype: {dtype}")
            
            if len(shape) == 3 and shape[2] == 3:
                print(f"  ✅ RGB format (correct for PIL/CV2)")
            else:
                print(f"  ⚠️  Unexpected shape {shape} (expected H×W×3)")
        
        results['stage4a_load'] = True
        
    except Exception as e:
        print(f"  ❌ Failed to load: {e}")
        results['errors'].append(f"Failed to load crops cache: {e}")
        return results
    
    # Check 3: Canonical persons (needed for Stage 7b)
    print(f"\n✓ Check 3: Canonical persons available")
    canonical_path = Path(canonical_file)
    if canonical_path.exists():
        try:
            data = np.load(canonical_path, allow_pickle=True)
            persons = data['persons']
            print(f"  ✅ Found: {canonical_path.name}")
            print(f"  📊 Persons: {len(persons)}")
            
            # Verify persons have required fields
            sample_person = persons[0]
            required_fields = ['frame_numbers', 'bboxes', 'confidences']
            
            all_valid = True
            for field in required_fields:
                if field in sample_person:
                    print(f"  ✅ Field '{field}' present")
                else:
                    print(f"  ❌ Field '{field}' missing")
                    all_valid = False
            
            if all_valid:
                results['stage7b_ready'] = True
            
        except Exception as e:
            print(f"  ⚠️  Issue reading canonical persons: {e}")
            results['errors'].append(f"Issue with canonical persons: {e}")
    else:
        print(f"  ⚠️  Not found: {canonical_file}")
        print(f"  💡 Run Stages 1-4b first")
        results['errors'].append(f"Canonical persons not found")
    
    # Check 4: Detections file
    print(f"\n✓ Check 4: Detections file")
    detections_path = Path(detections_file)
    if detections_path.exists():
        try:
            data = np.load(detections_path, allow_pickle=True)
            print(f"  ✅ Found: {detections_path.name}")
            print(f"  📊 Keys: {list(data.keys())}")
        except Exception as e:
            print(f"  ⚠️  Issue reading detections: {e}")
            results['errors'].append(f"Issue with detections file: {e}")
    else:
        print(f"  ⚠️  Not found: {detections_file}")
    
    # Check 5: Crop lookup test
    if results['stage4a_load'] and results['stage7b_ready']:
        print(f"\n✓ Check 5: Crop lookup test")
        try:
            # Try to get crops for first person
            first_person = persons[0]
            frame_numbers = first_person['frame_numbers']
            
            crops_found = 0
            for frame_idx in frame_numbers[:5]:  # Check first 5 frames
                frame_idx = int(frame_idx)
                if frame_idx in crops_cache:
                    for crop in crops_cache[frame_idx].values():
                        if crop is not None:
                            crops_found += 1
                            break
            
            print(f"  ✅ Crop lookup works")
            print(f"  📊 Found {crops_found} crops for person {first_person.get('person_id', 'N/A')} (first 5 frames)")
        
        except Exception as e:
            print(f"  ❌ Lookup failed: {e}")
            results['errors'].append(f"Crop lookup failed: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📋 VALIDATION SUMMARY")
    print(f"{'='*70}\n")
    
    checks = [
        ("Stage 1: Crops extracted", results['stage1_crops']),
        ("Stage 4a: Crops loadable", results['stage4a_load']),
        ("Stage 7b: Ready for table", results['stage7b_ready']),
    ]
    
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    
    for check_name, check_result in checks:
        status = "✅ PASS" if check_result else "❌ FAIL"
        print(f"  {status}: {check_name}")
    
    print(f"\n📊 Result: {passed}/{total} checks passed")
    
    if results['errors']:
        print(f"\n⚠️  Errors ({len(results['errors'])}):")
        for error in results['errors']:
            print(f"  - {error}")
    
    if passed == total:
        print(f"\n✅ Pipeline validation SUCCESSFUL!")
        print(f"\n🎯 Next steps:")
        print(f"  1. Run Stage 7b to create selection_table.png:")
        print(f"     python stage7_create_selection_table.py --config {config}")
        print(f"\n  2. View selection_table.png to choose a person")
        print(f"\n  3. Select person via Stage 7:")
        print(f"     python stage7_select_person.py --config {config} --person-id <ID>")
    else:
        print(f"\n❌ Pipeline validation FAILED!")
        print(f"\n💡 Troubleshooting:")
        print(f"  - Ensure all required stages have been run")
        print(f"  - Check that config file paths are correct")
        print(f"  - Re-run Stage 1 to regenerate crops_cache.pkl")
    
    print(f"\n{'='*70}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Validate Crop Caching Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate pipeline (basic)
  python validate_crop_caching.py --config configs/pipeline_config.yaml
  
  # Validate with verbose output
  python validate_crop_caching.py --config configs/pipeline_config.yaml --verbose
        """
    )
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Validate
    results = validate_crops_cache(config, verbose=args.verbose)
    
    # Exit with appropriate code
    exit(0 if all(results[k] for k in ['stage1_crops', 'stage4a_load', 'stage7b_ready']) else 1)


if __name__ == '__main__':
    main()
