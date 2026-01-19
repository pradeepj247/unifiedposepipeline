#!/usr/bin/env python3
"""
Unified Detection & Tracking Pipeline (Single-Process Version)

OPTIMIZED VERSION: Runs all stages in a single Python process instead of
spawning subprocesses. Saves ~8 seconds by eliminating process startup and
redundant module import overhead.

IDENTICAL FUNCTIONALITY to run_pipeline.py:
- Same --stages, --mode, --force, --verbose options
- Same configuration handling
- Same output formatting

USAGE: Replace run_pipeline.py with this for faster execution:
  python run_pipeline_new.py --config configs/pipeline_config.yaml
  python run_pipeline_new.py --config configs/pipeline_config.yaml --stages 3c,4
  python run_pipeline_new.py --config configs/pipeline_config.yaml --force
"""

import argparse
import yaml
import re
import time
import sys
from pathlib import Path

# Pre-import torch once to save ~1.5s across multiple stages
import torch

# Import stage functions directly (no subprocess overhead)
from stage0_normalize_video import run_stage0_normalize
from stage1_detect import run_detection
from stage2_track import run_tracking
from stage3a_analyze_tracklets import run_analysis
from stage3b_group_canonical import run_enhanced_grouping
from stage3c_filter_persons import run_filter
from stage3d_refine_visual import run_refine


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
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        config['global']['current_video'] = video_name
    
    return resolve_path_variables(config)


def apply_mode_overrides(config, mode, verbose=False):
    """Apply mode-specific configuration overrides"""
    if mode not in ['fast', 'balanced', 'full']:
        return config
    
    mode_config = config.get('modes', {}).get(mode, {})
    
    if not mode_config:
        return config
    
    # Apply stage-specific overrides
    for stage_key, stage_overrides in mode_config.items():
        if stage_key in config:
            # Deep merge overrides into stage config
            for key, value in stage_overrides.items():
                if isinstance(value, dict) and key in config[stage_key]:
                    config[stage_key][key].update(value)
                else:
                    config[stage_key][key] = value
    
    return config


def run_stage(stage_name, stage_func, config, config_path, verbose=False):
    """Run a single stage function directly (no subprocess)"""
    if verbose:
        print(f"\n🚀 Running {stage_name}...")
    
    t_start = time.time()
    
    try:
        # Stage 4 doesn't have a separate run function - call main() from subprocess
        if stage_func is None:
            # For Stage 4, we still need to use subprocess since it doesn't expose a run function
            # Pass dual_row argument based on config
            import subprocess
            dual_row = config.get('stage4_html', {}).get('dual_row', True)
            cmd = [sys.executable, '-u', 'stage4_generate_html.py', '--config', config_path,
                   '--dual-row', 'true' if dual_row else 'false']
            result = subprocess.run(cmd, capture_output=False, text=True)
            success = result.returncode == 0
        else:
            # Call stage function directly
            stage_func(config)
            success = True
    except Exception as e:
        print(f"❌ {stage_name} failed with error: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    t_end = time.time()
    
    if not success:
        print(f"❌ {stage_name} failed!")
        return False, t_end - t_start
    
    # Stage-specific completion messages
    if 'Stage 1' in stage_name or 'Detection' in stage_name:
        print(f"  ✅ Stage 1:  Detection completed in {t_end - t_start:.2f}s")
    elif 'Stage 2' in stage_name or 'Tracking' in stage_name:
        print(f"  ✅ {stage_name} completed")
    elif 'Stage 0' in stage_name or 'Video Normalization' in stage_name:
        print("="*70)
    
    return True, t_end - t_start


def check_stage_outputs_exist(config, stage_key):
    """Check if stage output files already exist"""
    stage_to_section = {
        'stage0': 'stage0_normalize',
        'stage1': 'stage1_detect',
        'stage2': 'stage2_track',
        'stage3a': 'stage3a_analyze',
        'stage3b': 'stage3b_group',
        'stage3c': 'stage3c_filter',
        'stage3d': 'stage3d_refine',
        'stage4': 'stage4_generate_html',
    }
    
    section = stage_to_section.get(stage_key)
    if not section or section not in config:
        return False
    
    stage_config = config[section]
    output_config = stage_config.get('output', {})
    
    if not output_config:
        return False
    
    # Check all output files
    for key, filepath in output_config.items():
        if isinstance(filepath, str):
            if not Path(filepath).exists():
                return False
    
    return True


def run_pipeline(config_path, stages_to_run=None, mode=None, verbose=False, force=False):
    """Run the full pipeline in a single process"""
    
    # Load config
    config = load_config(config_path)
    
    # Determine active mode (CLI overrides config default)
    active_mode = mode if mode else config.get('mode', 'full')
    
    # Apply mode overrides
    config = apply_mode_overrides(config, active_mode, verbose=verbose)
    
    # Display mode info
    mode_info = config.get('modes', {}).get(active_mode, {})
    if mode_info and not verbose:
        description = mode_info.get('description', '')
        print(f"\n⚙️  Pipeline Mode: {active_mode.upper()}")
        if description:
            print(f"   {description}")
        print()
    
    # Define all stages with their functions
    all_stages = [
        ('Stage 0: Video Normalization', run_stage0_normalize, 'stage0'),
        ('Stage 1: Detection', run_detection, 'stage1'),
        ('Stage 2: Tracking', run_tracking, 'stage2'),
        ('Stage 3a: Tracklet Analysis', run_analysis, 'stage3a'),
        ('Stage 3b: Canonical Grouping', run_enhanced_grouping, 'stage3b'),
        ('Stage 3c: Filter Persons & Extract Crops', run_filter, 'stage3c'),
        ('Stage 3d: Visual Refinement (OSNet)', run_refine, 'stage3d'),
        ('Stage 4: Generate HTML Viewer', None, 'stage4'),  # Stage 4 handled specially
    ]
    
    print(f"\n{'='*70}")
    print(f"🎬 UNIFIED DETECTION & TRACKING PIPELINE")
    print(f"{'='*70}\n")
    print(f"   Loaded config: {config_path}")
    
    # Determine which stages to run
    enabled_stages = config['pipeline']['stages']
    
    if stages_to_run is not None:
        # User specified stages
        stage_specs = [s.strip() for s in stages_to_run.split(',')]
        stages = []
        stage_nums = []
        
        for spec in stage_specs:
            matched = False
            search_key = spec if spec.startswith('stage') else f'stage{spec}'
            for idx, (name, func, key) in enumerate(all_stages):
                if key == search_key:
                    stages.append((name, func, key))
                    stage_nums.append(key)
                    matched = True
                    break
            
            if not matched:
                print(f"   ❌ ERROR: Could not match spec '{spec}'. Valid stages: 0, 1, 2, 3a, 3b, 3c, 3d, 4")
                return False
        
        print(f"   Running pipeline stages: {', '.join(stage_nums)}")
    else:
        # Run all enabled stages
        stages = [
            stage for stage in all_stages
            if enabled_stages.get(stage[2], True)
        ]
        enabled_keys = [stage[2] for stage in all_stages if enabled_stages.get(stage[2], True)]
        print(f"   Running pipeline stages: {', '.join(enabled_keys)}")
    
    print(f"\n{'='*70}\n")
    
    # Run stages
    pipeline_start = time.time()
    stage_times = []
    
    for stage_name, stage_func, stage_key in stages:
        # Check if outputs already exist (skip logic)
        if not force and check_stage_outputs_exist(config, stage_key):
            print(f"\n⏭️  Skipping {stage_name} (outputs already exist)")
            stage_times.append((stage_name, 0.0, True))  # Mark as skipped
            print(f"\n{'='*70}")
            continue
        
        print(f"\n{'='*70}")
        print(f"📍 {stage_name.upper()}")
        print(f"{'='*70}\n")
        
        # Run stage
        success, duration = run_stage(stage_name, stage_func, config, config_path, verbose)
        
        if not success:
            print(f"\n❌ Pipeline failed at {stage_name}")
            return False
        
        stage_times.append((stage_name, duration, False))
    
    pipeline_end = time.time()
    total_time = pipeline_end - pipeline_start
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✅ PIPELINE COMPLETE!")
    print(f"{'='*70}\n")
    
    # Timing summary
    print(f"⏱️  TIMING SUMMARY:")
    print("-" * 70)
    print(f"{'Stage':<40}  {'Time':<8}  {'% of Total'}")
    print("-" * 70)
    
    executed_times = [(name, duration) for name, duration, skipped in stage_times if not skipped]
    
    if executed_times:
        for stage_name, duration in executed_times:
            pct = (duration / total_time) * 100 if total_time > 0 else 0
            print(f"{stage_name:<40}  {duration:>6.2f}s  {pct:>5.1f}%")
        
        print("-" * 70)
        print(f"{'TOTAL':<40}  {total_time:>6.2f}s  {'100.0%':>8}")
    
    # Output files summary
    print(f"\n📦 Output Files:")
    output_files = {
        'stage0': config['stage0_normalize']['output']['canonical_video_file'],
        'stage1': config['stage1_detect']['output']['detections_file'],
        'stage2': config['stage2_track']['output']['tracklets_file'],
        'stage3a': config['stage3a_analyze']['output']['tracklet_stats_file'],
        'stage3b': config['stage3b_group']['output']['canonical_persons_file'],
        'stage3c': config['stage3c_filter']['output']['canonical_persons_3c_file'],
        'stage3d': config.get('stage3d_refine', {}).get('output', {}).get('canonical_persons_merged_file', ''),
        'stage4': config.get('stage4_html', {}).get('output', {}).get('webp_output_dir', ''),
    }
    
    for stage_key, filepath in output_files.items():
        if filepath:
            output_path = Path(filepath)
            if output_path.exists():
                if output_path.is_dir():
                    print(f"  ✅ {output_path.name}")
                else:
                    size_mb = output_path.stat().st_size / (1024 * 1024)
                    print(f"  ✅ {output_path.name} ({size_mb:.2f} MB)")
            else:
                print(f"  ⚠️  {output_path.name} (not found)")
    
    print(f"\n{'='*70}\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Unified Detection & Tracking Pipeline (Single-Process Optimized)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all enabled stages
  python run_pipeline_new.py --config configs/pipeline_config.yaml
  
  # Run specific stages
  python run_pipeline_new.py --config configs/pipeline_config.yaml --stages 3c,4
  
  # Force re-run all stages
  python run_pipeline_new.py --config configs/pipeline_config.yaml --force
        """
    )
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    parser.add_argument('--stages', type=str, default=None,
                       help='Comma-separated stage keys to run (e.g., "1,2,3a,3b")')
    parser.add_argument('--mode', type=str, choices=['fast', 'balanced', 'full'],
                       help='Pipeline mode: fast (10 crops), balanced (30 crops), full (50 crops + ReID)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--force', action='store_true',
                       help='Re-run stages even if outputs already exist')
    
    args = parser.parse_args()
    
    success = run_pipeline(args.config, args.stages, args.mode, args.verbose, args.force)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
