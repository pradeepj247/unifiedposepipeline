#!/usr/bin/env python3
"""
Unified Detection & Tracking Pipeline

STAGE NUMBERING (Simple & Clear):
  Stage 1:  YOLO Detection
  Stage 2:  ByteTrack Tracking
  Stage 3:  Tracklet Analysis
  Stage 4:  Load Crops Cache
  Stage 5:  Canonical Person Grouping
  Stage 6:  Enrich Crops with HDF5
  Stage 7:  Rank Persons
  Stage 8:  Visualize Grouping (Debug only)
  Stage 9:  Output Video Visualization
  Stage 10: HTML Selection Report
  Stage 11: Generate Person WebPs

USAGE EXAMPLES:
  # Run all enabled stages
  python run_pipeline.py --config configs/pipeline_config.yaml
  
  # Run specific stages (e.g., HTML + WebPs)
  python run_pipeline.py --config configs/pipeline_config.yaml --stages 10,11
  
  # Run specific stages with --force (skip cache check)
  python run_pipeline.py --config configs/pipeline_config.yaml --stages 10,11 --force
  
  # Run from detection through ranking
  python run_pipeline.py --config configs/pipeline_config.yaml --stages 1,2,3,4,5,6,7
"""

import argparse
import yaml
import re
import time
import subprocess
import sys
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


def run_stage(stage_name, stage_script, config_path, verbose=False):
    """Run a single stage script"""
    if verbose:
        print(f"\n🚀 Running {stage_name}...")
        print(f"   Script: {stage_script}")
        print(f"   Config: {config_path}")
    
    t_start = time.time()
    
    # Run stage script with real-time output streaming
    result = subprocess.run(
        [sys.executable, '-u', stage_script, '--config', config_path],
        capture_output=False,  # Stream output directly to console
        text=True
    )
    
    t_end = time.time()
    
    if result.returncode != 0:
        print(f"❌ {stage_name} failed!")
        return False

    # For YOLO detection stage, print a streamlined completion line with an extra leading space
    if 'YOLO' in stage_name:
        base = stage_name.split(':')[0]
        print(f" ✅ {base}:  Detection completed in {t_end - t_start:.2f}s")
    else:
        print(f"✅ {stage_name} completed in {t_end - t_start:.2f}s")
    return True


def check_stage_outputs_exist(config, stage_key):
    """Check if stage output files already exist"""
    # Map stage keys to config section names
    stage_to_section = {
        'stage1': 'stage1',
        'stage2': 'stage2',
        'stage3': 'stage3',
        'stage4': 'stage4',
        'stage5': 'stage5',
        'stage6': 'stage6',
        'stage7': 'stage7',
        'stage8': 'stage8',
        'stage9': 'stage9',
        'stage10': 'stage10',
        'stage11': 'stage11'
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


def run_pipeline(config_path, stages_to_run=None, verbose=False, force=False):
    """Run the full pipeline"""
    
    # Load config
    config = load_config(config_path)
    
    # Pipeline stages - SIMPLE NUMERIC IDs FOR CLEAR REFERENCING
    # Usage: --stages 1,2,3  or  --stages 10,11
    # NOTE: Stage 10 must run AFTER Stage 11 (generate WebPs before embedding in HTML)
    all_stages = [
        ('Stage 1: YOLO Detection', 'stage1_detect.py', 'stage1'),
        ('Stage 2: ByteTrack Tracking', 'stage2_track.py', 'stage2'),
        ('Stage 3: Tracklet Analysis', 'stage3_analyze_tracklets.py', 'stage3'),
        ('Stage 4: Load Crops Cache', 'stage4_load_crops_cache.py', 'stage4'),
        ('Stage 5: Canonical Person Grouping', 'stage5_group_canonical.py', 'stage5'),
        ('Stage 6: Enrich Crops with HDF5', 'stage6_enrich_crops.py', 'stage6'),
        ('Stage 7: Rank Persons', 'stage7_rank_persons.py', 'stage7'),
        ('Stage 8: Visualize Grouping (Debug)', 'stage8_visualize_grouping.py', 'stage8'),
        ('Stage 9: Output Video Visualization', 'stage9_create_output_video.py', 'stage9'),
        ('Stage 10: Generate Person Animated WebPs', 'stage10_generate_person_webps.py', 'stage10'),
        ('Stage 11: HTML Selection Report (Horizontal Tape)', 'stage11_create_selection_html_horizontal.py', 'stage11')
    ]
    
    print(f"\n{'='*70}")
    print(f"🎬 UNIFIED DETECTION & TRACKING PIPELINE")
    print(f"{'='*70}\n")
    print(f"   Loaded config: {config_path}")
    
    # Determine which stages to run
    enabled_stages = config['pipeline']['stages']
    
    if stages_to_run is not None:
        # User specified stages - can be numbers or stage keys
        stage_specs = [s.strip() for s in stages_to_run.split(',')]
        stages = []
        stage_nums = []
        
        for spec in stage_specs:
            try:
                # Try as number first
                stage_num = int(spec)
                if 1 <= stage_num <= len(all_stages):
                    stages.append(all_stages[stage_num - 1])
                    stage_nums.append(str(stage_num))
            except ValueError:
                # Try as stage key (e.g., "stage9_generate_gifs")
                for idx, (name, script, key) in enumerate(all_stages):
                    if key == spec:
                        stages.append((name, script, key))
                        stage_nums.append(f"{idx+1}")
                        break
        
        print(f"   Running pipeline stages: {', '.join(stage_nums)}")
    else:
        # Run all enabled stages
        stages = [
            stage for stage in all_stages
            if enabled_stages.get(stage[2], True)
        ]
        enabled_nums = [i+1 for i, stage in enumerate(all_stages) if enabled_stages.get(stage[2], True)]
        print(f"   Running enabled stages: {', '.join([str(i) for i in enabled_nums])}")
    
    print(f"\n{'='*70}\n")
    
    # Run stages
    pipeline_start = time.time()
    stage_times = []  # Track timing for each stage
    
    for stage_name, stage_script, stage_key in stages:
        # Check if stage outputs already exist (skip unless --force is used)
        if not force and check_stage_outputs_exist(config, stage_key):
            print(f"\n⏭️  Skipping {stage_name} (outputs already exist)")
            stage_times.append((stage_name, 0.0, True))  # Mark as skipped
            continue
        
        # Get script path (relative to this orchestrator)
        script_dir = Path(__file__).parent
        script_path = script_dir / stage_script
        
        if not script_path.exists():
            print(f"❌ Stage script not found: {script_path}")
            return False
        
        # Run stage with timing
        stage_start = time.time()
        success = run_stage(stage_name, str(script_path), config_path, verbose)
        stage_end = time.time()
        stage_duration = stage_end - stage_start
        
        stage_times.append((stage_name, stage_duration, False))  # Not skipped
        
        if not success:
            print(f"\n❌ Pipeline failed at {stage_name}")
            return False
    
    pipeline_end = time.time()
    total_time = pipeline_end - pipeline_start
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✅ PIPELINE COMPLETE!")
    print(f"{'='*70}\n")
    
    # Timing breakdown table
    print(f"⏱️  TIMING SUMMARY:")
    print(f"-" * 70)
    
    executed_times = [(name, duration) for name, duration, skipped in stage_times if not skipped]
    
    if executed_times:
        try:
            from tabulate import tabulate
            
            table_data = []
            for stage_name, duration in executed_times:
                percentage = (duration / total_time * 100) if total_time > 0 else 0
                table_data.append([
                    stage_name,
                    f"{duration:.2f}s",
                    f"{percentage:.1f}%"
                ])
            
            # Add total row
            table_data.append([
                "TOTAL",
                f"{total_time:.2f}s",
                "100.0%"
            ])
            
            headers = ['Stage', 'Time', '% of Total']
            print(tabulate(table_data, headers=headers, tablefmt='simple'))
        except ImportError:
            # Fallback if tabulate not available
            for stage_name, duration in executed_times:
                percentage = (duration / total_time * 100) if total_time > 0 else 0
                print(f"  {stage_name}: {duration:.2f}s ({percentage:.1f}%)")
            print(f"  TOTAL: {total_time:.2f}s (100.0%)")
    else:
        print(f"Total time: {total_time:.2f}s (all stages skipped)")
    
    print(f"\nStages executed: {len(executed_times)}")
    print(f"Stages skipped: {len([stage for stage, _, skipped in stage_times if skipped])}")
    
    # Show output files
    print(f"\n📦 Output Files:")
    
    stage_outputs = {
        'stage1': config['stage1']['output']['detections_file'],
        'stage2': config['stage2']['output']['tracklets_file'],
        'stage3': [
            config['stage3']['output']['tracklet_stats_file'],
            config['stage3']['output']['candidates_file']
        ],
        'stage4': [],  # Lightweight stage - no outputs, just loads crops cache
        'stage5': [
            config['stage5']['output']['canonical_persons_file'],
            config['stage5']['output']['grouping_log_file']
        ],
        'stage7': [
            config['stage7']['output']['primary_person_file'],
            config['stage7']['output']['ranking_report_file']
        ],
        'stage9': [
            config.get('stage9', {}).get('output', {}).get('video_file', 'N/A')
        ],
        'stage10': [
            # HTML report - check the person_selection_report.html file
            str(Path(config['stage5']['output']['canonical_persons_file']).parent / 'person_selection_report.html')
        ],
        'stage11': [
            # Check videos subfolder in video-specific outputs
            str(Path(config['stage5']['output']['canonical_persons_file']).parent / 'videos')
        ]
    }
    
    for stage_name, _, stage_key in stages:
        outputs = stage_outputs.get(stage_key, [])
        if isinstance(outputs, str):
            outputs = [outputs]
        
        for output_file in outputs:
            output_path = Path(output_file)
            if output_path.exists():
                if output_path.is_dir():
                    # For directories, sum up all file sizes
                    total_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
                    size_mb = total_size / (1024 * 1024)
                    file_count = len(list(output_path.rglob('*')))
                    print(f"  ✅ {output_path.name} ({size_mb:.2f} MB, {file_count} files)")
                else:
                    # For files, get file size
                    size_mb = output_path.stat().st_size / (1024 * 1024)
                    print(f"  ✅ {output_path.name} ({size_mb:.2f} MB)")
            else:
                print(f"  ⚠️  {output_path.name} (not found)")
    
    print(f"\n{'='*70}\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Unified Detection & Tracking Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all enabled stages
  python run_pipeline.py --config configs/pipeline_config.yaml
  
  # Run specific stages (1, 2, 3, 5)
  python run_pipeline.py --config configs/pipeline_config.yaml --stages 1,2,3,5
  
  # Run with verbose output
  python run_pipeline.py --config configs/pipeline_config.yaml --verbose
        """
    )
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    parser.add_argument('--stages', type=str, default=None,
                       help='Comma-separated stage numbers to run (e.g., "1,2,3,5")')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--force', action='store_true',
                       help='Re-run stages even if outputs already exist')
    
    args = parser.parse_args()
    
    # Run pipeline
    success = run_pipeline(args.config, args.stages, args.verbose, args.force)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
