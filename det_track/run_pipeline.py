#!/usr/bin/env python3
"""
Unified Detection & Tracking Pipeline

Orchestrates all stages of the detection-tracking pipeline:
  Stage 1: YOLO Detection
  Stage 2: ByteTrack Tracking
  Stage 3: Tracklet Analysis
  Stage 4a: ReID Recovery (optional)
  Stage 4b: Canonical Grouping (optional)
  Stage 5: Primary Person Ranking

Usage:
    python run_pipeline.py --config configs/pipeline_config.yaml
    python run_pipeline.py --config configs/pipeline_config.yaml --stages 1,2,3,5
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
    return resolve_path_variables(config)


def run_stage(stage_name, stage_script, config_path, verbose=False):
    """Run a single stage script"""
    print(f"\n🚀 Running {stage_name}...")
    
    if verbose:
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
    
    print(f"✅ {stage_name} completed in {t_end - t_start:.2f}s")
    return True


def run_pipeline(config_path, stages_to_run=None, verbose=False):
    """Run the full pipeline"""
    
    # Load config
    config = load_config(config_path)
    
    # Pipeline stages
    all_stages = [
        ('Stage 1: Detection', 'stage1_detect.py', 'stage1_detect'),
        ('Stage 2: Tracking', 'stage2_track.py', 'stage2_track'),
        ('Stage 3: Analysis', 'stage3_analyze_tracklets.py', 'stage3_analyze'),
        ('Stage 4a: ReID Recovery', 'stage4a_reid_recovery.py', 'stage4a_reid_recovery'),
        ('Stage 4b: Canonical Grouping', 'stage4b_group_canonical.py', 'stage4b_group_canonical'),
        ('Stage 5: Ranking', 'stage5_rank_persons.py', 'stage5_rank')
    ]
    
    # Print header
    print(f"\n{'='*70}")
    print(f"🎬 UNIFIED DETECTION & TRACKING PIPELINE")
    print(f"{'='*70}\n")
    print(f"Config: {config_path}")
    
    # Determine which stages to run
    enabled_stages = config['pipeline']['stages']
    
    if stages_to_run is not None:
        # User specified stages
        stage_indices = [int(s.strip()) for s in stages_to_run.split(',')]
        stages = [all_stages[i-1] for i in stage_indices if 1 <= i <= len(all_stages)]
        print(f"Running stages: {', '.join([str(i) for i in stage_indices])}")
    else:
        # Run all enabled stages
        stages = [
            stage for stage in all_stages
            if enabled_stages.get(stage[2], True)
        ]
        enabled_nums = [i+1 for i, stage in enumerate(all_stages) if enabled_stages.get(stage[2], True)]
        print(f"Running enabled stages: {', '.join([str(i) for i in enabled_nums])}")
    
    print(f"\n{'='*70}\n")
    
    # Run stages
    pipeline_start = time.time()
    
    for stage_name, stage_script, stage_key in stages:
        # Get script path (relative to this orchestrator)
        script_dir = Path(__file__).parent
        script_path = script_dir / stage_script
        
        if not script_path.exists():
            print(f"❌ Stage script not found: {script_path}")
            return False
        
        # Run stage
        success = run_stage(stage_name, str(script_path), config_path, verbose)
        
        if not success:
            print(f"\n❌ Pipeline failed at {stage_name}")
            return False
    
    pipeline_end = time.time()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✅ PIPELINE COMPLETE!")
    print(f"{'='*70}\n")
    print(f"Total time: {pipeline_end - pipeline_start:.2f}s")
    print(f"Stages run: {len(stages)}")
    
    # Show output files
    print(f"\n📦 Output Files:")
    
    stage_outputs = {
        'stage1_detect': config['stage1_detect']['output']['detections_file'],
        'stage2_track': config['stage2_track']['output']['tracklets_file'],
        'stage3_analyze': [
            config['stage3_analyze']['output']['tracklet_stats_file'],
            config['stage3_analyze']['output']['candidates_file']
        ],
        'stage4a_reid_recovery': [
            config['stage4a_reid_recovery']['output']['recovered_tracklets_file'],
            config['stage4a_reid_recovery']['output']['merge_log_file']
        ],
        'stage4b_group_canonical': [
            config['stage4b_group_canonical']['output']['canonical_persons_file'],
            config['stage4b_group_canonical']['output']['grouping_log_file']
        ],
        'stage5_rank': [
            config['stage5_rank']['output']['primary_person_file'],
            config['stage5_rank']['output']['ranking_report_file']
        ]
    }
    
    for stage_name, _, stage_key in stages:
        outputs = stage_outputs.get(stage_key, [])
        if isinstance(outputs, str):
            outputs = [outputs]
        
        for output_file in outputs:
            output_path = Path(output_file)
            if output_path.exists():
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
    
    args = parser.parse_args()
    
    # Run pipeline
    success = run_pipeline(args.config, args.stages, args.verbose)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
