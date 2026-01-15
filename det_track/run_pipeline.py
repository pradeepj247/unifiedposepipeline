#!/usr/bin/env python3
"""
Unified Detection & Tracking Pipeline

PIPELINE ARCHITECTURE (PHASE 4):
  Stage 0:  Video Normalization & Validation (runs FIRST)
  Stage 1:  Detection (YOLO)
  Stage 2:  Tracking (ByteTrack)
  Stage 3:  Analysis & Refinement:
            - Stage 3a: Tracklet Analysis
            - Stage 3b: Canonical Grouping
            - Stage 3c: Filter Persons & Extract Crops (40+ → 8)
            - Stage 3d: Visual Refinement via OSNet (8 → 6, ReID-based merging)
  Stage 4:  HTML Viewer Generation (visualization only)

USAGE EXAMPLES:
  # Run all enabled stages
  python run_pipeline.py --config configs/pipeline_config.yaml
  
  # Run specific stages (e.g., Stage 3 analysis sub-stages)
  python run_pipeline.py --config configs/pipeline_config.yaml --stages 3a,3b,3c,3d
  
  # Run with --force to skip cache checks
  python run_pipeline.py --config configs/pipeline_config.yaml --stages 4 --force
  
  # Run detection through refinement (skip normalization if already done)
  python run_pipeline.py --config configs/pipeline_config.yaml --stages 1,2,3a,3b,3c,3d
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
        return False, t_end - t_start

    # Stage-specific completion messages
    # Stage 0: No special handling needed (prints own completion)
    # Stage 1: Detection - streamlined message
    if 'Stage 1' in stage_name or 'Detection' in stage_name:
        print(f"  ✅ Stage 1:  Detection completed in {t_end - t_start:.2f}s")
    # Stage 2: Tracking - compact message (stage prints own breakdown)
    elif 'Stage 2' in stage_name or 'Tracking' in stage_name:
        print(f"  ✅ {stage_name} completed")
    # Stage 3a, 3b, 3c, 4 - completion with breakdown printed by orchestrator
    elif 'Stage 3a' in stage_name or 'Tracklet Analysis' in stage_name:
        pass  # Will print completion with breakdown below
    elif 'Stage 3b' in stage_name or 'Canonical Grouping' in stage_name:
        pass  # Will print completion with breakdown below
    elif 'Stage 3c' in stage_name or 'Person Ranking' in stage_name:
        pass  # Will print completion with breakdown below
    elif 'Stage 4' in stage_name or 'Generate HTML Viewer' in stage_name:
        pass  # Will print completion with breakdown below
    else:
        print(f"✅ {stage_name} completed in {t_end - t_start:.2f}s")
    
    return True, t_end - t_start


def check_stage_outputs_exist(config, stage_key):
    """Check if stage output files already exist"""
    # Map stage keys to config section names
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


def run_pipeline(config_path, stages_to_run=None, verbose=False, force=False):
    """Run the full pipeline"""
    
    # Load config
    config = load_config(config_path)
    
    # Simplified 5-stage pipeline: 0→1→2→3→4
    # Stage 3 splits into 3a (analyze) → 3b (group) → 3c (filter) → 3d (refine)
    # Usage: --stages 0,1,2,3a,3b,3c,3d,4  or  --stages stage0,stage1,stage4
    all_stages = [
        ('Stage 0: Video Normalization', 'stage0_normalize_video.py', 'stage0'),
        ('Stage 1: Detection', 'stage1_detect.py', 'stage1'),
        ('Stage 2: Tracking', 'stage2_track.py', 'stage2'),
        ('Stage 3a: Tracklet Analysis', 'stage3a_analyze_tracklets.py', 'stage3a'),
        ('Stage 3b: Canonical Grouping', 'stage3b_group_canonical.py', 'stage3b'),
        ('Stage 3c: Filter Persons & Extract Crops', 'stage3c_filter_persons.py', 'stage3c'),
        ('Stage 3d: Visual Refinement (OSNet)', 'stage3d_refine_visual.py', 'stage3d'),
        ('Stage 4: Generate HTML Viewer', 'stage4_generate_html.py', 'stage4'),
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
                # Try as stage key - handle shorthand (3c, 3d) or full key (stage3c, stage3d)
                search_key = spec if spec.startswith('stage') else f'stage{spec}'
                for idx, (name, script, key) in enumerate(all_stages):
                    if key == search_key:
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
        # Show stage keys instead of indices for clarity
        enabled_keys = [stage[2] for stage in all_stages if enabled_stages.get(stage[2], True)]
        print(f"   Running enabled stages: {', '.join(enabled_keys)}")
    
    print(f"\n{'='*70}\n")
    
    # Run stages
    pipeline_start = time.time()
    stage_times = []  # Track timing for each stage
    
    for stage_name, stage_script, stage_key in stages:
        # DEPENDENCY CHECK: Stage 4 requires Stage 3d output (final_crops.pkl)
        if stage_key == 'stage4':
            # Get the output directory from stage3d's final_crops_merged_file location
            # (same as where final_crops.pkl is saved after Stage 3d)
            final_crops_file = config.get('stage3d_refine', {}).get('output', {}).get('final_crops_merged_file', '')
            if final_crops_file:
                output_dir = Path(final_crops_file).parent
                final_crops_path = Path(final_crops_file)
            else:
                # Fallback to canonical output_dir if not configured
                output_dir = config.get('global', {}).get('output_dir', '')
                final_crops_path = Path(output_dir) / 'final_crops.pkl'
            
            if not final_crops_path.exists():
                print(f"\n❌ DEPENDENCY ERROR: Stage 4 requires final_crops.pkl from Stage 3c")
                print(f"   Missing file: {final_crops_path}")
                print(f"   Run Stage 3c first: python run_pipeline.py --config ... --stages 3c")
                print(f"   Or run the full pipeline: python run_pipeline.py --config ...")
                return False
        
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
        success, _ = run_stage(stage_name, str(script_path), config_path, verbose)
        stage_end = time.time()
        stage_duration = stage_end - stage_start
        
        stage_times.append((stage_name, stage_duration, False))  # Not skipped
        
        # Try to read the timings sidecar for stages that emit one and report overhead
        try:
            import json
            # Stage 1: YOLO detection (already supported)
            if stage_key == 'stage1':
                detections_file = config['stage1_detect']['output'].get('detections_file')
                if detections_file:
                    sidecar_path = Path(detections_file).parent / (Path(detections_file).name + '.timings.json')
                    if sidecar_path.exists():
                        with open(sidecar_path, 'r', encoding='utf-8') as sf:
                            data = json.load(sf)

                        model_load_time = float(data.get('model_load_time', 0.0))
                        detect_loop_time = float(data.get('detect_loop_time', data.get('detect_loop', 0.0)))
                        total_save_time = float(data.get('total_save_time', data.get('npz_save_time', 0.0) + data.get('crops_save_time', 0.0)))
                        num_frames_sidecar = int(data.get('num_frames', 0))

                        # Other overheads = orchestrator stage duration - (model + detect + save)
                        other_overhead = stage_duration - (model_load_time + detect_loop_time + total_save_time)
                        if other_overhead < 0 and abs(other_overhead) < 0.05:
                            other_overhead = 0.0

                        print(f"     Breakdown (stage parts):")
                        print(f"       model load: {model_load_time:.2f}s")
                        print(f"       detection processing: {detect_loop_time:.2f}s")
                        print(f"       files saving: {total_save_time:.2f}s")
                        print(f"       other overheads: {other_overhead:.2f}s")

                        # FPS metrics: detection-only, and active-FPS excluding model load
                        fps_detection_only = (num_frames_sidecar / detect_loop_time) if detect_loop_time > 0 else 0.0
                        active_time = stage_duration - model_load_time
                        fps_active = (num_frames_sidecar / active_time) if active_time > 0 else 0.0
                        print(f"     FPS (detection-only): {fps_detection_only:.1f}")
                        print(f"     FPS (detection+save+overhead): {fps_active:.1f}")
                        print(f"     Sum of parts (model+detect+save): {(model_load_time + detect_loop_time + total_save_time):.2f}s")
                    else:
                        if verbose:
                            print(f"     (No timings sidecar found at {sidecar_path.name})")

            # Stage 2: ByteTrack tracking
            # Note: Stage 2 prints its own compact reconciled breakdown; avoid duplicating it here.
            if stage_key == 'stage2':
                pass

            # Stage 3a: Tracklet Analysis
            if stage_key == 'stage3a':
                stats_file = config['stage3a_analyze']['output'].get('tracklet_stats_file')
                if stats_file:
                    sidecar_path = Path(stats_file).parent / (Path(stats_file).name + '.timings.json')
                    if sidecar_path.exists():
                        with open(sidecar_path, 'r', encoding='utf-8') as sf:
                            data = json.load(sf)

                        stats_time = float(data.get('stats_time', 0.0))
                        npz_save_time = float(data.get('npz_save_time', 0.0))
                        num_tracklets = int(data.get('num_tracklets', 0))

                        other_overhead = stage_duration - (stats_time + npz_save_time)
                        if other_overhead < 0 and abs(other_overhead) < 0.05:
                            other_overhead = 0.0

                        print(f"   ✅ Stage 3a: Tracklet Analysis completed in {stage_duration:.2f}s")
                        print(f"      Breakdown (stage parts):")
                        print(f"       compute stats: {stats_time:.2f}s")
                        print(f"       stats save: {npz_save_time:.2f}s")
                        print(f"       other overheads: {other_overhead:.2f}s")
                    else:
                        print(f"   ✅ Stage 3a: Tracklet Analysis completed in {stage_duration:.2f}s")

            # Stage 3b: Canonical Grouping
            if stage_key == 'stage3b':
                canonical_file = config['stage3b_group']['output'].get('canonical_persons_file')
                if canonical_file:
                    sidecar_path = Path(canonical_file).parent / (Path(canonical_file).name + '.timings.json')
                    if sidecar_path.exists():
                        with open(sidecar_path, 'r', encoding='utf-8') as sf:
                            data = json.load(sf)

                        grouping_time = float(data.get('grouping_time', 0.0))
                        save_time = float(data.get('npz_save_time', 0.0))
                        num_persons = int(data.get('num_persons', 0))
                        num_merged = int(data.get('num_merged_groups', 0))

                        other_overhead = stage_duration - (grouping_time + save_time)
                        if other_overhead < 0 and abs(other_overhead) < 0.05:
                            other_overhead = 0.0

                        print(f"   ✅ Stage 3b: Enhanced Canonical Grouping completed in {stage_duration:.2f}s")
                        print(f"      Breakdown (stage parts):")
                        print(f"       grouping (5 checks): {grouping_time:.2f}s")
                        print(f"       files saving: {save_time:.2f}s")
                        print(f"       other overheads: {other_overhead:.2f}s")
                        if verbose:
                            print(f"      Merge stats: {num_merged} groups merged from multiple tracklets")
                    else:
                        print(f"   ✅ Stage 3b: Enhanced Canonical Grouping completed in {stage_duration:.2f}s")

            # Stage 3c: Person Ranking
            if stage_key == 'stage3c':
                primary_file = config['stage3c_rank']['output'].get('primary_person_file')
                if primary_file:
                    sidecar_path = Path(primary_file).parent / (Path(primary_file).name + '.timings.json')
                    if sidecar_path.exists():
                        with open(sidecar_path, 'r', encoding='utf-8') as sf:
                            data = json.load(sf)

                        ranking_time = float(data.get('ranking_time', 0.0))
                        save_time = float(data.get('npz_save_time', 0.0))
                        num_persons = int(data.get('num_persons', 0))
                        primary_id = int(data.get('primary_person_id', -1))

                        other_overhead = stage_duration - (ranking_time + save_time)
                        if other_overhead < 0 and abs(other_overhead) < 0.05:
                            other_overhead = 0.0

                        print(f"   ✅ Stage 3c: Person Ranking completed in {stage_duration:.2f}s")
                        print(f"      Breakdown (stage parts):")
                        print(f"       ranking: {ranking_time:.2f}s")
                        print(f"       files saving: {save_time:.2f}s")
                        print(f"       other overheads: {other_overhead:.2f}s")
                        if verbose:
                            print(f"      Selected primary person: {primary_id} from {num_persons} candidates")
                    else:
                        print(f"   ✅ Stage 3c: Person Ranking completed in {stage_duration:.2f}s")

            # Stage 4: Generate HTML Viewer (loads from final_crops.pkl - Phase 5)
            if stage_key == 'stage4':
                output_dir = config.get('global', {}).get('output_dir', '')
                sidecar_path = Path(output_dir) / 'stage4.timings.json'
                if sidecar_path.exists():
                    with open(sidecar_path, 'r', encoding='utf-8') as sf:
                        data = json.load(sf)

                        webp_time = float(data.get('webp_time', 0.0))
                        clustering_time = float(data.get('clustering_time', 0.0))
                        num_persons = int(data.get('num_persons', 0))
                        total_crops = int(data.get('total_crops', 0))

                        other_overhead = stage_duration - webp_time - clustering_time
                        if other_overhead < 0 and abs(other_overhead) < 0.05:
                            other_overhead = 0.0

                        print(f"  ✅ Stage 4: Generate HTML Viewer completed in {stage_duration:.2f}s")
                        print(f"      Breakdown (stage parts):")
                        print(f"       webp generation: {webp_time:.2f}s")
                        if clustering_time > 0:
                            print(f"       clustering: {clustering_time:.2f}s")
                        print(f"       other overheads: {other_overhead:.2f}s")
                        if verbose:
                            print(f"      Processed {num_persons} persons, {total_crops} total crops")
                        print(f"{'='*70}")
                else:
                    print(f"  ✅ Stage 4: Generate HTML Viewer completed in {stage_duration:.2f}s")
                    print(f"{'='*70}")

        except Exception:
            if verbose:
                print("     ⚠️  Failed to read timings sidecar")
        
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
    
    # Show output files
    print(f"\n📦 Output Files:")
    
    stage_outputs = {
        'stage0': [
            config.get('stage0_normalize', {}).get('output', {}).get('canonical_video_file', 'N/A'),
            config.get('stage0_normalize', {}).get('output', {}).get('timing_file', 'N/A')
        ],
        'stage1': config['stage1_detect']['output']['detections_file'],
        'stage2': config['stage2_track']['output']['tracklets_file'],
        'stage3a': config.get('stage3a_analyze', {}).get('output', {}).get('tracklet_stats_file', 'N/A'),
        'stage3b': [
            config.get('stage3b_group', {}).get('output', {}).get('canonical_persons_file', 'N/A'),
            config.get('stage3b_group', {}).get('output', {}).get('grouping_log_file', 'N/A')
        ],
        'stage3c': [
            config.get('stage3c_rank', {}).get('output', {}).get('primary_person_file', 'N/A'),
            config.get('stage3c_rank', {}).get('output', {}).get('ranking_report_file', 'N/A')
        ],
        'stage4': config.get('stage4_generate_html', {}).get('output', {}).get('webp_dir', 'N/A'),
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
