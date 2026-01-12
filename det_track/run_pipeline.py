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
        return False, t_end - t_start

    # For YOLO detection stage, print a streamlined completion line with an extra leading space
    if 'YOLO' in stage_name:
        base = stage_name.split(':')[0]
        print(f"  ✅ {base}:  Detection completed in {t_end - t_start:.2f}s")
    # For ByteTrack (stage 2) we avoid printing the per-stage time here because
    # the stage script prints a compact, reconciled breakdown itself.
    elif 'ByteTrack' in stage_name or 'TRACKING' in stage_name.upper():
        print(f"  ✅ {stage_name} completed")
    # For Stage 3 and 4, completion message will be printed after reading sidecar
    elif 'Stage 3' in stage_name or 'Tracklet Analysis' in stage_name:
        pass  # Will print completion with breakdown below
    elif 'Stage 4' in stage_name or 'Load Crops Cache' in stage_name:
        pass  # Will print completion with breakdown below
    else:
        print(f"✅ {stage_name} completed in {t_end - t_start:.2f}s")
    
    return True, t_end - t_start


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
        success, _ = run_stage(stage_name, str(script_path), config_path, verbose)
        stage_end = time.time()
        stage_duration = stage_end - stage_start
        
        stage_times.append((stage_name, stage_duration, False))  # Not skipped
        
        # Try to read the timings sidecar for stages that emit one and report overhead
        try:
            import json
            # Stage 1: YOLO detection (already supported)
            if stage_key == 'stage1':
                detections_file = config['stage1']['output'].get('detections_file')
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

            # Stage 3: Analysis
            if stage_key == 'stage3':
                stats_file = config['stage3']['output'].get('tracklet_stats_file')
                if stats_file:
                    sidecar_path = Path(stats_file).parent / (Path(stats_file).name + '.timings.json')
                    if sidecar_path.exists():
                        with open(sidecar_path, 'r', encoding='utf-8') as sf:
                            data = json.load(sf)

                        stats_time = float(data.get('stats_time', 0.0))
                        npz_save_time = float(data.get('npz_save_time', 0.0))
                        candidate_time = float(data.get('candidate_id_time', 0.0))
                        candidate_save = float(data.get('candidate_save_time', 0.0))
                        num_tracklets = int(data.get('num_tracklets', 0))

                        other_overhead = stage_duration - (stats_time + npz_save_time + candidate_time + candidate_save)
                        if other_overhead < 0 and abs(other_overhead) < 0.05:
                            other_overhead = 0.0

                        # Print completion message with 3-space indent and breakdown
                        print(f"   ✅ Stage 3: Tracklet Analysis completed in {stage_duration:.2f}s")
                        print(f"      Breakdown (stage parts):")
                        print(f"       compute stats: {stats_time:.2f}s")
                        print(f"       stats save: {npz_save_time:.2f}s")
                        print(f"       candidate id: {candidate_time:.2f}s")
                        print(f"       candidate save: {candidate_save:.2f}s")
                        print(f"       other overheads: {other_overhead:.2f}s")
                    else:
                        if verbose:
                            print(f"     (No timings sidecar found at {sidecar_path.name})")
                        else:
                            print(f"   ✅ Stage 3: Tracklet Analysis completed in {stage_duration:.2f}s")

            # Stage 4: Load crops cache
            if stage_key == 'stage4':
                crops_file = config['stage4']['input'].get('crops_cache_file')
                if crops_file:
                    sidecar_path = Path(crops_file).parent / (Path(crops_file).name + '.timings.json')
                    if sidecar_path.exists():
                        with open(sidecar_path, 'r', encoding='utf-8') as sf:
                            data = json.load(sf)

                        load_time = float(data.get('load_time', 0.0))
                        num_frames_sidecar = int(data.get('num_frames', 0))

                        other_overhead = stage_duration - load_time
                        if other_overhead < 0 and abs(other_overhead) < 0.05:
                            other_overhead = 0.0

                        # Print completion message with 3-space indent and breakdown
                        print(f"   ✅ Stage 4: Load Crops Cache completed in {stage_duration:.2f}s")
                        print(f"      Breakdown (stage parts):")
                        print(f"       cache load: {load_time:.2f}s")
                        print(f"       other overheads: {other_overhead:.2f}s")
                    else:
                        if verbose:
                            print(f"     (No timings sidecar found at {sidecar_path.name})")
                        else:
                            print(f"   ✅ Stage 4: Load Crops Cache completed in {stage_duration:.2f}s")

            # Stage 5: Canonical grouping
            if stage_key == 'stage5':
                canonical_file = config['stage5']['output'].get('canonical_persons_file')
                if canonical_file:
                    sidecar_path = Path(canonical_file).parent / (Path(canonical_file).name + '.timings.json')
                    if sidecar_path.exists():
                        with open(sidecar_path, 'r', encoding='utf-8') as sf:
                            data = json.load(sf)

                        grouping_time = float(data.get('grouping_time', 0.0))
                        save_time = float(data.get('npz_save_time', 0.0))
                        num_persons = int(data.get('num_persons', 0))

                        other_overhead = stage_duration - (grouping_time + save_time)
                        if other_overhead < 0 and abs(other_overhead) < 0.05:
                            other_overhead = 0.0

                        print(f"     Breakdown (stage parts):")
                        print(f"       grouping: {grouping_time:.2f}s")
                        print(f"       files saving: {save_time:.2f}s")
                        print(f"       other overheads: {other_overhead:.2f}s")
                        print(f"     Output persons: {num_persons}")
                    else:
                        if verbose:
                            print(f"     (No timings sidecar found at {sidecar_path.name})")
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
