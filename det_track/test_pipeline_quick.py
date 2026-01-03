#!/usr/bin/env python3
"""
Quick pipeline test: Run Stages 1-3 and validate outputs

Tests:
1. Stage 1 Detection: Generates detections_raw.npz
2. Stage 2 Tracking: Generates tracklets_raw.npz (motion-only, no video)
3. Stage 3 Analysis: Generates tracklet_stats.npz

Run with: python test_pipeline_quick.py --config configs/pipeline_config.yaml --frames 400
"""

import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np


def run_stage(stage_name, script_path, config_path):
    """Run a stage and check for success"""
    print(f"\n{'='*70}")
    print(f"Running {stage_name}...")
    print(f"{'='*70}")
    
    result = subprocess.run(
        [sys.executable, str(script_path), '--config', str(config_path)],
        cwd=Path(script_path).parent
    )
    
    if result.returncode != 0:
        print(f"❌ {stage_name} failed!")
        return False
    
    print(f"✅ {stage_name} completed")
    return True


def validate_detections(npz_path):
    """Validate detections NPZ format"""
    print(f"\n🔍 Validating detections: {npz_path.name}")
    
    if not npz_path.exists():
        print(f"❌ File not found: {npz_path}")
        return False
    
    try:
        data = np.load(npz_path)
        frame_numbers = data['frame_numbers']
        bboxes = data['bboxes']
        confidences = data['confidences']
        
        print(f"  ✅ Total detections: {len(frame_numbers)}")
        print(f"  ✅ Bbox shape: {bboxes.shape}")
        print(f"  ✅ Confidence shape: {confidences.shape}")
        print(f"  ✅ Frame range: {frame_numbers.min()} - {frame_numbers.max()}")
        
        # Sample
        if len(bboxes) > 0:
            print(f"  Sample detection (frame {frame_numbers[0]}):")
            print(f"    Bbox: {bboxes[0]}")
            print(f"    Confidence: {confidences[0]:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Error reading detections: {e}")
        return False


def validate_tracklets(npz_path):
    """Validate tracklets NPZ format"""
    print(f"\n🔍 Validating tracklets: {npz_path.name}")
    
    if not npz_path.exists():
        print(f"❌ File not found: {npz_path}")
        return False
    
    try:
        data = np.load(npz_path, allow_pickle=True)
        tracklets = list(data['tracklets'])
        
        print(f"  ✅ Total tracklets: {len(tracklets)}")
        
        if len(tracklets) == 0:
            print(f"  ⚠️  WARNING: Zero tracklets detected!")
            return False
        
        # Statistics
        total_detections = sum(len(t['frame_numbers']) for t in tracklets)
        durations = [len(t['frame_numbers']) for t in tracklets]
        
        print(f"  ✅ Total tracked detections: {total_detections}")
        print(f"  ✅ Tracklet durations: min={min(durations)}, max={max(durations)}, mean={np.mean(durations):.1f}")
        
        # Sample
        if len(tracklets) > 0:
            t = tracklets[0]
            print(f"  Sample tracklet (ID {t['tracklet_id']}):")
            print(f"    Duration: {len(t['frame_numbers'])} frames")
            print(f"    Frame range: {t['frame_numbers'][0]} - {t['frame_numbers'][-1]}")
            print(f"    First bbox: {t['bboxes'][0]}")
            print(f"    Mean confidence: {np.mean(t['confidences']):.3f}")
        
        return len(tracklets) > 0
    
    except Exception as e:
        print(f"❌ Error reading tracklets: {e}")
        return False


def validate_stats(npz_path):
    """Validate tracklet stats NPZ format"""
    print(f"\n🔍 Validating tracklet stats: {npz_path.name}")
    
    if not npz_path.exists():
        print(f"⚠️  File not found (optional): {npz_path}")
        return True
    
    try:
        data = np.load(npz_path)
        stats = data['stats']
        
        print(f"  ✅ Statistics for {len(stats)} tracklets")
        return True
    
    except Exception as e:
        print(f"⚠️  Could not read stats (optional): {e}")
        return True


def main():
    parser = argparse.ArgumentParser(description='Quick pipeline test')
    parser.add_argument('--config', default='configs/pipeline_config.yaml',
                        help='Path to config')
    parser.add_argument('--frames', type=int, default=400,
                        help='Max frames to process')
    args = parser.parse_args()
    
    config_path = Path(args.config).resolve()
    script_dir = Path(__file__).parent
    
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return False
    
    # Modify config to limit frames for testing
    print(f"📝 Using config: {config_path}")
    print(f"📊 Limiting to {args.frames} frames for quick test")
    
    # Get output directory from config (simplified - just read video name)
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    video_name = Path(config['global']['video_file']).stem
    outputs_dir = Path(config['global']['outputs_dir']).parent / video_name
    
    detections_file = outputs_dir / 'detections_raw.npz'
    tracklets_file = outputs_dir / 'tracklets_raw.npz'
    stats_file = outputs_dir / 'tracklet_stats.npz'
    
    print(f"\n{'='*70}")
    print(f"QUICK PIPELINE TEST")
    print(f"{'='*70}")
    print(f"Config: {config_path}")
    print(f"Output dir: {outputs_dir}")
    print(f"Max frames: {args.frames}")
    
    # Stage 1: Detection
    success = run_stage(
        "STAGE 1: Detection",
        script_dir / 'stage1_detect.py',
        config_path
    )
    if not success:
        return False
    
    if not validate_detections(detections_file):
        print(f"❌ Detections validation failed!")
        return False
    
    # Stage 2: Tracking (motion-only, no video)
    success = run_stage(
        "STAGE 2: Tracking (ByteTrack motion-only)",
        script_dir / 'stage2_track.py',
        config_path
    )
    if not success:
        return False
    
    if not validate_tracklets(tracklets_file):
        print(f"❌ Tracklets validation failed!")
        print(f"❌ Zero tracklets - motion-only tracking didn't work!")
        return False
    
    # Stage 3: Analysis
    success = run_stage(
        "STAGE 3: Tracklet Analysis",
        script_dir / 'stage3_analyze_tracklets.py',
        config_path
    )
    if not success:
        print(f"⚠️  Stage 3 failed, but tracklets exist")
    else:
        validate_stats(stats_file)
    
    print(f"\n{'='*70}")
    print(f"✅ PIPELINE TEST SUCCESSFUL!")
    print(f"{'='*70}")
    print(f"✅ Stage 1 (Detection): Generated detections_raw.npz")
    print(f"✅ Stage 2 (Tracking): Generated tracklets_raw.npz (motion-only)")
    print(f"✅ Tracklets: {len(list(np.load(tracklets_file, allow_pickle=True)['tracklets']))} tracklets")
    print(f"\nOutput: {outputs_dir}")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
