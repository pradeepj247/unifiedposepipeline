"""
Benchmark RTMPose COCO (17 kpts) vs Halpe26 (26 kpts)

Compares:
- Processing speed (FPS)
- Accuracy metrics
- Memory usage
- Keypoint detection quality

Usage:
    python benchmark_halpe26.py --video demo_data/videos/dance.mp4 --frames 360
"""

import argparse
import time
from pathlib import Path
import subprocess
import numpy as np
import yaml


def run_pipeline(config_file, description):
    """Run UDP video pipeline and extract performance metrics"""
    print(f"\n{'='*80}")
    print(f"🏃 Running: {description}")
    print(f"{'='*80}\n")
    
    cmd = f"python udp_video.py --config {config_file}"
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    total_time = time.time() - start_time
    
    if result.returncode != 0:
        print(f"❌ Error running {description}")
        print(result.stderr)
        return None
    
    # Parse output for metrics
    output = result.stdout
    print(output)
    
    # Extract metrics from output
    metrics = {
        'description': description,
        'total_time': total_time,
        'output': output
    }
    
    # Parse stage times and FPS
    for line in output.split('\n'):
        if 'Stage 1 (Detection):' in line:
            parts = line.split()
            metrics['stage1_time'] = float(parts[3].rstrip('s'))
            metrics['stage1_fps'] = float(parts[5])
        elif 'Stage 2 (Pose):' in line:
            parts = line.split()
            metrics['stage2_time'] = float(parts[3].rstrip('s'))
            metrics['stage2_fps'] = float(parts[5])
        elif 'Stage 3 (Visualization):' in line:
            parts = line.split()
            metrics['stage3_time'] = float(parts[3].rstrip('s'))
    
    return metrics


def load_and_analyze_keypoints(npz_path, description):
    """Load keypoints and compute statistics"""
    print(f"\n📊 Analyzing keypoints: {description}")
    
    data = np.load(npz_path)
    keypoints = data['keypoints']
    scores = data['scores']
    
    num_frames, num_kpts, _ = keypoints.shape
    
    # Compute statistics
    valid_detections = np.sum(scores[:, 0] > 0)
    avg_confidence = np.mean(scores[scores > 0]) if np.any(scores > 0) else 0
    
    # Per-joint average confidence
    per_joint_conf = []
    for j in range(num_kpts):
        valid_scores = scores[:, j][scores[:, j] > 0]
        if len(valid_scores) > 0:
            per_joint_conf.append(np.mean(valid_scores))
        else:
            per_joint_conf.append(0.0)
    
    print(f"   Frames: {num_frames}")
    print(f"   Keypoints per frame: {num_kpts}")
    print(f"   Valid detections: {valid_detections}/{num_frames} ({100*valid_detections/num_frames:.1f}%)")
    print(f"   Average confidence: {avg_confidence:.3f}")
    print(f"   Min joint confidence: {np.min(per_joint_conf):.3f}")
    print(f"   Max joint confidence: {np.max(per_joint_conf):.3f}")
    
    if num_kpts == 26:
        print(f"\n   Body keypoints (0-16) avg: {np.mean(per_joint_conf[:17]):.3f}")
        print(f"   Feet keypoints (17-22) avg: {np.mean(per_joint_conf[17:23]):.3f}")
        print(f"   Extra keypoints (23-25) avg: {np.mean(per_joint_conf[23:]):.3f}")
    
    return {
        'num_frames': num_frames,
        'num_keypoints': num_kpts,
        'valid_detections': valid_detections,
        'avg_confidence': avg_confidence,
        'per_joint_conf': per_joint_conf
    }


def print_comparison(coco_metrics, halpe_metrics, coco_kpts, halpe_kpts):
    """Print detailed comparison"""
    print("\n" + "="*80)
    print("📊 BENCHMARK COMPARISON: COCO-17 vs Halpe26")
    print("="*80)
    
    print("\n⏱️  PERFORMANCE:")
    print(f"{'Metric':<30} {'COCO-17':<20} {'Halpe26':<20} {'Difference'}")
    print("-" * 80)
    
    if 'stage2_time' in coco_metrics and 'stage2_time' in halpe_metrics:
        coco_time = coco_metrics['stage2_time']
        halpe_time = halpe_metrics['stage2_time']
        diff = halpe_time - coco_time
        print(f"{'Pose Estimation Time':<30} {coco_time:<20.2f}s {halpe_time:<20.2f}s {diff:+.2f}s")
        
        coco_fps = coco_metrics['stage2_fps']
        halpe_fps = halpe_metrics['stage2_fps']
        diff_fps = halpe_fps - coco_fps
        print(f"{'Pose Estimation FPS':<30} {coco_fps:<20.1f} {halpe_fps:<20.1f} {diff_fps:+.1f}")
        
        # FPS percentage difference
        fps_percent = 100 * (halpe_fps - coco_fps) / coco_fps
        print(f"{'FPS Change':<30} {'':<20} {'':<20} {fps_percent:+.1f}%")
    
    print("\n🎯 ACCURACY:")
    print(f"{'Metric':<30} {'COCO-17':<20} {'Halpe26':<20} {'Difference'}")
    print("-" * 80)
    
    coco_conf = coco_kpts['avg_confidence']
    halpe_conf = halpe_kpts['avg_confidence']
    print(f"{'Average Confidence':<30} {coco_conf:<20.3f} {halpe_conf:<20.3f} {halpe_conf-coco_conf:+.3f}")
    
    coco_valid = coco_kpts['valid_detections']
    halpe_valid = halpe_kpts['valid_detections']
    total_frames = coco_kpts['num_frames']
    print(f"{'Valid Detections':<30} {coco_valid}/{total_frames:<14} {halpe_valid}/{total_frames:<14} {halpe_valid-coco_valid:+d}")
    
    print("\n📐 ADDITIONAL INFORMATION:")
    print(f"   Halpe26 provides {halpe_kpts['num_keypoints'] - coco_kpts['num_keypoints']} extra keypoints:")
    print(f"      - 6 foot keypoints (17-22): 3 per foot")
    print(f"      - 3 body keypoints (23-25): neck, chest, pelvis")
    
    if len(halpe_kpts['per_joint_conf']) == 26:
        feet_conf = np.mean(halpe_kpts['per_joint_conf'][17:23])
        extra_conf = np.mean(halpe_kpts['per_joint_conf'][23:])
        print(f"   Feet keypoints average confidence: {feet_conf:.3f}")
        print(f"   Extra body keypoints average confidence: {extra_conf:.3f}")
    
    print("\n💡 RECOMMENDATION:")
    if halpe_fps >= coco_fps * 0.95:  # Less than 5% slower
        print("   ✅ Halpe26 provides 9 extra keypoints with minimal speed impact")
        print("   ✅ Recommended if you need foot and detailed torso tracking")
    elif halpe_fps >= coco_fps * 0.85:  # 5-15% slower
        print("   ⚠️  Halpe26 is slightly slower but provides valuable extra keypoints")
        print("   💡 Consider using Halpe26 if foot/torso tracking is important")
    else:
        print("   ⚠️  Halpe26 has noticeable speed impact")
        print("   💡 Use COCO-17 if speed is critical, Halpe26 for detailed tracking")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark COCO-17 vs Halpe26")
    parser.add_argument("--video", type=str, help="Video path (optional, uses config default)")
    parser.add_argument("--frames", type=int, help="Max frames (optional, uses config default)")
    args = parser.parse_args()
    
    # Update configs if arguments provided
    if args.video or args.frames:
        for config_file in ['configs/udp_video.yaml', 'configs/udp_video_halpe26.yaml']:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            if args.video:
                config['video']['input_path'] = args.video
            if args.frames:
                config['video']['max_frames'] = args.frames
            
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
    
    print("\n" + "🔬" * 40)
    print("RTMPose COCO-17 vs Halpe26 Benchmark")
    print("🔬" * 40)
    
    # Run COCO-17
    coco_metrics = run_pipeline(
        "configs/udp_video.yaml",
        "RTMPose COCO-17 (17 keypoints)"
    )
    
    if coco_metrics is None:
        print("❌ COCO-17 benchmark failed")
        return 1
    
    # Run Halpe26
    halpe_metrics = run_pipeline(
        "configs/udp_video_halpe26.yaml",
        "RTMPose Halpe26 (26 keypoints)"
    )
    
    if halpe_metrics is None:
        print("❌ Halpe26 benchmark failed")
        return 1
    
    # Analyze keypoints
    coco_kpts = load_and_analyze_keypoints(
        "demo_data/outputs/keypoints.npz",
        "COCO-17"
    )
    
    halpe_kpts = load_and_analyze_keypoints(
        "demo_data/outputs/keypoints_halpe26.npz",
        "Halpe26"
    )
    
    # Print comparison
    print_comparison(coco_metrics, halpe_metrics, coco_kpts, halpe_kpts)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
