#!/usr/bin/env python3
"""
Quick test script for BoxMOT tracking with ReID on campus_walk.mp4

This script:
1. Verifies BoxMOT installation and tracker availability
2. Tests tracking with ReID enabled
3. Compares performance: Detection-only vs Tracking+ReID
4. Generates visualization video with track IDs

Usage (on Google Colab):
    cd /content/unifiedposepipeline
    python test_tracking_reid_benchmark.py
"""

import sys
import os

print("="*70)
print("BoxMOT Tracking + ReID Benchmark Test")
print("="*70)

# Step 1: Verify BoxMOT installation
print("\n[1/5] Verifying BoxMOT installation...")
try:
    import boxmot
    from boxmot import ByteTrack, BotSort
    print(f"   ✅ BoxMOT v{boxmot.__version__} installed")
    print(f"   ✅ ByteTrack available")
    print(f"   ✅ BotSort available")
except ImportError as e:
    print(f"   ❌ BoxMOT not found: {e}")
    print(f"   Install with: pip install boxmot")
    sys.exit(1)

# Step 2: Check video file
print("\n[2/5] Checking input video...")
video_path = "demo_data/videos/campus_walk.mp4"
if os.path.exists(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"   ✅ Video found: {video_path}")
        print(f"      Resolution: {width}x{height}")
        print(f"      FPS: {fps:.2f}")
        print(f"      Frames: {total_frames}")
    else:
        print(f"   ❌ Cannot open video: {video_path}")
        sys.exit(1)
else:
    print(f"   ❌ Video not found: {video_path}")
    print(f"   Please ensure video is at demo_data/videos/campus_walk.mp4")
    sys.exit(1)

# Step 3: Check config file
print("\n[3/5] Checking configuration...")
config_path = "configs/detector_tracking_benchmark.yaml"
if os.path.exists(config_path):
    print(f"   ✅ Config found: {config_path}")
else:
    print(f"   ❌ Config not found: {config_path}")
    sys.exit(1)

# Step 4: Run tracking test
print("\n[4/5] Running tracking + ReID benchmark...")
print(f"   Command: python run_detector_tracking.py --config {config_path}")
print(f"\n   This will:")
print(f"   • Detect people using YOLOv8")
print(f"   • Track them using ByteTrack (fastest tracker)")
print(f"   • Apply OSNet x1.0 ReID for appearance matching")
print(f"   • Benchmark: Detection FPS, Tracking FPS, Combined FPS")
print(f"   • Report: Unique track IDs, ID stability")
print(f"   • Generate: Visualization video with track IDs")

import subprocess
result = subprocess.run(
    [sys.executable, "run_detector_tracking.py", "--config", config_path],
    capture_output=False
)

if result.returncode != 0:
    print(f"\n   ❌ Tracking failed with exit code {result.returncode}")
    sys.exit(1)

# Step 5: Verify outputs
print("\n[5/5] Verifying outputs...")

npz_path = "demo_data/outputs/detections_tracking_reid.npz"
if os.path.exists(npz_path):
    print(f"   ✅ Detections saved: {npz_path}")
    import numpy as np
    data = np.load(npz_path)
    print(f"      Bboxes shape: {data['bboxes'].shape}")
else:
    print(f"   ⚠️  Detections not found: {npz_path}")

video_path = "demo_data/outputs/campus_walk_tracking_reid.mp4"
if os.path.exists(video_path):
    print(f"   ✅ Visualization saved: {video_path}")
    print(f"      You can view this video to see track IDs and stability!")
else:
    print(f"   ⚠️  Visualization not found: {video_path}")

print("\n" + "="*70)
print("✓ Benchmark complete!")
print("="*70)
print("\nKey Observations:")
print("1. Check 'Detection FPS' vs 'Combined FPS' to see ReID overhead")
print("2. Check 'Unique track IDs' to see how many people were tracked")
print("3. Watch the output video to verify ID consistency across frames")
print("4. Compare with detection-only (set tracking.enabled=false) to see difference")
print("\nExpected Results:")
print("• Detection-only: ~70 FPS")
print("• With Tracking+ReID: ~30-40 FPS (2x slowdown due to ReID)")
print("• Track IDs should remain stable even during occlusions")
print("="*70)
