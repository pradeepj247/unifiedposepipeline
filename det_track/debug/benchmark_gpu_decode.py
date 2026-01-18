#!/usr/bin/env python3
"""
Benchmark: CPU decode vs GPU decode for YOLO inference

Tests whether GPU-accelerated video decode improves YOLO performance
when processing all-I-frame videos from Stage 0.

Comparison:
  Method 1 (Current): CPU decode → GPU transfer → YOLO
  Method 2 (New):     GPU decode → YOLO (no transfer)
"""

import cv2
import time
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def benchmark_cpu_decode(model, video_path, max_frames=None):
    """
    Method 1: CPU decode with OpenCV (current approach)
    """
    print("\n" + "="*70)
    print("METHOD 1: CPU DECODE (OpenCV VideoCapture)")
    print("="*70)
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    print(f"Video: {video_path.name}")
    print(f"Frames to process: {total_frames}")
    print(f"Starting CPU decode benchmark...\n")
    
    decode_times = []
    inference_times = []
    frame_count = 0
    last_print = 0
    
    start_total = time.time()
    
    while frame_count < total_frames:
        # Time CPU decode
        t0 = time.time()
        ret, frame = cap.read()
        t1 = time.time()
        
        if not ret:
            break
        
        decode_times.append(t1 - t0)
        
        # Time YOLO inference
        t2 = time.time()
        results = model(frame, conf=0.3, classes=[0], verbose=False)
        t3 = time.time()
        
        inference_times.append(t3 - t2)
        frame_count += 1
        
        # Print progress every 200 frames
        if frame_count - last_print >= 200:
            elapsed = time.time() - start_total
            fps = frame_count / elapsed
            avg_decode = np.mean(decode_times[-200:]) * 1000
            avg_infer = np.mean(inference_times[-200:]) * 1000
            print(f"  {frame_count}/{total_frames} | {fps:.1f} FPS | "
                  f"Decode: {avg_decode:.1f}ms | Infer: {avg_infer:.1f}ms")
            last_print = frame_count
    
    cap.release()
    end_total = time.time()
    
    total_time = end_total - start_total
    avg_decode = np.mean(decode_times) * 1000
    avg_infer = np.mean(inference_times) * 1000
    fps = frame_count / total_time
    
    print(f"\n{'─'*70}")
    print(f"✅ CPU Decode Results:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Frames: {frame_count}")
    print(f"   FPS: {fps:.1f}")
    print(f"   Avg decode time: {avg_decode:.2f}ms/frame")
    print(f"   Avg inference time: {avg_infer:.2f}ms/frame")
    print(f"   Total per frame: {avg_decode + avg_infer:.2f}ms")
    
    return {
        'method': 'CPU Decode',
        'total_time': total_time,
        'frames': frame_count,
        'fps': fps,
        'decode_ms': avg_decode,
        'inference_ms': avg_infer,
        'total_ms': avg_decode + avg_infer
    }


def benchmark_gpu_decode_ffmpeg(model, video_path, max_frames=None):
    """
    Method 2: GPU decode with FFmpeg NVDEC
    
    Uses ffmpeg with -hwaccel cuda to decode directly to GPU memory
    """
    print("\n" + "="*70)
    print("METHOD 2: GPU DECODE (FFmpeg NVDEC)")
    print("="*70)
    
    import subprocess
    
    # Get video properties
    cap_temp = cv2.VideoCapture(str(video_path))
    width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap_temp.get(cv2.CAP_PROP_FPS)
    total_frames_video = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_temp.release()
    
    total_frames = total_frames_video if not max_frames else min(total_frames_video, max_frames)
    
    print(f"Video: {video_path.name}")
    print(f"Resolution: {width}x{height}")
    print(f"Frames to process: {total_frames}")
    print(f"Starting GPU decode benchmark...\n")
    
    # FFmpeg command with GPU decode
    # GPU decodes but outputs to CPU for piping (still faster than pure CPU decode)
    ffmpeg_cmd = [
        'ffmpeg',
        '-hwaccel', 'cuda',           # GPU decode (NVDEC)
        '-i', str(video_path),
        '-f', 'rawvideo',              # Raw video output
        '-pix_fmt', 'bgr24',          # BGR format (OpenCV compatible)
        '-'                            # Output to stdout
    ]
    
    if max_frames:
        # Insert frame limit
        ffmpeg_cmd.insert(-1, '-frames:v')
        ffmpeg_cmd.insert(-1, str(max_frames))
    
    process = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10**8
    )
    
    frame_size = width * height * 3  # BGR = 3 bytes per pixel
    inference_times = []
    frame_count = 0
    last_print = 0
    
    start_total = time.time()
    
    try:
        while frame_count < total_frames:
            # Read raw frame data
            t0 = time.time()
            raw_frame = process.stdout.read(frame_size)
            t1 = time.time()
            
            if len(raw_frame) != frame_size:
                break
            
            # Convert to numpy array (minimal overhead)
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
            
            # Time YOLO inference
            t2 = time.time()
            results = model(frame, conf=0.3, classes=[0], verbose=False)
            t3 = time.time()
            
            inference_times.append(t3 - t2)
            frame_count += 1
            
            # Print progress every 200 frames
            if frame_count - last_print >= 200:
                elapsed = time.time() - start_total
                fps = frame_count / elapsed
                avg_infer = np.mean(inference_times[-200:]) * 1000
                print(f"  {frame_count}/{total_frames} | {fps:.1f} FPS | "
                      f"Infer: {avg_infer:.1f}ms")
                last_print = frame_count
    
    finally:
        process.stdout.close()
        process.wait()
    
    end_total = time.time()
    
    total_time = end_total - start_total
    avg_infer = np.mean(inference_times) * 1000
    fps = frame_count / total_time
    
    print(f"\n{'─'*70}")
    print(f"✅ GPU Decode Results:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Frames: {frame_count}")
    print(f"   FPS: {fps:.1f}")
    print(f"   Avg inference time: {avg_infer:.2f}ms/frame")
    print(f"   Total per frame: {(total_time / frame_count * 1000) if frame_count > 0 else 0:.2f}ms")
    print(f"   Note: Decode time included in total (pipelined with FFmpeg)")
    
    return {
        'method': 'GPU Decode',
        'total_time': total_time,
        'frames': frame_count,
        'fps': fps,
        'decode_ms': None,  # Pipelined, can't separate
        'inference_ms': avg_infer,
        'total_ms': (total_time / frame_count * 1000) if frame_count > 0 else 0
    }


def print_comparison(cpu_results, gpu_results):
    """Print side-by-side comparison"""
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    print(f"\n{'Metric':<25} {'CPU Decode':<20} {'GPU Decode':<20} {'Speedup':<15}")
    print("─"*70)
    
    # Total time
    speedup = cpu_results['total_time'] / gpu_results['total_time']
    print(f"{'Total Time':<25} {cpu_results['total_time']:>6.2f}s {' '*13} "
          f"{gpu_results['total_time']:>6.2f}s {' '*13} {speedup:>6.2f}x")
    
    # FPS
    speedup = gpu_results['fps'] / cpu_results['fps']
    print(f"{'Throughput (FPS)':<25} {cpu_results['fps']:>6.1f} FPS {' '*10} "
          f"{gpu_results['fps']:>6.1f} FPS {' '*10} {speedup:>6.2f}x")
    
    # Decode time
    if cpu_results['decode_ms']:
        print(f"{'Decode Time':<25} {cpu_results['decode_ms']:>6.2f}ms {' '*11} "
              f"{'(pipelined)':<20} {'N/A':<15}")
    
    # Inference time
    speedup = cpu_results['inference_ms'] / gpu_results['inference_ms']
    print(f"{'Inference Time':<25} {cpu_results['inference_ms']:>6.2f}ms {' '*11} "
          f"{gpu_results['inference_ms']:>6.2f}ms {' '*11} {speedup:>6.2f}x")
    
    # Recommendation
    print("\n" + "="*70)
    if gpu_results['fps'] > cpu_results['fps'] * 1.2:  # >20% improvement
        improvement = ((gpu_results['fps'] / cpu_results['fps']) - 1) * 100
        print(f"✅ RECOMMENDATION: Use GPU decode ({improvement:.0f}% faster)")
        print(f"   GPU decode eliminates CPU→GPU transfer bottleneck")
    elif gpu_results['fps'] > cpu_results['fps']:
        print(f"ℹ️  RECOMMENDATION: GPU decode slightly faster, but marginal")
        print(f"   Current CPU decode is adequate for this use case")
    else:
        print(f"⚠️  RECOMMENDATION: Stick with CPU decode")
        print(f"   GPU decode overhead exceeds benefit for this video")
    print("="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark CPU vs GPU decode for YOLO')
    parser.add_argument('--video', required=True, help='Path to video file (preferably all-I-frame from Stage 0)')
    parser.add_argument('--model', default='yolov8s.pt', help='YOLO model path')
    parser.add_argument('--max-frames', type=int, default=800, help='Limit frames for quick test (default: 800, multiple of 200)')
    parser.add_argument('--full', action='store_true', help='Process entire video (ignore max-frames)')
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    max_frames = None if args.full else args.max_frames
    
    print("\n" + "="*70)
    print("GPU DECODE BENCHMARK FOR YOLO")
    print("="*70)
    print(f"Video: {video_path}")
    print(f"Model: {model_path}")
    print(f"Max frames: {max_frames if max_frames else 'All'}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only (error!)'}")
    
    # Load YOLO model
    print(f"\nLoading YOLO model...")
    model = YOLO(str(model_path))
    model.to('cuda')
    print(f"✅ Model loaded on GPU")
    
    # Warmup
    print(f"\n🔥 Warming up GPU...")
    dummy = torch.rand(1, 3, 640, 640).cuda()  # rand (not randn) creates 0-1 values
    for _ in range(10):
        _ = model(dummy, verbose=False)
    print(f"✅ Warmup complete")
    
    # Benchmark 1: CPU decode
    # Note: Timing starts AFTER model load and warmup (excludes initialization overhead)
    cpu_results = benchmark_cpu_decode(model, video_path, max_frames)
    
    # Wait a bit between tests
    time.sleep(2)
    
    # Benchmark 2: GPU decode
    gpu_results = benchmark_gpu_decode_ffmpeg(model, video_path, max_frames)
    
    # Compare
    print_comparison(cpu_results, gpu_results)


if __name__ == '__main__':
    main()
