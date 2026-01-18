#!/usr/bin/env python3
"""
Benchmark: Single-frame vs Batch inference for YOLO

Tests whether batched inference (processing multiple frames simultaneously)
improves throughput compared to single-frame processing.

Comparison:
  Method 1: Single-frame inference (batch_size=1) - current approach
  Method 2: Batched inference (batch_size=4) - process 4 frames at once
  Method 3: Batched inference (batch_size=8) - process 8 frames at once
"""

import cv2
import time
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def benchmark_single_frame(model, video_path, max_frames=None):
    """
    Method 1: Single-frame inference (current approach)
    """
    print("\n" + "="*70)
    print("METHOD 1: SINGLE-FRAME INFERENCE (batch_size=1)")
    print("="*70)
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    print(f"Video: {video_path.name}")
    print(f"Frames to process: {total_frames}")
    print(f"Starting single-frame benchmark...\n")
    
    decode_times = []
    inference_times = []
    frame_count = 0
    last_print = 0
    
    start_total = time.time()
    
    while frame_count < total_frames:
        # Decode
        t0 = time.time()
        ret, frame = cap.read()
        t1 = time.time()
        
        if not ret:
            break
        
        decode_times.append(t1 - t0)
        
        # Inference (single frame)
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
    print(f"✅ Single-frame Results:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Frames: {frame_count}")
    print(f"   FPS: {fps:.1f}")
    print(f"   Avg decode time: {avg_decode:.2f}ms/frame")
    print(f"   Avg inference time: {avg_infer:.2f}ms/frame")
    print(f"   Total per frame: {avg_decode + avg_infer:.2f}ms")
    
    return {
        'method': 'Single-frame',
        'batch_size': 1,
        'total_time': total_time,
        'frames': frame_count,
        'fps': fps,
        'decode_ms': avg_decode,
        'inference_ms': avg_infer,
        'total_ms': avg_decode + avg_infer
    }


def benchmark_batch_inference(model, video_path, batch_size=4, max_frames=None):
    """
    Method 2: Batched inference (process multiple frames simultaneously)
    """
    print("\n" + "="*70)
    print(f"METHOD 2: BATCHED INFERENCE (batch_size={batch_size})")
    print("="*70)
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    print(f"Video: {video_path.name}")
    print(f"Frames to process: {total_frames}")
    print(f"Batch size: {batch_size}")
    print(f"Starting batched inference benchmark...\n")
    
    decode_times = []
    inference_times = []
    frame_count = 0
    last_print = 0
    
    start_total = time.time()
    
    # Read and accumulate frames into batches
    frame_buffer = []
    
    while frame_count < total_frames:
        # Decode frames for batch
        t0 = time.time()
        for _ in range(batch_size):
            if frame_count + len(frame_buffer) >= total_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            frame_buffer.append(frame)
        t1 = time.time()
        
        if not frame_buffer:
            break
        
        batch_decode_time = t1 - t0
        decode_times.extend([batch_decode_time / len(frame_buffer)] * len(frame_buffer))
        
        # Batch inference
        t2 = time.time()
        results = model(frame_buffer, conf=0.3, classes=[0], verbose=False)
        t3 = time.time()
        
        batch_infer_time = t3 - t2
        # Amortize batch inference time across all frames in batch
        per_frame_infer_time = batch_infer_time / len(frame_buffer)
        inference_times.extend([per_frame_infer_time] * len(frame_buffer))
        
        frame_count += len(frame_buffer)
        frame_buffer = []  # Clear buffer
        
        # Print progress every ~200 frames
        if frame_count - last_print >= 200:
            elapsed = time.time() - start_total
            fps = frame_count / elapsed
            avg_decode = np.mean(decode_times[-200:]) * 1000
            avg_infer = np.mean(inference_times[-200:]) * 1000
            print(f"  {frame_count}/{total_frames} | {fps:.1f} FPS | "
                  f"Decode: {avg_decode:.1f}ms | Infer: {avg_infer:.1f}ms/frame")
            last_print = frame_count
    
    cap.release()
    end_total = time.time()
    
    total_time = end_total - start_total
    avg_decode = np.mean(decode_times) * 1000
    avg_infer = np.mean(inference_times) * 1000
    fps = frame_count / total_time
    
    print(f"\n{'─'*70}")
    print(f"✅ Batched Results (batch_size={batch_size}):")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Frames: {frame_count}")
    print(f"   FPS: {fps:.1f}")
    print(f"   Avg decode time: {avg_decode:.2f}ms/frame")
    print(f"   Avg inference time: {avg_infer:.2f}ms/frame (amortized)")
    print(f"   Total per frame: {avg_decode + avg_infer:.2f}ms")
    
    return {
        'method': f'Batch-{batch_size}',
        'batch_size': batch_size,
        'total_time': total_time,
        'frames': frame_count,
        'fps': fps,
        'decode_ms': avg_decode,
        'inference_ms': avg_infer,
        'total_ms': avg_decode + avg_infer
    }


def print_comparison(results_list):
    """Print side-by-side comparison of all methods"""
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    baseline = results_list[0]
    
    print(f"\n{'Method':<20} {'Batch':<8} {'Total Time':<12} {'FPS':<10} {'Speedup':<10}")
    print("─"*70)
    
    for result in results_list:
        speedup = result['fps'] / baseline['fps']
        print(f"{result['method']:<20} {result['batch_size']:<8} "
              f"{result['total_time']:>6.2f}s      "
              f"{result['fps']:>6.1f}     "
              f"{speedup:>6.2f}x")
    
    print("\n" + "─"*70)
    print("Per-frame breakdown:")
    print(f"{'Method':<20} {'Decode':<12} {'Inference':<12} {'Total':<12}")
    print("─"*70)
    
    for result in results_list:
        print(f"{result['method']:<20} "
              f"{result['decode_ms']:>6.2f}ms     "
              f"{result['inference_ms']:>6.2f}ms     "
              f"{result['total_ms']:>6.2f}ms")
    
    # Find best method
    best = max(results_list, key=lambda x: x['fps'])
    best_speedup = best['fps'] / baseline['fps']
    
    print("\n" + "="*70)
    if best_speedup > 1.15:  # >15% improvement
        improvement = ((best['fps'] / baseline['fps']) - 1) * 100
        print(f"✅ RECOMMENDATION: Use {best['method']} ({improvement:.0f}% faster)")
        print(f"   Batch processing improves GPU utilization")
        print(f"   Inference time: {baseline['inference_ms']:.2f}ms → {best['inference_ms']:.2f}ms per frame")
    elif best_speedup > 1.05:
        print(f"ℹ️  RECOMMENDATION: {best['method']} slightly faster, but marginal")
        print(f"   Batching helps but may add complexity")
    else:
        print(f"⚠️  RECOMMENDATION: Stick with single-frame processing")
        print(f"   Batching doesn't improve performance for this use case")
    print("="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark single-frame vs batch inference for YOLO')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--model', default='yolov8s.pt', help='YOLO model path')
    parser.add_argument('--max-frames', type=int, default=500, help='Limit frames for quick test (default: 500)')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[4, 8], 
                        help='Batch sizes to test (default: 4 8)')
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
    print("BATCH INFERENCE BENCHMARK FOR YOLO")
    print("="*70)
    print(f"Video: {video_path}")
    print(f"Model: {model_path}")
    print(f"Max frames: {max_frames if max_frames else 'All'}")
    print(f"Batch sizes to test: 1 (baseline), {', '.join(map(str, args.batch_sizes))}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only (error!)'}")
    
    # Load YOLO model
    print(f"\nLoading YOLO model...")
    model = YOLO(str(model_path))
    model.to('cuda')
    print(f"✅ Model loaded on GPU")
    
    # Warmup
    print(f"\n🔥 Warming up GPU...")
    dummy = torch.randn(1, 3, 640, 640).cuda()
    for _ in range(10):
        _ = model(dummy, verbose=False)
    print(f"✅ Warmup complete")
    
    results_list = []
    
    # Benchmark 1: Single-frame (baseline)
    single_results = benchmark_single_frame(model, video_path, max_frames)
    results_list.append(single_results)
    
    # Benchmark 2+: Batched inference
    for batch_size in args.batch_sizes:
        time.sleep(2)  # Cool down between tests
        batch_results = benchmark_batch_inference(model, video_path, batch_size, max_frames)
        results_list.append(batch_results)
    
    # Compare
    print_comparison(results_list)


if __name__ == '__main__':
    main()
