#!/usr/bin/env python3
"""
Benchmark: Compare different YOLO models (v8n vs v8s vs v8m vs v8l)

Tests inference speed and accuracy trade-offs across YOLO model sizes.
All tests use single-frame inference (batch_size=1).

Comparison:
  - YOLOv8n (nano): Smallest, fastest
  - YOLOv8s (small): Balanced (current default)
  - YOLOv8m (medium): Larger, more accurate
  - YOLOv8l (large): Largest, most accurate
"""

import cv2
import time
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def benchmark_model(model, model_name, video_path, max_frames=800, is_tensorrt=False):
    """
    Benchmark a YOLO model with single-frame inference
    
    Args:
        model: YOLO model instance
        model_name: Display name (e.g., "yolov8s", "yolov8s-trt")
        video_path: Path to video file
        max_frames: Number of frames to test
        is_tensorrt: Whether this is a TensorRT engine (for display)
    """
    print("\n" + "="*70)
    print(f"TESTING: {model_name}{' (TensorRT)' if is_tensorrt else ' (PyTorch)'}")
    print("="*70)
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    print(f"Video: {video_path.name}")
    print(f"Frames to process: {total_frames}")
    print(f"Starting benchmark...\n")
    
    decode_times = []
    inference_times = []
    detection_counts = []
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
        
        # Count detections
        num_detections = len(results[0].boxes) if results and len(results) > 0 else 0
        detection_counts.append(num_detections)
        
        frame_count += 1
        
        # Print progress every 200 frames
        if frame_count - last_print >= 200:
            elapsed = time.time() - start_total
            fps = frame_count / elapsed
            avg_decode = np.mean(decode_times[-200:]) * 1000
            avg_infer = np.mean(inference_times[-200:]) * 1000
            avg_detections = np.mean(detection_counts[-200:])
            print(f"  {frame_count}/{total_frames} | {fps:.1f} FPS | "
                  f"Decode: {avg_decode:.1f}ms | Infer: {avg_infer:.1f}ms | "
                  f"Detections: {avg_detections:.1f}")
            last_print = frame_count
    
    cap.release()
    end_total = time.time()
    
    total_time = end_total - start_total
    avg_decode = np.mean(decode_times) * 1000
    avg_infer = np.mean(inference_times) * 1000
    avg_detections = np.mean(detection_counts)
    fps = frame_count / total_time
    
    print(f"\n{'─'*70}")
    print(f"✅ {model_name} Results:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Frames: {frame_count}")
    print(f"   FPS: {fps:.1f}")
    print(f"   Avg decode time: {avg_decode:.2f}ms/frame")
    print(f"   Avg inference time: {avg_infer:.2f}ms/frame")
    print(f"   Total per frame: {avg_decode + avg_infer:.2f}ms")
    print(f"   Avg detections: {avg_detections:.1f} persons/frame")
    
    return {
        'model': model_name,
        'total_time': total_time,
        'frames': frame_count,
        'fps': fps,
        'decode_ms': avg_decode,
        'inference_ms': avg_infer,
        'total_ms': avg_decode + avg_infer,
        'avg_detections': avg_detections
    }


def print_comparison(results_list):
    """Print side-by-side comparison of all models"""
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    
    # Find baseline (usually first model tested)
    baseline = results_list[0]
    
    # Main comparison table
    print(f"\n{'Model':<15} {'FPS':<10} {'Total Time':<12} {'Speedup':<10}")
    print("─"*70)
    
    for result in results_list:
        speedup = result['fps'] / baseline['fps']
        print(f"{result['model']:<15} "
              f"{result['fps']:>6.1f}     "
              f"{result['total_time']:>6.2f}s      "
              f"{speedup:>6.2f}x")
    
    # Per-frame breakdown
    print("\n" + "─"*70)
    print("Per-frame timing:")
    print(f"{'Model':<15} {'Decode':<12} {'Inference':<12} {'Total':<12} {'Detections':<12}")
    print("─"*70)
    
    for result in results_list:
        print(f"{result['model']:<15} "
              f"{result['decode_ms']:>6.2f}ms     "
              f"{result['inference_ms']:>6.2f}ms     "
              f"{result['total_ms']:>6.2f}ms     "
              f"{result['avg_detections']:>6.1f}")
    
    # Inference time comparison
    print("\n" + "─"*70)
    print("Inference speedup comparison:")
    print(f"{'Model':<15} {'Inference Time':<20} {'vs Baseline':<15}")
    print("─"*70)
    
    for result in results_list:
        speedup = baseline['inference_ms'] / result['inference_ms']
        print(f"{result['model']:<15} "
              f"{result['inference_ms']:>6.2f}ms            "
              f"{speedup:>6.2f}x")
    
    # Find fastest model
    fastest = max(results_list, key=lambda x: x['fps'])
    fastest_speedup = fastest['fps'] / baseline['fps']
    
    print("\n" + "="*70)
    print("RECOMMENDATION:")
    print("="*70)
    
    if fastest != baseline:
        improvement = ((fastest['fps'] / baseline['fps']) - 1) * 100
        print(f"✅ {fastest['model']} is {improvement:.0f}% faster than {baseline['model']}")
        print(f"   FPS: {baseline['fps']:.1f} → {fastest['fps']:.1f}")
        print(f"   Inference: {baseline['inference_ms']:.2f}ms → {fastest['inference_ms']:.2f}ms per frame")
        print(f"   Time for 2027 frames: {baseline['total_time']*2027/baseline['frames']:.1f}s → "
              f"{fastest['total_time']*2027/fastest['frames']:.1f}s")
        
        # Check if detection accuracy is similar
        detection_diff = abs(fastest['avg_detections'] - baseline['avg_detections'])
        if detection_diff < 0.5:
            print(f"   ✅ Detection count similar ({fastest['avg_detections']:.1f} vs {baseline['avg_detections']:.1f})")
        else:
            print(f"   ⚠️  Detection count differs ({fastest['avg_detections']:.1f} vs {baseline['avg_detections']:.1f})")
            print(f"      Consider accuracy vs speed trade-off")
    else:
        print(f"✅ {baseline['model']} is already optimal for this use case")
    
    print("="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare different YOLO models (v8n, v8s, etc.)')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--models-dir', default='models/yolo', help='Directory containing YOLO models')
    parser.add_argument('--models', nargs='+', default=['yolov8s.pt', 'yolov8n.pt'], 
                        help='Model files to test (.pt or .engine) - default: yolov8s.pt yolov8n.pt')
    parser.add_argument('--max-frames', type=int, default=800, help='Frames to test (default: 800)')
    parser.add_argument('--full', action='store_true', help='Process entire video')
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return
    
    # Verify all models exist
    model_paths = []
    for model_file in args.models:
        model_path = models_dir / model_file
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return
        model_paths.append(model_path)
    
    max_frames = None if args.full else args.max_frames
    
    print("\n" + "="*70)
    print("YOLO MODEL COMPARISON BENCHMARK")
    print("="*70)
    print(f"Video: {video_path}")
    print(f"Max frames: {max_frames if max_frames else 'All'}")
    print(f"Models to test: {', '.join(args.models)}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only (error!)'}")
    print(f"Batch size: 1 (single-frame inference)")
    
    results_list = []
    
    # Test each model
    for i, model_path in enumerate(model_paths):
        model_name = model_path.stem  # e.g., "yolov8s"
        is_tensorrt = model_path.suffix == '.engine'
        
        print(f"\n{'='*70}")
        print(f"Loading model {i+1}/{len(model_paths)}: {model_name}")
        if is_tensorrt:
            print(f"Format: TensorRT Engine (optimized)")
        else:
            print(f"Format: PyTorch (.pt)")
        print(f"{'='*70}")
        
        # Load model
        model = YOLO(str(model_path))
        if not is_tensorrt:
            model.to('cuda')
        print(f"✅ {model_name} loaded on GPU")
        
        # Warmup
        print(f"🔥 Warming up {model_name}...")
        if is_tensorrt:
            # TensorRT needs actual frame warmup
            cap_warmup = cv2.VideoCapture(str(video_path))
            ret, warmup_frame = cap_warmup.read()
            cap_warmup.release()
            if ret:
                for _ in range(10):
                    _ = model(warmup_frame, verbose=False)
        else:
            # PyTorch can use dummy tensor
            dummy = torch.rand(1, 3, 640, 640).cuda()
            for _ in range(10):
                _ = model(dummy, verbose=False)
        print(f"✅ Warmup complete")
        
        # Benchmark
        # Note: Timing starts AFTER model load and warmup (excludes initialization overhead)
        result = benchmark_model(model, model_name, video_path, max_frames, is_tensorrt)
        results_list.append(result)
        
        # Clean up model
        del model
        torch.cuda.empty_cache()
        
        # Wait between tests
        if i < len(model_paths) - 1:
            time.sleep(2)
    
    # Compare all results
    print_comparison(results_list)


if __name__ == '__main__':
    main()
