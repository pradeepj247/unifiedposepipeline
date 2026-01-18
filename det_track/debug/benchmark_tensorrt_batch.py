#!/usr/bin/env python3
"""
Benchmark: TensorRT engines with different batch sizes

Tests batch inference performance (1, 4, 8) for TensorRT-optimized YOLO models.
Compares yolov8s.engine vs yolov8n.engine across all batch sizes.

Usage:
    python benchmark_tensorrt_batch.py \
      --video demo_data/outputs/kohli_nets/kohli_nets_allI_720p.mp4 \
      --models-dir models/yolo
"""

import cv2
import time
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def benchmark_tensorrt_batch(model, model_name, video_path, batch_size=1, max_frames=800):
    """
    Benchmark TensorRT engine with specific batch size
    """
    print(f"\n{'='*70}")
    print(f"TESTING: {model_name} (TensorRT) - Batch size: {batch_size}")
    print(f"{'='*70}")
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    print(f"Video: {video_path.name}")
    print(f"Frames to process: {total_frames}")
    print(f"Batch size: {batch_size}")
    print(f"Starting benchmark...\n")
    
    decode_times = []
    inference_times = []
    detection_counts = []
    frame_count = 0
    last_print = 0
    
    start_total = time.time()
    
    if batch_size == 1:
        # Single-frame inference
        while frame_count < total_frames:
            # Decode
            t0 = time.time()
            ret, frame = cap.read()
            t1 = time.time()
            
            if not ret:
                break
            
            decode_times.append(t1 - t0)
            
            # Inference
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
                avg_det = np.mean(detection_counts[-200:])
                print(f"  {frame_count}/{total_frames} | {fps:.1f} FPS | "
                      f"Decode: {avg_decode:.1f}ms | Infer: {avg_infer:.1f}ms | Det: {avg_det:.1f}")
                last_print = frame_count
    
    else:
        # Batched inference
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
            per_frame_infer_time = batch_infer_time / len(frame_buffer)
            inference_times.extend([per_frame_infer_time] * len(frame_buffer))
            
            # Count detections for each frame in batch
            for result in results:
                num_detections = len(result.boxes) if result and hasattr(result, 'boxes') else 0
                detection_counts.append(num_detections)
            
            frame_count += len(frame_buffer)
            frame_buffer = []
            
            # Print progress every ~200 frames
            if frame_count - last_print >= 200:
                elapsed = time.time() - start_total
                fps = frame_count / elapsed
                avg_decode = np.mean(decode_times[-200:]) * 1000
                avg_infer = np.mean(inference_times[-200:]) * 1000
                avg_det = np.mean(detection_counts[-200:])
                print(f"  {frame_count}/{total_frames} | {fps:.1f} FPS | "
                      f"Decode: {avg_decode:.1f}ms | Infer: {avg_infer:.1f}ms/frame | Det: {avg_det:.1f}")
                last_print = frame_count
    
    cap.release()
    end_total = time.time()
    
    total_time = end_total - start_total
    avg_decode = np.mean(decode_times) * 1000
    avg_infer = np.mean(inference_times) * 1000
    avg_detections = np.mean(detection_counts)
    fps = frame_count / total_time
    
    print(f"\n{'─'*70}")
    print(f"✅ {model_name} (batch={batch_size}) Results:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Frames: {frame_count}")
    print(f"   FPS: {fps:.1f}")
    print(f"   Avg decode time: {avg_decode:.2f}ms/frame")
    print(f"   Avg inference time: {avg_infer:.2f}ms/frame")
    print(f"   Total per frame: {avg_decode + avg_infer:.2f}ms")
    print(f"   Avg detections: {avg_detections:.1f} persons/frame")
    
    return {
        'model': model_name,
        'batch_size': batch_size,
        'total_time': total_time,
        'frames': frame_count,
        'fps': fps,
        'decode_ms': avg_decode,
        'inference_ms': avg_infer,
        'total_ms': avg_decode + avg_infer,
        'avg_detections': avg_detections
    }


def print_comparison(results_list):
    """Print comprehensive comparison table"""
    print("\n" + "="*70)
    print("TENSORRT BATCH SIZE COMPARISON")
    print("="*70)
    
    # Group by model
    models = sorted(set(r['model'] for r in results_list))
    
    # Main comparison table
    print(f"\n{'Model':<15} {'Batch':<8} {'FPS':<10} {'Total Time':<12} {'Speedup':<10}")
    print("─"*70)
    
    baseline = results_list[0]  # First result (v8s batch=1)
    
    for result in results_list:
        speedup = result['fps'] / baseline['fps']
        print(f"{result['model']:<15} "
              f"{result['batch_size']:<8} "
              f"{result['fps']:>6.1f}     "
              f"{result['total_time']:>6.2f}s      "
              f"{speedup:>6.2f}x")
    
    # Per-frame timing breakdown
    print("\n" + "─"*70)
    print("Per-frame timing breakdown:")
    print(f"{'Model':<15} {'Batch':<8} {'Decode':<12} {'Inference':<12} {'Total':<12} {'Detections':<12}")
    print("─"*70)
    
    for result in results_list:
        print(f"{result['model']:<15} "
              f"{result['batch_size']:<8} "
              f"{result['decode_ms']:>6.2f}ms     "
              f"{result['inference_ms']:>6.2f}ms     "
              f"{result['total_ms']:>6.2f}ms     "
              f"{result['avg_detections']:>6.1f}")
    
    # Model comparison (best batch for each model)
    print("\n" + "="*70)
    print("BEST BATCH SIZE PER MODEL:")
    print("="*70)
    
    for model in models:
        model_results = [r for r in results_list if r['model'] == model]
        best = max(model_results, key=lambda x: x['fps'])
        print(f"\n{model}:")
        print(f"  Best batch size: {best['batch_size']}")
        print(f"  FPS: {best['fps']:.1f}")
        print(f"  Inference: {best['inference_ms']:.2f}ms/frame")
        print(f"  Time for 2027 frames: {best['total_time']*2027/best['frames']:.1f}s")
    
    # Overall winner
    print("\n" + "="*70)
    print("OVERALL RECOMMENDATION:")
    print("="*70)
    
    fastest = max(results_list, key=lambda x: x['fps'])
    speedup = fastest['fps'] / baseline['fps']
    improvement = (speedup - 1) * 100
    
    print(f"✅ Fastest: {fastest['model']} with batch_size={fastest['batch_size']}")
    print(f"   FPS: {baseline['fps']:.1f} → {fastest['fps']:.1f} ({improvement:.0f}% faster)")
    print(f"   Inference: {baseline['inference_ms']:.2f}ms → {fastest['inference_ms']:.2f}ms")
    print(f"   Time for full video (2027 frames): {fastest['total_time']*2027/fastest['frames']:.1f}s")
    
    # Check if detection count is similar
    baseline_det = baseline['avg_detections']
    fastest_det = fastest['avg_detections']
    det_diff = abs(fastest_det - baseline_det)
    
    if det_diff < 0.5:
        print(f"   ✅ Detection accuracy maintained ({fastest_det:.1f} vs {baseline_det:.1f})")
    else:
        print(f"   ⚠️  Detection count differs ({fastest_det:.1f} vs {baseline_det:.1f})")
    
    print("="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark TensorRT engines with different batch sizes')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--models-dir', default='models/yolo', help='Directory containing TensorRT engines')
    parser.add_argument('--models', nargs='+', default=['yolov8s.engine', 'yolov8n.engine'],
                        help='TensorRT engine files to test (default: yolov8s.engine yolov8n.engine)')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4, 8],
                        help='Batch sizes to test (default: 1 4 8)')
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
    
    # Verify all engines exist
    engine_paths = []
    for engine_file in args.models:
        engine_path = models_dir / engine_file
        if not engine_path.exists():
            print(f"❌ TensorRT engine not found: {engine_path}")
            print(f"   Run export_yolo_tensorrt.py first to create .engine files")
            return
        engine_paths.append(engine_path)
    
    max_frames = None if args.full else args.max_frames
    
    print("\n" + "="*70)
    print("TENSORRT BATCH INFERENCE BENCHMARK")
    print("="*70)
    print(f"Video: {video_path}")
    print(f"Max frames: {max_frames if max_frames else 'All'}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Batch sizes: {', '.join(map(str, args.batch_sizes))}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only (error!)'}")
    print("="*70)
    
    results_list = []
    
    # Test each model with each batch size
    for engine_path in engine_paths:
        model_name = engine_path.stem  # e.g., "yolov8s"
        
        for batch_idx, batch_size in enumerate(args.batch_sizes):
            print(f"\n{'='*70}")
            print(f"Testing: {model_name} (batch_size={batch_size})")
            print(f"{'='*70}")
            
            # Load model
            print(f"Loading {model_name}.engine...")
            model = YOLO(str(engine_path))
            print(f"✅ {model_name} loaded")
            
            # Warmup
            print(f"🔥 Warming up {model_name}...")
            cap_warmup = cv2.VideoCapture(str(video_path))
            warmup_frames = []
            for _ in range(batch_size):
                ret, frame = cap_warmup.read()
                if ret:
                    warmup_frames.append(frame)
            cap_warmup.release()
            
            if warmup_frames:
                for _ in range(10):
                    if batch_size == 1:
                        _ = model(warmup_frames[0], verbose=False)
                    else:
                        _ = model(warmup_frames, verbose=False)
            print(f"✅ Warmup complete")
            
            # Benchmark
            result = benchmark_tensorrt_batch(model, model_name, video_path, batch_size, max_frames)
            results_list.append(result)
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
            # Wait between tests
            if batch_idx < len(args.batch_sizes) - 1:
                time.sleep(1)
    
    # Compare all results
    print_comparison(results_list)


if __name__ == '__main__':
    main()
