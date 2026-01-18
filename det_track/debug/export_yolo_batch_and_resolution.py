#!/usr/bin/env python3
"""
Export YOLO models with specific batch sizes and input resolutions

Creates optimized TensorRT engines:
1. Batch=4 at 640x640 (for batched inference)
2. Batch=1 at 576x576 (for lower resolution testing)

Usage:
    python export_yolo_batch_and_resolution.py --models-dir /content/unifiedposepipeline/models/yolo
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import torch


def export_engine(model_path, output_name, batch_size=1, imgsz=640, half=True, device=0):
    """
    Export a YOLO model to TensorRT engine with specific batch size
    
    Args:
        model_path: Path to .pt model file
        output_name: Name for output engine file (without .engine extension)
        batch_size: Fixed batch size for engine
        imgsz: Input image size
        half: Use FP16 precision
        device: GPU device ID
    
    Returns:
        Path to generated .engine file or None if failed
    """
    model_path = Path(model_path)
    print(f"\n{'='*70}")
    print(f"Exporting: {model_path.name} → {output_name}.engine")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {imgsz}x{imgsz}")
    print(f"Precision: {'FP16' if half else 'FP32'}")
    print(f"Device: cuda:{device}")
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    # Expected output path
    output_dir = model_path.parent
    engine_path = output_dir / f"{output_name}.engine"
    
    # Check if .engine already exists
    if engine_path.exists():
        response = input(f"\n⚠️  {engine_path.name} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print(f"Skipping {output_name}")
            return engine_path
        print(f"Removing existing {engine_path.name}...")
        engine_path.unlink()
    
    # Load model
    print(f"\nLoading {model_path.name}...")
    model = YOLO(str(model_path))
    print(f"✅ Model loaded")
    
    # Export to TensorRT with specific batch size
    print(f"\n🔧 Exporting to TensorRT engine...")
    print(f"   Batch size: {batch_size} (fixed)")
    print(f"   This may take 2-5 minutes (optimizing kernels)...")
    
    try:
        # Ultralytics export with batch parameter
        export_path = model.export(
            format='engine',
            imgsz=imgsz,
            batch=batch_size,  # Fixed batch size
            half=half,
            device=device,
            verbose=True
        )
        
        # Ultralytics creates file with default naming, rename it
        default_engine = Path(export_path)
        if default_engine.exists() and default_engine != engine_path:
            # Rename to our desired name
            default_engine.rename(engine_path)
            print(f"\n✅ Export successful!")
            print(f"   Renamed: {default_engine.name} → {engine_path.name}")
        elif engine_path.exists():
            print(f"\n✅ Export successful!")
        else:
            print(f"\n⚠️  Export completed but file not found at expected location")
            print(f"   Looking for: {engine_path}")
            return None
        
        # Verify and show file size
        if engine_path.exists():
            size_mb = engine_path.stat().st_size / (1024 * 1024)
            print(f"   Engine file: {engine_path.name}")
            print(f"   File size: {size_mb:.1f} MB")
            return engine_path
        else:
            return None
            
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Ensure TensorRT is installed: pip install tensorrt")
        print(f"  2. Check CUDA compatibility")
        print(f"  3. Try with half=False if FP16 fails")
        return None


def main():
    parser = argparse.ArgumentParser(description='Export YOLO models with batch sizes and resolutions')
    parser.add_argument('--models-dir', required=True, help='Directory containing .pt model files')
    parser.add_argument('--models', nargs='+', default=['yolov8s.pt', 'yolov8n.pt'],
                        help='Model files to export (default: yolov8s.pt yolov8n.pt)')
    parser.add_argument('--fp32', action='store_true', help='Use FP32 instead of FP16')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID (default: 0)')
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return
    
    print("\n" + "="*70)
    print("YOLO BATCH & RESOLUTION EXPORT")
    print("="*70)
    print(f"Models directory: {models_dir}")
    print(f"Models to export: {', '.join(args.models)}")
    print(f"Precision: {'FP32' if args.fp32 else 'FP16 (default)'}")
    print(f"GPU: {torch.cuda.get_device_name(args.device) if torch.cuda.is_available() else 'CUDA not available!'}")
    print("\nExport plan:")
    print("  1. Batch=4 at 640x640 (for batched inference)")
    print("  2. Batch=1 at 576x576 (for lower resolution)")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available. TensorRT requires GPU.")
        return
    
    results = []
    
    # Export each model in two configurations
    for model_file in args.models:
        model_path = models_dir / model_file
        model_stem = model_path.stem  # e.g., "yolov8s"
        
        if not model_path.exists():
            print(f"\n❌ Model not found: {model_path}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing: {model_file}")
        print(f"{'='*70}")
        
        # Configuration 1: Batch=4 at 640x640
        print(f"\n[1/2] Exporting batch=4 at 640x640...")
        output_name_b4 = f"{model_stem}_b4_640"
        engine_b4 = export_engine(
            model_path,
            output_name_b4,
            batch_size=4,
            imgsz=640,
            half=not args.fp32,
            device=args.device
        )
        results.append({
            'model': model_file,
            'config': 'batch=4, 640x640',
            'output': f"{output_name_b4}.engine" if engine_b4 else None,
            'success': engine_b4 is not None
        })
        
        # Configuration 2: Batch=1 at 576x576
        print(f"\n[2/2] Exporting batch=1 at 576x576...")
        output_name_576 = f"{model_stem}_576"
        engine_576 = export_engine(
            model_path,
            output_name_576,
            batch_size=1,
            imgsz=576,
            half=not args.fp32,
            device=args.device
        )
        results.append({
            'model': model_file,
            'config': 'batch=1, 576x576',
            'output': f"{output_name_576}.engine" if engine_576 else None,
            'success': engine_576 is not None
        })
    
    # Summary
    print("\n" + "="*70)
    print("EXPORT SUMMARY")
    print("="*70)
    
    success_count = sum(1 for r in results if r['success'])
    print(f"Total exports attempted: {len(results)}")
    print(f"Successful exports: {success_count}")
    print(f"Failed exports: {len(results) - success_count}")
    
    print(f"\n{'Model':<20} {'Configuration':<20} {'Output File':<30} {'Status':<10}")
    print("─"*90)
    for r in results:
        status = "✅ Success" if r['success'] else "❌ Failed"
        output = r['output'] if r['output'] else "N/A"
        print(f"{r['model']:<20} {r['config']:<20} {output:<30} {status:<10}")
    
    if success_count > 0:
        print("\n" + "="*70)
        print("✅ Export complete! Generated engines:")
        print("="*70)
        for r in results:
            if r['success']:
                print(f"  • {r['output']}")
                if 'b4' in r['output']:
                    print(f"    → Use for batched inference (4 frames at once)")
                else:
                    print(f"    → Use for lower resolution (faster, less accurate)")
        print("\nYou can now benchmark with benchmark_tensorrt_batch.py")
        print("="*70)


if __name__ == '__main__':
    main()
