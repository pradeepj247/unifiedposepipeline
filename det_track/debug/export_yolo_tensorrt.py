#!/usr/bin/env python3
"""
Export YOLO models to TensorRT engine format

Converts PyTorch YOLO models (.pt) to optimized TensorRT engines (.engine)
for 2-3x faster inference.

Usage:
    python export_yolo_tensorrt.py --models-dir /content/unifiedposepipeline/models/yolo

Output:
    yolov8s.pt → yolov8s.engine
    yolov8n.pt → yolov8n.engine
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import torch


def export_to_tensorrt(model_path, imgsz=640, half=True, device=0):
    """
    Export a YOLO model to TensorRT engine format
    
    Args:
        model_path: Path to .pt model file
        imgsz: Input image size (640 or 720)
        half: Use FP16 precision (faster, recommended)
        device: GPU device ID
    
    Returns:
        Path to generated .engine file
    """
    model_path = Path(model_path)
    print(f"\n{'='*70}")
    print(f"Exporting: {model_path.name}")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Image size: {imgsz}x{imgsz}")
    print(f"Precision: {'FP16' if half else 'FP32'}")
    print(f"Device: cuda:{device}")
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    # Check if .engine already exists
    engine_path = model_path.parent / f"{model_path.stem}.engine"
    if engine_path.exists():
        response = input(f"\n⚠️  {engine_path.name} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print(f"Skipping {model_path.name}")
            return engine_path
        print(f"Removing existing {engine_path.name}...")
        engine_path.unlink()
    
    # Load model
    print(f"\nLoading {model_path.name}...")
    model = YOLO(str(model_path))
    print(f"✅ Model loaded")
    
    # Export to TensorRT
    print(f"\n🔧 Exporting to TensorRT engine...")
    print(f"   This may take 2-5 minutes (optimizing kernels)...")
    
    try:
        export_path = model.export(
            format='engine',
            imgsz=imgsz,
            half=half,
            device=device,
            verbose=True
        )
        
        print(f"\n✅ Export successful!")
        print(f"   Engine file: {export_path}")
        
        # Verify engine exists
        if Path(export_path).exists():
            size_mb = Path(export_path).stat().st_size / (1024 * 1024)
            print(f"   File size: {size_mb:.1f} MB")
            return Path(export_path)
        else:
            print(f"⚠️  Engine file not found at expected location")
            return None
            
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Ensure TensorRT is installed: pip install tensorrt")
        print(f"  2. Check CUDA compatibility")
        print(f"  3. Try with half=False if FP16 fails")
        return None


def main():
    parser = argparse.ArgumentParser(description='Export YOLO models to TensorRT')
    parser.add_argument('--models-dir', required=True, help='Directory containing .pt model files')
    parser.add_argument('--models', nargs='+', default=['yolov8s.pt', 'yolov8n.pt'],
                        help='Model files to export (default: yolov8s.pt yolov8n.pt)')
    parser.add_argument('--imgsz', type=int, default=640, 
                        help='Input image size (default: 640, use 720 for 720p videos)')
    parser.add_argument('--fp32', action='store_true', help='Use FP32 instead of FP16 (slower but more precise)')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID (default: 0)')
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return
    
    print("\n" + "="*70)
    print("YOLO → TensorRT ENGINE EXPORT")
    print("="*70)
    print(f"Models directory: {models_dir}")
    print(f"Models to export: {', '.join(args.models)}")
    print(f"Image size: {args.imgsz}x{args.imgsz}")
    print(f"Precision: {'FP32' if args.fp32 else 'FP16 (default)'}")
    print(f"GPU: {torch.cuda.get_device_name(args.device) if torch.cuda.is_available() else 'CUDA not available!'}")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available. TensorRT requires GPU.")
        return
    
    # Export each model
    results = []
    for i, model_file in enumerate(args.models):
        model_path = models_dir / model_file
        
        print(f"\n[{i+1}/{len(args.models)}] Processing {model_file}...")
        
        engine_path = export_to_tensorrt(
            model_path,
            imgsz=args.imgsz,
            half=not args.fp32,
            device=args.device
        )
        
        if engine_path:
            results.append({
                'model': model_file,
                'engine': engine_path.name,
                'success': True
            })
        else:
            results.append({
                'model': model_file,
                'engine': None,
                'success': False
            })
    
    # Summary
    print("\n" + "="*70)
    print("EXPORT SUMMARY")
    print("="*70)
    
    success_count = sum(1 for r in results if r['success'])
    print(f"Total models: {len(results)}")
    print(f"Successful exports: {success_count}")
    print(f"Failed exports: {len(results) - success_count}")
    
    print(f"\n{'Model':<20} {'Engine File':<30} {'Status':<10}")
    print("─"*70)
    for r in results:
        status = "✅ Success" if r['success'] else "❌ Failed"
        engine = r['engine'] if r['engine'] else "N/A"
        print(f"{r['model']:<20} {engine:<30} {status:<10}")
    
    if success_count > 0:
        print("\n" + "="*70)
        print("✅ Export complete! You can now benchmark with:")
        print(f"   python benchmark_yolo_models.py \\")
        print(f"     --video <video_path> \\")
        print(f"     --models-dir {models_dir} \\")
        print(f"     --models {' '.join([r['engine'] for r in results if r['success']])} \\")
        print(f"     --models {' '.join(args.models)}")
        print("="*70)


if __name__ == '__main__':
    main()
