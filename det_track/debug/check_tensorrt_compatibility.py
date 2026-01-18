#!/usr/bin/env python3
"""
Check TensorRT engine compatibility and re-export if needed.

TensorRT engines are compiled for specific CUDA/TensorRT/GPU combinations.
This script validates compatibility and triggers re-export if versions mismatch.

Usage:
    python check_tensorrt_compatibility.py --models-dir models/yolo
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def get_current_environment():
    """Get current CUDA, TensorRT, PyTorch versions and GPU info."""
    env_info = {}
    
    try:
        import torch
        env_info['pytorch'] = torch.__version__
        env_info['cuda'] = torch.version.cuda
        env_info['cudnn'] = str(torch.backends.cudnn.version())
        
        if torch.cuda.is_available():
            env_info['gpu_name'] = torch.cuda.get_device_name(0)
            env_info['gpu_compute'] = '.'.join(map(str, torch.cuda.get_device_capability(0)))
        else:
            env_info['gpu_name'] = 'None'
            env_info['gpu_compute'] = 'None'
    except Exception as e:
        print(f"❌ Error getting PyTorch info: {e}")
        env_info['pytorch'] = 'unknown'
        env_info['cuda'] = 'unknown'
        env_info['cudnn'] = 'unknown'
        env_info['gpu_name'] = 'unknown'
        env_info['gpu_compute'] = 'unknown'
    
    try:
        import tensorrt as trt
        env_info['tensorrt'] = trt.__version__
    except ImportError:
        env_info['tensorrt'] = 'not_installed'
    
    try:
        from ultralytics import __version__ as ultralytics_version
        env_info['ultralytics'] = ultralytics_version
    except ImportError:
        env_info['ultralytics'] = 'not_installed'
    
    return env_info


def load_engine_metadata(metadata_path):
    """Load engine metadata from JSON file."""
    if not os.path.exists(metadata_path):
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Error reading metadata: {e}")
        return None


def save_engine_metadata(metadata_path, env_info, engine_files):
    """Save engine metadata to JSON file."""
    metadata = {
        'environment': env_info,
        'engines': engine_files,
        'exported_at': subprocess.check_output(['date', '+%Y-%m-%d %H:%M:%S'], 
                                               text=True).strip() if sys.platform != 'win32' 
                      else subprocess.check_output(['powershell', 'Get-Date', '-Format', '"yyyy-MM-dd HH:mm:ss"'], 
                                                   text=True).strip().strip('"')
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved metadata to {metadata_path}")


def check_compatibility(current_env, saved_env):
    """
    Check if current environment is compatible with saved environment.
    
    Critical fields: CUDA, GPU compute capability
    Warning fields: TensorRT, PyTorch
    """
    if saved_env is None:
        return False, "No metadata found - engines need to be exported"
    
    issues = []
    warnings = []
    
    # Critical: CUDA version
    if current_env['cuda'] != saved_env['cuda']:
        issues.append(f"CUDA version mismatch: {saved_env['cuda']} → {current_env['cuda']}")
    
    # Critical: GPU compute capability
    if current_env['gpu_compute'] != saved_env['gpu_compute']:
        issues.append(f"GPU compute capability mismatch: {saved_env['gpu_compute']} → {current_env['gpu_compute']}")
    
    # Warning: TensorRT version (may still work)
    if current_env['tensorrt'] != saved_env['tensorrt']:
        warnings.append(f"TensorRT version changed: {saved_env['tensorrt']} → {current_env['tensorrt']}")
    
    # Warning: PyTorch version (less critical)
    if current_env['pytorch'] != saved_env['pytorch']:
        warnings.append(f"PyTorch version changed: {saved_env['pytorch']} → {current_env['pytorch']}")
    
    compatible = len(issues) == 0
    
    return compatible, issues, warnings


def find_engine_files(models_dir):
    """Find all .engine files in models directory."""
    models_path = Path(models_dir)
    return sorted(models_path.glob("*.engine"))


def trigger_reexport(models_dir, models_to_export):
    """Trigger re-export of TensorRT engines."""
    print("\n" + "="*60)
    print("🔄 RE-EXPORTING TENSORRT ENGINES")
    print("="*60)
    
    export_script = Path(__file__).parent / "export_yolo_tensorrt.py"
    
    if not export_script.exists():
        print(f"❌ Export script not found: {export_script}")
        return False
    
    # Build command
    cmd = [
        sys.executable,
        str(export_script),
        "--models-dir", models_dir,
        "--models"
    ] + models_to_export
    
    print(f"\n📦 Running: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Re-export failed with code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Check TensorRT engine compatibility and re-export if needed"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="/content/models/yolo",
        help="Directory containing YOLO models and engines"
    )
    parser.add_argument(
        "--auto-reexport",
        action="store_true",
        help="Automatically re-export if incompatible (default: prompt user)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolov8n.pt", "yolov8s.pt"],
        help="Models to re-export if needed"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("🔍 TENSORRT ENGINE COMPATIBILITY CHECK")
    print("="*60)
    
    # Get current environment
    print("\n📊 Current Environment:")
    current_env = get_current_environment()
    for key, value in current_env.items():
        print(f"  {key:15s}: {value}")
    
    # Check for metadata file
    metadata_path = os.path.join(args.models_dir, "tensorrt_metadata.json")
    saved_metadata = load_engine_metadata(metadata_path)
    
    if saved_metadata is None:
        print("\n⚠️  No metadata found - engines may not exist or were exported manually")
        print(f"   Expected: {metadata_path}")
        
        # Check if engines exist
        engine_files = find_engine_files(args.models_dir)
        if engine_files:
            print(f"\n⚠️  Found {len(engine_files)} engine file(s) without metadata:")
            for eng in engine_files:
                print(f"     - {eng.name}")
            print("\n⚠️  These may be incompatible with current environment!")
        
        print("\n🔧 Recommended: Re-export engines to ensure compatibility")
        
        if args.auto_reexport:
            should_export = True
        else:
            response = input("\n🔄 Re-export engines now? [Y/n]: ").strip().lower()
            should_export = response in ['', 'y', 'yes']
        
        if should_export:
            success = trigger_reexport(args.models_dir, args.models)
            sys.exit(0 if success else 1)
        else:
            print("\n⏭️  Skipping re-export (engines may fail to load)")
            sys.exit(1)
    
    # Compare environments
    print("\n📊 Saved Environment (from engines):")
    saved_env = saved_metadata['environment']
    for key, value in saved_env.items():
        print(f"  {key:15s}: {value}")
    
    print(f"\n📅 Engines exported: {saved_metadata.get('exported_at', 'unknown')}")
    print(f"📦 Engine files: {len(saved_metadata.get('engines', []))}")
    for eng in saved_metadata.get('engines', []):
        print(f"     - {eng}")
    
    # Check compatibility
    compatible, issues, warnings = check_compatibility(current_env, saved_env)
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"     - {warning}")
        print("     (These may not cause issues, but monitor for errors)")
    
    if not compatible:
        print("\n❌ INCOMPATIBLE ENVIRONMENT!")
        print("   Critical issues detected:")
        for issue in issues:
            print(f"     - {issue}")
        
        print("\n💥 TensorRT engines will FAIL to load with these mismatches!")
        print("   Engines must be re-exported for current environment")
        
        if args.auto_reexport:
            should_export = True
        else:
            response = input("\n🔄 Re-export engines now? [Y/n]: ").strip().lower()
            should_export = response in ['', 'y', 'yes']
        
        if should_export:
            success = trigger_reexport(args.models_dir, args.models)
            
            if success:
                # Save new metadata
                engine_files = [e.name for e in find_engine_files(args.models_dir)]
                save_engine_metadata(metadata_path, current_env, engine_files)
                print("\n✅ Re-export complete - engines are now compatible!")
                sys.exit(0)
            else:
                sys.exit(1)
        else:
            print("\n⏭️  Skipping re-export")
            print("❌ Engines will NOT work - pipeline will fail!")
            sys.exit(1)
    
    else:
        print("\n✅ ENVIRONMENT COMPATIBLE!")
        print("   All engines should load successfully")
        
        # Verify engines exist
        engine_files = find_engine_files(args.models_dir)
        if not engine_files:
            print("\n⚠️  WARNING: No .engine files found in models directory!")
            print(f"   Expected location: {args.models_dir}")
            print("   Run export script to create engines")
            sys.exit(1)
        
        print(f"\n✅ Found {len(engine_files)} engine file(s):")
        for eng in engine_files:
            size_mb = eng.stat().st_size / 1024 / 1024
            print(f"     - {eng.name:30s} ({size_mb:6.2f} MB)")
        
        sys.exit(0)


if __name__ == "__main__":
    main()
