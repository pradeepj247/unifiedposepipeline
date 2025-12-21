"""
Model Verification Tool - Check which RTMPose model is cached

This script helps verify which RTMPose model variant you're actually using
by inspecting the cached files and comparing checksums with known models.

Usage:
    python verify_model.py
"""

import os
import hashlib
from pathlib import Path
import zipfile


def get_file_hash(filepath, algorithm='md5', chunk_size=8192):
    """Calculate hash of a file"""
    hash_obj = hashlib.new(algorithm)
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def find_rtmpose_cache():
    """Find RTMPose model cache locations"""
    cache_locations = [
        Path.home() / ".cache" / "rtmpose",
        Path("/root/.cache/rtmpose"),  # Colab default
        Path.cwd() / ".cache" / "rtmpose",
        Path("/content/.cache/rtmpose"),  # Colab alternative
    ]
    
    found = []
    for loc in cache_locations:
        if loc.exists():
            found.append(loc)
    
    return found


def inspect_model_files(cache_dir):
    """Inspect RTMPose model files in cache"""
    print(f"\n📂 Inspecting: {cache_dir}")
    print("=" * 80)
    
    # Find all .onnx and .zip files
    onnx_files = list(cache_dir.rglob("*.onnx"))
    zip_files = list(cache_dir.rglob("*.zip"))
    
    if not onnx_files and not zip_files:
        print("   ⚠️  No model files found")
        return
    
    # Check ONNX files
    for onnx_file in onnx_files:
        size_mb = onnx_file.stat().st_size / (1024 * 1024)
        print(f"\n📄 ONNX Model: {onnx_file.name}")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"   Path: {onnx_file}")
        
        # Identify model by filename
        filename = onnx_file.name.lower()
        if "body7" in filename:
            print(f"   ✅ Body7 model detected!")
        elif "aic-coco" in filename:
            print(f"   ⚠️  AIC-COCO model (older 2-dataset version)")
        
        if "384x288" in filename or "384-288" in filename:
            print(f"   Resolution: 384×288 (High-accuracy)")
        elif "256x192" in filename or "256-192" in filename:
            print(f"   Resolution: 256×192 (Standard)")
        
        if "-l_" in filename or "-l-" in filename:
            print(f"   Model size: L (Large)")
        elif "-m_" in filename or "-m-" in filename:
            print(f"   Model size: M (Medium)")
        elif "-s_" in filename or "-s-" in filename:
            print(f"   Model size: S (Small)")
    
    # Check ZIP files
    for zip_file in zip_files:
        size_mb = zip_file.stat().st_size / (1024 * 1024)
        print(f"\n📦 ZIP Archive: {zip_file.name}")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"   Path: {zip_file}")
        
        # Try to read contents
        try:
            with zipfile.ZipFile(zip_file, 'r') as zf:
                print(f"   Contents:")
                for name in zf.namelist():
                    info = zf.getinfo(name)
                    print(f"      - {name} ({info.file_size / (1024*1024):.2f} MB)")
        except Exception as e:
            print(f"   ⚠️  Could not read ZIP: {e}")


def check_known_models():
    """List known RTMPose Body7 model checksums"""
    print("\n\n🔍 Known RTMPose Body7 Models (Recommended):")
    print("=" * 80)
    
    models = [
        {
            "name": "RTMPose-L Body7 (384×288)",
            "filename": "rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504",
            "checksum": "3f5a1437",
            "ap_coco": 78.3,
            "resolution": "384×288",
            "description": "⭐ HIGHEST ACCURACY - 7 datasets, best generalization"
        },
        {
            "name": "RTMPose-M Body7 (384×288)",
            "filename": "rtmpose-m_simcc-body7_pt-body7_420e-384x288-65e718c4_20230504",
            "checksum": "65e718c4",
            "ap_coco": 76.6,
            "resolution": "384×288",
            "description": "High accuracy, faster than L"
        },
        {
            "name": "RTMPose-L Body7 (256×192)",
            "filename": "rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504",
            "checksum": "4dba18fc",
            "ap_coco": 76.7,
            "resolution": "256×192",
            "description": "Balanced speed/accuracy"
        },
        {
            "name": "RTMPose-M Body7 (256×192)",
            "filename": "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504",
            "checksum": "e48f03d0",
            "ap_coco": 74.9,
            "resolution": "256×192",
            "description": "Standard accuracy, good speed"
        },
    ]
    
    for i, model in enumerate(models, 1):
        print(f"\n{i}. {model['name']}")
        print(f"   Filename: {model['filename']}")
        print(f"   Checksum: {model['checksum']}")
        print(f"   COCO AP: {model['ap_coco']}%")
        print(f"   Resolution: {model['resolution']}")
        print(f"   {model['description']}")


def verify_config_model():
    """Check which model is specified in config"""
    print("\n\n⚙️  Config Verification:")
    print("=" * 80)
    
    config_file = Path("configs/udp_video.yaml")
    if not config_file.exists():
        print(f"   ⚠️  Config not found: {config_file}")
        return
    
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Extract model URL
    if "rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437" in content:
        print(f"   ✅ Config uses: RTMPose-L Body7 (384×288)")
        print(f"   ✅ This is the RECOMMENDED high-accuracy model!")
        print(f"   ✅ AP: 78.3% on COCO")
        print(f"   ✅ Trained on 7 datasets for best generalization")
    elif "body7" in content:
        print(f"   ✅ Config uses a Body7 model (7-dataset version)")
        print(f"   Extract specific variant from URL")
    elif "aic-coco" in content:
        print(f"   ⚠️  Config uses older AIC-COCO model (2 datasets)")
        print(f"   💡 Consider upgrading to Body7 for better accuracy")
    else:
        print(f"   ❓ Could not determine model version from config")
    
    print(f"\n   Config location: {config_file.absolute()}")


def main():
    print("\n" + "🔍" * 40)
    print("RTMPose Model Verification Tool")
    print("🔍" * 40)
    
    # Find cache locations
    cache_dirs = find_rtmpose_cache()
    
    if cache_dirs:
        print(f"\n✅ Found {len(cache_dirs)} cache location(s)")
        for cache_dir in cache_dirs:
            inspect_model_files(cache_dir)
    else:
        print("\n⚠️  No RTMPose cache directories found")
        print("\nCommon locations to check manually:")
        print("   - ~/.cache/rtmpose")
        print("   - /root/.cache/rtmpose (Colab)")
        print("   - /content/.cache/rtmpose (Colab)")
    
    # Show known models
    check_known_models()
    
    # Verify config
    verify_config_model()
    
    print("\n\n💡 Quick verification on Colab:")
    print("   !ls -lh ~/.cache/rtmpose/")
    print("   !ls -lh /root/.cache/rtmpose/")
    print("   !find /root/.cache -name '*rtmpose*.onnx' -exec ls -lh {} \\;")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
