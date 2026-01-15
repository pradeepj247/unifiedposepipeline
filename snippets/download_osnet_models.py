#!/usr/bin/env python
"""
Download OSNet models to Colab.

This script downloads the necessary OSNet ReID models to Colab if they don't exist.
It provides two options:

1. x0_25 (ONNX) - Lightweight, tested, recommended fallback
2. x1_0 (PyTorch) - Stronger model, better quality

Both can coexist. The pipeline will prefer x1_0 and fall back to x0_25.

RUN THIS SCRIPT ON COLAB:
    python download_osnet_models.py
"""

import os
from pathlib import Path
import urllib.request
import sys

# Colab paths
MODELS_DIR = Path("/content/unifiedposepipeline/models/reid")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("OSNet Model Downloader for Colab")
print("=" * 70)
print()

# Models to download
MODELS = {
    "osnet_x0_25_msmt17.onnx": {
        "url": "https://github.com/KaiyuYue/person-reid-lib/releases/download/osnet_ms/osnet_x0_25_msmt17.onnx",
        "size_mb": 3.2,
        "type": "ONNX (Lightweight)",
        "priority": "FALLBACK (safe, tested)",
    },
    "osnet_x1_0_msmt17.pt": {
        "url": "https://github.com/KaiyuYue/person-reid-lib/releases/download/osnet_ms/osnet_x1_0_msmt17.pt",
        "size_mb": 1200,  # Estimated
        "type": "PyTorch (Stronger)",
        "priority": "PRIMARY (best quality)",
    },
}

# Check which models exist
print("Current Status:")
print("-" * 70)
for model_name, info in MODELS.items():
    model_path = MODELS_DIR / model_name
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ {model_name:35} | {size_mb:7.1f} MB | {info['priority']}")
    else:
        print(f"✗ {model_name:35} | MISSING  | {info['priority']}")

print()
print("-" * 70)
print()

# Ask user what to download
print("Available models to download:")
print()
print("1. osnet_x0_25_msmt17.onnx (RECOMMENDED - Tested)")
print("   - Type: ONNX (fast inference)")
print("   - Size: ~3 MB")
print("   - Priority: Fallback (if x1_0 not available)")
print("   - Status: WORKS (0.32-0.88 similarity range)")
print()
print("2. osnet_x1_0_msmt17.pt (OPTIONAL - Better Quality)")
print("   - Type: PyTorch (stronger features)")
print("   - Size: ~1.2 GB (!)")
print("   - Priority: Primary (preferred if available)")
print("   - Status: Better discrimination expected")
print()
print("3. Both (recommended for best fallback support)")
print()

try:
    choice = input("Which models to download? (1/2/3, default=1): ").strip() or "1"
except KeyboardInterrupt:
    print("\nAborted by user")
    sys.exit(0)

models_to_download = []
if choice == "1":
    models_to_download = ["osnet_x0_25_msmt17.onnx"]
elif choice == "2":
    models_to_download = ["osnet_x1_0_msmt17.pt"]
elif choice == "3":
    models_to_download = list(MODELS.keys())
else:
    print("Invalid choice")
    sys.exit(1)

print()
print("=" * 70)
print(f"Downloading {len(models_to_download)} model(s)...")
print("=" * 70)
print()

downloaded_count = 0
for model_name in models_to_download:
    model_path = MODELS_DIR / model_name
    info = MODELS[model_name]
    
    # Skip if already exists
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"⏭️  SKIPPING {model_name} (already exists, {size_mb:.1f} MB)")
        print()
        continue
    
    print(f"⬇️  Downloading {model_name}...")
    print(f"   URL: {info['url']}")
    print(f"   Estimated size: {info['size_mb']:.0f} MB")
    
    try:
        # Download with progress
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded / total_size) * 100)
                mb_done = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"   [{percent:3.0f}%] {mb_done:6.1f} / {mb_total:6.1f} MB", end='\r')
        
        urllib.request.urlretrieve(info['url'], str(model_path), progress_hook)
        
        # Verify download
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"   ✓ Downloaded successfully ({size_mb:.1f} MB)     ")
        downloaded_count += 1
        print()
        
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        print(f"   Try downloading manually from: {info['url']}")
        if model_path.exists():
            model_path.unlink()
        print()

print("=" * 70)
print("Summary:")
print("-" * 70)

# Final status
any_missing = False
for model_name in models_to_download:
    model_path = MODELS_DIR / model_name
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ {model_name:35} | {size_mb:7.1f} MB")
    else:
        print(f"✗ {model_name:35} | FAILED TO DOWNLOAD")
        any_missing = True

print()
if any_missing:
    print("⚠️  Some models failed to download")
    print("   Try downloading manually from GitHub releases")
else:
    print(f"✅ Successfully downloaded {downloaded_count} model(s)")
    print()
    print("Next steps:")
    print("1. Run pipeline: python det_track/run_pipeline.py --config configs/pipeline_config.yaml")
    print("2. Check output for: [OSNet] ✓ Loaded ... model")
    print("3. Verify similarities are in 0.3-0.9 range (not 0.96-0.99)")

print("=" * 70)
