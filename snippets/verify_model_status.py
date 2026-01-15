#!/usr/bin/env python
"""
Verify OSNet model status on current system.

This script helps diagnose why OSNet is using random initialization instead of
loading the proper model. It checks:

1. Whether model files exist
2. Which model would be loaded first
3. Simulates the model loading priority
"""

import os
from pathlib import Path

# Simulated paths (based on Colab structure)
COLAB_MODELS_DIR = "/content/unifiedposepipeline/models"
WINDOWS_MODELS_DIR = "D:\\trials\\unifiedpipeline\\newrepo\\models"

# Detect current system
import platform
if platform.system() == "Windows":
    MODELS_DIR = WINDOWS_MODELS_DIR
    print(f"🖥️  Running on Windows")
else:
    MODELS_DIR = COLAB_MODELS_DIR
    print(f"🐧 Running on Linux/Colab")

print(f"📁 Models directory: {MODELS_DIR}")
print()

# Model paths to check
PRIMARY_MODEL = os.path.join(MODELS_DIR, "reid", "osnet_x1_0_msmt17.pt")
FALLBACK_MODEL = os.path.join(MODELS_DIR, "reid", "osnet_x0_25_msmt17.onnx")

models_to_check = [
    ("PRIMARY (x1_0 PyTorch)", PRIMARY_MODEL),
    ("FALLBACK (x0_25 ONNX)", FALLBACK_MODEL),
]

print("=" * 70)
print("MODEL STATUS CHECK")
print("=" * 70)
print()

# Check each model
found_any = False
for model_name, model_path in models_to_check:
    exists = Path(model_path).exists()
    size_mb = None
    if exists:
        size_mb = Path(model_path).stat().st_size / (1024 * 1024)
        found_any = True
        status = f"✓ EXISTS ({size_mb:.1f} MB)"
    else:
        status = "✗ NOT FOUND"
    
    print(f"{model_name:30} | {status:30}")
    print(f"{'':30} | {model_path}")
    print()

print("=" * 70)
print()

# Simulation of loading priority
print("LOADING PRIORITY (what happens when pipeline runs):")
print()
print("1️⃣  Try PRIMARY model (x1_0.pt)")
if Path(PRIMARY_MODEL).exists():
    print(f"   ✓ Found! Will load PyTorch model")
    print(f"   Expected behavior: Good feature discrimination (similarity 0.3-0.9)")
else:
    print(f"   ✗ Not found.")
    print()
    print("2️⃣  Try FALLBACK model (x0_25.onnx)")
    if Path(FALLBACK_MODEL).exists():
        print(f"   ✓ Found! Will load ONNX model")
        print(f"   Expected behavior: Moderate discrimination (similarity 0.3-0.88)")
    else:
        print(f"   ✗ Not found.")
        print()
        print("3️⃣  Fall back to RANDOM INITIALIZATION")
        print(f"   ⚠️  This is what happened in your last run!")
        print(f"   Expected behavior: Random embeddings, meaningless similarities (0.96-0.99)")

print()
print("=" * 70)
print()

# Recommendation
if not found_any:
    print("❌ RECOMMENDATION: No models found!")
    print()
    print("   Download models to Colab:")
    print("   1. Download osnet_x0_25_msmt17.onnx from: https://drive.google.com/...")
    print("   2. Place at: /content/unifiedposepipeline/models/reid/osnet_x0_25_msmt17.onnx")
    print()
    print("   OR download x1_0 PyTorch version for better quality")
elif Path(PRIMARY_MODEL).exists():
    print("✅ GOOD: Primary model x1_0.pt is available")
    print("   The stronger model should be used")
    print()
    print("   If you still see 'randomly initialized' message:")
    print("   - Check if PyTorch is installed: pip install torch")
    print("   - Check if model loading logs show errors")
    print("   - Run with verbose=True to see detailed diagnostics")
elif Path(FALLBACK_MODEL).exists():
    print("⚠️  PRIMARY NOT FOUND - Using fallback model x0_25.onnx")
    print("   This is the safe fallback. Clustering will work but with lower quality.")
    print()
    print("   If similarities are 0.96-0.99:")
    print("   - This means the fallback model is also not loaded")
    print("   - Pipeline fell back to random initialization")
    print("   - Check ONNX Runtime installation: pip install onnxruntime")
    print()
    print("   To get better results:")
    print("   - Download osnet_x1_0_msmt17.pt (stronger model)")
    print("   - Place at: /content/unifiedposepipeline/models/reid/osnet_x1_0_msmt17.pt")

print()
print("=" * 70)
