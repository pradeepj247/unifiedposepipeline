#!/usr/bin/env python3
"""
Test if boxmot works WITHOUT torchvision pre-imported
"""

import time
import torch  # Pre-import torch only (not torchvision)

print("✅ torch pre-imported")
print(f"   PyTorch version: {torch.__version__}")

# Now import boxmot
print("\n⏱️  Importing boxmot (with torch pre-loaded, NO torchvision)...")
t_start = time.time()
from boxmot import ByteTrack
elapsed = time.time() - t_start
print(f"   ✅ boxmot imported in {elapsed:.2f}s")

# Test tracker creation
print("\n⏱️  Creating ByteTrack tracker...")
t_start = time.time()
tracker = ByteTrack(
    track_thresh=0.15,
    track_buffer=30,
    match_thresh=0.8,
    min_hits=1,
    frame_rate=25
)
elapsed = time.time() - t_start
print(f"   ✅ Tracker created in {elapsed:.4f}s")

print("\n✅ SUCCESS: boxmot works fine WITHOUT torchvision!")
print(f"   Recommendation: Only pre-import torch, not torchvision")
