#!/usr/bin/env python3
"""
Deep investigation of boxmot import time
Breaks down what takes so long during import
"""

import time
import sys

print("=" * 70)
print("BOXMOT IMPORT INVESTIGATION")
print("=" * 70)

# Check what's already imported (from Stage 1)
print("\n1. Checking pre-existing imports...")
pre_existing = []
check_modules = ['torch', 'torchvision', 'ultralytics', 'cv2', 'numpy', 'scipy']
for mod in check_modules:
    if mod in sys.modules:
        pre_existing.append(mod)
        print(f"   ✅ {mod} already loaded")
    else:
        print(f"   ⏹️  {mod} not yet loaded")

# Import key dependencies individually and time them
print("\n2. Importing boxmot dependencies individually...")

deps_to_test = [
    ('numpy', 'import numpy'),
    ('torch', 'import torch'),
    ('torchvision', 'import torchvision'),
    ('cv2', 'import cv2'),
    ('ultralytics', 'from ultralytics import YOLO'),
    ('loguru', 'from loguru import logger'),
    ('scipy', 'import scipy'),
]

dep_times = {}
for name, import_stmt in deps_to_test:
    if name in sys.modules:
        print(f"   ⏭️  {name} - already loaded (0.00s)")
        dep_times[name] = 0.0
    else:
        t_start = time.time()
        try:
            exec(import_stmt)
            elapsed = time.time() - t_start
            dep_times[name] = elapsed
            print(f"   ✅ {name} - imported in {elapsed:.2f}s")
        except ImportError as e:
            print(f"   ⚠️  {name} - failed: {e}")
            dep_times[name] = 0.0

# Now import boxmot and measure
print("\n3. Importing boxmot.ByteTrack...")
t_start = time.time()
from boxmot import ByteTrack
boxmot_import_time = time.time() - t_start
print(f"   ✅ boxmot imported in {boxmot_import_time:.2f}s")

# Check what NEW modules appeared
print("\n4. New modules loaded by boxmot:")
new_modules = []
boxmot_related = [m for m in sys.modules.keys() if 'boxmot' in m.lower()]
print(f"   Found {len(boxmot_related)} boxmot-related modules:")
for mod in sorted(boxmot_related)[:20]:  # Show first 20
    print(f"     - {mod}")

# Test if subsequent imports are faster
print("\n5. Testing import caching...")
# Remove from cache
if 'boxmot' in sys.modules:
    # Can't really re-import, but we can measure a second ByteTrack creation
    t_start = time.time()
    tracker2 = ByteTrack(track_thresh=0.15, track_buffer=30, match_thresh=0.8, min_hits=1, frame_rate=25)
    second_creation = time.time() - t_start
    print(f"   ✅ Second tracker creation: {second_creation:.4f}s")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nPre-existing imports (saved time):")
total_saved = sum(dep_times.get(mod, 0) for mod in pre_existing)
print(f"   {', '.join(pre_existing)} - ~{total_saved:.1f}s saved")

print(f"\nNew imports by boxmot:")
new_import_time = boxmot_import_time - total_saved
print(f"   Net new import overhead: {new_import_time:.2f}s")
print(f"   Boxmot-specific modules: {len(boxmot_related)}")

print(f"\nTotal boxmot import time: {boxmot_import_time:.2f}s")

# Recommendations
print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)
if total_saved > 1.0:
    print(f"✅ GOOD: Pre-existing imports saved {total_saved:.1f}s")
    print(f"   (Would be {boxmot_import_time + total_saved:.1f}s without Stage 1)")

if new_import_time > 3.0:
    print(f"\n⚠️  SLOW: boxmot-specific import takes {new_import_time:.2f}s")
    print("   Possible causes:")
    print("   - Heavy sub-module imports (trackers, models, etc.)")
    print("   - Settings file creation/validation")
    print("   - Lazy module initialization")
    print("\n   Optimization ideas:")
    print("   1. Import boxmot at pipeline startup (not per-stage)")
    print("   2. Check if boxmot has a 'lightweight' import mode")
    print("   3. Contact boxmot maintainers about import time")

if boxmot_import_time < 2.0:
    print(f"✅ FAST: boxmot import under 2s - acceptable")

print("=" * 70)
