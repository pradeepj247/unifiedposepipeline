#!/usr/bin/env python3
"""
Test script to measure ByteTrack initialization timing breakdown
"""

import time
import sys

print("=" * 70)
print("BYTETRACK INITIALIZATION TIMING TEST")
print("=" * 70)

# Measure boxmot import time
print("\n1. Importing boxmot library...")
t_start = time.time()
try:
    from boxmot import ByteTrack
    import_time = time.time() - t_start
    print(f"   ✅ Import completed in {import_time:.2f}s")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Measure ByteTrack object creation time
print("\n2. Creating ByteTrack tracker object...")
t_start = time.time()
tracker = ByteTrack(
    track_thresh=0.15,
    track_buffer=30,
    match_thresh=0.8,
    min_hits=1,
    frame_rate=25
)
creation_time = time.time() - t_start
print(f"   ✅ Tracker created in {creation_time:.2f}s")

# Measure first tracking call
print("\n3. First tracking call (warm-up)...")
import numpy as np
dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
dummy_dets = np.array([[100, 100, 200, 200, 0.9, 0]])  # [x1,y1,x2,y2,conf,cls]
t_start = time.time()
_ = tracker.update(dummy_dets, dummy_frame)
first_call_time = time.time() - t_start
print(f"   ✅ First call completed in {first_call_time:.2f}s")

# Measure subsequent tracking calls
print("\n4. Subsequent tracking calls (5 iterations)...")
times = []
for i in range(5):
    t_start = time.time()
    _ = tracker.update(dummy_dets, dummy_frame)
    times.append(time.time() - t_start)
avg_time = sum(times) / len(times)
print(f"   ✅ Average: {avg_time:.4f}s ({1/avg_time:.0f} FPS)")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Import boxmot:           {import_time:.2f}s")
print(f"  Create tracker object:   {creation_time:.2f}s")
print(f"  First tracking call:     {first_call_time:.2f}s")
print(f"  Subsequent calls (avg):  {avg_time:.4f}s")
print(f"\n  Total init overhead:     {import_time + creation_time + first_call_time:.2f}s")
print("=" * 70)
