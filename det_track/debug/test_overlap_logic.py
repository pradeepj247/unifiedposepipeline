#!/usr/bin/env python3
"""
Quick debug: Test P14/P29 overlap detection logic
"""

import numpy as np

# Simulate the data
tracklet_14_frames = np.arange(103, 366)  # 103-365
tracklet_29_frames = np.arange(360, 786)  # 360-785

print(f"P14 frames: {tracklet_14_frames[0]}-{tracklet_14_frames[-1]}")
print(f"P29 frames: {tracklet_29_frames[0]}-{tracklet_29_frames[-1]}")

# Check overlap
gap = tracklet_29_frames[0] - tracklet_14_frames[-1]
print(f"\nGap: {gap}")
print(f"Gap < -100? {gap < -100}")
print(f"Gap > 50 (max_temporal_gap)? {gap > 50}")

# Check if frame 360 exists in both
overlap_start = tracklet_29_frames[0]  # 360
print(f"\nOverlap start frame: {overlap_start}")
print(f"Frame 360 in P14? {overlap_start in tracklet_14_frames}")
print(f"Frame 360 in P29? {overlap_start in tracklet_29_frames}")

# Test the actual numpy array membership check
print(f"\nUsing 'in' operator on numpy array:")
print(f"  360 in tracklet_14_frames: {360 in tracklet_14_frames}")
print(f"  360 in tracklet_29_frames: {360 in tracklet_29_frames}")

# Alternative: use np.isin
print(f"\nUsing np.isin:")
print(f"  np.isin(360, tracklet_14_frames): {np.isin(360, tracklet_14_frames)}")
print(f"  np.isin(360, tracklet_29_frames): {np.isin(360, tracklet_29_frames)}")
