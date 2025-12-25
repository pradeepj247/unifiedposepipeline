"""
Quick script to check what keys are in the keypoint .npz files
"""
import numpy as np
from pathlib import Path

# Update these paths to match your Colab paths
kps2d_path = "demo_data/outputs/kps_2d_wb3d.npz"
kps3d_path = "demo_data/outputs/kps_3d_wb3d.npz"

print("=" * 70)
print("Checking .npz file contents")
print("=" * 70)

print("\n2D Keypoints file:")
print(f"  Path: {kps2d_path}")
try:
    data = np.load(kps2d_path)
    print(f"  Keys: {list(data.keys())}")
    for key in data.keys():
        print(f"    {key}: shape={data[key].shape}, dtype={data[key].dtype}")
except Exception as e:
    print(f"  Error: {e}")

print("\n3D Keypoints file:")
print(f"  Path: {kps3d_path}")
try:
    data = np.load(kps3d_path)
    print(f"  Keys: {list(data.keys())}")
    for key in data.keys():
        print(f"    {key}: shape={data[key].shape}, dtype={data[key].dtype}")
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "=" * 70)
