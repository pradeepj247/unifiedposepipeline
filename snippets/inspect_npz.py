"""
Inspect NPZ files to see what keys they contain

Usage in Colab:
    python inspect_npz.py
"""

import numpy as np

print("=" * 80)
print("NPZ FILE INSPECTOR")
print("=" * 80)

# Inspect MAGF file
print("\n1. MAGF File: keypoints_3D_magf.npz")
print("-" * 80)
try:
    magf_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_3D_magf.npz')
    print(f"✅ File loaded successfully")
    print(f"Keys in file: {list(magf_data.keys())}")
    for key in magf_data.keys():
        print(f"  - '{key}': shape = {magf_data[key].shape}, dtype = {magf_data[key].dtype}")
except Exception as e:
    print(f"❌ Error: {e}")

# Inspect WB3D file
print("\n2. WB3D File: keypoints_3D_wb.npz")
print("-" * 80)
try:
    wb3d_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_3D_wb.npz')
    print(f"✅ File loaded successfully")
    print(f"Keys in file: {list(wb3d_data.keys())}")
    for key in wb3d_data.keys():
        print(f"  - '{key}': shape = {wb3d_data[key].shape}, dtype = {wb3d_data[key].dtype}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 80)
print("✅ Inspection complete!")
print("=" * 80)
