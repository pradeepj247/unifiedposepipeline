"""
Debug script: Verify what format features are actually being stored in.

Check if we're actually storing (16, 256) per person or accidentally averaging to (256,)
"""
import json
import numpy as np
from pathlib import Path

# Load the HTML output to see what was generated
html_path = Path(r"D:\trials\unifiedpipeline\viewer (4).html")
print(f"HTML file size: {html_path.stat().st_size / 1024 / 1024:.1f} MB")
print()

# Check if all_features.json exists
features_json_path = Path(r"D:\trials\unifiedpipeline\demo_data\outputs\all_features.json")
if features_json_path.exists():
    print(f"✓ all_features.json EXISTS ({features_json_path.stat().st_size / 1024:.1f} KB)")
    with open(features_json_path) as f:
        features_data = json.load(f)
    
    print(f"  Keys: {list(features_data.keys())}")
    
    # Check first person
    if 'feature_info' in features_data:
        print(f"  feature_info: {features_data['feature_info']}")
    
    # Check person features
    person_keys = [k for k in features_data.keys() if k.startswith('person_')]
    if person_keys:
        first_pid = person_keys[0]
        first_features = features_data[first_pid]
        if isinstance(first_features, list):
            print(f"\n  {first_pid}: list with {len(first_features)} elements")
            if len(first_features) > 0:
                if isinstance(first_features[0], list):
                    print(f"    -> Nested list, first element has {len(first_features[0])} values")
                    print(f"    -> This looks like: (num_crops, 256)")
                else:
                    print(f"    -> Flat list of {len(first_features)} values")
                    print(f"    -> This looks like: (256,) - a single embedding!")
else:
    print(f"✗ all_features.json NOT FOUND")

print()

# Check all_features.npy
features_npy_path = Path(r"D:\trials\unifiedpipeline\demo_data\outputs\all_features.npy")
if features_npy_path.exists():
    print(f"✓ all_features.npy EXISTS ({features_npy_path.stat().st_size / 1024:.1f} KB)")
    try:
        all_feat = np.load(features_npy_path)
        print(f"  Shape: {all_feat.shape}")
        if all_feat.ndim == 1:
            print(f"  -> This is 1D, looks like a single averaged embedding (256,)")
        elif all_feat.ndim == 2:
            print(f"  -> This is 2D (num_items, 256)")
            print(f"  -> Expected: (8 persons, 256) for averaged OR (128 crops, 256) for per-crop")
    except Exception as e:
        print(f"  Error loading: {e}")
else:
    print(f"✗ all_features.npy NOT FOUND")

print()

# Check individual person files
person_files = list(Path(r"D:\trials\unifiedpipeline\demo_data\outputs").glob("features_person_*.npy"))
if person_files:
    print(f"✓ Found {len(person_files)} person feature files")
    for pf in sorted(person_files)[:3]:  # Check first 3
        arr = np.load(pf)
        print(f"  {pf.name}: shape {arr.shape}")
else:
    print(f"✗ No person feature files (features_person_*.npy)")

print()
print("=" * 70)
print("INTERPRETATION:")
print("=" * 70)
print()
print("If per-crop approach is working:")
print("  - each person file should be (16, 256)")
print("  - OR all_features.json has nested lists")
print()
print("If averaging is happening (bug):")
print("  - each person file should be (256,)")
print("  - OR all_features.json has flat lists")
