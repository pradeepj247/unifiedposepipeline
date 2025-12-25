"""
Data Integrity Check: Inspect NPZ files for zeros and anomalies

Check all 4 NPZ files for:
1. Zero values (failed detections?)
2. NaN/Inf values
3. Data distribution statistics
4. Frame-by-frame validity

Usage in Colab:
    python check_data_integrity.py
"""

import numpy as np

def analyze_npz_file(filepath, expected_key=None):
    """Analyze NPZ file for data quality issues"""
    
    print("\n" + "=" * 80)
    print(f"FILE: {filepath.split('/')[-1]}")
    print("=" * 80)
    
    try:
        data = np.load(filepath)
        
        # List all keys
        print(f"\n📋 Available keys: {list(data.keys())}")
        
        # Auto-detect key if not specified
        if expected_key and expected_key in data:
            key = expected_key
        else:
            key = list(data.keys())[0]
            print(f"   Using key: '{key}'")
        
        arr = data[key]
        
        print(f"\n📊 Shape: {arr.shape}")
        print(f"   Dtype: {arr.dtype}")
        
        # Basic statistics
        print(f"\n📈 Value Statistics:")
        print(f"   Min:    {np.min(arr):.6f}")
        print(f"   Max:    {np.max(arr):.6f}")
        print(f"   Mean:   {np.mean(arr):.6f}")
        print(f"   Std:    {np.std(arr):.6f}")
        
        # Check for special values
        num_zeros = np.sum(arr == 0)
        num_nans = np.sum(np.isnan(arr))
        num_infs = np.sum(np.isinf(arr))
        total_values = arr.size
        
        print(f"\n🔍 Special Values:")
        print(f"   Zeros:    {num_zeros:>10d} / {total_values} ({num_zeros/total_values*100:6.2f}%)")
        print(f"   NaNs:     {num_nans:>10d} / {total_values} ({num_nans/total_values*100:6.2f}%)")
        print(f"   Infs:     {num_infs:>10d} / {total_values} ({num_infs/total_values*100:6.2f}%)")
        
        # Check per-frame zeros
        if len(arr.shape) == 3:  # (frames, joints, coords)
            print(f"\n🎞️  Per-Frame Analysis:")
            
            frames_with_all_zeros = 0
            frames_with_some_zeros = 0
            
            for i in range(len(arr)):
                frame = arr[i]
                num_frame_zeros = np.sum(frame == 0)
                total_frame_values = frame.size
                
                if num_frame_zeros == total_frame_values:
                    frames_with_all_zeros += 1
                elif num_frame_zeros > 0:
                    frames_with_some_zeros += 1
            
            print(f"   Frames with ALL zeros:  {frames_with_all_zeros} / {len(arr)}")
            print(f"   Frames with SOME zeros: {frames_with_some_zeros} / {len(arr)}")
            print(f"   Frames with NO zeros:   {len(arr) - frames_with_all_zeros - frames_with_some_zeros} / {len(arr)}")
            
            # Show first few frames summary
            print(f"\n   First 5 frames zero percentage:")
            for i in range(min(5, len(arr))):
                frame = arr[i]
                zero_pct = (np.sum(frame == 0) / frame.size) * 100
                print(f"      Frame {i}: {zero_pct:6.2f}% zeros")
        
        # Check per-joint zeros (if 3D array)
        if len(arr.shape) == 3:
            print(f"\n🦴 Per-Joint Analysis (averaged across frames):")
            
            num_joints = arr.shape[1]
            print(f"   Total joints: {num_joints}")
            
            # Show first 20 joints
            for j in range(min(20, num_joints)):
                joint_data = arr[:, j, :]  # All frames, this joint, all coords
                zero_pct = (np.sum(joint_data == 0) / joint_data.size) * 100
                mean_val = np.mean(np.abs(joint_data))
                print(f"      Joint {j:2d}: {zero_pct:6.2f}% zeros, mean |value|: {mean_val:8.3f}")
        
        # Sample values from first frame
        print(f"\n📝 Sample Values (Frame 0, First 5 Joints):")
        if len(arr.shape) == 3:
            for j in range(min(5, arr.shape[1])):
                print(f"      Joint {j}: {arr[0, j]}")
        
        return arr, key
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def compare_2d_files(rtm_arr, wb_arr):
    """Compare RTM vs WB 2D keypoints"""
    
    print("\n" + "=" * 80)
    print("COMPARISON: RTM 2D vs WB 2D")
    print("=" * 80)
    
    # Use first 30 frames for comparison
    num_frames = min(30, len(rtm_arr), len(wb_arr))
    
    # RTM has 17 joints, WB has 133 - compare first 17
    rtm_data = rtm_arr[:num_frames, :17, :]
    wb_data = wb_arr[:num_frames, :17, :]
    
    print(f"\n📊 Comparing first {num_frames} frames, first 17 joints")
    
    # Per-frame differences
    diffs = []
    for i in range(num_frames):
        diff = np.mean(np.abs(rtm_data[i] - wb_data[i]))
        diffs.append(diff)
    
    print(f"\n📏 Mean Absolute Difference per Frame:")
    print(f"   Overall mean: {np.mean(diffs):.4f}")
    print(f"   Std dev:      {np.std(diffs):.4f}")
    print(f"   Min:          {np.min(diffs):.4f}")
    print(f"   Max:          {np.max(diffs):.4f}")
    
    # Show first 10 frames
    print(f"\n   First 10 frames:")
    for i in range(min(10, num_frames)):
        print(f"      Frame {i}: {diffs[i]:.4f}")
    
    # Check correlation
    rtm_flat = rtm_data.flatten()
    wb_flat = wb_data.flatten()
    
    # Remove zeros for correlation
    mask = (rtm_flat != 0) & (wb_flat != 0)
    if np.sum(mask) > 0:
        correlation = np.corrcoef(rtm_flat[mask], wb_flat[mask])[0, 1]
        print(f"\n📈 Correlation (excluding zeros): {correlation:.4f}")


# ============================================================================
# Main Analysis
# ============================================================================

print("=" * 80)
print("DATA INTEGRITY CHECK")
print("=" * 80)

print("\n🔍 Checking all 4 NPZ files for data quality issues...")

# File paths
files = {
    'RTM_2D': ('/content/unifiedposepipeline/demo_data/outputs/keypoints_2D_rtm.npz', 'keypoints'),
    'WB_2D': ('/content/unifiedposepipeline/demo_data/outputs/keypoints_2D_wb.npz', 'keypoints'),
    'MAGF_3D': ('/content/unifiedposepipeline/demo_data/outputs/keypoints_3D_magf.npz', 'poses_3d'),
    'WB_3D': ('/content/unifiedposepipeline/demo_data/outputs/keypoints_3D_wb.npz', 'keypoints_3d'),
}

results = {}

# Analyze each file
for name, (filepath, key) in files.items():
    arr, actual_key = analyze_npz_file(filepath, key)
    results[name] = (arr, actual_key)

# Compare 2D files
print("\n" + "=" * 80)
print("CROSS-FILE COMPARISONS")
print("=" * 80)

if results['RTM_2D'][0] is not None and results['WB_2D'][0] is not None:
    compare_2d_files(results['RTM_2D'][0], results['WB_2D'][0])

# ============================================================================
# Summary and Recommendations
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 80)

print("""
✅ **What to look for:**

1. **Excessive zeros (>50%)**: Indicates failed detections or generation errors
2. **All-zero frames**: Completely failed frames
3. **NaN/Inf values**: Numerical instability in generation
4. **Huge differences between RTM and WB 2D**: Indicates different detections

🔧 **Action items:**

- If WB files have many zeros → Need to regenerate WB3D outputs
- If RTM files look good but WB has issues → Use RTM+MAGF pipeline
- If both are good but different → Expected (different models)
- If WB 3D has zeros but WB 2D is fine → 3D lifting failed

📊 **Expected patterns:**

RTM 2D:
  - Should have minimal zeros (only for occluded joints)
  - Values typically in range [0, image_width/height]
  
MAGF 3D:
  - Normalized coordinates, typically [-1, 1] or similar
  - Should have minimal zeros (anatomically centered)
  
WB3D 2D & 3D:
  - 133 joints (body + hands + face)
  - Should have data for all body joints (0-16)
  - Hand/face joints might have zeros if not detected
""")

print("\n✅ Analysis complete!\n")
