"""
Check if MAGF and WB3D are using the same 2D keypoint inputs

This helps explain why 3D outputs differ so much (PA-MPJPE = 0.86)

Usage in Colab:
    python check_2d_inputs.py
"""

import numpy as np

print("=" * 80)
print("2D INPUT COMPARISON")
print("=" * 80)

# Load RTM 2D keypoints (input to MAGF)
print("\n📂 Loading 2D keypoints...")
try:
    rtm_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_2D_rtm.npz')
    print(f"   RTM file keys: {list(rtm_data.keys())}")
    # Try common key names
    if 'keypoints' in rtm_data.keys():
        magf_2d = rtm_data['keypoints']  # (N, 17, 2) COCO format
    elif 'keypoints_2d' in rtm_data.keys():
        magf_2d = rtm_data['keypoints_2d']
    else:
        # Use first available key
        first_key = list(rtm_data.keys())[0]
        magf_2d = rtm_data[first_key]
    print(f"✅ RTM 2D keypoints: {magf_2d.shape}")
except Exception as e:
    print(f"❌ Could not load RTM 2D input: {e}")
    magf_2d = None

# Load WB3D's 2D keypoints
try:
    wb3d_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_2D_wb.npz')
    print(f"   WB3D file keys: {list(wb3d_data.keys())}")
    # Try common key names
    if 'keypoints' in wb3d_data.keys():
        wb3d_2d_full = wb3d_data['keypoints']
    elif 'keypoints_2d' in wb3d_data.keys():
        wb3d_2d_full = wb3d_data['keypoints_2d']
    else:
        # Use first available key
        first_key = list(wb3d_data.keys())[0]
        wb3d_2d_full = wb3d_data[first_key]
    
    # Extract body joints only (first 17) if WB3D has 133
    if wb3d_2d_full.shape[1] == 133:
        wb3d_2d = wb3d_2d_full[:, :17, :]  # (N, 17, 2) body only
        print(f"✅ WB3D 2D keypoints: {wb3d_2d.shape} (extracted body from 133)")
    else:
        wb3d_2d = wb3d_2d_full
        print(f"✅ WB3D 2D keypoints: {wb3d_2d.shape}")
except Exception as e:
    print(f"❌ Could not load WB3D 2D: {e}")
    wb3d_2d = None

if magf_2d is not None and wb3d_2d is not None:
    # Compare first frame
    n_frames = min(magf_2d.shape[0], wb3d_2d.shape[0])
    
    print(f"\n📊 Comparing first {n_frames} frames...")
    
    # Frame 0 comparison
    magf_frame0 = magf_2d[0]  # (17, 2)
    wb3d_frame0 = wb3d_2d[0]  # (17, 2)
    
    print("\n" + "=" * 80)
    print("FRAME 0: 2D KEYPOINT COMPARISON")
    print("=" * 80)
    
    joint_names = [
        'Nose', 'LEye', 'REye', 'LEar', 'REar',
        'LShoulder', 'RShoulder', 'LElbow', 'RElbow',
        'LWrist', 'RWrist', 'LHip', 'RHip',
        'LKnee', 'RKnee', 'LAnkle', 'RAnkle'
    ]
    
    print(f"\n{'Joint':<12s} {'MAGF (x, y)':<25s} {'WB3D (x, y)':<25s} {'Diff (px)':>10s}")
    print("-" * 80)
    
    total_diff = 0
    for i in range(17):
        m = magf_frame0[i]
        w = wb3d_frame0[i]
        diff = np.linalg.norm(m - w)
        total_diff += diff
        
        magf_str = f"({m[0]:7.2f}, {m[1]:7.2f})"
        wb3d_str = f"({w[0]:7.2f}, {w[1]:7.2f})"
        
        print(f"{joint_names[i]:<12s} {magf_str:<25s} {wb3d_str:<25s} {diff:>10.2f}")
    
    avg_diff = total_diff / 17
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Average 2D difference (frame 0): {avg_diff:.2f} pixels")
    
    if avg_diff < 5:
        print("✅ 2D inputs are VERY SIMILAR (< 5px difference)")
        print("   → 3D difference is due to different lifting methods")
    elif avg_diff < 20:
        print("⚠️ 2D inputs have MODERATE differences (5-20px)")
        print("   → 3D difference is partly due to different 2D detections")
    else:
        print("❌ 2D inputs are VERY DIFFERENT (> 20px)")
        print("   → 3D difference is PRIMARILY due to different 2D detections!")
        print("   → This explains the high PA-MPJPE = 0.86")
    
    # Calculate per-frame average difference
    frame_diffs = []
    for t in range(min(n_frames, 30)):  # First 30 frames
        diff = np.mean(np.linalg.norm(magf_2d[t] - wb3d_2d[t], axis=1))
        frame_diffs.append(diff)
    
    print(f"\nAverage 2D difference (first 30 frames): {np.mean(frame_diffs):.2f} pixels")
    print(f"Min: {np.min(frame_diffs):.2f}px, Max: {np.max(frame_diffs):.2f}px")

else:
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
Cannot compare 2D inputs directly.

IMPORTANT: MAGF and WB3D likely use DIFFERENT 2D keypoints:
- MAGF: Uses Stage 1+2 output (YOLOv8 + RTMPose 2D)
- WB3D: Uses its own detection (YOLOv8 + RTMPose 2D, but separate run)

This means they're lifting DIFFERENT 2D poses to 3D!
→ This explains the high PA-MPJPE = 0.86

To make a fair 3D comparison, you would need to:
1. Save WB3D's 2D keypoints during wb3d_demo.py
2. OR feed the SAME 2D keypoints to both methods
""")

print("\n✅ Analysis complete!\n")
