"""
Joint Mapping Diagnostic Tool

Shows current (potentially incorrect) understanding of joint semantics.
User will correct the mapping based on visual inspection.

Output:
1. Image with numbered joints (0-16)
2. Table showing current joint name assumptions
3. Table showing angle calculation assumptions

Usage in Colab:
    python diagnose_joint_mapping.py
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

# ============================================================================
# CORRECT JOINT MAPPING (from udp_3d_lifting_fixed.py)
# ============================================================================

# H36M-17 Joint Order (CORRECT from MotionAGFormer vis.py)
CORRECT_JOINT_NAMES = {
    0: 'Pelvis (Root)',
    1: 'RHip',
    2: 'RKnee',
    3: 'RAnkle',
    4: 'LHip',
    5: 'LKnee',
    6: 'LAnkle',
    7: 'Spine',
    8: 'Thorax',
    9: 'Neck/Nose',
    10: 'Head',
    11: 'LShoulder',
    12: 'LElbow',
    13: 'LWrist',
    14: 'RShoulder',
    15: 'RElbow',
    16: 'RWrist'
}

# Correct skeleton connections (from MotionAGFormer show3Dpose)
# I = [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9]
# J = [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10]
CORRECT_SKELETON = [
    (0, 1), (0, 4),      # Pelvis to hips
    (1, 2), (2, 3),      # Right leg
    (4, 5), (5, 6),      # Left leg
    (0, 7), (7, 8),      # Spine to thorax
    (8, 14), (14, 15), (15, 16),  # Right arm
    (8, 11), (11, 12), (12, 13),  # Left arm
    (8, 9), (9, 10),     # Neck to head
]

# Correct angle definitions (based on H36M-17 format)
CORRECT_ANGLE_DEFS = [
    (11, 12, 13, 'LElbow', 'Left Elbow'),        # LShoulder-LElbow-LWrist
    (14, 15, 16, 'RElbow', 'Right Elbow'),       # RShoulder-RElbow-RWrist
    (4, 5, 6, 'LKnee', 'Left Knee'),             # LHip-LKnee-LAnkle
    (1, 2, 3, 'RKnee', 'Right Knee'),            # RHip-RKnee-RAnkle
    (8, 11, 12, 'LShoulder', 'Left Shoulder'),   # Thorax-LShoulder-LElbow
    (8, 14, 15, 'RShoulder', 'Right Shoulder'),  # Thorax-RShoulder-RElbow
    (11, 0, 4, 'LHip', 'Left Hip'),              # LShoulder-Pelvis-LHip (torso bend)
    (14, 0, 1, 'RHip', 'Right Hip'),             # RShoulder-Pelvis-RHip (torso bend)
    (11, 8, 14, 'ShoulderBridge', 'Shoulder Bridge'),  # LShoulder-Thorax-RShoulder
]

# ============================================================================
# Load Data
# ============================================================================

print("=" * 80)
print("JOINT MAPPING DIAGNOSTIC - PLEASE CORRECT MY UNDERSTANDING!")
print("=" * 80)

FRAME_IDX = 1
VIDEO_PATH = '/content/unifiedposepipeline/demo_data/videos/dance.mp4'

# Load 2D keypoints
rtm_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_2D_rtm.npz')
keypoints_2d = rtm_data['keypoints'][FRAME_IDX]  # (17, 2)

print(f"\n✅ Loaded Frame {FRAME_IDX}")
print(f"   Keypoints shape: {keypoints_2d.shape}")

# Load video frame
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_IDX)
ret, frame = cap.read()
cap.release()

if not ret:
    print(f"❌ Failed to load frame {FRAME_IDX}")
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128

# ============================================================================
# Print Current Understanding (Table 1: Joint Names)
# ============================================================================

print("\n" + "=" * 80)
print("CORRECTED JOINT MAPPING (from udp_3d_lifting_fixed.py)")
print("=" * 80)

print(f"\n{'Joint #':<10s} {'Correct Name':<20s} {'2D Position (x, y)':<30s}")
print("-" * 80)

for joint_idx in range(17):
    joint_name = CORRECT_JOINT_NAMES[joint_idx]
    pos = keypoints_2d[joint_idx]
    print(f"{joint_idx:<10d} {joint_name:<20s} ({pos[0]:>7.2f}, {pos[1]:>7.2f})")

# ============================================================================
# Print Angle Calculation Definitions (Table 2)
# ============================================================================

print("\n" + "=" * 80)
print("CORRECTED ANGLE CALCULATION DEFINITIONS")
print("=" * 80)

print(f"\n{'Angle Name':<20s} {'Joint Names Used':<40s} {'Joint Numbers':<20s}")
print("-" * 80)

for j1, j2, j3, angle_id, angle_display in CORRECT_ANGLE_DEFS:
    joint_names = f"{CORRECT_JOINT_NAMES[j1]} - {CORRECT_JOINT_NAMES[j2]} - {CORRECT_JOINT_NAMES[j3]}"
    joint_numbers = f"({j1}, {j2}, {j3})"
    print(f"{angle_display:<20s} {joint_names:<40s} {joint_numbers:<20s}")

# ============================================================================
# Visualize with Numbers
# ============================================================================

print("\n🎨 Creating numbered visualization...")

fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Display frame
ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Draw skeleton connections (CORRECTED)
for j1, j2 in CORRECT_SKELETON:
    x = [keypoints_2d[j1, 0], keypoints_2d[j2, 0]]
    y = [keypoints_2d[j1, 1], keypoints_2d[j2, 1]]
    ax.plot(x, y, 'lime', linewidth=3, alpha=0.6)

# Draw joints with numbers
for joint_idx in range(17):
    pos = keypoints_2d[joint_idx]
    
    # Draw large circle for joint
    ax.scatter(pos[0], pos[1], c='red', s=200, edgecolors='white', 
               linewidths=2, zorder=10)
    
    # Draw number on joint
    ax.text(pos[0], pos[1], str(joint_idx), 
            fontsize=14, fontweight='bold', color='white',
            ha='center', va='center', zorder=11)
    
    # Draw label below joint
    joint_name = CORRECT_JOINT_NAMES[joint_idx]
    ax.text(pos[0], pos[1] + 20, joint_name, 
            fontsize=9, color='yellow',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
            ha='center', va='top')

ax.set_title(f'Frame {FRAME_IDX}: H36M-17 Joint Mapping (CORRECTED)\n'
             f'RED CIRCLES = Joint positions | WHITE NUMBERS = Joint indices (0-16) | '
             f'YELLOW LABELS = Correct H36M joint names | GREEN = Skeleton',
             fontsize=12, fontweight='bold')
ax.axis('off')

plt.tight_layout()

# Save image
output_path = f'/content/unifiedposepipeline/demo_data/outputs/frame_{FRAME_IDX}_numbered_joints.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ Saved: {output_path}")

plt.show()

# ============================================================================
# Instructions for User
# ============================================================================

print("\n" + "=" * 80)
print("CORRECTED MAPPING APPLIED!")
print("=" * 80)

print("""
✅ **Joint mapping has been CORRECTED based on udp_3d_lifting_fixed.py!**

Key corrections made:
1. Joint 0: 'Hip' → 'Pelvis (Root)' ✓
2. Joint 9: 'Nose' → 'Neck/Nose' ✓
3. Skeleton connections: Now match MotionAGFormer show3Dpose() ✓
4. Angle definitions: Updated shoulder angles to use Thorax (8) not Pelvis (0) ✓

📊 **New Angle Definitions:**
   - Left/Right Shoulder: Thorax-Shoulder-Elbow (was Pelvis-Shoulder-Elbow)
   - Shoulder Bridge: LShoulder-Thorax-RShoulder (was LShoulder-Pelvis-RShoulder)
   - Other angles remain the same

🔍 **Please verify the visualization:**
   - Does the skeleton look correct now?
   - Do the joint numbers match the body parts?
   - Are the connections anatomically correct?

If everything looks good, I'll update ALL comparison scripts with this corrected mapping!
""")

print("\n✅ Diagnostic complete!\n")
