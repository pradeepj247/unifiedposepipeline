"""
Visual Comparison: 2D vs 3D Joint Angles for a Single Frame

Creates side-by-side visualization with:
- Left: Original frame with 2D keypoints + angle annotations
- Right: 3D skeleton with angle annotations
- Console: Detailed table of all joint angles

Usage in Colab:
    python visualize_frame_angles.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

# ============================================================================
# Angle Calculation
# ============================================================================

def angle_between_vectors(v1, v2):
    """Calculate angle between two vectors in degrees [0, 180]"""
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    return np.degrees(angle_rad)


def calculate_joint_angle(j1, j2, j3, keypoints):
    """Calculate angle at joint j2 formed by j1-j2-j3"""
    p1 = keypoints[j1]
    p2 = keypoints[j2]
    p3 = keypoints[j3]
    
    v1 = p1 - p2
    v2 = p3 - p2
    
    return angle_between_vectors(v1, v2)


# ============================================================================
# Joint Definitions (H36M-17 format)
# ============================================================================

JOINT_NAMES = {
    0: 'Hip', 1: 'RHip', 2: 'RKnee', 3: 'RAnkle',
    4: 'LHip', 5: 'LKnee', 6: 'LAnkle',
    7: 'Spine', 8: 'Thorax', 9: 'Nose', 10: 'Head',
    11: 'LShoulder', 12: 'LElbow', 13: 'LWrist',
    14: 'RShoulder', 15: 'RElbow', 16: 'RWrist'
}

# Angle definitions: (j1, j2, j3, name, display_name)
ANGLE_DEFINITIONS = [
    (11, 12, 13, 'LElbow', 'L Elbow'),
    (14, 15, 16, 'RElbow', 'R Elbow'),
    (4, 5, 6, 'LKnee', 'L Knee'),
    (1, 2, 3, 'RKnee', 'R Knee'),
    (0, 11, 12, 'LShoulder', 'L Shoulder'),
    (0, 14, 15, 'RShoulder', 'R Shoulder'),
    (11, 0, 4, 'LHip', 'L Hip'),
    (14, 0, 1, 'RHip', 'R Hip'),
    (11, 0, 14, 'ShoulderBridge', 'Shoulder Bridge'),
]

# Skeleton connections for visualization
SKELETON_CONNECTIONS = [
    # Spine
    (0, 1), (0, 4), (0, 7), (7, 8), (8, 9), (9, 10),
    # Right leg
    (1, 2), (2, 3),
    # Left leg
    (4, 5), (5, 6),
    # Right arm
    (8, 14), (14, 15), (15, 16),
    # Left arm
    (8, 11), (11, 12), (12, 13),
]


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_2d_with_angles(ax, frame_img, keypoints_2d, angles_2d, frame_idx):
    """Plot 2D keypoints on frame image with angle annotations"""
    
    # Display image
    ax.imshow(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB))
    
    # Draw skeleton
    for j1, j2 in SKELETON_CONNECTIONS:
        x = [keypoints_2d[j1, 0], keypoints_2d[j2, 0]]
        y = [keypoints_2d[j1, 1], keypoints_2d[j2, 1]]
        ax.plot(x, y, 'g-', linewidth=2, alpha=0.7)
    
    # Draw joints
    ax.scatter(keypoints_2d[:, 0], keypoints_2d[:, 1], 
               c='lime', s=50, edgecolors='white', linewidths=1.5, zorder=10)
    
    # Annotate angles at joint locations
    for j1, j2, j3, angle_name, display_name in ANGLE_DEFINITIONS:
        angle = angles_2d[angle_name]
        pos = keypoints_2d[j2]  # Position at the middle joint
        
        # Add text with background for readability
        ax.text(pos[0], pos[1], f'{angle:.1f}°', 
                fontsize=9, fontweight='bold',
                color='yellow', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
                ha='center', va='bottom')
    
    ax.set_title(f'Frame {frame_idx}: RTM 2D Keypoints + Angles', fontsize=12, fontweight='bold')
    ax.axis('off')


def plot_3d_with_angles(ax, keypoints_3d, angles_3d, frame_idx):
    """Plot 3D skeleton with angle annotations"""
    
    # Draw skeleton
    for j1, j2 in SKELETON_CONNECTIONS:
        x = [keypoints_3d[j1, 0], keypoints_3d[j2, 0]]
        y = [keypoints_3d[j1, 1], keypoints_3d[j2, 1]]
        z = [keypoints_3d[j1, 2], keypoints_3d[j2, 2]]
        ax.plot(x, y, z, 'b-', linewidth=2, alpha=0.7)
    
    # Draw joints
    ax.scatter(keypoints_3d[:, 0], keypoints_3d[:, 1], keypoints_3d[:, 2],
               c='red', s=50, edgecolors='white', linewidths=1.5)
    
    # Annotate angles
    for j1, j2, j3, angle_name, display_name in ANGLE_DEFINITIONS:
        angle = angles_3d[angle_name]
        pos = keypoints_3d[j2]
        
        ax.text(pos[0], pos[1], pos[2], f'{angle:.1f}°',
                fontsize=9, fontweight='bold',
                color='yellow',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Frame {frame_idx}: MAGF 3D Skeleton + Angles', fontsize=12, fontweight='bold')
    
    # Set equal aspect ratio
    max_range = np.array([keypoints_3d[:, 0].max() - keypoints_3d[:, 0].min(),
                          keypoints_3d[:, 1].max() - keypoints_3d[:, 1].min(),
                          keypoints_3d[:, 2].max() - keypoints_3d[:, 2].min()]).max() / 2.0
    
    mid_x = (keypoints_3d[:, 0].max() + keypoints_3d[:, 0].min()) * 0.5
    mid_y = (keypoints_3d[:, 1].max() + keypoints_3d[:, 1].min()) * 0.5
    mid_z = (keypoints_3d[:, 2].max() + keypoints_3d[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


# ============================================================================
# Main Script
# ============================================================================

print("=" * 80)
print("VISUAL ANGLE COMPARISON: 2D vs 3D")
print("=" * 80)

# Configuration
FRAME_IDX = 1  # Frame to visualize
VIDEO_PATH = '/content/unifiedposepipeline/demo_data/videos/dance.mp4'

# Load data
print(f"\n📂 Loading data...")
rtm_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_2D_rtm.npz')
magf_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_3D_magf.npz')

keypoints_2d_all = rtm_data['keypoints']
keypoints_3d_all = magf_data['poses_3d']

print(f"✅ RTM 2D:  {keypoints_2d_all.shape}")
print(f"✅ MAGF 3D: {keypoints_3d_all.shape}")

# Get specific frame
keypoints_2d = keypoints_2d_all[FRAME_IDX]
keypoints_3d = keypoints_3d_all[FRAME_IDX]

# Load video frame
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_IDX)
ret, frame = cap.read()
cap.release()

if not ret:
    print(f"❌ Failed to load frame {FRAME_IDX} from video")
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128  # Gray placeholder

print(f"\n📊 Analyzing Frame {FRAME_IDX}...")

# Calculate all angles
angles_2d = {}
angles_3d = {}

for j1, j2, j3, angle_name, display_name in ANGLE_DEFINITIONS:
    angles_2d[angle_name] = calculate_joint_angle(j1, j2, j3, keypoints_2d)
    angles_3d[angle_name] = calculate_joint_angle(j1, j2, j3, keypoints_3d)

# ============================================================================
# Print Table
# ============================================================================

print("\n" + "=" * 80)
print(f"JOINT ANGLE COMPARISON - FRAME {FRAME_IDX}")
print("=" * 80)

print(f"\n{'Joint Name':<20s} {'Joints (j1-j2-j3)':<20s} {'RTM 2D':<12s} {'MAGF 3D':<12s} {'Δ (abs)':<12s}")
print("-" * 80)

for j1, j2, j3, angle_name, display_name in ANGLE_DEFINITIONS:
    angle_2d = angles_2d[angle_name]
    angle_3d = angles_3d[angle_name]
    delta = abs(angle_2d - angle_3d)
    
    joint_names = f"{JOINT_NAMES[j1]}-{JOINT_NAMES[j2]}-{JOINT_NAMES[j3]}"
    
    print(f"{display_name:<20s} {joint_names:<20s} "
          f"{angle_2d:>10.2f}° {angle_3d:>10.2f}° {delta:>10.2f}°")

# Summary statistics
all_deltas = [abs(angles_2d[name] - angles_3d[name]) for _, _, _, name, _ in ANGLE_DEFINITIONS]
mean_delta = np.mean(all_deltas)
max_delta = np.max(all_deltas)

print("\n" + "-" * 80)
print(f"{'SUMMARY':<20s} {'':20s} {'':12s} {'':12s} Mean: {mean_delta:>6.2f}°")
print(f"{'':20s} {'':20s} {'':12s} {'':12s} Max:  {max_delta:>6.2f}°")

# ============================================================================
# Create Visualization
# ============================================================================

print(f"\n🎨 Creating visualization...")

fig = plt.figure(figsize=(20, 10))

# Left panel: 2D with angles
ax1 = fig.add_subplot(121)
plot_2d_with_angles(ax1, frame, keypoints_2d, angles_2d, FRAME_IDX)

# Right panel: 3D with angles
ax2 = fig.add_subplot(122, projection='3d')
plot_3d_with_angles(ax2, keypoints_3d, angles_3d, FRAME_IDX)

# Add overall title
fig.suptitle(f'2D vs 3D Joint Angle Comparison - Frame {FRAME_IDX}\n'
             f'Mean Difference: {mean_delta:.2f}°, Max Difference: {max_delta:.2f}°',
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
output_path = f'/content/unifiedposepipeline/demo_data/outputs/frame_{FRAME_IDX}_angle_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ Saved: {output_path}")

plt.show()

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

print(f"""
Frame {FRAME_IDX} Analysis:

📊 Statistics:
   Mean angle difference: {mean_delta:.2f}°
   Max angle difference:  {max_delta:.2f}°

💡 What this shows:
   - LEFT: Original video frame with RTM 2D detections
   - RIGHT: MAGF 3D reconstruction
   - Yellow numbers: Joint angles in degrees

🔍 Look for:
   - Large differences (>20°): Significant depth ambiguity
   - Consistent patterns: Which joints MAGF struggles with
   - Visual plausibility: Does 3D skeleton match 2D pose?

⚠️  Common issues:
   - Knees often have large errors (depth ambiguity)
   - Elbows/shoulders usually better (more constrained)
   - Temporal smoothing may cause frame-to-frame variations
""")

print("\n✅ Analysis complete!\n")
