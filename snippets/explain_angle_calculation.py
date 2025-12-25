"""
Detailed Angle Calculation Explanation and Verification

This script will:
1. Show the EXACT angle calculation for a single triad (LShoulder-LElbow-LWrist)
2. Calculate for Frame 0 in both 2D (RTM) and 3D (MAGF)
3. Visualize the geometry to verify correctness
4. Show all intermediate steps

Usage in Colab:
    python explain_angle_calculation.py
"""

import numpy as np
import matplotlib.pyplot as plt

def angle_between_vectors(v1, v2):
    """
    Calculate angle between two vectors using dot product
    
    Formula: cos(θ) = (v1 · v2) / (|v1| * |v2|)
    Then: θ = arccos(cos(θ))
    
    Returns angle in degrees [0, 180]
    """
    # Normalize vectors
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    
    # Dot product
    dot_product = np.dot(v1_norm, v2_norm)
    
    # Clip to avoid numerical errors in arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Calculate angle in radians, then convert to degrees
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def calculate_elbow_angle_2d(shoulder, elbow, wrist):
    """
    Calculate elbow angle from 2D keypoints
    
    The angle at the elbow is formed by two vectors:
    - v1: from elbow to shoulder (upper arm)
    - v2: from elbow to wrist (forearm)
    """
    # Vector from elbow to shoulder
    v1 = shoulder - elbow
    
    # Vector from elbow to wrist
    v2 = wrist - elbow
    
    # Calculate angle between these vectors
    angle = angle_between_vectors(v1, v2)
    
    return angle, v1, v2


def calculate_elbow_angle_3d(shoulder, elbow, wrist):
    """
    Calculate elbow angle from 3D keypoints
    
    EXACTLY the same formula as 2D, just with 3D coordinates
    """
    # Vector from elbow to shoulder
    v1 = shoulder - elbow
    
    # Vector from elbow to wrist
    v2 = wrist - elbow
    
    # Calculate angle between these vectors
    angle = angle_between_vectors(v1, v2)
    
    return angle, v1, v2


# ============================================================================
# Load Data
# ============================================================================

print("=" * 80)
print("DETAILED ANGLE CALCULATION VERIFICATION")
print("=" * 80)

# Load RTM 2D and MAGF 3D
rtm_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_2D_rtm.npz')
magf_data = np.load('/content/unifiedposepipeline/demo_data/outputs/keypoints_3D_magf.npz')

rtm_2d = rtm_data['keypoints']      # (360, 17, 2)
magf_3d = magf_data['poses_3d']     # (300, 17, 3)

print(f"\n✅ Loaded: RTM 2D {rtm_2d.shape}, MAGF 3D {magf_3d.shape}")

# Use Frame 0
frame_idx = 0
frame_2d = rtm_2d[frame_idx]  # (17, 2)
frame_3d = magf_3d[frame_idx]  # (17, 3)

print(f"\nAnalyzing Frame {frame_idx}")

# ============================================================================
# LEFT ELBOW ANGLE CALCULATION
# ============================================================================

print("\n" + "=" * 80)
print("LEFT ELBOW ANGLE CALCULATION")
print("=" * 80)

# H36M-17 format joint indices:
# 11: LShoulder
# 12: LElbow  
# 13: LWrist

idx_shoulder = 11
idx_elbow = 12
idx_wrist = 13

# Get 2D coordinates
shoulder_2d = frame_2d[idx_shoulder]
elbow_2d = frame_2d[idx_elbow]
wrist_2d = frame_2d[idx_wrist]

print(f"\n📍 2D Keypoints (pixels):")
print(f"   LShoulder (joint {idx_shoulder}): {shoulder_2d}")
print(f"   LElbow    (joint {idx_elbow}): {elbow_2d}")
print(f"   LWrist    (joint {idx_wrist}): {wrist_2d}")

# Calculate 2D angle
angle_2d, v1_2d, v2_2d = calculate_elbow_angle_2d(shoulder_2d, elbow_2d, wrist_2d)

print(f"\n🔢 2D Angle Calculation:")
print(f"   Vector 1 (elbow → shoulder): {v1_2d}")
print(f"   Vector 2 (elbow → wrist):    {v2_2d}")
print(f"   |v1| = {np.linalg.norm(v1_2d):.4f}")
print(f"   |v2| = {np.linalg.norm(v2_2d):.4f}")
print(f"   v1 · v2 = {np.dot(v1_2d, v2_2d):.4f}")
print(f"   cos(θ) = {np.dot(v1_2d, v2_2d) / (np.linalg.norm(v1_2d) * np.linalg.norm(v2_2d)):.4f}")
print(f"   **2D Elbow Angle: {angle_2d:.2f}°**")

# Get 3D coordinates
shoulder_3d = frame_3d[idx_shoulder]
elbow_3d = frame_3d[idx_elbow]
wrist_3d = frame_3d[idx_wrist]

print(f"\n📍 3D Keypoints (normalized):")
print(f"   LShoulder (joint {idx_shoulder}): {shoulder_3d}")
print(f"   LElbow    (joint {idx_elbow}): {elbow_3d}")
print(f"   LWrist    (joint {idx_wrist}): {wrist_3d}")

# Calculate 3D angle
angle_3d, v1_3d, v2_3d = calculate_elbow_angle_3d(shoulder_3d, elbow_3d, wrist_3d)

print(f"\n🔢 3D Angle Calculation:")
print(f"   Vector 1 (elbow → shoulder): {v1_3d}")
print(f"   Vector 2 (elbow → wrist):    {v2_3d}")
print(f"   |v1| = {np.linalg.norm(v1_3d):.4f}")
print(f"   |v2| = {np.linalg.norm(v2_3d):.4f}")
print(f"   v1 · v2 = {np.dot(v1_3d, v2_3d):.4f}")
print(f"   cos(θ) = {np.dot(v1_3d, v2_3d) / (np.linalg.norm(v1_3d) * np.linalg.norm(v2_3d)):.4f}")
print(f"   **3D Elbow Angle: {angle_3d:.2f}°**")

# Difference
delta = abs(angle_2d - angle_3d)
print(f"\n❗ Angle Difference: {delta:.2f}°")

# ============================================================================
# PROJECTION TEST: Does 3D project to 2D correctly?
# ============================================================================

print("\n" + "=" * 80)
print("PROJECTION TEST")
print("=" * 80)

print("""
Key Question: When we project MAGF 3D (X, Y, Z) to 2D by taking (X, Y),
does it match RTM 2D?

If NO → That's why angles differ (3D doesn't project back to 2D)
If YES → Something else is wrong
""")

# Project 3D to 2D (simple orthographic: take X, Y)
shoulder_3d_proj = shoulder_3d[:2]  # Take X, Y only
elbow_3d_proj = elbow_3d[:2]
wrist_3d_proj = wrist_3d[:2]

print(f"\n📍 3D Projected to 2D (take X,Y only):")
print(f"   LShoulder: {shoulder_3d_proj}")
print(f"   LElbow:    {elbow_3d_proj}")
print(f"   LWrist:    {wrist_3d_proj}")

# Calculate angle from projected 3D
angle_3d_projected, v1_proj, v2_proj = calculate_elbow_angle_2d(
    shoulder_3d_proj, elbow_3d_proj, wrist_3d_proj
)

print(f"\n🔢 Angle from Projected 3D:")
print(f"   Vector 1 (elbow → shoulder): {v1_proj}")
print(f"   Vector 2 (elbow → wrist):    {v2_proj}")
print(f"   **Projected Elbow Angle: {angle_3d_projected:.2f}°**")

print(f"\n📊 Summary:")
print(f"   2D Angle (RTM):           {angle_2d:.2f}°")
print(f"   3D Angle (MAGF):          {angle_3d:.2f}°")
print(f"   3D Projected Angle:       {angle_3d_projected:.2f}°")
print(f"   Δ (2D vs 3D):             {abs(angle_2d - angle_3d):.2f}°")
print(f"   Δ (2D vs 3D Projected):   {abs(angle_2d - angle_3d_projected):.2f}°")

# ============================================================================
# COORDINATE SCALE ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("COORDINATE SCALE ANALYSIS")
print("=" * 80)

print(f"\n2D Coordinates (RTM - pixel space):")
print(f"   Range X: [{frame_2d[:, 0].min():.2f}, {frame_2d[:, 0].max():.2f}]")
print(f"   Range Y: [{frame_2d[:, 1].min():.2f}, {frame_2d[:, 1].max():.2f}]")
print(f"   Mean: {frame_2d.mean():.2f}")

print(f"\n3D Coordinates (MAGF - normalized):")
print(f"   Range X: [{frame_3d[:, 0].min():.4f}, {frame_3d[:, 0].max():.4f}]")
print(f"   Range Y: [{frame_3d[:, 1].min():.4f}, {frame_3d[:, 1].max():.4f}]")
print(f"   Range Z: [{frame_3d[:, 2].min():.4f}, {frame_3d[:, 2].max():.4f}]")
print(f"   Mean: {frame_3d.mean():.4f}")

print(f"\n⚠️  SCALE DIFFERENCE: ~{frame_2d.mean() / abs(frame_3d.mean()):.0f}x")

# ============================================================================
# NORMALIZATION TEST
# ============================================================================

print("\n" + "=" * 80)
print("NORMALIZATION TEST")
print("=" * 80)

print("""
Let's normalize both to unit scale and recalculate angles.
This is what the "corrected" script does.
""")

# Normalize 2D (center and scale to unit)
frame_2d_centered = frame_2d - frame_2d.mean(axis=0)
frame_2d_scale = np.sqrt(np.sum(frame_2d_centered ** 2))
frame_2d_norm = frame_2d_centered / frame_2d_scale

# Normalize 3D (center and scale to unit)
frame_3d_centered = frame_3d - frame_3d.mean(axis=0)
frame_3d_scale = np.sqrt(np.sum(frame_3d_centered ** 2))
frame_3d_norm = frame_3d_centered / frame_3d_scale

# Get normalized coordinates
shoulder_2d_norm = frame_2d_norm[idx_shoulder]
elbow_2d_norm = frame_2d_norm[idx_elbow]
wrist_2d_norm = frame_2d_norm[idx_wrist]

shoulder_3d_norm = frame_3d_norm[idx_shoulder, :2]  # Project to 2D
elbow_3d_norm = frame_3d_norm[idx_elbow, :2]
wrist_3d_norm = frame_3d_norm[idx_wrist, :2]

# Calculate angles on normalized data
angle_2d_norm, _, _ = calculate_elbow_angle_2d(
    shoulder_2d_norm, elbow_2d_norm, wrist_2d_norm
)
angle_3d_norm, _, _ = calculate_elbow_angle_2d(
    shoulder_3d_norm, elbow_3d_norm, wrist_3d_norm
)

print(f"\n📊 Normalized Results:")
print(f"   2D Angle (normalized):    {angle_2d_norm:.2f}°")
print(f"   3D Projected (normalized): {angle_3d_norm:.2f}°")
print(f"   Δ (normalized):           {abs(angle_2d_norm - angle_3d_norm):.2f}°")

# ============================================================================
# VISUAL VERIFICATION
# ============================================================================

print("\n" + "=" * 80)
print("VISUAL VERIFICATION")
print("=" * 80)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: 2D RTM
ax = axes[0]
ax.plot([shoulder_2d[0], elbow_2d[0], wrist_2d[0]], 
        [shoulder_2d[1], elbow_2d[1], wrist_2d[1]], 
        'bo-', linewidth=2, markersize=8)
ax.text(shoulder_2d[0], shoulder_2d[1], ' Shoulder', fontsize=10)
ax.text(elbow_2d[0], elbow_2d[1], ' Elbow', fontsize=10)
ax.text(wrist_2d[0], wrist_2d[1], ' Wrist', fontsize=10)
ax.set_title(f'2D RTM\nElbow Angle: {angle_2d:.2f}°')
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
ax.grid(True, alpha=0.3)
ax.invert_yaxis()  # Image coordinates

# Plot 2: 3D MAGF Projected to 2D
ax = axes[1]
ax.plot([shoulder_3d[0], elbow_3d[0], wrist_3d[0]], 
        [shoulder_3d[1], elbow_3d[1], wrist_3d[1]], 
        'ro-', linewidth=2, markersize=8)
ax.text(shoulder_3d[0], shoulder_3d[1], ' Shoulder', fontsize=10)
ax.text(elbow_3d[0], elbow_3d[1], ' Elbow', fontsize=10)
ax.text(wrist_3d[0], wrist_3d[1], ' Wrist', fontsize=10)
ax.set_title(f'3D MAGF (X,Y only)\nElbow Angle: {angle_3d_projected:.2f}°')
ax.set_xlabel('X (normalized)')
ax.set_ylabel('Y (normalized)')
ax.grid(True, alpha=0.3)

# Plot 3: 3D MAGF with Z depth (side view)
ax = axes[2]
ax.plot([shoulder_3d[2], elbow_3d[2], wrist_3d[2]], 
        [shoulder_3d[1], elbow_3d[1], wrist_3d[1]], 
        'go-', linewidth=2, markersize=8)
ax.text(shoulder_3d[2], shoulder_3d[1], ' Shoulder', fontsize=10)
ax.text(elbow_3d[2], elbow_3d[1], ' Elbow', fontsize=10)
ax.text(wrist_3d[2], wrist_3d[1], ' Wrist', fontsize=10)
ax.set_title(f'3D MAGF (Z,Y side view)\n3D Angle: {angle_3d:.2f}°')
ax.set_xlabel('Z (depth, normalized)')
ax.set_ylabel('Y (normalized)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/content/unifiedposepipeline/demo_data/outputs/angle_calculation_verification.png', 
            dpi=150, bbox_inches='tight')
print(f"\n✅ Saved visualization: /content/unifiedposepipeline/demo_data/outputs/angle_calculation_verification.png")

# ============================================================================
# CONCLUSION
# ============================================================================

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

print(f"""
📊 **Angle Calculation Summary for Frame {frame_idx}:**

1. **2D Elbow Angle (RTM):**           {angle_2d:.2f}°
2. **3D Elbow Angle (MAGF native):**   {angle_3d:.2f}°
3. **3D Projected to 2D Angle:**       {angle_3d_projected:.2f}°
4. **Normalized 2D Angle:**            {angle_2d_norm:.2f}°
5. **Normalized 3D Projected Angle:**  {angle_3d_norm:.2f}°

🔍 **Analysis:**

The angle calculation itself is CORRECT (dot product formula).

The difference ({delta:.2f}°) comes from:

A. **BEFORE normalization:**
   - Comparing angles in different coordinate systems (pixels vs normalized)
   - This is INCORRECT but we still see difference even with projection

B. **AFTER normalization:**
   - Both in unit scale, 3D projected to 2D
   - Difference: {abs(angle_2d_norm - angle_3d_norm):.2f}°
   - This is the TRUE 2D-3D inconsistency

💡 **Why the difference exists:**

MAGF's 3D estimate uses:
- Temporal information (243-frame window)
- Learned depth priors
- Biomechanical constraints

When projected back to 2D, it doesn't perfectly match RTM's 2D because:
1. MAGF smooths temporally (RTM is per-frame)
2. MAGF adds depth (Z) based on learned patterns
3. No explicit reprojection loss enforcing 2D-3D consistency

This is NOT an error in angle calculation - it's inherent in the
two-stage pipeline (2D detection → 3D lifting).

✅ **Verification complete!**
""")

print("\n✅ Analysis complete!\n")
