"""
Quick throwaway demo for Wholebody3D visualization
Run from: /content/unifiedposepipeline
"""

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / "lib"))

from ultralytics import YOLO
from rtmlib.tools import RTMPose3d


def plot_3d_skeleton(keypoints_3d, scores, output_path='wb3d_plot.png', threshold=0.3):
    """Create 4-subplot 3D visualization"""
    if len(keypoints_3d) == 0:
        print("⚠️ No keypoints to plot.")
        return
    
    kpts = keypoints_3d[0]  # (133, 3)
    score = scores[0]       # (133,)
    
    # Filter by confidence
    valid_mask = score > threshold
    valid_kpts = kpts[valid_mask]
    
    if len(valid_kpts) == 0:
        print(f"⚠️ No keypoints above threshold {threshold}.")
        return
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    
    # Plot 1: 3D scatter (XYZ)
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(valid_kpts[:, 0], valid_kpts[:, 1], valid_kpts[:, 2], 
                c='blue', marker='o', s=20, alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z (Depth)')
    ax1.set_title('3D Keypoints (Full View)')
    
    # Plot 2: Top-down view (X-Z plane)
    ax2 = fig.add_subplot(222)
    ax2.scatter(valid_kpts[:, 0], valid_kpts[:, 2], c='green', marker='o', s=20, alpha=0.6)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z (Depth)')
    ax2.set_title('Top-Down View (X-Z)')
    ax2.grid(True)
    ax2.invert_yaxis()
    
    # Plot 3: Side view (Y-Z plane)
    ax3 = fig.add_subplot(223)
    ax3.scatter(valid_kpts[:, 1], valid_kpts[:, 2], c='red', marker='o', s=20, alpha=0.6)
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z (Depth)')
    ax3.set_title('Side View (Y-Z)')
    ax3.grid(True)
    ax3.invert_yaxis()
    
    # Plot 4: Front view (X-Y plane)
    ax4 = fig.add_subplot(224)
    ax4.scatter(valid_kpts[:, 0], valid_kpts[:, 1], c='purple', marker='o', s=20, alpha=0.6)
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_title('Front View (X-Y, Standard 2D)')
    ax4.grid(True)
    ax4.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✅ 3D plot saved to: {output_path}")
    plt.close()


def print_keypoint_stats(keypoints_3d, scores, keypoints_2d):
    """Print detailed keypoint statistics"""
    if len(keypoints_3d) == 0:
        print("⚠️ No keypoints detected.")
        return
    
    kpts_3d = keypoints_3d[0]  # (133, 3)
    kpts_2d = keypoints_2d[0]  # (133, 2)
    score = scores[0]          # (133,)
    
    # Count keypoints by confidence
    high_conf = np.sum(score > 0.8)
    med_conf = np.sum((score > 0.5) & (score <= 0.8))
    low_conf = np.sum((score > 0.3) & (score <= 0.5))
    very_low = np.sum(score <= 0.3)
    
    print("\n📊 Keypoint Statistics:")
    print(f"   Total keypoints: 133 (body: 17, face: 68, left hand: 21, right hand: 21, feet: 6)")
    print(f"   High confidence (>0.8): {high_conf}")
    print(f"   Medium confidence (0.5-0.8): {med_conf}")
    print(f"   Low confidence (0.3-0.5): {low_conf}")
    print(f"   Very low (<0.3): {very_low}")
    
    # Depth range (Z coordinates)
    z_coords = kpts_3d[:, 2]
    valid_z = z_coords[score > 0.3]
    if len(valid_z) > 0:
        print(f"\n   Depth range (Z): {valid_z.min():.2f} to {valid_z.max():.2f}")
        print(f"   Depth span: {valid_z.max() - valid_z.min():.2f} units")
    
    # 2D pixel range
    x_coords = kpts_2d[:, 0]
    y_coords = kpts_2d[:, 1]
    valid_x = x_coords[score > 0.3]
    valid_y = y_coords[score > 0.3]
    if len(valid_x) > 0:
        print(f"\n   2D X range: {valid_x.min():.1f} to {valid_x.max():.1f} pixels")
        print(f"   2D Y range: {valid_y.min():.1f} to {valid_y.max():.1f} pixels")


def main():
    print("=" * 70)
    print("Wholebody3D Quick Demo")
    print("=" * 70)
    
    # Hardcoded paths
    image_path = "/content/unifiedposepipeline/demo_data/images/sample.jpg"
    yolo_path = "/content/models/yolo/yolov8s.pt"
    wb3d_model_path = "/content/models/wb3d/rtmw3d-l.onnx"
    output_plot = "/content/unifiedposepipeline/demo_data/outputs/wb3d_demo_plot.png"
    
    # Load image
    print(f"\n📷 Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Failed to load image")
        return
    print(f"   Image size: {img.shape[1]}x{img.shape[0]}")
    
    # Initialize models
    print(f"\n🔧 Initializing models...")
    detector = YOLO(yolo_path)
    pose_model = RTMPose3d(
        wb3d_model_path,
        model_input_size=(288, 384),
        backend='onnxruntime',
        device='cuda'
    )
    print("   ✅ Models loaded (YOLOv8s + RTMW3D-L)")
    
    # Detect person
    print(f"\n🔍 Detecting person...")
    results = detector(img, verbose=False)
    bboxes = []
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 0:  # Person
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bboxes.append([float(x1), float(y1), float(x2), float(y2)])
    
    if len(bboxes) == 0:
        print("⚠️ No person detected, using full image")
        bboxes = [[0, 0, img.shape[1], img.shape[0]]]
    else:
        print(f"✅ Found {len(bboxes)} person(s), using largest bbox")
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in bboxes]
        largest_idx = np.argmax(areas)
        bboxes = [bboxes[largest_idx]]
    
    # Run 3D pose estimation
    print(f"\n⚡ Running 3D pose estimation...")
    keypoints_3d, scores, keypoints_simcc, keypoints_2d = pose_model(img, bboxes=bboxes)
    print(f"   ✅ Inference complete!")
    print(f"   keypoints_3d shape: {keypoints_3d.shape}")
    print(f"   keypoints_2d shape: {keypoints_2d.shape}")
    print(f"   scores shape: {scores.shape}")
    
    # Print statistics
    print_keypoint_stats(keypoints_3d, scores, keypoints_2d)
    
    # Create 3D plot
    print(f"\n🎨 Creating 3D visualization...")
    plot_3d_skeleton(keypoints_3d, scores, output_plot, threshold=0.3)
    
    print("\n" + "=" * 70)
    print("✅ Demo complete!")
    print("=" * 70)
    print(f"\nOutput: {output_plot}")


if __name__ == '__main__':
    main()
