"""
Debug Halpe26 Skeleton - Visualize numbered keypoints and only extra connections

Shows:
- All 26 keypoints with numbers
- Only feet (17-22) and body (23-25) connections
- Helps identify incorrect skeleton edges

Usage:
    python debug_halpe26_skeleton.py --config configs/udp_image_halpe26.yaml
"""

import sys
import argparse
from pathlib import Path
import yaml
import time
import cv2
import numpy as np
import matplotlib.colors

REPO_ROOT = Path(__file__).parent
PARENT_DIR = REPO_ROOT.parent
MODELS_DIR = PARENT_DIR / "models"

# Only connections for extra keypoints (17-25)
HALPE26_EXTRA_EDGES = [
    # Right foot connections (from ankle 16)
    (16, 21), (16, 23), (16, 25),  # RAnkle to RBigToe, RSmallToe, RHeel
    # Left foot connections (from ankle 15)
    (15, 20), (15, 22), (15, 24),  # LAnkle to LBigToe, LSmallToe, LHeel
    # Body chain connections
    (17, 18),  # Head to Neck
    (18, 19),  # Neck to Hip/Pelvis
]

# Keypoint names (correct Halpe26 ordering)
HALPE26_NAMES = [
    "0:Nose", "1:LEye", "2:REye", "3:LEar", "4:REar",
    "5:LShoulder", "6:RShoulder", "7:LElbow", "8:RElbow",
    "9:LWrist", "10:RWrist", "11:LHip", "12:RHip",
    "13:LKnee", "14:RKnee", "15:LAnkle", "16:RAnkle",
    "17:Head", "18:Neck", "19:Hip",
    "20:LBigToe", "21:RBigToe", "22:LSmallToe",
    "23:RSmallToe", "24:LHeel", "25:RHeel"
]


def detect_persons_yolo(image, yolo, confidence_threshold):
    """Detect persons using YOLOv8s"""
    t0 = time.time()
    results = yolo(image, classes=[0], verbose=False)
    boxes = []
    for result in results:
        for box in result.boxes:
            if box.conf[0] >= confidence_threshold:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
    det_time = (time.time() - t0) * 1000
    return boxes, det_time


def estimate_poses_rtmpose(image, boxes, config):
    """Estimate poses using RTMPose"""
    sys.path.insert(0, str(REPO_ROOT / "lib"))
    from rtmlib.tools import RTMPose
    
    pose_model = RTMPose(
        onnx_model=config["pose_model_url"],
        model_input_size=tuple(config["pose_input_size"]),
        backend=config["backend"],
        device=config["device"]
    )
    
    t0 = time.time()
    keypoints, scores = pose_model(image, bboxes=boxes)
    pose_time = (time.time() - t0) * 1000
    
    return keypoints, scores, pose_time


def draw_debug_skeleton(image, keypoints, scores, kpt_thr=0.3):
    """
    Draw debug skeleton with numbered keypoints and only extra connections
    
    Args:
        image: Input image (BGR)
        keypoints: Keypoint coordinates (26 x 2)
        scores: Confidence scores (26,)
        kpt_thr: Confidence threshold
    
    Returns:
        result_image: Annotated debug image
    """
    result_image = image.copy()
    
    # Generate colors for extra edges
    num_edges = len(HALPE26_EXTRA_EDGES)
    edge_colors = [
        tuple([int(c * 255) for c in matplotlib.colors.hsv_to_rgb([i/float(num_edges), 1.0, 1.0])])
        for i in range(num_edges)
    ]
    
    # Draw skeleton lines (only extra keypoints)
    for ie, (start_idx, end_idx) in enumerate(HALPE26_EXTRA_EDGES):
        if scores[start_idx] > kpt_thr and scores[end_idx] > kpt_thr:
            cv2.line(result_image, 
                   (int(keypoints[start_idx, 0]), int(keypoints[start_idx, 1])),
                   (int(keypoints[end_idx, 0]), int(keypoints[end_idx, 1])),
                   edge_colors[ie], 1, lineType=cv2.LINE_AA)
    
    # Draw keypoints with numbers
    for p in range(26):
        if scores[p] > kpt_thr:
            x, y = int(keypoints[p, 0]), int(keypoints[p, 1])
            
            # Color coding
            if p < 17:  # Body keypoints
                color = (0, 0, 255)  # Red
            elif p < 23:  # Feet keypoints (17-22)
                color = (0, 255, 0)  # Green
            else:  # Additional body keypoints (23-25)
                color = (255, 0, 0)  # Blue
            
            # Draw circle
            cv2.circle(result_image, (x, y), 6, color, thickness=-1, lineType=cv2.FILLED)
            cv2.circle(result_image, (x, y), 7, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            
            # Draw number with background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.35
            thickness = 1
            text = str(p)
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw black background for number
            cv2.rectangle(result_image, 
                        (x + 10, y - text_height - 5),
                        (x + 10 + text_width + 4, y + 5),
                        (0, 0, 0), -1)
            
            # Draw number
            cv2.putText(result_image, text, (x + 12, y), 
                       font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return result_image


def main():
    parser = argparse.ArgumentParser(description="Debug Halpe26 Skeleton")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    
    config_file = Path(args.config)
    if not config_file.is_absolute():
        config_file = REPO_ROOT / config_file
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "🔍" * 40)
    print("HALPE26 DEBUG MODE - Numbered Keypoints")
    print("🔍" * 40 + "\n")
    
    # Load YOLO
    print("📦 Loading YOLO detector...")
    from ultralytics import YOLO
    yolo_filename = config["detection"]["model_path"]
    yolo_path = MODELS_DIR / "yolo" / yolo_filename
    if not yolo_path.exists():
        yolo_path = REPO_ROOT / yolo_filename
    yolo = YOLO(str(yolo_path))
    print(f"   ✅ Loaded {yolo_path.name}")
    
    # Load image
    print("\n📸 Processing image...")
    input_path = REPO_ROOT / config["input"]["path"]
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"   ❌ Could not read image: {input_path}")
        return 1
    print(f"   ✓ Loaded {input_path.name}")
    
    # Detect persons
    boxes, det_time = detect_persons_yolo(
        image, yolo, config["detection"]["confidence_threshold"]
    )
    print(f"   ✓ Detected {len(boxes)} persons")
    
    if not boxes:
        print("   ⚠️  No persons detected")
        return 1
    
    # Estimate poses
    keypoints, scores, pose_time = estimate_poses_rtmpose(
        image, boxes, config["pose_estimation"]["rtmpose_halpe26"]
    )
    print(f"   ✓ Estimated {len(keypoints)} poses with 26 keypoints")
    
    # Draw debug visualization
    result_image = draw_debug_skeleton(image, keypoints[0], scores[0], kpt_thr=0.3)
    
    # Save output
    output_path = REPO_ROOT / "demo_data/outputs/debug_halpe26.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result_image)
    
    print(f"\n✅ Saved debug image: {output_path}")
    print("\n📋 Keypoint Reference:")
    print("   BODY (Red): 0-16")
    print("   FEET (Green): 17-22")
    print("   EXTRA (Blue): 23-25")
    
    print("\n📊 Keypoint Confidences:")
    for i in range(26):
        if scores[0][i] > 0.3:
            print(f"   {HALPE26_NAMES[i]:<20} conf={scores[0][i]:.3f}")
    
    print("\n🔗 Connections being drawn:")
    for i, (start, end) in enumerate(HALPE26_EXTRA_EDGES):
        print(f"   {i+1:2d}. {start:2d} → {end:2d}  ({HALPE26_NAMES[start].split(':')[1]} → {HALPE26_NAMES[end].split(':')[1]})")
    
    print("\n💡 Review the image and identify incorrect connections!")
    print("=" * 80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
