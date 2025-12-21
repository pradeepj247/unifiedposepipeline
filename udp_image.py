"""
UDP Image Demo - Quick image processing test

Configurable demo supporting multiple pose estimation methods:
- RTMPose: Fast ONNX-based pose estimation
- ViTPose: High-accuracy transformer-based pose estimation

Both methods use YOLOv8s for person detection.

Usage:
    python udp_image.py --config configs/udp_image.yaml
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
MODELS_DIR = PARENT_DIR / "models"  # Models stored in parent directory

# COCO skeleton edges (17 keypoints - pre-defined for performance)
COCO_EDGES = [
    (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
    (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
    (12, 14), (14, 16), (5, 6), (11, 12)
]

# Halpe26 skeleton edges (26 keypoints)
# Based on: https://github.com/Fang-Haoshu/Halpe-FullBody/
# 0-16: COCO body, 17-19: head/neck/hip, 20-25: feet
HALPE26_EDGES = [
    # COCO body connections (0-16)
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
    # Right foot connections (from ankle 16)
    (16, 21), (16, 23), (16, 25),  # RAnkle to RBigToe, RSmallToe, RHeel
    # Left foot connections (from ankle 15)
    (15, 20), (15, 22), (15, 24),  # LAnkle to LBigToe, LSmallToe, LHeel
    # Body chain connections
    (17, 18),  # Head to Neck
    (18, 19),  # Neck to Hip/Pelvis
]


def get_edge_colors(num_edges):
    """Generate rainbow colors for skeleton edges"""
    return [
        tuple([int(c * 255) for c in matplotlib.colors.hsv_to_rgb([i/float(num_edges), 1.0, 1.0])])
        for i in range(num_edges)
    ]

# Pre-compute colors
COCO_EDGE_COLORS = get_edge_colors(len(COCO_EDGES))
HALPE26_EDGE_COLORS = get_edge_colors(len(HALPE26_EDGES))


def detect_persons_yolo(image, yolo, confidence_threshold):
    """
    Stage 1: Detect persons using YOLOv8s
    
    Args:
        image: Input image (BGR)
        yolo: YOLO model instance
        confidence_threshold: Minimum confidence for detection
    
    Returns:
        boxes: List of bounding boxes [[x1, y1, x2, y2], ...]
        det_time: Detection time in milliseconds
    """
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
    """
    Stage 2a: Estimate poses using RTMPose
    
    Args:
        image: Input image (BGR)
        boxes: List of bounding boxes from detection
        config: RTMPose configuration dict
    
    Returns:
        keypoints: Array of keypoints (N x 17 x 2)
        scores: Array of confidence scores (N x 17)
        pose_time: Pose estimation time in milliseconds
    """
    sys.path.insert(0, str(REPO_ROOT / "lib"))
    from rtmlib.tools import RTMPose
    
    # Initialize RTMPose
    pose_model = RTMPose(
        onnx_model=config["pose_model_url"],
        model_input_size=tuple(config["pose_input_size"]),
        backend=config["backend"],
        device=config["device"]
    )
    
    # Run pose estimation
    t0 = time.time()
    keypoints, scores = pose_model(image, bboxes=boxes)
    pose_time = (time.time() - t0) * 1000
    
    return keypoints, scores, pose_time


def estimate_poses_vitpose(image, boxes, config):
    """
    Stage 2b: Estimate poses using ViTPose
    
    Args:
        image: Input image (BGR)
        boxes: List of bounding boxes from detection
        config: ViTPose configuration dict
    
    Returns:
        keypoints: Array of keypoints (N x 17 x 3) - last dim is confidence
        scores: Array of confidence scores (N x 17)
        pose_time: Pose estimation time in milliseconds
    """
    sys.path.insert(0, str(REPO_ROOT / "lib"))
    from vitpose.pose_only import VitPoseOnly
    
    # Initialize ViTPose
    model_path = PARENT_DIR / config["model_path"]
    pose_model = VitPoseOnly(
        model=str(model_path),
        model_name=config["model_name"],
        dataset=config["dataset"],
        device=config["device"]
    )
    
    # Convert BGR to RGB for ViTPose
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run pose estimation for each bbox
    t0 = time.time()
    all_keypoints = []
    all_scores = []
    
    for bbox in boxes:
        kpts = pose_model.inference_bbox(image_rgb, bbox)
        if len(kpts) > 0:
            # ViTPose returns (17, 3) with [y, x, conf]
            # Convert to (17, 2) [x, y] and separate scores
            keypoints_xy = np.stack([kpts[:, 1], kpts[:, 0]], axis=1)  # Swap to [x, y]
            scores = kpts[:, 2]
            all_keypoints.append(keypoints_xy)
            all_scores.append(scores)
    
    pose_time = (time.time() - t0) * 1000
    
    if all_keypoints:
        keypoints = np.array(all_keypoints)
        scores = np.array(all_scores)
    else:
        keypoints = np.array([])
        scores = np.array([])
    
    return keypoints, scores, pose_time


def draw_skeleton_unified(image, keypoints, scores, kpt_thr=0.5):
    """
    Unified colorful skeleton drawing for both RTMPose and ViTPose
    Automatically detects COCO (17 kpts) or Halpe26 (26 kpts)
    
    Args:
        image: Input image (BGR)
        keypoints: Array of keypoints (N x 17 x 2) or (N x 26 x 2) in [x, y] format
        scores: Array of confidence scores (N x 17) or (N x 26)
        kpt_thr: Confidence threshold for drawing
    
    Returns:
        result_image: Annotated image with colorful skeleton
    """
    result_image = image.copy()
    
    for kpts, scrs in zip(keypoints, scores):
        num_keypoints = len(kpts)
        
        # Select skeleton and colors based on keypoint count
        if num_keypoints == 26:
            edges = HALPE26_EDGES
            edge_colors = HALPE26_EDGE_COLORS
        else:  # Default to COCO (17 keypoints)
            edges = COCO_EDGES
            edge_colors = COCO_EDGE_COLORS
        
        # Draw skeleton lines first (so keypoints are on top)
        for ie, (start_idx, end_idx) in enumerate(edges):
            if start_idx < num_keypoints and end_idx < num_keypoints:
                if scrs[start_idx] > kpt_thr and scrs[end_idx] > kpt_thr:
                    cv2.line(result_image, 
                           (int(kpts[start_idx, 0]), int(kpts[start_idx, 1])),
                           (int(kpts[end_idx, 0]), int(kpts[end_idx, 1])),
                           edge_colors[ie], 2, lineType=cv2.LINE_AA)
        
        # Draw keypoints (circles) on top with color coding
        for p in range(num_keypoints):
            if scrs[p] > kpt_thr:
                # Color coding for Halpe26
                if num_keypoints == 26:
                    if p < 17:  # Body keypoints
                        color = (0, 0, 255)  # Red
                    elif p < 23:  # Feet keypoints (17-22)
                        color = (0, 255, 0)  # Green
                    else:  # Additional body keypoints (23-25)
                        color = (255, 0, 0)  # Blue
                else:
                    color = (0, 0, 255)  # Red for COCO
                
                cv2.circle(result_image, 
                         (int(kpts[p, 0]), int(kpts[p, 1])), 
                         4, color, thickness=-1, lineType=cv2.FILLED)
    
    return result_image


def draw_results(image, boxes, keypoints, scores, method, draw_bbox=True):
    """
    Draw detection boxes and pose keypoints on image
    
    Args:
        image: Input image (BGR)
        boxes: List of bounding boxes
        keypoints: Array of keypoints
        scores: Array of confidence scores
        method: Pose estimation method used (for display only)
        draw_bbox: Whether to draw bounding boxes
    
    Returns:
        result_image: Annotated image
    """
    result_image = image.copy()
    
    # Draw bounding boxes (optional)
    if draw_bbox:
        for box in boxes:
            cv2.rectangle(result_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    
    # Draw skeleton (unified colorful visualization)
    if len(keypoints) > 0:
        result_image = draw_skeleton_unified(result_image, keypoints, scores, kpt_thr=0.5)
    
    return result_image


def main():
    parser = argparse.ArgumentParser(description="UDP Image Demo - Configurable pose estimation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    
    # Load config
    config_file = Path(args.config)
    if not config_file.is_absolute():
        config_file = REPO_ROOT / config_file
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required config sections
    if "input" not in config:
        print(f"❌ Error: Config file missing 'input' section")
        print(f"   Config file: {config_file}")
        print(f"   Available sections: {list(config.keys())}")
        return 1
    
    # Get method with default fallback
    method = config.get("pose_estimation", {}).get("method", "rtmpose")
    
    # Normalize method name to lowercase for comparison
    method = method.lower().strip()
    
    print("\n" + "🎯" * 35)
    print(f"UDP IMAGE DEMO - {method.upper()} Mode")
    print("🎯" * 35 + "\n")
    
    # Stage 1: Initialize YOLO detector
    print("📦 Stage 1: Loading YOLO detector...")
    from ultralytics import YOLO
    
    yolo_filename = config["detection"]["model_path"]
    yolo_path = MODELS_DIR / "yolo" / yolo_filename
    if not yolo_path.exists():
        yolo_path = REPO_ROOT / yolo_filename
    
    yolo = YOLO(str(yolo_path))
    print(f"   ✅ Loaded {yolo_path.name}")
    
    # Stage 2: Initialize pose estimator
    print(f"\n📦 Stage 2: Loading {method.upper()} pose estimator...")
    if method == "rtmpose":
        rtm_cfg = config["pose_estimation"]["rtmpose"]
        input_size = rtm_cfg["pose_input_size"]
        model_size = "L" if input_size[0] >= 288 else "M" if input_size[0] >= 256 else "S"
        print(f"   Model: RTMPose-{model_size} ({input_size[0]}×{input_size[1]}, ONNX, 17 keypoints)")
    elif method == "rtmpose_halpe26":
        rtm_cfg = config["pose_estimation"]["rtmpose_halpe26"]
        input_size = rtm_cfg["pose_input_size"]
        model_size = "L" if input_size[0] >= 288 else "M" if input_size[0] >= 256 else "S"
        print(f"   Model: RTMPose-{model_size} Halpe26 ({input_size[0]}×{input_size[1]}, ONNX, 26 keypoints)")
    elif method == "vitpose":
        vitpose_cfg = config["pose_estimation"]["vitpose"]
        print(f"   Model: ViTPose-{vitpose_cfg['model_name'].upper()} (PyTorch)")
    print(f"   ✅ Pose estimator ready")
    
    # Load image
    print("\n📸 Processing image...")
    input_path = REPO_ROOT / config["input"]["path"]
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"   ❌ Could not read image: {input_path}")
        return 1
    print(f"   ✓ Loaded {input_path.name} ({image.shape[1]}x{image.shape[0]})")
    
    # Detect persons with YOLO
    boxes, det_time = detect_persons_yolo(
        image, yolo, config["detection"]["confidence_threshold"]
    )
    print(f"   ✓ Detected {len(boxes)} persons ({det_time:.1f} ms)")
    
    # Estimate poses
    if boxes:
        if method == "rtmpose":
            keypoints, scores, pose_time = estimate_poses_rtmpose(
                image, boxes, config["pose_estimation"]["rtmpose"]
            )
        elif method == "rtmpose_halpe26":
            keypoints, scores, pose_time = estimate_poses_rtmpose(
                image, boxes, config["pose_estimation"]["rtmpose_halpe26"]
            )
        elif method == "vitpose":
            keypoints, scores, pose_time = estimate_poses_vitpose(
                image, boxes, config["pose_estimation"]["vitpose"]
            )
        else:
            print(f"   ❌ Unknown method: {method}")
            print(f"   Available methods: rtmpose, rtmpose_halpe26, vitpose")
            return 1
        
        num_kpts = len(keypoints[0]) if len(keypoints) > 0 else 0
        print(f"   ✓ Estimated {len(keypoints)} poses with {num_kpts} keypoints each ({pose_time:.1f} ms)")
        
        # Draw results
        result_image = draw_results(image, boxes, keypoints, scores, method)
    else:
        print(f"   ⚠️  No persons detected")
        result_image = image.copy()
    
    # Save output
    output_path = REPO_ROOT / config["output"]["path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result_image)
    print(f"\n✅ Saved result: {output_path}")
    print(f"   Total time: {(det_time + (pose_time if boxes else 0)) / 1000:.2f}s\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
