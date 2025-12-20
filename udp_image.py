"""
UDP Image Demo - Quick image processing test

Bare-bones demo for quick verification:
- Load single image
- Detect person with YOLO
- Estimate pose with RTMPose
- Save annotated result

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

REPO_ROOT = Path(__file__).parent
PARENT_DIR = REPO_ROOT.parent
MODELS_DIR = PARENT_DIR / "models"  # Models stored in parent directory

def main():
    parser = argparse.ArgumentParser(description="UDP Image Demo - Quick verification")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    
    # Load config
    config_file = Path(args.config)
    if not config_file.is_absolute():
        config_file = REPO_ROOT / config_file
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "🎯" * 35)
    print("UDP IMAGE DEMO - Quick Verification")
    print("🎯" * 35 + "\n")
    
    # Initialize YOLO
    print("📦 Loading YOLO detector...")
    from ultralytics import YOLO
    
    # Look for model in parent/models directory first, fallback to repo
    yolo_filename = config["detection"]["model_path"]
    yolo_path = MODELS_DIR / "yolo" / yolo_filename
    if not yolo_path.exists():
        yolo_path = REPO_ROOT / yolo_filename
    
    yolo = YOLO(str(yolo_path))
    print(f"   ✅ Loaded {yolo_path.name}")
    
    # Initialize RTMLib
    print("\n📦 Loading RTMPose estimator...")
    sys.path.insert(0, str(REPO_ROOT / "lib"))
    from rtmlib import Body
    
    pose_model = Body(
        pose=config["pose_estimation"]["pose_model_url"],
        pose_input_size=tuple(config["pose_estimation"]["pose_input_size"]),
        backend=config["pose_estimation"]["backend"],
        device=config["pose_estimation"]["device"]
    )
    print(f"   ✅ Loaded RTMPose")
    
    # Load image
    print("\n📸 Processing image...")
    input_path = REPO_ROOT / config["input"]["path"]
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"   ❌ Could not read image: {input_path}")
        return 1
    print(f"   ✓ Loaded {input_path.name} ({image.shape[1]}x{image.shape[0]})")
    
    # Detect persons
    t0 = time.time()
    results = yolo(image, classes=[0], verbose=False)
    boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
    t1 = time.time()
    print(f"   ✓ Detected {len(boxes)} persons ({(t1-t0)*1000:.1f} ms)")
    
    # Estimate poses
    if boxes:
        t2 = time.time()
        keypoints, scores = pose_model(image, boxes)
        t3 = time.time()
        print(f"   ✓ Estimated {len(keypoints)} poses ({(t3-t2)*1000:.1f} ms)")
        
        # Draw results
        from rtmlib import draw_skeleton
        result_image = image.copy()
        for box in boxes:
            cv2.rectangle(result_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        result_image = draw_skeleton(result_image, keypoints, scores, kpt_thr=0.3)
    else:
        print(f"   ⚠️  No persons detected")
        result_image = image.copy()
    
    # Save output
    output_path = REPO_ROOT / config["output"]["path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result_image)
    print(f"\n✅ Saved result: {output_path}")
    print(f"   Total time: {(time.time()-t0):.2f}s\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
