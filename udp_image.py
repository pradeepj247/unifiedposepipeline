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
    
    # Initialize RTMLib Body (handles detection + pose)
    print("📦 Loading RTMLib Body (detector + pose estimator)...")
    sys.path.insert(0, str(REPO_ROOT / "lib"))
    from rtmlib import Body, draw_skeleton
    
    body = Body(
        pose=config["pose_estimation"]["pose_model_url"],
        pose_input_size=tuple(config["pose_estimation"]["pose_input_size"]),
        backend=config["pose_estimation"]["backend"],
        device=config["pose_estimation"]["device"]
    )
    print(f"   ✅ Loaded RTMLib Body pipeline")
    
    # Load image
    print("\n📸 Processing image...")
    input_path = REPO_ROOT / config["input"]["path"]
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"   ❌ Could not read image: {input_path}")
        return 1
    print(f"   ✓ Loaded {input_path.name} ({image.shape[1]}x{image.shape[0]})")
    
    # Run detection + pose estimation
    t0 = time.time()
    keypoints, scores = body(image)
    t1 = time.time()
    print(f"   ✓ Detected and estimated {len(keypoints)} persons ({(t1-t0)*1000:.1f} ms)")
    
    # Draw results
    if len(keypoints) > 0:
        result_image = image.copy()
        result_image = draw_skeleton(result_image, keypoints, scores, kpt_thr=0.5)
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
