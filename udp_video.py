"""
UDP Video Demo - Comprehensive video processing test

Thorough testing with:
- Frame-by-frame processing
- Progress tracking
- Performance statistics
- Optional JSON export
- Quality assessment

Usage:
    python udp_video.py --config configs/udp_video.yaml
"""

import sys
import argparse
from pathlib import Path
import yaml
import time
import json
import cv2
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).parent
PARENT_DIR = REPO_ROOT.parent
MODELS_DIR = PARENT_DIR / "models"  # Models stored in parent directory

def main():
    parser = argparse.ArgumentParser(description="UDP Video Demo - Comprehensive testing")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    
    # Load config
    config_file = Path(args.config)
    if not config_file.is_absolute():
        config_file = REPO_ROOT / config_file
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "🎬" * 35)
    print("UDP VIDEO DEMO - Comprehensive Testing")
    print("🎬" * 35 + "\n")
    
    # Stage 1: Initialize YOLO detector
    print("=" * 70)
    print("📦 Stage 1: Initializing YOLO Detector")
    print("=" * 70)
    from ultralytics import YOLO
    
    yolo_config_path = config["detection"]["model_path"]
    if "/" in yolo_config_path:
        yolo_filename = yolo_config_path.split("/")[-1]
    else:
        yolo_filename = yolo_config_path
    
    yolo_path = MODELS_DIR / "yolo" / yolo_filename
    if not yolo_path.exists():
        yolo_path = REPO_ROOT / yolo_config_path
    
    yolo = YOLO(str(yolo_path))
    print(f"✅ YOLO loaded: {yolo_path.name}")
    print(f"   Confidence threshold: {config['detection']['confidence_threshold']}")
    
    # Stage 2: Initialize RTMPose (pose only, no detector)
    print("\n" + "=" * 70)
    print("📦 Stage 2: Initializing RTMPose Estimator")
    print("=" * 70)
    sys.path.insert(0, str(REPO_ROOT / "lib"))
    from rtmlib.tools import RTMPose
    from rtmlib import draw_skeleton
    
    pose_model = RTMPose(
        onnx_model=config["pose_estimation"]["pose_model_url"],
        model_input_size=tuple(config["pose_estimation"]["pose_input_size"]),
        backend=config["pose_estimation"]["backend"],
        device=config["pose_estimation"]["device"]
    )
    print(f"✅ RTMPose loaded")
    print(f"   Backend: {config['pose_estimation']['backend']}")
    print(f"   Device: {config['pose_estimation']['device']}")
    
    # Open video
    print("\n" + "=" * 70)
    print("🎬 Opening Video")
    print("=" * 70)
    input_path = REPO_ROOT / config["input"]["path"]
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"❌ Could not open video: {input_path}")
        return 1
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = config["processing"].get("max_frames")
    
    if max_frames:
        frames_to_process = min(total_frames, max_frames)
    else:
        frames_to_process = total_frames
    
    print(f"✅ Video opened: {input_path.name}")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps:.1f}")
    print(f"   Total frames: {total_frames}")
    print(f"   Processing: {frames_to_process} frames")
    
    # Setup output
    output_path = REPO_ROOT / config["output"]["path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Processing stats
    stats = {
        "frames_processed": 0,
        "persons_detected": 0,
        "poses_estimated": 0,
        "detection_times": [],
        "pose_times": [],
        "total_times": [],
    }
    
    # Optional JSON export
    save_json = config["output"].get("save_json", False)
    if save_json:
        json_data = {"frames": []}
    
    # Process video
    print("\n" + "=" * 70)
    print("⚙️  Processing Video")
    print("=" * 70)
    
    pbar = tqdm(total=frames_to_process, desc="Processing", unit="frame")
    frame_idx = 0
    
    while frame_idx < frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_start = time.time()
        
        # Stage 1: Detect persons with YOLO
        det_start = time.time()
        results = yolo(frame, classes=[0], verbose=False)
        boxes = []
        for result in results:
            for box in result.boxes:
                if box.conf[0] >= config["detection"]["confidence_threshold"]:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    boxes.append([int(x1), int(y1), int(x2), int(y2)])
        det_time = time.time() - det_start
        
        num_persons = len(boxes)
        stats["persons_detected"] += num_persons
        
        # Stage 2: Estimate poses with RTMPose
        result_frame = frame.copy()
        if boxes:
            pose_start = time.time()
            keypoints, scores = pose_model(frame, bboxes=boxes)
            pose_time = time.time() - pose_start
            stats["poses_estimated"] += len(keypoints)
            
            # Draw bounding boxes
            for box in boxes:
                cv2.rectangle(result_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            # Draw skeleton
            result_frame = draw_skeleton(result_frame, keypoints, scores, kpt_thr=0.5)
            
            # Save to JSON if requested
            if save_json:
                frame_data = {
                    "frame_id": frame_idx,
                    "persons": []
                }
                for i, (box, kpts, scrs) in enumerate(zip(boxes, keypoints, scores)):
                    person_data = {
                        "person_id": i,
                        "bbox": box,
                        "keypoints": kpts.tolist(),
                        "scores": scrs.tolist()
                    }
                    frame_data["persons"].append(person_data)
                json_data["frames"].append(frame_data)
        else:
            pose_time = 0
        
        out.write(result_frame)
        
        # Update stats
        total_time = time.time() - frame_start
        stats["detection_times"].append(det_time)
        stats["pose_times"].append(pose_time)
        stats["total_times"].append(total_time)
        stats["frames_processed"] += 1
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    # Save JSON if requested
    if save_json:
        json_path = REPO_ROOT / config["output"].get("json_path", "demo_data/outputs/keypoints.json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"\n✅ JSON data saved: {json_path}")
    
    # Print comprehensive statistics
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE STATISTICS")
    print("=" * 70)
    
    n = stats["frames_processed"]
    total_time = sum(stats["total_times"])
    det_time = sum(stats["detection_times"])
    pose_time = sum(stats["pose_times"])
    
    print(f"\n📹 Video Processing:")
    print(f"   Frames processed: {n}")
    print(f"   Total duration: {total_time:.2f}s")
    print(f"   Average FPS: {n/total_time:.2f}")
    
    print(f"\n👤 Detection (YOLO):")
    print(f"   Total persons detected: {stats['persons_detected']}")
    print(f"   Average per frame: {stats['persons_detected']/n:.1f}")
    print(f"   Total time: {det_time:.2f}s")
    print(f"   Average per frame: {det_time/n*1000:.1f}ms")
    print(f"   Detection FPS: {n/det_time:.2f}")
    
    print(f"\n🤸 Pose Estimation (RTMPose):")
    print(f"   Total poses estimated: {stats['poses_estimated']}")
    print(f"   Average per frame: {stats['poses_estimated']/n:.1f}")
    print(f"   Total time: {pose_time:.2f}s")
    print(f"   Average per frame: {pose_time/n*1000:.1f}ms")
    if pose_time > 0:
        print(f"   Pose estimation FPS: {n/pose_time:.2f}")
    
    print(f"\n⚡ Performance:")
    print(f"   Best frame time: {min(stats['total_times'])*1000:.1f}ms")
    print(f"   Worst frame time: {max(stats['total_times'])*1000:.1f}ms")
    print(f"   Average frame time: {total_time/n*1000:.1f}ms")
    print(f"   Std deviation: {np.std(stats['total_times'])*1000:.1f}ms")
    
    print(f"\n💾 Output:")
    print(f"   Video saved: {output_path}")
    if save_json:
        print(f"   JSON saved: {json_path}")
    
    print("\n" + "=" * 70)
    print("✅ VIDEO DEMO COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
