"""
UDP - Unified Demo Pipeline

Main entry point for running pose estimation demos.
Accepts a YAML configuration file that specifies:
- Detection method (YOLO)
- Pose estimation method (ViTPose, RTMPose, etc.)
- Model paths and config files
- Input/output paths
- Processing parameters

Usage:
    python udp.py --config configs/vitpose_demo.yaml
    python udp.py --config configs/rtmlib_demo.yaml
    python udp.py --config configs/tracking_pose_demo.yaml
"""

import sys
import argparse
from pathlib import Path
import yaml
import time
from typing import Dict, Any, Optional
import cv2
import numpy as np

# ============================================
# Configuration
# ============================================
REPO_ROOT = Path(__file__).parent
LIB_DIR = REPO_ROOT / "lib"
MODELS_DIR = REPO_ROOT / "models"
DEMO_DATA_DIR = REPO_ROOT / "demo_data"


# ============================================
# Pipeline Classes
# ============================================
class DetectionModule:
    """Object detection using YOLO"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        
    def initialize(self):
        """Initialize YOLO model"""
        from ultralytics import YOLO
        
        model_path = Path(self.config["model_path"])
        if not model_path.is_absolute():
            model_path = REPO_ROOT / model_path
        
        print(f"   📦 Loading YOLO model: {model_path.name}")
        self.model = YOLO(str(model_path))
        print(f"   ✅ YOLO initialized")
    
    def detect(self, image: np.ndarray) -> list:
        """Detect persons in image"""
        results = self.model(image, classes=[0], verbose=False)  # class 0 = person
        boxes = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                boxes.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(conf)
                })
        
        return boxes


class ViTPoseModule:
    """ViTPose-based pose estimation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        
    def initialize(self):
        """Initialize ViTPose model"""
        sys.path.insert(0, str(LIB_DIR))
        
        model_path = Path(self.config["model_path"])
        if not model_path.is_absolute():
            model_path = REPO_ROOT / model_path
        
        print(f"   📦 Loading ViTPose model: {model_path.name}")
        
        # Import and initialize ViTPose
        from vitpose import ViTPoseModel
        
        model_name = self.config.get("model_name", "vitpose-b")
        dataset = self.config.get("dataset", "coco")
        
        self.model = ViTPoseModel(model_name, str(model_path), dataset)
        print(f"   ✅ ViTPose initialized ({model_name}, {dataset})")
    
    def estimate(self, image: np.ndarray, boxes: list) -> list:
        """Estimate poses for detected persons"""
        if not boxes:
            return []
        
        keypoints_list = []
        for box_info in boxes:
            bbox = box_info["bbox"]
            keypoints = self.model.predict(image, [bbox])
            keypoints_list.append({
                "bbox": bbox,
                "keypoints": keypoints,
                "confidence": box_info["confidence"]
            })
        
        return keypoints_list


class RTMPoseModule:
    """RTMPose-based pose estimation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        
    def initialize(self):
        """Initialize RTMPose model"""
        from rtmlib import Body, draw_skeleton
        
        model_type = self.config.get("model_type", "rtmpose-l")
        backend = self.config.get("backend", "onnxruntime")
        
        print(f"   📦 Loading RTMPose model: {model_type}")
        
        self.model = Body(
            pose=model_type,
            to_openpose=False,
            backend=backend,
            device='cuda'
        )
        
        print(f"   ✅ RTMPose initialized ({model_type}, {backend})")
    
    def estimate(self, image: np.ndarray, boxes: list) -> list:
        """Estimate poses for detected persons"""
        if not boxes:
            return []
        
        keypoints_list = []
        for box_info in boxes:
            bbox = box_info["bbox"]
            keypoints, scores = self.model(image, [bbox])
            keypoints_list.append({
                "bbox": bbox,
                "keypoints": keypoints[0] if len(keypoints) > 0 else None,
                "scores": scores[0] if len(scores) > 0 else None,
                "confidence": box_info["confidence"]
            })
        
        return keypoints_list


# ============================================
# Main Pipeline
# ============================================
class UnifiedPipeline:
    """Main unified pose estimation pipeline"""
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.detector = None
        self.pose_estimator = None
        self.stats = {
            "frames_processed": 0,
            "total_time": 0,
            "detection_time": 0,
            "pose_time": 0,
        }
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = REPO_ROOT / config_file
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Loaded config: {config_file.name}")
        return config
    
    def initialize(self):
        """Initialize all pipeline components"""
        print("\n" + "=" * 70)
        print("🚀 Initializing Pipeline Components")
        print("=" * 70)
        
        # Initialize detector
        print("\n📍 Detection Module:")
        detection_config = self.config["detection"]
        self.detector = DetectionModule(detection_config)
        self.detector.initialize()
        
        # Initialize pose estimator
        print("\n📍 Pose Estimation Module:")
        pose_config = self.config["pose_estimation"]
        pose_type = pose_config["type"]
        
        if pose_type == "vitpose":
            self.pose_estimator = ViTPoseModule(pose_config)
        elif pose_type == "rtmlib" or pose_type == "rtmpose":
            self.pose_estimator = RTMPoseModule(pose_config)
        else:
            raise ValueError(f"Unknown pose estimation type: {pose_type}")
        
        self.pose_estimator.initialize()
        
        print("\n✅ All components initialized\n")
    
    def process_image(self, image_path: str, output_path: str):
        """Process a single image"""
        print(f"📸 Processing image: {Path(image_path).name}")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Detect persons
        t0 = time.time()
        boxes = self.detector.detect(image)
        t1 = time.time()
        print(f"   ✓ Detected {len(boxes)} persons ({(t1-t0)*1000:.1f} ms)")
        
        # Estimate poses
        t2 = time.time()
        poses = self.pose_estimator.estimate(image, boxes)
        t3 = time.time()
        print(f"   ✓ Estimated {len(poses)} poses ({(t3-t2)*1000:.1f} ms)")
        
        # Draw results
        result_image = self.draw_results(image.copy(), poses)
        
        # Save output
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_file), result_image)
        print(f"   ✅ Saved: {output_file}")
        
        self.stats["frames_processed"] += 1
        self.stats["detection_time"] += (t1 - t0)
        self.stats["pose_time"] += (t3 - t2)
        self.stats["total_time"] += (t3 - t0)
    
    def process_video(self, video_path: str, output_path: str, max_frames: Optional[int] = None):
        """Process a video file"""
        print(f"🎬 Processing video: {Path(video_path).name}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"   Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
        
        # Setup output video
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
        
        # Process frames
        frame_idx = 0
        try:
            from tqdm import tqdm
            pbar = tqdm(total=total_frames, desc="Processing")
        except ImportError:
            pbar = None
        
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break
            
            # Detect and estimate
            t0 = time.time()
            boxes = self.detector.detect(frame)
            t1 = time.time()
            poses = self.pose_estimator.estimate(frame, boxes)
            t2 = time.time()
            
            # Draw results
            result_frame = self.draw_results(frame.copy(), poses)
            out.write(result_frame)
            
            # Update stats
            frame_idx += 1
            self.stats["frames_processed"] += 1
            self.stats["detection_time"] += (t1 - t0)
            self.stats["pose_time"] += (t2 - t1)
            self.stats["total_time"] += (t2 - t0)
            
            if pbar:
                pbar.update(1)
        
        if pbar:
            pbar.close()
        
        cap.release()
        out.release()
        
        print(f"   ✅ Saved: {output_file}")
        print(f"   📊 Processed {frame_idx} frames")
    
    def draw_results(self, image: np.ndarray, poses: list) -> np.ndarray:
        """Draw detection and pose results on image"""
        # Draw bounding boxes and keypoints
        for pose in poses:
            bbox = pose["bbox"]
            keypoints = pose.get("keypoints")
            
            # Draw bbox
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Draw keypoints
            if keypoints is not None:
                if isinstance(keypoints, np.ndarray):
                    for kp in keypoints:
                        if len(kp) >= 2:
                            x, y = int(kp[0]), int(kp[1])
                            if x > 0 and y > 0:
                                cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
        
        return image
    
    def print_stats(self):
        """Print processing statistics"""
        print("\n" + "=" * 70)
        print("📊 Processing Statistics")
        print("=" * 70)
        
        n = self.stats["frames_processed"]
        if n > 0:
            print(f"\n   Frames Processed: {n}")
            print(f"   Total Time: {self.stats['total_time']:.2f} s")
            print(f"   Average FPS: {n / self.stats['total_time']:.2f}")
            print(f"\n   Detection Time: {self.stats['detection_time']:.2f} s ({self.stats['detection_time']/n*1000:.1f} ms/frame)")
            print(f"   Pose Time: {self.stats['pose_time']:.2f} s ({self.stats['pose_time']/n*1000:.1f} ms/frame)")
        else:
            print("   No frames processed")
        print()
    
    def run(self):
        """Main execution"""
        try:
            # Initialize
            self.initialize()
            
            # Get input/output config
            input_config = self.config["input"]
            output_config = self.config["output"]
            
            input_path = Path(input_config["path"])
            if not input_path.is_absolute():
                input_path = REPO_ROOT / input_path
            
            output_path = Path(output_config["path"])
            if not output_path.is_absolute():
                output_path = REPO_ROOT / output_path
            
            # Check input type
            input_type = input_config.get("type", "auto")
            if input_type == "auto":
                if input_path.suffix.lower() in [".mp4", ".avi", ".mov"]:
                    input_type = "video"
                else:
                    input_type = "image"
            
            # Process
            print("=" * 70)
            print("🎯 Starting Processing")
            print("=" * 70)
            
            if input_type == "image":
                self.process_image(str(input_path), str(output_path))
            else:
                max_frames = self.config.get("processing", {}).get("max_frames")
                self.process_video(str(input_path), str(output_path), max_frames)
            
            # Print stats
            self.print_stats()
            
            print("✅ Pipeline completed successfully!\n")
            return 0
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return 1


# ============================================
# Main Entry Point
# ============================================
def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="UDP - Unified Demo Pipeline for Pose Estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python udp.py --config configs/vitpose_demo.yaml
  python udp.py --config configs/rtmlib_demo.yaml
  python udp.py --config configs/tracking_pose_demo.yaml
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    
    args = parser.parse_args()
    
    print("\n" + "🎯" * 35)
    print("UDP - UNIFIED DEMO PIPELINE")
    print("🎯" * 35 + "\n")
    
    # Create and run pipeline
    pipeline = UnifiedPipeline(args.config)
    return pipeline.run()


if __name__ == "__main__":
    sys.exit(main())
