"""
Debug UDP Video - Halpe26 Single Frame Test

Tests Halpe26 pose estimation on a single frame with numbered keypoints.
Useful for debugging joint semantics and verifying the halpe26_2d.json definition.

Usage:
    python debug_udp_video_h26.py --video demo_data/videos/dance.mp4
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

REPO_ROOT = Path(__file__).parent
PARENT_DIR = REPO_ROOT.parent
MODELS_DIR = PARENT_DIR / "models"

# Halpe26 skeleton edges (26 keypoints)
HALPE26_EDGES = [
    # COCO body connections (0-16)
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
    # Additional body connections (head and pelvis)
    (0, 17),  # Nose to head_top
    (0, 18),  # Nose to neck
    (11, 19),  # Left_hip to pelvis
    (12, 19),  # Right_hip to pelvis
    # Left foot connections (from ankle 15)
    (15, 20), (15, 22), (15, 24),  # LAnkle to LBigToe, LSmallToe, LHeel
    # Right foot connections (from ankle 16)
    (16, 21), (16, 23), (16, 25),  # RAnkle to RBigToe, RSmallToe, RHeel
    # Foot internal connections (optional for better visualization)
    (20, 22), (21, 23),  # Toe connections
    (22, 24), (23, 25),  # Toe to heel
]


def main():
    parser = argparse.ArgumentParser(description="Debug Halpe26 - Single Frame Test")
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--output', type=str, default='demo_data/outputs/debug_halpe26_frame1.png',
                        help='Output image path')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("DEBUG: Halpe26 Single Frame Test")
    print("=" * 70)
    
    # Initialize YOLO
    print("\n📦 Loading YOLO detector...")
    from ultralytics import YOLO
    yolo_path = MODELS_DIR / "yolo" / "yolov8s.pt"
    yolo = YOLO(str(yolo_path))
    print(f"   ✅ Loaded {yolo_path.name}")
    
    # Initialize RTMPose Halpe26
    print("\n📦 Loading RTMPose Halpe26...")
    sys.path.insert(0, str(REPO_ROOT / "lib"))
    from rtmlib.tools import RTMPose
    
    #model_path = "/content/models/rtmlib/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288_halpe26.onnx"
    model_path = "/content/models/rtmlib/rtmpose-l-halpe26-384x288.onnx"
    pose_model = RTMPose(
        onnx_model=model_path,
        model_input_size=(288, 384),
        backend="onnxruntime",
        device="cuda"
    )
    print(f"   ✅ RTMPose-L Halpe26 loaded (288x384, 26 keypoints)")
    
    # Load video and get first frame
    print(f"\n🎬 Loading video: {args.video}")
    video_path = Path(args.video)
    if not video_path.exists():
        video_path = REPO_ROOT / args.video
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return 1
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Could not read first frame")
        return 1
    
    height, width = frame.shape[:2]
    print(f"   ✅ Frame loaded: {width}x{height}")
    
    # Stage 1: Detect person
    print("\n🎯 Stage 1: Detecting person...")
    results = yolo(frame, classes=[0], verbose=False)
    
    boxes = []
    for result in results:
        for box in result.boxes:
            if box.conf[0] >= 0.5:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
    
    if not boxes:
        print("❌ No person detected")
        return 1
    
    # Use largest box
    boxes_sorted = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    bbox = boxes_sorted[0]
    print(f"   ✅ Detected person: bbox {bbox}")
    
    # Stage 2: Estimate pose
    print("\n🎯 Stage 2: Estimating pose (Halpe26)...")
    keypoints, scores = pose_model(frame, bboxes=[bbox])
    
    if len(keypoints) == 0:
        print("❌ No pose detected")
        return 1
    
    kpts = keypoints[0]  # (26, 2)
    score = scores[0]    # (26,)
    
    print(f"   ✅ Pose detected: {kpts.shape[0]} keypoints")
    print(f"   Mean confidence: {score.mean():.3f}")
    
    # Draw on frame
    result_frame = frame.copy()
    
    # Draw bbox
    cv2.rectangle(result_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    
    # Draw skeleton
    for i, (start, end) in enumerate(HALPE26_EDGES):
        if start < len(kpts) and end < len(kpts):
            if score[start] > 0.3 and score[end] > 0.3:
                pt1 = (int(kpts[start, 0]), int(kpts[start, 1]))
                pt2 = (int(kpts[end, 0]), int(kpts[end, 1]))
                cv2.line(result_frame, pt1, pt2, (0, 255, 255), 2)
    
    # Draw keypoints with numbers
    for idx in range(len(kpts)):
        if score[idx] > 0.3:
            x, y = int(kpts[idx, 0]), int(kpts[idx, 1])
            
            # Draw point
            cv2.circle(result_frame, (x, y), 5, (0, 0, 255), -1)
            
            # Draw number (white text with black outline for visibility)
            text = str(idx)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            
            # Get text size for positioning
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Position text slightly offset from point
            text_x = x + 8
            text_y = y - 8
            
            # Draw black outline
            cv2.putText(result_frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            # Draw white text
            cv2.putText(result_frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    # Add title
    cv2.putText(result_frame, "Halpe26 - Frame 1 (Joint Numbers)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Save output
    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result_frame)
    
    print(f"\n✅ Saved: {output_path}")
    
    # Print keypoint coordinates for reference
    print("\n" + "=" * 70)
    print("KEYPOINT COORDINATES (for verification)")
    print("=" * 70)
    
    # Expected joint names from halpe26_2d.json
    joint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
        "head_top", "neck", "pelvis",
        "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe",
        "left_heel", "right_heel"
    ]
    
    for idx in range(len(kpts)):
        x, y = kpts[idx, 0], kpts[idx, 1]
        conf = score[idx]
        name = joint_names[idx] if idx < len(joint_names) else "unknown"
        status = "✓" if conf > 0.3 else "✗"
        print(f"  {status} Joint {idx:2d} ({name:20s}): x={x:6.1f}, y={y:6.1f}, conf={conf:.3f}")
    
    print("\n" + "=" * 70)
    print("📋 VERIFICATION CHECKLIST")
    print("=" * 70)
    print("""
Please check the output image and verify:

1. Joint 0 (nose) - Should be at nose position
2. Joints 1-4 (eyes, ears) - Should be on face
3. Joints 5-6 (shoulders) - Should be at shoulder positions
4. Joints 7-10 (elbows, wrists) - Should be on arms
5. Joints 11-12 (hips) - Should be at hip positions
6. Joints 13-16 (knees, ankles) - Should be on legs
7. Joints 17-19 (head_top, neck, pelvis) - Additional body keypoints
8. Joints 20-25 (foot keypoints) - Should be on feet (toes and heels)

If any joint numbers don't match their expected body positions,
we need to update the joint_definitions/halpe26_2d.json file!
""")
    print("=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
