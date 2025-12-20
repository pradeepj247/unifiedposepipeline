# Unified Pose Pipeline - New Structure

## 🎯 Simplified Design

Instead of one monolithic `udp.py`, we now have two focused scripts:

### 1️⃣ Image Demo (Quick Verification)
**Purpose:** Fast, bare-bones test to verify everything works
- **Script:** `udp_image.py`
- **Config:** `configs/udp_image.yaml`
- **What it does:**
  - Load single image
  - Detect person with YOLO
  - Estimate pose with RTMPose
  - Save annotated result
  - Show timing stats
- **Use case:** Quick smoke test after setup

### 2️⃣ Video Demo (Comprehensive Testing)  
**Purpose:** Thorough testing with full statistics and analysis
- **Script:** `udp_video.py`
- **Config:** `configs/udp_video.yaml`
- **What it does:**
  - Process video frame-by-frame
  - Track progress with progress bar
  - Collect detailed statistics
  - Export JSON keypoints (optional)
  - Show comprehensive performance report
- **Use case:** Full pipeline validation and benchmarking

---

## 🚀 Quick Start

### Fresh Colab Session

```bash
# 1. Setup (one time)
python setup_unified.py

# 2. Verify installation
python verify.py

# 3. Quick image test (5 seconds)
python udp_image.py --config configs/udp_image.yaml

# 4. Full video test (thorough)
python udp_video.py --config configs/udp_video.yaml
```

### Using Helper Script

```bash
# Setup
python run.py setup

# Verify
python run.py verify

# Quick image demo
python run.py demo image

# Full video demo
python run.py demo video
```

---

## 📋 Configuration Files

### `configs/udp_image.yaml`
```yaml
detection:
  model_path: models/yolo/yolov8n.pt  # Fast nano model
  confidence_threshold: 0.5

pose_estimation:
  model_type: rtmpose-m  # Balanced medium model
  backend: onnxruntime
  device: cuda

input:
  path: demo_data/images/sample.jpg

output:
  path: demo_data/outputs/image_result.jpg
```

### `configs/udp_video.yaml`
```yaml
detection:
  model_path: models/yolo/yolov8s.pt  # Small model
  confidence_threshold: 0.5

pose_estimation:
  model_type: rtmpose-l  # Large model for quality
  backend: onnxruntime
  device: cuda

input:
  path: demo_data/videos/dance.mp4

output:
  path: demo_data/outputs/video_result.mp4
  save_json: true  # Export keypoints
  json_path: demo_data/outputs/video_keypoints.json

processing:
  max_frames: 100  # Limit for testing (null = full video)
```

---

## 📊 Output Comparison

### Image Demo Output
```
🎯 UDP IMAGE DEMO - Quick Verification
📦 Loading YOLO detector...
   ✅ Loaded yolov8n.pt
📦 Loading RTMPose estimator...
   ✅ Loaded rtmpose-m
📸 Processing image...
   ✓ Loaded sample.jpg (640x480)
   ✓ Detected 2 persons (23.5 ms)
   ✓ Estimated 2 poses (45.2 ms)
✅ Saved result: demo_data/outputs/image_result.jpg
   Total time: 0.08s
```

### Video Demo Output
```
🎬 UDP VIDEO DEMO - Comprehensive Testing
══════════════════════════════════════════════════════════════════════
📦 Initializing Detection Module
✅ YOLO loaded: yolov8s.pt
   Confidence threshold: 0.5

📦 Initializing Pose Estimation Module
✅ RTMPose loaded: rtmpose-l
   Backend: onnxruntime
   Device: cuda

🎬 Opening Video
✅ Video opened: dance.mp4
   Resolution: 1920x1080
   FPS: 30.0
   Processing: 100 frames

⚙️  Processing Video
Processing: 100%|████████████████| 100/100 [00:05<00:00, 19.12frame/s]

📊 COMPREHENSIVE STATISTICS
══════════════════════════════════════════════════════════════════════

📹 Video Processing:
   Frames processed: 100
   Total duration: 5.23s
   Average FPS: 19.12

👤 Detection (YOLO):
   Total persons detected: 150
   Average per frame: 1.5
   Total time: 2.11s
   Average per frame: 21.1ms
   Detection FPS: 47.39

🤸 Pose Estimation (RTMPose):
   Total poses estimated: 150
   Average per frame: 1.5
   Total time: 3.12s
   Average per frame: 31.2ms
   Pose estimation FPS: 32.05

⚡ Performance:
   Best frame time: 45.2ms
   Worst frame time: 68.5ms
   Average frame time: 52.3ms
   Std deviation: 8.7ms

💾 Output:
   Video saved: demo_data/outputs/video_result.mp4
   JSON saved: demo_data/outputs/video_keypoints.json

✅ VIDEO DEMO COMPLETED SUCCESSFULLY
```

---

## 🎨 Customization

### Quick Test (10 frames)
Edit `configs/udp_video.yaml`:
```yaml
processing:
  max_frames: 10
```

### Full Video Processing
Edit `configs/udp_video.yaml`:
```yaml
processing:
  max_frames: null  # Process all frames
```

### Change Models
Edit either config file:
```yaml
detection:
  model_path: models/yolo/yolov8m.pt  # Use larger model

pose_estimation:
  model_type: rtmpose-x  # Use extra-large model
```

### Disable JSON Export
Edit `configs/udp_video.yaml`:
```yaml
output:
  save_json: false
```

---

## 📁 File Structure

```
newrepo/
├── udp_image.py              # Quick image test
├── udp_video.py              # Comprehensive video test
├── setup_unified.py          # Setup script
├── verify.py                 # Verification script
├── run.py                    # Helper launcher
│
├── configs/
│   ├── udp_image.yaml       # Image demo config
│   ├── udp_video.yaml       # Video demo config
│   └── default.yaml         # Template
│
├── lib/
│   ├── vitpose/             # ViTPose (future use)
│   └── rtmlib/              # RTMLib implementation
│
├── models/
│   ├── yolo/                # YOLO models
│   └── rtmlib/              # RTMLib models
│
└── demo_data/
    ├── images/              # Test images
    ├── videos/              # Test videos
    └── outputs/             # Results
```

---

## 💡 Workflow Recommendations

### Development/Testing
1. Run image demo first (quick feedback)
2. Verify output looks correct
3. Run video demo with `max_frames: 10`
4. Review statistics and adjust models
5. Run full video when satisfied

### Production
1. Set `max_frames: null` in video config
2. Enable JSON export if needed for analysis
3. Use larger models for better accuracy
4. Monitor performance statistics

---

## 🔄 Migration from Old Structure

**Before:**
```bash
python udp.py --config configs/vitpose_demo.yaml
python udp.py --config configs/rtmlib_demo.yaml
```

**Now:**
```bash
python udp_image.py --config configs/udp_image.yaml  # Quick test
python udp_video.py --config configs/udp_video.yaml  # Full test
```

**Benefits:**
- ✅ Clearer purpose for each script
- ✅ Faster iteration (image demo is instant)
- ✅ Better testing workflow
- ✅ More detailed video statistics
- ✅ Simpler config files

---

## ✅ Summary

| Feature | Image Demo | Video Demo |
|---------|------------|------------|
| **Speed** | ⚡⚡⚡ 5sec | ⚡ 30-60sec |
| **Purpose** | Quick test | Full validation |
| **Statistics** | Basic timing | Comprehensive |
| **JSON Export** | No | Yes (optional) |
| **Progress Bar** | No | Yes |
| **Use Case** | Smoke test | Benchmarking |

**Recommended workflow:**
1. Setup → Verify → Image Demo → Video Demo (short) → Video Demo (full)
