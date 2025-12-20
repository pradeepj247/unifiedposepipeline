# Unified Pose Estimation Pipeline

A unified, easy-to-use pipeline for human pose estimation combining multiple state-of-the-art methods:
- **ViTPose**: Vision Transformer-based pose estimation
- **RTMPose**: Real-time pose estimation 
- **YOLO**: Object detection for person localization
- **BoxMOT**: Multi-object tracking (optional)

## 🚀 Quick Start

### 1. Setup (First Time)

```bash
# On Google Colab
python setup_unified.py

# This will:
# - Mount Google Drive (Colab only)
# - Install all dependencies
# - Download models
# - Setup demo data
# - Verify installation
```

### 2. Verify Environment

```bash
python verify.py
```

This checks:
- ✅ All library imports
- ✅ Model files
- ✅ Demo data
- ✅ CUDA/GPU availability
- ✅ Functional tests

### 3. Run Demo

```bash
# ViTPose on image
python udp.py --config configs/vitpose_demo.yaml

# RTMPose on video
python udp.py --config configs/udp.yaml

# Process video with frame limit
python udp.py --config configs/video_demo.yaml
```

## 📋 Configuration Files

All settings are controlled via YAML config files in `configs/`.

### Example: `vitpose_demo.yaml`

```yaml
# Detection (YOLO)
detection:
  type: yolo
  model_path: models/yolo/yolov8s.pt
  confidence_threshold: 0.5

# Pose Estimation (ViTPose)
pose_estimation:
  type: vitpose
  model_name: vitpose-b
  model_path: models/vitpose/vitpose-b-coco.pth
  dataset: coco

# Input
input:
  type: auto  # auto-detect from file extension
  path: demo_data/images/sample.jpg

# Output
output:
  path: demo_data/outputs/result.jpg
  draw_bbox: true
  draw_keypoints: true

# Processing
processing:
  max_frames: null  # null = all frames
  device: cuda
```

### Key Configuration Options

**Detection:**
- `model_path`: YOLO model (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
- `confidence_threshold`: Detection confidence (0.0-1.0)

**Pose Estimation:**
- `type`: Choose `vitpose` or `rtmlib`

**For ViTPose:**
- `model_name`: vitpose-s, vitpose-b, vitpose-l, vitpose-h
- `model_path`: Path to .pth file
- `dataset`: coco, coco_wholebody, mpii

**For RTMPose/RTMLib:**
- `model_type`: rtmpose-m, rtmpose-l, rtmpose-x
- `backend`: onnxruntime or openvino

**Processing:**
- `max_frames`: Limit frames for videos (null = all)
- `device`: cuda or cpu

## 📁 Directory Structure

```
newrepo/
├── setup_unified.py          # Main setup script
├── verify.py                 # Environment verification
├── udp.py                    # Unified Demo Pipeline (main entry point)
│
├── configs/                  # Configuration files
│   ├── default.yaml         # Template config
│   ├── vitpose_demo.yaml    # ViTPose example
│   ├── rtmlib_demo.yaml     # RTMPose example
│   └── video_demo.yaml      # Video processing example
│
├── lib/                      # Library code
│   ├── vitpose/             # ViTPose implementation
│   └── rtmlib/              # RTMLib implementation
│
├── models/                   # Model files
│   ├── yolo/                # YOLO detection models
│   ├── vitpose/             # ViTPose models (.pth)
│   └── rtmlib/              # RTMLib models (auto-downloaded)
│
├── demo_data/               # Demo media
│   ├── videos/              # Test videos (dance.mp4, etc.)
│   ├── images/              # Test images
│   └── outputs/             # Results saved here
│
└── requirements.txt         # Python dependencies
```

## 🎯 Workflow

### Fresh Colab Session

```python
# 1. Setup
!python setup_unified.py

# 2. Verify
!python verify.py

# 3. Run demo
!python udp.py --config configs/rtmlib_demo.yaml
```

### Configuration-Based Workflow

1. **Create/edit config file** in `configs/`
2. **Set parameters**:
   - Model paths (detection + pose)
   - Input/output paths
   - Processing options (frame limit, device)
3. **Run**: `python udp.py --config your_config.yaml`

## 📊 Output

The pipeline produces:

**For Images:**
- Annotated image with bounding boxes and keypoints
- Optional JSON with keypoint coordinates

**For Videos:**
- Annotated video with pose overlay
- Processing statistics (FPS, timing)
- Optional JSON with per-frame keypoints

**Statistics Shown:**
```
📊 Processing Statistics
══════════════════════════════════════════════════════════════════════

   Frames Processed: 100
   Total Time: 5.23 s
   Average FPS: 19.12

   Detection Time: 2.11 s (21.1 ms/frame)
   Pose Time: 3.12 s (31.2 ms/frame)
```

## 🔧 Customization

### Create New Config

1. Copy `configs/default.yaml`
2. Modify for your use case:
   - Change model sizes (faster vs more accurate)
   - Adjust input/output paths
   - Set frame limits for testing
   - Enable JSON export for data analysis

### Example Use Cases

**Fast Processing (Real-time):**
```yaml
detection:
  model_path: models/yolo/yolov8n.pt  # Nano - fastest
pose_estimation:
  type: rtmlib
  model_type: rtmpose-m  # Medium - balanced
```

**High Accuracy:**
```yaml
detection:
  model_path: models/yolo/yolov8x.pt  # Extra large
pose_estimation:
  type: vitpose
  model_name: vitpose-h  # Huge - most accurate
```

**Quick Testing (100 frames):**
```yaml
processing:
  max_frames: 100
```

## 🛠️ Troubleshooting

### Setup Issues

**Problem**: Missing packages
```bash
# Re-run setup
python setup_unified.py
```

**Problem**: Model not found
```bash
# Check models directory
ls models/yolo/
ls models/vitpose/

# Download models manually if needed
```

**Problem**: Import errors
```bash
# Run verification to see what's missing
python verify.py
```

### Runtime Issues

**Problem**: CUDA out of memory
- Use smaller model (yolov8n, rtmpose-m)
- Reduce batch size
- Process fewer frames at a time

**Problem**: Slow processing
- Check device setting (should be `cuda` not `cpu`)
- Use faster models (rtmpose-m instead of vitpose-h)
- Verify GPU is being used: `python verify.py`

## 📚 Documentation

- **ViTPose**: [GitHub](https://github.com/ViTAE-Transformer/ViTPose)
- **RTMLib**: [GitHub](https://github.com/Tau-J/rtmlib)
- **YOLO**: [Ultralytics Docs](https://docs.ultralytics.com/)

## 🎓 Model Performance

### ViTPose on MS COCO
- **ViTPose-S**: 73.8 AP (Small, ~25M params)
- **ViTPose-B**: 75.8 AP (Base, ~86M params)
- **ViTPose-L**: 78.3 AP (Large, ~307M params)
- **ViTPose-H**: 79.1 AP (Huge, ~632M params)

### RTMPose
- **RTMPose-m**: ~70 AP, ~40 FPS (GPU)
- **RTMPose-l**: ~73 AP, ~30 FPS (GPU)
- **RTMPose-x**: ~75 AP, ~20 FPS (GPU)

### YOLO Detection
- **YOLOv8n**: 37.3 mAP, ~200 FPS (Nano)
- **YOLOv8s**: 44.9 mAP, ~150 FPS (Small)
- **YOLOv8m**: 50.2 mAP, ~100 FPS (Medium)

## 💡 Tips

1. **Start with small models** for testing, then scale up
2. **Use max_frames** parameter to test on short segments
3. **Check verify.py output** before running demos
4. **Monitor GPU memory** with large videos
5. **Save JSON output** for downstream analysis

## 🔄 Updates

To update the pipeline:
```bash
cd oldrepos/
git pull  # Update source repos

# Copy updated code to lib/ if needed
```

## 📝 License

This unified pipeline combines multiple open-source projects. Check individual component licenses:
- ViTPose: Apache 2.0
- RTMLib: Apache 2.0
- YOLO: AGPL-3.0

## 🤝 Contributing

This is a unified wrapper. Contribute to upstream projects:
- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
- [RTMLib](https://github.com/Tau-J/rtmlib)
- [Ultralytics](https://github.com/ultralytics/ultralytics)

---

**Need Help?** Check `verify.py` output for diagnostics.
