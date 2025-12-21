# Unified Pose Pipeline

Production-ready 2D pose estimation pipeline with configurable methods (RTMPose/ViTPose).

## 🎯 Features

- **3-Stage Video Pipeline**: Detection → Pose → Visualization (optional)
- **Unified Image Pipeline**: Single-shot image processing
- **Configurable Methods**: Switch between RTMPose (fast) and ViTPose (accurate)
- **NPZ Storage**: Efficient intermediate data storage
- **Colorful Visualization**: Rainbow skeleton overlay
- **Performance**: 77 FPS detection, 48 FPS RTMPose, 39 FPS ViTPose

## 🚀 Quick Start

### 1. Setup (Google Colab)

```bash
# Clone repository
git clone https://github.com/pradeepj247/unifiedposepipeline.git
cd unifiedposepipeline

# Run automated setup
python setup_unified.py
```

**Downloads:**
- YOLOv8s model → `/content/models/yolo/`
- ViTPose-B model → `/content/models/vitpose/`
- RTMPose models → cached by rtmlib during first run

### 2. Process Images

```bash
# RTMPose (fast)
python udp_image.py --config configs/udp_image.yaml

# ViTPose (accurate) - edit config:
# Set method: vitpose in configs/udp_image.yaml
python udp_image.py --config configs/udp_image.yaml
```

### 3. Process Videos

```bash
# 3-stage pipeline
python udp_video.py --config configs/udp_video.yaml

# Outputs:
# - demo_data/outputs/detections.npz (Stage 1: Bounding boxes)
# - demo_data/outputs/keypoints.npz (Stage 2: Pose keypoints)  
# - demo_data/outputs/result.mp4 (Stage 3: Annotated video)
```

## ⚙️ Configuration

**Image Pipeline** (`configs/udp_image.yaml`):
```yaml
detection:
  model_path: yolov8s.pt
  confidence_threshold: 0.5

pose_estimation:
  method: rtmpose  # Options: rtmpose, vitpose
```

**Video Pipeline** (`configs/udp_video.yaml`):
```yaml
video:
  input_path: demo_data/videos/dance.mp4
  max_frames: 360  # Limit for quick testing

output:
  plot: true  # Set false to skip visualization (NPZ only)
```

## 📊 Performance Comparison (360 frames, 720p video)

| Method | Detection | Pose | Visualization | Total | Speed vs Accuracy |
|--------|-----------|------|---------------|-------|-------------------|
| **RTMPose** | 4.64s @ 77.7 FPS | 7.46s @ 48.3 FPS | 2.68s | **14.77s** | ⚡ **25% faster** |
| **ViTPose** | 4.65s @ 77.4 FPS | 9.31s @ 38.7 FPS | 2.70s | **16.66s** | 🎯 **Higher confidence** |

**Agreement**: 4.6 pixel average difference (excellent)

## 🔧 Tools

**Compare Keypoints**:
```bash
python compare_keypoints.py \
    demo_data/outputs/keypoints_rtm.npz \
    demo_data/outputs/keypoints_vit.npz
```

Shows:
- Per-joint confidence comparison
- Spatial differences (pixels)
- Frame-by-frame analysis
- Best/worst agreement frames

## 📁 Architecture

```
Stage 1: YOLO Detection (YOLOv8s)
    ↓ (largest bbox per frame → NPZ)
Stage 2: Pose Estimation (RTMPose/ViTPose)  
    ↓ (17 keypoints × [x,y,confidence] → NPZ)
Stage 3: Visualization (optional)
    ↓ (colorful rainbow skeleton)
Output: Annotated Video + NPZ Data
```

## 📂 Repository Structure

```
unifiedposepipeline/
├── udp_image.py          # Image processing pipeline
├── udp_video.py          # 3-stage video pipeline
├── compare_keypoints.py  # RTMPose vs ViTPose comparison
├── setup_unified.py      # Automated setup script
├── configs/
│   ├── udp_image.yaml    # Image pipeline config
│   └── udp_video.yaml    # Video pipeline config
├── lib/
│   ├── rtmlib/           # RTMPose library (ONNX)
│   └── vitpose/          # ViTPose library (PyTorch)
└── demo_data/
    ├── videos/           # Input videos
    └── outputs/          # Results (NPZ + MP4)
```

## 🎓 Model Details

**RTMPose-L**:
- ONNX-optimized for speed
- 384×288 input resolution
- Auto-cached during first run
- Best for: Real-time, production

**ViTPose-B**:
- Transformer-based (PyTorch)
- Higher accuracy on challenging poses
- Better on: Ears, shoulders, core joints
- Best for: Quality-critical applications

**YOLOv8s**:
- Person detection (class 0)
- 77 FPS on CUDA
- Single person tracking (largest bbox)

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- See `requirements.txt` for dependencies

## 🙏 Credits

- **RTMPose**: [rtmlib](https://github.com/Tau-J/rtmlib) - Real-time pose estimation
- **ViTPose**: [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) - Vision Transformer poses
- **Detection**: [YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection

## 📄 License

MIT License - See individual libraries for their licenses.

## 📝 Notes

**RTMPose Model Caching**:
- Models auto-download to `~/.cache/rtmlib/` on first run
- No manual download needed
- RTMPose-L: ~99MB download

**Choosing a Method**:
- **RTMPose**: Production, real-time, embedded systems
- **ViTPose**: Research, quality-critical, challenging poses
- **Both excellent**: 4.6 pixel average agreement

## 🐛 Troubleshooting

**"Could not open video"**: Check video path in config
**"CUDA out of memory"**: Reduce `max_frames` or use CPU
**"Model not found"**: Run `setup_unified.py` again


```python
from lib.unified_pose import UnifiedPoseEstimator

# Use either 'vitpose' or 'rtmlib'
estimator = UnifiedPoseEstimator(backend='vitpose')

# Process with consistent API
results = estimator.estimate('path/to/image.jpg')
```

## 📁 Project Structure

```
newrepo/
├── lib/                      # Core library code
│   ├── vitpose/             # ViTPose+HybrIK implementation
│   ├── rtmlib/              # RTMLib implementation
│   ├── vitpose_wrapper.py   # ViTPose API wrapper
│   ├── rtmlib_wrapper.py    # RTMLib API wrapper
│   └── unified_pose.py      # Unified interface
├── demos/                   # Example scripts
│   ├── demo_vitpose.py
│   ├── demo_rtmlib.py
│   └── demo_comparison.py
├── notebooks/               # Jupyter notebooks
│   ├── 01_setup.ipynb
│   ├── 02_vitpose_demo.ipynb
│   ├── 03_rtmlib_demo.ipynb
│   └── 04_comparison.ipynb
├── models/                  # Pre-trained models (downloaded)
├── configs/                 # Configuration files
├── requirements.txt         # Python dependencies
├── setup.py                # Package setup
└── README.md               # This file
```

## 🔧 Configuration

Configuration files are stored in `configs/`:

- `vitpose_config.yaml`: ViTPose settings
- `rtmlib_config.yaml`: RTMLib settings
- `unified_config.yaml`: Unified pipeline settings

## 📊 Model Zoo

### ViTPose Models
- ViTPose-Small (256x192)
- ViTPose-Base (256x192)
- ViTPose-Large (256x192)
- ViTPose-Huge (256x192)

### RTMLib Models
- RTMPose-t (tiny)
- RTMPose-s (small)
- RTMPose-m (medium)
- RTMPose-l (large)

## 🎓 Citation

If you use this unified pipeline in your research, please cite the original works:

### ViTPose
```bibtex
@inproceedings{xu2022vitpose,
  title={ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation},
  author={Xu, Yufei and Zhang, Jing and Zhang, Qiming and Tao, Dacheng},
  booktitle={NeurIPS},
  year={2022}
}
```

### RTMLib
```bibtex
@misc{rtmlib2023,
  title={RTMLib: Real-time Multi-person Pose Estimation Library},
  author={Tau-J and contributors},
  year={2023},
  howpublished={\url{https://github.com/Tau-J/rtmlib}}
}
```

## 📝 License

This unified repository maintains the original licenses:
- ViTPose components: See original license
- RTMLib components: See original license

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions and issues, please open an issue on GitHub.

---

**Note**: This is a unified implementation combining easy-pose-pipeline and rtmlib. All essential components have been consolidated for ease of use and maintenance.
