# Unified Pose Estimation Pipeline

A comprehensive pose estimation framework combining **ViTPose+HybrIK** and **RTMLib** into a single, unified pipeline.

## 🎯 Overview

This repository unifies two powerful pose estimation approaches:

1. **ViTPose + HybrIK Pipeline**: Vision Transformer-based pose estimation with SMPL body model support
2. **RTMLib Pipeline**: Lightweight real-time pose estimation using RTMPose models

### Key Features

- ✅ **Unified Installation**: Single requirements.txt for both pipelines
- ✅ **Shared YOLO Detection**: Common object detection framework
- ✅ **Flexible Model Selection**: Switch between ViTPose and RTMLib easily
- ✅ **Multiple Output Formats**: 2D keypoints, 3D poses, SMPL parameters
- ✅ **Production Ready**: Optimized for both research and deployment

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (optional, for GPU acceleration)
- 10GB+ disk space (for models)

### 🚀 Quick Start (Automated Setup)

**Recommended**: Use the automated setup script that handles everything:

```bash
# Clone the repository
git clone <your-repo-url>
cd newrepo

# Run the complete setup (installs everything + downloads models)
python setup_environment.py

# Verify the installation
python verify_environment.py
```

The `setup_environment.py` script will:
- ✅ Check Python version and environment
- ✅ Install all dependencies (PyTorch, OpenCV, YOLO, etc.)
- ✅ Verify library structure
- ✅ Download essential YOLO models
- ✅ Provide detailed progress and error messages
- ✅ Run verification tests

### 📝 Manual Installation (Alternative)

If you prefer manual control:

```bash
# 1. Install core dependencies
pip install -r requirements.txt

# 2. Install PyTorch with CUDA (for GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Download models
python download_models.py

# 4. Verify installation
python verify_environment.py
```

### GPU Setup Notes

- The automated setup detects GPU and installs appropriate drivers
- For CPU-only: Setup script automatically configures for CPU
- For custom CUDA versions: Edit `setup_environment.py` Step 2

## 🚀 Quick Usage

### Using ViTPose Pipeline

```python
from lib.vitpose_wrapper import ViTPosePipeline

# Initialize pipeline
pipeline = ViTPosePipeline()

# Process image
results = pipeline.process_image('path/to/image.jpg')
```

### Using RTMLib Pipeline

```python
from lib.rtmlib_wrapper import RTMLibPipeline

# Initialize pipeline
pipeline = RTMLibPipeline()

# Process image
results = pipeline.process_image('path/to/image.jpg')
```

### Unified Interface

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
