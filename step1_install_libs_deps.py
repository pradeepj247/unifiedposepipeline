#!/usr/bin/env python3
"""
Step 1: Install Libraries and Dependencies

This script installs all required Python packages and creates necessary directories.
Corresponds to Steps 0-7 in the original setup_unified.py.

Usage:
    python step1_install_libs_deps.py
"""

import os
import sys
from setup_utils import (
    is_colab_environment, print_header, print_step, run_command,
    create_directory, print_success, print_error
)


# Configuration
REPO_ROOT = "/content/unifiedposepipeline" if is_colab_environment() else os.getcwd()
MODELS_DIR = "/content/models"  # Parent directory to persist across repo deletions


def step0_mount_drive():
    """Mount Google Drive (Colab only)"""
    print_step(0, "Mount Google Drive")
    
    if not is_colab_environment():
        print("⊘ Skipping (not in Colab environment)")
        return
    
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✓ Google Drive mounted successfully")
    except Exception as e:
        print(f"✗ Failed to mount Drive: {e}")
        print("Continuing without Drive (some features may be unavailable)")


def step1_install_core_dependencies():
    """Install core Python packages"""
    print_step(1, "Install Core Dependencies")
    
    packages = [
        "numpy",
        "scipy",
        "pillow",
        "pyyaml",
        "tqdm",
        "matplotlib",
        "pandas"
    ]
    
    cmd = f"pip install -q {' '.join(packages)}"
    try:
        run_command(cmd)
        print("✓ Core dependencies installed")
    except Exception as e:
        print_error(f"Failed to install core dependencies: {e}")
        sys.exit(1)


def step2_install_pytorch():
    """Install PyTorch with CUDA support"""
    print_step(2, "Install PyTorch with CUDA")
    
    cmd = "pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    
    try:
        run_command(cmd)
        print("✓ PyTorch installed")
        
        # Verify installation
        import torch
        print(f"  - PyTorch version: {torch.__version__}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - GPU device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print_error(f"Failed to install PyTorch: {e}")
        sys.exit(1)


def step3_install_opencv_yolo():
    """Install OpenCV and YOLO (Ultralytics)"""
    print_step(3, "Install OpenCV and YOLO")
    
    packages = [
        "opencv-python",
        "opencv-contrib-python",
        "ultralytics"
    ]
    
    cmd = f"pip install -q {' '.join(packages)}"
    
    try:
        run_command(cmd)
        print("✓ OpenCV and YOLO installed")
    except Exception as e:
        print_error(f"Failed to install OpenCV/YOLO: {e}")
        sys.exit(1)


def step4_install_pose_estimation():
    """Install pose estimation libraries (ONNX Runtime)"""
    print_step(4, "Install Pose Estimation Libraries")
    
    # Check if GPU is available
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except:
        has_cuda = False
    
    if has_cuda:
        print("GPU detected, installing ONNX Runtime GPU")
        packages = ["onnx", "onnxruntime-gpu"]
    else:
        print("No GPU detected, installing ONNX Runtime CPU")
        packages = ["onnx", "onnxruntime"]
    
    cmd = f"pip install -q {' '.join(packages)}"
    
    try:
        run_command(cmd)
        print("✓ Pose estimation libraries installed")
    except Exception as e:
        print_error(f"Failed to install pose estimation libraries: {e}")
        sys.exit(1)


def step5_install_tracking():
    """Install tracking and ReID libraries"""
    print_step(5, "Install Tracking Libraries")
    
    packages = [
        "boxmot",
        "gdown",
        "supervision",
        "filterpy",
        "scikit-learn"
    ]
    
    cmd = f"pip install -q {' '.join(packages)}"
    
    try:
        run_command(cmd)
        print("✓ Tracking libraries installed")
    except Exception as e:
        print_error(f"Failed to install tracking libraries: {e}")
        sys.exit(1)


def step6_install_motionagformer_deps():
    """Install MotionAGFormer dependencies"""
    print_step(6, "Install MotionAGFormer Dependencies")
    
    packages = [
        "timm",
        "easydict",
        "yacs",
        "numba",
        "scikit-image"
    ]
    
    cmd = f"pip install -q {' '.join(packages)}"
    
    try:
        run_command(cmd)
        print("✓ MotionAGFormer dependencies installed")
    except Exception as e:
        print(f"⚠ Some MotionAGFormer dependencies failed: {e}")
        print("  (This may not be critical if you don't use 3D lifting)")


def step7_create_directories():
    """Create necessary directory structure"""
    print_step(7, "Create Directory Structure")
    
    directories = [
        MODELS_DIR,
        os.path.join(MODELS_DIR, "yolo"),
        os.path.join(MODELS_DIR, "rtmlib"),
        os.path.join(MODELS_DIR, "vitpose"),
        os.path.join(MODELS_DIR, "wb3d"),
        os.path.join(MODELS_DIR, "motionagformer"),
        os.path.join(MODELS_DIR, "reid"),
        os.path.join(REPO_ROOT, "demo_data"),
        os.path.join(REPO_ROOT, "demo_data", "videos"),
        os.path.join(REPO_ROOT, "demo_data", "images"),
        os.path.join(REPO_ROOT, "demo_data", "outputs"),
        os.path.join(REPO_ROOT, "configs")
    ]
    
    for directory in directories:
        create_directory(directory)
    
    print("✓ Directory structure created")


def main():
    """Main execution function"""
    print_header("STEP 1: Install Libraries and Dependencies")
    
    print("This script will install all required Python packages.")
    print(f"Repository root: {REPO_ROOT}")
    print(f"Models directory: {MODELS_DIR}")
    
    try:
        step0_mount_drive()
        step1_install_core_dependencies()
        step2_install_pytorch()
        step3_install_opencv_yolo()
        step4_install_pose_estimation()
        step5_install_tracking()
        step6_install_motionagformer_deps()
        step7_create_directories()
        
        print_success("All libraries and dependencies installed successfully!")
        print("\nNext steps:")
        print("  python step2_install_models.py   # Download model files")
        print("  python step3_pull_demodata.py    # Setup demo data")
        print("  python step4_verify_envt.py      # Verify installation")
        
    except KeyboardInterrupt:
        print("\n\n⊘ Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
