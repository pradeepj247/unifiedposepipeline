"""
Unified Pose Estimation Pipeline - Complete Setup Script
Installs all dependencies, clones required repos, downloads models, and verifies environment.
Works for both local and Google Colab environments.
"""

import os
import sys
import subprocess
from pathlib import Path
import urllib.request
from typing import Dict, Tuple, Optional

# ============================================
# Configuration Settings
# ============================================
REPO_ROOT = Path(__file__).parent
LIB_DIR = REPO_ROOT / "lib"
MODELS_DIR = REPO_ROOT / "models"
DEMOS_DIR = REPO_ROOT / "demos"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

# Model URLs and configurations
YOLO_MODELS = {
    "yolov8n": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "filename": "yolov8n.pt",
        "size_mb": 6.2
    },
    "yolov8m": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
        "filename": "yolov8m.pt",
        "size_mb": 49.7
    },
    "yolov8x": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt",
        "filename": "yolov8x.pt",
        "size_mb": 130.5
    },
}

VITPOSE_MODELS = {
    "vitpose-b": {
        "info": "ViTPose-Base model",
        "manual": True,
        "download_url": "https://github.com/ViTAE-Transformer/ViTPose#results-from-this-repo-on-ms-coco"
    },
    "vitpose-l": {
        "info": "ViTPose-Large model",
        "manual": True,
        "download_url": "https://github.com/ViTAE-Transformer/ViTPose#results-from-this-repo-on-ms-coco"
    },
}

# RTMLib models are downloaded automatically by the library


# ============================================
# Helper Functions
# ============================================
def print_header(message: str, char: str = "=", width: int = 70):
    """Print a formatted header"""
    print("\n" + char * width)
    print(message)
    print(char * width)


def print_step(step_num: int, total_steps: int, message: str):
    """Print a step header"""
    print_header(f"Step {step_num}/{total_steps}: {message}")


def run_command(cmd: str, error_msg: str = None, capture_output: bool = False) -> Tuple[int, str]:
    """
    Run a shell command with error handling
    
    Args:
        cmd: Command to run
        error_msg: Custom error message if command fails
        capture_output: Whether to capture and return output
    
    Returns:
        Tuple of (return_code, output)
    """
    print(f"$ {cmd}")
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.returncode, result.stdout
        else:
            return_code = subprocess.call(cmd, shell=True)
            return return_code, ""
    except Exception as e:
        if error_msg:
            print(f"❌ {error_msg}: {e}")
        else:
            print(f"❌ Command failed: {e}")
        return 1, str(e)


def check_python_version():
    """Verify Python version is compatible"""
    print("🔍 Checking Python version...")
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        print(f"❌ Python 3.8+ required, found {major}.{minor}")
        sys.exit(1)
    print(f"✅ Python {major}.{minor} detected")


def check_gpu_availability():
    """Check if CUDA/GPU is available"""
    print("\n🔍 Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available: {gpu_name}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  No CUDA GPU detected - will use CPU")
            return False
    except ImportError:
        print("⚠️  PyTorch not yet installed - GPU check deferred")
        return False


def create_directories():
    """Create necessary directory structure"""
    print("\n📁 Creating directory structure...")
    dirs = [MODELS_DIR, DEMOS_DIR, NOTEBOOKS_DIR]
    subdirs = [
        MODELS_DIR / "yolo",
        MODELS_DIR / "vitpose",
        MODELS_DIR / "rtmlib",
    ]
    
    for directory in dirs + subdirs:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {directory.relative_to(REPO_ROOT)}/")
    
    print("✅ Directory structure created")


def download_file(url: str, output_path: Path, desc: str = None) -> bool:
    """
    Download a file with progress indication
    
    Args:
        url: URL to download from
        output_path: Where to save the file
        desc: Description for progress bar
    
    Returns:
        True if successful, False otherwise
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 ** 2)
        print(f"   ✓ {output_path.name} already exists ({size_mb:.1f} MB)")
        return True
    
    desc = desc or output_path.name
    print(f"   ⬇️  Downloading {desc}...")
    
    try:
        # Use wget if available, otherwise urllib
        wget_available = subprocess.call("which wget", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
        
        if wget_available:
            cmd = f'wget -q --show-progress -O "{output_path}" "{url}"'
            return_code, _ = run_command(cmd)
            if return_code == 0:
                size_mb = output_path.stat().st_size / (1024 ** 2)
                print(f"   ✅ Downloaded {desc} ({size_mb:.1f} MB)")
                return True
        else:
            urllib.request.urlretrieve(url, output_path)
            size_mb = output_path.stat().st_size / (1024 ** 2)
            print(f"   ✅ Downloaded {desc} ({size_mb:.1f} MB)")
            return True
    except Exception as e:
        print(f"   ❌ Download failed for {desc}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


# ============================================
# Installation Steps
# ============================================
def step_0_environment_check():
    """Step 0: Environment validation"""
    print_step(0, 7, "Environment Validation")
    
    check_python_version()
    
    # Check if running in Colab
    try:
        import google.colab
        print("✅ Google Colab environment detected")
        is_colab = True
    except ImportError:
        print("✅ Local environment detected")
        is_colab = False
    
    return is_colab


def step_1_install_core_dependencies(is_colab: bool):
    """Step 1: Install core Python dependencies"""
    print_step(1, 7, "Installing Core Dependencies")
    
    # Core dependencies
    core_deps = [
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pillow>=10.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "packaging",
    ]
    
    print("📦 Installing core dependencies...")
    cmd = f"pip install {' '.join(core_deps)}"
    return_code, _ = run_command(cmd, "Failed to install core dependencies")
    if return_code != 0:
        print("❌ Core dependencies installation failed")
        sys.exit(1)
    
    print("✅ Core dependencies installed")


def step_2_install_pytorch(is_colab: bool):
    """Step 2: Install PyTorch"""
    print_step(2, 7, "Installing PyTorch")
    
    # Check if PyTorch is already installed
    try:
        import torch
        print(f"✓ PyTorch already installed: {torch.__version__}")
        return
    except ImportError:
        pass
    
    print("📦 Installing PyTorch (this may take a few minutes)...")
    
    if is_colab:
        # Colab usually has PyTorch pre-installed
        cmd = "pip install torch torchvision torchaudio"
    else:
        # Local installation with CUDA support
        print("   ℹ️  Installing PyTorch with CUDA 11.8 support")
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    
    return_code, _ = run_command(cmd, "Failed to install PyTorch")
    if return_code != 0:
        print("❌ PyTorch installation failed")
        sys.exit(1)
    
    print("✅ PyTorch installed")
    check_gpu_availability()


def step_3_install_cv_dependencies():
    """Step 3: Install computer vision dependencies"""
    print_step(3, 7, "Installing Computer Vision Dependencies")
    
    cv_deps = [
        "opencv-python>=4.8.0",
        "opencv-contrib-python",  # Required by RTMLib
    ]
    
    print("📦 Installing OpenCV packages...")
    cmd = f"pip install {' '.join(cv_deps)}"
    return_code, _ = run_command(cmd, "Failed to install CV dependencies")
    if return_code != 0:
        print("❌ CV dependencies installation failed")
        sys.exit(1)
    
    print("✅ CV dependencies installed")


def step_4_install_pose_frameworks():
    """Step 4: Install YOLO and ONNX Runtime"""
    print_step(4, 7, "Installing Pose Estimation Frameworks")
    
    # YOLO (Ultralytics)
    print("📦 Installing Ultralytics (YOLO)...")
    return_code, _ = run_command("pip install ultralytics>=8.0.0", "Failed to install Ultralytics")
    if return_code != 0:
        print("❌ Ultralytics installation failed")
        sys.exit(1)
    print("✅ Ultralytics installed")
    
    # ONNX Runtime
    print("\n📦 Installing ONNX Runtime...")
    # Try GPU version first, fall back to CPU
    return_code, _ = run_command("pip install onnxruntime-gpu", capture_output=True)
    if return_code != 0:
        print("   ⚠️  GPU version failed, installing CPU version...")
        return_code, _ = run_command("pip install onnxruntime", "Failed to install ONNX Runtime")
        if return_code != 0:
            print("❌ ONNX Runtime installation failed")
            sys.exit(1)
    print("✅ ONNX Runtime installed")
    
    # Additional ML utilities
    print("\n📦 Installing additional utilities...")
    utils = ["pandas>=2.0.0", "seaborn>=0.12.0"]
    cmd = f"pip install {' '.join(utils)}"
    run_command(cmd)  # Non-critical, don't exit on failure
    print("✅ Additional utilities installed")


def step_5_verify_libraries():
    """Step 5: Verify library imports"""
    print_step(5, 7, "Verifying Library Imports")
    
    imports_to_check = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("ultralytics", "Ultralytics"),
        ("onnxruntime", "ONNX Runtime"),
        ("matplotlib", "Matplotlib"),
        ("pandas", "Pandas"),
        ("yaml", "PyYAML"),
    ]
    
    failed_imports = []
    
    for module_name, display_name in imports_to_check:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"   ✅ {display_name:20} v{version}")
        except ImportError as e:
            print(f"   ❌ {display_name:20} FAILED")
            failed_imports.append(display_name)
    
    # Check lib directories
    print("\n🔍 Checking library structure...")
    vitpose_init = LIB_DIR / "vitpose" / "__init__.py"
    rtmlib_init = LIB_DIR / "rtmlib" / "__init__.py"
    
    if vitpose_init.exists():
        print(f"   ✅ ViTPose library found")
    else:
        print(f"   ⚠️  ViTPose library not found at {vitpose_init}")
    
    if rtmlib_init.exists():
        print(f"   ✅ RTMLib library found")
    else:
        print(f"   ⚠️  RTMLib library not found at {rtmlib_init}")
    
    if failed_imports:
        print(f"\n⚠️  Warning: {len(failed_imports)} imports failed: {', '.join(failed_imports)}")
    else:
        print("\n✅ All critical imports successful")


def step_6_download_models():
    """Step 6: Download essential models"""
    print_step(6, 7, "Downloading Essential Models")
    
    create_directories()
    
    # Download YOLO models
    print("\n📦 Downloading YOLO models...")
    yolo_dir = MODELS_DIR / "yolo"
    
    # Download at least the smallest model
    essential_yolo = ["yolov8n"]
    for model_name in essential_yolo:
        model_info = YOLO_MODELS[model_name]
        output_path = yolo_dir / model_info["filename"]
        success = download_file(
            model_info["url"],
            output_path,
            f"{model_name} ({model_info['size_mb']}MB)"
        )
        if not success:
            print(f"   ⚠️  Failed to download {model_name}, continuing...")
    
    # ViTPose models info
    print("\n📦 ViTPose Models:")
    print("   ℹ️  ViTPose models are large and require manual download")
    print("   📍 Visit: https://github.com/ViTAE-Transformer/ViTPose")
    print(f"   📁 Place models in: {MODELS_DIR / 'vitpose'}")
    
    # RTMLib models info
    print("\n📦 RTMLib Models:")
    print("   ✅ RTMLib models download automatically on first use")
    print("   📁 Models will be cached in: ~/.cache/")
    
    print("\n✅ Essential models setup complete")


def step_7_setup_demo_data():
    """Step 7: Setup demo data files"""
    print_step(7, 8, "Setting Up Demo Data")
    
    print("📂 Creating demo data structure...")
    demo_data_dir = REPO_ROOT / "demo_data"
    dirs = [
        demo_data_dir / "videos",
        demo_data_dir / "images", 
        demo_data_dir / "outputs",
    ]
    
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("✅ Demo data directories created")
    print(f"\n💡 To download demo files, run:")
    print("   python setup_demo_data.py")


def step_8_final_verification():
    """Step 8: Final environment verification"""
    print_step(8, 8, "Final Verification")
    
    # Test imports with version info
    print("🔍 Testing complete environment...\n")
    
    try:
        import torch
        import cv2
        import numpy as np
        from ultralytics import YOLO
        
        print("✅ Environment Test Results:")
        print(f"   • PyTorch: {torch.__version__}")
        print(f"   • OpenCV: {cv2.__version__}")
        print(f"   • NumPy: {np.__version__}")
        print(f"   • CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   • CUDA Version: {torch.version.cuda}")
            print(f"   • GPU Device: {torch.cuda.get_device_name(0)}")
        
        print("\n✅ All components verified successfully!")
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================
# Main Execution
# ============================================
def main():
    """Main setup execution"""
    print("\n" + "🚀" * 35)
    print("UNIFIED POSE ESTIMATION PIPELINE - SETUP")
    print("🚀" * 35)
    print(f"\nInstallation directory: {REPO_ROOT.absolute()}\n")
    
    try:
        # Execute all setup steps
        is_colab = step_0_environment_check()
        step_1_install_core_dependencies(is_colab)
        step_2_install_pytorch(is_colab)
        step_3_install_cv_dependencies()
        step_4_install_pose_frameworks()
        step_5_verify_libraries()
        step_6_download_models()
        step_7_setup_demo_data()
        step_8_final_verification()
        
        # Success message
        print("\n" + "=" * 70)
        print("🎉 SETUP COMPLETE!")
        print("=" * 70)
        print("\n📚 Next Steps:")
        print("   1. Run: python setup_demo_data.py (to download demo files)")
        print("   2. Check out the notebooks/ directory for tutorials")
        print("   3. Run demo scripts in demos/ directory")
        print("   4. Read the README.md for detailed usage")
        print("\n💡 Quick Start:")
        print("   python setup_demo_data.py")
        print("   python demos/demo_vitpose.py")
        print("   python demos/demo_rtmlib.py")
        print("\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
