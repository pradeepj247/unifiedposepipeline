"""
Unified Pose Estimation Pipeline - Complete Setup
For Google Colab and Local Environments

Run this first in a fresh Colab session to:
1. Mount Google Drive
2. Install all dependencies
3. Download models
4. Setup demo data
5. Verify environment
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Tuple, Optional

# ============================================
# Configuration
# ============================================
REPO_ROOT = Path(__file__).parent
PARENT_DIR = REPO_ROOT.parent  # Store models in parent directory (persistent across repo deletions)
LIB_DIR = REPO_ROOT / "lib"
MODELS_DIR = PARENT_DIR / "models"  # Models persist when repo is deleted
DEMO_DATA_DIR = REPO_ROOT / "demo_data"  # Demo data stays with repo

# Google Drive paths (for Colab)
DRIVE_ROOT = Path("/content/drive/MyDrive")
DRIVE_MODELS_PATH = DRIVE_ROOT / "HybrIK_TRT_Backups/models"
DRIVE_DEMO_DATA_PATH = DRIVE_ROOT / "HybrIK_TRT_Backups/demodata"


# ============================================
# Helper Functions
# ============================================
def print_header(message: str, char: str = "=", width: int = 70):
    """Print formatted header"""
    print("\n" + char * width)
    print(message)
    print(char * width)


def print_step(step_num: int, total_steps: int, message: str):
    """Print step header"""
    print_header(f"Step {step_num}/{total_steps}: {message}")


def run_command(cmd: str, error_msg: str = None) -> int:
    """Run shell command with logging"""
    print(f"$ {cmd}")
    return_code = subprocess.call(cmd, shell=True)
    if return_code != 0 and error_msg:
        print(f"❌ {error_msg}")
    return return_code


def is_colab_environment() -> bool:
    """Check if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False


# ============================================
# Setup Steps
# ============================================
def step_0_mount_drive() -> bool:
    """Step 0: Mount Google Drive (Colab only)"""
    print_step(0, 9, "Mounting Google Drive")
    
    if not is_colab_environment():
        print("✓ Local environment - skipping Drive mount")
        return False
    
    drive_path = Path("/content/drive")
    if drive_path.exists() and (drive_path / "MyDrive").exists():
        print("✓ Google Drive already mounted")
        return True
    
    try:
        from google.colab import drive
        print("📂 Mounting Google Drive...")
        drive.mount("/content/drive", force_remount=False)
        print("✅ Google Drive mounted successfully")
        return True
    except Exception as e:
        print(f"⚠️  Could not mount Drive: {e}")
        return False


def step_1_install_core_dependencies():
    """Step 1: Install core Python dependencies"""
    print_step(1, 9, "Installing Core Dependencies")
    
    core_deps = [
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pillow>=10.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "packaging",
    ]
    
    print("📦 Installing core Python packages...")
    # Use quotes around each package to prevent shell redirection issues
    packages_quoted = ' '.join([f'"{pkg}"' for pkg in core_deps])
    if run_command(f"pip install {packages_quoted}", 
                   "Failed to install core dependencies") != 0:
        sys.exit(1)
    print("✅ Core dependencies installed")


def step_2_install_pytorch():
    """Step 2: Install PyTorch"""
    print_step(2, 9, "Installing PyTorch")
    
    try:
        import torch
        print(f"✓ PyTorch already installed: {torch.__version__}")
        return
    except ImportError:
        pass
    
    print("📦 Installing PyTorch with CUDA support...")
    is_colab = is_colab_environment()
    
    if is_colab:
        cmd = "pip install torch torchvision torchaudio"
    else:
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    
    if run_command(cmd, "Failed to install PyTorch") != 0:
        sys.exit(1)
    
    print("✅ PyTorch installed")


def step_3_install_cv_and_detection():
    """Step 3: Install OpenCV and YOLO"""
    print_step(3, 9, "Installing Computer Vision & Detection")
    
    packages = [
        "opencv-python>=4.8.0",
        "opencv-contrib-python",
        "ultralytics>=8.0.0",
    ]
    
    print("📦 Installing OpenCV and YOLO (Ultralytics)...")
    packages_quoted = ' '.join([f'"{pkg}"' for pkg in packages])
    if run_command(f"pip install {packages_quoted}", 
                   "Failed to install CV packages") != 0:
        sys.exit(1)
    print("✅ CV and detection packages installed")


def step_4_install_pose_estimation():
    """Step 4: Install pose estimation frameworks"""
    print_step(4, 9, "Installing Pose Estimation Frameworks")
    
    # Note: RTMLib is used from local copy in lib/rtmlib/ (sys.path.insert prioritizes it)
    # No pip install needed - local copy has all required code
    
    # ONNX and ONNX Runtime
    print("📦 Installing ONNX and ONNX Runtime GPU...")
    # Install ONNX first
    if run_command('pip install "onnx>=1.12.0"') != 0:
        print("⚠️  ONNX install warning (non-critical)")
    
    # Try GPU version of ONNX Runtime
    if run_command("pip install onnxruntime-gpu") != 0:
        print("⚠️  GPU version failed, installing CPU version...")
        if run_command("pip install onnxruntime", "Failed to install ONNX Runtime") != 0:
            sys.exit(1)
    print("✅ ONNX and ONNX Runtime installed")


def step_5_install_tracking():
    """Step 5: Install tracking frameworks (BoxMOT)"""
    print_step(5, 9, "Installing Tracking Frameworks")
    
    # Skip BoxMOT for now - has NumPy compatibility issues
    # Only install lightweight dependencies
    packages = [
        "supervision>=0.15.0",
        "filterpy>=1.4.5",
        "scikit-learn>=1.0.0",
    ]
    
    print("📦 Installing tracking dependencies (BoxMOT skipped - optional)...")
    packages_quoted = ' '.join([f'"{pkg}"' for pkg in packages])
    if run_command(f"pip install {packages_quoted}") != 0:
        print("⚠️  Some tracking packages failed (non-critical)")
    else:
        print("✅ Tracking dependencies installed")
    print("ℹ️  BoxMOT skipped due to compatibility issues (not needed for basic demos)")


def step_6_install_motionagformer_deps():
    """Step 6: Install MotionAGFormer dependencies"""
    print_step(6, 10, "Installing MotionAGFormer Dependencies")
    
    # Note: MotionAGFormer model code is in lib/motionagformer/ (local copy)
    
    packages = [
        '"timm>=0.9.0"',       # Vision transformers (avoid 0.6.11 - Python 3.10+ issues)
        'easydict',            # Easy dictionary access
        'yacs',                # Configuration system
        'gdown',               # Google Drive downloads
        'numba',               # JIT compilation for speed
        'scikit-image',        # Image processing
    ]
    
    print("📦 Installing MotionAGFormer packages...")
    packages_str = ' '.join(packages)
    if run_command(f"pip install {packages_str}") != 0:
        print("⚠️  Some packages failed (non-critical - will retry later)")
    else:
        print("✅ MotionAGFormer dependencies installed")
    
    # Optional packages (can fail without breaking)
    optional = ['wandb', 'torchprofile']
    print("\n📦 Installing optional packages (logging/profiling)...")
    if run_command(f"pip install {' '.join(optional)}") != 0:
        print("ℹ️  Optional packages skipped (not needed for inference)")
    else:
        print("✅ Optional packages installed")


def step_7_setup_directories():
    """Step 7: Create directory structure"""
    print_step(7, 10, "Setting Up Directory Structure")
    
    dirs = [
        MODELS_DIR / "yolo",
        MODELS_DIR / "vitpose",
        MODELS_DIR / "rtmlib",
        MODELS_DIR / "motionagformer",
        MODELS_DIR / "wb3d",
        DEMO_DATA_DIR / "videos",
        DEMO_DATA_DIR / "images",
        DEMO_DATA_DIR / "outputs",
        REPO_ROOT / "configs",
    ]
    
    print("📁 Creating directories...")
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
        # Print relative to PARENT_DIR for models, REPO_ROOT for others
        if directory.is_relative_to(PARENT_DIR):
            print(f"   ✓ {directory.relative_to(PARENT_DIR)}/")
        else:
            print(f"   ✓ {directory}/")
    
    print("✅ Directory structure created")


def step_8_download_models(drive_mounted: bool):
    """Step 8: Download/copy models"""
    print_step(8, 10, "Downloading Models")
    
    # YOLO models
    print("\n📦 Downloading YOLO model...")
    yolo_models = {
        "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
    }
    
    yolo_dir = MODELS_DIR / "yolo"
    for model_name, url in yolo_models.items():
        output_path = yolo_dir / model_name
        if output_path.exists():
            print(f"   ✓ {model_name} already exists")
        else:
            print(f"   ⬇️  Downloading {model_name}...")
            if run_command(f'wget -q --show-progress -O "{output_path}" "{url}"') == 0:
                print(f"   ✅ Downloaded {model_name}")
    
    # ViTPose models
    print("\n📦 Downloading ViTPose model...")
    vitpose_dir = MODELS_DIR / "vitpose"
    vitpose_model = vitpose_dir / "vitpose-b.pth"
    vitpose_config = vitpose_dir / "ViTPose_base_coco_256x192.py"
    
    # Download model from GitHub releases (343 MB)
    if vitpose_model.exists():
        print(f"   ✓ vitpose-b.pth already exists")
    else:
        print(f"   ⬇️  Downloading vitpose-b.pth (343 MB)...")
        vitpose_url = "https://github.com/pradeepj247/easy-pose-pipeline/releases/download/v1.0/vitpose-b.pth"
        if run_command(f'curl -L -o "{vitpose_model}" "{vitpose_url}"') == 0:
            size_mb = vitpose_model.stat().st_size / (1024 ** 2)
            print(f"   ✅ Downloaded vitpose-b.pth ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ Download failed")
    
    # Copy config from lib/vitpose
    source_config = REPO_ROOT / "lib" / "vitpose" / "configs" / "train_configs" / "ViTPose_base_coco_256x192.py"
    if source_config.exists() and not vitpose_config.exists():
        print(f"   📋 Copying config file...")
        run_command(f'cp "{source_config}" "{vitpose_config}"')
        print(f"   ✅ Copied ViTPose_base_coco_256x192.py")
    elif vitpose_config.exists():
        print(f"   ✓ ViTPose_base_coco_256x192.py already exists")
    else:
        print(f"   ⚠️  Config not found at {source_config}")
    
    # RTMPose models
    print("\n📦 Downloading RTMPose ONNX models...")
    rtmlib_dir = MODELS_DIR / "rtmlib"
    rtmpose_models = {
        "rtmpose-l-coco-384x288.onnx": "https://github.com/pradeepj247/unifiedposepipeline/releases/download/v1.0.0/rtmpose-l-coco-384x288.onnx",
        "rtmpose-l-halpe26-384x288.onnx": "https://github.com/pradeepj247/unifiedposepipeline/releases/download/v1.0.0/rtmpose-l-halpe26-384x288.onnx",
    }
    
    for model_name, url in rtmpose_models.items():
        output_path = rtmlib_dir / model_name
        if output_path.exists():
            print(f"   ✓ {model_name} already exists")
        else:
            print(f"   ⬇️  Downloading {model_name} (~110 MB)...")
            if run_command(f'curl -L -o "{output_path}" "{url}"') == 0:
                size_mb = output_path.stat().st_size / (1024 ** 2)
                print(f"   ✅ Downloaded {model_name} ({size_mb:.1f} MB)")
            else:
                print(f"   ❌ Download failed for {model_name}")
    
    # Also check Drive for backup
    if drive_mounted and DRIVE_MODELS_PATH.exists():
        print(f"   📂 Checking Drive for additional models: {DRIVE_MODELS_PATH}")
        vitpose_files = list(DRIVE_MODELS_PATH.glob("vitpose*.pth"))
        if vitpose_files:
            for model_file in vitpose_files:
                dest = vitpose_dir / model_file.name
                if not dest.exists():
                    print(f"   📋 Copying {model_file.name} from Drive...")
                    run_command(f'cp "{model_file}" "{dest}"')
                else:
                    print(f"   ✓ {model_file.name} already exists")
    
    # MotionAGFormer checkpoint
    print("\n📦 Downloading MotionAGFormer checkpoint...")
    magf_dir = MODELS_DIR / "motionagformer"
    magf_checkpoint = magf_dir / "motionagformer-base-h36m.pth.tr"
    
    if magf_checkpoint.exists():
        size_mb = magf_checkpoint.stat().st_size / (1024 ** 2)
        print(f"   ✓ motionagformer-base-h36m.pth.tr already exists ({size_mb:.1f} MB)")
    else:
        print(f"   ⬇️  Downloading MotionAGFormer-Base checkpoint (~200 MB)...")
        print(f"   📥 Using gdown for Google Drive download...")
        
        # Google Drive file ID for MotionAGFormer-Base H36M
        gdown_id = "1Iii5EwsFFm9_9lKBUPfN8bV5LmfkNUMP"
        
        if run_command(f'gdown {gdown_id} -O "{magf_checkpoint}"') == 0:
            if magf_checkpoint.exists():
                size_mb = magf_checkpoint.stat().st_size / (1024 ** 2)
                print(f"   ✅ Downloaded motionagformer-base-h36m.pth.tr ({size_mb:.1f} MB)")
            else:
                print(f"   ❌ Download completed but file not found")
        else:
            print(f"   ❌ Download failed - you may need to download manually:")
            print(f"      URL: https://drive.google.com/file/d/1Iii5EwsFFm9_9lKBUPfN8bV5LmfkNUMP/view")
            print(f"      Save to: {magf_checkpoint}")
    
    # Whole-Body 3D model (rtmw3d-l.onnx)
    print("\n📦 Copying Whole-Body 3D model from Drive...")
    wb3d_dir = MODELS_DIR / "wb3d"
    wb3d_model = wb3d_dir / "rtmw3d-l.onnx"
    
    if wb3d_model.exists():
        size_mb = wb3d_model.stat().st_size / (1024 ** 2)
        print(f"   ✓ rtmw3d-l.onnx already exists ({size_mb:.1f} MB)")
    elif drive_mounted:
        drive_wb3d_path = DRIVE_ROOT / "rtmw3d_onnx_exports" / "rtmw3d-l.onnx"
        if drive_wb3d_path.exists():
            print(f"   📋 Copying from Drive: {drive_wb3d_path}")
            if run_command(f'cp "{drive_wb3d_path}" "{wb3d_model}"') == 0:
                size_mb = wb3d_model.stat().st_size / (1024 ** 2)
                print(f"   ✅ Copied rtmw3d-l.onnx ({size_mb:.1f} MB)")
            else:
                print(f"   ❌ Failed to copy model from Drive")
        else:
            print(f"   ⚠️  Model not found on Drive: {drive_wb3d_path}")
            print(f"      Expected location: /content/drive/MyDrive/rtmw3d_onnx_exports/rtmw3d-l.onnx")
    else:
        print(f"   ⚠️  Drive not mounted - skipping wb3d model")
        print(f"      Manual copy: Place rtmw3d-l.onnx in {wb3d_model}")
    
    # RTMLib models (auto-download note)
    print("\n📦 RTMLib Models:")
    print("   ✅ Will be downloaded automatically on first use")
    
    print("\n✅ Models setup complete")


def step_9_setup_demo_data(drive_mounted: bool):
    """Step 9: Setup demo data"""
    print_step(9, 10, "Setting Up Demo Data")
    
    # Copy from Drive if available
    if drive_mounted and DRIVE_DEMO_DATA_PATH.exists():
        print(f"📂 Copying demo files from Drive...")
        
        # Copy dance.mp4
        dance_src = DRIVE_DEMO_DATA_PATH / "dance.mp4"
        dance_dst = DEMO_DATA_DIR / "videos" / "dance.mp4"
        
        if dance_src.exists():
            if not dance_dst.exists():
                print(f"   📋 Copying dance.mp4...")
                if run_command(f'cp "{dance_src}" "{dance_dst}"') == 0:
                    size_mb = dance_dst.stat().st_size / (1024 ** 2)
                    print(f"   ✅ Copied dance.mp4 ({size_mb:.1f} MB)")
            else:
                print(f"   ✓ dance.mp4 already exists")
        else:
            print(f"   ⚠️  dance.mp4 not found on Drive")
    else:
        print("   ℹ️  No Drive access - demo files can be added manually")
    
    # Download sample image
    sample_img_url = "https://raw.githubusercontent.com/open-mmlab/mmpose/main/tests/data/coco/000000000785.jpg"
    sample_img_path = DEMO_DATA_DIR / "images" / "sample.jpg"
    
    if not sample_img_path.exists():
        print(f"\n   ⬇️  Downloading sample image...")
        if run_command(f'wget -q -O "{sample_img_path}" "{sample_img_url}"') == 0:
            print(f"   ✅ Downloaded sample.jpg")
    
    print("✅ Demo data setup complete")


def step_10_verify_installation():
    """Step 10: Verify installation"""
    print_step(10, 10, "Verifying Installation")
    
    imports_to_check = [
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("ultralytics", "Ultralytics/YOLO"),
        ("onnxruntime", "ONNX Runtime"),
    ]
    
    print("🔍 Checking imports...")
    all_ok = True
    for module_name, display_name in imports_to_check:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "installed")
            print(f"   ✅ {display_name:20} v{version}")
        except ImportError:
            print(f"   ❌ {display_name:20} NOT FOUND")
            all_ok = False
    
    # Check local RTMLib copy
    rtmlib_path = REPO_ROOT / "lib" / "rtmlib"
    if rtmlib_path.exists() and (rtmlib_path / "__init__.py").exists():
        try:
            version_file = rtmlib_path / "version.py"
            if version_file.exists():
                # Read version from version.py
                version_content = version_file.read_text()
                if "__version__" in version_content:
                    version = "local copy"
                else:
                    version = "local copy"
            else:
                version = "local copy"
            print(f"   ✅ {'RTMLib':20} {version} (lib/rtmlib/)")
        except Exception:
            print(f"   ✅ {'RTMLib':20} local copy (lib/rtmlib/)")
    else:
        print(f"   ❌ {'RTMLib':20} NOT FOUND (expected at lib/rtmlib/)")
        all_ok = False
    
    if not all_ok:
        print("\n⚠️  Some imports failed - check errors above")
        return False
    
    # Check CUDA
    print("\n🔍 Checking GPU/CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅ CUDA Available: {torch.cuda.get_device_name(0)}")
            print(f"   ✅ CUDA Version: {torch.version.cuda}")
        else:
            print(f"   ⚠️  No CUDA GPU - will use CPU")
    except Exception as e:
        print(f"   ❌ GPU check failed: {e}")
    
    print("\n✅ Installation verified")
    return True


# ============================================
# Main Execution
# ============================================
def main():
    """Main setup execution"""
    print("\n" + "🚀" * 35)
    print("UNIFIED POSE ESTIMATION PIPELINE - COMPLETE SETUP")
    print("🚀" * 35)
    print(f"\nSetup directory: {REPO_ROOT.absolute()}\n")
    
    try:
        # Execute all setup steps
        drive_mounted = step_0_mount_drive()
        step_1_install_core_dependencies()
        step_2_install_pytorch()
        step_3_install_cv_and_detection()
        step_4_install_pose_estimation()
        step_5_install_tracking()
        step_6_install_motionagformer_deps()
        step_7_setup_directories()
        step_8_download_models(drive_mounted)
        step_9_setup_demo_data(drive_mounted)
        step_10_verify_installation()
        
        # Success message
        print("\n" + "=" * 70)
        print("🎉 SETUP COMPLETE!")
        print("=" * 70)
        print("\n📚 Next Steps:")
        print("   1. Run: python verify_unified.py (comprehensive verification)")
        print("   2. Run: python udp_image.py --config configs/udp_image.yaml (quick test)")
        print("   3. Run: python udp_video.py --config configs/udp_video.yaml (2D pose)")
        print("   4. Run: python udp_3d_lifting.py (3D pose from 2D keypoints)")
        print("\n💡 Quick Commands:")
        print("   python udp_image.py --config configs/udp_image.yaml")
        print("   python udp_video.py --config configs/udp_video.yaml")
        print("   python udp_3d_lifting.py --keypoints demo_data/outputs/stage2_keypoints.npz")
        print("\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
