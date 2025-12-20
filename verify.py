"""
Unified Pose Estimation Pipeline - Environment Verification

Comprehensive verification of:
- All library imports and versions
- Model files and their locations
- Demo media files (images and videos)
- GPU/CUDA availability
- Directory structure
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ============================================
# Configuration
# ============================================
REPO_ROOT = Path(__file__).parent
LIB_DIR = REPO_ROOT / "lib"
MODELS_DIR = REPO_ROOT / "models"
DEMO_DATA_DIR = REPO_ROOT / "demo_data"
CONFIGS_DIR = REPO_ROOT / "configs"


# ============================================
# Helper Functions
# ============================================
def print_header(message: str, char: str = "=", width: int = 70):
    """Print formatted header"""
    print("\n" + char * width)
    print(message)
    print(char * width)


def print_section(title: str):
    """Print section header"""
    print(f"\n{'─' * 70}")
    print(f"📋 {title}")
    print('─' * 70)


# ============================================
# Verification Functions
# ============================================
def verify_imports() -> Tuple[bool, List[Dict]]:
    """Verify all required library imports"""
    print_section("Library Imports & Versions")
    
    imports_to_check = [
        # Core ML
        ("torch", "PyTorch", True),
        ("torchvision", "TorchVision", True),
        # Computer Vision
        ("cv2", "OpenCV", True),
        ("PIL", "Pillow", True),
        # Detection
        ("ultralytics", "Ultralytics/YOLO", True),
        # Pose Estimation
        ("rtmlib", "RTMLib", True),
        # ONNX
        ("onnx", "ONNX", False),
        ("onnxruntime", "ONNX Runtime", True),
        # Tracking (optional - has compatibility issues)
        # ("boxmot", "BoxMOT", False),
        ("supervision", "Supervision", False),
        # Scientific
        ("numpy", "NumPy", True),
        ("scipy", "SciPy", False),
        ("pandas", "Pandas", False),
        ("matplotlib", "Matplotlib", False),
        # Utilities
        ("yaml", "PyYAML", True),
        ("tqdm", "tqdm", True),
    ]
    
    results = []
    all_critical_ok = True
    
    for module_name, display_name, is_critical in imports_to_check:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "installed")
            status = "✅"
            success = True
            print(f"   {status} {display_name:25} v{version}")
        except ImportError as e:
            status = "❌" if is_critical else "⚠️ "
            success = False
            if is_critical:
                all_critical_ok = False
            print(f"   {status} {display_name:25} NOT FOUND")
        
        results.append({
            "module": module_name,
            "name": display_name,
            "critical": is_critical,
            "success": success,
        })
    
    return all_critical_ok, results


def verify_cuda() -> Tuple[bool, Dict]:
    """Verify CUDA/GPU availability"""
    print_section("GPU & CUDA")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        info = {
            "cuda_available": cuda_available,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
        }
        
        if cuda_available:
            info["device_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
            
            print(f"   ✅ CUDA Available: YES")
            print(f"   ✅ Device: {info['device_name']}")
            print(f"   ✅ CUDA Version: {info['cuda_version']}")
            print(f"   ✅ cuDNN Version: {info['cudnn_version']}")
            print(f"   ✅ Device Count: {info['device_count']}")
            
            # Check ONNX Runtime GPU
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in providers:
                    print(f"   ✅ ONNX Runtime GPU: Available")
                else:
                    print(f"   ⚠️  ONNX Runtime GPU: Not available (CPU only)")
                info["onnx_gpu"] = 'CUDAExecutionProvider' in providers
            except:
                print(f"   ⚠️  ONNX Runtime: Not installed")
                info["onnx_gpu"] = False
            
            return True, info
        else:
            print(f"   ⚠️  CUDA Available: NO (CPU mode)")
            print(f"   ℹ️  GPU acceleration will not be available")
            return False, info
            
    except ImportError:
        print(f"   ❌ PyTorch not installed")
        return False, {"error": "PyTorch not found"}


def verify_models() -> Tuple[bool, Dict]:
    """Verify model files"""
    print_section("Model Files")
    
    models_status = {
        "yolo": [],
        "vitpose": [],
        "rtmlib": [],
    }
    
    all_ok = True
    
    # YOLO models
    print("\n   🔍 YOLO Models:")
    yolo_dir = MODELS_DIR / "yolo"
    if yolo_dir.exists():
        yolo_files = list(yolo_dir.glob("*.pt"))
        if yolo_files:
            for model_file in yolo_files:
                size_mb = model_file.stat().st_size / (1024 ** 2)
                print(f"      ✅ {model_file.name:30} ({size_mb:.1f} MB)")
                models_status["yolo"].append(model_file.name)
        else:
            print(f"      ⚠️  No YOLO models found")
            all_ok = False
    else:
        print(f"      ❌ Directory not found: {yolo_dir}")
        all_ok = False
    
    # ViTPose models
    print("\n   🔍 ViTPose Models:")
    vitpose_dir = MODELS_DIR / "vitpose"
    if vitpose_dir.exists():
        vitpose_files = list(vitpose_dir.glob("*.pth")) + list(vitpose_dir.glob("*.onnx"))
        if vitpose_files:
            for model_file in vitpose_files:
                size_mb = model_file.stat().st_size / (1024 ** 2)
                print(f"      ✅ {model_file.name:30} ({size_mb:.1f} MB)")
                models_status["vitpose"].append(model_file.name)
        else:
            print(f"      ⚠️  No ViTPose models found")
            print(f"      ℹ️  Download from: https://github.com/ViTAE-Transformer/ViTPose")
    else:
        print(f"      ❌ Directory not found: {vitpose_dir}")
    
    # RTMLib models
    print("\n   🔍 RTMLib Models:")
    rtmlib_dir = MODELS_DIR / "rtmlib"
    if rtmlib_dir.exists():
        rtmlib_files = list(rtmlib_dir.glob("*.onnx")) + list(rtmlib_dir.glob("*.pth"))
        if rtmlib_files:
            for model_file in rtmlib_files:
                size_mb = model_file.stat().st_size / (1024 ** 2)
                print(f"      ✅ {model_file.name:30} ({size_mb:.1f} MB)")
                models_status["rtmlib"].append(model_file.name)
        else:
            print(f"      ℹ️  No local models (will download on first use)")
    else:
        print(f"      ℹ️  Directory not found (will be created on first use)")
    
    return all_ok, models_status


def verify_demo_data() -> Tuple[bool, Dict]:
    """Verify demo media files"""
    print_section("Demo Media Files")
    
    demo_status = {
        "videos": [],
        "images": [],
    }
    
    has_media = False
    
    # Videos
    print("\n   🎬 Videos:")
    videos_dir = DEMO_DATA_DIR / "videos"
    if videos_dir.exists():
        video_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.avi"))
        if video_files:
            for video_file in video_files:
                size_mb = video_file.stat().st_size / (1024 ** 2)
                print(f"      ✅ {video_file.name:30} ({size_mb:.1f} MB)")
                demo_status["videos"].append(video_file.name)
                has_media = True
        else:
            print(f"      ⚠️  No video files found")
    else:
        print(f"      ❌ Directory not found: {videos_dir}")
    
    # Images
    print("\n   🖼️  Images:")
    images_dir = DEMO_DATA_DIR / "images"
    if images_dir.exists():
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        if image_files:
            for image_file in image_files:
                size_kb = image_file.stat().st_size / 1024
                print(f"      ✅ {image_file.name:30} ({size_kb:.1f} KB)")
                demo_status["images"].append(image_file.name)
                has_media = True
        else:
            print(f"      ⚠️  No image files found")
    else:
        print(f"      ❌ Directory not found: {images_dir}")
    
    return has_media, demo_status


def verify_configs() -> Tuple[bool, List[str]]:
    """Verify configuration files"""
    print_section("Configuration Files")
    
    configs = []
    
    if CONFIGS_DIR.exists():
        config_files = list(CONFIGS_DIR.glob("*.yaml")) + list(CONFIGS_DIR.glob("*.yml"))
        if config_files:
            for config_file in config_files:
                print(f"   ✅ {config_file.name}")
                configs.append(config_file.name)
            return True, configs
        else:
            print(f"   ⚠️  No configuration files found")
            print(f"   ℹ️  Create config files in: {CONFIGS_DIR}")
            return False, []
    else:
        print(f"   ❌ Directory not found: {CONFIGS_DIR}")
        return False, []


def verify_directory_structure() -> bool:
    """Verify directory structure"""
    print_section("Directory Structure")
    
    required_dirs = [
        ("lib", LIB_DIR),
        ("models", MODELS_DIR),
        ("models/yolo", MODELS_DIR / "yolo"),
        ("models/vitpose", MODELS_DIR / "vitpose"),
        ("models/rtmlib", MODELS_DIR / "rtmlib"),
        ("demo_data", DEMO_DATA_DIR),
        ("demo_data/videos", DEMO_DATA_DIR / "videos"),
        ("demo_data/images", DEMO_DATA_DIR / "images"),
        ("demo_data/outputs", DEMO_DATA_DIR / "outputs"),
        ("configs", CONFIGS_DIR),
    ]
    
    all_ok = True
    for name, path in required_dirs:
        if path.exists():
            print(f"   ✅ {name:25} {path}")
        else:
            print(f"   ❌ {name:25} NOT FOUND")
            all_ok = False
    
    return all_ok


def run_functional_test() -> bool:
    """Run quick functional test"""
    print_section("Functional Tests")
    
    all_ok = True
    
    # Test PyTorch
    print("\n   🔬 Testing PyTorch...")
    try:
        import torch
        x = torch.randn(2, 3)
        y = x * 2
        print(f"      ✅ PyTorch tensor operations work")
    except Exception as e:
        print(f"      ❌ PyTorch test failed: {e}")
        all_ok = False
    
    # Test OpenCV
    print("\n   🔬 Testing OpenCV...")
    try:
        import cv2
        import numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"      ✅ OpenCV image operations work")
    except Exception as e:
        print(f"      ❌ OpenCV test failed: {e}")
        all_ok = False
    
    # Test YOLO
    print("\n   🔬 Testing YOLO...")
    try:
        from ultralytics import YOLO
        print(f"      ✅ YOLO import successful")
    except Exception as e:
        print(f"      ❌ YOLO test failed: {e}")
        all_ok = False
    
    # Test RTMLib
    print("\n   🔬 Testing RTMLib...")
    try:
        import rtmlib
        print(f"      ✅ RTMLib import successful")
    except Exception as e:
        print(f"      ❌ RTMLib test failed: {e}")
        all_ok = False
    
    return all_ok


# ============================================
# Main Execution
# ============================================
def main():
    """Main verification execution"""
    print("\n" + "🔍" * 35)
    print("UNIFIED POSE ESTIMATION PIPELINE - VERIFICATION")
    print("🔍" * 35)
    print(f"\nRepository root: {REPO_ROOT.absolute()}\n")
    
    # Run all verifications
    results = {}
    
    print_header("VERIFICATION REPORT", "=", 70)
    
    results["imports_ok"], results["imports_detail"] = verify_imports()
    results["cuda_ok"], results["cuda_detail"] = verify_cuda()
    results["models_ok"], results["models_detail"] = verify_models()
    results["demo_ok"], results["demo_detail"] = verify_demo_data()
    results["configs_ok"], results["configs_detail"] = verify_configs()
    results["dirs_ok"] = verify_directory_structure()
    results["functional_ok"] = run_functional_test()
    
    # Summary
    print_header("VERIFICATION SUMMARY", "=", 70)
    
    checks = [
        ("Library Imports", results["imports_ok"]),
        ("CUDA/GPU", results["cuda_ok"]),
        ("Model Files", results["models_ok"]),
        ("Demo Data", results["demo_ok"]),
        ("Config Files", results["configs_ok"]),
        ("Directory Structure", results["dirs_ok"]),
        ("Functional Tests", results["functional_ok"]),
    ]
    
    print()
    all_passed = True
    for check_name, status in checks:
        icon = "✅" if status else "⚠️ "
        print(f"   {icon} {check_name:25} {'PASS' if status else 'NEEDS ATTENTION'}")
        if not status:
            all_passed = False
    
    # Final verdict
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL CHECKS PASSED - Environment ready!")
        print("=" * 70)
        print("\n✨ Ready to run demos:")
        print("   python udp_image.py --config configs/udp_image.yaml")
        print("   python udp_video.py --config configs/udp_video.yaml")
        print()
        return 0
    else:
        print("⚠️  SOME CHECKS FAILED - Review above for details")
        print("=" * 70)
        print("\n📝 Recommendations:")
        if not results["imports_ok"]:
            print("   - Re-run setup_unified.py to install missing packages")
        if not results["models_ok"]:
            print("   - Download required model files")
        if not results["configs_ok"]:
            print("   - Create configuration files in configs/ directory")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
