"""
Quick verification script to check if the environment is properly set up
Run this after setup_environment.py to ensure everything is working
"""

import sys
from pathlib import Path

def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def check_imports():
    """Check if all required packages can be imported"""
    print_section("Checking Package Imports")
    
    packages = {
        "Core ML": [
            ("torch", "PyTorch"),
            ("torchvision", "TorchVision"),
            ("torchaudio", "TorchAudio"),
        ],
        "Computer Vision": [
            ("cv2", "OpenCV"),
            ("PIL", "Pillow"),
        ],
        "YOLO & Detection": [
            ("ultralytics", "Ultralytics"),
        ],
        "ONNX Runtime": [
            ("onnxruntime", "ONNX Runtime"),
        ],
        "Scientific Computing": [
            ("numpy", "NumPy"),
            ("scipy", "SciPy"),
            ("pandas", "Pandas"),
        ],
        "Visualization": [
            ("matplotlib", "Matplotlib"),
            ("seaborn", "Seaborn"),
        ],
        "Utilities": [
            ("yaml", "PyYAML"),
            ("tqdm", "tqdm"),
        ],
    }
    
    all_success = True
    
    for category, pkg_list in packages.items():
        print(f"\n{category}:")
        for module_name, display_name in pkg_list:
            try:
                module = __import__(module_name)
                version = getattr(module, "__version__", "installed")
                print(f"  ✅ {display_name:20} v{version}")
            except ImportError:
                print(f"  ❌ {display_name:20} NOT FOUND")
                all_success = False
    
    return all_success


def check_cuda():
    """Check CUDA availability"""
    print_section("Checking GPU/CUDA")
    
    try:
        import torch
        
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"    Compute Capability: {props.major}.{props.minor}")
        else:
            print("⚠️  No CUDA GPU detected - CPU mode only")
            
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False
    
    return True


def check_library_structure():
    """Check if library files are in place"""
    print_section("Checking Library Structure")
    
    repo_root = Path(__file__).parent
    
    checks = [
        (repo_root / "lib" / "vitpose" / "__init__.py", "ViTPose library"),
        (repo_root / "lib" / "rtmlib" / "__init__.py", "RTMLib library"),
        (repo_root / "models", "Models directory"),
        (repo_root / "demos", "Demos directory"),
        (repo_root / "notebooks", "Notebooks directory"),
        (repo_root / "requirements.txt", "Requirements file"),
        (repo_root / "setup.py", "Setup file"),
    ]
    
    all_exist = True
    
    for path, description in checks:
        if path.exists():
            print(f"  ✅ {description:30} {path.name}")
        else:
            print(f"  ❌ {description:30} NOT FOUND")
            all_exist = False
    
    return all_exist


def check_models():
    """Check if any models are downloaded"""
    print_section("Checking Downloaded Models")
    
    repo_root = Path(__file__).parent
    models_dir = repo_root / "models"
    
    if not models_dir.exists():
        print("❌ Models directory not found")
        return False
    
    # Check for YOLO models
    yolo_dir = models_dir / "yolo"
    if yolo_dir.exists():
        yolo_models = list(yolo_dir.glob("*.pt"))
        if yolo_models:
            print(f"\n✅ YOLO Models ({len(yolo_models)} found):")
            for model in yolo_models:
                size_mb = model.stat().st_size / (1024 ** 2)
                print(f"  • {model.name} ({size_mb:.1f} MB)")
        else:
            print("\n⚠️  No YOLO models found - run download_models.py")
    
    # Check for ViTPose models
    vitpose_dir = models_dir / "vitpose"
    if vitpose_dir.exists():
        vitpose_models = list(vitpose_dir.glob("*.pth")) + list(vitpose_dir.glob("*.pt"))
        if vitpose_models:
            print(f"\n✅ ViTPose Models ({len(vitpose_models)} found):")
            for model in vitpose_models:
                size_mb = model.stat().st_size / (1024 ** 2)
                print(f"  • {model.name} ({size_mb:.1f} MB)")
        else:
            print("\n⚠️  No ViTPose models found - download manually from GitHub")
    
    # RTMLib models are cached automatically
    print("\n✅ RTMLib models will be downloaded automatically on first use")
    
    return True


def run_quick_test():
    """Run a quick functional test"""
    print_section("Running Quick Functional Test")
    
    try:
        import torch
        import cv2
        import numpy as np
        from ultralytics import YOLO
        
        # Test tensor creation
        print("\n1. Testing PyTorch tensor creation...")
        tensor = torch.randn(3, 224, 224)
        print(f"   ✅ Created tensor with shape {tensor.shape}")
        
        if torch.cuda.is_available():
            tensor_gpu = tensor.cuda()
            print(f"   ✅ Moved tensor to GPU")
        
        # Test OpenCV
        print("\n2. Testing OpenCV image creation...")
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        print(f"   ✅ Created image with shape {img.shape}")
        
        # Test YOLO loading (if model exists)
        print("\n3. Testing YOLO model loading...")
        repo_root = Path(__file__).parent
        yolo_model_path = repo_root / "models" / "yolo" / "yolov8n.pt"
        
        if yolo_model_path.exists():
            model = YOLO(str(yolo_model_path))
            print(f"   ✅ Loaded YOLO model from {yolo_model_path.name}")
        else:
            print(f"   ⚠️  YOLO model not found at {yolo_model_path}")
            print(f"      Run setup_environment.py or download_models.py")
        
        print("\n✅ All functional tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Functional test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main verification function"""
    print("\n" + "🔍" * 30)
    print("UNIFIED POSE ESTIMATION - ENVIRONMENT VERIFICATION")
    print("🔍" * 30)
    
    results = {
        "Imports": check_imports(),
        "CUDA": check_cuda(),
        "Library Structure": check_library_structure(),
        "Models": check_models(),
        "Functional Test": run_quick_test(),
    }
    
    # Summary
    print_section("Verification Summary")
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {check_name:20} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("🎉 Environment is fully set up and ready to use!")
        print("\n💡 Next steps:")
        print("   • Check out notebooks/ for tutorials")
        print("   • Run demos in demos/ directory")
        print("   • Read README.md for documentation")
    else:
        print("⚠️  Some checks failed. Please review the output above.")
        print("\n💡 Troubleshooting:")
        print("   • Run: python setup_environment.py")
        print("   • Run: python download_models.py")
        print("   • Check README.md for manual setup instructions")
    
    print("\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
