#!/usr/bin/env python3
"""
Step 4: Verify Environment

This script performs comprehensive verification of the installation.
Merges Step 10 from setup_unified.py and verify_unified.py.

Usage:
    python step4_verify_envt.py [--quick] [--detailed]
"""

import os
import sys
import argparse
from setup_utils import (
    is_colab_environment, print_header, print_step, check_import,
    check_file_exists, print_success, print_error, print_warning
)


# Configuration
REPO_ROOT = "/content/unifiedposepipeline" if is_colab_environment() else os.getcwd()
MODELS_DIR = "/content/models"


def verify_python_packages():
    """Verify all required Python packages are installed"""
    print_step("10.1", "Verify Python Packages")
    
    packages = [
        ("numpy", None),
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("ultralytics", "YOLO"),
        ("onnx", None),
        ("onnxruntime", "ONNX Runtime"),
        ("boxmot", "BoxMOT"),
        ("supervision", None),
        ("timm", None),
        ("easydict", None),
        ("yaml", "PyYAML"),
        ("matplotlib", None),
        ("PIL", "Pillow"),
        ("scipy", None),
        ("pandas", None),
        ("tqdm", None)
    ]
    
    success_count = 0
    failed_packages = []
    
    for module_name, display_name in packages:
        if check_import(module_name, display_name):
            success_count += 1
        else:
            failed_packages.append(display_name or module_name)
    
    print(f"\n✓ {success_count}/{len(packages)} packages verified")
    
    if failed_packages:
        print_warning(f"Missing packages: {', '.join(failed_packages)}")
        return False
    
    return True


def verify_cuda_gpu():
    """Verify CUDA and GPU availability"""
    print_step("10.2", "Verify CUDA and GPU")
    
    try:
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print_warning("CUDA not available - will use CPU (slower)")
            return False
            
    except Exception as e:
        print_error(f"Failed to check CUDA: {e}")
        return False


def verify_model_files():
    """Verify all required model files exist"""
    print_step("10.3", "Verify Model Files")
    
    models = [
        # YOLO models
        (os.path.join(MODELS_DIR, "yolo", "yolov8s.pt"), "YOLO Small"),
        (os.path.join(MODELS_DIR, "yolo", "yolov8x.pt"), "YOLO Extra Large"),
        
        # RTMPose models
        (os.path.join(MODELS_DIR, "rtmlib", "rtmpose-l-coco-384x288.onnx"), "RTMPose COCO"),
        (os.path.join(MODELS_DIR, "rtmlib", "rtmpose-l-halpe26-384x288.onnx"), "RTMPose Halpe26"),
        
        # ViTPose model
        (os.path.join(MODELS_DIR, "vitpose", "vitpose-b.pth"), "ViTPose-B"),
        
        # Wholebody 3D model
        (os.path.join(MODELS_DIR, "wb3d", "rtmw3d-l.onnx"), "Wholebody 3D"),
        
        # MotionAGFormer model
        (os.path.join(MODELS_DIR, "motionagformer", "motionagformer-l.pth.tar"), "MotionAGFormer"),
        
        # ReID model
        (os.path.join(MODELS_DIR, "reid", "osnet_x1_0_msmt17.pt"), "ReID OSNet x1.0")
    ]
    
    success_count = 0
    missing_models = []
    
    for model_path, model_name in models:
        if check_file_exists(model_path):
            success_count += 1
        else:
            missing_models.append(model_name)
    
    print(f"\n✓ {success_count}/{len(models)} model files verified")
    
    if missing_models:
        print_warning(f"Missing models: {', '.join(missing_models)}")
        return False
    
    return True


def verify_demo_data():
    """Verify demo data files exist"""
    print_step("10.4", "Verify Demo Data")
    
    demo_files = [
        (os.path.join(REPO_ROOT, "demo_data", "videos", "dance.mp4"), "dance.mp4"),
        (os.path.join(REPO_ROOT, "demo_data", "videos", "campus_walk.mp4"), "campus_walk.mp4"),
        (os.path.join(REPO_ROOT, "demo_data", "images", "sample.jpg"), "sample.jpg")
    ]
    
    success_count = 0
    missing_files = []
    
    for file_path, file_name in demo_files:
        if check_file_exists(file_path):
            success_count += 1
        else:
            missing_files.append(file_name)
    
    print(f"\n✓ {success_count}/{len(demo_files)} demo files verified")
    
    if missing_files:
        print_warning(f"Missing demo files: {', '.join(missing_files)}")
        return False
    
    return True


def verify_directories():
    """Verify directory structure"""
    print_step("10.5", "Verify Directory Structure")
    
    directories = [
        (MODELS_DIR, "Models root"),
        (os.path.join(MODELS_DIR, "yolo"), "YOLO models"),
        (os.path.join(MODELS_DIR, "rtmlib"), "RTMPose models"),
        (os.path.join(MODELS_DIR, "vitpose"), "ViTPose models"),
        (os.path.join(MODELS_DIR, "wb3d"), "Wholebody 3D models"),
        (os.path.join(MODELS_DIR, "motionagformer"), "MotionAGFormer models"),
        (os.path.join(MODELS_DIR, "reid"), "ReID models"),
        (os.path.join(REPO_ROOT, "demo_data"), "Demo data root"),
        (os.path.join(REPO_ROOT, "demo_data", "videos"), "Demo videos"),
        (os.path.join(REPO_ROOT, "demo_data", "images"), "Demo images"),
        (os.path.join(REPO_ROOT, "demo_data", "outputs"), "Demo outputs"),
        (os.path.join(REPO_ROOT, "configs"), "Configs")
    ]
    
    success_count = 0
    missing_dirs = []
    
    for dir_path, dir_name in directories:
        if os.path.isdir(dir_path):
            print(f"✓ {dir_name}")
            success_count += 1
        else:
            print(f"✗ {dir_name}")
            missing_dirs.append(dir_name)
    
    print(f"\n✓ {success_count}/{len(directories)} directories verified")
    
    if missing_dirs:
        print_warning(f"Missing directories: {', '.join(missing_dirs)}")
        return False
    
    return True


def run_quick_inference_test():
    """Run quick inference tests (optional)"""
    print_step("10.6", "Quick Inference Test (Optional)")
    
    print("Testing YOLO detection...")
    try:
        from ultralytics import YOLO
        model_path = os.path.join(MODELS_DIR, "yolo", "yolov8s.pt")
        
        if not os.path.exists(model_path):
            print("⊘ YOLO model not found, skipping test")
            return False
        
        model = YOLO(model_path)
        print("✓ YOLO model loaded successfully")
        
        # Test on sample image if available
        sample_image = os.path.join(REPO_ROOT, "demo_data", "images", "sample.jpg")
        if os.path.exists(sample_image):
            results = model(sample_image, verbose=False)
            print(f"✓ YOLO inference successful ({len(results[0].boxes)} detections)")
        
        return True
        
    except Exception as e:
        print_warning(f"YOLO test failed: {e}")
        return False


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Verify environment setup")
    parser.add_argument("--quick", action="store_true",
                       help="Skip optional inference tests")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed information for all checks")
    args = parser.parse_args()
    
    print_header("STEP 4: Verify Environment")
    
    print("This script will verify your installation.")
    print(f"Repository root: {REPO_ROOT}")
    print(f"Models directory: {MODELS_DIR}")
    print()
    
    results = {}
    
    try:
        # Run all verification steps
        results["packages"] = verify_python_packages()
        results["cuda"] = verify_cuda_gpu()
        results["directories"] = verify_directories()
        results["models"] = verify_model_files()
        results["demo_data"] = verify_demo_data()
        
        if not args.quick:
            results["inference"] = run_quick_inference_test()
        
        # Summary
        print("\n" + "=" * 70)
        print("VERIFICATION SUMMARY")
        print("=" * 70)
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        for check, status in results.items():
            status_str = "✓ PASS" if status else "✗ FAIL"
            print(f"{check.upper():20s} {status_str}")
        
        print(f"\nOverall: {passed}/{total} checks passed")
        
        if passed == total:
            print_success("Environment verification complete! All checks passed.")
            print("\nYou can now use the pipeline:")
            print("  python udp_video.py --config configs/udp_video.yaml")
            print("  python run_detector_tracking.py --config configs/detector_tracking_benchmark.yaml")
        else:
            print_warning("Some checks failed. Please review the output above.")
            print("\nYou may need to:")
            print("  - Re-run step1_install_libs_deps.py for missing packages")
            print("  - Re-run step2_install_models.py for missing models")
            print("  - Re-run step3_pull_demodata.py for missing demo data")
        
    except KeyboardInterrupt:
        print("\n\n⊘ Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
