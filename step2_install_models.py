#!/usr/bin/env python3
"""
Step 2: Download Model Files

This script downloads all required model weights and checkpoints.
Corresponds to Step 8 in the original setup_unified.py.

Usage:
    python step2_install_models.py [--skip-existing]
"""

import os
import sys
import argparse
from setup_utils import (
    is_colab_environment, print_header, print_step, run_command,
    check_file_exists, print_success, print_error, print_warning
)


# Configuration
REPO_ROOT = "/content/unifiedposepipeline" if is_colab_environment() else os.getcwd()
MODELS_DIR = "/content/models"
DRIVE_MODELS = "/content/drive/MyDrive/models" if is_colab_environment() else None


def download_yolo_models():
    """Download YOLO detection models"""
    print_step("8.1", "Download YOLO Models")
    
    yolo_dir = os.path.join(MODELS_DIR, "yolo")
    
    models = {
        "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt",
        "yolov8x.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt"
    }
    
    for model_name, url in models.items():
        model_path = os.path.join(yolo_dir, model_name)
        
        if check_file_exists(model_path):
            print(f"  Skipping {model_name} (already exists)")
            continue
        
        # Check Drive backup
        if DRIVE_MODELS:
            drive_path = os.path.join(DRIVE_MODELS, "yolo", model_name)
            if os.path.exists(drive_path):
                print(f"  Copying from Drive: {model_name}")
                run_command(f"cp '{drive_path}' '{model_path}'")
                continue
        
        # Download from GitHub
        print(f"  Downloading {model_name}...")
        cmd = f"curl -L '{url}' -o '{model_path}'"
        try:
            run_command(cmd)
            print(f"  ✓ Downloaded {model_name}")
        except Exception as e:
            print_warning(f"Failed to download {model_name}: {e}")


def download_vitpose_models():
    """Download ViTPose models"""
    print_step("8.2", "Download ViTPose Models")
    
    vitpose_dir = os.path.join(MODELS_DIR, "vitpose")
    model_path = os.path.join(vitpose_dir, "vitpose-b.pth")
    
    if check_file_exists(model_path):
        print("  Skipping ViTPose-b (already exists)")
        return
    
    # Check Drive backup
    if DRIVE_MODELS:
        drive_path = os.path.join(DRIVE_MODELS, "vitpose", "vitpose-b.pth")
        if os.path.exists(drive_path):
            print("  Copying from Drive: vitpose-b.pth")
            run_command(f"cp '{drive_path}' '{model_path}'")
            return
    
    # Download from GitHub releases
    print("  Downloading ViTPose-b (343 MB)...")
    url = "https://github.com/ViTAE-Transformer/ViTPose/releases/download/v1.0/vitpose-b.pth"
    cmd = f"curl -L '{url}' -o '{model_path}'"
    
    try:
        run_command(cmd)
        print("  ✓ Downloaded vitpose-b.pth")
    except Exception as e:
        print_warning(f"Failed to download ViTPose: {e}")


def download_rtmpose_models():
    """Download RTMPose ONNX models"""
    print_step("8.3", "Download RTMPose Models")
    
    rtmpose_dir = os.path.join(MODELS_DIR, "rtmlib")
    
    models = {
        "rtmpose-l-coco-384x288.onnx": "https://github.com/open-mmlab/mmpose/releases/download/v1.0.0/rtmpose-l-coco-384x288.onnx",
        "rtmpose-l-halpe26-384x288.onnx": "https://github.com/open-mmlab/mmpose/releases/download/v1.0.0/rtmpose-l-halpe26-384x288.onnx"
    }
    
    for model_name, url in models.items():
        model_path = os.path.join(rtmpose_dir, model_name)
        
        if check_file_exists(model_path):
            print(f"  Skipping {model_name} (already exists)")
            continue
        
        # Check Drive backup
        if DRIVE_MODELS:
            drive_path = os.path.join(DRIVE_MODELS, "rtmlib", model_name)
            if os.path.exists(drive_path):
                print(f"  Copying from Drive: {model_name}")
                run_command(f"cp '{drive_path}' '{model_path}'")
                continue
        
        # Download from GitHub
        print(f"  Downloading {model_name} (~110 MB)...")
        cmd = f"curl -L '{url}' -o '{model_path}'"
        try:
            run_command(cmd)
            print(f"  ✓ Downloaded {model_name}")
        except Exception as e:
            print_warning(f"Failed to download {model_name}: {e}")


def download_motionagformer_models():
    """Download MotionAGFormer checkpoint"""
    print_step("8.4", "Download MotionAGFormer Checkpoint")
    
    magf_dir = os.path.join(MODELS_DIR, "motionagformer")
    model_path = os.path.join(magf_dir, "motionagformer-l.pth.tar")
    
    if check_file_exists(model_path):
        print("  Skipping MotionAGFormer (already exists)")
        return
    
    # Check Drive backup
    if DRIVE_MODELS:
        drive_path = os.path.join(DRIVE_MODELS, "motionagformer", "motionagformer-l.pth.tar")
        if os.path.exists(drive_path):
            print("  Copying from Drive: motionagformer-l.pth.tar")
            run_command(f"cp '{drive_path}' '{model_path}'")
            return
    
    # Download from Google Drive using gdown
    print("  Downloading MotionAGFormer checkpoint (~200 MB)...")
    gdrive_id = "1RJKHZsNaLhZSYwcY0ofWgU_LI-_-s81F"
    
    try:
        run_command(f"gdown {gdrive_id} -O '{model_path}'")
        print("  ✓ Downloaded motionagformer-l.pth.tar")
    except Exception as e:
        print_warning(f"Failed to download MotionAGFormer: {e}")


def download_wb3d_models():
    """Download Wholebody 3D models"""
    print_step("8.5", "Download Wholebody 3D Models")
    
    wb3d_dir = os.path.join(MODELS_DIR, "wb3d")
    model_path = os.path.join(wb3d_dir, "rtmw3d-l.onnx")
    
    if check_file_exists(model_path):
        print("  Skipping rtmw3d-l.onnx (already exists)")
        return
    
    # Check Drive backup (primary source)
    if DRIVE_MODELS:
        drive_path = os.path.join(DRIVE_MODELS, "wb3d", "rtmw3d-l.onnx")
        if os.path.exists(drive_path):
            print("  Copying from Drive: rtmw3d-l.onnx")
            run_command(f"cp '{drive_path}' '{model_path}'")
            return
        else:
            print_warning("rtmw3d-l.onnx not found in Drive")
            print("  Please manually place the model in:")
            print(f"    {drive_path}")
            print("  or")
            print(f"    {model_path}")
    else:
        print_warning("Drive not mounted, cannot copy rtmw3d-l.onnx")
        print(f"  Please manually download and place in: {model_path}")


def download_reid_models():
    """Download ReID models"""
    print_step("8.6", "Download ReID Models")
    
    reid_dir = os.path.join(MODELS_DIR, "reid")
    model_path = os.path.join(reid_dir, "osnet_x1_0_msmt17.pt")
    
    if check_file_exists(model_path):
        print("  Skipping OSNet x1.0 (already exists)")
        return
    
    # Check Drive backup
    if DRIVE_MODELS:
        drive_path = os.path.join(DRIVE_MODELS, "reid", "osnet_x1_0_msmt17.pt")
        if os.path.exists(drive_path):
            print("  Copying from Drive: osnet_x1_0_msmt17.pt")
            run_command(f"cp '{drive_path}' '{model_path}'")
            return
    
    # Download from Google Drive using gdown
    print("  Downloading OSNet x1.0 MSMT17 (~25 MB)...")
    gdrive_id = "1IosIFlLiulGIjwbXkxQF0EF_eY9S5p-4"
    
    try:
        run_command(f"gdown {gdrive_id} -O '{model_path}'")
        print("  ✓ Downloaded osnet_x1_0_msmt17.pt")
    except Exception as e:
        print_warning(f"Failed to download ReID model: {e}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Download all model files")
    parser.add_argument("--skip-existing", action="store_true",
                       help="Skip downloads if models already exist")
    args = parser.parse_args()
    
    print_header("STEP 2: Download Model Files")
    
    print("This script will download all required model weights.")
    print(f"Models directory: {MODELS_DIR}")
    if DRIVE_MODELS:
        print(f"Drive backup: {DRIVE_MODELS}")
    print()
    
    try:
        download_yolo_models()
        download_vitpose_models()
        download_rtmpose_models()
        download_motionagformer_models()
        download_wb3d_models()
        download_reid_models()
        
        print_success("Model download complete!")
        print("\nNext steps:")
        print("  python step3_pull_demodata.py    # Setup demo data")
        print("  python step4_verify_envt.py      # Verify installation")
        
    except KeyboardInterrupt:
        print("\n\n⊘ Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
