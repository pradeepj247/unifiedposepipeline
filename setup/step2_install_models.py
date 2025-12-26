#!/usr/bin/env python3
"""
Step 2: Download Model Files

This script downloads all required model weights and checkpoints.
Corresponds to Step 8 in the original setup_unified.py.

Usage:
    python step2_install_models.py [--skip-existing] [--verbose]
"""

import os
import sys
import argparse
import subprocess
import re
from setup_utils import (
    is_colab_environment, print_header, print_step, run_command,
    check_file_exists, print_success, print_error, print_warning, COLOR_YELLOW
)


# Configuration
REPO_ROOT = "/content/unifiedposepipeline" if is_colab_environment() else os.getcwd()
MODELS_DIR = "/content/models"
DRIVE_MODELS = "/content/drive/MyDrive/models" if is_colab_environment() else None

# Global verbose flag
VERBOSE = False


def run_command_with_progress(cmd, description, expected_size_mb=None):
    """Run command with progress tracking for silent mode"""
    if VERBOSE:
        # Verbose mode: show all output
        return run_command(cmd)
    
    # Silent mode: show progress
    if expected_size_mb:
        print(f"  Downloading {description} (~{expected_size_mb} MB)")
    else:
        print(f"  Downloading {description}...")
    
    try:
        # Run command and capture output
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Parse gdown output for progress (if applicable)
        output = result.stdout
        if 'Downloading' in output or 'To:' in output:
            # Extract progress info from gdown output
            progress_lines = [line for line in output.split('\n') if '%' in line]
            if progress_lines:
                # Show last progress line
                print(f"  Progress: 100%")
        
        print(f"  ✓ Downloaded {description} successfully")
        return 0
        
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to download {description}")
        if VERBOSE:
            print(e.output)
        return e.returncode
    except Exception as e:
        print_error(f"Error downloading {description}: {e}")
        return 1


def download_yolo_models():
    """Download YOLO detection models"""
    print_step("2.1", "Download YOLO Models", indent=True)
    
    yolo_dir = os.path.join(MODELS_DIR, "yolo")
    
    models = {
        "yolov8s.pt": ("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt", 11.5)
    }
    
    for model_name, (url, size_mb) in models.items():
        model_path = os.path.join(yolo_dir, model_name)
        
        if check_file_exists(model_path):
            if not VERBOSE:
                print(f"  ✓ {model_name} (already exists)")
            else:
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
        cmd = f"curl -L '{url}' -o '{model_path}'"
        if VERBOSE:
            print(f"  Downloading {model_name}...")
            try:
                run_command(cmd)
                print(f"  ✓ Downloaded {model_name}")
            except Exception as e:
                print_warning(f"Failed to download {model_name}: {e}")
        else:
            run_command_with_progress(cmd, model_name, size_mb)


def download_vitpose_models():
    """Download ViTPose models"""
    print_step("2.2", "Download ViTPose Models", indent=True)
    
    vitpose_dir = os.path.join(MODELS_DIR, "vitpose")
    model_path = os.path.join(vitpose_dir, "vitpose-b.pth")
    
    if check_file_exists(model_path):
        if not VERBOSE:
            print("  ✓ vitpose-b.pth (already exists)")
        else:
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
    url = "https://github.com/ViTAE-Transformer/ViTPose/releases/download/v1.0/vitpose-b.pth"
    cmd = f"curl -L '{url}' -o '{model_path}'"
    
    if VERBOSE:
        print("  Downloading ViTPose-b (343 MB)...")
        try:
            run_command(cmd)
            print("  ✓ Downloaded vitpose-b.pth")
        except Exception as e:
            print_warning(f"Failed to download ViTPose: {e}")
    else:
        run_command_with_progress(cmd, "vitpose-b.pth", 343)


def download_rtmpose_models():
    """Download RTMPose ONNX models"""
    print_step("2.3", "Download RTMPose Models", indent=True)
    
    rtmpose_dir = os.path.join(MODELS_DIR, "rtmlib")
    
    models = {
        "rtmpose-l-coco-384x288.onnx": ("https://github.com/open-mmlab/mmpose/releases/download/v1.0.0/rtmpose-l-coco-384x288.onnx", 110),
        "rtmpose-l-halpe26-384x288.onnx": ("https://github.com/open-mmlab/mmpose/releases/download/v1.0.0/rtmpose-l-halpe26-384x288.onnx", 110)
    }
    
    for model_name, (url, size_mb) in models.items():
        model_path = os.path.join(rtmpose_dir, model_name)
        
        if check_file_exists(model_path):
            if not VERBOSE:
                print(f"  ✓ {model_name} (already exists)")
            else:
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
        cmd = f"curl -L '{url}' -o '{model_path}'"
        if VERBOSE:
            print(f"  Downloading {model_name} (~{size_mb} MB)...")
            try:
                run_command(cmd)
                print(f"  ✓ Downloaded {model_name}")
            except Exception as e:
                print_warning(f"Failed to download {model_name}: {e}")
        else:
            run_command_with_progress(cmd, model_name, size_mb)


def download_motionagformer_models():
    """Download MotionAGFormer checkpoint"""
    print_step("2.4", "Download MotionAGFormer Checkpoint", indent=True)
    
    magf_dir = os.path.join(MODELS_DIR, "motionagformer")
    model_path = os.path.join(magf_dir, "motionagformer-base-h36m.pth.tr")
    
    if check_file_exists(model_path):
        if not VERBOSE:
            print("  ✓ motionagformer-base-h36m.pth.tr (already exists)")
        else:
            print("  Skipping MotionAGFormer (already exists)")
        return
    
    # Check Drive backup
    if DRIVE_MODELS:
        drive_path = os.path.join(DRIVE_MODELS, "motionagformer", "motionagformer-base-h36m.pth.tr")
        if os.path.exists(drive_path):
            print("  Copying from Drive: motionagformer-base-h36m.pth.tr")
            run_command(f"cp '{drive_path}' '{model_path}'")
            return
    
    # Download from Google Drive using gdown with --fuzzy flag
    gdrive_id = "1Iii5EwsFFm9_9lKBUPfN8bV5LmfkNUMP"
    
    if VERBOSE:
        print("  Downloading MotionAGFormer checkpoint (~200 MB)...")
        try:
            run_command(f"gdown --fuzzy {gdrive_id} -O '{model_path}'")
            print("  ✓ Downloaded motionagformer-base-h36m.pth.tr")
        except Exception as e:
            print_warning(f"Failed to download MotionAGFormer: {e}")
    else:
        cmd = f"gdown --fuzzy {gdrive_id} -O '{model_path}'"
        run_command_with_progress(cmd, "motionagformer-base-h36m.pth.tr", 200)


def download_wb3d_models():
    """Download Wholebody 3D models"""
    print_step("2.5", "Download Wholebody 3D Models", indent=True)
    
    wb3d_dir = os.path.join(MODELS_DIR, "wb3d")
    model_path = os.path.join(wb3d_dir, "rtmw3d-l.onnx")
    
    if check_file_exists(model_path):
        print("  Skipping rtmw3d-l.onnx (already exists)")
        return
    
    # Check Drive backup (primary source - stored in rtmw3d_onnx_exports/)
    if is_colab_environment() and os.path.exists("/content/drive/MyDrive"):
        drive_path = "/content/drive/MyDrive/rtmw3d_onnx_exports/rtmw3d-l.onnx"
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
    print_step("2.6", "Download ReID Models", indent=True)
    
    reid_dir = os.path.join(MODELS_DIR, "reid")
    model_path = os.path.join(reid_dir, "osnet_x1_0_msmt17.pt")
    
    if check_file_exists(model_path):
        if not VERBOSE:
            print("  ✓ osnet_x1_0_msmt17.pt (already exists)")
        else:
            print("  Skipping OSNet x1.0 (already exists)")
        return
    
    # Check Drive backup
    if DRIVE_MODELS:
        drive_path = os.path.join(DRIVE_MODELS, "reid", "osnet_x1_0_msmt17.pt")
        if os.path.exists(drive_path):
            print("  Copying from Drive: osnet_x1_0_msmt17.pt")
            run_command(f"cp '{drive_path}' '{model_path}'")
            return
    
    # Download from Google Drive using gdown with --fuzzy flag
    gdrive_id = "1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY"
    
    if VERBOSE:
        print("  Downloading OSNet x1.0 MSMT17 (~25 MB)...")
        try:
            run_command(f"gdown --fuzzy {gdrive_id} -O '{model_path}'")
            print("  ✓ Downloaded osnet_x1_0_msmt17.pt")
        except Exception as e:
            print_warning(f"Failed to download ReID model: {e}")
    else:
        cmd = f"gdown --fuzzy {gdrive_id} -O '{model_path}'"
        run_command_with_progress(cmd, "osnet_x1_0_msmt17.pt", 25)


def main():
    """Main execution function"""
    global VERBOSE
    
    parser = argparse.ArgumentParser(description="Download all model files")
    parser.add_argument("--skip-existing", action="store_true",
                       help="Skip downloads if models already exist")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed download output (default: silent mode)")
    args = parser.parse_args()
    
    VERBOSE = args.verbose
    
    print_header("STEP 2: Download Model Files", color=COLOR_YELLOW)
    
    if not VERBOSE:
        print("Running in silent mode. Use --verbose for detailed output.")
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
        
        print_success("Model download complete!", color=COLOR_YELLOW)
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
