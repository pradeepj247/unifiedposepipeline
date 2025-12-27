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

# Color codes for output
COLOR_ORANGE = "\033[38;5;208m"  # Orange color for missing files
COLOR_GREEN = "\033[92m"  # Green color for success
COLOR_RESET = "\033[0m"  # Reset color

# Model name mapping (filename -> friendly display name)
MODEL_NAMES = {
    "vitpose-b.pth": "VITPose",
    "yolov8s.pt": "YOLOv8s",
    "rtmpose-l-coco-384x288.onnx": "RTMPose (COCO)",
    "rtmw3d-l.onnx": "RTM WholeBody 3D",
    "osnet_x1_0_msmt17.pt": "OSNet x1.0 (PyTorch)",
    "osnet_x0_25_msmt17.pt": "OSNet x0.25 (PyTorch)",
    "osnet_x0_25_msmt17.onnx": "OSNet x0.25 (ONNX)",
    "motionagformer-base-h36m.pth.tr": "MotionAGFormer",
    "rtmpose-l-halpe26-384x288.onnx": "RTMPose (Halpe26)"
}

def get_model_display_name(filename):
    """Get friendly display name for a model file"""
    return MODEL_NAMES.get(filename, filename)


def run_command_with_progress(cmd, filename, model_path, expected_size_mb=None):
    """Run command with progress tracking for silent mode"""
    display_name = get_model_display_name(filename)
    
    if VERBOSE:
        # Verbose mode: show all output
        return run_command(cmd)
    
    # Silent mode: show progress with rocket emoji
    if expected_size_mb:
        print(f"  🚀 Downloading {display_name} (~{expected_size_mb} MB)")
    else:
        print(f"  🚀 Downloading {display_name}...")
    
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
        
        print(f"  {COLOR_GREEN}✓{COLOR_RESET} Downloaded {display_name} successfully to {model_path}")
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
        "yolov8s.pt": ("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt", 22)
    }
    
    for model_name, (url, size_mb) in models.items():
        model_path = os.path.join(yolo_dir, model_name)
        display_name = get_model_display_name(model_name)
        
        if check_file_exists(model_path, quiet=True):
            # Get file size
            file_size_bytes = os.path.getsize(model_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            print(f"  {COLOR_GREEN}✓{COLOR_RESET} {display_name} already exists: {model_path} ({file_size_mb:.1f} MB)")
            continue
        
        print(f"  {COLOR_ORANGE}✗{COLOR_RESET} {display_name} not found")
        
        # Check Drive backup
        if DRIVE_MODELS:
            drive_path = os.path.join(DRIVE_MODELS, "yolo", model_name)
            if os.path.exists(drive_path):
                print(f"  Copying from Drive: {display_name}")
                run_command(f"cp '{drive_path}' '{model_path}'")
                continue
        
        # Download from GitHub
        cmd = f"curl -L '{url}' -o '{model_path}'"
        if VERBOSE:
            print(f"  Downloading {display_name}...")
            try:
                run_command(cmd)
                print(f"  {COLOR_GREEN}✓{COLOR_RESET} Downloaded {display_name}")
            except Exception as e:
                print_warning(f"Failed to download {display_name}: {e}")
        else:
            run_command_with_progress(cmd, model_name, model_path, size_mb)


def download_vitpose_models():
    """Download ViTPose models"""
    print_step("2.2", "Download ViTPose Models", indent=True)
    
    vitpose_dir = os.path.join(MODELS_DIR, "vitpose")
    model_name = "vitpose-b.pth"
    model_path = os.path.join(vitpose_dir, model_name)
    display_name = get_model_display_name(model_name)
    
    if check_file_exists(model_path, quiet=True):
        # Get file size
        file_size_bytes = os.path.getsize(model_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        print(f"  {COLOR_GREEN}✓{COLOR_RESET} {display_name} already exists: {model_path} ({file_size_mb:.1f} MB)")
        return
    
    print(f"  {COLOR_ORANGE}✗{COLOR_RESET} {display_name} not found")
    
    # Check Drive backup
    if DRIVE_MODELS:
        drive_path = os.path.join(DRIVE_MODELS, "vitpose", model_name)
        if os.path.exists(drive_path):
            print(f"  Copying from Drive: {display_name}")
            run_command(f"cp '{drive_path}' '{model_path}'")
            return
    
    # Download from GitHub releases
    url = "https://github.com/pradeepj247/easy-pose-pipeline/releases/download/v1.0/vitpose-b.pth"
    cmd = f"curl -L '{url}' -o '{model_path}'"
    
    if VERBOSE:
        print(f"  Downloading {display_name} (343 MB)...")
        try:
            run_command(cmd)
            print(f"  {COLOR_GREEN}✓{COLOR_RESET} Downloaded {display_name}")
        except Exception as e:
            print_warning(f"Failed to download {display_name}: {e}")
    else:
        run_command_with_progress(cmd, model_name, model_path, 343)


def download_rtmpose_models():
    """Download RTMPose ONNX models"""
    print_step("2.3", "Download RTMPose Models", indent=True)
    
    rtmpose_dir = os.path.join(MODELS_DIR, "rtmlib")
    
    models = {
        "rtmpose-l-coco-384x288.onnx": ("https://github.com/pradeepj247/unifiedposepipeline/releases/download/v1.0.0/rtmpose-l-coco-384x288.onnx", 110),
        "rtmpose-l-halpe26-384x288.onnx": ("https://github.com/pradeepj247/unifiedposepipeline/releases/download/v1.0.0/rtmpose-l-halpe26-384x288.onnx", 110)
    }
    
    for model_name, (url, size_mb) in models.items():
        model_path = os.path.join(rtmpose_dir, model_name)
        display_name = get_model_display_name(model_name)
        
        if check_file_exists(model_path, quiet=True):
            # Get file size
            file_size_bytes = os.path.getsize(model_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            print(f"  {COLOR_GREEN}✓{COLOR_RESET} {display_name} already exists: {model_path} ({file_size_mb:.1f} MB)")
            continue
        
        print(f"  {COLOR_ORANGE}✗{COLOR_RESET} {display_name} not found")
        
        # Check Drive backup
        if DRIVE_MODELS:
            drive_path = os.path.join(DRIVE_MODELS, "rtmlib", model_name)
            if os.path.exists(drive_path):
                print(f"  Copying from Drive: {display_name}")
                run_command(f"cp '{drive_path}' '{model_path}'")
                continue
        
        # Download from GitHub
        cmd = f"curl -L '{url}' -o '{model_path}'"
        if VERBOSE:
            print(f"  Downloading {display_name} (~{size_mb} MB)...")
            try:
                run_command(cmd)
                print(f"  {COLOR_GREEN}✓{COLOR_RESET} Downloaded {display_name}")
            except Exception as e:
                print_warning(f"Failed to download {display_name}: {e}")
        else:
            run_command_with_progress(cmd, model_name, model_path, size_mb)


def download_motionagformer_models():
    """Download MotionAGFormer checkpoint"""
    print_step("2.4", "Download MotionAGFormer Checkpoint", indent=True)
    
    magf_dir = os.path.join(MODELS_DIR, "motionagformer")
    model_name = "motionagformer-base-h36m.pth.tr"
    model_path = os.path.join(magf_dir, model_name)
    display_name = get_model_display_name(model_name)
    
    if check_file_exists(model_path, quiet=True):
        # Get file size
        file_size_bytes = os.path.getsize(model_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        print(f"  {COLOR_GREEN}✓{COLOR_RESET} {display_name} already exists: {model_path} ({file_size_mb:.1f} MB)")
        return
    
    print(f"  {COLOR_ORANGE}✗{COLOR_RESET} {display_name} not found")
    
    # Check Drive backup
    if DRIVE_MODELS:
        drive_path = os.path.join(DRIVE_MODELS, "motionagformer", model_name)
        if os.path.exists(drive_path):
            print(f"  Copying from Drive: {display_name}")
            run_command(f"cp '{drive_path}' '{model_path}'")
            return
    
    # Download from Google Drive using gdown with --fuzzy flag
    gdrive_id = "1Iii5EwsFFm9_9lKBUPfN8bV5LmfkNUMP"
    
    if VERBOSE:
        print(f"  Downloading {display_name} (~200 MB)...")
        try:
            run_command(f"gdown --fuzzy {gdrive_id} -O '{model_path}'")
            print(f"  {COLOR_GREEN}✓{COLOR_RESET} Downloaded {display_name}")
        except Exception as e:
            print_warning(f"Failed to download {display_name}: {e}")
    else:
        cmd = f"gdown --fuzzy {gdrive_id} -O '{model_path}'"
        run_command_with_progress(cmd, model_name, model_path, 200)


def download_wb3d_models():
    """Download Wholebody 3D models"""
    print_step("2.5", "Download Wholebody 3D Models", indent=True)
    
    wb3d_dir = os.path.join(MODELS_DIR, "wb3d")
    model_path = os.path.join(wb3d_dir, "rtmw3d-l.onnx")
    
    if check_file_exists(model_path, quiet=True):
        print("  Skipping rtmw3d-l.onnx (already exists)")
        return
    
    # Check Drive backup (primary source - stored in rtmw3d_onnx_exports/)
    if is_colab_environment() and os.path.exists("/content/drive/MyDrive"):
        drive_path = "/content/drive/MyDrive/rtmw3d_onnx_exports/rtmw3d-l.onnx"
        if os.path.exists(drive_path):
            if not VERBOSE:
                print("  Copying from Drive: rtmw3d-l.onnx")
            else:
                print("  Copying from Drive: rtmw3d-l.onnx")
                print(f"  Running: cp '{drive_path}' '{model_path}'")
            
            # Run copy silently in non-verbose mode
            if VERBOSE:
                run_command(f"cp '{drive_path}' '{model_path}'")
            else:
                subprocess.run(f"cp '{drive_path}' '{model_path}'", shell=True, check=True, capture_output=True)
            
            if not VERBOSE:
                print("  ✓ Copied rtmw3d-l.onnx")
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
    """Download ReID models (both PyTorch and ONNX variants)"""
    print_step("2.6", "Download ReID Models", indent=True)
    
    reid_dir = os.path.join(MODELS_DIR, "reid")
    
    # Model 1: OSNet x1.0 (PyTorch) - for high accuracy
    model_name_pt = "osnet_x1_0_msmt17.pt"
    model_path_pt = os.path.join(reid_dir, model_name_pt)
    display_name_pt = get_model_display_name(model_name_pt)
    
    if check_file_exists(model_path_pt, quiet=True):
        file_size_bytes = os.path.getsize(model_path_pt)
        file_size_mb = file_size_bytes / (1024 * 1024)
        print(f"  {COLOR_GREEN}✓{COLOR_RESET} {display_name_pt} already exists: {model_path_pt} ({file_size_mb:.1f} MB)")
    else:
        print(f"  {COLOR_ORANGE}✗{COLOR_RESET} {display_name_pt} not found")
        
        # Check Drive backup
        if DRIVE_MODELS:
            drive_path = os.path.join(DRIVE_MODELS, "reid", model_name_pt)
            if os.path.exists(drive_path):
                print(f"  Copying from Drive: {display_name_pt}")
                run_command(f"cp '{drive_path}' '{model_path_pt}'")
            else:
                # Download from Google Drive
                gdrive_id = "1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY"
                if VERBOSE:
                    print(f"  Downloading {display_name_pt} (~25 MB)...")
                    try:
                        run_command(f"gdown --fuzzy {gdrive_id} -O '{model_path_pt}'")
                        print(f"  {COLOR_GREEN}✓{COLOR_RESET} Downloaded {display_name_pt}")
                    except Exception as e:
                        print_warning(f"Failed to download {display_name_pt}: {e}")
                else:
                    cmd = f"gdown --fuzzy {gdrive_id} -O '{model_path_pt}'"
                    run_command_with_progress(cmd, model_name_pt, model_path_pt, 25)
        else:
            # Download from Google Drive
            gdrive_id = "1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY"
            if VERBOSE:
                print(f"  Downloading {display_name_pt} (~25 MB)...")
                try:
                    run_command(f"gdown --fuzzy {gdrive_id} -O '{model_path_pt}'")
                    print(f"  {COLOR_GREEN}✓{COLOR_RESET} Downloaded {display_name_pt}")
                except Exception as e:
                    print_warning(f"Failed to download {display_name_pt}: {e}")
            else:
                cmd = f"gdown --fuzzy {gdrive_id} -O '{model_path_pt}'"
                run_command_with_progress(cmd, model_name_pt, model_path_pt, 25)
    
    # Model 2: OSNet x0.25 (PyTorch) - for speed (2x faster than x1.0)
    model_name_x025 = "osnet_x0_25_msmt17.pt"
    model_path_x025 = os.path.join(reid_dir, model_name_x025)
    display_name_x025 = get_model_display_name(model_name_x025)
    
    if check_file_exists(model_path_x025, quiet=True):
        file_size_bytes = os.path.getsize(model_path_x025)
        file_size_mb = file_size_bytes / (1024 * 1024)
        print(f"  {COLOR_GREEN}✓{COLOR_RESET} {display_name_x025} already exists: {model_path_x025} ({file_size_mb:.1f} MB)")
    else:
        print(f"  {COLOR_ORANGE}✗{COLOR_RESET} {display_name_x025} not found")
        
        # Check Drive backup
        if DRIVE_MODELS:
            drive_path = os.path.join(DRIVE_MODELS, "reid", model_name_x025)
            if os.path.exists(drive_path):
                print(f"  Copying from Drive: {display_name_x025}")
                run_command(f"cp '{drive_path}' '{model_path_x025}'")
            else:
                # Download from HuggingFace
                hf_url = "https://huggingface.co/paulosantiago/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.pt"
                if VERBOSE:
                    print(f"  Downloading {display_name_x025} (~2 MB)...")
                    try:
                        run_command(f"wget -O '{model_path_x025}' {hf_url}")
                        print(f"  {COLOR_GREEN}✓{COLOR_RESET} Downloaded {display_name_x025}")
                    except Exception as e:
                        print_warning(f"Failed to download {display_name_x025}: {e}")
                else:
                    cmd = f"wget -q -O '{model_path_x025}' {hf_url}"
                    run_command_with_progress(cmd, model_name_x025, model_path_x025, 2)
        else:
            # Download from HuggingFace
            hf_url = "https://huggingface.co/paulosantiago/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.pt"
            if VERBOSE:
                print(f"  Downloading {display_name_x025} (~2 MB)...")
                try:
                    run_command(f"wget -O '{model_path_x025}' {hf_url}")
                    print(f"  {COLOR_GREEN}✓{COLOR_RESET} Downloaded {display_name_x025}")
                except Exception as e:
                    print_warning(f"Failed to download {display_name_x025}: {e}")
            else:
                cmd = f"wget -q -O '{model_path_x025}' {hf_url}"
                run_command_with_progress(cmd, model_name_x025, model_path_x025, 2)
    
    # Model 3: OSNet x0.25 (ONNX) - kept for reference (has batch size issues)
    model_name_onnx = "osnet_x0_25_msmt17.onnx"
    model_path_onnx = os.path.join(reid_dir, model_name_onnx)
    display_name_onnx = get_model_display_name(model_name_onnx)
    
    if check_file_exists(model_path_onnx, quiet=True):
        file_size_bytes = os.path.getsize(model_path_onnx)
        file_size_mb = file_size_bytes / (1024 * 1024)
        print(f"  {COLOR_GREEN}✓{COLOR_RESET} {display_name_onnx} already exists: {model_path_onnx} ({file_size_mb:.1f} MB)")
    else:
        print(f"  {COLOR_ORANGE}✗{COLOR_RESET} {display_name_onnx} not found")
        
        # Check Drive backup
        if DRIVE_MODELS:
            drive_path = os.path.join(DRIVE_MODELS, "reid", model_name_onnx)
            if os.path.exists(drive_path):
                print(f"  Copying from Drive: {display_name_onnx}")
                run_command(f"cp '{drive_path}' '{model_path_onnx}'")
            else:
                # Download from HuggingFace
                hf_url = "https://huggingface.co/anriha/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.onnx"
                if VERBOSE:
                    print(f"  Downloading {display_name_onnx} (~2 MB)...")
                    try:
                        run_command(f"wget -O '{model_path_onnx}' {hf_url}")
                        print(f"  {COLOR_GREEN}✓{COLOR_RESET} Downloaded {display_name_onnx}")
                    except Exception as e:
                        print_warning(f"Failed to download {display_name_onnx}: {e}")
                else:
                    cmd = f"wget -q -O '{model_path_onnx}' {hf_url}"
                    run_command_with_progress(cmd, model_name_onnx, model_path_onnx, 2)
        else:
            # Download from HuggingFace
            hf_url = "https://huggingface.co/anriha/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.onnx"
            if VERBOSE:
                print(f"  Downloading {display_name_onnx} (~2 MB)...")
                try:
                    run_command(f"wget -O '{model_path_onnx}' {hf_url}")
                    print(f"  {COLOR_GREEN}✓{COLOR_RESET} Downloaded {display_name_onnx}")
                except Exception as e:
                    print_warning(f"Failed to download {display_name_onnx}: {e}")
            else:
                cmd = f"wget -q -O '{model_path_onnx}' {hf_url}"
                run_command_with_progress(cmd, model_name_onnx, model_path_onnx, 2)


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
