#!/usr/bin/env python3
"""
Copy all models from /content/models to Google Drive

This script copies all downloaded models to a centralized location in Google Drive
at /content/drive/MyDrive/pipelinemodels/ with the same folder structure.

Usage:
    python copy_models_to_drive.py
"""

import os
import shutil
import sys

# Source and destination paths
SOURCE_DIR = "/content/models"
DRIVE_DIR = "/content/drive/MyDrive/pipelinemodels"

# Subdirectories to create and copy
SUBDIRS = ["motionagformer", "reid", "rtmlib", "vitpose", "wb3d", "yolo"]

# Color codes
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"
COLOR_RESET = "\033[0m"


def print_success(msg):
    print(f"{COLOR_GREEN}✓{COLOR_RESET} {msg}")


def print_info(msg):
    print(f"{COLOR_YELLOW}ℹ{COLOR_RESET} {msg}")


def print_error(msg):
    print(f"{COLOR_RED}✗{COLOR_RESET} {msg}")


def create_drive_structure():
    """Create the pipelinemodels directory structure in Google Drive"""
    print_info(f"Creating directory structure in Google Drive...")
    
    # Create main directory
    os.makedirs(DRIVE_DIR, exist_ok=True)
    print_success(f"Created: {DRIVE_DIR}")
    
    # Create subdirectories
    for subdir in SUBDIRS:
        subdir_path = os.path.join(DRIVE_DIR, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        print_success(f"Created: {subdir_path}")
    
    print()


def copy_models():
    """Copy all model files from /content/models to Google Drive"""
    print_info("Copying model files to Google Drive...\n")
    
    total_files = 0
    total_size = 0
    
    for subdir in SUBDIRS:
        source_subdir = os.path.join(SOURCE_DIR, subdir)
        dest_subdir = os.path.join(DRIVE_DIR, subdir)
        
        if not os.path.exists(source_subdir):
            print_error(f"Source directory not found: {source_subdir}")
            continue
        
        # List all files in source subdirectory
        files = [f for f in os.listdir(source_subdir) if os.path.isfile(os.path.join(source_subdir, f))]
        
        if not files:
            print_info(f"No files in {subdir}/")
            continue
        
        print(f"📁 Copying {subdir}/ files:")
        for filename in files:
            source_file = os.path.join(source_subdir, filename)
            dest_file = os.path.join(dest_subdir, filename)
            
            # Get file size
            file_size = os.path.getsize(source_file)
            file_size_mb = file_size / (1024 * 1024)
            
            # Copy file
            print(f"   ⚡ Copying {filename} ({file_size_mb:.1f} MB)...")
            shutil.copy2(source_file, dest_file)
            print_success(f"   Copied to {dest_file}")
            
            total_files += 1
            total_size += file_size
        
        print()
    
    # Summary
    total_size_mb = total_size / (1024 * 1024)
    print("=" * 70)
    print_success(f"Copy complete! {total_files} files copied ({total_size_mb:.1f} MB total)")
    print("=" * 70)


def main():
    print("\n" + "=" * 70)
    print(f"{COLOR_YELLOW}🚀 Copy Models to Google Drive{COLOR_RESET}")
    print("=" * 70 + "\n")
    
    # Check if source directory exists
    if not os.path.exists(SOURCE_DIR):
        print_error(f"Source directory not found: {SOURCE_DIR}")
        print("Please ensure models are downloaded first using step2_install_models.py")
        sys.exit(1)
    
    # Check if Google Drive is mounted
    if not os.path.exists("/content/drive/MyDrive"):
        print_error("Google Drive not mounted!")
        print("Please mount Google Drive first:")
        print("  from google.colab import drive")
        print("  drive.mount('/content/drive')")
        sys.exit(1)
    
    try:
        # Create directory structure
        create_drive_structure()
        
        # Copy all model files
        copy_models()
        
        print(f"\n{COLOR_GREEN}✅ All models successfully copied to Google Drive!{COLOR_RESET}")
        print(f"📂 Location: {DRIVE_DIR}\n")
        
    except Exception as e:
        print_error(f"Error during copy: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
