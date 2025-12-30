#!/usr/bin/env python3
"""
Verify model files on Google Drive match those in /content/models

This script compares file sizes between the local models directory and
Google Drive to ensure all models were copied correctly.

Usage:
    python verify_drive_models.py
"""

import os
import sys

# Source and destination paths
SOURCE_DIR = "/content/models"
DRIVE_DIR = "/content/drive/MyDrive/pipelinemodels"

# Subdirectories to check
SUBDIRS = ["motionagformer", "reid", "rtmlib", "vitpose", "wb3d", "yolo"]

# Color codes
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"


def print_success(msg):
    print(f"{COLOR_GREEN}✓{COLOR_RESET} {msg}")


def print_info(msg):
    print(f"{COLOR_BLUE}ℹ{COLOR_RESET} {msg}")


def print_error(msg):
    print(f"{COLOR_RED}✗{COLOR_RESET} {msg}")


def print_warning(msg):
    print(f"{COLOR_YELLOW}⚠{COLOR_RESET} {msg}")


def format_size(size_bytes):
    """Format bytes to MB with 1 decimal place"""
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def verify_models():
    """Compare file sizes between source and Drive"""
    print_info("Verifying model files...\n")
    
    all_match = True
    total_files_checked = 0
    mismatches = []
    missing_files = []
    
    for subdir in SUBDIRS:
        source_subdir = os.path.join(SOURCE_DIR, subdir)
        drive_subdir = os.path.join(DRIVE_DIR, subdir)
        
        if not os.path.exists(source_subdir):
            print_warning(f"Source directory not found: {source_subdir}")
            continue
        
        if not os.path.exists(drive_subdir):
            print_error(f"Drive directory not found: {drive_subdir}")
            all_match = False
            continue
        
        # Get files from source
        source_files = [f for f in os.listdir(source_subdir) if os.path.isfile(os.path.join(source_subdir, f))]
        
        if not source_files:
            continue
        
        print(f"📁 Checking {subdir}/:")
        
        for filename in source_files:
            source_file = os.path.join(source_subdir, filename)
            drive_file = os.path.join(drive_subdir, filename)
            
            # Check if file exists on Drive
            if not os.path.exists(drive_file):
                print_error(f"   Missing on Drive: {filename}")
                missing_files.append(f"{subdir}/{filename}")
                all_match = False
                continue
            
            # Compare file sizes
            source_size = os.path.getsize(source_file)
            drive_size = os.path.getsize(drive_file)
            
            total_files_checked += 1
            
            if source_size == drive_size:
                print_success(f"   {filename}: {format_size(source_size)}")
            else:
                print_error(f"   {filename}: SIZE MISMATCH!")
                print(f"      Source: {format_size(source_size)}")
                print(f"      Drive:  {format_size(drive_size)}")
                mismatches.append(f"{subdir}/{filename}")
                all_match = False
        
        print()
    
    # Summary
    print("=" * 70)
    if all_match and total_files_checked > 0:
        print_success(f"Verification complete! All {total_files_checked} files match.")
        print("=" * 70)
        return True
    else:
        print_error("Verification failed!")
        if missing_files:
            print(f"\n{COLOR_RED}Missing files ({len(missing_files)}):{COLOR_RESET}")
            for f in missing_files:
                print(f"  - {f}")
        if mismatches:
            print(f"\n{COLOR_RED}Size mismatches ({len(mismatches)}):{COLOR_RESET}")
            for f in mismatches:
                print(f"  - {f}")
        print("=" * 70)
        return False


def main():
    print("\n" + "=" * 70)
    print(f"{COLOR_YELLOW}🔍 Verify Models on Google Drive{COLOR_RESET}")
    print("=" * 70 + "\n")
    
    # Check if source directory exists
    if not os.path.exists(SOURCE_DIR):
        print_error(f"Source directory not found: {SOURCE_DIR}")
        sys.exit(1)
    
    # Check if Drive directory exists
    if not os.path.exists(DRIVE_DIR):
        print_error(f"Drive directory not found: {DRIVE_DIR}")
        print("Please run copy_models_to_drive.py first.")
        sys.exit(1)
    
    try:
        success = verify_models()
        
        if success:
            print(f"\n{COLOR_GREEN}✅ All models verified successfully!{COLOR_RESET}\n")
            sys.exit(0)
        else:
            print(f"\n{COLOR_RED}❌ Verification failed. Please check the errors above.{COLOR_RESET}\n")
            sys.exit(1)
            
    except Exception as e:
        print_error(f"Error during verification: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
