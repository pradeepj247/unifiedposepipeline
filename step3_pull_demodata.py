#!/usr/bin/env python3
"""
Step 3: Pull Demo Data

This script sets up demo videos and images for testing the pipeline.
Corresponds to Step 9 in the original setup_unified.py.

Usage:
    python step3_pull_demodata.py
"""

import os
import sys
from setup_utils import (
    is_colab_environment, print_header, print_step, run_command,
    check_file_exists, print_success, print_error, print_warning, COLOR_YELLOW
)


# Configuration
REPO_ROOT = "/content/unifiedposepipeline" if is_colab_environment() else os.getcwd()
DEMO_DATA_DIR = os.path.join(REPO_ROOT, "demo_data")
DRIVE_ROOT = "/content/drive/MyDrive" if is_colab_environment() else None


def setup_demo_videos():
    """Copy demo videos from Google Drive"""
    print_step("3.1", "Setup Demo Videos", indent=True)
    
    videos_dir = os.path.join(DEMO_DATA_DIR, "videos")
    
    # List of video files to copy
    video_files = [
        "campus_walk.mp4",
        "dance.mp4",
        "kohli_nets.mp4",
        "practice1.mp4",
        "practice2.mp4"
    ]
    
    # Source directory in Google Drive
    drive_videos_dir = os.path.join(DRIVE_ROOT, "samplevideos") if DRIVE_ROOT else None
    
    # Count files
    existing_count = 0
    copied_count = 0
    missing_count = 0
    
    if not drive_videos_dir:
        print_warning("Drive not mounted, cannot copy videos")
        print(f"  Please manually download and place in: {videos_dir}")
        return
    
    print("  Fetching demo videos from Google Drive backup...")
    
    for video_name in video_files:
        local_path = os.path.join(videos_dir, video_name)
        
        if check_file_exists(local_path):
            existing_count += 1
            continue
        
        # Check Drive
        drive_path = os.path.join(drive_videos_dir, video_name)
        if os.path.exists(drive_path):
            try:
                run_command(f"cp '{drive_path}' '{local_path}'")
                copied_count += 1
            except Exception as e:
                print_warning(f"Failed to copy {video_name}: {e}")
                missing_count += 1
        else:
            missing_count += 1
    
    # Print summary
    if copied_count > 0:
        print(f"  ✓ Copied: {copied_count} video(s)")
    if existing_count > 0:
        print(f"  ✓ Already present: {existing_count} video(s)")
    if missing_count > 0:
        print_warning(f"{missing_count} video(s) not found in Drive")
        print(f"  Please manually place videos in: {videos_dir}")


def setup_demo_images():
    """Download demo images"""
    print_step("3.2", "Setup Demo Images", indent=True)
    
    images_dir = os.path.join(DEMO_DATA_DIR, "images")
    image_path = os.path.join(images_dir, "sample.jpg")
    
    if check_file_exists(image_path):
        print("  ✓ Already present: sample.jpg")
        return
    
    # Download a sample image from internet
    print("  Fetching demo image from internet...")
    url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"
    
    try:
        run_command(f"curl -L '{url}' -o '{image_path}'")
        print("  ✓ Downloaded sample.jpg")
    except Exception as e:
        print_warning(f"Failed to download sample image: {e}")
        print("  You can manually place any image in:")
        print(f"    {image_path}")


def main():
    """Main execution function"""
    print_header("STEP 3: Pull Demo Data", color=COLOR_YELLOW)
    
    print("This script will setup demo videos and images.")
    print(f"Demo data directory: {DEMO_DATA_DIR}")
    if DRIVE_ROOT:
        print(f"Drive root: {DRIVE_ROOT}")
    print()
    
    try:
        setup_demo_videos()
        setup_demo_images()
        
        print_success("Demo data setup complete!", color=COLOR_YELLOW)
        print("\nNext steps:")
        print("  python step4_verify_envt.py      # Verify installation")
        print("\nOr start using the pipeline:")
        print("  python udp_video.py --config configs/udp_video.yaml")
        
    except KeyboardInterrupt:
        print("\n\n⊘ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
