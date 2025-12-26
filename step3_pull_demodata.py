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
    check_file_exists, print_success, print_error, print_warning
)


# Configuration
REPO_ROOT = "/content/unifiedposepipeline" if is_colab_environment() else os.getcwd()
DEMO_DATA_DIR = os.path.join(REPO_ROOT, "demo_data")
DRIVE_ROOT = "/content/drive/MyDrive" if is_colab_environment() else None


def setup_demo_videos():
    """Copy demo videos from Google Drive"""
    print_step("9.1", "Setup Demo Videos")
    
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
    
    for video_name in video_files:
        local_path = os.path.join(videos_dir, video_name)
        
        if check_file_exists(local_path):
            print(f"  Skipping {video_name} (already exists)")
            continue
        
        # Check Drive
        if drive_videos_dir:
            drive_path = os.path.join(drive_videos_dir, video_name)
            if os.path.exists(drive_path):
                print(f"  Copying from Drive: {video_name}")
                try:
                    run_command(f"cp '{drive_path}' '{local_path}'")
                    print(f"  ✓ Copied {video_name}")
                except Exception as e:
                    print_warning(f"Failed to copy {video_name}: {e}")
            else:
                print_warning(f"{video_name} not found in Drive")
                print(f"  Expected location: {drive_path}")
                print(f"  Please manually place video in: {local_path}")
        else:
            print_warning("Drive not mounted, cannot copy videos")
            print(f"  Please manually download and place in: {videos_dir}")
            break  # No need to repeat warning for each file


def setup_demo_images():
    """Download demo images"""
    print_step("9.2", "Setup Demo Images")
    
    images_dir = os.path.join(DEMO_DATA_DIR, "images")
    image_path = os.path.join(images_dir, "sample.jpg")
    
    if check_file_exists(image_path):
        print("  Skipping sample.jpg (already exists)")
        return
    
    # Download a sample image from internet
    print("  Downloading sample image...")
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
    print_header("STEP 3: Pull Demo Data")
    
    print("This script will setup demo videos and images.")
    print(f"Demo data directory: {DEMO_DATA_DIR}")
    if DRIVE_ROOT:
        print(f"Drive root: {DRIVE_ROOT}")
    print()
    
    try:
        setup_demo_videos()
        setup_demo_images()
        
        print_success("Demo data setup complete!")
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
