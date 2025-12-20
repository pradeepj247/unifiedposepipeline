"""
Download and setup demo data files for testing the unified pose pipeline
Handles both local and Google Colab environments
"""

import os
import sys
import subprocess
from pathlib import Path
import urllib.request
from typing import Optional

# ============================================
# Configuration
# ============================================
REPO_ROOT = Path(__file__).parent
DEMO_DATA_DIR = REPO_ROOT / "demo_data"

# Demo files to download/copy
DEMO_FILES = {
    "dance.mp4": {
        "description": "Dance video for testing",
        "colab_path": "/content/drive/MyDrive/HybrIK_TRT_Backups/demodata/dance.mp4",
        "url": None,  # Set to URL if available online
        "size_mb": "~50MB",
        "required": True,
    },
    "sample_image.jpg": {
        "description": "Sample image for testing",
        "url": "https://raw.githubusercontent.com/open-mmlab/mmpose/main/tests/data/coco/000000000785.jpg",
        "size_mb": "~0.1MB",
        "required": False,
    },
}

# Test video URLs (fallback if drive files not available)
FALLBACK_VIDEOS = {
    "sample_walk.mp4": "https://github.com/pradeepj247/unified_pose_pipeline/raw/main/test_videos/sample_walk.mp4",
    # Add more fallback URLs as needed
}


# ============================================
# Helper Functions
# ============================================
def print_header(message: str, char: str = "=", width: int = 70):
    """Print formatted header"""
    print("\n" + char * width)
    print(message)
    print(char * width)


def is_colab_environment() -> bool:
    """Check if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def mount_drive_if_needed():
    """Mount Google Drive if in Colab and not already mounted"""
    if not is_colab_environment():
        return False
    
    drive_path = Path("/content/drive")
    if drive_path.exists() and (drive_path / "MyDrive").exists():
        print("✓ Google Drive already mounted")
        return True
    
    try:
        from google.colab import drive
        print("📂 Mounting Google Drive...")
        drive.mount("/content/drive", force_remount=False)
        print("✅ Google Drive mounted")
        return True
    except Exception as e:
        print(f"⚠️  Could not mount Drive: {e}")
        return False


def download_file(url: str, output_path: Path, description: str = None) -> bool:
    """Download a file with progress"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 ** 2)
        print(f"   ✓ {output_path.name} already exists ({size_mb:.1f} MB)")
        return True
    
    desc = description or output_path.name
    print(f"   ⬇️  Downloading {desc}...")
    
    try:
        # Try wget first (with progress)
        wget_available = subprocess.call(
            "which wget", 
            shell=True, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        ) == 0
        
        if wget_available:
            cmd = f'wget -q --show-progress -O "{output_path}" "{url}"'
            print(f"   $ {cmd}")
            result = subprocess.call(cmd, shell=True)
            if result == 0:
                size_mb = output_path.stat().st_size / (1024 ** 2)
                print(f"   ✅ Downloaded {desc} ({size_mb:.1f} MB)")
                return True
        else:
            # Fallback to urllib
            urllib.request.urlretrieve(url, output_path)
            size_mb = output_path.stat().st_size / (1024 ** 2)
            print(f"   ✅ Downloaded {desc} ({size_mb:.1f} MB)")
            return True
            
    except Exception as e:
        print(f"   ❌ Download failed for {desc}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def copy_from_drive(src_path: Path, dst_path: Path, description: str) -> bool:
    """Copy file from Google Drive"""
    if not src_path.exists():
        print(f"   ⚠️  Source not found: {src_path}")
        return False
    
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dst_path.exists():
        size_mb = dst_path.stat().st_size / (1024 ** 2)
        print(f"   ✓ {dst_path.name} already exists ({size_mb:.1f} MB)")
        return True
    
    print(f"   📋 Copying {description} from Drive...")
    try:
        cmd = f'cp "{src_path}" "{dst_path}"'
        print(f"   $ {cmd}")
        result = subprocess.call(cmd, shell=True)
        if result == 0:
            size_mb = dst_path.stat().st_size / (1024 ** 2)
            print(f"   ✅ Copied {description} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"   ❌ Copy failed for {description}")
            return False
    except Exception as e:
        print(f"   ❌ Error copying {description}: {e}")
        return False


def create_demo_structure():
    """Create demo data directory structure"""
    print_header("Creating Demo Data Structure")
    
    # Create directories
    dirs_to_create = [
        DEMO_DATA_DIR,
        DEMO_DATA_DIR / "videos",
        DEMO_DATA_DIR / "images",
        DEMO_DATA_DIR / "outputs",
    ]
    
    for directory in dirs_to_create:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {directory.relative_to(REPO_ROOT)}/")
    
    print("✅ Demo structure created")


def setup_demo_files():
    """Download or copy demo files"""
    print_header("Setting Up Demo Files")
    
    is_colab = is_colab_environment()
    drive_mounted = False
    
    if is_colab:
        drive_mounted = mount_drive_if_needed()
        print()
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for filename, info in DEMO_FILES.items():
        print(f"\n📄 {filename}:")
        print(f"   Description: {info['description']}")
        print(f"   Size: {info['size_mb']}")
        
        # Determine output path based on file type
        if filename.endswith(('.mp4', '.avi', '.mov')):
            output_path = DEMO_DATA_DIR / "videos" / filename
        elif filename.endswith(('.jpg', '.jpeg', '.png')):
            output_path = DEMO_DATA_DIR / "images" / filename
        else:
            output_path = DEMO_DATA_DIR / filename
        
        # Try different sources
        success = False
        
        # 1. Try Google Drive (if in Colab and drive mounted)
        if is_colab and drive_mounted and info.get("colab_path"):
            colab_src = Path(info["colab_path"])
            if copy_from_drive(colab_src, output_path, filename):
                success = True
        
        # 2. Try direct download (if URL available)
        if not success and info.get("url"):
            if download_file(info["url"], output_path, filename):
                success = True
        
        # 3. Try fallback URLs
        if not success and filename in FALLBACK_VIDEOS:
            print(f"   🔄 Trying fallback URL...")
            if download_file(FALLBACK_VIDEOS[filename], output_path, filename):
                success = True
        
        # Count results
        if success:
            success_count += 1
        elif info.get("required"):
            failed_count += 1
            print(f"   ❌ REQUIRED file not available!")
        else:
            skipped_count += 1
            print(f"   ⚠️  Optional file not available (skipped)")
    
    # Summary
    print(f"\n{'=' * 70}")
    print(f"Demo Files Summary:")
    print(f"  ✅ Success: {success_count}")
    print(f"  ❌ Failed (required): {failed_count}")
    print(f"  ⚠️  Skipped (optional): {skipped_count}")
    print(f"{'=' * 70}")
    
    return failed_count == 0


def create_demo_readme():
    """Create README in demo_data directory"""
    readme_content = """# Demo Data Directory

This directory contains demo files for testing the unified pose estimation pipeline.

## Directory Structure

```
demo_data/
├── videos/          # Demo videos
├── images/          # Demo images
└── outputs/         # Generated outputs from demos
```

## Files

### Videos
- **dance.mp4**: Dance video for testing pose tracking and 3D estimation
  - Source: Google Drive backup (if available in Colab)
  - Fallback: Manual download required

### Images
- **sample_image.jpg**: Single image for testing pose detection
  - Automatically downloaded from COCO dataset

## Usage

These files are used by demo scripts in the `demos/` directory:

```bash
# Run ViTPose demo on image
python demos/demo_vitpose.py --image demo_data/images/sample_image.jpg

# Run RTMLib demo on video
python demos/demo_rtmlib.py --video demo_data/videos/dance.mp4
```

## Adding Your Own Files

Place your test files in the appropriate subdirectory:
- Videos: `demo_data/videos/`
- Images: `demo_data/images/`

Outputs will be saved to `demo_data/outputs/`
"""
    
    readme_path = DEMO_DATA_DIR / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n✅ Created {readme_path.relative_to(REPO_ROOT)}")


def verify_demo_data():
    """Verify demo data setup"""
    print_header("Verifying Demo Data")
    
    # Check for any video files
    video_files = list((DEMO_DATA_DIR / "videos").glob("*.*"))
    image_files = list((DEMO_DATA_DIR / "images").glob("*.*"))
    
    print(f"\n📹 Video files: {len(video_files)}")
    for vf in video_files:
        size_mb = vf.stat().st_size / (1024 ** 2)
        print(f"   • {vf.name} ({size_mb:.1f} MB)")
    
    print(f"\n🖼️  Image files: {len(image_files)}")
    for img in image_files:
        size_kb = img.stat().st_size / 1024
        print(f"   • {img.name} ({size_kb:.1f} KB)")
    
    if len(video_files) == 0 and len(image_files) == 0:
        print("\n⚠️  No demo files found!")
        print("\n💡 Manual setup instructions:")
        print("   1. Place video files in: demo_data/videos/")
        print("   2. Place image files in: demo_data/images/")
        print("   3. Recommended: dance.mp4 (dance sequence for testing)")
        return False
    
    print("\n✅ Demo data verification complete")
    return True


# ============================================
# Main Execution
# ============================================
def main():
    """Main setup function"""
    print("\n" + "📂" * 35)
    print("UNIFIED POSE PIPELINE - DEMO DATA SETUP")
    print("📂" * 35)
    print(f"\nSetup directory: {REPO_ROOT.absolute()}\n")
    
    try:
        # Create structure
        create_demo_structure()
        
        # Setup files
        success = setup_demo_files()
        
        # Create README
        create_demo_readme()
        
        # Verify
        verify_demo_data()
        
        # Final message
        print("\n" + "=" * 70)
        if success:
            print("🎉 Demo data setup complete!")
            print("\n💡 Demo files are ready in: demo_data/")
            print("   • Videos: demo_data/videos/")
            print("   • Images: demo_data/images/")
            print("   • Outputs will be saved to: demo_data/outputs/")
        else:
            print("⚠️  Demo data setup completed with warnings")
            print("\n💡 Some files may need manual setup:")
            print("   • Check demo_data/README.md for instructions")
            print("   • Place your own test files in the directories")
        print("=" * 70 + "\n")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
