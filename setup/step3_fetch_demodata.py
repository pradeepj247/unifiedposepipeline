#!/usr/bin/env python3
"""
Step 3: Pull Demo Data

This script copies demo videos and images from Google Drive for testing.
Configuration is loaded from demodata.yaml for maintainability.

Usage:
    python step3_pull_demodata.py
"""

import os
import sys
import time
import yaml
import shutil
from pathlib import Path


def is_colab_environment():
    """Check if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def print_header(text, emoji="🛠️"):
    """Print a formatted section header"""
    print("\n  " + "─" * 66)
    print(f"  {emoji} {text}")
    print("  " + "─" * 66 + "\n")


def print_error(text):
    """Print error message in red"""
    print(f"\033[91m✗ {text}\033[0m")


def print_success(text):
    """Print success message"""
    print(f"  ✅ {text}")


def print_warning(text):
    """Print warning message"""
    print(f"  ⚠️ {text}")


def load_demodata_config():
    """Load demo data configuration from YAML file"""
    config_path = Path(__file__).parent / "demodata.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def copy_demo_group(group, base_source, base_dest):
    """Copy all files from a demo group source folder to destination"""
    name = group['name']
    subfolder = group['subfolder']
    description = group.get('description', '')
    
    # Construct full source and destination paths
    source_folder = os.path.join(base_source, subfolder)
    
    print_header(f"Fetching {name} from: {source_folder}")
    
    # Build destination path
    dest_folder = os.path.join(base_dest, subfolder)
    
    # Create destination if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)
    
    # Check if source exists
    if not os.path.exists(source_folder):
        print_warning(f"Source folder not found: {source_folder}")
        print(f"  Please ensure Google Drive is mounted and folder exists")
        return 0, 0
    
    # Get list of files in source
    try:
        source_files = [f for f in os.listdir(source_folder) 
                       if os.path.isfile(os.path.join(source_folder, f))]
    except Exception as e:
        print_error(f"Failed to read source folder: {e}")
        return 0, 0
    
    if not source_files:
        print_warning(f"No files found in {source_folder}")
        return 0, 0
    
    # Copy files
    copied_count = 0
    existing_count = 0
    
    for filename in source_files:
        source_path = os.path.join(source_folder, filename)
        dest_path = os.path.join(dest_folder, filename)
        
        if os.path.exists(dest_path):
            existing_count += 1
            continue
        
        try:
            print(f"  ⬇️ Copying: {filename}")
            shutil.copy2(source_path, dest_path)
            copied_count += 1
        except Exception as e:
            print_warning(f"Failed to copy {filename}: {e}")
    
    # Print summary
    print()
    if copied_count > 0:
        print_success(f"Copied {copied_count} file(s)")
    if existing_count > 0:
        print_success(f"Already present: {existing_count} file(s)")
    
    return copied_count, existing_count


def main():
    """Main execution function"""
    start_time = time.time()
    
    # Configuration
    repo_root = "/content/unifiedposepipeline" if is_colab_environment() else os.getcwd()
    
    # Top header
    print("\n" + "=" * 70)
    print(f"\033[93m🚀 STEP 3: Pull Demo Data\033[0m")
    print("=" * 70 + "\n")
    
    try:
        # Load configuration
        config = load_demodata_config()
        print("   ✅ Loaded configuration from demodata.yaml")
        
        base_source = config['global_settings']['source_folder']
        base_dest = config['global_settings']['destination_folder']
        print(f"   📁 Demo data folder: {base_dest}")
        
        # Check if Drive is mounted (Colab only)
        if is_colab_environment():
            if not os.path.exists('/content/drive/MyDrive'):
                print_warning("Google Drive is not mounted!")
                print("   Please mount Drive first by running:")
                print("   from google.colab import drive")
                print("   drive.mount('/content/drive')")
                sys.exit(1)
            print("   ✅ Google Drive is mounted")
        
        # Process demo groups
        demo_groups = config.get('demo_groups', [])
        total_copied = 0
        total_existing = 0
        
        for group in demo_groups:
            copied, existing = copy_demo_group(group, base_source, base_dest)
            total_copied += copied
            total_existing += existing
        
        # Final success message
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print()
        print(f"\033[93m✅ SUCCESS: Demo data setup complete!\033[0m")
        print(f"📊 Total files copied: {total_copied}")
        print(f"📊 Total files already present: {total_existing}")
        print(f"⏱️ TOTAL TIME TAKEN: {total_time:.2f}s")
        print("=" * 70 + "\n")
        print()
        print("💡 Next steps to try:")
        print("    ✓ python step4_verify_envt.py     # Verify installation")
        print()
        print("   Or start using the pipeline:")
        print("    ✓ python udp_video.py --config configs/udp_video.yaml")
        
    except KeyboardInterrupt:
        print("\n\n⊘ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
