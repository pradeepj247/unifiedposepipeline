#!/usr/bin/env python3
"""
Step 3: Pull Demo Data

This script downloads demo videos and images from GitHub releases.
Configuration is loaded from demodata.yaml for maintainability.

Usage:
    python step3_fetch_demodata.py
"""

import os
import sys
import time
import yaml
import subprocess
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


def download_and_extract_demo_group(group, base_url, base_dest):
    """Download and extract demo files from GitHub releases"""
    name = group['name']
    subfolder = group['subfolder']
    zip_filename = group['zip_filename']
    
    # Build paths
    dest_folder = os.path.join(base_dest, subfolder)
    zip_url = f"{base_url}/{zip_filename}"
    temp_zip = os.path.join("/tmp", zip_filename)
    
    print_header(f"Fetching {name}")
    
    # Create destination if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)
    
    # Check if files already exist
    try:
        existing_files = [f for f in os.listdir(dest_folder) 
                         if os.path.isfile(os.path.join(dest_folder, f))]
        if existing_files:
            print_success(f"Already present: {len(existing_files)} file(s)")
            return 0, len(existing_files)
    except:
        pass
    
    # Download zip file
    print(f"  ⬇️ Downloading from: {zip_url}")
    try:
        subprocess.run(
            f"curl -sL '{zip_url}' -o '{temp_zip}'",
            shell=True,
            check=True,
            capture_output=True
        )
    except Exception as e:
        print_error(f"Failed to download: {e}")
        return 0, 0
    
    # Extract zip file to base_dest (parent folder)
    # This avoids nested folder issue since zip already contains subfolder
    print(f"  📦 Extracting to: {dest_folder}")
    try:
        subprocess.run(
            f"unzip -q -o '{temp_zip}' -d '{base_dest}'",
            shell=True,
            check=True,
            capture_output=True
        )
    except Exception as e:
        print_error(f"Failed to extract: {e}")
        # Clean up zip file
        if os.path.exists(temp_zip):
            os.remove(temp_zip)
        return 0, 0
    
    # Clean up zip file
    if os.path.exists(temp_zip):
        os.remove(temp_zip)
    
    # Count extracted files in the correct location
    try:
        extracted_files = [f for f in os.listdir(dest_folder) 
                          if os.path.isfile(os.path.join(dest_folder, f))]
        file_count = len(extracted_files)
    except:
        file_count = 0
    
    print_success(f"Extracted {file_count} file(s)")
    
    return file_count, 0


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
        
        base_url = config['global_settings']['github_release_url']
        base_dest = config['global_settings']['destination_folder']
        print(f"   📁 Demo data folder: {base_dest}")
        
        # Process demo groups
        demo_groups = config.get('demo_groups', [])
        total_downloaded = 0
        total_existing = 0
        
        for group in demo_groups:
            downloaded, existing = download_and_extract_demo_group(group, base_url, base_dest)
            total_downloaded += downloaded
            total_existing += existing
        
        # Final success message
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print()
        print(f"\033[93m✅ SUCCESS: Demo data setup complete!\033[0m")
        print(f"📊 Total files downloaded: {total_downloaded}")
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
