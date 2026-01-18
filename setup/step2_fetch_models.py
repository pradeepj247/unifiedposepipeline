#!/usr/bin/env python3
"""
Step 2: Fetch Model Files (YAML-driven)

This script downloads all required model weights from sources defined in models.yaml.
The YAML file in this directory controls all model sources, destinations, and fetch methods.
Supports automatic fallback between GitHub and Google Drive sources.

Usage:
    python step2_fetch_models.py
"""

import os
import sys
import subprocess
import time
import yaml
from pathlib import Path
from setup_utils import (
    is_colab_environment, print_header, print_step, run_command,
    check_file_exists, print_success, print_error, print_warning, COLOR_YELLOW
)


# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "models.yaml")
MODELS_DIR = "/content/models"

# Display settings (can be modified in YAML or here)
VERBOSE = False  # Set to True for detailed output

# Color codes
COLOR_GREEN = "\033[92m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"


def load_model_config():
    """Load model configuration from YAML file"""
    if not os.path.exists(CONFIG_FILE):
        print_error(f"Config file not found: {CONFIG_FILE}")
        sys.exit(1)
    
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_model_directories(models, base_dir):
    """Create all required model directories"""
    dirs = set()
    for model in models:
        subfolder = model['subfolder']
        full_path = os.path.join(base_dir, subfolder)
        dirs.add(full_path)
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    print(f"  🛠️ Creating directory structure for models")
    print()

def download_from_github(url, destination, size_mb, model_name):
    """Download file from GitHub using curl"""
    cmd = f"curl -L '{url}' -o '{destination}'"
    
    if VERBOSE:
        print(f"     ⚡ Downloading {model_name} from GitHub (~{size_mb} MB)")
        try:
            run_command(cmd)
            print(f"     {COLOR_GREEN}✔️{COLOR_RESET} Downloaded {model_name}")
            print(f"     📡 Source: GitHub (curl)")
            return True
        except Exception as e:
            print_warning(f"GitHub download failed: {e}")
            return False
    else:
        # Silent mode
        print(f"     ⚡ Downloading {model_name} (~{size_mb} MB)")
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True
            )
            print(f"     {COLOR_GREEN}✔️{COLOR_RESET} Downloaded {model_name}")
            print(f"     📡 Source: GitHub (curl)")
            return True
        except subprocess.CalledProcessError as e:
            print_warning(f"GitHub download failed")
            return False


def copy_from_drive(drive_path, destination, model_name):
    """Copy file from Google Drive"""
    if not os.path.exists(drive_path):
        if VERBOSE:
            print_warning(f"Drive file not found: {drive_path}")
        return False
    
    cmd = f"cp '{drive_path}' '{destination}'"
    
    try:
        if VERBOSE:
            print(f"     ⚡ Copying {model_name} from Drive")
            run_command(cmd)
        else:
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
        
        print(f"     {COLOR_GREEN}✔️{COLOR_RESET} Copied {model_name}")
        print(f"     📁 Source: Drive (copy)")
        return True
    except Exception as e:
        if VERBOSE:
            print_warning(f"Drive copy failed: {e}")
        return False


def fetch_model(model, preferred_source, base_dir):
    """Fetch a single model using configured sources"""
    name = model['name']
    filename = model['filename']
    subfolder = model['subfolder']
    destination = os.path.join(base_dir, subfolder, filename)
    source_url = model['source_url']
    fallback_location = model['fallback_location']
    size_mb = model['size_mb']
    
    # Check if already exists
    # Check if already exists
    if check_file_exists(destination, quiet=True):
        file_size_bytes = os.path.getsize(destination)
        file_size_mb = file_size_bytes / (1024 * 1024)
        print(f"  ✅  {name} already exists: {destination} ({file_size_mb:.1f} MB)")
        print("  " + "─" * 65 + "\n")
        return True
    
    print(f"  ⚠️ {name} not found")
    
    # Start timing
    start = time.time()
    
    success = False
    
    # Try preferred source first
    if preferred_source == "drive":
        if VERBOSE:
            print(f"     Trying Drive first (preferred)")
        success = copy_from_drive(fallback_location, destination, name)
        
        if not success:
            if VERBOSE:
                print(f"     Falling back to GitHub")
            success = download_from_github(source_url, destination, size_mb, name)
    else:  # preferred_source == "github" (default)
        if VERBOSE:
            print(f"     Trying GitHub first (preferred)")
        success = download_from_github(source_url, destination, size_mb, name)
        
        if not success:
            if VERBOSE:
                print(f"     Falling back to Drive")
            success = copy_from_drive(fallback_location, destination, name)
    if success:
        elapsed = time.time() - start
        print(f"     ⏱️ Time taken: {elapsed:.2f}s")
        print("  " + "─" * 65 + "\n")
        return True
    else:
        print_error(f"Failed to fetch {name} from both sources")
        print("  " + "─" * 65 + "\n")
        return False


def main():
    """Main execution function"""
    
    # Load configuration from YAML file in same directory
    config = load_model_config()
    models = config['models']
    preferred_source = config.get('preferred_fetch_location', 'github')
    base_dir = config.get('destination_folder', MODELS_DIR)
    
    # Start timer
    start_time = time.time()
    
    # Print header
    print("\n" + "=" * 70)
    print(f"{COLOR_YELLOW}🚀 STEP 2: Fetch Model Files{COLOR_RESET}")
    print("=" * 70 + "\n")
    
    if not VERBOSE:
        print("  💡 Running in silent mode. Set VERBOSE=True in script for detailed output.\n")
    
    print(f"  📋 Loading configuration from: models.yaml")
    print(f"  🎯 Preferred source: {preferred_source.upper()}")
    print(f"  📂 Destination folder: {base_dir}")
    print(f"  📦 Total models to fetch: {len(models)}")
    print("  " + "─" * 65 + "\n")
    
    # Create directories
    create_model_directories(models, base_dir)
    print("  " + "─" * 65 + "\n")
    
    # Fetch each model
    success_count = 0
    failed_models = []
    skipped_models = []
    
    try:
        for i, model in enumerate(models, 1):
            # Check if fetch is disabled (handle both boolean and string values)
            fetch_value = model.get('fetch', 'yes')
            if isinstance(fetch_value, bool):
                fetch_enabled = fetch_value
            else:
                fetch_enabled = str(fetch_value).lower() in ['yes', 'true', '1']
            
            if not fetch_enabled:
                print(f"  ⏭️  [{i}/{len(models)}] Skipping: {model['name']} (fetch=no)")
                skipped_models.append(model['name'])
                continue
            
            print(f"  ⬇️  [{i}/{len(models)}] Fetching: {model['name']}")
            
            if fetch_model(model, preferred_source, base_dir):
                success_count += 1
            else:
                failed_models.append(model['name'])
        
        # Summary
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        
        if skipped_models:
            print(f"{COLOR_BLUE}⏭️  SKIPPED: {len(skipped_models)} model(s) (fetch=no){COLOR_RESET}")
            for model_name in skipped_models:
                print(f"  ⏭️  {model_name}")
            print()
        
        if failed_models:
            print(f"{COLOR_YELLOW}⚠️  PARTIAL SUCCESS: {success_count}/{len(models) - len(skipped_models)} models fetched{COLOR_RESET}")
            print(f"\nFailed models:")
            for model_name in failed_models:
                print(f"  ✗ {model_name}")
        else:
            fetched_count = len(models) - len(skipped_models)
            if fetched_count > 0:
                print(f"{COLOR_GREEN}✅  SUCCESS: All {fetched_count} enabled models fetched!{COLOR_RESET}")
            else:
                print(f"{COLOR_YELLOW}⚠️  No models fetched (all marked fetch=no){COLOR_RESET}")
        
        print(f"⏱️  TOTAL TIME TAKEN: {total_time:.2f}s")
        print("=" * 70 + "\n")
        
        # Copy optimized YOLO TensorRT engines from Drive backup (if available)
        yolo_dest = os.path.join(base_dir, "yolo")
        drive_backup_dir = "/content/drive/MyDrive/yolomodels/new"
        
        # Try to copy yolov8n.engine first (best: 121.7 FPS), then yolov8s_b4_640.engine (fallback: 113.8 FPS)
        engine_files = [
            ("yolov8n.engine", "121.7 FPS, 45% faster than PyTorch"),
            ("yolov8s_b4_640.engine", "113.8 FPS, 40% faster than PyTorch (batch=4)")
        ]
        
        if os.path.exists(drive_backup_dir) and os.path.exists(yolo_dest):
            copied_any = False
            
            for engine_name, description in engine_files:
                source_file = os.path.join(drive_backup_dir, engine_name)
                
                if os.path.exists(source_file):
                    if not copied_any:
                        print("=" * 70)
                        print(f"{COLOR_BLUE}📦 Copying optimized YOLO TensorRT engine from Drive backup{COLOR_RESET}")
                        print("=" * 70 + "\n")
                    
                    dest_file = os.path.join(yolo_dest, engine_name)
                    copy_cmd = f"cp '{source_file}' '{dest_file}'"
                    
                    try:
                        subprocess.run(copy_cmd, shell=True, check=True, capture_output=True)
                        file_size_mb = os.path.getsize(dest_file) / (1024 * 1024)
                        
                        print(f"  ✅ Copied {engine_name} ({file_size_mb:.1f} MB)")
                        print(f"     💡 Performance: {description}")
                        copied_any = True
                        
                        # If we got yolov8n.engine (best), don't need fallback
                        if engine_name == "yolov8n.engine":
                            break
                            
                    except subprocess.CalledProcessError as e:
                        print_warning(f"Could not copy {engine_name}: {e}")
            
            if copied_any:
                print(f"\n  {COLOR_GREEN}✔️ Optimized YOLO model restored!{COLOR_RESET}")
                print("=" * 70 + "\n")
            else:
                print_warning(f"No TensorRT engines found in backup: {drive_backup_dir}")
                print("  (Not critical - can be re-exported later with det_track/debug/export_yolo_tensorrt.py)\n")
                
        elif os.path.exists(drive_backup_dir):
            print_warning(f"Drive backup found but destination not created: {yolo_dest}")
        
        if not failed_models:
            print("📌 Next steps to try:")
            print("    python step3_pull_demodata.py    # Setup demo data")
            print("    python step4_verify_envt.py      # Verify installation")
        
    except KeyboardInterrupt:
        print("\n\n⊘ Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
