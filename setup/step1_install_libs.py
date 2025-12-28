#!/usr/bin/env python3
"""
Step 1: Install Libraries and Dependencies

This script installs all required Python packages and creates necessary directories.
Configuration is loaded from libraries.yaml for maintainability.

Usage:
    python step1_install_libs.py
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


def print_header(text, emoji="💡"):
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


def print_info(text):
    """Print info message"""
    print(f"  🔍 {text}")


def run_command(cmd, message=None, allow_failure=False):
    """Execute a shell command with optional message"""
    if message:
        print()
        print(message)
    
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0 and not allow_failure:
        raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
    
    return result


def load_library_config():
    """Load library configuration from YAML file"""
    config_path = Path(__file__).parent / "libraries.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def is_gpu_available():
    """Check if CUDA GPU is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


def mount_google_drive():
    """Mount Google Drive (Colab only)"""
    print_header("STEP 1.0: Mount Google Drive")
    
    if not is_colab_environment():
        print("      ⊘ Skipping (not in Colab environment)")
        return
    
    # Check if Drive is already mounted
    if os.path.exists('/content/drive/MyDrive'):
        print_success("Google Drive already mounted")
        return
    
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print_success("Google Drive mounted successfully")
    except Exception as e:
        print_error(f"Failed to mount Drive: {e}")
        print("  Continuing without Drive (some features may be unavailable)")


def install_library_group(group, config, gpu_available):
    """Install a single library group"""
    name = group['name']
    sequence = group['sequence']
    description = group.get('description', '')
    packages = group['packages']
    allow_failures = group.get('allow_failures', False)
    requires_index_url = group.get('requires_index_url', False)
    gpu_aware = group.get('gpu_aware', False)
    
    print_header(f"STEP 1.{sequence}: Install {name}")
    
    if description:
        print(f"  {description}\n")
    
    # Build package list
    package_list = []
    for pkg in packages:
        pkg_name = pkg['package_name']
        version = pkg.get('version', '')
        gpu_variant = pkg.get('gpu_variant', '')
        
        # Handle GPU-aware packages
        if gpu_aware and gpu_variant and gpu_available:
            if pkg_name == "onnxruntime":
                print_info(f"GPU detected, using {gpu_variant} instead of {pkg_name}")
                pkg_name = gpu_variant
        
        # Add version if specified
        if version:
            package_list.append(f"{pkg_name}{version}")
        else:
            package_list.append(pkg_name)
    
    # Build pip command
    pip_flags = config['global_settings']['pip_flags']
    cmd = f"pip install {pip_flags} {' '.join(package_list)}"
    
    # Add index URL if required (PyTorch)
    if requires_index_url:
        index_url = config['global_settings']['pytorch_index_url']
        cmd += f" --index-url {index_url}"
    
    # Execute installation
    try:
        run_command(
            cmd,
            message=f"  🛠️ Installing {', '.join(package_list)}",
            allow_failure=allow_failures
        )
        print_success(f"{name} installed")
        
    except Exception as e:
        if allow_failures:
            print(f"  ⚠ Some {name} failed: {e}")
            print("  (This may not be critical depending on your use case)")
        else:
            print_error(f"Failed to install {name}: {e}")
            raise


def main():
    """Main execution function"""
    start_time = time.time()
    
    # Configuration
    repo_root = "/content/unifiedposepipeline" if is_colab_environment() else os.getcwd()
    models_dir = "/content/models"
    
    # Top header
    print("\n" + "=" * 70)
    print(f"\033[93m🚀 STEP 1: Install Libraries and Dependencies\033[0m")
    print("=" * 70 + "\n")
    
    print("   This script will install all required Python packages.")
    print(f"   📁 Repository root: {repo_root}")
    print(f"   📁 Models directory: {models_dir}")
    
    try:
        # Load configuration
        config = load_library_config()
        print("   ✅ Loaded configuration from libraries.yaml")
        
        # Check GPU availability
        gpu_available = is_gpu_available()
        if gpu_available:
            print("   ✅ GPU detected - will use GPU-optimized packages where available")
        else:
            print("   ✅ No GPU detected - will use CPU-only packages")
        
        # Mount Drive (Colab only)
        mount_google_drive()
        
        # Install library groups in sequence order
        library_groups = sorted(config['library_groups'], key=lambda x: x['sequence'])
        
        for group in library_groups:
            install_library_group(group, config, gpu_available)
        
        # Final success message
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"\033[93m✅ SUCCESS: All libraries and dependencies installed!\033[0m")
        print(f"⏱️ TOTAL TIME TAKEN: {total_time:.2f}s")
        print("=" * 70 + "\n")
        print("🛠️ Next steps to try:")
        print("    ✓ python step2_fetch_models.py    # Download model files")
        print("    ✓ python step3_pull_demodata.py   # Setup demo data")
        print("    ✓ python step4_verify_envt.py     # Verify installation")
        
    except KeyboardInterrupt:
        print("\n\n⊘ Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
