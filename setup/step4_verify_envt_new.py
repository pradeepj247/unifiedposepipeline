#!/usr/bin/env python3
"""
Step 4: Verify Environment (YAML-driven)

This script performs comprehensive verification of the installation by reading
configuration from the YAML files used in steps 1-3.

Verification includes:
- Package imports and versions (from libraries.yaml)
- CUDA/GPU availability
- Model files (from models.yaml)
- Demo data (from demodata.yaml)

Usage:
    python step4_verify_envt_new.py
"""

import os
import sys
import yaml
import importlib
import importlib.metadata
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


def print_step(step_num, text):
    """Print a step header"""
    print(f"\n  📋 Step {step_num}: {text}")
    print("  " + "─" * 60)


def print_error(text):
    """Print error message in red"""
    print(f"\033[91m  ✗ {text}\033[0m")


def print_success(text):
    """Print success message"""
    print(f"  ✅ {text}")


def print_warning(text):
    """Print warning message"""
    print(f"  ⚠️  {text}")


def print_info(text):
    """Print info message"""
    print(f"  {text}")


# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIBRARIES_YAML = os.path.join(SCRIPT_DIR, "libraries.yaml")
MODELS_YAML = os.path.join(SCRIPT_DIR, "models.yaml")
DEMODATA_YAML = os.path.join(SCRIPT_DIR, "demodata.yaml")

REPO_ROOT = "/content/unifiedposepipeline" if is_colab_environment() else os.getcwd()


def load_yaml_config(yaml_file):
    """Load configuration from YAML file"""
    if not os.path.exists(yaml_file):
        print_error(f"Config file not found: {yaml_file}")
        return None
    
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)


def get_package_version(package_name):
    """Get installed version of a package"""
    try:
        # Try importlib.metadata first (Python 3.8+)
        version = importlib.metadata.version(package_name)
        return version
    except importlib.metadata.PackageNotFoundError:
        # Try package.__version__ as fallback
        try:
            module = importlib.import_module(package_name)
            if hasattr(module, '__version__'):
                return module.__version__
            return "unknown"
        except:
            return "unknown"


def get_import_name(package_name):
    """
    Map package names to their import names.
    Some packages have different pip names vs import names.
    """
    mapping = {
        'opencv-python': 'cv2',
        'opencv-contrib-python': 'cv2',
        'pillow': 'PIL',
        'pyyaml': 'yaml',
        'scikit-learn': 'sklearn',
        'scikit-image': 'skimage',
        'onnxruntime-gpu': 'onnxruntime',  # GPU variant imports as onnxruntime
    }
    return mapping.get(package_name, package_name)


def check_package_import(package_name):
    """Check if a package can be imported"""
    import_name = get_import_name(package_name)
    
    try:
        # Special handling for ultralytics (suppress settings output)
        if package_name == 'ultralytics':
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                importlib.import_module(import_name)
        else:
            importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def verify_python_packages():
    """Verify all packages from libraries.yaml"""
    print_step("4.1", "Package Verification")
    
    config = load_yaml_config(LIBRARIES_YAML)
    if not config:
        return False
    
    all_packages = []
    success_count = 0
    failed_packages = []
    
    # Extract all packages from library groups
    for group in config.get('library_groups', []):
        for pkg in group.get('packages', []):
            package_name = pkg.get('package_name')
            
            # Handle GPU variants
            gpu_variant = pkg.get('gpu_variant')
            if gpu_variant:
                # Check if GPU is available to determine which package to verify
                try:
                    import torch
                    if torch.cuda.is_available():
                        package_name = gpu_variant
                except:
                    pass
            
            all_packages.append(package_name)
    
    # Verify each package
    print()
    for package_name in all_packages:
        can_import = check_package_import(package_name)
        
        if can_import:
            version = get_package_version(package_name)
            import_name = get_import_name(package_name)
            
            # Show import name if different from package name
            if import_name != package_name:
                print(f"  📦 {package_name} → {import_name} ({version}) ✓")
            else:
                print(f"  📦 {package_name} ({version}) ✓")
            
            success_count += 1
        else:
            print_error(f"{package_name} ✗ FAILED TO IMPORT")
            failed_packages.append(package_name)
    
    print(f"\n  📊 {success_count}/{len(all_packages)} packages verified")
    
    if failed_packages:
        print_warning(f"Missing packages: {', '.join(failed_packages)}")
        return False
    
    return True


def verify_cuda_gpu():
    """Verify CUDA and GPU availability"""
    print_step("4.2", "Environment Verification (CUDA/GPU)")
    
    try:
        import torch
        
        print(f"\n  🔹 PyTorch version: {torch.__version__}")
        print(f"  🔹 CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  🔹 CUDA version: {torch.version.cuda}")
            print(f"  🔹 GPU count: {torch.cuda.device_count()}")
            print(f"  🔹 GPU device: {torch.cuda.get_device_name(0)}")
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  🔹 GPU memory: {gpu_memory_gb:.1f} GB")
            print_success("CUDA environment verified")
            return True
        else:
            print_warning("CUDA not available - will use CPU (slower)")
            return False
            
    except Exception as e:
        print_error(f"Failed to check CUDA: {e}")
        return False


def verify_model_files():
    """Verify all model files from models.yaml"""
    print_step("4.3", "Model Files Verification")
    
    config = load_yaml_config(MODELS_YAML)
    if not config:
        return False
    
    models = config.get('models', [])
    destination_folder = config.get('destination_folder', '/content/models/')
    
    success_count = 0
    missing_models = []
    
    print()
    for model in models:
        model_name = model.get('name')
        filename = model.get('filename')
        subfolder = model.get('subfolder', '')
        
        # Build full path
        model_path = os.path.join(destination_folder, subfolder, filename)
        
        if os.path.isfile(model_path):
            # Get file size
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"  📦 {model_name} ({size_mb:.1f} MB) ✓")
            success_count += 1
        else:
            print_error(f"{model_name} ✗ NOT FOUND")
            print(f"      Expected: {model_path}")
            missing_models.append(model_name)
    
    print(f"\n  📊 {success_count}/{len(models)} model files verified")
    
    if missing_models:
        print_warning(f"Missing models: {', '.join(missing_models)}")
        return False
    
    return True


def verify_demo_data():
    """Verify demo data from demodata.yaml"""
    print_step("4.4", "Demo Data Verification")
    
    config = load_yaml_config(DEMODATA_YAML)
    if not config:
        return False
    
    demo_groups = config.get('demo_groups', [])
    destination_folder = config.get('global_settings', {}).get('destination_folder', '/content/unifiedposepipeline/demo_data')
    
    success_count = 0
    missing_data = []
    
    print()
    for group in demo_groups:
        group_name = group.get('name')
        subfolder = group.get('subfolder')
        
        # Build directory path
        data_path = os.path.join(destination_folder, subfolder)
        
        if os.path.isdir(data_path):
            # Count files in directory
            files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
            file_count = len(files)
            
            if file_count > 0:
                print(f"  📁 {group_name}: {file_count} file(s) ✓")
                print(f"      Location: {data_path}")
                success_count += 1
            else:
                print_warning(f"{group_name}: directory exists but empty")
                print(f"      Location: {data_path}")
                missing_data.append(group_name)
        else:
            print_error(f"{group_name} ✗ DIRECTORY NOT FOUND")
            print(f"      Expected: {data_path}")
            missing_data.append(group_name)
    
    print(f"\n  📊 {success_count}/{len(demo_groups)} demo data groups verified")
    
    if missing_data:
        print_warning(f"Missing demo data: {', '.join(missing_data)}")
        return False
    
    return True


def main():
    """Main execution function"""
    print_header("STEP 4: Verify Environment", emoji="🔍")
    
    print("  This script will verify your installation by reading configurations")
    print("  from the YAML files used in steps 1-3.\n")
    print(f"  📂 Repository root: {REPO_ROOT}")
    print(f"  📂 Libraries config: {LIBRARIES_YAML}")
    print(f"  📂 Models config: {MODELS_YAML}")
    print(f"  📂 Demo data config: {DEMODATA_YAML}")
    
    results = {}
    
    try:
        # Run all verification steps
        results["packages"] = verify_python_packages()
        results["cuda"] = verify_cuda_gpu()
        results["models"] = verify_model_files()
        results["demo_data"] = verify_demo_data()
        
        # Summary
        print("\n" + "  " + "=" * 66)
        print("  📋 VERIFICATION SUMMARY")
        print("  " + "=" * 66 + "\n")
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        for check, status in results.items():
            status_str = "✓ PASS" if status else "✗ FAIL"
            status_emoji = "✅" if status else "❌"
            print(f"  {status_emoji} {check.upper():20s} {status_str}")
        
        print(f"\n  📊 Overall: {passed}/{total} checks passed")
        
        if passed == total:
            print_success("Environment verification complete! All checks passed.")
            print("\n  You can now use the pipeline:")
            print("    python udp_video.py --config configs/udp_video.yaml")
            print("    python udp_image.py --config configs/udp_image.yaml")
        else:
            print_warning("Some checks failed. Please review the output above.")
            print("\n  💡 You may need to:")
            print("    - Re-run step1_install_libs.py for missing packages")
            print("    - Re-run step2_fetch_models.py for missing models")
            print("    - Re-run step3_fetch_demodata.py for missing demo data")
        
        print()
        
    except KeyboardInterrupt:
        print("\n\n  ⊘ Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
