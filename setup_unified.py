"""
Unified Pose Estimation Pipeline - Complete Setup (Legacy Wrapper)
For Google Colab and Local Environments

DEPRECATED: This script is maintained for backward compatibility.
Please use the new modular setup scripts instead:
  - step1_install_libs_deps.py  (Install libraries and dependencies)
  - step2_install_models.py     (Download model files)
  - step3_pull_demodata.py      (Setup demo data)
  - step4_verify_envt.py        (Verify installation)
  - setup_all.py                (Run all steps)

This wrapper now calls the modular scripts internally.
"""

import os
import sys
import subprocess


def print_header(message: str, char: str = "=", width: int = 70):
    """Print formatted header"""
    print("\n" + char * width)
    print(f"  {message}")
    print(char * width + "\n")


def print_deprecation_notice():
    """Print deprecation notice"""
    print("\n" + "=" * 70)
    print("DEPRECATION NOTICE")
    print("=" * 70)
    print("\nThis script (setup_unified.py) is deprecated.")
    print("Please use the new modular setup scripts:")
    print("  - step1_install_libs_deps.py")
    print("  - step2_install_models.py")
    print("  - step3_pull_demodata.py")
    print("  - step4_verify_envt.py")
    print("\nOr use the master script:")
    print("  - setup_all.py")
    print("\nContinuing with legacy wrapper (calls new scripts internally)...")
    print("=" * 70 + "\n")
    
    import time
    time.sleep(3)


def run_script(script_name):
    """Run a setup script"""
    print(f"\n{'─' * 70}")
    print(f"Running: {script_name}")
    print(f"{'─' * 70}\n")
    
    try:
        result = subprocess.run([sys.executable, script_name], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"✗ Script {script_name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ Failed to run {script_name}: {e}")
        return False


def main():
    """Main setup function - now calls modular scripts"""
    print_deprecation_notice()
    
    print("\n" + "=" * 70)
    print("UNIFIED POSE PIPELINE - COMPLETE SETUP (LEGACY WRAPPER)")
    print("=" * 70 + "\n")
    
    print("This wrapper will call the new modular setup scripts.")
    print("Press Ctrl+C to cancel and use the new scripts directly.\n")
    
    try:
        # Run all 4 steps
        results = {}
        
        print_header("Calling Step 1: Install Libraries and Dependencies")
        results["step1"] = run_script("step1_install_libs_deps.py")
        
        print_header("Calling Step 2: Download Model Files")
        results["step2"] = run_script("step2_install_models.py")
        
        print_header("Calling Step 3: Pull Demo Data")
        results["step3"] = run_script("step3_pull_demodata.py")
        
        print_header("Calling Step 4: Verify Environment")
        results["step4"] = run_script("step4_verify_envt.py")
        
        # Summary
        print("\n" + "=" * 70)
        print("SETUP SUMMARY")
        print("=" * 70)
        
        step_names = {
            "step1": "Install dependencies",
            "step2": "Download models",
            "step3": "Pull demo data",
            "step4": "Verify environment"
        }
        
        for step, status in results.items():
            status_str = "✓ SUCCESS" if status else "✗ FAILED"
            print(f"{step_names[step]:25s} {status_str}")
        
        all_passed = all(results.values())
        
        if all_passed:
            print("\n" + "=" * 70)
            print("🎉 SETUP COMPLETE!")
            print("=" * 70)
            print("\n📚 Next Steps:")
            print("   python udp_video.py --config configs/udp_video.yaml")
            print("   python run_detector_tracking.py --config configs/detector_tracking_benchmark.yaml")
        else:
            print("\n⚠️  Setup completed with some issues.")
            print("Please check the output above and re-run individual steps if needed.")
        
        return 0 if all_passed else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        print("Consider using the new modular scripts directly:")
        print("  python setup_all.py")
        return 1
    except Exception as e:
        print(f"\n\n❌ Setup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
