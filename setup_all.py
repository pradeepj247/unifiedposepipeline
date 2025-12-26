#!/usr/bin/env python3
"""
Master Setup Script - Run All Setup Steps

This script orchestrates the complete setup process by running all 4 steps in sequence.
Provides command-line flags to skip certain steps.

Usage:
    python setup_all.py                    # Run all steps
    python setup_all.py --skip-models      # Skip model downloads
    python setup_all.py --skip-data        # Skip demo data setup
    python setup_all.py --verify-only      # Only run verification
    python setup_all.py --quick-verify     # Skip optional inference tests in verification
"""

import os
import sys
import argparse
import subprocess
from setup_utils import print_header, print_success, print_error, print_warning


def run_script(script_name, args=None):
    """
    Run a setup script and handle errors.
    
    Args:
        script_name (str): Name of the script to run
        args (list, optional): Additional arguments to pass to the script
        
    Returns:
        bool: True if successful, False otherwise
    """
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    
    print(f"\n{'─' * 70}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'─' * 70}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print_error(f"Script {script_name} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n\n⊘ Interrupted by user")
        raise
    except Exception as e:
        print_error(f"Failed to run {script_name}: {e}")
        return False


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Run complete setup process for Unified Pose Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_all.py                    # Run all steps
  python setup_all.py --skip-models      # Skip model downloads
  python setup_all.py --skip-data        # Skip demo data setup
  python setup_all.py --verify-only      # Only run verification
  python setup_all.py --quick-verify     # Quick verification (no inference tests)
        """
    )
    
    parser.add_argument("--skip-models", action="store_true",
                       help="Skip model downloads (Step 2)")
    parser.add_argument("--skip-data", action="store_true",
                       help="Skip demo data setup (Step 3)")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only run verification (Step 4)")
    parser.add_argument("--quick-verify", action="store_true",
                       help="Skip optional inference tests in verification")
    
    args = parser.parse_args()
    
    print_header("UNIFIED POSE PIPELINE - COMPLETE SETUP")
    
    print("This script will setup your environment for the Unified Pose Pipeline.")
    print("\nSetup steps:")
    if not args.verify_only:
        print("  1. Install libraries and dependencies")
        if not args.skip_models:
            print("  2. Download model files")
        else:
            print("  2. Download model files (SKIPPED)")
        if not args.skip_data:
            print("  3. Pull demo data")
        else:
            print("  3. Pull demo data (SKIPPED)")
    print("  4. Verify environment")
    print()
    
    results = {}
    
    try:
        # Step 1: Install dependencies (unless verify-only)
        if not args.verify_only:
            print_header("Starting Step 1 of 4")
            results["step1"] = run_script("step1_install_libs_deps.py")
            
            if not results["step1"]:
                print_error("Step 1 failed. Cannot continue.")
                sys.exit(1)
        
        # Step 2: Download models (unless skipped or verify-only)
        if not args.verify_only and not args.skip_models:
            print_header("Starting Step 2 of 4")
            results["step2"] = run_script("step2_install_models.py")
            
            if not results["step2"]:
                print_warning("Step 2 had issues. Continuing anyway...")
        
        # Step 3: Pull demo data (unless skipped or verify-only)
        if not args.verify_only and not args.skip_data:
            print_header("Starting Step 3 of 4")
            results["step3"] = run_script("step3_pull_demodata.py")
            
            if not results["step3"]:
                print_warning("Step 3 had issues. Continuing to verification...")
        
        # Step 4: Verify environment
        print_header("Starting Step 4 (Final)")
        verify_args = ["--quick"] if args.quick_verify else []
        results["step4"] = run_script("step4_verify_envt.py", verify_args)
        
        # Final summary
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
            print_success("Complete setup finished successfully!")
            print("\nYou can now use the pipeline:")
            print("  python udp_video.py --config configs/udp_video.yaml")
            print("  python run_detector_tracking.py --config configs/detector_tracking_benchmark.yaml")
        else:
            print_warning("Setup completed with some issues. Check output above.")
            print("\nYou can re-run individual steps:")
            print("  python step1_install_libs_deps.py")
            print("  python step2_install_models.py")
            print("  python step3_pull_demodata.py")
            print("  python step4_verify_envt.py")
        
    except KeyboardInterrupt:
        print("\n\n⊘ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
