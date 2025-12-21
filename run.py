#!/usr/bin/env python3
"""
Unified Pose Pipeline - Helper Launcher

Quick shortcuts for common tasks.
Usage:
    python run.py setup          # Run setup
    python run.py verify         # Run verification
    python run.py demo image     # Run image demo (quick test)
    python run.py demo video     # Run video demo (full test)
    python run.py list-configs   # List available configs
    python run.py help           # Show help
"""

import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parent


def run_command(cmd: list):
    """Run command and return exit code"""
    print(f"$ {' '.join(cmd)}")
    return subprocess.call(cmd)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    
    command = sys.argv[1].lower()
    
    # Setup
    if command == "setup":
        print("🚀 Running setup...")
        return run_command([sys.executable, "setup_unified.py"])
    
    # Verify
    elif command == "verify":
        print("🔍 Running verification...")
        return run_command([sys.executable, "verify_unified.py"])
    
    # Demo
    elif command == "demo":
        if len(sys.argv) < 3:
            print("❌ Usage: python run.py demo <image|video>")
            return 1
        
        demo_type = sys.argv[2].lower()
        
        if demo_type == "image":
            print("🎯 Running image demo (quick verification)...")
            return run_command([sys.executable, "udp_image.py", "--config", "configs/udp_image.yaml"])
        elif demo_type == "video":
            print("🎯 Running video demo (comprehensive test)...")
            return run_command([sys.executable, "udp_video.py", "--config", "configs/udp_video.yaml"])
        else:
            print(f"❌ Unknown demo type: {demo_type}")
            print(f"   Available: image, video")
            return 1
    
    # List configs
    elif command == "list-configs" or command == "configs":
        print("📋 Available configurations:\n")
        configs_dir = REPO_ROOT / "configs"
        if configs_dir.exists():
            for config_file in sorted(configs_dir.glob("*.yaml")):
                print(f"   • {config_file.name}")
                # Try to read description from first comment
                try:
                    with open(config_file) as f:
                        first_line = f.readline().strip()
                        if first_line.startswith("#"):
                            desc = first_line.lstrip("#").strip()
                            print(f"     {desc}")
                except:
                    pass
        else:
            print("   ⚠️  configs/ directory not found")
        return 0
    
    # Custom config
    elif command == "run":
        if len(sys.argv) < 3:
            print("❌ Usage: python run.py run <config_file.yaml>")
            return 1
        
        config = sys.argv[2]
        print(f"🎯 Running with config: {config}")
        return run_command([sys.executable, "udp.py", "--config", config])
    
    # Help
    elif command == "help" or command == "-h" or command == "--help":
        print(__doc__)
        return 0
    
    else:
        print(f"❌ Unknown command: {command}")
        print(__doc__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
