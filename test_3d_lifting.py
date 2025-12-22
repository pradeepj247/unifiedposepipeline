"""
Quick test script for 3D lifting pipeline

This script:
1. Runs Stage 1-2 of udp_video.py to get 2D keypoints (RTMPose)
2. Runs udp_3d_lifting.py to lift 2D → 3D with MotionAGFormer
3. Creates visualization

Usage:
    python test_3d_lifting.py
"""

import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parent

def run_command(cmd, description):
    """Run a command and print status"""
    print(f"\n{'='*70}")
    print(f"🚀 {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    
    if result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        return False
    
    print(f"\n✅ Success: {description}")
    return True


def main():
    print("\n" + "🎬" * 35)
    print("3D LIFTING TEST - Full Pipeline")
    print("🎬" * 35)
    
    # Check if checkpoint exists
    checkpoint = REPO_ROOT.parent / "models" / "motionagformer" / "motionagformer-base-h36m.pth.tr"
    if not checkpoint.exists():
        print(f"\n❌ MotionAGFormer checkpoint not found: {checkpoint}")
        print("   Run setup_unified.py first to download the checkpoint")
        return 1
    
    size_mb = checkpoint.stat().st_size / (1024 ** 2)
    print(f"\n✅ Found checkpoint: motionagformer-base-h36m.pth.tr ({size_mb:.1f} MB)")
    
    # Check if 2D keypoints already exist
    keypoints_file = REPO_ROOT / "demo_data" / "outputs" / "keypoints_2D.npz"
    detections_file = REPO_ROOT / "demo_data" / "outputs" / "detections.npz"
    
    if keypoints_file.exists() and detections_file.exists():
        print("\n" + "="*70)
        print("⏭️  STEP 1 & 2: SKIPPED (outputs already exist)")
        print("="*70)
        print(f"   Found: {keypoints_file}")
        print(f"   Found: {detections_file}")
        print("   Using existing 2D keypoints for 3D lifting")
    else:
        # Step 1: Generate 2D keypoints (RTMPose)
        print("\n" + "="*70)
        print("STEP 1 & 2: Generate 2D Keypoints (RTMPose)")
        print("="*70)
        
        success = run_command(
            ["python", "udp_video.py", "--config", "configs/udp_video.yaml"],
            "Running 2D pose estimation (RTMPose)"
        )
        
        if not success:
            return 1
        
        # Check if keypoints were generated
        if not keypoints_file.exists():
            print(f"\n❌ Keypoints file not found: {keypoints_file}")
            return 1
        
        print(f"\n✅ Generated keypoints: {keypoints_file}")
    
    # Step 3: Run 3D lifting
    print("\n" + "="*70)
    print("STEP 3: 3D Lifting (MotionAGFormer)")
    print("="*70)
    
    video_file = REPO_ROOT / "demo_data" / "videos" / "dance.mp4"
    
    success = run_command(
        [
            "python", "udp_3d_lifting.py",
            "--keypoints", str(keypoints_file),
            "--video", str(video_file),
            "--visualize",
            "--max-frames", "360"
        ],
        "Running 3D pose lifting with visualization"
    )
    
    if not success:
        return 1
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST COMPLETE!")
    print("="*70)
    
    outputs_dir = REPO_ROOT / "demo_data" / "outputs"
    print(f"\nOutput files in: {outputs_dir}")
    print("  - detections.npz (Stage 1: Person detection)")
    print("  - keypoints_2D.npz (Stage 2: 2D pose)")
    print("  - keypoints_2D_3d.npy (Stage 3: 3D pose)")
    print("  - keypoints_2D_3d.mp4 (Stage 3: 3D visualization)")
    print("  - result_2D.mp4 (Stage 2: 2D visualization)")
    
    print("\n✨ You can now view the 3D visualization video!")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
