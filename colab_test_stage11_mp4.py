"""
COLAB TEST SCRIPT: Stage 11 MP4 Generation
Run this in a Colab cell to test the MP4 video generation
"""

import os
import sys
import time
from pathlib import Path

# Ensure we're in the right directory
os.chdir('/content/unifiedposepipeline/det_track')

# Run Stage 11 directly
print("=" * 70)
print("🎬 TESTING STAGE 11: MP4 VIDEO GENERATION")
print("=" * 70)
print()

# Import and run stage 11
from stage9_generate_person_gifs import main
import sys
from argparse import Namespace

# Override sys.argv to pass config
original_argv = sys.argv
sys.argv = ['stage9_generate_person_gifs.py', '--config', 'configs/pipeline_config.yaml']

start_time = time.time()
try:
    exit_code = main()
except SystemExit as e:
    exit_code = e.code
finally:
    sys.argv = original_argv

end_time = time.time()

print()
print("=" * 70)
print(f"TOTAL EXECUTION TIME: {end_time - start_time:.2f} seconds")
print("=" * 70)

# Show what was generated
videos_dir = Path('/content/unifiedposepipeline/demo_data/outputs/kohli_nets/videos')
if videos_dir.exists():
    mp4_files = list(videos_dir.glob('*.mp4'))
    print(f"\n✅ Generated {len(mp4_files)} MP4 files:")
    total_size = 0
    for f in sorted(mp4_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        total_size += f.stat().st_size
        print(f"  - {f.name} ({size_mb:.2f} MB)")
    print(f"\nTotal size: {total_size / (1024 * 1024):.2f} MB")
else:
    print(f"\n❌ Videos directory not found: {videos_dir}")
