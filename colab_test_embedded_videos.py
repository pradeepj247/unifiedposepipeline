"""
COLAB TEST SCRIPT: Stage 10 & 11 with Embedded Videos
Run this in a Colab cell to generate the complete HTML report with embedded MP4 videos
"""

import os
import sys
import time
from pathlib import Path

os.chdir('/content/unifiedposepipeline/det_track')

print("=" * 70)
print("🎬 TESTING STAGES 10 & 11: HTML WITH EMBEDDED MP4 VIDEOS")
print("=" * 70)
print()

# Import both stages
from stage6b_create_selection_html import main as stage10_main
from stage9_generate_person_gifs import main as stage11_main
import sys as sys_module
from argparse import Namespace

# Override sys.argv for both stages
original_argv = sys_module.argv

print("\n🚀 Running Stage 11: Generate MP4 Videos...")
print("-" * 70)
sys_module.argv = ['stage9_generate_person_gifs.py', '--config', 'configs/pipeline_config.yaml']
t1_start = time.time()
try:
    exit_code = stage11_main()
except SystemExit as e:
    exit_code = e.code
t1_end = time.time()

print("\n🚀 Running Stage 10: Generate HTML with Embedded Videos...")
print("-" * 70)
sys_module.argv = ['stage6b_create_selection_html.py', '--config', 'configs/pipeline_config.yaml']
t10_start = time.time()
try:
    exit_code = stage10_main()
except SystemExit as e:
    exit_code = e.code
t10_end = time.time()

sys_module.argv = original_argv

print()
print("=" * 70)
print("📊 FINAL RESULTS")
print("=" * 70)
print(f"Stage 11 (MP4 videos):  {t1_end - t1_start:.2f}s")
print(f"Stage 10 (HTML embed):  {t10_end - t10_start:.2f}s")
print(f"TOTAL TIME:             {(t1_end - t1_start) + (t10_end - t10_start):.2f}s")
print()

# Show file sizes
videos_dir = Path('/content/unifiedposepipeline/demo_data/outputs/kohli_nets/videos')
html_file = Path('/content/unifiedposepipeline/demo_data/outputs/kohli_nets/person_selection_report.html')

if videos_dir.exists():
    total_video_size = sum(f.stat().st_size for f in videos_dir.glob('*.mp4')) / (1024 * 1024)
    print(f"✅ Video files: {len(list(videos_dir.glob('*.mp4')))} MP4s = {total_video_size:.2f} MB")

if html_file.exists():
    html_size_mb = html_file.stat().st_size / (1024 * 1024)
    print(f"✅ HTML file: {html_file.name} = {html_size_mb:.2f} MB")
    print(f"   📍 Open: file://{html_file}")
    print(f"\n   💾 Total combined: {total_video_size + html_size_mb:.2f} MB (all videos + HTML)")

print("=" * 70)
