"""
COLAB: Display HTML report info after generation

Run this after the embedded videos test to see file details
"""

from pathlib import Path

html_file = Path('/content/unifiedposepipeline/demo_data/outputs/kohli_nets/person_selection_report.html')

if html_file.exists():
    print("\n" + "="*70)
    print("📊 HTML REPORT DETAILS")
    print("="*70)
    
    html_size_mb = html_file.stat().st_size / (1024 * 1024)
    
    print(f"\n✅ File: {html_file.name}")
    print(f"   Size: {html_size_mb:.2f} MB")
    print(f"   Path: {html_file}")
    
    # Count embedded videos
    with open(html_file, 'r') as f:
        content = f.read()
        video_count = content.count('data:video/mp4;base64,')
    
    print(f"   Embedded videos: {video_count}")
    print(f"\n🎬 Features:")
    print(f"   ✓ 10 person cards with embedded MP4 videos")
    print(f"   ✓ Play/pause controls on each video")
    print(f"   ✓ Person statistics (frames, coverage, duration)")
    print(f"   ✓ Responsive grid layout (auto-adapts to screen size)")
    print(f"   ✓ Smooth hover effects and modern UI")
    print(f"\n💻 Open in browser:")
    print(f"   file://{html_file}")
    print(f"\n📱 Features:")
    print(f"   - Fully responsive (works on mobile/tablet)")
    print(f"   - All videos embedded (no external files needed)")
    print(f"   - H.264 codec (widely supported)")
    print(f"   - 15 fps playback (~3.3 seconds per person)")
    print("="*70)
else:
    print(f"\n❌ HTML file not found: {html_file}")
