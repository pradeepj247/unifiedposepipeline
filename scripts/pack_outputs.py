"""
Pack Outputs - Create ZIP of analysis files (excluding videos)

Creates a ZIP archive of all files in demo_data/outputs/ except MP4 videos.
This allows easy transfer to Colab for analysis without re-running the pipeline.

Usage:
    python pack_outputs.py
"""

import zipfile
from pathlib import Path
from datetime import datetime

def pack_outputs():
    """Create ZIP of outputs folder (excluding MP4 files)"""
    
    # Paths
    outputs_dir = Path("demo_data/outputs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = Path(f"outputs_backup_{timestamp}.zip")
    
    if not outputs_dir.exists():
        print(f"❌ Outputs directory not found: {outputs_dir}")
        return
    
    # Collect files (exclude MP4)
    files_to_pack = []
    for file in outputs_dir.rglob("*"):
        if file.is_file() and file.suffix.lower() != ".mp4":
            files_to_pack.append(file)
    
    if not files_to_pack:
        print(f"⚠️  No files found to pack in {outputs_dir}")
        return
    
    print(f"\n📦 Packing {len(files_to_pack)} files...")
    print(f"   Excluding: *.mp4 files")
    print(f"   Output: {zip_path}")
    
    # Create ZIP
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_pack:
            arcname = file.relative_to(outputs_dir.parent)
            zipf.write(file, arcname)
            print(f"   ✓ {arcname}")
    
    # Summary
    size_mb = zip_path.stat().st_size / (1024 ** 2)
    print(f"\n✅ Created: {zip_path}")
    print(f"   Size: {size_mb:.2f} MB")
    print(f"   Files: {len(files_to_pack)}")
    
    # Instructions
    print(f"\n📋 To unpack in Colab:")
    print(f"   !unzip -q {zip_path.name}")
    print(f"   # Files will be restored to demo_data/outputs/")

if __name__ == "__main__":
    pack_outputs()
