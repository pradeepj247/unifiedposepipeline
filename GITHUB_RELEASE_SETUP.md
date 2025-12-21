# GitHub Release Setup Guide for RTMPose Models

This guide explains how to extract RTMPose ONNX models and upload them to GitHub Releases.

## 📦 Why GitHub Releases?

- **Predictable setup**: All models download during `setup_unified.py`
- **Version control**: Lock specific model versions
- **Faster downloads**: GitHub CDN is reliable and fast
- **Consistency**: All models follow same download pattern (YOLO, ViTPose, RTMPose)
- **Offline-friendly**: Download once, share internally

## 🔧 One-Time Extraction (Run on Colab)

### Step 1: Extract ONNX Files

Run this code block in a Colab notebook to download and extract the ONNX files:

```python
# ===== Download and Extract RTMPose ONNX Files =====
import zipfile
import shutil
from pathlib import Path
import urllib.request

# Create extraction directory
extract_dir = Path("/content/rtmpose_onnx_extraction")
extract_dir.mkdir(exist_ok=True)

# Model URLs
models = {
    "rtmpose-l-coco": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_700e-384x288-24e67831_20230504.zip",
        "zip_name": "rtmpose-l-coco.zip",
        "onnx_name": "rtmpose-l-coco-384x288.onnx"
    },
    "rtmpose-l-halpe26": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7-halpe26_700e-384x288-734182ce_20230605.zip",
        "zip_name": "rtmpose-l-halpe26.zip",
        "onnx_name": "rtmpose-l-halpe26-384x288.onnx"
    }
}

print("=" * 70)
print("📥 Downloading and Extracting RTMPose ONNX Models")
print("=" * 70)

for model_name, model_info in models.items():
    print(f"\n🔄 Processing: {model_name}")
    
    # Download zip
    zip_path = extract_dir / model_info["zip_name"]
    print(f"   Downloading from {model_info['url'][:60]}...")
    urllib.request.urlretrieve(model_info["url"], zip_path)
    print(f"   ✅ Downloaded: {zip_path.name} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Extract zip
    temp_extract = extract_dir / f"{model_name}_temp"
    temp_extract.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract)
    print(f"   ✅ Extracted to temporary folder")
    
    # Find and copy ONNX file
    onnx_files = list(temp_extract.rglob("*.onnx"))
    if onnx_files:
        onnx_src = onnx_files[0]
        onnx_dst = extract_dir / model_info["onnx_name"]
        shutil.copy2(onnx_src, onnx_dst)
        print(f"   ✅ ONNX extracted: {onnx_dst.name} ({onnx_dst.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print(f"   ❌ No ONNX file found in {model_name}")
    
    # Cleanup
    zip_path.unlink()
    shutil.rmtree(temp_extract)

print("\n" + "=" * 70)
print("✅ Extraction Complete!")
print("=" * 70)
print(f"\n📁 ONNX files ready in: {extract_dir}")
print("\nFiles to upload to GitHub Releases:")
for f in extract_dir.glob("*.onnx"):
    print(f"   • {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
```

### Step 2: Download Files to Local Machine

```python
# ===== Download Files from Colab =====
from google.colab import files

print("📥 Downloading ONNX files to your computer...")
print("(This may take a minute due to file sizes ~100MB each)\n")

for onnx_file in extract_dir.glob("*.onnx"):
    print(f"Downloading: {onnx_file.name}")
    files.download(str(onnx_file))

print("\n✅ Downloads complete!")
```

## 📤 Upload to GitHub Releases

### Step 3: Create GitHub Release

1. Go to: https://github.com/pradeepj247/unifiedposepipeline/releases
2. Click **"Create a new release"**
3. Fill in release details:
   - **Tag**: `v1.0-models`
   - **Title**: `RTMPose ONNX Models v1.0`
   - **Description**:
     ```
     RTMPose ONNX models for unified pose pipeline
     
     Models included:
     - rtmpose-l-coco-384x288.onnx (~106 MB) - Standard COCO 17 keypoints
     - rtmpose-l-halpe26-384x288.onnx (~110 MB) - Halpe26 26 keypoints
     
     These models are extracted from OpenMMLab releases and repackaged for easier deployment.
     
     Original sources:
     - https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
     ```

4. **Upload files**: Drag and drop both `.onnx` files
5. Click **"Publish release"**

## ✅ Verification

After creating the release, the URLs will be:
- `https://github.com/pradeepj247/unifiedposepipeline/releases/download/v1.0.0/rtmpose-l-coco-384x288.onnx`
- `https://github.com/pradeepj247/unifiedposepipeline/releases/download/v1.0.0/rtmpose-l-halpe26-384x288.onnx`

These URLs are already configured in `setup_unified.py`!

## 🔄 Code Changes Summary

The following files have been updated to use local ONNX paths:

### Modified Files:
1. **setup_unified.py** - Added RTMPose downloads from GitHub releases
2. **configs/udp_video.yaml** - Changed `pose_model_url` → `pose_model_path`
3. **configs/udp_image.yaml** - Changed `pose_model_url` → `pose_model_path`
4. **udp_video.py** - Load from local path instead of URL
5. **udp_image.py** - Load from local path + added Halpe26 function
6. **verify_unified.py** - Check for required ONNX files in `models/rtmlib/`

### Model Storage Structure:
```
/content/
├── models/
│   ├── yolo/
│   │   └── yolov8s.pt                        (22 MB)
│   ├── vitpose/
│   │   └── vitpose-b.pth                     (343 MB)
│   └── rtmlib/                               ← NEW!
│       ├── rtmpose-l-coco-384x288.onnx       (106 MB)
│       └── rtmpose-l-halpe26-384x288.onnx    (110 MB)
└── unifiedposepipeline/
    └── ...
```

## 🚀 Usage After Setup

Users now have a consistent experience:

```bash
# 1. Clone repo
git clone https://github.com/pradeepj247/stage3hybrik.git
cd stage3hybrik/unifiedposepipeline

# 2. Run setup (downloads ALL models including RTMPose)
python setup_unified.py

# 3. Verify (checks for RTMPose ONNX files)
python verify_unified.py

# 4. Run demos (no surprise downloads!)
python udp_video.py --config configs/udp_video.yaml
```

All models download during setup - no surprises on first run! 🎉
