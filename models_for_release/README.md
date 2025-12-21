# RTMPose ONNX Models for GitHub Release

## 📦 Files Ready for Release

These ONNX models have been extracted from the official OpenMMLab RTMPose releases:

### 1. rtmpose-l-coco-384x288.onnx (105.97 MB)
- **Purpose**: Standard COCO 17 keypoints pose estimation
- **Resolution**: 384×288
- **Model**: RTMPose-L
- **Source**: `rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.zip`
- **Original URL**: https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.zip

### 2. rtmpose-l-halpe26-384x288.onnx (107.69 MB)
- **Purpose**: Extended Halpe26 26 keypoints pose estimation (includes feet)
- **Resolution**: 384×288
- **Model**: RTMPose-L Halpe26
- **Source**: `rtmpose-l_simcc-body7_pt-body7-halpe26_700e-384x288-734182ce_20230605.zip`
- **Original URL**: https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-body7_pt-body7-halpe26_700e-384x288-734182ce_20230605.zip

---

## 🚀 How to Create GitHub Release

### Step 1: Go to GitHub Releases
Visit: https://github.com/pradeepj247/stage3hybrik/releases

### Step 2: Create New Release
Click **"Draft a new release"** or **"Create a new release"**

### Step 3: Fill in Release Details

**Tag version**: `v1.0-models`

**Release title**: `RTMPose ONNX Models v1.0`

**Description**:
```markdown
# RTMPose ONNX Models for Unified Pose Pipeline

Pre-extracted ONNX models for fast deployment without zip extraction overhead.

## 📦 Models Included

- **rtmpose-l-coco-384x288.onnx** (105.97 MB)
  - Standard COCO 17 keypoints
  - High accuracy body pose estimation
  - Best for: General pose estimation tasks

- **rtmpose-l-halpe26-384x288.onnx** (107.69 MB)
  - Extended Halpe26 26 keypoints (includes feet: toes, heels)
  - Enhanced body representation
  - Best for: Full-body tracking with foot details

## 🎯 Performance

- **RTMPose COCO**: ~48 FPS on 360 frames
- **RTMPose Halpe26**: ~54 FPS on 360 frames (11% faster!)

## 📥 Usage

These models are automatically downloaded by `setup_unified.py`:

```bash
python setup_unified.py
```

Models will be installed to: `/content/models/rtmlib/`

## 📚 Original Sources

Both models extracted from OpenMMLab's official RTMPose releases:
- https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose

## ⚖️ License

These models follow the original RTMPose licensing from OpenMMLab.
```

### Step 4: Upload Files
Drag and drop both ONNX files:
- `rtmpose-l-coco-384x288.onnx`
- `rtmpose-l-halpe26-384x288.onnx`

### Step 5: Publish Release
Click **"Publish release"**

---

## ✅ After Publishing

The download URLs will be:
```
https://github.com/pradeepj247/stage3hybrik/releases/download/v1.0-models/rtmpose-l-coco-384x288.onnx
https://github.com/pradeepj247/stage3hybrik/releases/download/v1.0-models/rtmpose-l-halpe26-384x288.onnx
```

These URLs are already configured in `setup_unified.py`! ✨

---

## 🧹 Cleanup (After Release)

Once the release is published and tested, you can safely delete:
- `D:\trials\unifiedpipeline\newrepo\models\*.zip` (original ZIP files)
- `D:\trials\unifiedpipeline\newrepo\models\rtmlib_extracted\` (extracted folders)
- Keep `models_for_release\` for future reference or delete after uploading

---

## 📊 File Integrity

Original ONNX files extracted from official OpenMMLab releases with no modifications.
MD5 checksums can be verified against source archives if needed.
