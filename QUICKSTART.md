# Quick Start Guide - Unified Pose Pipeline

## 🎯 For Google Colab (Fresh Session)

### Step 1: Setup Everything
```python
# Run this ONCE per Colab session
!python setup_unified.py
```

**What it does:**
- ✅ Mounts Google Drive
- ✅ Installs PyTorch, OpenCV, YOLO, RTMLib, etc.
- ✅ Downloads YOLO models
- ✅ Copies ViTPose models from Drive (if available)
- ✅ Copies demo videos (dance.mp4, etc.)
- ✅ Creates directory structure
- ✅ Runs basic verification

**Time:** ~5-10 minutes

---

### Step 2: Verify Installation
```python
!python verify.py
```

**What it checks:**
- ✅ All library imports and versions
- ✅ CUDA/GPU availability
- ✅ Model files exist
- ✅ Demo data present
- ✅ Directory structure
- ✅ Functional tests (PyTorch, OpenCV, YOLO, RTMLib)

**Time:** ~30 seconds

---

### Step 3: Run Your First Demo

#### Option A: Image Demo (ViTPose)
```python
!python udp.py --config configs/vitpose_demo.yaml
```

#### Option B: Video Demo (RTMPose)
```python
!python udp.py --config configs/udp.yaml
```

#### Option C: Custom Config
```python
!python udp.py --config configs/video_demo.yaml
```

---

## 🔧 Configuration Quick Reference

### Config File Structure
```yaml
# configs/my_demo.yaml

detection:
  model_path: models/yolo/yolov8s.pt    # YOLO model
  confidence_threshold: 0.5              # Detection confidence

pose_estimation:
  type: rtmlib                          # or 'vitpose'
  model_type: rtmpose-l                 # RTM model size
  device: cuda                          # or 'cpu'

input:
  path: demo_data/videos/dance.mp4      # Input file

output:
  path: demo_data/outputs/result.mp4    # Output file
  save_json: false                      # Export keypoints?

processing:
  max_frames: 100                       # Limit frames (null=all)
  device: cuda
```

---

## 🎨 Common Modifications

### Change Model Speed/Accuracy

**Fastest (Real-time):**
```yaml
detection:
  model_path: models/yolo/yolov8n.pt    # Nano YOLO
pose_estimation:
  type: rtmlib
  model_type: rtmpose-m                  # Medium RTMPose
```

**Most Accurate:**
```yaml
detection:
  model_path: models/yolo/yolov8x.pt    # Extra-large YOLO
pose_estimation:
  type: vitpose
  model_name: vitpose-h                  # Huge ViTPose
  model_path: models/vitpose/vitpose-h.pth
```

**Balanced:**
```yaml
detection:
  model_path: models/yolo/yolov8s.pt    # Small YOLO
pose_estimation:
  type: rtmlib
  model_type: rtmpose-l                  # Large RTMPose
```

---

### Process Different Media

**Single Image:**
```yaml
input:
  type: image
  path: demo_data/images/person.jpg
output:
  path: demo_data/outputs/person_pose.jpg
```

**Full Video:**
```yaml
input:
  type: video
  path: demo_data/videos/dance.mp4
output:
  path: demo_data/outputs/dance_output.mp4
processing:
  max_frames: null    # Process all frames
```

**Video Segment (Testing):**
```yaml
processing:
  max_frames: 100     # Only first 100 frames
```

---

### Export Data for Analysis

```yaml
output:
  path: demo_data/outputs/result.mp4
  save_json: true                        # Enable JSON export
  json_path: demo_data/outputs/keypoints.json
```

JSON format:
```json
{
  "frame_0": {
    "person_0": {
      "bbox": [x1, y1, x2, y2],
      "keypoints": [[x, y, conf], ...]
    }
  }
}
```

---

## 📊 Performance Tips

### GPU Memory Issues?
1. Use smaller models (yolov8n, rtmpose-m)
2. Limit frames: `max_frames: 100`
3. Check GPU: `python verify.py`

### Slow Processing?
1. Verify CUDA is working: `python verify.py`
2. Use RTMPose instead of ViTPose (faster)
3. Use smaller YOLO (yolov8n)

### Want Best Quality?
1. Use ViTPose-H (most accurate)
2. Use YOLOv8x for detection
3. Process full video: `max_frames: null`

---

## 🐛 Troubleshooting

### "Module not found" error
```bash
# Re-run setup
python setup_unified.py
```

### "Model file not found"
```bash
# Check what's available
ls models/yolo/
ls models/vitpose/

# Re-run setup to download
python setup_unified.py
```

### "CUDA out of memory"
```yaml
# In your config, reduce model size:
detection:
  model_path: models/yolo/yolov8n.pt    # Smaller model
pose_estimation:
  model_type: rtmpose-m                  # Smaller model
processing:
  max_frames: 50                         # Fewer frames
```

### Verify what's wrong
```bash
python verify.py
```

This shows:
- ✅/❌ Each library status
- ✅/❌ CUDA availability
- ✅/❌ Model files
- ✅/❌ Demo data

---

## 📋 Checklist

Before running demos:

- [ ] Ran `setup_unified.py` (one time per session)
- [ ] Ran `verify.py` (all checks pass)
- [ ] Have config file ready in `configs/`
- [ ] Input file exists (video or image)
- [ ] Output directory is writable

---

## 🚀 Example Session

```python
# 1. Setup (first time)
!python setup_unified.py

# 2. Verify
!python verify.py

# 3. Test with image (fast)
!python udp.py --config configs/vitpose_demo.yaml

# 4. Test with video segment (100 frames)
!python udp.py --config configs/video_demo.yaml

# 5. Process full video
# Edit configs/video_demo.yaml: set max_frames: null
!python udp.py --config configs/video_demo.yaml

# 6. Check output
from IPython.display import Video
Video('demo_data/outputs/result.mp4')
```

---

## 💡 Pro Tips

1. **Always test with small frame count first** (`max_frames: 10`)
2. **Use verify.py if anything fails** (shows what's wrong)
3. **Start with default configs**, then customize
4. **Monitor GPU memory** in Colab (Settings → Runtime → Hardware accelerator)
5. **Save JSON output** for post-processing/analysis

---

## 📞 Help

| Issue | Solution |
|-------|----------|
| Setup fails | Check Google Drive is mounted |
| Import error | Run `verify.py` to see what's missing |
| Model not found | Check `models/` directories |
| Slow processing | Verify CUDA works: `verify.py` |
| Out of memory | Use smaller models, fewer frames |

---

**Ready to go? Start with:**
```bash
python setup_unified.py && python verify.py && python udp.py --config configs/rtmlib_demo.yaml
```
