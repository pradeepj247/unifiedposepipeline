# Quick Start: Tracking + ReID Benchmark on Google Colab

**Video:** `/content/campus_walk.mp4`  
**Goal:** Benchmark BoxMOT tracking with ReID, observe ID stability and FPS impact

---

## 🚀 Quick Run (One Command)

```bash
cd /content/unifiedposepipeline
python test_tracking_reid_benchmark.py
```

This automated test will:
1. ✅ Verify BoxMOT installation
2. ✅ Check video file
3. ✅ Run tracking + ReID benchmark
4. ✅ Generate visualization video
5. ✅ Report performance metrics

---

## 📋 Manual Run (Step by Step)

If you want more control:

```bash
cd /content/unifiedposepipeline

# Run tracking with ReID
python run_detector_tracking.py --config configs/detector_tracking_benchmark.yaml
```

---

## 📊 Expected Output

### Console:
```
⏱️  Performance Breakdown:
  Detection FPS: 70.5 (14.2ms/frame)      ← YOLO speed (unchanged)
  Tracking FPS:  150.3 (6.7ms/frame)      ← ByteTrack speed
  Combined FPS:  42.8 (23.4ms/frame)      ← With ReID (2x slower!)
  Overall FPS:   30.0 (including I/O)

📊 Tracking Statistics:
  Unique track IDs: 12                     ← People tracked
  Track IDs seen: [1, 2, 3, 4, 5, ...]    ← All IDs
```

### Files Created:
- **NPZ:** `demo_data/outputs/detections_tracking_reid.npz`
- **Video:** `demo_data/outputs/campus_walk_tracking_reid.mp4` ← **Watch this!**

---

## 🎥 What You'll See in the Video

- ✅ Each person gets unique **colored bbox**
- ✅ **Track ID** displayed above each bbox (e.g., "ID:1", "ID:2")
- ✅ Same ID maintained across frames (with ReID)
- ✅ Frame number + active track count shown

---

## 🔧 Configuration Used

**Tracker:** ByteTrack (fastest: 1265 Hz baseline)  
**ReID Model:** OSNet x1.0 MSMT17 (~2M params)  
**Detection:** YOLOv8s (humans only)  
**Visualization:** Enabled

See: `configs/detector_tracking_benchmark.yaml`

---

## 📈 Performance Comparison

| Mode | FPS | Track ID Stability | Use Case |
|------|-----|-------------------|----------|
| Detection-only | 70 | N/A | Speed critical |
| Motion tracking | 65 | Medium | Balanced |
| **Tracking + ReID** | **40** | **High** | **Quality critical** |

**ReID Impact:** 70 → 40 FPS (2x slower, but stable IDs!)

---

## 🔄 Test Different Configurations

### 1. Disable ReID (faster, less stable):
Edit `configs/detector_tracking_benchmark.yaml`:
```yaml
reid:
  enabled: false  # Motion-only tracking
```
Expected: ~65 FPS, unstable IDs

### 2. Try Different Tracker:
```yaml
tracker: botsort  # Best accuracy (HOTA: 69.4)
```
or
```yaml
tracker: boosttrack  # Best IDF1 (83.2)
```

### 3. Use Lighter ReID Model:
```yaml
weights_path: models/reid/osnet_x0_25_msmt17.pt  # Faster
```

---

## 🐛 Troubleshooting

**Issue:** "Video not found"  
**Solution:** Ensure `campus_walk.mp4` is at `/content/`

**Issue:** "BoxMOT not installed"  
**Solution:** `pip install boxmot`

**Issue:** "ReID weights not found"  
**Solution:** Weights auto-download on first run (need internet)

**Issue:** Low FPS (<20)  
**Solution:** Use lighter ReID model or disable it

---

## 📚 Documentation

- **Full Guide:** `TRACKING_REID_BENCHMARK_GUIDE.md`
- **BoxMOT Integration:** `BOXMOT_INTEGRATION_GUIDE.md`
- **Case Sensitivity Fix:** `snippets/BOXMOT_CASE_SENSITIVITY_FIX.md`

---

**Ready to run!** 🎬  
Command: `python test_tracking_reid_benchmark.py`
