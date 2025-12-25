# Tracking + ReID Benchmark Setup

**Date:** December 25, 2024  
**Goal:** Benchmark BoxMOT tracking with ReID on campus_walk.mp4  
**Baseline:** Raw YOLO detection at ~70 FPS

---

## Setup Summary

### Video Input
- **Path:** `/content/campus_walk.mp4`
- **Content:** Multiple people and objects in campus scene
- **Objective:** Track only humans, measure ID stability

### Configuration Selected

**Tracker:** ByteTrack (fastest: 1265 Hz motion-only baseline)
- Chosen for speed benchmark
- Will show maximum ReID overhead impact

**ReID Model:** OSNet x1.0 MSMT17
- Better accuracy than OSNet x0.25
- ~2M parameters (medium weight)
- Trained on MSMT17 dataset (diverse scenarios)

**Detection:** YOLOv8s
- Person class only (filter out other objects)
- Confidence threshold: 0.3
- Device: CUDA

---

## Files Created/Modified

### 1. Enhanced Tracking Script
**File:** `snippets/run_detector_tracking.py` (updated)

**New Features:**
- ✅ Per-frame timing (detection + tracking separately)
- ✅ Track ID storage and analysis
- ✅ Unique track ID counting
- ✅ Visualization with colored bboxes per track ID
- ✅ FPS breakdown (Detection / Tracking / Combined / Overall)
- ✅ Tracking statistics reporting

**Key Functions Added:**
```python
draw_tracks(frame, tracked_bboxes, show_conf=False)
    # Draws bboxes with track IDs, color-coded per ID
    
save_visualization(video_path, detections_data, output_path)
    # Generates annotated video with track IDs
```

---

### 2. Benchmark Configuration
**File:** `configs/detector_tracking_benchmark.yaml`

**Key Settings:**
```yaml
tracking:
  enabled: true
  tracker: bytetrack  # Fastest (1265 Hz)
  reid:
    enabled: true  # Enable appearance matching
    weights_path: models/reid/osnet_x1_0_msmt17.pt

input:
  video_path: /content/campus_walk.mp4
  max_frames: 0  # Process all frames

output:
  save_visualization: true
  visualization_path: demo_data/outputs/campus_walk_tracking_reid.mp4
```

---

### 3. Test Script
**File:** `snippets/test_tracking_reid_benchmark.py`

**What it does:**
1. Verifies BoxMOT installation
2. Checks video file exists
3. Runs tracking benchmark
4. Verifies outputs (NPZ + video)
5. Reports key metrics

---

## Running the Benchmark

### Quick Test (Recommended):
```bash
python snippets/test_tracking_reid_benchmark.py
```

### Manual Run:
```bash
python snippets/run_detector_tracking.py --config configs/detector_tracking_benchmark.yaml
```

---

## Expected Output

### Console Output:
```
======================================================================
Video: /content/campus_walk.mp4
Resolution: 1920x1080 @ 30.00 fps
Total frames: 900
Detector: YOLO
Tracking: BYTETRACK (ReID: ON)
Output mode: Largest tracked bbox per frame
======================================================================

Processing: 100%|████████████████████| 900/900 [00:30<00:00, 30.00it/s]

✓ Processing complete!
  Total frames processed: 900
  Frames with valid output: 895
  Success rate: 99.4%

⏱️  Performance Breakdown:
  Detection FPS: 70.5 (14.2ms/frame)
  Tracking FPS:  150.3 (6.7ms/frame)
  Combined FPS:  42.8 (23.4ms/frame)
  Overall FPS:   30.0 (including I/O)
  Time taken: 30.00s

📊 Tracking Statistics:
  Unique track IDs: 12
  Track IDs seen: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
```

### Files Generated:

**1. Detections NPZ:**
- `demo_data/outputs/detections_tracking_reid.npz`
- Format: Same as pipeline standard
- Contains: `frame_numbers`, `bboxes` (largest track per frame)

**2. Visualization Video:**
- `demo_data/outputs/campus_walk_tracking_reid.mp4`
- Shows: Colored bboxes with track IDs
- Displays: Frame number and active track count
- Purpose: Visual verification of ID stability

---

## Interpretation Guide

### FPS Metrics Explained:

**Detection FPS:** ~70 (Raw YOLO detection speed)
- This should match baseline (no overhead from tracking)

**Tracking FPS:** ~150 (ByteTrack processing speed)
- ByteTrack is very fast even with multiple people

**ReID Overhead:**
- Implicit in tracking time
- OSNet x1.0 inference: ~20-30ms per person
- More people = more overhead

**Combined FPS:** ~40 (Detection + Tracking + ReID)
- **Expected drop:** 70 → 40 FPS (43% slower)
- This is the actual processing speed with ReID

**Overall FPS:** ~30 (Including video I/O)
- Real-world throughput
- Includes frame reading/writing

---

### Track ID Stability:

**Without ReID (motion-only):**
- ID switches during occlusions
- New IDs assigned when person reappears
- Higher total unique IDs (ID fragmentation)

**With ReID (appearance matching):**
- Same ID maintained during occlusions
- Person re-identified after disappearance
- Lower total unique IDs (ID consistency)

**Example:**
```
# Without ReID - person walks behind tree
Frame 100: ID=1
Frame 120: ID lost (occluded)
Frame 140: ID=5 (re-appeared, new ID assigned)
Total IDs: 2 for same person

# With ReID - same scenario
Frame 100: ID=1
Frame 120: ID lost (occluded)
Frame 140: ID=1 (re-appeared, same ID matched by appearance)
Total IDs: 1 for same person
```

---

## Comparison Tests

### Test 1: Detection-Only (Baseline)
**Config:** Set `tracking.enabled: false` in YAML

**Expected:**
- FPS: ~70 (no tracking overhead)
- Bbox jitter: Higher (frame-to-frame variation)
- No track IDs

### Test 2: Motion-Only Tracking
**Config:** Set `reid.enabled: false`

**Expected:**
- FPS: ~65 (slight tracking overhead)
- Track IDs: Present but may switch during occlusions
- Faster than with ReID

### Test 3: Tracking + ReID (Current)
**Config:** Both tracking and ReID enabled

**Expected:**
- FPS: ~40 (ReID overhead significant)
- Track IDs: Stable across occlusions
- Best for identity consistency

---

## Visual Analysis Checklist

When watching `campus_walk_tracking_reid.mp4`:

✅ **Check for:**
1. Each person has consistent colored bbox
2. Track IDs remain same across frames
3. IDs don't switch when people cross paths
4. IDs maintained after brief occlusions
5. New people get new unique IDs
6. No flickering bboxes (smooth tracking)

❌ **Red flags:**
1. Track IDs changing frequently (ReID not working)
2. Multiple IDs for same person (ID fragmentation)
3. Missing bboxes where people visible (detection miss)
4. Bboxes jittering heavily (unstable tracking)

---

## Troubleshooting

### Issue: ReID Model Not Found
**Symptom:** Warning about ReID weights not found
**Solution:** 
- Model will auto-download on first run
- Check internet connection
- Verify path: `models/reid/osnet_x1_0_msmt17.pt`

### Issue: Low FPS (<20)
**Possible Causes:**
- Too many people in scene (ReID overhead per person)
- CPU fallback (check CUDA availability)
- High resolution video (resize input)

**Solutions:**
- Use lighter ReID: `osnet_x0_25_msmt17.pt`
- Reduce video resolution
- Use motion-only tracking (disable ReID)

### Issue: ID Switches Still Happening
**Possible Causes:**
- Very long occlusions (exceeds tracker memory)
- Significant appearance change (lighting, angle)
- Similar looking people (ReID confusion)

**Solutions:**
- Try different tracker: `botsort` or `boosttrack`
- Adjust tracker parameters
- Use higher quality ReID model

---

## Performance Targets

| Metric | Detection-Only | Motion Tracking | Tracking+ReID |
|--------|---------------|-----------------|---------------|
| **FPS** | 70 | 65 | 40 |
| **ID Stability** | N/A | Medium | High |
| **Occlusion Handling** | N/A | Poor | Good |
| **Use Case** | Speed critical | Balanced | Quality critical |

---

## Next Steps

1. ✅ **Run benchmark** and collect metrics
2. **Analyze visualization** for ID stability
3. **Compare trackers:**
   - ByteTrack (fastest: 1265 Hz)
   - BotSort (best accuracy: HOTA 69.4)
   - BoostTrack (best IDF1: 83.2)
4. **Test ReID models:**
   - OSNet x0.25 (lighter, faster)
   - OSNet x1.0 (current, balanced)
   - MobileNetV2 x1.4 (heavier, better)
5. **Optimize if needed:**
   - Export ReID to ONNX for speed
   - Adjust confidence thresholds
   - Tune tracker hyperparameters

---

## References

- **BoxMOT Documentation:** https://github.com/mikel-brostrom/boxmot
- **Tracker Benchmark (MOT17):** `BOXMOT_INTEGRATION_GUIDE.md`
- **Case Sensitivity Fix:** `snippets/BOXMOT_CASE_SENSITIVITY_FIX.md`
- **ReID Model Zoo:** https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO

---

**Status:** Ready to run benchmark! 🚀
**Command:** `python snippets/test_tracking_reid_benchmark.py`
