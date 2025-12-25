# File Organization Update - Tracking Scripts Moved to Main Directory

**Date:** December 25, 2024  
**Reason:** Snippets directory not available on Google Colab  
**Action:** Moved tracking scripts from `snippets/` to main directory

---

## Files Moved

### 1. run_detector_tracking.py
**Old location:** `snippets/run_detector_tracking.py` (local only)  
**New location:** `run_detector_tracking.py` (main directory, in repo)

**Purpose:** Main tracking script with visualization
- Detection + Tracking (7 trackers available)
- ReID support (appearance-based tracking)
- Visualization with track IDs
- FPS benchmarking

**Usage on Colab:**
```bash
cd /content/unifiedposepipeline
python run_detector_tracking.py --config configs/detector_tracking_benchmark.yaml
```

---

### 2. test_tracking_reid_benchmark.py
**Old location:** `snippets/test_tracking_reid_benchmark.py` (local only)  
**New location:** `test_tracking_reid_benchmark.py` (main directory, in repo)

**Purpose:** Automated test script
- Verifies BoxMOT installation
- Checks video file
- Runs benchmark
- Reports metrics

**Usage on Colab:**
```bash
cd /content/unifiedposepipeline
python test_tracking_reid_benchmark.py
```

---

## Files That Remain in Snippets (Local Only)

These files are for local development/debugging and won't be pushed to GitHub:

1. `snippets/run_detector_tracking.py` (original copy, kept for reference)
2. `snippets/test_tracking_reid_benchmark.py` (original copy)
3. `snippets/explore_boxmot_api.py` (API exploration tool)
4. `snippets/test_boxmot_import.py` (Import testing tool)
5. `snippets/BOXMOT_CASE_SENSITIVITY_FIX.md` (debugging notes)

---

## Path Updates Made

### In test_tracking_reid_benchmark.py:
**Before:**
```python
result = subprocess.run(
    [sys.executable, "snippets/run_detector_tracking.py", "--config", config_path],
    capture_output=False
)
```

**After:**
```python
result = subprocess.run(
    [sys.executable, "run_detector_tracking.py", "--config", config_path],
    capture_output=False
)
```

---

## Documentation Updates

### New Files in Main Directory:
1. **`TRACKING_BENCHMARK_QUICKSTART.md`** - Quick start guide for Colab
2. **`TRACKING_REID_BENCHMARK_GUIDE.md`** - Comprehensive benchmark guide
3. **`BOXMOT_INTEGRATION_GUIDE.md`** - BoxMOT integration documentation

---

## Repository Structure (Relevant Files)

```
unifiedposepipeline/
├── run_detector_tracking.py          ← NEW: Main tracking script (moved from snippets)
├── test_tracking_reid_benchmark.py   ← NEW: Test script (moved from snippets)
├── configs/
│   ├── detector.yaml                 ← Existing detector config
│   └── detector_tracking_benchmark.yaml  ← NEW: Benchmark config
├── TRACKING_BENCHMARK_QUICKSTART.md  ← NEW: Quick start guide
├── TRACKING_REID_BENCHMARK_GUIDE.md  ← NEW: Full guide
├── BOXMOT_INTEGRATION_GUIDE.md       ← NEW: BoxMOT docs
└── snippets/                         ← Local development only (not in repo)
    ├── run_detector_tracking.py      (original copy)
    ├── test_tracking_reid_benchmark.py (original copy)
    ├── explore_boxmot_api.py
    └── test_boxmot_import.py
```

---

## .gitignore Update

The `snippets/` directory should already be in `.gitignore`:

```gitignore
# Local development/debug scripts
snippets/
```

This ensures local debugging files don't get committed to the repository.

---

## Running on Google Colab

### Setup (One-time):
```bash
cd /content/unifiedposepipeline
pip install boxmot
```

### Quick Test:
```bash
python test_tracking_reid_benchmark.py
```

### Manual Run:
```bash
python run_detector_tracking.py --config configs/detector_tracking_benchmark.yaml
```

---

## Verification

**Check files exist on Colab:**
```bash
cd /content/unifiedposepipeline
ls -la run_detector_tracking.py
ls -la test_tracking_reid_benchmark.py
ls -la configs/detector_tracking_benchmark.yaml
```

**All should return:** `-rw-r--r-- 1 root root <size> <date> <filename>`

---

## Next Steps

1. ✅ Push updated files to GitHub (run_detector_tracking.py, test script, configs, docs)
2. ✅ Pull on Colab: `git pull origin main`
3. ✅ Run test: `python test_tracking_reid_benchmark.py`
4. ✅ Watch output video to verify tracking + ReID

---

**Status:** Files organized for Colab compatibility ✅  
**Ready to:** Push to GitHub and test on Colab 🚀
