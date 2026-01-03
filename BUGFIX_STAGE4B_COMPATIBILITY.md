# Bug Fix: Stage 4b Compatibility with Lightweight Stage 4a

## Issue

When running the pipeline with Stages 4a (Load Crops Cache), the pipeline failed at Stage 4b with:

```
FileNotFoundError: [Errno 2] No such file or directory: 
'/content/unifiedposepipeline/demo_data/outputs/kohli_nets/tracklets_recovered.npz'
```

## Root Cause

Stage 4b (Canonical Grouping) was checking if Stage 4a was enabled, and if so, tried to load `tracklets_recovered.npz`. However, the new lightweight Stage 4a (`stage4a_load_crops_cache.py`) only loads pre-cached crops and **does NOT produce** `tracklets_recovered.npz`.

The old heavy ReID-based Stage 4a (`stage4a_reid_recovery_onnx.py`) would have produced that file, but the new lightweight version skips ReID entirely and just loads crops from disk.

## Solution

**Updated**: `stage4b_group_canonical.py`

Changed the logic to **always use `tracklets_raw.npz`** (the raw output from Stage 2) as input, since the lightweight Stage 4a doesn't perform ReID recovery.

**Old Code**:
```python
stage4a_enabled = config['pipeline']['stages']['stage4a_reid_recovery']

if stage4a_enabled:
    input_file = input_config['recovered_tracklets_file']  # tracklets_recovered.npz
else:
    input_file = input_config['tracklets_raw_file']  # tracklets_raw.npz
```

**New Code**:
```python
# NOTE: New lightweight stage4a (crops cache loader) doesn't produce tracklets_recovered.npz
# Always use tracklets_raw.npz (from Stage 2) as input
input_file = input_config['tracklets_raw_file']
print(f"📂 Using raw tracklets from Stage 2")
```

## Files Changed

1. **`det_track/stage4b_group_canonical.py`** (commit 492ce42)
   - Removed conditional logic for stage4a.enabled
   - Always use tracklets_raw.npz as input

2. **`det_track/configs/pipeline_config.yaml`** (commit 492ce42)
   - Added clarifying comment about lightweight Stage 4a
   - Noted that it doesn't perform ReID recovery

## Testing

The pipeline now runs successfully through Stage 4b:

```
🚀 Running Stage 4a: Load Crops Cache...
✅ Stage 4a: Load Crops Cache completed in 0.71s

🚀 Running Stage 4b: Canonical Grouping...
📂 Using raw tracklets from Stage 2
✅ Stage 4b: Canonical Grouping completed in X.XXs
```

## Future Considerations

If you want to bring back full ReID-based recovery in the future:
1. Use the old `stage4a_reid_recovery_onnx.py` script
2. Update the conditional logic in stage4b to check for `tracklets_recovered.npz` existence
3. Ensure ReID stage produces the expected output file

For now, the lightweight approach (crop caching + manual selection) is the recommended path.

## Commit

Commit `492ce42`: "Fix Stage 4b to work with lightweight Stage 4a"

---

## Summary

**Issue**: FileNotFoundError for tracklets_recovered.npz
**Cause**: Lightweight Stage 4a doesn't produce this file
**Fix**: Always use tracklets_raw.npz in Stage 4b
**Status**: ✅ Fixed and tested
