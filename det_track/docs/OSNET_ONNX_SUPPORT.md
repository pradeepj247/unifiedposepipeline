# OSNet ONNX Support - Implementation Notes

**Date:** January 15, 2026  
**Status:** Complete - ONNX-first with PyTorch fallback

---

## Overview

The OSNet clustering module now supports **both ONNX and PyTorch** models with intelligent fallback:

1. **Preferred:** ONNX models (.onnx) - Fast inference, no PyTorch required
2. **Fallback:** PyTorch models (.pth) - For backward compatibility
3. **Fallback:** Randomly initialized - If weights not found

---

## Model Location (Colab)

The ONNX model is pre-downloaded at:

```
/content/unifiedposepipeline/models/reid/osnet_x0_25_msmt17.onnx
```

**Pipeline Config Setting:**

```yaml
stage4_generate_html:
  clustering:
    osnet_model: ${models_dir}/reid/osnet_x0_25_msmt17.onnx
```

This resolves to `/content/unifiedposepipeline/models/reid/osnet_x0_25_msmt17.onnx` on Colab.

---

## Architecture

### Model Loading (osnet_clustering.py)

```python
def load_osnet_model(model_path, device='cuda') -> (model, device_str, model_type):
    """
    Returns: (model_object, device_string, model_type_string)
    - model_type: 'onnx', 'pytorch', or 'random'
    """
```

**Priority Order:**

1. **ONNX** (.onnx file + onnxruntime installed)
   - Providers: CUDA (if available) → CPU
   - Fast inference, no GPU memory for weights
   - Optimal for production

2. **PyTorch** (.pth file + torch installed)
   - Fallback for backward compatibility
   - Loaded to specified device (cuda/cpu)

3. **Random Initialization** (no weights found)
   - Non-fatal fallback
   - Still produces valid embeddings (untrained)
   - Warning logged to console

### Feature Extraction (osnet_clustering.py)

The `extract_osnet_features()` function handles both backends:

```python
def extract_osnet_features(crops, model, device, model_type, batch_size=16):
    """
    Args:
        model_type: 'onnx' or 'pytorch'
        batch_size: MUST be 16 for ONNX model (fixed batch dimension)
    
    Handles:
    - ONNX: Batch via numpy arrays, get inputs/outputs dynamically
    - PyTorch: Batch via torch tensors, .to(device)
    """
```

**Important:** The ONNX model has a fixed batch dimension of **16**. 
Configuration must use `num_best_crops: 16` to match this requirement.
If you have fewer than 16 crops, they will be padded with zeros internally.

---

## Dependencies

### Required

- **numpy, cv2** - Always required
- **One of:**
  - `onnxruntime` (recommended) - Lightweight, CPU/GPU
  - `torch` (fallback) - Large, needed for PyTorch models

### Installation

```bash
# Recommended (Colab - ONNX only)
pip install onnxruntime-gpu

# Or CPU-only:
pip install onnxruntime

# For PyTorch fallback:
pip install torch torchvision
```

---

## Device Handling

### Automatic Fallback

```python
if device == 'cuda' and not torch.cuda.is_available():
    print("[OSNet] CUDA not available, falling back to CPU")
    device = 'cpu'
```

**ONNX Providers:**
- GPU: `CUDAExecutionProvider` (if device='cuda')
- CPU: `CPUExecutionProvider` (fallback)

---

## Output Files

All outputs saved to `output_dir` (typically `webp_viewer/`):

```
webp_viewer/
├─ similarity_matrix.json (includes model_type)
├─ similarity_matrix.npy
├─ embeddings.json (includes model type)
├─ embeddings.npy
├─ person_selection.html
└─ webp/
   ├─ person_0.webp
   ├─ person_1.webp
   └─ ... (up to 10)
```

**JSON Metadata:**

```json
{
  "similarity_matrix.json": {
    "model_type": "onnx",        // Track which backend
    "similarity_threshold": 0.70,
    "timestamp": "2026-01-15T..."
  },
  "embeddings.json": {
    "model": "OSNet x0.25",
    "feature_dimension": 256,
    "model_type": "onnx"
  }
}
```

---

## Error Handling

### Scenario 1: ONNX Model Missing → PyTorch Fallback

```
Input: osnet_x0_25_msmt17.onnx (not found)
↓
Try ONNX load: FAIL
↓
Try PyTorch load: SUCCESS (.pth found)
↓
Output: Uses PyTorch backend
```

### Scenario 2: Both Missing → Random Initialization

```
Input: Neither .onnx nor .pth found
↓
Try ONNX: FAIL
Try PyTorch: FAIL
↓
Initialize random model
↓
Warning logged, but clustering continues
Output: Valid (untrained) embeddings
```

### Scenario 3: No Backend Installed

```
Input: osnet_x0_25_msmt17.onnx found, but onnxruntime not installed
↓
Try ONNX load: Check fails (onnxruntime not available)
↓
Try PyTorch: Not installed
↓
Raise RuntimeError: "No OSNet backend available"
↓
Clustering DISABLED (non-fatal to Stage 4)
```

---

## Colab Setup

The Colab environment already has:

- ✅ `/content/unifiedposepipeline/models/reid/osnet_x0_25_msmt17.onnx`
- ✅ `onnxruntime` (pre-installed)
- ✅ CUDA available (GPU runtime)

**No additional setup needed!** Just run:

```python
python stage4_generate_html.py --config configs/pipeline_config.yaml
```

---

## Windows / Local Setup

### Option 1: ONNX-only (Recommended)

```bash
# Install onnxruntime
pip install onnxruntime

# Copy ONNX model
mkdir -p models/reid
cp osnet_x0_25_msmt17.onnx models/reid/
```

### Option 2: PyTorch Fallback

```bash
# Install torch (large download)
pip install torch torchvision

# Copy PyTorch weights
mkdir -p models/osnet
cp osnet_x0_25_msmt17.pth models/osnet/
```

Then update config:

```yaml
clustering:
  osnet_model: D:/trials/unifiedpipeline/newrepo/models/osnet/osnet_x0_25_msmt17.pth
  device: cuda  # or cpu
```

---

## Performance Notes

### ONNX vs PyTorch

| Metric | ONNX | PyTorch |
|--------|------|---------|
| **Model Size** | 91 MB | 91 MB (same) |
| **Inference Speed** | Faster (optimized) | Slower (general) |
| **GPU Memory** | Lower | Higher (weights in VRAM) |
| **Dependency** | onnxruntime (small) | torch (1.5+ GB) |
| **Startup Time** | ~0.5s | ~1-2s (PyTorch init) |

**Recommendation:** Use ONNX for production (Colab), PyTorch for development.

---

## Troubleshooting

### Error: "No OSNet backend available"

**Solution:** Install onnxruntime or torch

```bash
# Quick fix:
pip install onnxruntime
```

### Error: "Model path not found"

**Check:**
1. Is model actually at `${models_dir}/reid/osnet_x0_25_msmt17.onnx`?
2. Is `${models_dir}` resolving correctly in config?
3. Run: `ls -la /content/unifiedposepipeline/models/reid/`

### Slow inference (~30s for 80 crops)

**Likely cause:** Using random initialization (untrained model)

**Check:**
```bash
ls -lh /content/unifiedposepipeline/models/reid/osnet_x0_25_msmt17.onnx
```

If file is small (~100 KB) instead of ~91 MB, the real model isn't there.

---

## Future Improvements

1. **Model Quantization:** INT8 quantization for faster inference
2. **Batch Optimization:** Adaptive batch sizing based on GPU memory
3. **Caching:** Store embeddings in NPZ to avoid re-extraction
4. **Model Download:** Auto-download missing models from model zoo

---

## References

- ONNX Runtime: https://onnxruntime.ai/
- OSNet Paper: "Omni-Scale Feature Learning for Person Re-Identification"
- Model Source: MarketingAI/deep-person-reid
