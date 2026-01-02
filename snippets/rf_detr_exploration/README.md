# RF-DETR Exploration

Standalone project to evaluate [Roboflow's RF-DETR](https://github.com/roboflow/rf-detr) for potential integration into our unified pose estimation pipeline.

## 🎯 Purpose

Test RF-DETR as an alternative to YOLO for person detection in our pipeline:
- **Speed**: Compare inference performance
- **Accuracy**: Evaluate detection quality for pose estimation use cases
- **Integration**: Assess ease of adoption into existing pipeline
- **Benefits**: Identify advantages over current YOLO setup

## 📋 Contents

- `test_rf_detr.ipynb` - Complete Colab notebook for evaluation
- This README

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. Upload `test_rf_detr.ipynb` to Google Colab
2. Enable GPU runtime: `Runtime` → `Change runtime type` → `GPU`
3. Run cells sequentially
4. Upload your test video when prompted

### Option 2: Local Setup

```bash
# Clone RF-DETR
git clone https://github.com/roboflow/rf-detr.git
cd rf-detr

# Install dependencies
pip install -r requirements.txt

# Use the notebook or adapt code snippets
```

## 📊 Evaluation Checklist

### Performance Metrics
- [ ] Inference speed (ms/frame, FPS)
- [ ] GPU memory usage
- [ ] CPU usage (if CPU mode available)
- [ ] Comparison with YOLO (yolov8s)

### Detection Quality
- [ ] Person detection accuracy
- [ ] False positive rate
- [ ] False negative rate
- [ ] Consistency across frames
- [ ] Performance in crowded scenes
- [ ] Performance with occlusions

### Integration Considerations
- [ ] API compatibility with our pipeline
- [ ] Output format compatibility
- [ ] Model size and loading time
- [ ] Dependency conflicts
- [ ] License compatibility
- [ ] Community support and maintenance

## 🎬 Test Scenarios

Recommended test videos:
1. **Single person**: Simple tracking scenario (like kohli_nets.mp4)
2. **Multiple persons**: Crowded scene with 5-10 people
3. **Occlusions**: People partially hidden or overlapping
4. **Fast motion**: Sports or dance video
5. **Scale variation**: People at different distances

## 📈 Expected Outputs

The notebook generates:
- Detection videos with bounding boxes
- Performance statistics (FPS, inference time)
- Detection data in NPZ format (pipeline-compatible)
- Comparison charts (RF-DETR vs YOLO)
- Summary report with recommendations

## 🔄 Integration Path (If Approved)

If RF-DETR proves beneficial:

1. **Create detector wrapper**: `det_track/detectors/rf_detr_detector.py`
2. **Update config**: Add `rf-detr` as detection backend option
3. **Test end-to-end**: Run full pipeline with RF-DETR
4. **Benchmark**: Compare complete pipeline performance
5. **Documentation**: Update main README with RF-DETR instructions
6. **Release**: Merge into main pipeline

## 📝 Notes

- This is a **temporary exploration** folder
- Code here is **experimental** and not production-ready
- Will be removed after evaluation (keep only learnings)
- All findings should be documented before cleanup

## 🤔 Decision Criteria

**Integrate if:**
- ✅ Faster than YOLO with comparable accuracy
- ✅ Better detection quality for pose estimation
- ✅ Easy integration with minimal dependencies
- ✅ Active maintenance and good documentation

**Don't integrate if:**
- ❌ Slower or significantly more memory intensive
- ❌ Lower detection quality
- ❌ Complex integration or dependency conflicts
- ❌ Licensing issues or maintenance concerns

## 🔗 References

- [RF-DETR Repository](https://github.com/roboflow/rf-detr)
- [Roboflow Blog Post](https://blog.roboflow.com/) (check for RF-DETR announcements)
- [DETR Paper](https://arxiv.org/abs/2005.12872) (original DETR)
- Our pipeline: `det_track/README.md`

---

**Status**: 🔬 Experimental  
**Created**: 2025-12-31  
**Maintainer**: Pipeline team
