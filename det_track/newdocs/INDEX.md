# Detection & Tracking Pipeline Documentation Index

**Last Updated**: January 17, 2026 ✅ (LATEST)

## 📚 Complete Documentation Set

This directory contains comprehensive documentation for the unified detection & tracking pipeline. Start with the appropriate entry point based on your needs.

---

## 🚀 Quick Start (5 minutes)

**For new users or quick reference:**
- 📄 **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** ← Start here
  - One-minute overview
  - Essential commands
  - Common troubleshooting
  - Cheat sheet format

---

## 📖 Complete Understanding (30 minutes)

**For learning the full architecture:**
1. 📄 **[README_MASTER.md](README_MASTER.md)** - Master overview
   - Core objective & pipeline flow
   - Execution modes (fast/balanced/full)
   - Performance summary
   - Key design decisions

2. 📄 **[PIPELINE_CONFIG_REFERENCE.md](PIPELINE_CONFIG_REFERENCE.md)** - Configuration guide
   - All YAML settings explained
   - Path resolution (`${variable}` syntax)
   - Stage-specific parameters
   - Common configuration changes

3. 📄 **[RUN_PIPELINE_EXECUTION.md](RUN_PIPELINE_EXECUTION.md)** - Pipeline orchestrator
   - How `run_pipeline.py` works
   - Command-line arguments
   - Stage execution flow
   - Error handling & debugging

---

## 🔬 Stage-by-Stage Deep Dives (5 minutes each)

**For understanding individual stages:**

| Stage | File | Purpose |
|-------|------|---------|
| **0** | [STAGE0_VIDEO_VALIDATION.md](STAGE0_VIDEO_VALIDATION.md) | Video normalization & format validation |
| **1** | [STAGE1_DETECTION.md](STAGE1_DETECTION.md) | YOLO detection + eager crop extraction (48.75s) |
| **2** | [STAGE2_TRACKING.md](STAGE2_TRACKING.md) | ByteTrack offline identity tracking (7.91s) |
| **3a** | [STAGE3A_ANALYSIS.md](STAGE3A_ANALYSIS.md) | Tracklet statistics & ranking (0.23s) |
| **3b** | [STAGE3B_GROUPING.md](STAGE3B_GROUPING.md) | Canonical person grouping (0.47s) |
| **3c** | [STAGE3C_FILTER_PERSONS.md](STAGE3C_FILTER_PERSONS.md) | Filter persons & 3-bin crop selection (0.95s) |
| **3d** | [STAGE3D_VISUAL_REFINEMENT.md](STAGE3D_VISUAL_REFINEMENT.md) | OSNet visual ReID matching (optional, 8-12s) |
| **4** | [STAGE4_HTML_GENERATION.md](STAGE4_HTML_GENERATION.md) | WebP + HTML viewer generation (2.51s) |

Each stage doc includes:
- ✅ Purpose & when it runs
- ✅ Inputs & outputs
- ✅ Processing pipeline (visual flow)
- ✅ Performance metrics
- ✅ Key design decisions & rationale
- ✅ Configuration options
- ✅ Data format specifications
- ✅ Related cross-references

---

## 🎯 Use Cases

### "I just want to run the pipeline"
→ Go to [QUICK_REFERENCE.md](QUICK_REFERENCE.md) → copy-paste commands

### "It's running slow, how do I speed it up?"
→ Read [README_MASTER.md](README_MASTER.md#performance-summary) performance section
→ See bottleneck analysis: Stage 1 (79.3% of time) is video I/O limited, not GPU limited

### "How do I process a different video?"
→ [QUICK_REFERENCE.md](QUICK_REFERENCE.md#change-video) → Change `video_file` in YAML

### "I want fast mode (60s) vs full mode (77s)"
→ [QUICK_REFERENCE.md](QUICK_REFERENCE.md#switch-mode) → Change `mode: fast` in YAML

### "I'm debugging Stage X, what are the inputs/outputs?"
→ Go to `STAGEX_*.md` file → See "Inputs" and "Outputs" sections

### "Why did YOLOv8n not speed things up?"
→ [STAGE1_DETECTION.md](STAGE1_DETECTION.md#1-yolov8s-model-selection) → Explains video decoding bottleneck

### "What's this detection_idx thing everyone mentions?"
→ [README_MASTER.md](README_MASTER.md#critical-linkage-detection_idx) → Data flow explanation

### "How does 3-bin contiguous selection work?"
→ [STAGE3C_FILTER_PERSONS.md](STAGE3C_FILTER_PERSONS.md#3-bin-contiguous-selection-algorithm) → Algorithm with example

### "Why is crops_cache automatically deleted?"
→ [STAGE3C_FILTER_PERSONS.md](STAGE3C_FILTER_PERSONS.md#why-delete-crops_cache-after) → Design rationale

### "I want to understand the full config file"
→ [PIPELINE_CONFIG_REFERENCE.md](PIPELINE_CONFIG_REFERENCE.md) → Every setting explained

---

## 📊 Documentation Organization

```
newdocs/
├── README_MASTER.md                    ← Master overview (start here)
├── QUICK_REFERENCE.md                  ← Cheat sheet (run commands)
├── INDEX.md                            ← This file
│
├── Configuration & Execution
│   ├── PIPELINE_CONFIG_REFERENCE.md   ← All YAML settings
│   └── RUN_PIPELINE_EXECUTION.md      ← How orchestrator works
│
└── Stage Documentation (0-4)
    ├── STAGE0_VIDEO_VALIDATION.md     ← Video normalization
    ├── STAGE1_DETECTION.md            ← YOLOv8 + eager crops
    ├── STAGE2_TRACKING.md             ← ByteTrack optimization
    ├── STAGE3A_ANALYSIS.md            ← Ranklet stats
    ├── STAGE3B_GROUPING.md            ← Canonical persons
    ├── STAGE3C_FILTER_PERSONS.md      ← Crop selection
    ├── STAGE3D_VISUAL_REFINEMENT.md   ← OSNet ReID
    └── STAGE4_HTML_GENERATION.md      ← WebP + HTML
```

---

## 🎓 Learning Path

### Path 1: "I just need to run it"
1. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) (5 min)
2. Run command from there
3. Done! ✅

### Path 2: "I want to understand what's happening"
1. [README_MASTER.md](README_MASTER.md) (15 min)
2. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) (5 min)
3. Pick 1-2 stage docs that interest you (10 min each)
4. Done! ✅

### Path 3: "I need to debug or modify the code"
1. [README_MASTER.md](README_MASTER.md) (15 min)
2. [RUN_PIPELINE_EXECUTION.md](RUN_PIPELINE_EXECUTION.md) (10 min)
3. [PIPELINE_CONFIG_REFERENCE.md](PIPELINE_CONFIG_REFERENCE.md) (10 min)
4. Relevant stage docs (10 min each, as needed)
5. Source code in `stage*.py` files
6. Done! ✅

### Path 4: "I'm optimizing performance"
1. [README_MASTER.md](README_MASTER.md#performance-summary) (5 min)
2. [STAGE1_DETECTION.md](STAGE1_DETECTION.md) (15 min) - where 79% time is spent
3. [STAGE2_TRACKING.md](STAGE2_TRACKING.md) (10 min) - optimization examples
4. [STAGE3C_FILTER_PERSONS.md](STAGE3C_FILTER_PERSONS.md) (10 min) - 11× speedup achieved
5. Done! ✅

---

## 🔑 Key Concepts Explained

| Concept | Where to Learn |
|---------|----------------|
| **detection_idx** | [README_MASTER.md - Data Flow](README_MASTER.md#critical-linkage-detection_idx) |
| **Eager extraction** | [STAGE1_DETECTION.md - Design Decision](STAGE1_DETECTION.md#2-eager-crop-extraction) |
| **3-bin selection** | [STAGE3C_FILTER_PERSONS.md - Algorithm](STAGE3C_FILTER_PERSONS.md#3-bin-contiguous-selection-algorithm) |
| **ByteTrack optimization** | [STAGE2_TRACKING.md - Optimizations](STAGE2_TRACKING.md#key-optimizations) |
| **Path resolution** | [PIPELINE_CONFIG_REFERENCE.md - Global Config](PIPELINE_CONFIG_REFERENCE.md#global-configuration) |
| **Mode selection** | [QUICK_REFERENCE.md - Config](QUICK_REFERENCE.md#switch-mode) |
| **O(1) lookup** | [STAGE3C_FILTER_PERSONS.md - Optimization](STAGE3C_FILTER_PERSONS.md#o1-lookup-optimization) |
| **Video bottleneck** | [STAGE1_DETECTION.md - YOLOv8s decision](STAGE1_DETECTION.md#1-yolov8s-model-selection) |
| **ReID merging** | [STAGE3D_VISUAL_REFINEMENT.md - OSNet](STAGE3D_VISUAL_REFINEMENT.md) |
| **WebP encoding** | [STAGE4_HTML_GENERATION.md - WebP](STAGE4_HTML_GENERATION.md#webp-encoding) |

---

## 📈 Performance at a Glance

```
Total: 60.24 seconds (2025 frames, 1920×1080)

Stage 1: 48.75s (79.3%) ████████████    Bottleneck: Video I/O
Stage 2:  7.91s (13.1%) ██              ByteTrack optimized
Stage 3c: 0.95s (1.5%)  ▌                11× faster via eager extraction
Stage 4:  2.51s (3.9%)  ▌                WebP generation
Stage 3a: 0.23s (0.4%)  ▌                Analysis
Stage 3b: 0.47s (0.7%)  ▌                Grouping
```

**Key Insight**: Video decoding (H.264 CPU) is the bottleneck at 53 FPS, not GPU inference.

---

## ✅ Quality Assurance

All documentation has been:
- ✅ Written on January 17, 2026
- ✅ Based on production-tested code
- ✅ Cross-referenced for consistency
- ✅ Includes performance metrics & timing
- ✅ Explains design decisions & rationale
- ✅ Tested on Google Colab + Windows

---

## 🔗 External References

- **GitHub Repository**: Code available in `det_track/` directory
- **Requirements**: See project README for dependencies
- **Related Projects**: 
  - ViTPose: Pose estimation backbone
  - RTMPose: Alternative pose model
  - ByteTrack: Multi-object tracking
  - OSNet: Person ReID

---

## 📝 Document Metadata

| Property | Value |
|----------|-------|
| **Version** | 1.0 |
| **Last Updated** | January 17, 2026 |
| **Status** | ✅ Production Ready |
| **Coverage** | Stages 0-4 (detection/tracking/HTML) |
| **Total Pages** | 30+ (this index + 8 stage docs + 3 reference docs) |
| **Example Videos** | kohli_nets.mp4 (2025 frames, 81s) |
| **Tested On** | Google Colab (T4 GPU) + Windows (RTX GPU) |

---

## 🚀 Getting Started Now

**Fastest path (< 2 minutes):**
1. Open [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
2. Copy command: `python run_pipeline.py --config configs/pipeline_config.yaml`
3. Paste into Colab terminal
4. Wait 60 seconds
5. View output: `person_selection_slideshow.html`

Done! ✅

---

**Questions or issues?**  
Consult the appropriate doc based on your question:
- Running? → [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- Understanding? → [README_MASTER.md](README_MASTER.md)
- Debugging? → Relevant stage doc
- Configuring? → [PIPELINE_CONFIG_REFERENCE.md](PIPELINE_CONFIG_REFERENCE.md)
- Performance? → [STAGE1_DETECTION.md](STAGE1_DETECTION.md) or [STAGE2_TRACKING.md](STAGE2_TRACKING.md)

---

**Happy detecting! 🎉**

Generated: January 17, 2026  
Status: Production Ready ✅
