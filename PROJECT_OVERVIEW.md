# Unified Pose Pipeline - Project Overview & Cold Start Guide

**Last Updated**: January 17, 2026  
**Status**: Detection/Tracking Complete ✅ | Pose Pipeline Active 🔄 | Ready for Cleanup & Refinement

---

## 🎯 Project Goal

Build a **unified pose estimation pipeline** that processes sports videos to extract biomechanical analytics for coaching and performance analysis. The pipeline handles the complete workflow:

1. **Detect & Track**: Identify and track all persons in a video
2. **Select**: User selects the target athlete visually
3. **Extract**: Get 2D and 3D pose keypoints for the selected person
4. **Analyze**: Derive biomechanics (joint angles, COM, forces, etc.)

**Target Use Case**: Sports coaching, movement analysis, performance optimization

---

## 📊 Pipeline Stages Overview

### Stage Group 1: Detection, Tracking & Selection (COMPLETE ✅)
Detailed documentation: [`det_track/newdocs/`](det_track/newdocs/)

```
Video Input
    ↓
[Stage 0] Video Normalization (H.264, GOP=30, ≤1080p)
    ↓
[Stage 1] YOLO Detection (YOLOv8s) + Eager Crop Extraction
    ↓
[Stage 2] ByteTrack Offline Tracking
    ↓
[Stage 3] Multi-Stage Analysis & Refinement
    ├─ Stage 3a: Tracklet Statistics
    ├─ Stage 3b: Canonical Grouping (67 tracklets → 40+ persons)
    ├─ Stage 3c: Filter & Crop Selection (40+ → 8-10 persons)
    └─ Stage 3d: OSNet Visual ReID (8-10 → 7-8 persons, optional)
    ↓
[Stage 4] HTML Viewer Generation (WebP animations)
    ↓
[Stage 5] Person Selection (User picks person_id)
    ↓
Output: selected_person.npz (bboxes for target athlete)
```

**Documentation**: See [`det_track/newdocs/README_MASTER.md`](det_track/newdocs/README_MASTER.md) for complete details.

**Performance**: ~60 seconds for 2025 frames (1920×1080) on T4 GPU

### Stage Group 2: Pose Estimation & 3D Lifting (ACTIVE 🔄)
Current implementation files (need cleanup):

- **`udp_video.py`**: Main unified detection pipeline (2D pose)
- **`run_posedet.py`**: Pose detection runner
- **`run_detector.py`**: Detector runner
- **`test_detector.py`**: Detector testing
- **`vis_wb3d.py`**: 3D wholebody visualization
- **`udp_3d_lifting.py`**: 3D lifting pipeline

**Models Configured** (see [`setup/models.yaml`](setup/models.yaml)):
- **2D Pose**: RTMPose (COCO/Halpe26), ViTPose-B, RTM Wholebody 3D
- **3D Lifting**: MotionAGFormer
- **Detection**: YOLOv8s (PyTorch + TensorRT)
- **ReID**: OSNet x0.25, OSNet x1.0 (PyTorch + ONNX)

### Stage Group 3: Biomechanics Analysis (PLANNED 🔜)
- Joint angle computation
- Center of mass tracking
- Ground reaction force estimation
- Movement quality metrics

---

## 🗂️ Repository Structure

### Local Development (Windows)
```
D:\trials\unifiedpipeline\newrepo\              # Main repository root
├── det_track/                                  # Detection & tracking pipeline (COMPLETE)
│   ├── stage0_normalize_video.py              # Video normalization
│   ├── stage1_detect.py                       # YOLO detection + crop extraction
│   ├── stage2_track.py                        # ByteTrack tracking
│   ├── stage3a_analyze_tracklets.py           # Tracklet statistics
│   ├── stage3b_group_canonical.py             # Canonical grouping
│   ├── stage3c_filter_persons.py              # Person filtering + crop selection
│   ├── stage3d_refine_visual.py               # OSNet ReID merging
│   ├── stage4_generate_html.py                # HTML viewer generation
│   ├── stage5_extract_person.py               # Person bbox extraction (to implement)
│   ├── run_pipeline.py                        # Orchestrator script
│   ├── configs/
│   │   └── pipeline_config.yaml               # All stage configurations
│   ├── newdocs/                               # 📚 COMPREHENSIVE DOCUMENTATION
│   │   ├── README_MASTER.md                   # Start here!
│   │   ├── STAGE0_VIDEO_VALIDATION.md
│   │   ├── STAGE1_DETECTION.md
│   │   ├── STAGE2_TRACKING.md
│   │   ├── STAGE3A_ANALYSIS.md
│   │   ├── STAGE3B_GROUPING.md
│   │   ├── STAGE3C_FILTER_PERSONS.md
│   │   ├── STAGE3D_VISUAL_REFINEMENT.md
│   │   ├── STAGE4_HTML_GENERATION.md
│   │   ├── STAGE5_PERSON_SELECTION.md
│   │   ├── PIPELINE_CONFIG_REFERENCE.md
│   │   └── RUN_PIPELINE_EXECUTION.md
│   └── outputs/                               # Pipeline outputs (gitignored)
│
├── setup/                                      # Environment setup scripts
│   ├── step1_install_packages.py              # Install Python dependencies
│   ├── step2_fetch_models.py                  # Download model weights
│   ├── step3_fetch_demodata.py                # Download demo videos
│   ├── step4_verify_envt.py                   # Verify installation
│   ├── models.yaml                            # Model configuration (all sources)
│   └── setup_utils.py                         # Shared utilities
│
├── configs/                                    # Global configs (pose estimation)
│   ├── udp_video.yaml
│   └── udp_image.yaml
│
├── lib/                                        # Core libraries
│   ├── vitpose/                               # ViTPose implementation
│   └── ...
│
├── utils/                                      # Shared utilities
│   ├── logger.py                              # Pipeline logging
│   └── ...
│
├── demo_data/                                  # Demo videos & images
│   ├── videos/
│   │   └── dance.mp4
│   └── images/
│       └── sample.jpg
│
├── snippets/                                   # Temporary debug code (NOT committed)
│
├── udp_video.py                                # 🔄 Main pose pipeline (needs cleanup)
├── run_posedet.py                              # 🔄 Pose detection runner
├── run_detector.py                             # 🔄 Detector runner
├── test_detector.py                            # 🔄 Testing script
├── vis_wb3d.py                                 # 🔄 3D wholebody visualization
├── udp_3d_lifting.py                           # 🔄 3D lifting pipeline
│
├── requirements.txt                            # Python dependencies
├── README.md                                   # Main project README
└── PROJECT_OVERVIEW.md                         # ← This file (cold start guide)
```

### Google Colab Deployment (Production)
```
/content/
├── unifiedposepipeline/                        # Cloned repository
│   ├── det_track/                             # Detection & tracking pipeline
│   ├── setup/                                 # Setup scripts (run these first)
│   ├── configs/
│   ├── lib/
│   ├── utils/
│   ├── demo_data/
│   ├── udp_video.py
│   ├── run_posedet.py
│   ├── vis_wb3d.py
│   └── ...
│
├── models/                                     # Downloaded model weights
│   ├── yolo/
│   │   ├── yolov8s.pt                         # 22.6 MB
│   │   └── yolov8s.engine                     # 23.0 MB (TensorRT)
│   ├── vitpose/
│   │   └── vitpose-b.pth                      # 360 MB
│   ├── rtmlib/
│   │   ├── rtmpose-l-coco-384x288.onnx        # 111.1 MB
│   │   └── rtmpose-l-halpe26-384x288.onnx     # 112.9 MB
│   ├── motionagformer/
│   │   └── motionagformer-base-h36m.pth.tr    # 141.9 MB
│   ├── wb3d/
│   │   └── rtmw3d-l.onnx                      # 230 MB
│   └── reid/
│       ├── osnet_x0_25_msmt17.onnx            # 0.9 MB
│       └── osnet_x1_0_msmt17.pt               # 10.9 MB
│
└── drive/                                      # Google Drive mount (optional backup)
    └── MyDrive/
        └── pipelinemodels/                     # Fallback model location
```

---

## 🚀 Getting Started on Google Colab

### Step 1: Clone Repository
```python
# In Colab notebook
!git clone https://github.com/pradeepj247/unifiedposepipeline.git
%cd unifiedposepipeline
```

**GitHub Repository**: `https://github.com/pradeepj247/unifiedposepipeline`  
**Owner**: `pradeepj247`

### Step 2: Run Setup Scripts
```python
# Install Python packages
!python setup/step1_install_packages.py

# Download model weights (~1 GB total)
!python setup/step2_fetch_models.py

# Download demo data (optional)
!python setup/step3_fetch_demodata.py

# Verify environment
!python setup/step4_verify_envt.py
```

**Note**: Models are fetched from GitHub releases with automatic fallback to Google Drive if needed.

### Step 3: Run Detection & Tracking Pipeline
```python
# Process demo video (outputs HTML viewer for person selection)
%cd det_track
!python run_pipeline.py --config configs/pipeline_config.yaml

# Open HTML viewer to select person
from google.colab import files
files.download('outputs/dance/webp_viewer/person_selection.html')
```

### Step 4: Extract Selected Person
```python
# After visual selection, extract person bboxes
!python stage5_extract_person.py \
    --config configs/pipeline_config.yaml \
    --person_id 5 \
    --source 3c
```

### Step 5: Run Pose Estimation (Current Implementation)
```python
# Navigate back to root
%cd /content/unifiedposepipeline

# Run unified pose pipeline (using selected_person.npz)
!python udp_video.py --config configs/udp_video.yaml

# Or run pose detection separately
!python run_posedet.py --input outputs/dance/selected_person.npz

# Or visualize 3D wholebody
!python vis_wb3d.py --video demo_data/videos/dance.mp4
```

---

## 🔄 Development Workflow

### Local → GitHub → Colab Cycle

1. **Local Development (Windows)**:
   ```powershell
   # Work on code locally at:
   D:\trials\unifiedpipeline\newrepo\
   
   # Make changes, test locally if possible
   python det_track/run_pipeline.py --config det_track/configs/pipeline_config.yaml
   ```

2. **Commit to GitHub**:
   ```bash
   # Stage changes
   git add det_track/stage2_track.py
   
   # Commit with descriptive message
   git commit -m "Optimize ByteTrack: reuse dummy frame, target 800+ FPS"
   
   # Push to remote
   git push origin main
   ```

3. **Pull on Colab**:
   ```python
   # In Colab notebook
   %cd /content/unifiedposepipeline
   !git pull origin main
   ```

4. **Test on Colab**:
   ```python
   # Run pipeline with GPU
   !python det_track/run_pipeline.py --config det_track/configs/pipeline_config.yaml
   
   # Check outputs
   !ls -lh det_track/outputs/dance/
   ```

5. **Share Results Back**:
   ```python
   # Download outputs for local analysis
   from google.colab import files
   files.download('det_track/outputs/dance/webp_viewer/person_selection.html')
   
   # Or zip and download entire outputs folder
   !zip -r outputs.zip det_track/outputs/
   files.download('outputs.zip')
   ```

6. **Debug Locally**:
   - Review downloaded outputs
   - Analyze logs, HTML viewers
   - Identify issues, fix code locally
   - Repeat cycle

**Why This Workflow?**
- **Local**: Fast iteration, full IDE support, debugging tools
- **Colab**: Free GPU (T4), reproducible environment, easy sharing
- **GitHub**: Version control, collaboration, backup

---

## 🎓 What Has Been Achieved

### ✅ Detection, Tracking & Selection Pipeline (COMPLETE)
**Location**: `det_track/` folder  
**Documentation**: [`det_track/newdocs/README_MASTER.md`](det_track/newdocs/README_MASTER.md)

**Highlights**:
- 8-stage pipeline (0 → 5) with comprehensive documentation
- Performance: 60 seconds for 2025 frames on T4 GPU
- YOLOv8s detection with eager crop extraction (11× faster Stage 3c)
- ByteTrack optimization (694 FPS, 27% improvement)
- 3-bin contiguous crop selection (temporal diversity)
- Optional OSNet ReID visual matching
- WebP-based HTML viewer for person selection
- Standardized NPZ output format for pose pipelines

**Key Design Decisions** (all documented):
- YOLOv8s over v8n (no speed penalty, better accuracy)
- Eager crop extraction (net +5-6s speedup)
- 3-bin contiguous selection (deterministic, smooth animations)
- ByteTrack dummy frame reuse (27% faster)
- detection_idx linkage (O(1) lookup throughout pipeline)

**Output**: `selected_person.npz` with all bboxes for target athlete

### ✅ Pose Estimation Pipeline (IMPLEMENTED, NEEDS CLEANUP)
**Current Files**:
- `udp_video.py`: Main unified detection pipeline
- `run_posedet.py`: Pose detection runner
- `run_detector.py`: Detector runner
- `test_detector.py`: Testing utilities
- `vis_wb3d.py`: 3D wholebody visualization
- `udp_3d_lifting.py`: 3D lifting pipeline

**Models Integrated**:
| Model | Type | Size | Purpose |
|-------|------|------|---------|
| YOLOv8s | Detection | 22.6 MB | Person detection |
| YOLOv8s TensorRT | Detection | 23.0 MB | Accelerated detection |
| ViTPose-B | 2D Pose | 360 MB | High-quality 2D keypoints (17 COCO) |
| RTMPose-L (COCO) | 2D Pose | 111.1 MB | Fast 2D keypoints (17 COCO) |
| RTMPose-L (Halpe26) | 2D Pose | 112.9 MB | Extended keypoints (26 Halpe) |
| RTM Wholebody 3D | 3D Pose | 230 MB | Wholebody 3D (133 keypoints) |
| MotionAGFormer | 3D Lifting | 141.9 MB | Temporal 3D pose refinement |
| OSNet x0.25 | ReID | 0.9 MB | Visual person matching (ONNX) |
| OSNet x1.0 | ReID | 10.9 MB | Visual person matching (PyTorch) |

**Capabilities**:
- 2D pose estimation (17-133 keypoints)
- 3D pose lifting with temporal smoothing
- Multi-backend support (RTMPose ONNX, ViTPose PyTorch)
- Wholebody estimation (body + hands + face)

**Current Status**: Working implementation, but spread across multiple files. **Needs consolidation and cleanup**.

---

## 🎯 Next Steps & Cleanup Plan

### Immediate Tasks
1. **Implement Stage 5**: Create `det_track/stage5_extract_person.py` (spec ready in docs)
2. **Consolidate Pose Pipeline**: Merge `udp_video.py`, `run_posedet.py` logic into unified stage
3. **Create Stage 6**: 2D pose estimation using `selected_person.npz` as input
4. **Create Stage 7**: 3D lifting using Stage 6 output
5. **Clean Up**: Remove redundant files (`test_detector.py`, etc.)

### Documentation Needed
- Stage 6 documentation (2D pose estimation)
- Stage 7 documentation (3D lifting)
- Stage 8+ documentation (biomechanics)
- Integration guide (det_track → pose → analytics)

### Testing Strategy
- Test full pipeline end-to-end on Colab
- Validate NPZ format compatibility
- Benchmark performance (2D/3D stages)
- Document model switching (RTMPose ↔ ViTPose)

---

## 📚 Key Documentation Files

### Must-Read for Cold Start
1. **This file** (`PROJECT_OVERVIEW.md`): Overall project context
2. [`det_track/newdocs/README_MASTER.md`](det_track/newdocs/README_MASTER.md): Detection/tracking pipeline master doc
3. [`det_track/newdocs/QUICK_REFERENCE.md`](det_track/newdocs/QUICK_REFERENCE.md): Command cheat sheet
4. [`setup/models.yaml`](setup/models.yaml): All model sources and configurations

### Deep Dives (When Needed)
- [`det_track/newdocs/STAGE*.md`](det_track/newdocs/): Individual stage documentation (0-5)
- [`det_track/newdocs/PIPELINE_CONFIG_REFERENCE.md`](det_track/newdocs/PIPELINE_CONFIG_REFERENCE.md): Config file explained
- [`det_track/newdocs/RUN_PIPELINE_EXECUTION.md`](det_track/newdocs/RUN_PIPELINE_EXECUTION.md): Orchestrator logic
- [`README.md`](README.md): Main project README

---

## 🛠️ Troubleshooting

### Common Issues on Colab

**Issue**: Models not downloading  
**Solution**: Mount Google Drive and use fallback location:
```python
from google.colab import drive
drive.mount('/content/drive')
!python setup/step2_fetch_models.py  # Will auto-fallback to Drive
```

**Issue**: Out of GPU memory  
**Solution**: Reduce batch size or use smaller models:
```yaml
# In configs/udp_video.yaml
pose_estimation:
  method: rtmpose  # Use RTMPose instead of ViTPose (smaller)
```

**Issue**: Path resolution errors (`${repo_root}` in logs)  
**Solution**: Check YAML syntax in config files, ensure multi-pass resolution works

**Issue**: HTML viewer not rendering in Colab  
**Solution**: Download HTML file and open locally:
```python
from google.colab import files
files.download('det_track/outputs/dance/webp_viewer/person_selection.html')
```

### Getting Help

1. **Check documentation**: `det_track/newdocs/` has comprehensive answers
2. **Review logs**: Pipeline outputs detailed logs with timing and errors
3. **Inspect outputs**: HTML viewers, NPZ files show intermediate results
4. **Test incrementally**: Run individual stages to isolate issues

---

## 📊 Performance Benchmarks

### Detection & Tracking (Stages 0-4)
- **Hardware**: Google Colab T4 GPU
- **Video**: 2025 frames, 1920×1080, 30 FPS
- **Total Time**: 60.24 seconds (33.6 FPS overall)
- **Breakdown**: See [`det_track/newdocs/README_MASTER.md`](det_track/newdocs/README_MASTER.md) for detailed table

### Pose Estimation (Estimated)
- **RTMPose**: ~50 FPS (ONNX optimized)
- **ViTPose**: ~40 FPS (PyTorch)
- **MotionAGFormer**: ~30 FPS (temporal refinement)
- **Wholebody 3D**: ~25 FPS (133 keypoints)

---

## 🔗 Quick Links

- **GitHub Repository**: https://github.com/pradeepj247/unifiedposepipeline
- **Main README**: [README.md](README.md)
- **Detection/Tracking Docs**: [det_track/newdocs/README_MASTER.md](det_track/newdocs/README_MASTER.md)
- **Model Configuration**: [setup/models.yaml](setup/models.yaml)
- **Setup Scripts**: [setup/](setup/)

---

**Document Version**: 1.0  
**Last Updated**: January 17, 2026  
**For Questions**: Review comprehensive documentation in `det_track/newdocs/` first

