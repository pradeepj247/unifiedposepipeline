# 🚀 Google Colab Quick Start Guide

**Last Updated**: January 10, 2026  
**Purpose**: Step-by-step instructions to run the unified detection & tracking pipeline on Google Colab

---

## STEP 0: Fresh Git Clone

```python
%cd /content/
!git clone https://github.com/pradeepj247/unifiedposepipeline.git
```

**Expected output**:
```
Cloning into 'unifiedposepipeline'...
remote: Enumerating objects...
...
done.
```

---

## STEP 1: Install Libraries & Packages

```python
%cd /content/unifiedposepipeline/setup/
!python step1_install_libs.py
```

**What this script does**:
- Installs all required Python packages
- Sets up repository root: `/content/unifiedposepipeline`
- Creates models directory: `/content/models`
- Loads configuration from `libraries.yaml`

**Expected output**:
```
📁 Repository root: /content/unifiedposepipeline
📁 Models directory: /content/models
✅ Loaded configuration from libraries.yaml

Installing required packages...
...
✅ All installations complete!
```

---

## STEP 2: Fetch All Model Files

```python
%cd /content/unifiedposepipeline/setup/
!python step2_fetch_models.py
```

**What this script does**:
- Loads model configuration from `models.yaml`
- Fetches models from GitHub (preferred source)
- Downloads to `/content/unifiedposepipeline/models/`
- Validates all 10 required models

**Expected output**:
```
======================================================================
🚀 STEP 2: Fetch Model Files
======================================================================
  📋 Loading configuration from: models.yaml
  🎯 Preferred source: GITHUB
  📂 Destination folder: /content/unifiedposepipeline/models/
  📦 Total models to fetch: 10
  ─────────────────────────────────────────────────────────────────

  ✅ Model 1/10: YOLO detection...
  ✅ Model 2/10: RTMPose backbone...
  ...
  ✅ All models fetched successfully!
```

---

## STEP 3: Install Demo Data & Folders

```python
%cd /content/unifiedposepipeline/setup/
!python step3_fetch_demodata.py
```

**What this script does**:
- Loads demo data configuration from `demodata.yaml`
- Creates demo data folder structure
- Sets up sample videos and images
- Initializes outputs directory: `/content/unifiedposepipeline/demo_data`

**Expected output**:
```
======================================================================
🚀 STEP 3: Pull Demo Data
======================================================================

   ✅ Loaded configuration from demodata.yaml
   📁 Demo data folder: /content/unifiedposepipeline/demo_data
   
   ✅ Videos downloaded: dance.mp4, kohli_nets.mp4
   ✅ Images downloaded: sample.jpg
   ✅ Outputs folder created: /content/unifiedposepipeline/demo_data/outputs
   
   ✅ Demo data setup complete!
```

---

## STEP 4: Verify Installation & Environment

```python
%cd /content/unifiedposepipeline/setup/
!python step4_verify_envt_new.py
```

**What this script does**:
- Verifies all installations from Steps 1-3
- Reads configuration from YAML files:
  - `libraries.yaml` - Python packages
  - `models.yaml` - Model files
  - `demodata.yaml` - Demo data
- Validates all imports and directory structure
- Checks that all required files are in place

**Expected output**:
```
This script will verify your installation by reading configurations
  from the YAML files used in steps 1-3.

  📂 Repository root: /content/unifiedposepipeline
  📂 Libraries config: /content/unifiedposepipeline/setup/libraries.yaml
  📂 Models config: /content/unifiedposepipeline/setup/models.yaml
  📂 Demo data config: /content/unifiedposepipeline/setup/demodata.yaml

  ✅ All imports successful
  ✅ All folders created
  ✅ All models downloaded
  ✅ All demo data ready
  
  🎉 Environment verification complete!
```

---

## STEP 5: Run Detection & Tracking Pipeline

```python
%cd /content/unifiedposepipeline/det_track

!python run_pipeline.py --config configs/pipeline_config.yaml
```

**What this script does**:
- Runs the complete 11-stage pipeline
- Enabled stages: 1, 2, 3, 4, 5, 6, 7, 9, 10, 11
  - Stage 1: YOLO Detection
  - Stage 2: ByteTrack Tracking
  - Stage 3: Tracklet Analysis
  - Stage 4: Load Crops Cache
  - Stage 5: Canonical Grouping
  - Stage 6: HDF5 Enrichment
  - Stage 7: Ranking
  - Stage 9: Output Video (Top 10)
  - Stage 10: HTML Selection Report
  - Stage 11: WebP Generation
- Processes demo video and generates outputs

**Expected output**:
```
🎬 UNIFIED DETECTION & TRACKING PIPELINE
======================================================================

Config: configs/pipeline_config.yaml
Running enabled stages: 1, 2, 3, 4, 5, 6, 7, 9, 10, 11

🚀 Running Stage 1: YOLO Detection...
  ✅ Stage 1 completed in 4.60s

🚀 Running Stage 2: ByteTrack Tracking...
  ✅ Stage 2 completed in 3.00s

... (stages 3-11 progress) ...

======================================================================
🎉 Pipeline completed successfully in ~18s!

📂 Outputs saved to: /content/unifiedposepipeline/demo_data/outputs/
   - detections_raw.npz
   - tracklets_raw.npz
   - canonical_persons.npz
   - crops_enriched.h5
   - ranking_report.json
   - webp/ folder (10 animated WebPs)
   - person_selection_report.html (2.35 MB)
```

---

