# Backup & Restore Scripts for Pipeline Outputs

## Overview
These scripts help you backup detection/tracking/selection outputs to Google Drive and restore them later for quick pose estimation testing.

**Benefits:**
- ⏱️ Skip 60+ seconds of detection/tracking processing
- 💾 Persist outputs across Colab sessions
- 🔄 Iterate quickly on pose models without re-running detection
- 📦 Manage multiple video outputs easily

## Scripts

### 1. list_outputs.sh
**Purpose:** List all backed-up video outputs in Google Drive

```bash
./list_outputs.sh
```

**Output:**
- Lists all video directories in `/content/drive/MyDrive/pipelineoutputs/`
- Shows size and file count for each video
- Displays which key files are present (selected_person.npz, canonical_video.mp4, etc.)

### 2. copy_outputs.sh
**Purpose:** Backup outputs from Colab to Google Drive

```bash
./copy_outputs.sh <video_name>
```

**Example:**
```bash
# After running det_track pipeline on kohli_nets.mp4
./copy_outputs.sh kohli_nets
```

**What it does:**
- Copies everything from `/content/unifiedposepipeline/demo_data/outputs/<video_name>/`
- To `/content/drive/MyDrive/pipelineoutputs/<video_name>/`
- Verifies key files are backed up
- Shows backup summary (size, file count)

### 3. restore_outputs.sh
**Purpose:** Restore outputs from Google Drive to Colab

```bash
./restore_outputs.sh <video_name>
```

**Example:**
```bash
# Restore kohli_nets outputs to start pose detection immediately
./restore_outputs.sh kohli_nets
```

**What it does:**
- Copies everything from `/content/drive/MyDrive/pipelineoutputs/<video_name>/`
- To `/content/unifiedposepipeline/demo_data/outputs/<video_name>/`
- Shows next steps for running pose detection
- Provides command to update config file

## Typical Workflow

### Initial Run (with detection/tracking)
```bash
# 1. Run full pipeline (takes 60+ seconds)
cd /content/unifiedposepipeline/det_track
python run_pipeline.py --config configs/pipeline_config.yaml

# 2. Select person from HTML viewer
python stage5_select_person.py --config configs/pipeline_config.yaml --person_id 3

# 3. Backup outputs to Google Drive
cd /content/unifiedposepipeline/setup/bkup_restore
./copy_outputs.sh kohli_nets
```

### Later Sessions (skip detection/tracking)
```bash
# 1. See what's available
cd /content/unifiedposepipeline/setup/bkup_restore
./list_outputs.sh

# 2. Restore outputs (takes 1-2 seconds)
./restore_outputs.sh kohli_nets

# 3. Run pose detection immediately
cd /content/unifiedposepipeline
python run_posedet.py --config configs/posedet.yaml
```

## Directory Structure

**Google Drive Backup Location:**
```
/content/drive/MyDrive/pipelineoutputs/
├── kohli_nets/
│   ├── selected_person.npz          # ✅ Ready for pose detection
│   ├── canonical_video.mp4          # Normalized video
│   ├── canonical_persons_3c.npz     # Filtered persons
│   ├── detections_raw.npz           # YOLO detections
│   ├── tracklets_raw.npz            # Tracking results
│   └── ...
├── dance_sequence/
│   └── ...
└── cricket_shot/
    └── ...
```

**Colab Working Location:**
```
/content/unifiedposepipeline/demo_data/outputs/
├── kohli_nets/
│   └── (same structure as above)
├── dance_sequence/
│   └── ...
└── cricket_shot/
    └── ...
```

## Requirements

- Google Drive mounted at `/content/drive/`
- Run scripts from `/content/unifiedposepipeline/setup/bkup_restore/`

**To mount Google Drive:**
```python
from google.colab import drive
drive.mount('/content/drive')
```

## File Permissions

Make scripts executable on first use:
```bash
cd /content/unifiedposepipeline/setup/bkup_restore
chmod +x *.sh
```
