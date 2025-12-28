# Directory Structure

This document describes the directory structure used by the unified pose estimation pipeline.

## Demo Data Directories

The pipeline uses the following directory structure for demo data:

### `demo_data/`
Root directory for all demo data files.

### `demo_data/videos/`
Contains sample video files for testing the pipeline.
- Videos are downloaded from GitHub releases during setup (step3)
- Used for testing video-based pose estimation

### `demo_data/images/`
Contains sample image files for testing the pipeline.
- Images are downloaded from GitHub releases during setup (step3)
- Used for testing image-based pose estimation

### `demo_data/outputs/`
Output directory for processed results.
- Generated automatically when running demos
- Contains visualization videos, pose data, and analysis results

## Configuration Directories

### `configs/`
Contains YAML configuration files for different pipeline modes.
- Already tracked in git repository
- Includes configurations for:
  - `udp_video.yaml` - Video processing
  - `udp_image.yaml` - Image processing
  - `detector.yaml` - Detection settings
  - `detector_tracking_benchmark.yaml` - Tracking benchmarks

## Model Directories

### `/content/models/` (Colab) or `models/` (Local)
Contains all downloaded model files.
- Models are organized in subfolders by type:
  - `detection/` - YOLO detection models
  - `pose2d/` - 2D pose estimation models
  - `pose3d/` - 3D pose estimation models
  - `tracking/` - ReID and tracking models

All directories (except `/content/models/`) are created automatically during setup or runtime as needed.
