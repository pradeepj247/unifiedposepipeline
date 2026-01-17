# Stage 5: Person Selection & Bbox Extraction

**Implementation**: `stage5_select_person.py` ✅ **COMPLETE**

## Purpose
Extract all bounding boxes for a user-selected person from the HTML viewer, creating a standardized NPZ file for downstream pose estimation pipelines (RTMPose, ViTPose, MotionAGFormer, etc.).

## 🚀 Quick Start: Using Pre-computed Outputs

For rapid testing of the pose estimation pipeline, you can skip Stages 0-4 by using pre-computed outputs stored in Google Drive.

### Pre-computed Pipeline Outputs Location
```
/content/drive/MyDrive/pipelineoutputs/kohli_nets/
```

**Available Files** (from complete pipeline run):
```
canonical_persons_3c.npz          # 139 KB - Filtered persons (Stage 3c)
canonical_persons_3c.npz.timings.json
canonical_persons.npz             # 185 KB - All grouped persons (Stage 3b)
canonical_persons.npz.timings.json
detections_raw.npz                # 170 KB - YOLO detections (Stage 1)
detections_raw.npz.timings.json
final_crops_3c.pkl                # 41 MB - Extracted crops for persons
grouping_log.json                 # 8 KB - Grouping statistics
selected_person.npz               # 25 KB - ✅ Ready for pose estimation!
stage3c_sidecar.json              # 7 KB - Filtering details
tracklets_raw.npz                 # 183 KB - Tracking results (Stage 2)
tracklets_raw.npz.timings.json
tracklet_stats.npz                # 188 KB - Tracklet statistics (Stage 3a)
tracklet_stats.npz.timings.json
webp_viewer/                      # HTML viewer for person selection
```

### Copy Pre-computed Outputs for Testing
```bash
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy pre-computed outputs to your working directory
mkdir -p /content/unifiedposepipeline/demo_data/outputs/kohli_nets
cp -r /content/drive/MyDrive/pipelineoutputs/kohli_nets/* \
      /content/unifiedposepipeline/demo_data/outputs/kohli_nets/

# Verify selected_person.npz is ready
ls -lh /content/unifiedposepipeline/demo_data/outputs/kohli_nets/selected_person.npz
```

**Benefits**:
- ⚡ Skip 60+ seconds of detection/tracking/selection
- 🔄 Iterate quickly on pose estimation models
- 💾 Reuse outputs across different pose backends (RTMPose, ViTPose, etc.)
- 🧪 Test 3D lifting without re-running detection

**Extracted Person Info** (kohli_nets video):
- **Person ID**: 3
- **Frames**: 2019 out of 2027 (99.7% coverage)
- **Duration**: ~80 seconds @ 25 FPS
- **Quality**: Mean confidence 0.873 (high quality detections)
- **Format**: Backward-compatible with all pose pipelines

## User Workflow

### Step 1: Visual Selection
User opens the interactive HTML viewer from Stage 4:
```bash
# Open in browser
open det_track/outputs/dance/webp_viewer/person_selection.html
```

The viewer shows:
- **Row 1**: Stage 3c persons (8-10 persons before ReID merging)
- **Row 2**: Stage 3d persons (7-8 persons after ReID merging, if enabled)
- Each person displayed as animated WebP (60 frames @ 200ms = 12s loop)

### Step 2: Identify Person ID
User visually inspects animations and notes the **person_id** of interest:
```
Example: "I want person_3c_005" or "person_3d_002"
```

Person IDs are displayed in the HTML viewer below each animation.

### Step 3: Run Extraction
```bash
python stage5_select_person.py \
    --config configs/pipeline_config.yaml \
    --person_id 5
```

**Note**: Current implementation works with `canonical_persons_3c.npz` only.

## Inputs

### Required
- `canonical_persons_3c.npz`: Person records from Stage 3c
- `canonical_video.mp4`: Video metadata (frame count, resolution, FPS)

### User Provides
- `--person_id`: Integer ID of selected person (e.g., 5)

## Outputs

### Primary Output
**`selected_person.npz`**: Standardized format for pose estimation pipelines

NPZ structure:
```python
{
    'person_id': int,                    # Selected person ID
    'frame_numbers': np.ndarray,         # (N,) Frame indices where person appears
    'bboxes': np.ndarray,                # (N, 4) Bounding boxes [x1, y1, x2, y2]
    'confidences': np.ndarray,           # (N,) YOLO detection confidences
    'video_metadata': {
        'video_path': str,               # Path to canonical_video.mp4
        'total_frames': int,             # Total video frames
        'fps': float,                    # Video frame rate
        'resolution': tuple              # (width, height)
    },
    'person_metadata': {
        'source': str,                   # '3c' or '3d'
        'tracklet_ids': list,            # Original tracklet IDs merged into this person
        'duration_frames': int,          # Number of frames person appears
        'coverage_ratio': float          # duration / (last_frame - first_frame)
    }
}
```

### Example Usage in Pose Pipeline
```python
import numpy as np

# Load selected person
data = np.load('selected_person.npz', allow_pickle=True)

frame_numbers = data['frame_numbers']      # [10, 11, 12, ..., 1995]
bboxes = data['bboxes']                    # [[x1,y1,x2,y2], ...]
confidences = data['confidences']          # [0.92, 0.91, ...]
metadata = data['video_metadata'].item()   # Video info

# Use in pose estimation
video_path = metadata['video_path']
for frame_num, bbox in zip(frame_numbers, bboxes):
    # Extract frame, crop person, run pose model
    crop = extract_frame_crop(video_path, frame_num, bbox)
    keypoints = pose_model(crop)
```

## Processing Flow

```
HTML Viewer (Stage 4 output)
    ↓
User visually selects person (e.g., person_3c_005)
    ↓
Run: python stage5_extract_person.py --person_id 5 --source 3c
    ↓
Load canonical_persons_3c.npz
    ↓
Find person with person_id=5
    ├─→ Extract frame_numbers (where person appears)
    ├─→ Extract bboxes (from person['bboxes'])
    ├─→ Extract confidences (from person['confidences'])
    └─→ Extract tracklet_ids (person['tracklet_ids'])
    ↓
Load video metadata from canonical_video.mp4
    ├─→ Total frames
    ├─→ FPS
    └─→ Resolution
    ↓
Package into selected_person.npz
    ↓
Output: selected_person.npz (ready for pose estimation)
```

## Data Continuity via `detection_idx`

The `detection_idx` linkage ensures perfect correspondence:
```
Stage 1: YOLO detection creates detection_idx=42
    └─→ Stored in detections_raw.npz[42]
    
Stage 2: ByteTrack creates tracklet_id=7, includes detection_idx=42
    └─→ tracklets_raw.npz: tracklet 7 contains detection_idx=42
    
Stage 3b: Tracklet 7 merged into person_id=5
    └─→ canonical_persons.npz: person 5 has detection_indices=[..., 42, ...]
    
Stage 5: User selects person_id=5
    └─→ Extract all detection_indices → lookup bboxes in detections_raw.npz
    └─→ Result: bbox from frame where detection_idx=42 occurred
```

## Configuration

```yaml
stage5_extract:
  input:
    # Automatically determined based on --source flag
    canonical_persons_3c_file: ${outputs_dir}/${current_video}/canonical_persons_3c.npz
    canonical_persons_3d_file: ${outputs_dir}/${current_video}/canonical_persons_3d.npz
    detections_file: ${outputs_dir}/${current_video}/detections_raw.npz
    video_file: ${outputs_dir}/${current_video}/canonical_video.mp4
  
  output:
    selected_person_file: ${outputs_dir}/${current_video}/selected_person.npz
  
  advanced:
    validate_bbox_consistency: true    # Verify bboxes are within frame bounds
    interpolate_missing_frames: false  # Option to fill gaps (future feature)
```

## Command-Line Interface

### Basic Usage
```bash
# Extract person 5 from Stage 3c results
python stage5_extract_person.py --config configs/pipeline_config.yaml \
    --person_id 5 --source 3c

# Extract person 2 from Stage 3d results (after ReID)
python stage5_extract_person.py --config configs/pipeline_config.yaml \
    --person_id 2 --source 3d
```

### Advanced Options
```bash
# Custom output path
python stage5_extract_person.py \
    --config configs/pipeline_config.yaml \
    --person_id 5 --source 3c \
    --output custom_output/dancer_main.npz

# Validate and show person info
python stage5_extract_person.py \
    --config configs/pipeline_config.yaml \
    --person_id 5 --source 3c \
    --verbose
```

## Output Validation

The script performs automatic validation:

```python
✓ Person ID 5 found in canonical_persons_3c.npz
✓ Extracted 1847 frames (91.2% coverage)
✓ Bounding boxes: min=(45, 120), max=(1876, 1065)
✓ All bboxes within video bounds (1920x1080)
✓ Mean confidence: 0.89 (high quality detections)
✓ Saved to: outputs/dance/selected_person.npz (2.3 MB)
```

## Integration with Pose Pipelines

### RTMPose Example
```python
import numpy as np
from rtmlib import RTMPose

# Load selected person
data = np.load('selected_person.npz', allow_pickle=True)
video_path = data['video_metadata'].item()['video_path']

# Initialize pose model
pose_model = RTMPose(
    onnx_model='rtmpose-m_8xb256-420e_coco-256x192.onnx',
    device='cuda'
)

# Process each detection
keypoints_list = []
for frame_num, bbox in zip(data['frame_numbers'], data['bboxes']):
    frame = extract_frame(video_path, frame_num)
    x1, y1, x2, y2 = bbox.astype(int)
    crop = frame[y1:y2, x1:x2]
    
    keypoints, scores = pose_model(crop)
    keypoints_list.append(keypoints)
```

### ViTPose Example
```python
from vitpose import VitPoseOnly

data = np.load('selected_person.npz', allow_pickle=True)
pose_model = VitPoseOnly(model='vitpose-b.pth', dataset='coco')

# Batch process for efficiency
bboxes_xyxy = data['bboxes']  # Already in xyxy format
keypoints = pose_model.inference(video_path, bboxes=bboxes_xyxy)
```

### MotionAGFormer (3D Lifting)
```python
# After 2D pose estimation, use selected_person.npz for frame alignment
data = np.load('selected_person.npz', allow_pickle=True)
frame_numbers = data['frame_numbers']

# Align 2D keypoints with frame numbers
keypoints_2d_aligned = align_keypoints_to_frames(
    keypoints_2d, frame_numbers, 
    total_frames=data['video_metadata'].item()['total_frames']
)

# Run 3D lifting
from model.MotionAGFormer import MotionAGFormer
model_3d = MotionAGFormer(...)
keypoints_3d = model_3d(keypoints_2d_aligned)
```

## NPZ File Format Details

### Field Descriptions

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `person_id` | int | scalar | Selected person ID (from HTML viewer) |
| `frame_numbers` | int | (N,) | Absolute frame indices where person detected |
| `bboxes` | float | (N, 4) | Bounding boxes in xyxy format [x1, y1, x2, y2] |
| `confidences` | float | (N,) | YOLO detection confidence scores (0-1) |
| `video_metadata` | dict | - | Video properties (path, fps, resolution) |
| `person_metadata` | dict | - | Person properties (tracklets, duration, coverage) |

### Example Data
```python
>>> data = np.load('selected_person.npz', allow_pickle=True)
>>> data['person_id']
5

>>> data['frame_numbers'][:5]
array([  10,   11,   12,   13,   14])

>>> data['bboxes'][:2]
array([[  785.2,  245.1, 1023.8,  876.4],
       [  790.1,  248.3, 1025.2,  879.1]])

>>> data['confidences'][:5]
array([0.923, 0.918, 0.921, 0.915, 0.919])

>>> data['video_metadata'].item()
{
    'video_path': 'outputs/dance/canonical_video.mp4',
    'total_frames': 2025,
    'fps': 30.0,
    'resolution': (1920, 1080)
}

>>> data['person_metadata'].item()
{
    'source': '3c',
    'tracklet_ids': [3, 7, 12],
    'duration_frames': 1847,
    'coverage_ratio': 0.912
}
```

## Design Rationale

### Why Separate Selection Stage?
1. **Human-in-the-loop**: User expertise in identifying "main person" > automated ranking
2. **Flexibility**: User can choose different persons for different analyses
3. **Reproducibility**: Selection recorded (person_id) for future reference
4. **Decoupling**: Pose estimation independent from detection/tracking

### Why NPZ Format?
- **Standard**: NumPy native, widely supported
- **Compact**: Binary format, smaller than JSON/CSV
- **Type-safe**: Preserves int/float types
- **Fast I/O**: Efficient loading for batch processing

### Why Include Both 3c and 3d Options?
- **3c**: More persons (8-10), use if ReID incorrectly merged your target
- **3d**: Fewer persons (7-8), use if same person split across tracklets

## Performance Notes

- **Extraction time**: <0.1s (simple array slicing)
- **File size**: ~2-3 MB for 2000 frames (very compact)
- **Memory usage**: <10 MB (loads only selected person, not all data)

## Error Handling

### Person ID Not Found
```
Error: Person ID 99 not found in canonical_persons_3c.npz
Available person IDs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### Invalid Source
```
Error: --source must be '3c' or '3d', got 'invalid'
```

### Missing Input Files
```
Error: Required file not found: outputs/dance/canonical_persons_3c.npz
Run Stage 3c first: python run_pipeline.py --stages 3c
```

## Next Steps After Selection

1. **2D Pose Estimation** (Stage 6): RTMPose or ViTPose
2. **3D Pose Lifting** (Stage 7): HybrIK or MotionAGFormer
3. **Biomechanics Analysis** (Stage 8+): Joint angles, COM, ground reaction forces

---

**Related**: [Back to Master](README_MASTER.md) | [← Stage 4](STAGE4_HTML_GENERATION.md) | [Stage 6 (Pose) →]

