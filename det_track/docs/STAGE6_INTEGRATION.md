# Stage 6: Visualization Integration

## Overview
Stage 6 creates a visualization video showing the top 10 canonical persons (those appearing for ≥5 seconds) with bounding boxes.

## Integration Status: ✅ COMPLETE

### Files Modified

1. **`stage9_create_output_video.py`** (previously `create_visualization_video.py`)
   - Updated to use config file instead of CLI args
   - Added top 10 person limit (was showing all qualifying persons)
   - Added config resolution logic
   - Now accepts `--config` parameter like other stages

2. **`configs/pipeline_config.yaml`**
   - Added `stage9_create_output_video: true` to pipeline stages
   - Added complete Stage 6 configuration section:
     ```yaml
     stage9_create_output_video:
       enabled: true
       visualization:
         min_duration_seconds: 5
         max_persons_shown: 10
         output_resolution: 720
         output_fps: 22.5
       input:
         video_file: ${video_dir}${video_file}
         canonical_persons_file: ${outputs_dir}/${current_video}/canonical_persons.npz
       output:
         video_file: ${outputs_dir}/${current_video}/top_persons_visualization.mp4
     ```

3. **`run_pipeline.py`**
   - Added Stage 6 to `all_stages` list
   - Added Stage 6 output to file listing
   - Stage 6 now runs automatically after Stage 5

## Usage

### Run Full Pipeline (Stages 1-6)
```bash
python run_pipeline.py --config configs/pipeline_config.yaml
```

### Run Only Stage 6 (after other stages complete)
```bash
python run_pipeline.py --config configs/pipeline_config.yaml --stages 6
```

### Run Stages 1-5, then Stage 6
```bash
python run_pipeline.py --config configs/pipeline_config.yaml --stages 1,2,3,4b,5,6
```

## Output

**File**: `{outputs_dir}/{video_name}/top_persons_visualization.mp4`

**Content**:
- Shows top 10 persons (or all if <10 meet criteria)
- Each person has unique colored bounding box
- Person ID labeled on bbox
- 720p resolution @ 22.5 FPS
- Only includes persons appearing ≥5 seconds

## Configuration Options

In `pipeline_config.yaml`:

- **`min_duration_seconds`**: Minimum appearance time (default: 5 seconds)
- **`max_persons_shown`**: Maximum number of persons to show (default: 10)
- **`output_resolution`**: Output video height in pixels (default: 720)
- **`output_fps`**: Output video frame rate (default: 22.5)

## Next Steps: Manual Person Selection

Currently Stage 5 auto-selects the primary person. To enable manual selection:

1. Run pipeline through Stage 6 to generate visualization
2. Watch `top_persons_visualization.mp4` to see all qualifying persons
3. Note the person ID you want to select
4. Create a tool to extract that specific person from `canonical_persons.npz`
5. Generate new `primary_person.npz` with selected person data

### Planned Tool: `select_person_manually.py`
```bash
python select_person_manually.py --config configs/pipeline_config.yaml --person-id 3
```

This will:
- Read `canonical_persons.npz`
- Extract person with ID 3
- Create `primary_person.npz` for downstream pose estimation
- Skip Stage 5 (auto-ranking) entirely

## Example: kohli_nets.mp4

**Results** (from previous run):
- 44 canonical persons identified
- 14 persons appear ≥5 seconds
- Top 10 shown in visualization:
  - Person 3: 2016 frames (99.5% coverage) ← Auto-selected by Stage 5
  - Person 7, 17, 22, 29, etc.

**Manual Selection Workflow**:
1. ✅ Run Stages 1-6 → Get visualization video
2. 📹 Watch visualization → See all 10 persons
3. ✏️ Pick person ID (e.g., Person 7)
4. 🔧 Run `select_person_manually.py --person-id 7`
5. 🎯 Continue with pose estimation on Person 7

## Color Palette (First 15 Persons)
```python
COLORS = [
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 0, 128),    # Purple
    (0, 128, 128),    # Olive
    (128, 128, 0),    # Teal
    (255, 128, 0),    # Orange
    (0, 128, 255),    # Light Blue
    (128, 255, 0),    # Lime
    (255, 0, 128),    # Pink
    (128, 0, 255),    # Violet
    (0, 255, 128),    # Spring Green
]
```

## Technical Details

**Video Processing**:
- Input: Original video (e.g., 1920x1080 @ 25 FPS)
- Output: 720p @ 22.5 FPS (90% of original FPS)
- Encoding: H.264 (mp4v codec)
- Bounding boxes drawn every frame where person appears

**Person Filtering**:
```python
min_frames = int(min_duration_seconds * video_fps)  # 5s * 25fps = 125 frames
top_n = min(10, len(persons_with_duration))  # Limit to 10
```

**Sorting**: Persons ranked by total frame count (descending)

## Testing

After integration, verify:
```bash
# 1. Check Stage 6 is enabled
grep -A2 "stage9_create_output_video" configs/pipeline_config.yaml

# 2. Run pipeline
python run_pipeline.py --config configs/pipeline_config.yaml

# 3. Verify output exists
ls -lh demo_data/outputs/kohli_nets/top_persons_visualization.mp4
```

Expected output:
```
✅ Stage 6: Visualization completed in XX.XXs
📦 Output Files:
  ✅ top_persons_visualization.mp4 (XX.XX MB)
```
