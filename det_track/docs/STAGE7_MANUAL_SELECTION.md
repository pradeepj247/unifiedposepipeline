# Stage 7: Manual Person Selection

## Overview
`stage7_select_person.py` allows you to manually select a specific person from the canonical persons data, creating the `primary_person.npz` file needed for pose estimation.

## Purpose
- **Alternative to Stage 5**: Instead of auto-ranking, manually choose which person to analyze
- **Use Case**: When you want to select a specific person after reviewing the visualization video

## Workflow

```
Stage 1-4b → Generate canonical_persons.npz
     ↓
Stage 6 → Create visualization video (shows top 10 persons with IDs)
     ↓
Watch video → Note the person ID you want
     ↓
Stage 7 → Select that person manually
     ↓
Continue → Pose estimation on selected person
```

## Usage

### Basic Usage
```bash
python stage7_select_person.py --config configs/pipeline_config.yaml --person-id 3
```

### With Verbose Output
```bash
python stage7_select_person.py --config configs/pipeline_config.yaml --person-id 7 --verbose
```

### List Available Persons
```bash
# Use an invalid ID (like 999) to see all available persons
python stage7_select_person.py --config configs/pipeline_config.yaml --person-id 999
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--config` | ✅ | Path to pipeline configuration YAML |
| `--person-id` | ✅ | Person ID to select (as shown in visualization) |
| `--verbose` | ❌ | Show detailed statistics (optional) |

## Input Files

Reads from `canonical_persons.npz`:
```
demo_data/outputs/{video_name}/canonical_persons.npz
```

## Output Files

Creates two files:

1. **`primary_person.npz`** - Selected person data (same format as Stage 5 output)
   - Location: `demo_data/outputs/{video_name}/primary_person.npz`
   - Contents: Person's bboxes, frames, confidences, tracklet IDs

2. **`selection_report.json`** - Selection metadata
   - Location: `demo_data/outputs/{video_name}/selection_report.json`
   - Contents: Which person was selected and why (manual selection)

## Example Output

```
======================================================================
📌 STAGE 7: MANUAL PERSON SELECTION
======================================================================

📂 Loading canonical persons...
  ✅ Loaded 44 canonical persons

🎯 Selected Person: 3
  Frame count: 2016 frames
  Frame range: 0 - 2026
  Tracklets: [0, 3, 7, 10, 15, ...]

💾 Saving primary person...
  ✅ Saved: /content/unifiedposepipeline/demo_data/outputs/kohli_nets/primary_person.npz
  ✅ Saved selection report: /content/unifiedposepipeline/demo_data/outputs/kohli_nets/selection_report.json

======================================================================
✅ PERSON SELECTION COMPLETE!
======================================================================
📦 Output: primary_person.npz
🎯 Selected: Person 3 (2016 frames)

💡 Next: Use this person for pose estimation
======================================================================
```

## Error Handling

### Person Not Found
If you specify an invalid person ID, the script will show all available persons:

```
❌ Person ID 999 not found in canonical persons!

📋 Available Person IDs:
   1. Person  3: 2016 frames, tracklets [0, 3, 7, ...]
   2. Person  7: 1842 frames, tracklets [1, 5, ...]
   3. Person 17: 1523 frames, tracklets [2, 8, ...]
   ...
```

### Missing canonical_persons.npz
```
❌ Canonical persons file not found: /path/to/canonical_persons.npz
   Please run Stages 1-4b first to generate canonical persons.
```

## Comparison: Stage 5 vs Stage 7

| Feature | Stage 5 (Auto) | Stage 7 (Manual) |
|---------|----------------|------------------|
| Selection Method | Automatic ranking | Manual choice |
| Input Required | None | Person ID from user |
| Use Case | Default workflow | When you want specific person |
| Speed | Fast | Requires reviewing visualization |
| Flexibility | None | Full control |

## Complete Workflow Example

### 1. Run Detection & Tracking Pipeline
```bash
python run_pipeline.py --config configs/pipeline_config.yaml --stages 1,2,3,4b
```

### 2. Create Visualization (Stage 6)
```bash
python stage6_create_output_video.py --config configs/pipeline_config.yaml
```

### 3. Watch Visualization Video
```bash
# Output: demo_data/outputs/kohli_nets/top_persons_visualization.mp4
# Note: Shows top 10 persons with their IDs labeled
```

### 4. Select Person Manually (Stage 7)
```bash
# After watching video, choose Person 7
python stage7_select_person.py --config configs/pipeline_config.yaml --person-id 7
```

### 5. Continue with Pose Estimation
```bash
# Use the generated primary_person.npz for downstream pose estimation
# (Your pose estimation stages would go here)
```

## Verbose Output

With `--verbose` flag, get detailed statistics:

```
📊 Detailed Statistics:
  Duration: 2016 frames over 2027 frame range
  Coverage: 99.5%
  Avg confidence: 0.847
  Avg bbox size: 234.5 x 512.3 pixels
```

## Integration Notes

- **Replaces Stage 5**: Run either Stage 5 (auto) OR Stage 7 (manual), not both
- **Output Format**: Same as Stage 5 - downstream stages work identically
- **Config Compatible**: Uses same config file as other pipeline stages
- **Standalone**: Can be run independently after Stages 1-4b complete

## Tips

1. **Always run Stage 6 first** to see which persons are available
2. **Use verbose mode** (`--verbose`) to verify you selected the right person
3. **Invalid ID trick**: Run with `--person-id 999` to list all available persons
4. **Check tracklets**: Person with more tracklets = more consistent tracking

## Selection Report Format

`selection_report.json`:
```json
{
  "selection_method": "manual",
  "selected_person_id": 3,
  "frame_count": 2016,
  "frame_range": [0, 2026],
  "tracklet_ids": [0, 3, 7, 10, 15, ...],
  "total_canonical_persons": 44
}
```
