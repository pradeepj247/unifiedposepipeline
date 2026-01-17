# run_pipeline.py - Execution Orchestrator

**Location**: `det_track/run_pipeline.py`

## Purpose
Central orchestrator that:
1. Loads and validates YAML configuration
2. Resolves all `${variable}` paths
3. Executes stages in correct order
4. Handles caching and dependencies
5. Reports timing and status

## Quick Start

### Basic Execution
```bash
cd det_track
python run_pipeline.py --config configs/pipeline_config.yaml
```

### Run Specific Stages Only
```bash
# Run only tracking (skip detection)
python run_pipeline.py --config configs/pipeline_config.yaml --stages 2

# Run all analysis + output
python run_pipeline.py --config configs/pipeline_config.yaml --stages 3a,3b,3c,3d,4

# Run detection through HTML generation
python run_pipeline.py --config configs/pipeline_config.yaml --stages 1,2,3a,3b,3c,3d,4
```

### Force Re-Run (Skip Cache)
```bash
# Re-generate HTML even if already exists
python run_pipeline.py --config configs/pipeline_config.yaml --stages 4 --force

# Re-run entire pipeline
python run_pipeline.py --config configs/pipeline_config.yaml --force
```

## Command-Line Arguments

```
python run_pipeline.py [OPTIONS]

OPTIONS:
  --config FILE         Path to pipeline_config.yaml (required)
  --stages STAGE_LIST   Comma-separated stages to run (e.g., "1,2,3a,3b,3c,4")
                        If omitted, runs all enabled stages from config
  --force               Force re-run even if outputs exist (skip cache)
  --help                Show this help message
```

## Configuration Loading & Path Resolution

### Step 1: Load YAML
```python
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
```

### Step 2: Resolve Variables (Multi-Pass)
```python
def resolve_path_variables(config):
    """Recursively resolve ${variable} in config"""
    global_vars = config.get('global', {})
    
    # Multi-pass resolution (handles nested variables)
    for _ in range(10):  # Max 10 iterations
        resolved = {}
        changed = False
        for key, value in global_vars.items():
            if isinstance(value, str) and '${' in value:
                resolved_value = re.sub(
                    r'\$\{(\w+)\}',
                    lambda m: str(global_vars.get(m.group(1), m.group(0))),
                    value
                )
                resolved[key] = resolved_value
                changed = changed or (resolved_value != value)
        
        if not changed:
            break
    
    return resolved
```

**Example Resolution**:
```
Initial:  repo_root = /content/unifiedposepipeline
          models_dir = ${repo_root}/models

Pass 1:   models_dir = /content/unifiedposepipeline/models ✓
```

### Step 3: Extract Current Video Name
```python
# auto-extract from video_file (e.g., "kohli_nets.mp4" → "kohli_nets")
video_file = config['global']['video_file']
current_video = os.path.splitext(video_file)[0]
config['current_video'] = current_video
```

Result: All subsequent stages use `${outputs_dir}/${current_video}/` for outputs.

## Stage Execution

### Stage Ordering & Dependencies
```
[0] Normalize Video (independent)
    ↓
[1] Detection (uses normalized video)
    ↓
[2] Tracking (uses detections)
    ↓
[3a] Tracklet Analysis (uses tracklets)
    ↓
[3b] Canonical Grouping (uses stats)
    ↓
[3c] Filter Persons (uses canonical)
    ↓
[3d] Visual Refinement (uses tracklets)
    ↓
[4] HTML Generation (uses crops)
```

### Runtime Stage Selection
```python
# If --stages specified, use those
if args.stages:
    requested_stages = args.stages.split(',')
    stages_to_run = [s.strip() for s in requested_stages]
else:
    # Otherwise, use enabled stages from config
    stages_to_run = []
    for stage in ['stage0', 'stage1', 'stage2', 'stage3a', 'stage3b', 'stage3c', 'stage3d', 'stage4']:
        if config['pipeline']['stages'].get(stage, False):
            stages_to_run.append(stage.replace('stage', ''))
```

### Cache Checking
```python
def should_run_stage(stage_name, output_files):
    """Check if outputs already exist"""
    if args.force:
        return True  # Force re-run
    
    for output_file in output_files:
        if not os.path.exists(output_file):
            return True  # Missing output, must run
    
    return False  # All outputs exist, skip
```

## Stage Invocation

Each stage is called via subprocess:

```python
def run_stage(stage_num, config):
    """Execute single stage"""
    stage_name = f"stage{stage_num}"
    stage_file = f"det_track/{stage_name}.py"
    
    if not os.path.exists(stage_file):
        print(f"⚠ Stage file not found: {stage_file}")
        return False
    
    # Build command
    cmd = [
        'python', stage_file,
        '--config', config_path,
        '--output-dir', output_dir
    ]
    
    # Run with timing
    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"❌ Stage {stage_num} failed")
        return False
    
    print(f"✅ Stage {stage_num} completed in {elapsed:.2f}s")
    return True
```

## Output & Timing Report

### During Execution
```
======================================================================
🎬 UNIFIED DETECTION & TRACKING PIPELINE
======================================================================
   Loaded config: configs/pipeline_config.yaml
   Running pipeline stages: stage1, stage2, stage3a, stage3b, stage3c, stage4

[Stage 1: Detection] ▐████████▌ 2025/2025 [00:48<00:00, 40.8 FPS]
[Stage 2: Tracking] ✓ 694.7 FPS
[Stage 3a: Analysis] ✓ 0.23s
[Stage 3b: Grouping] ✓ 0.47s
[Stage 3c: Filter] ✓ 0.95s
[Stage 4: HTML] ✓ 2.51s
```

### Final Summary
```
======================================================================
✅ PIPELINE COMPLETE!
======================================================================

⏱️  TIMING SUMMARY:
----------------------------------------------------------------------
Stage               Time    % of Total
------------------  ------  -----------
Stage 1: Detection  48.75s  79.3%
Stage 2: Tracking    7.91s  13.1%
Stage 3a: Analysis   0.23s   0.4%
Stage 3b: Grouping   0.47s   0.7%
Stage 3c: Filter     0.95s   1.5%
Stage 4: HTML        2.51s   3.9%
TOTAL               60.24s  100.0%

📦 Output Files:
  ✅ detections_raw.npz (0.16 MB)
  ✅ tracklets_raw.npz (0.18 MB)
  ✅ canonical_persons_3c.npz (0.13 MB)
  ✅ final_crops_3c.pkl (39.07 MB)
  ✅ person_selection_slideshow.html (5.09 MB)
  ✅ person_selection_slideshow.gif (12.5 MB)

======================================================================
```

## Error Handling

### Missing Configuration
```python
if not os.path.exists(config_path):
    print(f"❌ Config not found: {config_path}")
    sys.exit(1)
```

### Invalid Stage Specification
```python
valid_stages = ['0', '1', '2', '3a', '3b', '3c', '3d', '4']
for stage in requested_stages:
    if stage not in valid_stages:
        print(f"❌ Unknown stage: {stage}")
        sys.exit(1)
```

### Missing Stage File
```python
stage_file = f"det_track/stage{stage_num}.py"
if not os.path.exists(stage_file):
    print(f"❌ Stage implementation not found: {stage_file}")
    sys.exit(1)
```

### Stage Execution Failure
```python
result = subprocess.run(cmd, ...)
if result.returncode != 0:
    print(f"❌ Stage {stage_num} failed with exit code {result.returncode}")
    print("Continue? (y/n)")
    # Can continue or abort based on user input
```

## Practical Examples

### Run Fast Mode (60 crops, no ReID)
```bash
python run_pipeline.py --config configs/pipeline_config.yaml
# Config already set to fast mode
```

### Test Only Detection + Tracking
```bash
python run_pipeline.py --config configs/pipeline_config.yaml --stages 1,2
```

### Re-generate HTML After Tuning
```bash
python run_pipeline.py --config configs/pipeline_config.yaml --stages 4 --force
```

### Process New Video
```bash
# Edit pipeline_config.yaml: video_file: new_video.mp4
python run_pipeline.py --config configs/pipeline_config.yaml
```

### Full Pipeline with Timing
```bash
python run_pipeline.py --config configs/pipeline_config.yaml --stages 0,1,2,3a,3b,3c,3d,4
```

## Architecture

```
run_pipeline.py
├── Load YAML config
├── Resolve ${variables}
├── Parse --stages argument
│
├── For each stage:
│   ├── Check cache (skip if exists)
│   ├── Import stage module
│   ├── Call stage.main(config)
│   ├── Record timing
│   └── Handle errors
│
├── Aggregate timing
└── Print summary report
```

## Key Design Decisions

### Why Subprocess Execution?
- **Isolation**: Each stage runs in separate Python process
- **Memory**: Cleans up memory after each stage (no accumulation)
- **Fault tolerance**: One stage failure doesn't crash pipeline
- **Flexibility**: Stages can be mixed/matched easily

### Why ${variable} Resolution?
- **Portability**: Single config works on Colab + Windows + Linux
- **DRY**: Define paths once, reference throughout
- **Maintainability**: Changing repo_root updates all derived paths

### Why Cache Checking?
- **Efficiency**: Don't re-run expensive stages unnecessarily
- **Debugging**: Can re-run single stages without full pipeline
- **Iteration**: Developers can tweak Stage 4 without re-detecting

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `Config not found` | Wrong path | Use `--config configs/pipeline_config.yaml` |
| `Unknown stage: 5` | Invalid stage number | Use 0-4 or 3a-3d |
| `Stage X failed` | Stage has error | Check stage output for details, use --force to retry |
| `${repo_root}` in path | Unresolved variable | Check YAML syntax, ensure global section defined |
| Missing dependencies | Import error in stage | `pip install package_name` |

---

**Related**: [Back to Master](README_MASTER.md) | [← Config Reference](PIPELINE_CONFIG_REFERENCE.md)
