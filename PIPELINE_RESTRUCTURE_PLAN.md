# Pipeline Restructuring Plan: Stage 3c → 3d, 4, 5

## Overview

**Current Problem**: All merging and ranking logic is scattered. Person filtering happens in Stage 3c, but it should be split into distinct concerns.

**Solution**: Restructure Stages 3c, 3d, 4, 5 to have clear responsibilities:
- **Stage 3c**: Filter & Crop Extraction (top 8 persons)
- **Stage 3d**: Visual Refinement via OSNet (merge duplicates: 8 → 6)
- **Stage 4**: Visualization (WebP + HTML)
- **Stage 5**: Manual Selection (user chooses person)

---

## Stage Architecture

### Stage 3a: Tracklet Analysis ✅ (UNCHANGED)
**Purpose**: Compute motion statistics for all tracklets
- **Input**: `tracklets_raw.npz` (from Stage 2)
- **Output**: `tracklet_stats.npz` (motion features)
- **Status**: Existing, no changes

### Stage 3b: Canonical Grouping ✅ (UNCHANGED)
**Purpose**: Merge tracklets into canonical persons using geometric heuristics
- **Input**: `tracklet_stats.npz` (from Stage 3a)
- **Output**: `canonical_persons.npz` (v1, all persons, ~40+)
- **Status**: Existing, no changes

### Stage 3c: Filter & Crop Extraction 🔄 (REFACTORED)
**Purpose**: Select top persons and extract crops
- **Input**: `canonical_persons.npz` (v1, all persons from 3b)
- **Output**:
  - `canonical_persons.npz` (v2, top 8 filtered)
  - `final_crops.pkl` (crops for 8 persons, ~50 crops each)
- **Logic**:
  1. Load all canonical persons
  2. Compute ranking scores for each:
     - Duration (longest presence)
     - Coverage (% of frames)
     - Center bias (proximity to frame center)
     - Smoothness (motion consistency)
  3. Apply late-appearance penalty (persons starting after 50% of video get penalty)
  4. Rank and select top 8
  5. Extract crops from video for top 8 persons
  6. Save filtered canonical_persons.npz and final_crops.pkl

### Stage 3d: Visual Refinement via OSNet 🆕 (NEW)
**Purpose**: Detect and merge person duplicates using ReID
- **Input**: `final_crops.pkl` (8 persons)
- **Output**:
  - `final_crops.pkl` (merged to ~6 persons)
  - `canonical_persons.npz` (v3, updated with merges)
- **Logic**:
  1. Load final_crops.pkl (8 persons)
  2. Load canonical_persons.npz (v2, 8 persons)
  3. Extract OSNet features for each person (averaged across crops)
  4. Find non-overlapping person pairs
  5. Compute cosine similarity on ReID features
  6. Find person chains using Union-Find:
     - Constraints: gap ≤ 60 frames, overlap ≤ 15 frames, similarity ≥ 0.60
     - Example: person_4 + person_40 → single person
  7. Merge person chains in canonical_persons.npz and final_crops.pkl
  8. Save updated files

### Stage 4: Visualization 🔄 (REFACTORED)
**Purpose**: Generate visual representation for user selection
- **Input**: `final_crops.pkl` (6 persons)
- **Output**:
  - WebP animations (6 persons × ~50 frames each)
  - HTML viewer with person gallery
- **Logic**:
  1. Load final_crops.pkl (6 persons)
  2. For each person:
     - Resize crops to fixed size (256×256)
     - Create WebP animation (50 frames @ 10ms each)
     - Generate HTML card with person info
  3. Generate HTML viewer with:
     - Gallery of 6 person cards
     - Each card shows WebP animation + metadata
     - Click handler for person selection (Stage 5)

### Stage 5: Manual Selection 🆕 (NEW)
**Purpose**: User selects person of interest
- **Input**: HTML viewer (from Stage 4)
- **Action**: User clicks person card
- **Output**: Selected person ID + data for pose estimation
- **Logic**:
  1. User opens HTML in browser
  2. User clicks on one of 6 person cards
  3. System extracts full data for selected person
  4. Data passed to pose estimation pipeline

---

## Data Flow Diagram

```
Stage 2: ByteTrack
    ↓ tracklets_raw.npz
Stage 3a: Tracklet Analysis
    ↓ tracklet_stats.npz
Stage 3b: Canonical Grouping
    ↓ canonical_persons.npz (v1, ~40+ persons)
Stage 3c: Filter & Crop Extraction
    ├─→ canonical_persons.npz (v2, 8 persons)
    ├─→ final_crops.pkl (8 persons, ~400 crops total)
    ↓
Stage 3d: Visual Refinement (OSNet)
    ├─→ canonical_persons.npz (v3, 6 persons, merged)
    ├─→ final_crops.pkl (6 persons, ~300 crops total, merged)
    ↓
Stage 4: Visualization
    ├─→ 6 × WebP animations
    ├─→ HTML viewer
    ↓
Stage 5: Manual Selection
    ├─→ User selects 1 person from HTML
    ├─→ Selected person data for pose estimation
```

---

## Config Changes Required

### pipeline_config.yaml

**Add Stage 3c section**:
```yaml
stage3c_filter:
  filtering:
    top_n_persons: 8              # Select top 8
    late_appearance_penalty: 0.3  # Penalty for persons starting after 50%
    crops_per_person: 50          # Extract 50 crops per person
  
  input:
    canonical_persons_file: ${outputs_dir}/${current_video}/canonical_persons.npz
  
  output:
    canonical_persons_file: ${outputs_dir}/${current_video}/canonical_persons_filtered.npz
    final_crops_file: ${outputs_dir}/${current_video}/final_crops.pkl
```

**Add Stage 3d section**:
```yaml
stage3d_refine:
  reid:
    enabled: true
    osnet_model: ${models_dir}/reid/osnet_x0_25_msmt17.onnx
    device: cuda
    num_best_crops: 16            # Extract features from top 16 crops per person
    
    # Merge constraints
    max_temporal_gap: 60          # Max gap between persons (frames)
    max_temporal_overlap: 15      # Max overlap allowed (frames)
    similarity_threshold: 0.60    # Min cosine similarity to merge
  
  input:
    final_crops_file: ${outputs_dir}/${current_video}/final_crops.pkl
    canonical_persons_file: ${outputs_dir}/${current_video}/canonical_persons_filtered.npz
  
  output:
    canonical_persons_file: ${outputs_dir}/${current_video}/canonical_persons.npz
    final_crops_file: ${outputs_dir}/${current_video}/final_crops.pkl
    reid_report_file: ${outputs_dir}/${current_video}/reid_merges.json
```

**Update Stage 4 section**:
```yaml
stage4_generate_html:
  enabled: true
  
  visualization:
    resize_to: [256, 256]
    webp_duration_ms: 100         # 10fps
    webp_quality: 80
  
  input:
    final_crops_file: ${outputs_dir}/${current_video}/final_crops.pkl
    canonical_persons_file: ${outputs_dir}/${current_video}/canonical_persons.npz
  
  output:
    output_dir: ${outputs_dir}/${current_video}/viewer/
    html_file: ${outputs_dir}/${current_video}/viewer/index.html
    webp_dir: ${outputs_dir}/${current_video}/viewer/webps/
```

**Update run_pipeline.py**:
- Add Stage 3c to stage list: `('Stage 3c: Filter & Crops', 'stage3c_filter_persons.py', 'stage3c')`
- Add Stage 3d to stage list: `('Stage 3d: Visual Refinement', 'stage3d_refine_visual.py', 'stage3d')`
- Update stage dependencies

---

## Implementation Checklist

### Phase 1: Refactor Stage 3c ⬜

- [ ] **1.1**: Understand current Stage 3c ranking logic
- [ ] **1.2**: Extract ranking logic (duration, coverage, center, smoothness)
- [ ] **1.3**: Implement late-appearance penalty
- [ ] **1.4**: Create stage3c_filter_persons.py
- [ ] **1.5**: Add Stage 3c config to pipeline_config.yaml
- [ ] **1.6**: Update run_pipeline.py to include Stage 3c
- [ ] **1.7**: Test Stage 3c on Colab (outputs: canonical_persons_filtered.npz + final_crops.pkl)

### Phase 2: Create Stage 3d ⬜

- [ ] **2.1**: Copy helper functions from tracklet_recovery_candidates.py (OSNet loading, feature extraction)
- [ ] **2.2**: Implement core functions:
  - Load final_crops.pkl + canonical_persons.npz
  - Extract OSNet features
  - Find non-overlapping pairs
  - Compute similarity
  - Build Union-Find graph
- [ ] **2.3**: Implement merging logic:
  - Update canonical_persons with merged persons
  - Update final_crops.pkl with merged crops
- [ ] **2.4**: Create stage3d_refine_visual.py
- [ ] **2.5**: Add Stage 3d config to pipeline_config.yaml
- [ ] **2.6**: Update run_pipeline.py to include Stage 3d
- [ ] **2.7**: Test Stage 3d on Colab (outputs: canonical_persons.npz + final_crops.pkl, merged)

### Phase 3: Refactor Stage 4 ⬜

- [ ] **3.1**: Remove ranking logic from current Stage 3c
- [ ] **3.2**: Update Stage 4 to read final_crops.pkl from output directory
- [ ] **3.3**: Implement WebP generation for 6 persons
- [ ] **3.4**: Update HTML template for person gallery
- [ ] **3.5**: Test Stage 4 on Colab (outputs: HTML + WebPs)

### Phase 4: Testing & Cleanup ⬜

- [ ] **4.1**: Run full pipeline (Stage 3a → 3b → 3c → 3d → 4) on Colab
- [ ] **4.2**: Verify person counts:
  - After 3b: ~40+ persons
  - After 3c: 8 persons
  - After 3d: 6 persons
  - After 4: 6 WebP files + HTML
- [ ] **4.3**: Verify OSNet merging works correctly
- [ ] **4.4**: Delete temporary debug scripts from snippets/
- [ ] **4.5**: Commit all changes to git

### Phase 5: Documentation ⬜

- [ ] **5.1**: Update README.md with new pipeline structure
- [ ] **5.2**: Create Stage 3d documentation
- [ ] **5.3**: Document OSNet merge constraints and thresholds

---

## Key Learnings & Constraints

1. **OSNet Model**: Use `osnet_x0_25_msmt17.onnx` (NOT x1_0, which collapses features)
2. **Temporal Constraints**: 
   - Max gap: 60 frames (don't try to merge if occlusion is too long)
   - Max overlap: 15 frames (small overlaps OK at tracklet boundaries)
3. **Similarity Threshold**: 0.60 (determined empirically from cricket data)
4. **Late Appearance Penalty**: Persons appearing after 50% mark get penalty (avoid short detections)
5. **Top-N Selection**: 8 persons balances between keeping enough diversity and manageable visualization

---

## Success Criteria

✅ **Phase 1 Success**: Stage 3c outputs top 8 persons with extracted crops
✅ **Phase 2 Success**: Stage 3d merges 8 → 6 persons (person_4+40, person_29+65)
✅ **Phase 3 Success**: Stage 4 generates 6 WebP animations + HTML viewer
✅ **Overall Success**: End-to-end pipeline runs without errors on Colab cricket video
