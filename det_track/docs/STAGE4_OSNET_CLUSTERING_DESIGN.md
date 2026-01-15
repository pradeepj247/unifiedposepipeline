# Stage 4: Enhanced HTML Viewer with OSNet Clustering

**Document Date:** January 15, 2026  
**Purpose:** Design specification for Stage 4 enhancement with integrated clustering function

---

## Overview

Stage 4 is enhanced to include **built-in ReID clustering** without creating a separate stage 4b.

**Key Principle:** Process video once, fork output into two paths (WebP + OSNet clustering)

---

## Architecture

### Single Stage, Two Output Paths

```
Stage 4: Generate HTML Viewer + Similarity Matrix

Input: canonical_persons.npz (8-10 canonical persons)
Output: 
  - webp_viewer/ (HTML + WebP animations)
  - similarity_matrix.json / .npy (10×10 matrix)
  - embeddings.json / .npy (person embeddings)

STEP 1: Sequential Video Processing
└─→ Extract and Fill Buckets
    ├─ Read video frame by frame
    ├─ For each frame in canonical_persons.npz:
    ├─ Extract person crops on-demand
    └─ Fill 50-crop buckets for top-10 canonical persons

STEP 2: Fork Logic (Process same buckets two ways)

  PATH 1: WebP Generation (Existing)
  └─→ create_webp_animations(buckets)
      ├─ For each person's 50-crop bucket:
      ├─ Resize crops to 256×256
      ├─ Compress with 80% quality
      ├─ Create animated WebP (100ms per frame)
      └─ Save to webp_viewer/

  PATH 2: OSNet Clustering (NEW)
  └─→ create_similarity_matrix(buckets)
      ├─ For each person's 50-crop bucket:
      ├─ select_best_crops() → 8 best crops
      ├─ extract_osnet_features() → (8, 256) feature matrix
      ├─ average() → (256,) mean features
      ├─ L2_normalize() → (256,) unit embedding
      ├─ Compute 10×10 cosine similarity matrix
      └─ Color-code by threshold (70%, 50%, etc.)

STEP 3: Generate Unified HTML Report
└─→ generate_html()
    ├─ Section 1: Top 10 persons (horizontal with WebPs)
    ├─ Section 2: Similarity heatmap (color-coded)
    └─ Section 3: Recommendations (pairs > 70% similarity)

STEP 4: Save All Outputs
└─→ webp_viewer/
    ├─ person_selection.html (main report)
    ├─ webp/
    │  ├─ person_0.webp
    │  ├─ person_1.webp
    │  └─ ... (10 WebPs)
    ├─ similarity_matrix.json
    ├─ similarity_matrix.npy
    ├─ embeddings.json
    └─ embeddings.npy
```

---

## Function Specifications

### 1. extract_and_fill_buckets(config, canonical_persons_npz, video_path)

**Purpose:** Sequential video read, on-demand crop extraction, bucket filling

**Input:**
- `config`: Pipeline configuration
- `canonical_persons_npz`: Path to canonical_persons.npz
- `video_path`: Path to canonical_video.mp4

**Output:**
```python
buckets = {
    0: [crop_1, crop_2, ..., crop_50],  # person 0: list of (H, W, 3) BGR images
    1: [crop_1, crop_2, ..., crop_50],  # person 1
    ...
    9: [crop_1, crop_2, ..., crop_50],  # person 9
}
```

**Logic:**
```
for frame_idx in range(total_frames):
    frame = video.read()
    for person_id in canonical_persons:
        if person_id has bbox in frame_idx:
            crop = extract_crop(frame, bbox)
            buckets[person_id].append(crop)
            if len(buckets[person_id]) >= 50:
                break  # bucket full
```

---

### 2. create_webp_animations(buckets, output_dir, quality=80, fps=10)

**Purpose:** Convert 50 crops per person into animated WebP

**Input:**
- `buckets`: Dict[person_id → list of 50 crops]
- `output_dir`: Where to save WebPs
- `quality`: Compression quality (default: 80)
- `fps`: Frames per second (default: 10)

**Output:**
```
webp_dir/
├─ person_0.webp (50 frames @ 10fps = 5 seconds)
├─ person_1.webp
└─ ... (10 WebPs)
```

**Logic:**
```
for person_id, crops in buckets.items():
    # Resize all crops to 256×256
    resized = [cv2.resize(crop, (256, 256)) for crop in crops]
    
    # Write animated WebP
    imageio.imwrite(
        f"person_{person_id}.webp",
        resized,
        duration=1000/fps,  # 100ms per frame
        quality=quality
    )
```

---

### 3. create_similarity_matrix(buckets, osnet_model, device='cuda')

**Purpose:** Extract OSNet embeddings and compute similarity matrix

**Input:**
- `buckets`: Dict[person_id → list of 50 crops]
- `osnet_model`: Loaded OSNet x0.25 model
- `device`: 'cuda' or 'cpu'

**Output:**
```python
{
    'similarity_matrix': (10, 10) float32 array,  # Cosine similarities [0-1]
    'embeddings': {
        0: (256,) float32 embedding,
        1: (256,) float32 embedding,
        ...
    },
    'person_ids': [0, 1, 2, ..., 9]
}
```

**Sub-functions:**

#### 3a. select_best_crops(crops, num=8)
```
Purpose: Select 8 representative crops from 50-crop bucket

Criteria:
- Exclude very small/large crops (size outliers)
- Prefer high-confidence detections (if metadata available)
- Avoid blurry crops (optional: use Laplacian variance)
- Random sample from remaining top candidates

Return: List of 8 crops
```

#### 3b. extract_osnet_features(crops, osnet_model, device)
```
Purpose: Extract feature vectors from 8 crops using OSNet

Logic:
- Preprocess crops: resize to 256×128, normalize (ImageNet stats)
- Stack into batch: (8, 3, 256, 128)
- Forward through OSNet: output (8, 256) features
- Return: (8, 256) numpy array

Note: OSNet x0.25 outputs 256-dim feature vectors
```

#### 3c. compute_embedding(features)
```
Purpose: Create single representative embedding from 8 features

Logic:
1. Average across 8 features: mean = (256,)
2. L2 normalize: embedding = mean / ||mean||_2
3. Return: (256,) unit vector
```

#### 3d. compute_similarity_matrix(embeddings_dict)
```
Purpose: Compute NxN cosine similarity between all persons

Logic:
1. Stack embeddings: (N, 256) matrix
2. For each pair (i, j):
   similarity[i,j] = dot(emb[i], emb[j]) / (||emb[i]|| * ||emb[j]||)
   (Already unit vectors, so just dot product)
3. Return: (10, 10) symmetric matrix with 1.0 on diagonal

Output format:
     P0   P1   P2  ...  P9
P0 [1.0  0.45 0.32 ... 0.18]
P1 [0.45 1.0  0.78 ... 0.22]
P2 [0.32 0.78 1.0  ... 0.31]
...
P9 [0.18 0.22 0.31 ... 1.0]
```

---

### 4. generate_html(webp_dir, similarity_matrix, embeddings, output_path)

**Purpose:** Create HTML report with WebPs + similarity heatmap

**Output:** `person_selection.html`

**Structure:**

```html
<html>
  <head>
    <title>Person Selection Report</title>
    <!-- Include Plotly for heatmap visualization -->
  </head>
  <body>
    <h1>Top 10 Canonical Persons</h1>
    
    <!-- SECTION 1: Horizontal person display with WebPs -->
    <div class="persons-grid">
      <div class="person-card">
        <h2>Person 0</h2>
        <img src="webp/person_0.webp" width="256" height="256">
        <p>Duration: 150 frames | Coverage: 42%</p>
      </div>
      <div class="person-card">
        <h2>Person 1</h2>
        <img src="webp/person_1.webp" width="256" height="256">
        <p>Duration: 140 frames | Coverage: 39%</p>
      </div>
      <!-- ... 10 persons ... -->
    </div>
    
    <!-- SECTION 2: Similarity heatmap -->
    <h2>Person Similarity Matrix</h2>
    <p>Values above 70% indicate likely duplicates (same person, different timeframe)</p>
    <div id="heatmap"></div>
    
    <!-- Plotly heatmap code -->
    <script>
      var data = [{
        z: similarity_matrix,
        x: ['P0', 'P1', ..., 'P9'],
        y: ['P0', 'P1', ..., 'P9'],
        type: 'heatmap',
        colorscale: 'RdYlGn',  // Red(low) → Yellow → Green(high)
      }];
      Plotly.newPlot('heatmap', data);
    </script>
    
    <!-- SECTION 3: Recommendations -->
    <h2>Recommendations</h2>
    <p>Pairs with >70% similarity (likely same person):</p>
    <ul>
      <li>Person 0 & Person 5: 78% similar</li>
      <li>Person 2 & Person 7: 72% similar</li>
      <li>Person 3 & Person 8: 68% similar (near threshold)</li>
    </ul>
  </body>
</html>
```

---

## Data Flow Diagram

```
canonical_persons.npz
        ↓
extract_and_fill_buckets()
        ↓
    buckets {person_id: [crop×50]}
        ↓
        ├─→ create_webp_animations() ──→ webp/
        │                                  ├─ person_0.webp
        │                                  └─ person_9.webp
        │
        └─→ create_similarity_matrix() ──→ embeddings.json
                                          ├─ similarity_matrix.json
                                          └─ similarity_matrix.npy
                                                    ↓
generate_html()
        ↓
person_selection.html (combines both outputs)
        ↓
User views HTML:
├─ Top 10 persons (WebP animations)
└─ Similarity heatmap (ReID recommendations)
        ↓
Stage 5: User manually selects persons based on visual + ReID recommendations
```

---

## Configuration

Add to `pipeline_config.yaml`:

```yaml
stage4_generate_html:
  # Input files
  video_file: ${outputs_dir}/${current_video}/canonical_video.mp4
  canonical_persons_file: ${outputs_dir}/${current_video}/canonical_persons.npz
  
  # Output directory
  output_dir: ${outputs_dir}/${current_video}/webp_viewer
  
  # WebP generation parameters
  webp_fps: 10                        # Frames per second
  webp_quality: 80                    # Quality (0-100)
  webp_resize_to: [256, 256]          # Resize crops to this size
  
  # OSNet clustering parameters
  clustering:
    enabled: true                     # Toggle on/off
    osnet_model: ${models_dir}/osnet/osnet_x0_25_msmt17.pth
    device: cuda                      # cuda or cpu
    num_best_crops: 8                 # Select 8 best from 50
    similarity_threshold: 0.70        # Highlight similar pairs
    
  # Logging
  log_file: ${outputs_dir}/${current_video}/stage4_generate_html.log
  verbose: false
  
  # Output paths
  output:
    webp_dir: ${outputs_dir}/${current_video}/webp_viewer
    html_file: ${outputs_dir}/${current_video}/webp_viewer/person_selection.html
    similarity_matrix_json: ${outputs_dir}/${current_video}/similarity_matrix.json
    similarity_matrix_npy: ${outputs_dir}/${current_video}/similarity_matrix.npy
    embeddings_json: ${outputs_dir}/${current_video}/embeddings.json
    embeddings_npy: ${outputs_dir}/${current_video}/embeddings.npy
```

---

## Implementation Notes

### Performance Considerations

1. **Video Reading:** Sequential, on-demand extraction (already optimized)
2. **OSNet Batch Size:** Fixed at 8 (matches num_best_crops)
3. **Clustering:** Minimal overhead (~1-2 seconds for 10 persons)
4. **Total Stage 4 Time:** ~6-8 seconds (vs ~6s without clustering)

### Error Handling

1. **Bucket underfilled:** If person has <50 crops, use available
2. **OSNet unavailable:** Graceful fallback (log warning, skip clustering)
3. **Similarity NaN:** Handle edge cases (all features identical, etc.)

### Output Format Specifications

**similarity_matrix.json:**
```json
{
  "similarity_matrix": [
    [1.0, 0.45, 0.32, ...],
    [0.45, 1.0, 0.78, ...],
    ...
  ],
  "person_ids": [0, 1, 2, ..., 9],
  "threshold_70": [[0, 5], [2, 7]],  // Pairs above 70%
  "timestamp": "2026-01-15T10:30:00Z"
}
```

**embeddings.json:**
```json
{
  "embeddings": {
    "0": [0.12, 0.34, -0.45, ...],  // (256,) values
    "1": [0.23, 0.15, -0.67, ...],
    ...
  },
  "model": "OSNet x0.25",
  "timestamp": "2026-01-15T10:30:00Z"
}
```

---

## Stage 5 Integration

Stage 5 (Person Selection) can now use similarity matrix:

```
User workflow:
1. View person_selection.html
2. See Top 10 WebPs
3. See similarity heatmap
4. Identify likely duplicates (>70%)
5. Select persons in Stage 5 command:
   python stage5_select_person.py --config ... --persons p0,p5,p7
```

---

## Summary

**Stage 4 Enhancement:**
- ✅ Single video read (efficient)
- ✅ Two output paths from same data (WebP + OSNet)
- ✅ No separate stage 4b needed
- ✅ Integrated clustering function
- ✅ Unified HTML report
- ✅ User-friendly recommendations
- ✅ Machine-readable similarity matrix

**Key Files:**
- `stage4_generate_html.py` (enhanced with clustering)
- `person_selection.html` (unified report)
- `similarity_matrix.json/npy` (for reference)
- `embeddings.json/npy` (for advanced use)
