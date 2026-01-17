# Stage 4: HTML Viewer Generation

**Implementation**: `stage4_generate_html.py`

## Purpose
Generate interactive HTML viewer with WebP animations showing crops from both Stage 3c (before ReID) and Stage 3d (after ReID) for comparison.

## Inputs
- `final_crops_3c.pkl`: Crops from Stage 3c (8-10 persons)
- `canonical_persons_3c.npz`: Person records from Stage 3c
- `final_crops_3d.pkl`: Crops from Stage 3d (7-8 persons, optional if Stage 3d enabled)
- `canonical_persons_3d.npz`: Person records from Stage 3d (optional)
- `merging_report.json`: OSNet merge report (optional)
- `canonical_video.mp4`: For frame metadata

## Outputs
- `webp_viewer/`: Directory containing all generated files
- `webp_viewer/person_selection.html`: Interactive HTML viewer
- `webp_viewer/person_3c_*.webp`: Animated WebP files for Stage 3c persons (200ms per frame)
- `webp_viewer/person_3d_*.webp`: Animated WebP files for Stage 3d persons (if Stage 3d enabled)

## Processing Flow

```
final_crops_3c.pkl + canonical_persons_3c.npz (Stage 3c: 8-10 persons)
final_crops_3d.pkl + canonical_persons_3d.npz (Stage 3d: 7-8 persons, optional)
    ↓
For each person in Stage 3c:
    ├─→ Extract crops (60 per person)
    ├─→ Resize to 256×256 (consistent dimensions)
    ├─→ Encode as WebP (200ms per frame = 5 FPS, 60 frames = 12s loop)
    └─→ Save to webp_viewer/person_3c_N.webp
    ↓
For each person in Stage 3d (if enabled):
    ├─→ Extract crops (60 per person)
    ├─→ Resize to 256×256
    ├─→ Encode as WebP (200ms per frame)
    └─→ Save to webp_viewer/person_3d_N.webp
    ↓
Generate HTML with 2 rows (3c vs 3d) or 1 row (3c only)
    ↓
webp_viewer/person_selection.html (interactive viewer)
Generate HTML viewer with:
    ├─→ Animated WebP carousels
    ├─→ Person selection buttons
    ├─→ Ranking badges
    └─→ Metadata display
    ↓
person_selection_slideshow.html (5.09 MB)
```

## Performance

| Metric | Value |
|--------|-------|
| Time | 2.51s |
| Persons processed | 10 |
| WebPs generated | 10 |
| HTML output | 5.09 MB |
| WebP duration | 12 seconds (60 × 200ms) |

## WebP Encoding

### Frame Duration Selection
```
100ms/frame: 6s total (too fast, hard to follow)
200ms/frame: 12s total ✅ SELECTED (comfortable viewing)
300ms/frame: 18s total (acceptable but slow)
700ms/frame: 42s total (unwatchable, boring)
```

**Why 200ms?** User testing showed 5 FPS is comfortable for person recognition.

### WebP Quality Settings
```python
import imageio

imageio.mimsave(
    output_path,
    frames,           # List of numpy arrays
    format='webp',
    duration=0.2,     # 200ms per frame
    quality=90        # Good quality/size tradeoff
)
```

### Dimension Consistency
**Critical**: All frames must have identical dimensions.

```python
# Ensure consistent sizing
target_height, target_width = 256, 256  # Fixed size

resized_frames = []
for crop in person_crops:
    resized = cv2.resize(crop, (target_width, target_height))
    resized_frames.append(resized)

imageio.mimsave(output_path, resized_frames)
```

**Why fixed size?**
- Prevents dimension mismatch errors
- Consistent file sizes
- Better HTML layout

## HTML Viewer Structure

### Generated HTML
```html
<!DOCTYPE html>
<html>
<head>
    <title>Person Selection Viewer</title>
    <style>
        .person-card {
            border: 2px solid #ccc;
            padding: 15px;
            margin: 10px;
        }
        .webp-container {
            width: 256px;
            height: 256px;
        }
        .selection-btn {
            background: #007bff;
            color: white;
            padding: 10px;
        }
    </style>
</head>
<body>
    <h1>Person Selection</h1>
    
    <div class="persons-grid">
        <!-- For each person -->
        <div class="person-card">
            <h3>Person 1</h3>
            <img src="webp_viewer/person_1.webp" class="webp-container" loop>
            <p>Rank: #1 | Duration: 120 frames | Confidence: 0.92</p>
            <button class="selection-btn" onclick="select_person(1)">
                Select This Person
            </button>
        </div>
        
        <!-- ... more persons ... -->
    </div>
    
    <script>
        function select_person(person_id) {
            console.log("Selected person: " + person_id);
            // Send to backend or next stage
        }
    </script>
</body>
</html>
```

## Dual-Row vs Single-Row Output

### Single Row (Fast Mode, Stage 3d Disabled)
```
Person 1  Person 2  Person 3  Person 4  Person 5
Person 6  Person 7  Person 8  Person 9  Person 10
```
Each row: 5 WebPs (256×256 each)

### Dual Row (Full Mode, Stage 3d Enabled)
```
Top Row:    [Person Rank #1 - High Confidence]
Bottom Row: [Person Rank #2 - High Confidence]
            [Person Rank #3] [Person Rank #4] ... [Person Rank #10]
```

Highlights top 2 persons with larger thumbnails.

## Configuration

```yaml
stage4_html:
  webp:
    duration_ms: 200           # Frame duration
    quality: 90                # 0-100 (higher = better)
    target_width: 256          # Fixed dimensions
    target_height: 256
  
  html:
    dual_row: true             # Highlight top 2 persons
    include_metadata: true     # Show rank, duration, confidence
    responsive_layout: true    # Mobile-friendly
```

## Output Files

```
demo_data/outputs/kohli_nets/
├── webp_viewer/
│   ├── person_0.webp        (400-500 KB each)
│   ├── person_1.webp
│   ├── ...
│   └── person_9.webp
├── person_selection_slideshow.html  (5 MB total)
└── person_selection_slideshow.gif   (optional, larger)
```

## Design Decisions

### Why WebP Instead of MP4?
| Format | Pros | Cons |
|--------|------|------|
| **WebP** | Native browser support, loops smoothly, smaller files | Limited control |
| **MP4** | Highest compression, precise seeking | Requires plugins, auto-loop issues |
| **GIF** | Universal, old-school | Huge file sizes (100+ MB) |

**Selected**: WebP (good balance of compatibility and size)

### Why 60 Crops per Person?
- **Too few (<30)**: Misses interesting moments
- **Optimal (60)**: Captures diverse poses + expressions, 12s viewing time
- **Too many (>100)**: Diminishing returns, larger files

### Why Embed WebPs vs Link?
**Embedded**:
- Single HTML file (portable)
- No external dependencies
- Works offline

**vs Linked**:
- Smaller HTML, faster loading
- Can update WebPs independently

**Selected**: Embedded (portability more important)

## Auto-Cleanup

After successful HTML generation:
```python
# Delete temporary WebP files (already embedded)
if success:
    shutil.rmtree('webp_viewer')
    print("Cleaned up temporary WebP files")
```

**Why?** WebPs are embedded in HTML, keeping separate files is redundant.

## Performance Notes

- **WebP encoding**: 0.2s per person (GPU-accelerated if available)
- **HTML generation**: 0.1s (DOM construction)
- **File I/O**: 2.2s (writing 5+ MB to disk)
- **Bottleneck**: Disk write speed

## Common Issues

### Issue: "Dimensions mismatch" Error
**Cause**: Crops have different sizes (e.g., 240×180, 300×200)  
**Solution**: Stage 3c 3-bin selection ensures consistent ~256×256 size

### Issue: WebP Doesn't Loop
**Cause**: Browser doesn't support animated WebP  
**Solution**: Fallback to GIF format (generated separately)

### Issue: HTML File Too Large
**Cause**: High WebP quality + large dimensions  
**Solution**: Reduce quality (85 instead of 90) or dimensions (192×192)

## Next Steps (Stage 5+)

After HTML generation:
1. **User selects person** from interactive viewer
2. **Stage 5**: Extract all bboxes for selected person
3. **Stage 6**: Run 2D pose estimation (RTMPose/ViTPose)
4. **Stage 7**: Run 3D pose lifting (HybrIK/MotionAGFormer)
5. **Stage 8+**: Biomechanics analysis

---

**Related**: [Back to Master](README_MASTER.md) | [← Stage 3d](STAGE3D_VISUAL_REFINEMENT.md)
