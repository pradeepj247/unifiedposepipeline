# Demo Data Files Guide

## Required Demo Files

### For Local Development

If you have access to the original files, place them in the following locations:

#### Videos

1. **dance.mp4** (Recommended for testing)
   - **Source**: `/content/drive/MyDrive/HybrIK_TRT_Backups/demodata/dance.mp4` (Google Drive)
   - **Local Path**: `demo_data/videos/dance.mp4`
   - **Size**: ~50MB
   - **Description**: Dance sequence video for testing pose tracking and 3D pose estimation

#### Images

1. **sample_image.jpg** (Auto-downloaded)
   - Automatically downloaded from COCO dataset
   - Used for single-image pose estimation tests

## Setup Methods

### Method 1: Automated Setup (Recommended)

```bash
# Run the demo data setup script
python setup_demo_data.py
```

This script will:
- ✅ Create directory structure
- ✅ Download available demo files
- ✅ Copy files from Google Drive (if in Colab)
- ✅ Create README files

### Method 2: Manual Setup

1. Create the directory structure:
   ```bash
   mkdir -p demo_data/videos
   mkdir -p demo_data/images
   mkdir -p demo_data/outputs
   ```

2. Copy your test files:
   ```bash
   # Copy dance video (if you have it)
   cp /path/to/your/dance.mp4 demo_data/videos/
   
   # Or use your own videos
   cp /path/to/your/video.mp4 demo_data/videos/
   ```

3. For images:
   ```bash
   # Copy test images
   cp /path/to/your/images/*.jpg demo_data/images/
   ```

### Method 3: Google Colab

If running in Google Colab with access to the original Drive backup:

```python
# In Colab notebook
from google.colab import drive
drive.mount('/content/drive')

# Copy demo files
!cp /content/drive/MyDrive/HybrIK_TRT_Backups/demodata/dance.mp4 demo_data/videos/
```

## Using Your Own Demo Files

You can use any video or image files for testing:

### Supported Video Formats
- `.mp4` (recommended)
- `.avi`
- `.mov`
- `.mkv`

### Supported Image Formats
- `.jpg` / `.jpeg` (recommended)
- `.png`
- `.bmp`

### Recommended Video Characteristics
- Resolution: 720p or 1080p
- Frame rate: 24-30 fps
- Duration: 5-60 seconds (for quick testing)
- Content: Clear view of people with visible body poses

## Alternative Demo Videos

If you don't have access to the original `dance.mp4`, you can use any video with people:

### Free Resources:
1. **Pexels** (https://www.pexels.com/videos/)
   - Search for "people walking", "dance", "exercise"
   - Free to use, no attribution required

2. **COCO Dataset**
   - Download sample images from COCO dataset
   - https://cocodataset.org/

3. **Your Own Videos**
   - Record your own video with a phone/camera
   - Ensure good lighting and clear view of people

## Verification

After setup, verify your demo data:

```bash
python setup_demo_data.py
```

Or manually check:

```bash
ls -lh demo_data/videos/
ls -lh demo_data/images/
```

## Troubleshooting

### Issue: Files not copying from Google Drive

**Solution**: 
1. Ensure Drive is mounted: `drive.mount('/content/drive')`
2. Verify path exists: `ls /content/drive/MyDrive/HybrIK_TRT_Backups/demodata/`
3. Check permissions on Drive files

### Issue: Video won't play or process

**Solution**:
1. Check video codec: `ffmpeg -i demo_data/videos/your_video.mp4`
2. Re-encode if needed: `ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4`
3. Verify file isn't corrupted: Try playing in VLC or other media player

### Issue: Out of disk space

**Solution**:
1. Use shorter videos for testing
2. Reduce video resolution/quality
3. Delete output files after testing

## Notes

- Demo files are NOT included in the git repository (too large)
- Add your demo files to `.gitignore` to avoid committing large files
- Outputs will be saved to `demo_data/outputs/` by default
- Clean up outputs periodically to save space

## Need Help?

Check the main README.md or open an issue on GitHub.
