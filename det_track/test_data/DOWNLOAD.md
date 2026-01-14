# Download canonical_persons.npz

## Option 1: Manual Download from Google Drive

1. Open Google Drive: https://drive.google.com/drive/folders/
2. Navigate to: `MyDrive/pipelineoutputs/kohli_nets/`
3. Find file: `canonical_persons.npz` (169 KB)
4. Download it
5. Save to: `D:\trials\unifiedpipeline\newrepo\det_track\test_data\canonical_persons.npz`

## Option 2: Use Colab to Download

Run this in Google Colab:

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy from Drive to Colab
!cp /content/drive/MyDrive/pipelineoutputs/kohli_nets/canonical_persons.npz /content/

# Download to your local machine
from google.colab import files
files.download('/content/canonical_persons.npz')
```

Then move the downloaded file to:
`D:\trials\unifiedpipeline\newrepo\det_track\test_data\canonical_persons.npz`

## What's in this file?

`canonical_persons.npz` contains:
- Top 10 detected persons
- Their frame numbers (when they appear)
- Their bboxes for each frame
- Total size: 169 KB (tiny!)

This is all we need to test on-demand extraction!
