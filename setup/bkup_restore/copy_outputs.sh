#!/bin/bash
# copy_outputs.sh - Backup detection/tracking/selection outputs to Google Drive
#
# This copies outputs from Colab working directory to Google Drive for persistence
#
# Usage:
#   ./copy_outputs.sh <video_name>
#
# Example:
#   ./copy_outputs.sh kohli_nets
#   ./copy_outputs.sh dance_sequence

if [ -z "$1" ]; then
    echo "❌ Error: Video name required"
    echo ""
    echo "Usage: ./copy_outputs.sh <video_name>"
    echo ""
    echo "Available videos in Colab:"
    ls -1 /content/unifiedposepipeline/demo_data/outputs/ 2>/dev/null || echo "  (No outputs directory found)"
    exit 1
fi

VIDEO_NAME="$1"
LOCAL_SOURCE="/content/unifiedposepipeline/demo_data/outputs/${VIDEO_NAME}"
GDRIVE_DEST="/content/drive/MyDrive/pipelineoutputs/${VIDEO_NAME}"

echo "💾 Backing up outputs for: ${VIDEO_NAME}"
echo "   Source: ${LOCAL_SOURCE}"
echo "   Destination: ${GDRIVE_DEST}"
echo ""

# Check if source exists
if [ ! -d "$LOCAL_SOURCE" ]; then
    echo "❌ Error: Source directory not found"
    echo "   ${LOCAL_SOURCE}"
    echo ""
    echo "Available videos:"
    ls -1 /content/unifiedposepipeline/demo_data/outputs/ 2>/dev/null || echo "  (No outputs found)"
    exit 1
fi

# Check if Google Drive is mounted
if [ ! -d "/content/drive/MyDrive" ]; then
    echo "❌ Error: Google Drive not mounted"
    echo ""
    echo "To mount Google Drive, run:"
    echo "   from google.colab import drive"
    echo "   drive.mount('/content/drive')"
    exit 1
fi

# Create destination directory
mkdir -p "$GDRIVE_DEST"

# Copy outputs
echo "📦 Copying files to Google Drive..."
cp -r "${LOCAL_SOURCE}"/* "${GDRIVE_DEST}/"

if [ $? -eq 0 ]; then
    echo "✅ Backup complete!"
    echo ""
    echo "📊 Backup summary:"
    SIZE=$(du -sh "$GDRIVE_DEST" 2>/dev/null | cut -f1)
    NUM_FILES=$(find "$GDRIVE_DEST" -type f | wc -l)
    echo "   Total size: ${SIZE}"
    echo "   Total files: ${NUM_FILES}"
    echo ""
    echo "📁 Key files backed up:"
    [ -f "$GDRIVE_DEST/selected_person.npz" ] && echo "   ✅ selected_person.npz"
    [ -f "$GDRIVE_DEST/canonical_video.mp4" ] && echo "   ✅ canonical_video.mp4"
    [ -f "$GDRIVE_DEST/canonical_persons_3c.npz" ] && echo "   ✅ canonical_persons_3c.npz"
    [ -f "$GDRIVE_DEST/detections_raw.npz" ] && echo "   ✅ detections_raw.npz"
    [ -f "$GDRIVE_DEST/tracklets_raw.npz" ] && echo "   ✅ tracklets_raw.npz"
    echo ""
    echo "💡 To restore this backup later:"
    echo "   ./restore_outputs.sh ${VIDEO_NAME}"
else
    echo "❌ Error during backup"
    exit 1
fi
