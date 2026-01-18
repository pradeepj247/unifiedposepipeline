#!/bin/bash
# list_outputs.sh - List all video outputs stored in Google Drive backup location
#
# Usage:
#   ./list_outputs.sh

GDRIVE_ROOT="/content/drive/MyDrive/pipelineoutputs"

echo "📂 Listing all video outputs in Google Drive backup:"
echo "   ${GDRIVE_ROOT}/"
echo ""

if [ ! -d "$GDRIVE_ROOT" ]; then
    echo "❌ Google Drive not mounted or backup directory doesn't exist"
    echo ""
    echo "To mount Google Drive, run:"
    echo "   from google.colab import drive"
    echo "   drive.mount('/content/drive')"
    echo ""
    echo "To create backup directory:"
    echo "   mkdir -p ${GDRIVE_ROOT}"
    exit 1
fi

# List all directories
ls -l "$GDRIVE_ROOT"

echo ""
echo "📊 Summary:"
NUM_VIDEOS=$(find "$GDRIVE_ROOT" -maxdepth 1 -type d | tail -n +2 | wc -l)
echo "   Total videos: ${NUM_VIDEOS}"

if [ $NUM_VIDEOS -gt 0 ]; then
    echo ""
    echo "💾 Detailed breakdown:"
    for dir in "$GDRIVE_ROOT"/*; do
        if [ -d "$dir" ]; then
            VIDEO_NAME=$(basename "$dir")
            SIZE=$(du -sh "$dir" 2>/dev/null | cut -f1)
            NUM_FILES=$(find "$dir" -type f | wc -l)
            
            echo ""
            echo "   📹 ${VIDEO_NAME}"
            echo "      Size: ${SIZE}"
            echo "      Files: ${NUM_FILES}"
            
            # Check for key files
            [ -f "$dir/selected_person.npz" ] && echo "      ✅ selected_person.npz"
            [ -f "$dir/canonical_video.mp4" ] && echo "      ✅ canonical_video.mp4"
            [ -f "$dir/canonical_persons_3c.npz" ] && echo "      ✅ canonical_persons_3c.npz"
        fi
    done
fi

echo ""
echo "💡 To restore a video:"
echo "   ./restore_outputs.sh <video_name>"
echo ""
echo "💡 To backup a video:"
echo "   ./copy_outputs.sh <video_name>"
