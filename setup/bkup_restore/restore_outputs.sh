#!/bin/bash
# restore_outputs.sh - Restore pre-computed detection/tracking/selection outputs from Google Drive
#
# This copies outputs from Google Drive backup to Colab working directory
# Allows you to skip the 60+ second detection/tracking pipeline and go straight to pose estimation
#
# Usage:
#   ./restore_outputs.sh <video_name>
#
# Example:
#   ./restore_outputs.sh kohli_nets
#   ./restore_outputs.sh dance_sequence

if [ -z "$1" ]; then
    echo "❌ Error: Video name required"
    echo ""
    echo "Usage: ./restore_outputs.sh <video_name>"
    echo ""
    echo "Available videos in Google Drive:"
    ls -1 /content/drive/MyDrive/pipelineoutputs/ 2>/dev/null || echo "  (Google Drive not mounted or no backups found)"
    echo ""
    echo "💡 To see detailed list:"
    echo "   ./list_outputs.sh"
    exit 1
fi

VIDEO_NAME="$1"
GDRIVE_SOURCE="/content/drive/MyDrive/pipelineoutputs/${VIDEO_NAME}"
LOCAL_DEST="/content/unifiedposepipeline/demo_data/outputs/${VIDEO_NAME}"

echo "🔄 Restoring outputs for: ${VIDEO_NAME}"
echo "   Source: ${GDRIVE_SOURCE}"
echo "   Destination: ${LOCAL_DEST}"
echo ""

# Check if source exists
if [ ! -d "$GDRIVE_SOURCE" ]; then
    echo "❌ Error: Source directory not found in Google Drive"
    echo "   ${GDRIVE_SOURCE}"
    echo ""
    echo "Available videos:"
    ls -1 /content/drive/MyDrive/pipelineoutputs/ 2>/dev/null || echo "  (Google Drive not mounted)"
    echo ""
    echo "💡 To see detailed list:"
    echo "   ./list_outputs.sh"
    exit 1
fi

# Create destination directory
mkdir -p "$LOCAL_DEST"

# Copy outputs
echo "📦 Copying files from Google Drive..."
cp -r "${GDRIVE_SOURCE}"/* "${LOCAL_DEST}/"

if [ $? -eq 0 ]; then
    echo "✅ Restore complete!"
    echo ""
    echo "📊 Restored summary:"
    SIZE=$(du -sh "$LOCAL_DEST" 2>/dev/null | cut -f1)
    NUM_FILES=$(find "$LOCAL_DEST" -type f | wc -l)
    echo "   Total size: ${SIZE}"
    echo "   Total files: ${NUM_FILES}"
    echo ""
    echo "📁 Key files restored:"
    [ -f "$LOCAL_DEST/selected_person.npz" ] && echo "   ✅ selected_person.npz (ready for pose detection)"
    [ -f "$LOCAL_DEST/canonical_video.mp4" ] && echo "   ✅ canonical_video.mp4"
    [ -f "$LOCAL_DEST/canonical_persons_3c.npz" ] && echo "   ✅ canonical_persons_3c.npz"
    [ -f "$LOCAL_DEST/detections_raw.npz" ] && echo "   ✅ detections_raw.npz"
    [ -f "$LOCAL_DEST/tracklets_raw.npz" ] && echo "   ✅ tracklets_raw.npz"
    echo ""
    echo "🎯 Next steps:"
    echo "   1. Update configs/posedet.yaml with this video:"
    echo "      input_video_primary: demo_data/outputs/${VIDEO_NAME}/canonical_video.mp4"
    echo "      input_video_fallback: demo_data/videos/${VIDEO_NAME}.mp4"
    echo "      detections_file: demo_data/outputs/${VIDEO_NAME}/selected_person.npz"
    echo ""
    echo "   2. Run pose detection:"
    echo "      python run_posedet.py --config configs/posedet.yaml"
    echo ""
    echo "   Or use the quick command:"
    echo "      python run_posedet.py --config configs/posedet.yaml --video ${VIDEO_NAME}"
else
    echo "❌ Error during restore"
    exit 1
fi
