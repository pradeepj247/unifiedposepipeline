"""
Minimal test - just read video and save frame count
"""
import cv2
import pickle
import sys
from pathlib import Path

video_path = "demo_data/outputs/kohli_nets/canonical_video.mp4"
output_path = "demo_data/outputs/kohli_nets/test_output.pkl"

print(f"Opening video: {video_path}")
sys.stdout.flush()

cap = cv2.VideoCapture(video_path)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 100 == 0:
        print(f"Frame {count}")
        sys.stdout.flush()

cap.release()

print(f"\nTotal frames read: {count}")
print(f"Saving to: {output_path}")
sys.stdout.flush()

# Try to save
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'wb') as f:
    pickle.dump({'frame_count': count, 'test': 'success'}, f)

file_size = Path(output_path).stat().st_size
print(f"✅ Saved! Size: {file_size} bytes")
print(f"Verify: ls -lh {output_path}")
