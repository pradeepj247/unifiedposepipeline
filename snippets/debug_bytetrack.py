"""Debug script to test ByteTrack input format"""
import numpy as np

# Load detections
data = np.load('D:/trials/unifiedpipeline/detections_raw.npz', allow_pickle=True)

print("=== Detection Data ===")
print(f"Total detections: {len(data['frame_numbers'])}")
print(f"\nFirst detection:")
print(f"  frame: {data['frame_numbers'][0]}")
print(f"  bbox: {data['bboxes'][0]}")
print(f"  conf: {data['confidences'][0]}")
print(f"  class: {data['classes'][0]}")

# Reconstruct per-frame (like stage2 does)
frame_numbers = data['frame_numbers']
bboxes = data['bboxes']
confidences = data['confidences']
classes = data['classes']

unique_frames = np.unique(frame_numbers)
print(f"\nUnique frames with detections: {len(unique_frames)}")

# Test first frame
frame_id = unique_frames[0]
mask = frame_numbers == frame_id
frame_bboxes = bboxes[mask]
frame_confs = confidences[mask]
frame_classes = classes[mask]

print(f"\nFrame {frame_id}:")
print(f"  Detections: {len(frame_bboxes)}")
print(f"  bboxes shape: {frame_bboxes.shape}")
print(f"  confs shape: {frame_confs.shape}")
print(f"  classes shape: {frame_classes.shape}")

# Test column_stack (what we're doing)
print("\n=== Testing column_stack ===")
print(f"Before reshape:")
print(f"  confs: {frame_confs.shape}")
print(f"  classes: {frame_classes.shape}")

confs_reshaped = frame_confs.reshape(-1, 1)
classes_reshaped = frame_classes.reshape(-1, 1)

print(f"\nAfter reshape:")
print(f"  confs: {confs_reshaped.shape}")
print(f"  classes: {classes_reshaped.shape}")

dets_for_tracker = np.column_stack([
    frame_bboxes,
    confs_reshaped,
    classes_reshaped
])

print(f"\nFinal tracker input:")
print(f"  Shape: {dets_for_tracker.shape}")
print(f"  Dtype: {dets_for_tracker.dtype}")
print(f"  First row: {dets_for_tracker[0]}")
print(f"  Expected: [x1, y1, x2, y2, conf, cls]")

# Test with ByteTrack
print("\n=== Testing ByteTrack ===")
try:
    from boxmot import ByteTrack
    
    tracker = ByteTrack(
        det_thresh=0.3,
        track_thresh=0.45,
        match_thresh=0.8,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        device='cpu',
        half=False
    )
    
    print(f"✅ ByteTrack initialized")
    print(f"Updating with detections shape: {dets_for_tracker.shape}")
    
    tracked = tracker.update(dets_for_tracker, None)
    
    print(f"\nByteTrack output:")
    print(f"  Shape: {tracked.shape if len(tracked) > 0 else 'empty'}")
    if len(tracked) > 0:
        print(f"  First track: {tracked[0]}")
        print(f"  Track IDs: {tracked[:, 4]}")
    else:
        print(f"  ⚠️ No tracks returned!")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
