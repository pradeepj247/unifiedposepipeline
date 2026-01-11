#!/usr/bin/env python3
"""
Debug ID Switch Analysis

Analyzes tracking detections to understand why track IDs switch.
Automatically finds the switch point and provides parameter recommendations.

Usage:
    # Auto-detect switch point (recommended):
    python debug_id_switch.py --detections demo_data/outputs/raw_detections_debug.npz \
                               --old-id 3 --new-id 24
    
    # With frame extraction:
    python debug_id_switch.py --detections demo_data/outputs/raw_detections_debug.npz \
                               --old-id 3 --new-id 24 \
                               --video demo_data/videos/kohli_nets.mp4 \
                               --extract-frames
"""

import argparse
import numpy as np
import cv2
from pathlib import Path


def load_detections_debug(npz_path):
    """Load detections from debug NPZ file (structured per-frame format)"""
    data = np.load(npz_path, allow_pickle=True)
    frame_numbers = data['frame_numbers']
    detections_per_frame = data['detections_per_frame']
    metadata = data['metadata'].item() if 'metadata' in data else {}
    
    return frame_numbers, detections_per_frame, metadata


def load_detections_flattened(npz_path):
    """Load detections from flattened NPZ file and reconstruct per-frame structure"""
    data = np.load(npz_path, allow_pickle=True)
    frame_numbers_flat = data['frame_numbers']
    bboxes_flat = data['bboxes']
    track_ids_flat = data['track_ids']
    scores_flat = data['scores']
    
    # Group by frame
    frame_to_detections = {}
    for i in range(len(frame_numbers_flat)):
        frame_idx = int(frame_numbers_flat[i])
        if frame_idx not in frame_to_detections:
            frame_to_detections[frame_idx] = []
        
        frame_to_detections[frame_idx].append({
            'track_id': int(track_ids_flat[i]),
            'bbox': bboxes_flat[i].tolist(),
            'confidence': float(scores_flat[i]),
            'class_id': 0  # Assume person class
        })
    
    # Convert to arrays
    max_frame = max(frame_to_detections.keys())
    frame_numbers = np.arange(max_frame + 1)
    detections_per_frame = []
    
    for frame_idx in frame_numbers:
        if frame_idx in frame_to_detections:
            detections_per_frame.append(frame_to_detections[frame_idx])
        else:
            detections_per_frame.append([])
    
    return frame_numbers, np.array(detections_per_frame, dtype=object), {}


def load_detections(npz_path):
    """
    Smart loader: Try debug format first, fall back to flattened format
    """
    npz_path = Path(npz_path)
    
    # Try debug format first
    debug_path = npz_path.parent / npz_path.name.replace('raw_detections.npz', 'raw_detections_debug.npz')
    
    if debug_path.exists() and debug_path != npz_path:
        print(f"✓ Found debug format: {debug_path}")
        return load_detections_debug(debug_path)
    elif npz_path.name.endswith('_debug.npz'):
        print(f"✓ Loading debug format: {npz_path}")
        return load_detections_debug(npz_path)
    else:
        print(f"⚠️  Debug format not found, reconstructing from flattened format: {npz_path}")
        print(f"   (Run with --debug flag for better performance)\n")
        return load_detections_flattened(npz_path)


def find_id_switch_point(frame_numbers, detections_per_frame, old_id, new_id):
    """
    Automatically find where old_id disappears and new_id appears
    
    Returns:
        (last_old_frame, first_new_frame, switch_frame)
    """
    last_old_frame = None
    first_new_frame = None
    
    for frame_idx, detections in zip(frame_numbers, detections_per_frame):
        frame_ids = [det['track_id'] for det in detections]
        
        if old_id in frame_ids:
            last_old_frame = frame_idx
        
        if new_id in frame_ids and first_new_frame is None:
            first_new_frame = frame_idx
    
    if last_old_frame is None:
        print(f"⚠️  Warning: Old ID {old_id} not found in any frame!")
    if first_new_frame is None:
        print(f"⚠️  Warning: New ID {new_id} not found in any frame!")
    
    # Switch frame is approximately where old disappears / new appears
    if last_old_frame is not None and first_new_frame is not None:
        switch_frame = (last_old_frame + first_new_frame) // 2
    elif last_old_frame is not None:
        switch_frame = last_old_frame
    elif first_new_frame is not None:
        switch_frame = first_new_frame
    else:
        switch_frame = 0
    
    return last_old_frame, first_new_frame, switch_frame


def analyze_id_switch(frame_numbers, detections_per_frame, old_id, new_id, switch_frame=None, context_frames=20):
    """
    Analyze what happened around an ID switch
    
    Args:
        frame_numbers: Array of frame indices
        detections_per_frame: List of detection lists per frame
        old_id: Original track ID that disappeared
        new_id: New track ID that appeared
        switch_frame: Frame number where switch occurred (auto-detected if None)
        context_frames: Number of frames before/after to analyze
    """
    # Auto-detect switch point if not provided
    if switch_frame is None:
        last_old, first_new, switch_frame = find_id_switch_point(frame_numbers, detections_per_frame, old_id, new_id)
        print(f"\n🔍 Auto-detected switch point:")
        print(f"   Last appearance of ID {old_id}: Frame {last_old}")
        print(f"   First appearance of ID {new_id}: Frame {first_new}")
        print(f"   Analyzing around frame {switch_frame}\n")
    
    print(f"\n{'='*80}")
    print(f"ID SWITCH ANALYSIS: ID {old_id} → ID {new_id} around frame {switch_frame}")
    print(f"{'='*80}\n")
    
    
    # Find indices for analysis window
    start_frame = max(0, switch_frame - context_frames)
    end_frame = min(len(frame_numbers), switch_frame + context_frames)
    
    print(f"📊 Analyzing frames {start_frame} to {end_frame} (±{context_frames} from switch)\n")
    
    # Track presence and properties
    old_id_data = []
    new_id_data = []
    all_ids_per_frame = []
    
    for i in range(start_frame, end_frame):
        if i >= len(detections_per_frame):
            break
            
        frame_idx = frame_numbers[i]
        detections = detections_per_frame[i]
        
        frame_ids = []
        old_id_det = None
        new_id_det = None
        
        for det in detections:
            track_id = int(det['track_id'])
            frame_ids.append(track_id)
            
            if track_id == old_id:
                old_id_det = det
            elif track_id == new_id:
                new_id_det = det
        
        all_ids_per_frame.append((frame_idx, sorted(frame_ids)))
        
        if old_id_det:
            old_id_data.append({
                'frame': frame_idx,
                'bbox': old_id_det['bbox'],
                'conf': old_id_det['confidence']
            })
        
        if new_id_det:
            new_id_data.append({
                'frame': frame_idx,
                'bbox': new_id_det['bbox'],
                'conf': new_id_det['confidence']
            })
    
    # Print timeline
    print(f"🎯 Track ID Timeline:\n")
    print(f"{'Frame':<8} {'All IDs Present':<30} {'ID {old_id}':<15} {'ID {new_id}':<15}")
    print("-" * 80)
    
    for frame_idx, ids in all_ids_per_frame:
        ids_str = str(ids)[:28]
        old_status = "✓ Present" if any(d['frame'] == frame_idx for d in old_id_data) else "✗ Missing"
        new_status = "✓ Present" if any(d['frame'] == frame_idx for d in new_id_data) else "✗ Missing"
        
        marker = "  >>> SWITCH POINT <<<" if frame_idx == switch_frame else ""
        print(f"{frame_idx:<8} {ids_str:<30} {old_status:<15} {new_status:<15} {marker}")
    
    # Analyze last appearance of old ID
    print(f"\n\n{'='*80}")
    print(f"📉 OLD ID {old_id} Analysis:")
    print(f"{'='*80}\n")
    
    if old_id_data:
        print(f"First appearance: Frame {old_id_data[0]['frame']}")
        print(f"Last appearance:  Frame {old_id_data[-1]['frame']}")
        print(f"Total frames tracked: {len(old_id_data)}")
        
        # Check confidence trend
        print(f"\nConfidence scores (last 5 frames):")
        for d in old_id_data[-5:]:
            print(f"  Frame {d['frame']}: conf={d['conf']:.3f}, bbox={d['bbox']}")
        
        # Check if confidence dropped
        if len(old_id_data) >= 2:
            avg_conf_early = np.mean([d['conf'] for d in old_id_data[:5]])
            avg_conf_late = np.mean([d['conf'] for d in old_id_data[-5:]])
            print(f"\nAverage confidence (first 5 frames): {avg_conf_early:.3f}")
            print(f"Average confidence (last 5 frames):  {avg_conf_late:.3f}")
            if avg_conf_late < avg_conf_early - 0.1:
                print("⚠️  Confidence dropped significantly before disappearing!")
    else:
        print("⚠️  Old ID not found in analysis window")
    
    # Analyze first appearance of new ID
    print(f"\n\n{'='*80}")
    print(f"📈 NEW ID {new_id} Analysis:")
    print(f"{'='*80}\n")
    
    if new_id_data:
        print(f"First appearance: Frame {new_id_data[0]['frame']}")
        print(f"Last appearance:  Frame {new_id_data[-1]['frame']}")
        print(f"Total frames tracked: {len(new_id_data)}")
        
        # Check confidence trend
        print(f"\nConfidence scores (first 5 frames):")
        for d in new_id_data[:5]:
            print(f"  Frame {d['frame']}: conf={d['conf']:.3f}, bbox={d['bbox']}")
        
        # Check gap between old and new
        if old_id_data and new_id_data:
            gap = new_id_data[0]['frame'] - old_id_data[-1]['frame']
            print(f"\n⏱️  Gap between old ID disappearing and new ID appearing: {gap} frames")
            
            if gap > 0:
                print(f"   → Track was lost for {gap} frames (max_age={gap} would be needed)")
            elif gap == 0:
                print(f"   → IDs coexisted in same frame (likely occlusion/overlap)")
            else:
                print(f"   → New ID appeared {abs(gap)} frames BEFORE old ID disappeared!")
            
            # Compare bboxes to see if it's likely the same person
            old_last_bbox = old_id_data[-1]['bbox']
            new_first_bbox = new_id_data[0]['bbox']
            
            iou = compute_iou(old_last_bbox, new_first_bbox)
            print(f"\n📦 Bounding Box Comparison:")
            print(f"   Old ID last bbox:  {old_last_bbox}")
            print(f"   New ID first bbox: {new_first_bbox}")
            print(f"   IoU (overlap):     {iou:.3f}")
            
            if iou > 0.5:
                print(f"   ✓ High overlap ({iou:.3f}) - likely same person!")
            elif iou > 0.2:
                print(f"   ⚠️  Moderate overlap ({iou:.3f}) - could be same person")
            else:
                print(f"   ✗ Low overlap ({iou:.3f}) - might be different people")
    else:
        print("⚠️  New ID not found in analysis window")
    
    # Recommendations
    print(f"\n\n{'='*80}")
    print(f"💡 RECOMMENDATIONS:")
    print(f"{'='*80}\n")
    
    if old_id_data and new_id_data:
        gap = new_id_data[0]['frame'] - old_id_data[-1]['frame']
        
        if gap > 0 and gap <= 60:
            print(f"1. INCREASE max_age:")
            print(f"   Current: Need at least {gap} frames")
            print(f"   Suggested: max_age={gap + 10} (with some buffer)")
            print(f"   Why: Track was lost for {gap} frames, max_age wasn't long enough\n")
        
        if old_id_data:
            avg_conf_late = np.mean([d['conf'] for d in old_id_data[-5:]])
            if avg_conf_late < 0.4:
                print(f"2. LOWER det_thresh or track_thresh:")
                print(f"   Confidence dropped to {avg_conf_late:.3f} before loss")
                print(f"   Suggested: det_thresh=0.25, track_thresh=0.35")
                print(f"   Why: Lower thresholds can maintain tracks with lower confidence\n")
        
        if new_id_data and old_id_data:
            iou = compute_iou(old_id_data[-1]['bbox'], new_id_data[0]['bbox'])
            if iou > 0.3:
                print(f"3. DECREASE match_thresh:")
                print(f"   Current IoU: {iou:.3f}")
                print(f"   Suggested: match_thresh=0.6 or 0.7 (from default 0.8)")
                print(f"   Why: More permissive matching can maintain ID during occlusions\n")
                
                print(f"4. ENABLE or VERIFY ReID:")
                print(f"   ReID uses appearance features, not just IoU")
                print(f"   Can maintain IDs even with low bbox overlap")
                print(f"   Check: reid.enabled=true in config\n")
        
        print(f"5. CONSIDER DIFFERENT TRACKER:")
        print(f"   ByteTrack: Motion-only (fast but less robust)")
        print(f"   BoT-SORT: Motion + ReID (better for occlusions)")
        print(f"   DeepOCSORT: Deep learning features (best accuracy)")
        print(f"   Try: tracker=botsort with ReID enabled\n")


def compute_iou(bbox1, bbox2):
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]"""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height
    
    # Union
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def visualize_frames(video_path, frame_numbers, detections_per_frame, old_id, new_id, switch_frame, context_frames=5):
    """
    Extract and save frames around the ID switch for visual inspection
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"⚠️  Could not open video: {video_path}")
        return
    
    output_dir = Path("demo_data/outputs/debug_frames")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n\n{'='*80}")
    print(f"🎬 Extracting frames for visual inspection...")
    print(f"{'='*80}\n")
    
    frames_to_extract = range(switch_frame - context_frames, switch_frame + context_frames + 1)
    
    for frame_idx in frames_to_extract:
        if frame_idx < 0 or frame_idx >= len(frame_numbers):
            continue
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Find detections for this frame
        if frame_idx < len(detections_per_frame):
            detections = detections_per_frame[frame_idx]
        else:
            detections = []
        
        # Draw all bboxes
        for det in detections:
            track_id = int(det['track_id'])
            bbox = det['bbox']
            conf = det['confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Color code: old ID=red, new ID=green, others=blue
            if track_id == old_id:
                color = (0, 0, 255)  # Red
                thickness = 3
            elif track_id == new_id:
                color = (0, 255, 0)  # Green
                thickness = 3
            else:
                color = (255, 0, 0)  # Blue
                thickness = 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw ID and confidence
            label = f"ID:{track_id} ({conf:.2f})"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if frame_idx == switch_frame:
            cv2.putText(frame, "<<< SWITCH FRAME >>>", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        
        # Save frame
        output_path = output_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(output_path), frame)
        print(f"   Saved: {output_path}")
    
    cap.release()
    print(f"\n✓ Frames saved to: {output_dir}")
    print(f"   Red = ID {old_id}, Green = ID {new_id}, Blue = Other IDs")


def main():
    parser = argparse.ArgumentParser(description="Debug track ID switches")
    parser.add_argument("--detections", required=True, help="Path to raw_detections.npz or raw_detections_debug.npz")
    parser.add_argument("--video", help="Path to video file (for frame extraction)")
    parser.add_argument("--old-id", type=int, required=True, help="Original track ID")
    parser.add_argument("--new-id", type=int, required=True, help="New track ID")
    parser.add_argument("--frame", type=int, help="Approximate switch frame (optional, will auto-detect)")
    parser.add_argument("--context", type=int, default=20, help="Frames before/after to analyze (default: 20)")
    parser.add_argument("--extract-frames", action="store_true", help="Extract video frames for visual inspection")
    
    args = parser.parse_args()
    
    # Load detections (smart loader handles both formats)
    print(f"Loading detections from: {args.detections}")
    frame_numbers, detections_per_frame, metadata = load_detections(args.detections)
    print(f"✓ Loaded {len(frame_numbers)} frames with detections\n")
    
    # Show metadata if available
    if metadata:
        print(f"📋 Metadata:")
        print(f"   Video: {metadata.get('video_name', 'unknown')}")
        print(f"   Tracker: {metadata.get('tracker_type', 'unknown').upper()}")
        print(f"   ReID: {'ON' if metadata.get('reid_enabled', False) else 'OFF'}")
        print(f"   Total unique IDs: {len(metadata.get('unique_track_ids', []))}")
    
    # Analyze ID switch (auto-detects switch point if --frame not provided)
    analyze_id_switch(frame_numbers, detections_per_frame, args.old_id, args.new_id, 
                     args.frame, args.context)
    
    # Extract frames if requested
    if args.extract_frames and args.video:
        # Use detected switch point if not provided
        if args.frame is None:
            _, _, switch_frame = find_id_switch_point(frame_numbers, detections_per_frame, 
                                                     args.old_id, args.new_id)
        else:
            switch_frame = args.frame
            
        visualize_frames(args.video, frame_numbers, detections_per_frame, 
                        args.old_id, args.new_id, switch_frame)


if __name__ == "__main__":
    main()
