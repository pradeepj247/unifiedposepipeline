#!/usr/bin/env python3
"""
Stage 2: Tracking (ByteTrack Offline)

Runs ByteTrack tracker on pre-stored detections from Stage 1.
Motion-only tracking (no video frames needed).

Usage:
    python stage2_track.py --config configs/pipeline_config.yaml
"""

import argparse
import yaml
import numpy as np
import time
import re
import sys
from pathlib import Path
from tqdm import tqdm


def resolve_path_variables(config):
    """Recursively resolve ${variable} in config"""
    global_vars = config.get('global', {})
    
    # First pass: resolve variables within global section itself
    def resolve_string_once(s, vars_dict):
        if not isinstance(s, str):
            return s
        return re.sub(
            r'\$\{(\w+)\}',
            lambda m: str(vars_dict.get(m.group(1), m.group(0))),
            s
        )
    
    # Resolve global variables iteratively
    max_iterations = 10
    for _ in range(max_iterations):
        resolved_globals = {}
        changed = False
        for key, value in global_vars.items():
            if isinstance(value, str):
                resolved = resolve_string_once(value, global_vars)
                resolved_globals[key] = resolved
                if resolved != value:
                    changed = True
            else:
                resolved_globals[key] = value
        global_vars = resolved_globals
        if not changed:
            break
    
    def resolve_string(s):
        return re.sub(
            r'\$\{(\w+)\}',
            lambda m: str(global_vars.get(m.group(1), m.group(0))),
            s
        )
    
    def resolve_recursive(obj):
        if isinstance(obj, dict):
            return {k: resolve_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_recursive(v) for v in obj]
        elif isinstance(obj, str):
            return resolve_string(obj)
        return obj
    
    result = resolve_recursive(config)
    result['global'] = global_vars
    return result


def load_config(config_path):
    """Load and resolve YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return resolve_path_variables(config)


def load_detections(detections_file):
    """Load detections from NPZ file"""
    data = np.load(detections_file)
    return {
        'frame_numbers': data['frame_numbers'],
        'bboxes': data['bboxes'],
        'confidences': data['confidences'],
        'classes': data['classes'],
        'num_detections_per_frame': data['num_detections_per_frame']
    }


def reconstruct_detections_per_frame(detections_data):
    """Reconstruct per-frame detections from flat arrays"""
    frame_numbers = detections_data['frame_numbers']
    bboxes = detections_data['bboxes']
    confidences = detections_data['confidences']
    classes = detections_data['classes']
    
    # Get unique frame numbers
    unique_frames = np.unique(frame_numbers)
    
    detections_by_frame = {}
    for frame_id in unique_frames:
        mask = frame_numbers == frame_id
        detections_by_frame[frame_id] = {
            'bboxes': bboxes[mask],
            'confidences': confidences[mask],
            'classes': classes[mask]
        }
    
    return detections_by_frame, unique_frames


def init_bytetrack_tracker(params, frame_rate, verbose=False):
    """Initialize ByteTrack tracker"""
    try:
        from boxmot import ByteTrack
    except ImportError:
        raise ImportError("boxmot not found. Install with: pip install boxmot")
    
    # Suppress BoxMOT's verbose logging
    import logging
    logging.getLogger('boxmot').setLevel(logging.WARNING)
    
    if verbose:
        print(f"  ✅ Initializing ByteTrack tracker")
        print(f"     track_thresh: {params.get('track_thresh', 0.25)}")
        print(f"     track_buffer: {params.get('track_buffer', 30)}")
        print(f"     match_thresh: {params.get('match_thresh', 0.8)}")
        print(f"     frame_rate: {frame_rate}")
    
    tracker = ByteTrack(
        track_thresh=params.get('track_thresh', 0.25),
        track_buffer=params.get('track_buffer', 30),
        match_thresh=params.get('match_thresh', 0.8),
        frame_rate=frame_rate
    )
    
    return tracker


def run_tracking(config):
    """Run Stage 2: Tracking"""
    
    stage_config = config['stage2_track']
    verbose = stage_config.get('advanced', {}).get('verbose', False)
    
    # Extract configuration
    tracker_config = stage_config['tracker']
    params = stage_config['params']
    input_config = stage_config['input']
    output_config = stage_config['output']
    
    detections_file = input_config['detections_file']
    tracklets_file = output_config['tracklets_file']
    video_path = input_config.get('video_path', None)
    
    # Print header
    print(f"\n{'='*70}")
    print(f"📍 STAGE 2: TRACKING (BYTETRACK OFFLINE)")
    print(f"{'='*70}\n")
    
    # Get video metadata for frame_rate
    frame_rate = 30  # default
    if video_path:
        import cv2
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if verbose:
            print(f"📹 Video frame rate: {frame_rate:.2f} fps")
    
    # Load detections
    print(f"📂 Loading detections: {detections_file}")
    detections_data = load_detections(detections_file)
    
    total_detections = len(detections_data['frame_numbers'])
    print(f"  ✅ Loaded {total_detections} detections")
    
    # Reconstruct per-frame detections
    print(f"\n🔄 Reconstructing per-frame detections...")
    detections_by_frame, unique_frames = reconstruct_detections_per_frame(detections_data)
    num_frames = len(unique_frames)
    print(f"  ✅ {num_frames} frames with detections")
    
    # Initialize tracker
    print(f"\n🛠️  Initializing ByteTrack tracker...")
    tracker = init_bytetrack_tracker(params, frame_rate, verbose)
    
    # Track
    print(f"\n⚡ Running ByteTrack (offline mode)...")
    t_start = time.time()
    
    tracklets_dict = {}  # {tracklet_id: {'frame_numbers': [], 'bboxes': [], 'confidences': []}}
    
    pbar = tqdm(total=num_frames, desc="Tracking")
    
    debug_first_frame = True
    for frame_id in sorted(unique_frames):
        frame_data = detections_by_frame[frame_id]
        
        # Prepare detections for tracker: (N, 6) = [x1, y1, x2, y2, conf, cls]
        if len(frame_data['bboxes']) > 0:
            dets_for_tracker = np.column_stack([
                frame_data['bboxes'],       # (N, 4)
                frame_data['confidences'],  # (N,)
                frame_data['classes']       # (N,)
            ])
        else:
            dets_for_tracker = np.empty((0, 6))
        
        # Debug first frame
        if debug_first_frame and len(dets_for_tracker) > 0:
            print(f"\n🔍 DEBUG - First frame with detections (frame {frame_id}):")
            print(f"   Shape: {dets_for_tracker.shape}")
            print(f"   First detection: {dets_for_tracker[0]}")
            print(f"   Bbox format: [x1={dets_for_tracker[0,0]:.1f}, y1={dets_for_tracker[0,1]:.1f}, x2={dets_for_tracker[0,2]:.1f}, y2={dets_for_tracker[0,3]:.1f}, conf={dets_for_tracker[0,4]:.2f}, cls={dets_for_tracker[0,5]:.0f}]")
            debug_first_frame = False
        
        # Update tracker (pass actual frame for context)
        try:
            tracked = tracker.update(dets_for_tracker, None)  # ByteTrack motion-only, no frame needed
            
            # Debug first tracking result
            if debug_first_frame and len(dets_for_tracker) > 0:
                print(f"   Tracker returned: shape={tracked.shape if len(tracked) > 0 else 'empty'}, count={len(tracked)}")
                if len(tracked) > 0:
                    print(f"   First track: {tracked[0]}")
            
            # Store tracklets
            # tracked: (N, 8) = [x1, y1, x2, y2, track_id, conf, cls, det_ind]
            if len(tracked) > 0:
                for track in tracked:
                    track_id = int(track[4])
                    bbox = track[:4].astype(np.float32)
                    conf = float(track[5])
                    
                    if track_id not in tracklets_dict:
                        tracklets_dict[track_id] = {
                            'frame_numbers': [],
                            'bboxes': [],
                            'confidences': []
                        }
                    
                    tracklets_dict[track_id]['frame_numbers'].append(int(frame_id))
                    tracklets_dict[track_id]['bboxes'].append(bbox)
                    tracklets_dict[track_id]['confidences'].append(conf)
        
        except Exception as e:
            # Tracker error - skip this frame
            if verbose:
                print(f"\n⚠️  Tracker error at frame {frame_id}: {e}")
        
        pbar.update(1)
    
    pbar.close()
    
    t_end = time.time()
    total_time = t_end - t_start
    tracking_fps = num_frames / total_time if total_time > 0 else 0
    
    # Convert tracklets to list format
    tracklets = []
    for track_id, data in tracklets_dict.items():
        tracklets.append({
            'tracklet_id': track_id,
            'frame_numbers': np.array(data['frame_numbers'], dtype=np.int64),
            'bboxes': np.array(data['bboxes'], dtype=np.float32),
            'confidences': np.array(data['confidences'], dtype=np.float32)
        })
    
    # Sort by tracklet ID
    tracklets.sort(key=lambda x: x['tracklet_id'])
    
    # Summary
    num_tracklets = len(tracklets)
    total_tracked_detections = sum(len(t['frame_numbers']) for t in tracklets)
    
    print(f"\n✅ Tracking complete!")
    print(f"  Frames processed: {num_frames}")
    print(f"  Unique tracklets: {num_tracklets}")
    print(f"  Total tracked detections: {total_tracked_detections}")
    print(f"  Tracking FPS: {tracking_fps:.1f}")
    print(f"  Time taken: {total_time:.2f}s")
    
    if verbose and num_tracklets > 0:
        print(f"\n📊 Tracklet Statistics:")
        for t in tracklets[:10]:  # Show first 10
            duration = len(t['frame_numbers'])
            start_frame = t['frame_numbers'][0]
            end_frame = t['frame_numbers'][-1]
            print(f"  Tracklet {t['tracklet_id']}: {duration} frames "
                  f"(frames {start_frame}-{end_frame})")
        if num_tracklets > 10:
            print(f"  ... and {num_tracklets - 10} more")
    
    # Save NPZ
    print(f"\n💾 Saving tracklets...")
    output_path = Path(tracklets_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as structured array
    np.savez_compressed(
        output_path,
        tracklets=np.array(tracklets, dtype=object)
    )
    
    print(f"  ✅ Saved: {output_path}")
    print(f"  Format: {num_tracklets} tracklets")
    
    return {
        'tracklets_file': str(output_path),
        'num_tracklets': num_tracklets,
        'num_frames': num_frames
    }


def main():
    parser = argparse.ArgumentParser(description='Stage 2: Tracking')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Check if stage is enabled
    if not config['pipeline']['stages']['stage2_track']:
        print("⏭️  Stage 2 is disabled in config")
        return
    
    # Run tracking
    run_tracking(config)
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
