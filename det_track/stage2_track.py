#!/usr/bin/env python3
"""
Stage 2: Tracking (ByteTrack Offline)

Runs ByteTrack tracker on pre-stored detections from Stage 1.
Motion-only tracking using Kalman filters (no video pixels needed).
Video frames are NOT read - only detection bboxes and motion are used.

Usage:
    python stage2_track.py --config configs/pipeline_config.yaml
"""

import argparse
import yaml
import numpy as np
import time
import json
from datetime import datetime, timezone
import re
import sys
from pathlib import Path
import os
import contextlib
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
    
    # Auto-extract current_video from video_file
    video_file = config.get('global', {}).get('video_file', '')
    if video_file:
        import os
        video_name = os.path.splitext(video_file)[0]
        config['global']['current_video'] = video_name
    
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
        track_thresh=params.get('track_thresh', 0.15),
        track_buffer=params.get('track_buffer', 30),
        match_thresh=params.get('match_thresh', 0.8),
        min_hits=params.get('min_hits', 1),
        frame_rate=frame_rate
    )
    
    return tracker


def run_tracking(config):
    """Run Stage 2: Tracking"""
    
    stage_config = config['stage2']
    verbose = stage_config.get('advanced', {}).get('verbose', False)
    
    # Extract configuration
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
    
    # ByteTrack uses Kalman filters (motion-only), doesn't need actual frame pixels
    # Skip video loading for performance (frame is only placeholder for BoxMOT API)
    import cv2
    frame_rate = 30  # default
    video_width, video_height = 1920, 1080  # defaults
    
    if verbose:
        print(f"📹 Using motion-only tracking (no video pixels needed)")
        print(f"   Frame rate: {frame_rate} fps, Resolution: {video_width}x{video_height}")
    
    # Load detections
    if verbose:
        print(f"📂 Loading detections: {detections_file}")
    t_dl_start = time.time()
    detections_data = load_detections(detections_file)
    detections_load_time = time.time() - t_dl_start
    
    total_detections = len(detections_data['frame_numbers'])
    if verbose:
        print(f"  ✅ Loaded {total_detections} detections")
    
    # Reconstruct per-frame detections
    if verbose:
        print(f"\n🔄 Reconstructing per-frame detections...")
    t_rec_start = time.time()
    detections_by_frame, unique_frames = reconstruct_detections_per_frame(detections_data)
    reconstruct_time = time.time() - t_rec_start
    num_frames = len(unique_frames)
    if verbose:
        print(f"  ✅ {num_frames} frames with detections")
    
    # Initialize tracker
    if verbose:
        print(f"\n🛠️  Initializing ByteTrack tracker...")
    t_init_start = time.time()
    # Suppress boxmot/loguru output during tracker init when not verbose
    if verbose:
        tracker = init_bytetrack_tracker(params, frame_rate, verbose)
    else:
        try:
            from loguru import logger as _loguru_logger
            _loguru_logger.disable("boxmot")
        except Exception:
            _loguru_logger = None

        tracker = init_bytetrack_tracker(params, frame_rate, verbose)

        try:
            if _loguru_logger is not None:
                _loguru_logger.enable("boxmot")
        except Exception:
            pass
    tracker_init_time = time.time() - t_init_start
    
    # Track
    print(f"\n⚡ Running ByteTrack (offline mode)...")
    t_start = time.time()
    
    tracklets_dict = {}  # {tracklet_id: {'frame_numbers': [], 'bboxes': [], 'confidences': []}}
    
    pbar = tqdm(total=num_frames, desc="Tracking", mininterval=1.0)
    
    debug_first_frame = bool(verbose)
    debug_first_tracking = bool(verbose)
    frame_count = 0
    for frame_id in sorted(unique_frames):
        frame_data = detections_by_frame[frame_id]
        
        # Use dummy frame (ByteTrack only uses Kalman filters, not pixel features)
        # Passing None would fail BoxMOT API, so we pass empty frame placeholder
        frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        
        # Prepare detections for tracker: (N, 6) = [x1, y1, x2, y2, conf, cls]
        if len(frame_data['bboxes']) > 0:
            dets_for_tracker = np.column_stack([
                frame_data['bboxes'],       # (N, 4)
                frame_data['confidences'],  # (N,)
                frame_data['classes']       # (N,)
            ])
        else:
            dets_for_tracker = np.empty((0, 6))
        
        # Debug first frame (verbose only)
        if debug_first_frame and len(dets_for_tracker) > 0:
            print(f"\n🔍 DEBUG - First frame with detections (frame {frame_id}):")
            print(f"   Shape: {dets_for_tracker.shape}")
            print(f"   First detection: {dets_for_tracker[0]}")
            print(f"   Bbox format: [x1={dets_for_tracker[0,0]:.1f}, y1={dets_for_tracker[0,1]:.1f}, x2={dets_for_tracker[0,2]:.1f}, y2={dets_for_tracker[0,3]:.1f}, conf={dets_for_tracker[0,4]:.2f}, cls={dets_for_tracker[0,5]:.0f}]")
            debug_first_frame = False
        
        # Update tracker (pass frame - required by BoxMOT even for motion-only tracking)
        try:
            tracked = tracker.update(dets_for_tracker, frame)
            
            # Debug first tracking result (verbose only)
            if debug_first_tracking and len(dets_for_tracker) > 0:
                print(f"   Tracker returned: shape={tracked.shape if len(tracked) > 0 else 'empty'}, count={len(tracked)}")
                if len(tracked) > 0:
                    print(f"   First track: {tracked[0]}")
                else:
                    print(f"   ⚠️ WARNING: Tracker returned 0 tracks despite {len(dets_for_tracker)} detections!")
                debug_first_tracking = False
            
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
                            'confidences': [],
                            'detection_indices': []
                        }
                    
                    # Extract detection index from ByteTrack output
                    # tracked: (N, 8) = [x1, y1, x2, y2, track_id, conf, cls, det_ind]
                    det_ind = int(track[7]) if len(track) > 7 else -1
                    
                    tracklets_dict[track_id]['frame_numbers'].append(int(frame_id))
                    tracklets_dict[track_id]['bboxes'].append(bbox)
                    tracklets_dict[track_id]['confidences'].append(conf)
                    tracklets_dict[track_id]['detection_indices'].append(det_ind)
        
        except Exception as e:
            # Tracker error - skip this frame
            if verbose:
                print(f"\n⚠️  Tracker error at frame {frame_id}: {e}")
        
        frame_count += 1
        if frame_count % 100 == 0 or frame_count == num_frames:
            pbar.update(100 if frame_count + 100 <= num_frames else num_frames - frame_count + 100)
    
    pbar.close()
    
    t_end = time.time()
    tracking_loop_time = t_end - t_start
    total_time = tracking_loop_time
    tracking_fps = num_frames / total_time if total_time > 0 else 0
    
    # Convert tracklets to list format
    tracklets = []
    for track_id, data in tracklets_dict.items():
        tracklets.append({
            'tracklet_id': track_id,
            'frame_numbers': np.array(data['frame_numbers'], dtype=np.int64),
            'bboxes': np.array(data['bboxes'], dtype=np.float32),
            'confidences': np.array(data['confidences'], dtype=np.float32),
            'detection_indices': np.array(data['detection_indices'], dtype=np.int64)
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
    
    output_path = Path(tracklets_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as structured array
    t_save_start = time.time()
    np.savez_compressed(
        output_path,
        tracklets=np.array(tracklets, dtype=object)
    )
    npz_save_time = time.time() - t_save_start
    total_save_time = npz_save_time

    # Compact save message (filename only)
    print(f"  ✅ Saved:  {output_path.name}")

    # Write timings sidecar silently
    try:
        sidecar_path = output_path.parent / (output_path.name + '.timings.json')
        sidecar = {
            'detections_file': str(detections_file),
            'tracklets_file': str(output_path),
            'detections_load_time': float(detections_load_time),
            'reconstruct_time': float(reconstruct_time),
            'tracker_init_time': float(tracker_init_time),
            'tracking_loop_time': float(tracking_loop_time),
            'npz_save_time': float(npz_save_time),
            'total_save_time': float(total_save_time),
            'num_frames': int(num_frames),
            'num_tracklets': int(num_tracklets),
            'num_detections': int(total_detections),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        with open(sidecar_path, 'w', encoding='utf-8') as sf:
            json.dump(sidecar, sf, indent=2)
    except Exception:
        if verbose:
            print(f"  ⚠️  Failed to write timings sidecar")
    
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
    if not config['pipeline']['stages']['stage2']:
        print("⏭️  Stage 2 is disabled in config")
        return
    
    # Run tracking
    run_tracking(config)
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
