#!/usr/bin/env python3
"""
Stage 4a: ReID Recovery (Optional)

Performs selective ReID-based tracklet merging at transition points.
Extracts frames only at tracklet boundaries for efficiency.

Usage:
    python stage4a_reid_recovery.py --config configs/pipeline_config.yaml
"""

import argparse
import yaml
import numpy as np
import json
import time
import re
import cv2
import torch
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
    
    # Auto-extract current_video from video_file
    video_file = config.get('global', {}).get('video_file', '')
    if video_file:
        import os
        video_name = os.path.splitext(video_file)[0]
        config['global']['current_video'] = video_name
    
    return resolve_path_variables(config)


def load_reid_model(model_name, device):
    """Load OSNet ReID model"""
    try:
        from boxmot.appearance.reid_auto_backend import ReidAutoBackend
        print(f"  🛠️  Loading ReID model: {model_name}")
        model = ReidAutoBackend(model_name, device)
        return model
    except Exception as e:
        print(f"  ❌ Failed to load ReID model: {e}")
        raise


def extract_frames_at_transitions(video_path, candidates, frames_per_tracklet=5):
    """
    Extract frames near tracklet transitions.
    Returns dict: {tracklet_id: [frame_indices]}
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine which frames to extract
    frames_to_extract = {}
    for cand in candidates:
        tid1 = cand['tracklet_1']
        tid2 = cand['tracklet_2']
        
        # Extract frames near transition
        frame1 = cand['transition_frame_1']
        frame2 = cand['transition_frame_2']
        
        # Add frames around transitions
        for tid, frame in [(tid1, frame1), (tid2, frame2)]:
            if tid not in frames_to_extract:
                frames_to_extract[tid] = set()
            
            # Get frames within window
            start = max(0, frame - frames_per_tracklet // 2)
            end = min(total_frames, frame + frames_per_tracklet // 2 + 1)
            frames_to_extract[tid].update(range(start, end))
    
    # Sort frames for sequential reading
    all_frames_sorted = sorted(set(sum([list(f) for f in frames_to_extract.values()], [])))
    
    # Read frames
    frame_cache = {}
    for frame_idx in tqdm(all_frames_sorted, desc="Extracting frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_cache[frame_idx] = frame
    
    cap.release()
    return frame_cache, frames_to_extract


def extract_reid_features(tracklets, frame_cache, frames_to_extract, bboxes_by_frame, reid_model):
    """Extract ReID features for each tracklet"""
    features = {}
    
    for tid in tqdm(frames_to_extract.keys(), desc="Extracting features"):
        # Find tracklet
        tracklet = next((t for t in tracklets if t['tracklet_id'] == tid), None)
        if tracklet is None:
            continue
        
        # Extract features from available frames
        tid_features = []
        for frame_idx in frames_to_extract[tid]:
            if frame_idx not in frame_cache:
                continue
            
            frame = frame_cache[frame_idx]
            
            # Find bbox for this frame
            frame_bboxes = bboxes_by_frame.get(tid, {}).get(frame_idx)
            if frame_bboxes is None:
                continue
            
            # Crop and extract
            x1, y1, x2, y2 = map(int, frame_bboxes)
            crop = frame[y1:y2, x1:x2]
            
            if crop.size > 0:
                # Get ReID feature
                feature = reid_model(crop)
                tid_features.append(feature)
        
        if tid_features:
            # Average features
            features[tid] = np.mean(tid_features, axis=0)
    
    return features


def compute_similarity_matrix(features, tracklet_ids):
    """Compute cosine similarity between all tracklet features"""
    n = len(tracklet_ids)
    similarity = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if tracklet_ids[i] in features and tracklet_ids[j] in features:
                feat_i = features[tracklet_ids[i]]
                feat_j = features[tracklet_ids[j]]
                
                # Cosine similarity
                sim = np.dot(feat_i, feat_j) / (np.linalg.norm(feat_i) * np.linalg.norm(feat_j) + 1e-8)
                similarity[i, j] = sim
                similarity[j, i] = sim
    
    return similarity


def merge_tracklets(tracklets, merge_pairs):
    """Merge tracklets based on ReID similarities"""
    # Build merge groups
    groups = {}
    for tid1, tid2 in merge_pairs:
        # Find existing groups
        group1 = next((g for g, members in groups.items() if tid1 in members), None)
        group2 = next((g for g, members in groups.items() if tid2 in members), None)
        
        if group1 is None and group2 is None:
            # New group
            new_id = min(tid1, tid2)
            groups[new_id] = {tid1, tid2}
        elif group1 is not None and group2 is None:
            groups[group1].add(tid2)
        elif group1 is None and group2 is not None:
            groups[group2].add(tid1)
        elif group1 != group2:
            # Merge groups
            groups[group1].update(groups[group2])
            del groups[group2]
    
    # Add singletons
    all_merged = set(sum([list(g) for g in groups.values()], []))
    for tracklet in tracklets:
        tid = tracklet['tracklet_id']
        if tid not in all_merged:
            groups[tid] = {tid}
    
    # Create merged tracklets
    merged = []
    for canonical_id, member_ids in groups.items():
        # Combine all member tracklets
        member_tracklets = [t for t in tracklets if t['tracklet_id'] in member_ids]
        
        # Sort by start frame
        member_tracklets.sort(key=lambda t: t['frame_numbers'][0])
        
        # Concatenate
        merged_frames = np.concatenate([t['frame_numbers'] for t in member_tracklets])
        merged_bboxes = np.concatenate([t['bboxes'] for t in member_tracklets])
        merged_confs = np.concatenate([t['confidences'] for t in member_tracklets])
        
        merged.append({
            'tracklet_id': canonical_id,
            'frame_numbers': merged_frames,
            'bboxes': merged_bboxes,
            'confidences': merged_confs,
            'original_ids': sorted(member_ids)
        })
    
    return merged


def run_reid_recovery(config):
    """Run Stage 4a: ReID Recovery"""
    
    stage_config = config['stage4a_reid_recovery']
    verbose = stage_config.get('advanced', {}).get('verbose', False)
    
    # Extract configuration
    input_config = stage_config['input']
    output_config = stage_config['output']
    reid_config = stage_config['reid']
    
    video_file = config['global']['current_video']
    tracklet_stats_file = input_config['tracklet_stats_file']
    candidates_file = input_config['candidates_file']
    recovered_file = output_config['recovered_tracklets_file']
    merge_log_file = output_config['merge_log_file']
    
    # Print header
    print(f"\n{'='*70}")
    print(f"📍 STAGE 4a: ReID RECOVERY")
    print(f"{'='*70}\n")
    
    # Load data
    print(f"📂 Loading tracklets and candidates...")
    data = np.load(tracklet_stats_file, allow_pickle=True)
    tracklets = data['tracklets']
    
    with open(candidates_file, 'r') as f:
        candidates = json.load(f)
    
    print(f"  ✅ Loaded {len(tracklets)} tracklets, {len(candidates)} candidates")
    
    if len(candidates) == 0:
        print(f"  ⏭️  No candidates to process")
        # Just copy tracklets
        output_path = Path(recovered_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, tracklets=tracklets)
        return
    
    # Load ReID model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  🛠️  Device: {device}")
    reid_model = load_reid_model(reid_config['model_name'], device)
    
    # Extract frames
    print(f"\n📹 Extracting frames at transitions...")
    t_start = time.time()
    frame_cache, frames_to_extract = extract_frames_at_transitions(
        video_file, candidates, reid_config.get('frames_per_tracklet', 5)
    )
    t_end = time.time()
    print(f"  ✅ Extracted {len(frame_cache)} frames ({t_end - t_start:.2f}s)")
    
    # Build bbox lookup
    bboxes_by_frame = {}
    for tracklet in tracklets:
        tid = tracklet['tracklet_id']
        bboxes_by_frame[tid] = {}
        for i, frame_idx in enumerate(tracklet['frame_numbers']):
            bboxes_by_frame[tid][frame_idx] = tracklet['bboxes'][i]
    
    # Extract features
    print(f"\n🔬 Extracting ReID features...")
    t_start = time.time()
    features = extract_reid_features(tracklets, frame_cache, frames_to_extract, bboxes_by_frame, reid_model)
    t_end = time.time()
    print(f"  ✅ Extracted features for {len(features)} tracklets ({t_end - t_start:.2f}s)")
    
    # Compute similarities and merge
    print(f"\n🔗 Computing similarities and merging...")
    tracklet_ids = [t['tracklet_id'] for t in tracklets]
    similarity = compute_similarity_matrix(features, tracklet_ids)
    
    # Apply threshold
    threshold = reid_config['similarity_threshold']
    merge_pairs = []
    merge_log = []
    
    for cand in candidates:
        tid1 = cand['tracklet_1']
        tid2 = cand['tracklet_2']
        
        if tid1 in features and tid2 in features:
            idx1 = tracklet_ids.index(tid1)
            idx2 = tracklet_ids.index(tid2)
            sim = similarity[idx1, idx2]
            
            if sim >= threshold:
                merge_pairs.append((tid1, tid2))
                merge_log.append({
                    'tracklet_1': tid1,
                    'tracklet_2': tid2,
                    'similarity': float(sim),
                    'merged': True
                })
            else:
                merge_log.append({
                    'tracklet_1': tid1,
                    'tracklet_2': tid2,
                    'similarity': float(sim),
                    'merged': False
                })
    
    print(f"  ✅ Merging {len(merge_pairs)} pairs (threshold={threshold})")
    
    # Perform merging
    recovered_tracklets = merge_tracklets(tracklets, merge_pairs)
    
    # Save results
    output_path = Path(recovered_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, tracklets=np.array(recovered_tracklets, dtype=object))
    print(f"  ✅ Saved: {output_path}")
    
    # Save log
    log_path = Path(merge_log_file)
    with open(log_path, 'w') as f:
        json.dump(merge_log, f, indent=2)
    print(f"  ✅ Saved merge log: {log_path}")
    
    print(f"\n✅ ReID recovery complete!")
    print(f"  Original tracklets: {len(tracklets)}")
    print(f"  Recovered tracklets: {len(recovered_tracklets)}")
    print(f"  Merged pairs: {len(merge_pairs)}")


def main():
    parser = argparse.ArgumentParser(description='Stage 4a: ReID Recovery')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Check if stage is enabled
    if not config['pipeline']['stages']['stage4a_reid_recovery']:
        print("⏭️  Stage 4a is disabled in config")
        return
    
    # Run recovery
    run_reid_recovery(config)
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
