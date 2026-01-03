#!/usr/bin/env python3
"""
Stage 4a: ReID Recovery using ONNX OSNet model

Performs ReID-based tracklet merging at transition points.
Uses ONNX model instead of boxmot.appearance.

Usage:
    python stage4a_reid_recovery_onnx.py --config configs/pipeline_config.yaml
"""

import argparse
import yaml
import numpy as np
import json
import cv2
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import cosine
import re


def resolve_path_variables(config):
    """Recursively resolve ${variable} in config"""
    global_vars = config.get('global', {})
    
    def resolve_string_once(s, vars_dict):
        if not isinstance(s, str):
            return s
        return re.sub(
            r'\$\{(\w+)\}',
            lambda m: str(vars_dict.get(m.group(1), m.group(0))),
            s
        )
    
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
    
    video_file = config.get('global', {}).get('video_file', '')
    if video_file:
        import os
        video_name = os.path.splitext(video_file)[0]
        config['global']['current_video'] = video_name
    
    return resolve_path_variables(config)


class ReIDModel:
    """Wrapper for ONNX OSNet ReID model"""
    
    def __init__(self, model_path):
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def get_crop(self, frame, bbox):
        """Extract and resize crop to 256x128 (H x W)"""
        x1, y1, x2, y2 = bbox.astype(int)
        # Clip to frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        
        crop = cv2.resize(crop, (128, 256))  # width, height
        return crop
    
    def get_embedding_batch(self, crops_batch):
        """Get ReID embeddings for batch of 16 crops
        
        Args:
            crops_batch: np.array shape (16, 256, 128, 3) uint8
            
        Returns:
            embeddings: np.array shape (16, 512) float32
        """
        # Normalize to [0, 1]
        batch = crops_batch.astype(np.float32) / 255.0
        
        # Convert BGR to RGB
        batch = batch[:, :, :, ::-1]
        
        # Convert to CHW format
        batch = batch.transpose(0, 3, 1, 2)
        
        # Run inference
        embeddings = self.session.run([self.output_name], {self.input_name: batch})[0]
        return embeddings  # shape: (16, 512)
    
    def compute_similarity(self, embeddings1, embeddings2):
        """Compute average cosine similarity between two sets of embeddings
        
        Args:
            embeddings1: np.array shape (16, 512)
            embeddings2: np.array shape (16, 512)
            
        Returns:
            similarity: float in [0, 1]
        """
        avg_emb1 = embeddings1.mean(axis=0)
        avg_emb2 = embeddings2.mean(axis=0)
        
        # Cosine similarity
        sim = 1 - cosine(avg_emb1, avg_emb2)
        return float(sim)


def extract_crop_batch(video_path, tracklet, frame_range, reid_model, batch_size=16):
    """Extract crops from a tracklet for frames in given range
    
    Args:
        video_path: path to video file
        tracklet: dict with keys ['tracklet_id', 'frame_numbers', 'bboxes', 'confidences']
        frame_range: numpy array of frame numbers to include
        reid_model: ReIDModel instance
        batch_size: target batch size (will pad if needed)
        
    Returns:
        crop_batch: np.array shape (batch_size, 256, 128, 3) uint8
    """
    cap = cv2.VideoCapture(video_path)
    
    # Find which frames from tracklet fall in frame_range
    mask = np.isin(tracklet['frame_numbers'], frame_range)
    tracklet_frames = tracklet['frame_numbers'][mask]
    tracklet_bboxes = tracklet['bboxes'][mask]
    
    # Initialize batch with zeros
    crop_batch = np.zeros((batch_size, 256, 128, 3), dtype=np.uint8)
    
    # Extract crops
    for i, (frame_num, bbox) in enumerate(zip(tracklet_frames, tracklet_bboxes)):
        if i >= batch_size:
            break
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))
        ret, frame = cap.read()
        
        if ret:
            crop = reid_model.get_crop(frame, bbox)
            if crop is not None:
                crop_batch[i] = crop
    
    # Pad with last crop if needed
    if len(tracklet_frames) > 0 and len(tracklet_frames) < batch_size:
        last_valid_idx = min(len(tracklet_frames) - 1, batch_size - 1)
        for i in range(len(tracklet_frames), batch_size):
            crop_batch[i] = crop_batch[last_valid_idx]
    
    cap.release()
    return crop_batch


def run_reid_recovery(config):
    """Run Stage 4a: ReID Recovery"""
    
    stage_config = config['stage4a_reid_recovery']
    verbose = stage_config.get('advanced', {}).get('verbose', False)
    
    input_config = stage_config['input']
    output_config = stage_config['output']
    reid_config = stage_config['reid']
    
    tracklets_file = input_config['tracklets_file']
    candidates_file = input_config['candidates_file']
    reid_model_path = reid_config['model_path']
    similarity_threshold = reid_config['similarity_threshold']
    
    tracklets_recovered_file = output_config['tracklets_recovered_file']
    reid_results_file = output_config['reid_results_file']
    
    print(f"\n{'='*80}")
    print(f"📊 STAGE 4a: ReID RECOVERY (ONNX OSNet)")
    print(f"{'='*80}")
    
    # Load data
    print(f"\n📂 Loading tracklets: {tracklets_file}")
    data = np.load(tracklets_file, allow_pickle=True)
    tracklets = data['tracklets']
    print(f"✅ Loaded {len(tracklets)} tracklets")
    
    print(f"📂 Loading candidates: {candidates_file}")
    with open(candidates_file, 'r') as f:
        candidates = json.load(f)
    print(f"✅ Loaded {len(candidates)} candidate pairs")
    
    # Load ReID model
    print(f"\n🔧 Loading ReID model: {reid_model_path}")
    reid_model = ReIDModel(reid_model_path)
    print(f"✅ ReID model loaded")
    
    # Get video path
    global_vars = config['global']
    video_file = global_vars.get('video_file', '')
    video_dir = global_vars.get('video_dir', '')
    video_path = str(Path(video_dir) / video_file)
    print(f"📹 Video: {video_path}")
    
    # Process candidates
    print(f"\n🔍 Processing {len(candidates)} candidates with ReID...")
    reid_merges = []
    reid_rejections = []
    
    for cand in tqdm(candidates, desc="ReID matching"):
        tid1 = cand['tracklet_1']
        tid2 = cand['tracklet_2']
        gap = cand['gap']
        
        # Find tracklet indices
        idx1 = None
        idx2 = None
        for i, t in enumerate(tracklets):
            if int(t['tracklet_id']) == tid1:
                idx1 = i
            if int(t['tracklet_id']) == tid2:
                idx2 = i
        
        if idx1 is None or idx2 is None:
            if verbose:
                print(f"  ⚠️  Could not find tracklets {tid1}, {tid2}")
            continue
        
        # Determine frame range for extraction
        # For overlapping tracklets (gap < 0), use overlap region
        # For non-overlapping, use transition region
        if gap < 0:
            # Overlapping: use overlap frames
            start_frame = int(cand['transition_frame_2'])  # tracklet2 start
            end_frame = int(cand['transition_frame_1'])    # tracklet1 end
        else:
            # Non-overlapping: use frames around transition
            start_frame = int(cand['transition_frame_1']) - 10
            end_frame = int(cand['transition_frame_2']) + 10
        
        frame_range = np.arange(max(0, start_frame), end_frame + 1)
        
        try:
            # Extract crops and compute embeddings
            crops1 = extract_crop_batch(video_path, tracklets[idx1], frame_range, reid_model)
            crops2 = extract_crop_batch(video_path, tracklets[idx2], frame_range, reid_model)
            
            emb1 = reid_model.get_embedding_batch(crops1)
            emb2 = reid_model.get_embedding_batch(crops2)
            
            similarity = reid_model.compute_similarity(emb1, emb2)
            
            # Decide merge
            if similarity >= similarity_threshold:
                reid_merges.append({
                    'tracklet_1': tid1,
                    'tracklet_2': tid2,
                    'similarity': similarity,
                    'decision': 'MERGE'
                })
                if verbose:
                    print(f"  ✅ {tid1} + {tid2}: {similarity:.4f} (MERGE)")
            else:
                reid_rejections.append({
                    'tracklet_1': tid1,
                    'tracklet_2': tid2,
                    'similarity': similarity,
                    'decision': 'REJECT'
                })
                if verbose:
                    print(f"  ❌ {tid1} + {tid2}: {similarity:.4f} (REJECT)")
        
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Error processing {tid1} + {tid2}: {e}")
            reid_rejections.append({
                'tracklet_1': tid1,
                'tracklet_2': tid2,
                'similarity': -1.0,
                'decision': 'ERROR'
            })
    
    # Save results
    print(f"\n📊 ReID Results:")
    print(f"  ✅ Merges: {len(reid_merges)}")
    print(f"  ❌ Rejections: {len(reid_rejections)}")
    
    results = {
        'merges': reid_merges,
        'rejections': reid_rejections,
        'threshold': similarity_threshold
    }
    
    Path(reid_results_file).parent.mkdir(parents=True, exist_ok=True)
    with open(reid_results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Saved: {reid_results_file}")
    
    # For now, just save original tracklets (stage 4b will apply merges)
    np.savez_compressed(tracklets_recovered_file, tracklets=tracklets)
    print(f"💾 Saved: {tracklets_recovered_file}")
    
    print(f"\n{'='*80}")
    print(f"✅ STAGE 4a: ReID RECOVERY COMPLETE")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Stage 4a: ReID Recovery')
    parser.add_argument('--config', required=True, help='Config file path')
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_reid_recovery(config)


if __name__ == '__main__':
    main()
