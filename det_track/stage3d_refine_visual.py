#!/usr/bin/env python3
"""
Stage 3d: Visual Refinement (OSNet ReID-based Person Merging)

Uses OSNet embeddings to identify and merge split detections of the same person.
Takes output from Stage 3c (8-10 persons) and potentially reduces via ReID merging.

Input:
  - canonical_persons_filtered.npz: Filtered persons from Stage 3c (8-10 persons)
  - final_crops.pkl: Crops extracted by Stage 3c (8-10 persons, 50 crops each)

Algorithm:
  1. Load crops and canonical persons from Stage 3c
  2. Extract OSNet features (averaged per person)
  3. Find non-overlapping temporal pairs
  4. Compute cosine similarity for all pairs
  5. Build connected components (Union-Find) for same-person chains
  6. Merge crops and canonical person records
  7. Output merged files (OVERWRITE Stage 3c outputs with same filenames)

Output (overwrites Stage 3c outputs):
  - canonical_persons_filtered.npz: Merged persons (potentially fewer than input)
  - final_crops.pkl: Merged crops (potentially fewer than input)
  - merging_report.json: Details of detected person chains

Note: If no merges found, outputs are identical to inputs (just copied).

Usage:
    python stage3d_refine_visual.py --config configs/pipeline_config.yaml
"""

import argparse
import yaml
import numpy as np
import json
import time
import re
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import PipelineLogger
from datetime import datetime, timezone

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

import cv2


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


def load_osnet_model(model_path: str, device: str = 'cuda'):
    """Load OSNet model (ONNX or PyTorch)"""
    if str(model_path).endswith('.onnx'):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        session = ort.InferenceSession(str(model_path), providers=providers)
        return session, 'onnx'
    elif str(model_path).endswith(('.pt', '.pth')):
        # Simplified model loading - just load state dict
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available for .pt model")
        device_obj = torch.device(device)
        # For now, we'll use ONNX models primarily
        raise NotImplementedError("PyTorch model loading not implemented, use ONNX model")


def preprocess_crops(crops: List[np.ndarray], target_size: Tuple[int, int] = (256, 128)):
    """Preprocess crops for OSNet"""
    resized = [cv2.resize(crop, (target_size[1], target_size[0])) for crop in crops]
    batch_list = []
    for crop in resized:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        rgb = np.transpose(rgb, (2, 0, 1))
        batch_list.append(rgb)
    return np.stack(batch_list), resized


def extract_osnet_features(crops: List[np.ndarray], model, device: str = 'cuda', model_type: str = 'onnx'):
    """Extract OSNet features from crops (averaged)"""
    if len(crops) == 0:
        return np.array([]).reshape(0, 256)
    
    batch_array, _ = preprocess_crops(crops, target_size=(256, 128))
    
    if model_type == 'onnx':
        # Pad to 16 crops if needed
        if len(batch_array) != 16:
            padding = np.zeros((16 - len(batch_array), 3, 256, 128), dtype=np.float32)
            batch_array = np.concatenate([batch_array, padding], axis=0)
            original_len = len(crops)
        else:
            original_len = len(crops)
        
        input_name = model.get_inputs()[0].name
        feat = model.run(None, {input_name: batch_array})
        return feat[0][:original_len]


def load_crops_from_pkl(pkl_path: str) -> Dict:
    """Load crops from final_crops.pkl"""
    with open(pkl_path, 'rb') as f:
        final_crops = pickle.load(f)
    
    person_ids = final_crops.get('person_ids', [])
    crops_by_person = final_crops.get('crops', {})
    
    crops_dict = {}
    for person_id in person_ids:
        if person_id in crops_by_person:
            crop_list = crops_by_person[person_id]
            crops_dict[person_id] = crop_list
    
    return crops_dict


def load_canonical_persons(npz_path: str) -> Tuple[Dict, List]:
    """Load canonical persons and extract frame ranges"""
    data = np.load(npz_path, allow_pickle=True)
    persons_list = list(data['persons'])
    
    person_info = {}
    for person in persons_list:
        person_id = person['person_id']
        frame_numbers = person['frame_numbers']
        min_frame = frame_numbers.min()
        max_frame = frame_numbers.max()
        person_info[person_id] = {
            'min_frame': int(min_frame),
            'max_frame': int(max_frame),
            'num_frames': len(frame_numbers)
        }
    
    return person_info, persons_list


def check_temporal_overlap(frame_range1: Tuple[int, int], frame_range2: Tuple[int, int], overlap_tolerance: int = 0) -> bool:
    """Check if two frame ranges overlap beyond tolerance"""
    min1, max1 = frame_range1
    min2, max2 = frame_range2
    
    overlap_start = max(min1, min2)
    overlap_end = min(max1, max2)
    
    if overlap_end < overlap_start:
        return False
    
    overlap_size = overlap_end - overlap_start + 1
    return overlap_size > overlap_tolerance


def find_non_overlapping_pairs(person_info: Dict, temporal_gap_max: int, overlap_tolerance: int) -> List[Tuple[int, int]]:
    """Find all pairs with minimal temporal overlap and reasonable gaps"""
    person_ids = sorted(person_info.keys())
    pairs = []
    
    for i in range(len(person_ids)):
        for j in range(i+1, len(person_ids)):
            id1, id2 = person_ids[i], person_ids[j]
            range1 = (person_info[id1]['min_frame'], person_info[id1]['max_frame'])
            range2 = (person_info[id2]['min_frame'], person_info[id2]['max_frame'])
            
            if not check_temporal_overlap(range1, range2, overlap_tolerance=overlap_tolerance):
                # Calculate gap
                if range1[1] < range2[0]:
                    gap = range2[0] - range1[1]
                else:
                    gap = range1[0] - range2[1]
                
                # Only consider pairs with reasonable gaps
                if gap <= temporal_gap_max:
                    pairs.append((id1, id2))
    
    return pairs


class UnionFind:
    """Union-Find for connected components"""
    def __init__(self, elements):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
    
    def get_components(self):
        components = defaultdict(list)
        for e in self.parent.keys():
            root = self.find(e)
            components[root].append(e)
        return list(components.values())


def merge_crops(person_ids: List[int], crops_dict: Dict) -> Tuple[List[np.ndarray], Dict]:
    """Merge crops from multiple persons into one"""
    merged_crops = []
    person_crop_counts = {}
    
    for person_id in person_ids:
        if person_id in crops_dict:
            crops = crops_dict[person_id]
            merged_crops.extend(crops)
            person_crop_counts[person_id] = len(crops)
    
    return merged_crops, person_crop_counts


def merge_canonical_persons(person_ids: List[int], persons_dict: Dict) -> Dict:
    """Merge canonical person records"""
    merged_person = {
        'person_id': person_ids[0],  # Use first ID as merged person ID
        'merged_from_ids': person_ids,
        'frame_numbers': np.array([]),
        'bboxes': np.array([]).reshape(0, 4),
        'confidences': np.array([])
    }
    
    all_frames = []
    all_bboxes = []
    all_confidences = []
    
    for person_id in person_ids:
        if person_id in persons_dict:
            person = persons_dict[person_id]
            all_frames.extend(person['frame_numbers'])
            all_bboxes.extend(person['bboxes'])
            all_confidences.extend(person['confidences'])
    
    # Sort by frame number
    if all_frames:
        indices = np.argsort(all_frames)
        merged_person['frame_numbers'] = np.array(all_frames)[indices]
        merged_person['bboxes'] = np.array(all_bboxes)[indices]
        merged_person['confidences'] = np.array(all_confidences)[indices]
    
    return merged_person


def run_refine(config):
    """Run Stage 3d: Visual Refinement"""
    
    stage_config = config['stage3d_refine']
    verbose = stage_config.get('advanced', {}).get('verbose', False) or config.get('global', {}).get('verbose', False)
    
    logger = PipelineLogger("Stage 3d: Visual Refinement", verbose=verbose)
    logger.header()
    
    # Extract configuration
    input_config = stage_config['input']
    output_config = stage_config['output']
    osnet_config = stage_config['osnet']
    merging_config = stage_config['merging']
    
    crops_file = input_config['final_crops_file']
    canonical_file = input_config['canonical_persons_file']
    
    # Get output files with fallbacks
    output_crops_file = output_config.get('final_crops_merged_file', crops_file)  # Default to same name as input
    output_canonical_file = output_config.get('canonical_persons_merged_file', canonical_file)  # Default to same name
    output_report_file = output_config.get('merging_report_file', str(Path(canonical_file).parent / 'merging_report.json'))
    
    # Load crops and canonical persons
    logger.info(f"Loading crops: {Path(crops_file).name}")
    crops_dict = load_crops_from_pkl(crops_file)
    logger.stat("Persons with crops", len(crops_dict))
    
    logger.info(f"Loading canonical persons: {Path(canonical_file).name}")
    person_info, persons_list = load_canonical_persons(canonical_file)
    persons_dict = {p['person_id']: p for p in persons_list}
    logger.stat("Total persons to process", len(person_info))
    
    # Load OSNet model
    logger.info(f"Loading OSNet model...")
    model_path = osnet_config['model_path']
    device = osnet_config['device']
    
    try:
        model, model_type = load_osnet_model(model_path, device=device)
        logger.found(f"Model loaded ({model_type}): {Path(model_path).name}")
    except Exception as e:
        logger.error(f"Failed to load OSNet model: {e}")
        return
    
    # Extract features for each person
    logger.info(f"Extracting OSNet features (averaged per person)...")
    t_feature_start = time.time()
    
    person_features = {}
    sorted_persons = sorted(person_info.keys())
    
    for person_id in sorted_persons:
        if person_id in crops_dict:
            crops = crops_dict[person_id]
            # Use top N crops
            num_crops = osnet_config.get('num_crops_per_person', 16)
            selected_crops = crops[:min(num_crops, len(crops))]
            
            try:
                features = extract_osnet_features(selected_crops, model, device=device, model_type=model_type)
                avg_feature = features.mean(axis=0)
                person_features[person_id] = avg_feature
            except Exception as e:
                logger.warning(f"Failed to extract features for person_{person_id}: {e}")
                continue
    
    t_feature_elapsed = time.time() - t_feature_start
    logger.stat("Features extracted", len(person_features))
    logger.timing("Feature extraction", t_feature_elapsed)
    
    # Find non-overlapping pairs and compute similarities
    logger.info(f"Finding candidate pairs and computing similarities...")
    t_pair_start = time.time()
    
    temporal_gap_max = merging_config['temporal_gap_max']
    overlap_tolerance = merging_config['temporal_overlap_tolerance']
    similarity_threshold = merging_config['similarity_threshold']
    
    pairs = find_non_overlapping_pairs(person_info, temporal_gap_max, overlap_tolerance)
    logger.stat("Candidate pairs found", len(pairs))
    
    pair_similarities = {}
    connections = []
    
    for id1, id2 in pairs:
        if id1 not in person_features or id2 not in person_features:
            continue
        
        feat1 = person_features[id1]
        feat2 = person_features[id2]
        
        # Normalize and compute cosine similarity
        feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-10)
        feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-10)
        similarity = float(np.dot(feat1_norm, feat2_norm))
        
        pair_similarities[(id1, id2)] = similarity
        
        if similarity >= similarity_threshold:
            connections.append((id1, id2))
            if verbose:
                logger.verbose_info(f"  person_{id1} ↔ person_{id2}: {similarity:.4f} ✓")
    
    t_pair_elapsed = time.time() - t_pair_start
    logger.stat("High-similarity pairs", len(connections))
    logger.timing("Similarity computation", t_pair_elapsed)
    
    # Build connected components
    logger.info(f"Building connected components...")
    uf = UnionFind(sorted_persons)
    for id1, id2 in connections:
        uf.union(id1, id2)
    
    components = uf.get_components()
    components.sort(key=lambda x: min(x))
    
    num_merged = sum(1 for c in components if len(c) > 1)
    logger.found(f"{num_merged} person group(s) identified (merges needed)")
    
    if verbose:
        for group_idx, component in enumerate(components, 1):
            component.sort()
            if len(component) > 1:
                logger.verbose_info(f"  Group {group_idx}: {[f'person_{p}' for p in component]}")
    
    # Merge crops and persons
    logger.info(f"Merging crops and canonical persons...")
    t_merge_start = time.time()
    
    merged_crops_dict = {}
    merged_persons_list = []
    merging_report = []
    
    for component in components:
        component.sort()
        
        if len(component) > 1:
            # Merge multiple persons
            merged_crops, crop_counts = merge_crops(component, crops_dict)
            merged_crops_dict[component[0]] = merged_crops
            
            merged_person = merge_canonical_persons(component, persons_dict)
            merged_persons_list.append(merged_person)
            
            # Record merge info
            merging_report.append({
                'merged_persons': component,
                'merged_into_id': component[0],
                'similarity_scores': {
                    f"{id1}-{id2}": float(pair_similarities.get((min(id1, id2), max(id1, id2)), -1))
                    for id1, id2 in combinations(component, 2)
                },
                'total_merged_frames': int(merged_person['frame_numbers'].shape[0])
            })
        else:
            # Keep single person as-is
            person_id = component[0]
            if person_id in crops_dict:
                merged_crops_dict[person_id] = crops_dict[person_id]
            if person_id in persons_dict:
                merged_persons_list.append(persons_dict[person_id])
    
    t_merge_elapsed = time.time() - t_merge_start
    logger.stat("Persons after merging", len(merged_persons_list))
    logger.timing("Merging", t_merge_elapsed)
    
    # Save merged outputs
    logger.info(f"Saving merged crops: {Path(output_crops_file).name}")
    output_crops_path = Path(output_crops_file)
    output_crops_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Re-organize crops_dict with new person IDs
    final_crops_dict = {
        'person_ids': [p['person_id'] for p in merged_persons_list],
        'crops': merged_crops_dict,
        'metadata': {
            'stage': 'stage3d_refine',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'num_merged_groups': num_merged,
            'num_persons_final': len(merged_persons_list)
        }
    }
    
    with open(output_crops_path, 'wb') as f:
        pickle.dump(final_crops_dict, f)
    
    logger.info(f"Saving merged canonical persons: {Path(output_canonical_file).name}")
    output_canonical_path = Path(output_canonical_file)
    output_canonical_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_canonical_path,
        persons=merged_persons_list
    )
    
    logger.info(f"Saving merging report: {Path(output_report_file).name}")
    output_report_path = Path(output_report_file)
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_report_path, 'w') as f:
        json.dump(merging_report, f, indent=2)
    
    logger.success()


def main():
    parser = argparse.ArgumentParser(description='Stage 3d: Visual Refinement')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Check if stage is enabled
    if not config['pipeline']['stages'].get('stage3d', False):
        logger = PipelineLogger("Stage 3d: Visual Refinement", verbose=False)
        logger.info("Stage 3d is disabled in config")
        return
    
    # Run refinement
    run_refine(config)


if __name__ == '__main__':
    main()
