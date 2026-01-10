#!/usr/bin/env python3
"""
Stage 4b.5: Enrich Crops with Person IDs and Metadata (HDF5)

Purpose:
    Creates crops_enriched.h5 from canonical_persons.npz and crops_cache.pkl
    Provides fast indexed lookups for crop retrieval by person_id and frame_id
    Enables Stage 9 (GIF generation) and future stages to access correct person crops

Data Flow:
    INPUT:
        - canonical_persons.npz: Person IDs and frame ranges
        - crops_cache.pkl: BGR crop images indexed by {frame_idx: {local_det_idx: crop_image}}
        - detections_raw.npz: Frame numbers, bboxes, confidences for bbox reconstruction
    
    OUTPUT:
        - crops_enriched.h5: HDF5 file with structure:
            /person_03/
                frame_123: {frame_id, image_bgr, bbox, width, height}
                frame_125: {frame_id, image_bgr, bbox, width, height}
            /person_65/
                frame_100: {frame_id, image_bgr, bbox, width, height}
                ...

Key Features:
    - O(1) indexed lookups: enriched[person_id][frame_id]
    - BGR format (native OpenCV)
    - Bbox coordinates for context
    - Fast HDF5 access with Gzip compression
    - No external dependencies on crops_cache.pkl after creation

Author: Unified Pose Pipeline
"""

import os
import pickle
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_canonical_persons(canonical_persons_file: str) -> List[Dict]:
    """
    Load canonical persons from NPZ file.
    
    Returns:
        List of person dicts with structure:
        {
            'person_id': int,
            'tracklet_ids': [int, ...],
            'frame_numbers': np.array([...]),
            'bboxes': np.array([...]),
            'confidences': np.array([...])
        }
    """
    data = np.load(canonical_persons_file, allow_pickle=True)
    persons = data['persons'].tolist()
    return persons


def load_crops_cache(crops_cache_file: str) -> Dict:
    """
    Load crops cache from pickle file.
    
    Returns:
        Dict: {frame_idx: {local_det_idx: crop_image_BGR}}
    """
    with open(crops_cache_file, 'rb') as f:
        crops_cache = pickle.load(f)
    return crops_cache


def load_detections(detections_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load detections from NPZ file.
    
    Returns:
        Tuple of (frame_numbers, bboxes, confidences)
    """
    data = np.load(detections_file, allow_pickle=True)
    frame_numbers = data['frame_numbers']
    bboxes = data['bboxes']
    confidences = data['confidences']
    return frame_numbers, bboxes, confidences


def build_detection_index(frame_numbers: np.ndarray, bboxes: np.ndarray, 
                         confidences: np.ndarray) -> Dict:
    """
    Build a lookup index for detections by frame.
    
    Returns:
        Dict: {frame_idx: [(local_det_idx, bbox, confidence), ...]}
    """
    frame_index = defaultdict(list)
    
    for global_det_idx in range(len(frame_numbers)):
        frame_idx = frame_numbers[global_det_idx]
        bbox = bboxes[global_det_idx]
        confidence = confidences[global_det_idx]
        frame_index[frame_idx].append((global_det_idx, bbox, confidence))
    
    return dict(frame_index)


def compute_bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two bboxes.
    
    Args:
        bbox1, bbox2: [x1, y1, x2, y2] format
    
    Returns:
        float: IoU score in [0, 1]
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def find_best_crop_for_bbox(frame_idx: int, target_bbox: np.ndarray, 
                            frame_index: Dict, crops_cache: Dict,
                            iou_threshold: float = 0.3) -> Tuple[np.ndarray, bool]:
    """
    Find the best matching crop for a given bbox in a frame using IoU.
    
    Args:
        frame_idx: Frame number to search
        target_bbox: Target bbox [x1, y1, x2, y2]
        frame_index: Detection index by frame
        crops_cache: Crops indexed by frame and detection
        iou_threshold: Minimum IoU to accept a match
    
    Returns:
        Tuple of (crop_image or None, match_found)
    """
    if frame_idx not in frame_index:
        return None, False
    
    if frame_idx not in crops_cache:
        return None, False
    
    frame_detections = frame_index[frame_idx]
    frame_crops = crops_cache[frame_idx]
    
    best_iou = -1
    best_crop = None
    
    # frame_detections is list of (global_det_idx, detection_bbox, confidence)
    # We need to match each detection to its corresponding crop
    # The local_det_idx in frame_crops should correspond to the position in frame_detections
    
    for local_idx, (global_det_idx, detection_bbox, confidence) in enumerate(frame_detections):
        # local_idx is the position within this frame's detections
        if local_idx not in frame_crops:
            continue
        
        crop = frame_crops[local_idx]
        if crop is None or crop.size == 0:
            continue
        
        iou = compute_bbox_iou(target_bbox, detection_bbox)
        
        if iou > best_iou:
            best_iou = iou
            best_crop = crop
    
    if best_iou >= iou_threshold:
        return best_crop, True
    
    # Fallback: if no good match, return ANY crop from frame
    if len(frame_crops) > 0:
        first_crop = frame_crops[min(frame_crops.keys())]
        return first_crop, False
    
    return None, False


def enrich_crops_to_hdf5(canonical_persons_file: str, crops_cache_file: str,
                         detections_file: str, output_file: str,
                         compression: str = 'gzip',
                         iou_threshold: float = 0.3) -> None:
    """
    Create enriched crops HDF5 file from canonical persons and crops cache.
    
    Args:
        canonical_persons_file: Path to canonical_persons.npz
        crops_cache_file: Path to crops_cache.pkl
        detections_file: Path to detections_raw.npz
        output_file: Path to output crops_enriched.h5
        compression: HDF5 compression method (gzip, lzf, None)
        iou_threshold: Minimum IoU for bbox matching
    """
    logger.info("=" * 80)
    logger.info("STAGE 4b.5: ENRICH CROPS WITH PERSON IDS AND METADATA (HDF5)")
    logger.info("=" * 80)
    
    # Load data
    logger.info(f"Loading canonical persons from {canonical_persons_file}")
    persons = load_canonical_persons(canonical_persons_file)
    logger.info(f"  → Loaded {len(persons)} persons")
    
    logger.info(f"Loading crops cache from {crops_cache_file}")
    crops_cache = load_crops_cache(crops_cache_file)
    logger.info(f"  → Loaded crops for {len(crops_cache)} frames")
    
    logger.info(f"Loading detections from {detections_file}")
    frame_numbers, bboxes, confidences = load_detections(detections_file)
    logger.info(f"  → Loaded {len(frame_numbers)} detections")
    
    # Build detection index for fast frame lookup
    logger.info("Building detection index by frame...")
    frame_index = build_detection_index(frame_numbers, bboxes, confidences)
    logger.info(f"  → Indexed {len(frame_index)} unique frames")
    
    # Create HDF5 file
    logger.info(f"Creating HDF5 file: {output_file}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_file, 'w') as h5f:
        h5f.attrs['description'] = 'Enriched crops with person IDs and metadata'
        h5f.attrs['format_version'] = '1.0'
        h5f.attrs['color_space'] = 'BGR'
        
        total_frames_stored = 0
        total_frames_skipped = 0
        
        # Process each person
        for person_idx, person in enumerate(persons, 1):
            person_id = person['person_id']
            frame_numbers_person = person['frame_numbers']
            bboxes_person = person['bboxes']
            
            logger.info(f"Processing person {person_idx}/{len(persons)} "
                       f"(ID: {person_id}, frames: {len(frame_numbers_person)})")
            
            # Create group for this person
            person_group = h5f.create_group(f'person_{person_id:02d}')
            person_group.attrs['person_id'] = person_id
            person_group.attrs['num_frames'] = len(frame_numbers_person)
            
            frames_stored = 0
            frames_skipped = 0
            
            # Store crop for each frame of this person
            for frame_local_idx, frame_idx in enumerate(frame_numbers_person):
                bbox = bboxes_person[frame_local_idx]
                
                # Find best matching crop
                crop_image, match_found = find_best_crop_for_bbox(
                    frame_idx, bbox, frame_index, crops_cache, 
                    iou_threshold=iou_threshold
                )
                
                if crop_image is None:
                    frames_skipped += 1
                    total_frames_skipped += 1
                    continue
                
                # Store in HDF5
                frame_dataset_name = f'frame_{frame_idx:06d}'
                frame_group = person_group.create_group(frame_dataset_name)
                
                # Store metadata
                frame_group.attrs['frame_id'] = frame_idx
                frame_group.attrs['bbox'] = bbox  # [x1, y1, x2, y2]
                frame_group.attrs['width'] = crop_image.shape[1]
                frame_group.attrs['height'] = crop_image.shape[0]
                frame_group.attrs['match_found'] = match_found
                
                # Store image data
                frame_group.create_dataset('image_bgr', data=crop_image, 
                                          compression=compression)
                
                frames_stored += 1
                total_frames_stored += 1
            
            logger.info(f"  → Stored {frames_stored} frames, skipped {frames_skipped}")
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"HDF5 Enrichment Complete")
        logger.info(f"  Total frames stored: {total_frames_stored}")
        logger.info(f"  Total frames skipped: {total_frames_skipped}")
        logger.info(f"  Output file: {output_file}")
        logger.info(f"  File size: {Path(output_file).stat().st_size / (1024**2):.2f} MB")
        logger.info(f"{'=' * 80}\n")


def main():
    """
    Main entry point for stage4b5_enrich_crops.py.
    Reads from YAML config like other pipeline stages.
    """
    import argparse
    from pathlib import Path
    import sys
    import yaml
    import os
    
    parser = argparse.ArgumentParser(
        description='Stage 4b.5: Enrich crops with person IDs and metadata (HDF5)'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Auto-extract current_video from video_file (same as run_pipeline.py)
    video_file = config.get('global', {}).get('video_file', '')
    if video_file:
        video_name = os.path.splitext(video_file)[0]
        config['global']['current_video'] = video_name
    
    # Resolve paths using the same resolver as other stages
    from pathlib import Path
    import re
    
    def resolve_path_variables(cfg):
        """Recursively resolve ${variable} in config"""
        global_vars = cfg.get('global', {})
        
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
        
        result = resolve_recursive(cfg)
        result['global'] = global_vars
        return result
    
    config = resolve_path_variables(config)
    
    # Get paths from config
    stage_config = config['stage6']
    canonical_persons_file = stage_config['input']['canonical_persons_file']
    crops_cache_file = stage_config['input']['crops_cache_file']
    detections_file = stage_config['input']['detections_file']
    crops_enriched_file = stage_config['output']['crops_enriched_file']
    
    # Validate inputs
    if not Path(canonical_persons_file).exists():
        logger.error(f"canonical_persons.npz not found: {canonical_persons_file}")
        sys.exit(1)
    if not Path(crops_cache_file).exists():
        logger.error(f"crops_cache.pkl not found: {crops_cache_file}")
        sys.exit(1)
    if not Path(detections_file).exists():
        logger.error(f"detections_raw.npz not found: {detections_file}")
        sys.exit(1)
    
    # Run enrichment
    enrich_crops_to_hdf5(
        canonical_persons_file,
        crops_cache_file,
        detections_file,
        crops_enriched_file,
        compression=stage_config.get('enrichment', {}).get('compression', 'gzip'),
        iou_threshold=0.3
    )



if __name__ == '__main__':
    main()
