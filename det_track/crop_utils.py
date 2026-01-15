#!/usr/bin/env python3
"""
Crop utilities for Stage 3c and Stage 4.

Provides functions to:
1. Extract crops from video with quality metadata
2. Save to final_crops.pkl with quality scores
3. Load and filter crops by quality
"""

import numpy as np
import cv2
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime, timezone


def compute_crop_quality_metrics(
    bbox: List[float],
    frame_shape: Tuple[int, int],
    confidence: float = 1.0,
    frame_number: int = 0
) -> Dict[str, float]:
    """
    Compute quality metrics for a crop.
    
    Args:
        bbox: [x1, y1, x2, y2] bounding box
        frame_shape: (height, width) of frame
        confidence: Detection confidence (0-1)
        frame_number: Frame number in video
    
    Returns:
        Dict with quality metrics:
        - confidence: Detection confidence
        - width, height: Crop dimensions in pixels
        - area: Total pixels
        - aspect_ratio: width/height
        - visibility_score: 0-1 based on bbox normalization and aspect ratio
    """
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    frame_h, frame_w = frame_shape[:2]
    
    # Clamp to frame boundaries
    x1 = max(0, min(x1, frame_w))
    x2 = max(0, min(x2, frame_w))
    y1 = max(0, min(y1, frame_h))
    y2 = max(0, min(y2, frame_h))
    
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    area = width * height if width > 0 and height > 0 else 0
    
    # Aspect ratio: penalize very wide or very tall crops
    aspect_ratio = width / height if height > 0 else 0
    aspect_penalty = 1.0
    if aspect_ratio > 0:
        # Ideal ratio is ~2.0 (256x128), penalize deviations
        ideal_ratio = 2.0
        ratio_error = abs(aspect_ratio - ideal_ratio) / ideal_ratio
        aspect_penalty = max(0.3, 1.0 - ratio_error)  # Min 0.3x penalty
    
    # Visibility score based on:
    # 1. How centered the crop is (normalized coordinates)
    # 2. How well-sized it is relative to frame
    # 3. Aspect ratio quality
    
    # Normalized center coordinates (0.5 = perfectly centered)
    norm_x_center = (x1 + x2) / 2.0 / frame_w if frame_w > 0 else 0.5
    norm_y_center = (y1 + y2) / 2.0 / frame_h if frame_h > 0 else 0.5
    
    # Center penalty: optimal is 0.5, penalize extremes
    center_error = min(abs(norm_x_center - 0.5), abs(norm_y_center - 0.5)) * 2
    center_penalty = max(0.5, 1.0 - center_error)
    
    # Size penalty: encourage medium-sized crops (not too small, not too large)
    frame_area = frame_w * frame_h
    crop_ratio = area / frame_area if frame_area > 0 else 0
    if crop_ratio < 0.01:  # Too small
        size_penalty = 0.3
    elif crop_ratio > 0.3:  # Too large
        size_penalty = 0.5
    else:  # Ideal range
        size_penalty = 1.0
    
    visibility_score = (aspect_penalty * size_penalty * center_penalty)
    
    return {
        'confidence': float(confidence),
        'width': int(width),
        'height': int(height),
        'area': int(area),
        'aspect_ratio': float(aspect_ratio),
        'visibility_score': float(visibility_score),
        'frame_number': int(frame_number)
    }


def compute_crop_quality_rank(
    quality_metrics: Dict[str, float],
    max_area: float = 50000.0
) -> Tuple[float, int]:
    """
    Compute composite quality score and rank for a crop.
    
    Combines: confidence × (area / max_area) × visibility_score
    
    Args:
        quality_metrics: Output from compute_crop_quality_metrics()
        max_area: Maximum area for normalization
    
    Returns:
        (quality_score, rank_priority)
        - quality_score: 0-1 composite score
        - rank_priority: integer for sorting (higher = better)
    """
    conf = quality_metrics['confidence']
    area = quality_metrics['area']
    visibility = quality_metrics['visibility_score']
    
    # Composite score: confidence × area_norm × visibility
    area_norm = min(1.0, area / max_area)
    quality_score = conf * area_norm * visibility
    
    # Rank priority: 0-100 scale
    rank_priority = int(quality_score * 100)
    
    return quality_score, rank_priority


def extract_crops_with_quality(
    video_path: str,
    persons: List[Dict[str, Any]],
    target_crops_per_person: int = 50,
    top_n: int = 10,
    max_first_appearance_ratio: float = 0.5,
    verbose: bool = False
) -> Dict[int, Dict[str, Any]]:
    """
    Extract person crops with quality metrics.
    
    Same as ondemand_crop_extraction.extract_crops_from_video()
    but also computes quality metrics for each crop.
    
    Returns:
        Dict mapping person_id -> {
            'crops': List[np.ndarray],  # (50, H, W, 3)
            'metadata': List[Dict],      # Quality metrics per crop
            'max_area': float            # Used for quality normalization
        }
    """
    from ondemand_crop_extraction import extract_crops_from_video
    
    if verbose:
        print("[Crop Utils] Extracting crops with quality metrics...")
    
    start = time.time()
    
    # Use existing extraction logic
    person_buckets, extraction_metadata = extract_crops_from_video(
        video_path=video_path,
        persons=persons,
        target_crops_per_person=target_crops_per_person,
        top_n=top_n,
        max_first_appearance_ratio=max_first_appearance_ratio,
        verbose=verbose
    )
    
    # Now compute quality metrics for each crop
    crops_with_quality = {}
    
    cap = cv2.VideoCapture(str(video_path))
    frame_shape = None
    
    # First pass: get frame shape
    ret, frame = cap.read()
    if ret:
        frame_shape = frame.shape
    cap.release()
    
    if frame_shape is None:
        raise RuntimeError(f"Cannot read frame from video: {video_path}")
    
    # Compute max area across all crops for normalization
    all_areas = []
    
    for person_id, crops in person_buckets.items():
        metadata_list = []
        for crop in crops:
            h, w = crop.shape[:2]
            all_areas.append(h * w)
    
    max_area = np.percentile(all_areas, 95) if all_areas else 1.0
    if verbose:
        print(f"[Crop Utils] Max area (p95): {max_area:.0f} pixels")
    
    # Second pass: extract crops and their metadata
    # Re-extract to maintain alignment
    person_buckets, _ = extract_crops_from_video(
        video_path=video_path,
        persons=persons,
        target_crops_per_person=target_crops_per_person,
        top_n=top_n,
        max_first_appearance_ratio=max_first_appearance_ratio,
        verbose=False
    )
    
    for person_id, crops in person_buckets.items():
        person = next((p for p in persons if int(p['person_id']) == person_id), None)
        if person is None:
            continue
        
        bboxes = person['bboxes']
        frame_nums = person['frame_numbers']
        
        metadata_list = []
        for idx, crop in enumerate(crops):
            if idx < len(bboxes):
                bbox = bboxes[idx]
                frame_num = frame_nums[idx] if idx < len(frame_nums) else 0
                
                # Get confidence from tracklet data
                # Fallback: assume high confidence if not available
                confidence = 0.95
                
                metrics = compute_crop_quality_metrics(
                    bbox=bbox,
                    frame_shape=frame_shape,
                    confidence=confidence,
                    frame_number=frame_num
                )
                metadata_list.append(metrics)
            else:
                # Fallback if alignment issue
                metrics = compute_crop_quality_metrics(
                    bbox=[0, 0, frame_shape[1], frame_shape[0]],
                    frame_shape=frame_shape,
                    confidence=0.9,
                    frame_number=0
                )
                metadata_list.append(metrics)
        
        crops_with_quality[person_id] = {
            'crops': crops,
            'metadata': metadata_list,
            'max_area': max_area
        }
    
    elapsed = time.time() - start
    if verbose:
        print(f"[Crop Utils] Quality metrics computed in {elapsed:.2f}s")
    
    return crops_with_quality


def save_final_crops(
    crops_with_quality: Dict[int, Dict[str, Any]],
    output_path: str,
    video_source: str = "",
    verbose: bool = False
) -> None:
    """
    Save crops with quality metadata to final_crops.pkl
    
    Args:
        crops_with_quality: Output from extract_crops_with_quality()
        output_path: Path to save final_crops.pkl
        video_source: Source video path (for metadata)
        verbose: Print details
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"[Crop Utils] Saving final_crops.pkl...")
    
    start = time.time()
    
    # Build pickle data structure
    data_to_save = {
        'person_ids': sorted(crops_with_quality.keys()),
        'crops': {},
        'metadata': {}
    }
    
    total_crops = 0
    
    for person_id, data in crops_with_quality.items():
        crops = np.array(data['crops'])  # (num_crops, H, W, 3)
        metadata = data['metadata']
        max_area = data['max_area']
        
        # Compute quality ranks for this person
        quality_ranks = []
        for i, metrics in enumerate(metadata):
            score, rank = compute_crop_quality_rank(metrics, max_area)
            metrics['quality_score'] = score
            metrics['quality_rank'] = rank
        
        data_to_save['crops'][person_id] = crops
        data_to_save['metadata'][person_id] = metadata
        
        total_crops += len(crops)
    
    # Add global metadata
    data_to_save['global_metadata'] = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'video_source': video_source,
        'total_persons': len(data_to_save['person_ids']),
        'total_crops': total_crops,
        'crops_per_person': 50,
        'format_version': '1.0'
    }
    
    # Save to pickle
    with open(output_path, 'wb') as f:
        pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    elapsed = time.time() - start
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    
    if verbose:
        print(f"[Crop Utils] Saved {len(data_to_save['person_ids'])} persons, {total_crops} crops to {output_path.name}")
        print(f"[Crop Utils] File size: {file_size_mb:.1f} MB, Time: {elapsed:.2f}s")


def load_final_crops(output_path: str, verbose: bool = False) -> Dict[int, Dict[str, Any]]:
    """
    Load final_crops.pkl
    
    Args:
        output_path: Path to final_crops.pkl
        verbose: Print details
    
    Returns:
        Dict with 'crops', 'metadata', 'person_ids', 'global_metadata'
    """
    output_path = Path(output_path)
    
    if not output_path.exists():
        raise FileNotFoundError(
            f"final_crops.pkl not found: {output_path}\n"
            "Stage 4 requires Stage 3c to be completed first.\n"
            "Run: python run_pipeline.py --stages 3c,4"
        )
    
    if verbose:
        print(f"[Crop Utils] Loading final_crops.pkl...")
    
    start = time.time()
    
    with open(output_path, 'rb') as f:
        data = pickle.load(f)
    
    elapsed = time.time() - start
    
    if verbose:
        num_persons = len(data.get('person_ids', []))
        total_crops = data.get('global_metadata', {}).get('total_crops', 0)
        print(f"[Crop Utils] Loaded {num_persons} persons, {total_crops} crops in {elapsed:.2f}s")
    
    return data
