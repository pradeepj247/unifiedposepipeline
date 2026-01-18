#!/usr/bin/env python3
"""
OSNet Clustering Module - ReID-based Person Duplicate Detection

Extracts OSNet embeddings from person crops and computes similarity matrix
to detect duplicate canonical persons (same physical person appearing at different times).

Algorithm:
1. Load OSNet x0.25 model (ReID network)
2. Select 8 best crops per person (high-confidence, diverse)
3. Extract features in batch (batch_size=8)
4. Average features and L2 normalize to get person embedding
5. Compute NxN cosine similarity matrix
6. Identify duplicate pairs (>70% similarity)

Performance: ~2 seconds for 10 persons (8 crops × 10 = 80 crops)

Usage:
    from osnet_clustering import create_similarity_matrix
    
    result = create_similarity_matrix(
        buckets=person_buckets,          # Dict[person_id: [crop1, crop2, ...]]
        osnet_model_path='osnet_x0_25_msmt17.pth',
        device='cuda'
    )
    
    similarity_matrix = result['similarity_matrix']  # (10, 10) array
    embeddings = result['embeddings']                # Dict[person_id: (256,) array]
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import time
import json

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class OSNetModel(nn.Module):
    """Lightweight OSNet x0.25 model for ReID (256-dim embeddings)"""
    
    def __init__(self, num_classes: int = 1000, feature_dim: int = 256):
        super(OSNetModel, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # Simplified OSNet x0.25 architecture
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Build residual blocks
        self.layer1 = self._make_layer(32, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, feature_dim)
        self.feat_bn = nn.BatchNorm1d(feature_dim)
        self.feat_bn.bias.requires_grad_(False)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        for i in range(blocks):
            stride_i = stride if i == 0 else 1
            layers.append(ResBlock(in_channels if i == 0 else out_channels, 
                                  out_channels, stride=stride_i))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        x = self.feat_bn(x)
        return x


class ResBlock(nn.Module):
    """Simple residual block"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out


def load_osnet_model(model_path: Optional[str] = None, 
                     fallback_model_path: Optional[str] = None,
                     device: str = 'cuda') -> Tuple[Any, str, str, str]:
    """
    Load OSNet model with fallback support. Priority: ONNX → PyTorch → fallback ONNX → fallback PyTorch → random init
    
    Supports:
    1. ONNX models (.onnx) - recommended for production, fast inference
    2. PyTorch models (.pt, .pth) - fallback if ONNX not available
    3. Fallback model path - used if primary model not found
    4. Randomly initialized - fallback if all models missing
    
    Args:
        model_path: Primary model path (.onnx or .pt/.pth)
        fallback_model_path: Secondary model path if primary not found
        device: 'cuda' or 'cpu'
    
    Returns:
        (model, device_str, model_type, actual_path): Loaded model, device, model type, and actual path loaded
        - model_type: 'onnx', 'pytorch', 'random'
        - actual_path: Path that was actually loaded, or 'random_init'
    """
    # Ensure device is available
    if device == 'cuda' and not torch.cuda.is_available() if TORCH_AVAILABLE else True:
        print("[OSNet] CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Try paths in priority order
    paths_to_try = []
    if model_path:
        paths_to_try.append(('primary', model_path))
    if fallback_model_path:
        paths_to_try.append(('fallback', fallback_model_path))
    
    # Attempt each path
    for priority_name, path in paths_to_try:
        if not path:
            continue
            
        # Try ONNX if ends with .onnx
        if Path(path).exists() and str(path).endswith('.onnx'):
            if ONNX_AVAILABLE:
                try:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
                    session = ort.InferenceSession(str(path), providers=providers)
                    print(f"[OSNet] ✓ Loaded ONNX model ({priority_name})")
                    print(f"[OSNet]   Path: {path}")
                    print(f"[OSNet]   Providers: {session.get_providers()}")
                    return session, device, 'onnx', str(path)
                except Exception as e:
                    print(f"[OSNet] ✗ Failed to load ONNX model ({priority_name}): {e}")
            else:
                print(f"[OSNet] ✗ ONNX Runtime not available for ({priority_name}) model")
        
        # Try PyTorch if ends with .pt or .pth
        elif Path(path).exists() and str(path).endswith(('.pt', '.pth')):
            if TORCH_AVAILABLE:
                try:
                    device_obj = torch.device(device)
                    model = OSNetModel(feature_dim=256)
                    state_dict = torch.load(path, map_location=device_obj)
                    model.load_state_dict(state_dict, strict=False)
                    print(f"[OSNet] ✓ Loaded PyTorch model ({priority_name})")
                    print(f"[OSNet]   Path: {path}")
                    print(f"[OSNet]   Device: {device}")
                    model.eval()
                    model.to(device_obj)
                    return model, device, 'pytorch', str(path)
                except Exception as e:
                    print(f"[OSNet] ✗ Failed to load PyTorch model ({priority_name}): {e}")
            else:
                print(f"[OSNet] ✗ PyTorch not available for ({priority_name}) model")
        else:
            print(f"[OSNet] ✗ ({priority_name}) path not found or unsupported format: {path}")
    
    # All paths failed, fall back to random initialization
    if TORCH_AVAILABLE:
        print("[OSNet]")
        print("[OSNet] ⚠️  WARNING: Using RANDOMLY INITIALIZED PyTorch model")
        print("[OSNet] ⚠️  All model paths failed to load!")
        print("[OSNet] ⚠️  Clustering similarity scores will be UNRELIABLE and MEANINGLESS")
        print("[OSNet]")
        device_obj = torch.device(device)
        model = OSNetModel(feature_dim=256)
        model.eval()
        model.to(device_obj)
        return model, device, 'random', 'random_init'
    
    # No suitable backend found
    print("[OSNet] ERROR: Neither ONNX nor PyTorch available!")
    print("[OSNet] Install one of:")
    print("  - pip install onnxruntime  (recommended)")
    print("  - pip install torch  (fallback)")
    raise RuntimeError("No OSNet backend available")


def select_best_crops(crops: List[np.ndarray], 
                      num: int = 8,
                      verbose: bool = False) -> List[np.ndarray]:
    """
    Select num best crops from bucket.
    
    Criteria:
    - Exclude size outliers (too small/large)
    - Prefer center-positioned crops (not edge cuts)
    - Random sample from remaining candidates
    
    Args:
        crops: List of crop images (H, W, 3) in BGR
        num: Number of crops to select
        verbose: Print selection details
    
    Returns:
        List of num selected crops
    """
    if len(crops) <= num:
        return crops
    
    # Score each crop by size (prefer medium-sized, exclude very small/large)
    scores = []
    for crop in crops:
        h, w = crop.shape[:2]
        area = h * w
        # Prefer crops in 100k-300k pixel range (not too small, not too large)
        if 50000 < area < 500000:
            # Bonus for square-ish crops (more centered)
            aspect = max(h/w, w/h)
            size_score = 1.0 - abs(area - 200000) / 200000  # Prefer ~200k
            aspect_score = 1.0 - (aspect - 1.0) / 4.0  # Prefer square
            scores.append(0.7 * size_score + 0.3 * aspect_score)
        else:
            scores.append(0.0)  # Exclude outliers
    
    # Get top num candidates by score
    sorted_indices = np.argsort(scores)[::-1]
    selected_indices = sorted_indices[:num]
    
    # Sort selected indices to preserve temporal order
    selected_indices = sorted(selected_indices)
    
    if verbose:
        print(f"[Select Best Crops] Selected {len(selected_indices)}/{len(crops)} crops")
    
    return [crops[i] for i in selected_indices]


def preprocess_crops(crops: List[np.ndarray], 
                    target_size: Tuple[int, int] = (256, 128),
                    verbose: bool = False) -> Tuple[torch.Tensor, List[np.ndarray]]:
    """
    Preprocess crops for OSNet inference.
    
    Args:
        crops: List of crop images (H, W, 3) BGR
        target_size: Resize to (height, width)
        verbose: Print preprocessing details
    
    Returns:
        (batch_tensor, resized_crops): Tensor (N, 3, H, W) and list of resized images
    """
    resized = []
    for crop in crops:
        # Resize
        resized_crop = cv2.resize(crop, (target_size[1], target_size[0]))
        resized.append(resized_crop)
    
    # Convert BGR -> RGB
    batch_list = []
    for crop in resized:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        # Normalize: ImageNet mean/std
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        # (H, W, 3) -> (3, H, W)
        rgb = np.transpose(rgb, (2, 0, 1))
        batch_list.append(rgb)
    
    batch_tensor = torch.tensor(np.stack(batch_list), dtype=torch.float32)
    
    if verbose:
        print(f"[Preprocess] Batch shape: {batch_tensor.shape}")
    
    return batch_tensor, resized


def extract_osnet_features(crops: List[np.ndarray],
                          model: Any,
                          device: str = 'cuda',
                          model_type: str = 'onnx',
                          batch_size: int = 16,
                          verbose: bool = False) -> np.ndarray:
    """
    Extract OSNet embeddings from crops (ONNX or PyTorch).
    
    IMPORTANT for ONNX: The model has a fixed batch size of 16.
    This function will pad crops to 16 if needed.
    
    Args:
        crops: List of crop images (H, W, 3) BGR
        model: Loaded OSNet model (onnx session or pytorch model)
        device: 'cuda' or 'cpu'
        model_type: 'onnx' or 'pytorch'
        batch_size: Batch size for inference (ONNX MUST be 16, default: 16)
        verbose: Print extraction details
    
    Returns:
        Features (N, 256) array
    """
    # Preprocess
    batch_tensor, _ = preprocess_crops(crops, target_size=(256, 128), verbose=verbose)
    
    # For ONNX: Must use fixed batch size of 16 (pad if needed)
    if model_type == 'onnx':
        if len(batch_tensor) != 16:
            # Pad with zeros to match ONNX model batch size
            padding = torch.zeros((16 - len(batch_tensor), 3, 256, 128), dtype=torch.float32)
            batch_tensor = torch.cat([batch_tensor, padding], dim=0)
            original_len = len(crops)
        else:
            original_len = len(crops)
    
    # Forward pass
    features_list = []
    
    if model_type == 'onnx':
        # ONNX Runtime inference - must pass all 16 at once
        batch = batch_tensor.numpy()  # Convert to numpy for ONNX
        input_name = model.get_inputs()[0].name
        feat = model.run(None, {input_name: batch})
        features_list.append(feat[0][:original_len])  # Only keep features for original crops
    else:
        # PyTorch inference - can use variable batch size
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        device_obj = torch.device(device)
        with torch.no_grad():
            for i in range(0, len(batch_tensor), batch_size):
                batch = batch_tensor[i:i+batch_size].to(device_obj)
                feat = model(batch)
                features_list.append(feat.cpu().numpy())
    
    features = np.vstack(features_list)
    
    if verbose:
        print(f"[Extract Features] Output shape: {features.shape}, "
              f"Min: {features.min():.3f}, Max: {features.max():.3f}")
    
    return features


def compute_embedding(features: np.ndarray, 
                     verbose: bool = False) -> np.ndarray:
    """
    Create single embedding from multiple feature vectors.
    
    Algorithm:
    1. Average across samples: (N, 256) -> (256,)
    2. L2 normalize: embedding / ||embedding||_2
    3. Return unit vector
    
    Args:
        features: (N, 256) array of features
        verbose: Print details
    
    Returns:
        (256,) unit embedding vector
    """
    # Average
    mean_feat = features.mean(axis=0)
    
    # L2 normalize
    norm = np.linalg.norm(mean_feat)
    if norm > 0:
        embedding = mean_feat / norm
    else:
        embedding = mean_feat
    
    if verbose:
        print(f"[Compute Embedding] Norm before: {norm:.4f}, "
              f"Norm after: {np.linalg.norm(embedding):.4f}")
    
    return embedding


def compute_similarity_matrix_from_features(features_dict: Dict[int, np.ndarray],
                                           person_ids: Optional[List[int]] = None,
                                           threshold: float = 0.70,
                                           verbose: bool = False) -> Dict[str, Any]:
    """
    Compute similarity matrix from per-crop features (NO AVERAGING).
    
    NEW APPROACH: Instead of averaging crops into 1 embedding per person,
    keep all crop features and compute similarities between sets of features.
    
    Similarity between Person A and Person B = Mean similarity between their crops
    
    Args:
        features_dict: Dict[person_id: (num_crops, 256) feature array]
        person_ids: Optional list of person IDs (for ordering). If None, use sorted keys.
        threshold: Highlight pairs above this
        verbose: Print details
    
    Returns:
        {
            'similarity_matrix': (N, N) array of mean similarities,
            'person_ids': [sorted person IDs],
            'high_similarity_pairs': [[id1, id2, score], ...],
            'timestamp': ISO timestamp
        }
    """
    if person_ids is None:
        person_ids = sorted(features_dict.keys())
    
    n_persons = len(person_ids)
    similarity_matrix = np.zeros((n_persons, n_persons))
    
    # For each pair of persons, compute mean similarity between their crops
    for i, pid1 in enumerate(person_ids):
        for j, pid2 in enumerate(person_ids):
            if i == j:
                # Self-similarity should be high
                similarity_matrix[i, j] = 1.0
            else:
                # Compute similarities between all crops of person i and person j
                features1 = features_dict[pid1]  # (num_crops1, 256)
                features2 = features_dict[pid2]  # (num_crops2, 256)
                
                # Normalize each feature vector
                features1_norm = features1 / (np.linalg.norm(features1, axis=1, keepdims=True) + 1e-8)
                features2_norm = features2 / (np.linalg.norm(features2, axis=1, keepdims=True) + 1e-8)
                
                # Compute pairwise similarities: (num_crops1, num_crops2)
                pairwise_sims = np.dot(features1_norm, features2_norm.T)
                
                # Use MEAN of all pairwise similarities
                mean_sim = pairwise_sims.mean()
                similarity_matrix[i, j] = mean_sim
    
    # Find high-similarity pairs
    high_pairs = []
    for i in range(len(person_ids)):
        for j in range(i+1, len(person_ids)):
            score = float(similarity_matrix[i, j])
            if score > threshold:
                high_pairs.append([int(person_ids[i]), int(person_ids[j]), score])
    
    # Sort by similarity descending
    high_pairs.sort(key=lambda x: x[2], reverse=True)
    
    if verbose:
        print(f"[Similarity Matrix] Shape: {similarity_matrix.shape}")
        print(f"[Similarity Matrix] Diagonal (should be 1.0): {np.diag(similarity_matrix)}")
        print(f"[Similarity Matrix] High-similarity pairs (>{threshold}):")
        for p1, p2, score in high_pairs:
            print(f"  Person {p1} & {p2}: {score:.3f}")
    
    return {
        'similarity_matrix': similarity_matrix.astype(np.float32),
        'person_ids': person_ids,
        'high_similarity_pairs': high_pairs,
        'timestamp': __import__('datetime').datetime.now(
            __import__('datetime').timezone.utc).isoformat()
    }


def compute_similarity_matrix(embeddings_dict: Dict[int, np.ndarray],
                             threshold: float = 0.70,
                             verbose: bool = False) -> Dict[str, Any]:
    """
    Compute NxN cosine similarity matrix.
    
    Args:
        embeddings_dict: Dict[person_id: (256,) embedding]
        threshold: Highlight pairs above this similarity (default: 0.70)
        verbose: Print details
    
    Returns:
        {
            'similarity_matrix': (N, N) array,
            'person_ids': [list of person IDs in order],
            'high_similarity_pairs': [[id1, id2, score], ...],
            'timestamp': ISO timestamp
        }
    """
    person_ids = sorted(embeddings_dict.keys())
    embeddings = np.array([embeddings_dict[pid] for pid in person_ids])
    
    # Compute cosine similarity: already normalized, so just dot product
    similarity_matrix = np.dot(embeddings, embeddings.T)
    
    # Find high-similarity pairs
    high_pairs = []
    for i in range(len(person_ids)):
        for j in range(i+1, len(person_ids)):
            score = float(similarity_matrix[i, j])
            if score > threshold:
                high_pairs.append([int(person_ids[i]), int(person_ids[j]), score])
    
    # Sort by similarity descending
    high_pairs.sort(key=lambda x: x[2], reverse=True)
    
    if verbose:
        print(f"[Similarity Matrix] Shape: {similarity_matrix.shape}")
        print(f"[Similarity Matrix] Diagonal (should be ~1.0): {np.diag(similarity_matrix)}")
        print(f"[Similarity Matrix] High-similarity pairs (>{threshold}):")
        for p1, p2, score in high_pairs:
            print(f"  Person {p1} & {p2}: {score:.3f}")
    
    return {
        'similarity_matrix': similarity_matrix.astype(np.float32),
        'person_ids': person_ids,
        'high_similarity_pairs': high_pairs,
        'timestamp': __import__('datetime').datetime.now(
            __import__('datetime').timezone.utc).isoformat()
    }


def create_similarity_matrix(buckets: Dict[int, List[np.ndarray]],
                            osnet_model_path: Optional[str] = None,
                            osnet_fallback_model_path: Optional[str] = None,
                            device: str = 'cuda',
                            num_best_crops: int = 16,
                            similarity_threshold: float = 0.70,
                            verbose: bool = False) -> Dict[str, Any]:
    """
    Main function: Extract OSNet embeddings and compute similarity matrix.
    
    This is the entry point for Stage 4 clustering. Handles both ONNX and PyTorch models with fallback support.
    
    Args:
        buckets: Dict[person_id: [crop1, crop2, ..., crop50]]
        osnet_model_path: Path to OSNet model (.onnx or .pt/.pth) - primary
        osnet_fallback_model_path: Path to fallback model if primary not found
        device: 'cuda' or 'cpu'
        num_best_crops: Number of crops to use per person (default: 16, MUST match ONNX model batch size)
        similarity_threshold: Highlight pairs above this (default: 0.70)
        verbose: Print progress
    
    Returns:
        {
            'similarity_matrix': (N, N) array,
            'embeddings': Dict[person_id: (256,) embedding],
            'person_ids': [sorted person IDs],
            'high_similarity_pairs': [[id1, id2, score], ...],
            'timing': {'load_model': ..., 'select_crops': ..., 'extract': ..., 'similarity': ...},
            'timestamp': ISO timestamp
        }
    """
    start_total = time.time()
    
    if verbose:
        print("\n" + "="*70)
        print("Stage 4: OSNet Clustering - ReID Similarity Matrix")
        print("="*70)
    
    # Load model
    start = time.time()
    try:
        model, device_str, model_type, actual_model_path = load_osnet_model(
            osnet_model_path, 
            osnet_fallback_model_path, 
            device
        )
    except RuntimeError as e:
        print(f"[OSNet] Critical error: {e}")
        raise
    
    time_load = time.time() - start
    if verbose:
        print(f"Model loaded ({model_type}): {time_load:.2f}s")
        if model_type == 'random':
            print("  ⚠️  WARNING: Using random initialization - results unreliable!")
        else:
            print(f"  Path: {actual_model_path}")
    else:
        print(f"[OSNet] Model: {model_type} ({actual_model_path})")
    
    # Select best crops per person
    start = time.time()
    best_crops_dict = {}
    for person_id, crops in buckets.items():
        best_crops = select_best_crops(crops, num=num_best_crops, verbose=False)
        best_crops_dict[person_id] = best_crops
    time_select = time.time() - start
    if verbose:
        print(f"Crop selection: {time_select:.2f}s")
        total_crops = sum(len(c) for c in best_crops_dict.values())
        print(f"  Selected {total_crops} crops from {sum(len(c) for c in buckets.values())} total")
    
    # Extract features (NO AVERAGING - keep all per-crop features)
    start = time.time()
    all_features_dict = {}  # person_id: (num_crops, 256) array
    for person_id, crops in best_crops_dict.items():
        # Extract features for each crop individually
        features = extract_osnet_features(
            crops, 
            model, 
            device_str, 
            model_type, 
            batch_size=8,  # Use batch_size=8 for PyTorch x1_0
            verbose=False
        )
        # Store all features (no averaging!)
        # features shape: (num_crops, 256)
        all_features_dict[person_id] = features
        if verbose:
            print(f"  Person {person_id}: {features.shape} array stored")
    time_extract = time.time() - start
    if verbose:
        print(f"Feature extraction ({model_type}): {time_extract:.2f}s")
        total_features = sum(f.shape[0] for f in all_features_dict.values())
        print(f"  Total features stored: {total_features} (no averaging)")
    else:
        # Always log feature count for debugging
        total_features = sum(f.shape[0] for f in all_features_dict.values())
        total_dims = sum(f.shape[1] if f.ndim > 1 else 1 for f in all_features_dict.values())
        print(f"✓ Per-crop features stored: {len(all_features_dict)} persons, {total_features} total features")
    
    # Compute similarity using per-crop features (not averaged embeddings)
    start = time.time()
    similarity_result = compute_similarity_matrix_from_features(
        all_features_dict,
        person_ids=sorted(all_features_dict.keys()),
        threshold=similarity_threshold,
        verbose=verbose
    )
    time_similarity = time.time() - start
    if verbose:
        print(f"Similarity computation: {time_similarity:.2f}s")
    
    time_total = time.time() - start_total
    
    if verbose:
        print(f"\nTotal time: {time_total:.2f}s")
        print("="*70 + "\n")
    
    # Return all results
    return {
        'similarity_matrix': similarity_result['similarity_matrix'],
        'all_features': all_features_dict,  # Per-crop features (for clustering)
        'person_ids': similarity_result['person_ids'],
        'high_similarity_pairs': similarity_result['high_similarity_pairs'],
        'model_type': model_type,  # Track which backend was used
        'timing': {
            'load_model': time_load,
            'select_crops': time_select,
            'extract_features': time_extract,
            'similarity': time_similarity,
            'total': time_total
        },
        'timestamp': similarity_result['timestamp']
    }


def save_similarity_results(results: Dict[str, Any],
                           output_dir: Path,
                           verbose: bool = False) -> None:
    """
    Save similarity matrix and per-crop features to disk.
    
    NEW APPROACH: Instead of saving averaged embeddings, save all per-crop features
    for later clustering and analysis.
    
    Creates:
    - similarity_matrix.json (human-readable with metadata and matrix)
    - all_features.json (human-readable with person_ids and crop counts)
    
    Args:
        results: Output from create_similarity_matrix()
        output_dir: Directory to save files
        verbose: Print details
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save similarity matrix as JSON (human-readable, used by HTML viewer)
    sim_json = {
        'matrix': results['similarity_matrix'].tolist(),
        'person_ids': results['person_ids'],
        'high_similarity_pairs': results['high_similarity_pairs'],
        'model_type': results.get('model_type', 'unknown'),  # Track which backend
        'similarity_threshold': 0.70,
        'approach': 'per-crop features (no averaging)',  # NEW: track approach
        'timestamp': results['timestamp']
    }
    with open(output_dir / 'similarity_matrix.json', 'w') as f:
        json.dump(sim_json, f, indent=2)
    
    # Save all per-crop features
    all_features = results.get('all_features', {})
    
    # Create a flattened version with person ID tracking
    feature_info = {
        'person_ids': results['person_ids'],
        'feature_dimension': 256,
        'num_crops_per_person': {},
        'model': results.get('model_type', 'unknown'),
        'approach': 'per-crop features (all 16 crops kept)',
        'timestamp': results['timestamp']
    }
    
    # For each person, save their features
    all_features_json = {}
    for pid in results['person_ids']:
        if pid in all_features:
            features_array = all_features[pid]  # (num_crops, 256)
            feature_info['num_crops_per_person'][str(pid)] = features_array.shape[0]
            # Save as list for JSON
            all_features_json[str(pid)] = features_array.tolist()
    
    # Save features as JSON
    emb_json = {
        **feature_info,
        'all_features': all_features_json
    }
    with open(output_dir / 'all_features.json', 'w') as f:
        json.dump(emb_json, f, indent=2)
    
    if verbose:
        print(f"[Save Results] Saved to {output_dir}:")
        print(f"  - similarity_matrix.json (used by HTML viewer)")
        print(f"  - all_features.json (per-crop features, no averaging)")
        for pid in results['person_ids']:
            if pid in all_features:
                n_crops = all_features[pid].shape[0]
                print(f"    Person {pid}: {n_crops} crops × 256 dims")



if __name__ == '__main__':
    # Simple test
    print("OSNet Clustering Module - Ready for integration with Stage 4")
