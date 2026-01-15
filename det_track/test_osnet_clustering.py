#!/usr/bin/env python3
"""
Unit Tests for OSNet Clustering Module

Tests individual components before full integration:
1. select_best_crops()
2. preprocess_crops()
3. extract_osnet_features()
4. compute_embedding()
5. compute_similarity_matrix()
6. Full create_similarity_matrix()
"""

import numpy as np
import cv2
from pathlib import Path
import sys

# Add det_track to path
sys.path.insert(0, str(Path(__file__).parent))

from osnet_clustering import (
    select_best_crops,
    preprocess_crops,
    extract_osnet_features,
    compute_embedding,
    compute_similarity_matrix,
    create_similarity_matrix,
    load_osnet_model
)


def create_dummy_crops(num_crops: int = 50, height: int = 100, width: int = 80) -> list:
    """Create dummy crop images for testing"""
    crops = []
    for i in range(num_crops):
        # Create random BGR image
        crop = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        crops.append(crop)
    return crops


def test_select_best_crops():
    """Test 1: Crop selection"""
    print("\n" + "="*70)
    print("TEST 1: select_best_crops()")
    print("="*70)
    
    crops = create_dummy_crops(50)
    selected = select_best_crops(crops, num=8, verbose=True)
    
    assert len(selected) == 8, f"Expected 8 crops, got {len(selected)}"
    print(f"✓ Successfully selected {len(selected)} crops from {len(crops)}")
    return True


def test_preprocess_crops():
    """Test 2: Crop preprocessing"""
    print("\n" + "="*70)
    print("TEST 2: preprocess_crops()")
    print("="*70)
    
    crops = create_dummy_crops(8)
    batch_tensor, resized_crops = preprocess_crops(crops, target_size=(256, 128), verbose=True)
    
    assert batch_tensor.shape[0] == 8, f"Expected 8 samples, got {batch_tensor.shape[0]}"
    assert batch_tensor.shape[1] == 3, f"Expected 3 channels, got {batch_tensor.shape[1]}"
    assert batch_tensor.shape[2] == 256, f"Expected height 256, got {batch_tensor.shape[2]}"
    assert batch_tensor.shape[3] == 128, f"Expected width 128, got {batch_tensor.shape[3]}"
    print(f"✓ Batch shape correct: {batch_tensor.shape}")
    return True


def test_extract_osnet_features():
    """Test 3: OSNet feature extraction"""
    print("\n" + "="*70)
    print("TEST 3: extract_osnet_features()")
    print("="*70)
    
    # Load model
    print("Loading OSNet model...")
    model, device = load_osnet_model(device='cpu')
    print(f"Model loaded on {device}")
    
    # Create dummy crops
    crops = create_dummy_crops(8)
    
    # Extract features
    features = extract_osnet_features(
        crops=crops,
        model=model,
        device='cpu',
        batch_size=8,
        verbose=True
    )
    
    assert features.shape == (8, 256), f"Expected (8, 256), got {features.shape}"
    print(f"✓ Feature shape correct: {features.shape}")
    print(f"  - Min: {features.min():.4f}, Max: {features.max():.4f}")
    print(f"  - Mean: {features.mean():.4f}, Std: {features.std():.4f}")
    return True


def test_compute_embedding():
    """Test 4: Embedding computation"""
    print("\n" + "="*70)
    print("TEST 4: compute_embedding()")
    print("="*70)
    
    # Create dummy features
    features = np.random.randn(8, 256).astype(np.float32)
    
    # Compute embedding
    embedding = compute_embedding(features, verbose=True)
    
    assert embedding.shape == (256,), f"Expected (256,), got {embedding.shape}"
    norm = np.linalg.norm(embedding)
    assert abs(norm - 1.0) < 0.01, f"Expected L2 norm ~1.0, got {norm}"
    print(f"✓ Embedding norm: {norm:.4f} (unit vector)")
    return True


def test_compute_similarity_matrix():
    """Test 5: Similarity matrix computation"""
    print("\n" + "="*70)
    print("TEST 5: compute_similarity_matrix()")
    print("="*70)
    
    # Create dummy embeddings for 5 persons
    embeddings_dict = {}
    for i in range(5):
        embeddings_dict[i] = np.random.randn(256).astype(np.float32)
        embeddings_dict[i] /= np.linalg.norm(embeddings_dict[i])  # L2 norm
    
    # Compute similarity
    result = compute_similarity_matrix(embeddings_dict, threshold=0.70, verbose=True)
    
    matrix = result['similarity_matrix']
    assert matrix.shape == (5, 5), f"Expected (5, 5), got {matrix.shape}"
    assert abs(matrix[0, 0] - 1.0) < 0.01, f"Diagonal should be ~1.0, got {matrix[0,0]}"
    
    print(f"✓ Similarity matrix shape: {matrix.shape}")
    print(f"  - Diagonal (should be 1.0): {np.diag(matrix)}")
    print(f"  - High-similarity pairs: {result['high_similarity_pairs']}")
    return True


def test_full_pipeline():
    """Test 6: Full create_similarity_matrix()"""
    print("\n" + "="*70)
    print("TEST 6: create_similarity_matrix() - Full Pipeline")
    print("="*70)
    
    # Create dummy buckets (5 persons, 50 crops each)
    buckets = {}
    for person_id in range(5):
        buckets[person_id] = create_dummy_crops(50, height=100, width=80)
    
    print(f"Created {len(buckets)} persons with {len(buckets[0])} crops each")
    
    # Run full pipeline
    result = create_similarity_matrix(
        buckets=buckets,
        osnet_model_path=None,  # Will use random model
        device='cpu',
        num_best_crops=8,
        similarity_threshold=0.70,
        verbose=True
    )
    
    # Verify outputs
    assert result['similarity_matrix'].shape == (5, 5), f"Wrong matrix shape"
    assert len(result['embeddings']) == 5, f"Wrong number of embeddings"
    assert len(result['person_ids']) == 5, f"Wrong number of person IDs"
    assert 'timing' in result, f"Missing timing info"
    
    print(f"✓ Full pipeline succeeded")
    print(f"  - Matrix shape: {result['similarity_matrix'].shape}")
    print(f"  - Embeddings: {len(result['embeddings'])}")
    print(f"  - Timing: {result['timing']}")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("OSNet Clustering Unit Tests")
    print("="*70)
    
    tests = [
        ("select_best_crops", test_select_best_crops),
        ("preprocess_crops", test_preprocess_crops),
        ("extract_osnet_features", test_extract_osnet_features),
        ("compute_embedding", test_compute_embedding),
        ("compute_similarity_matrix", test_compute_similarity_matrix),
        ("full_pipeline", test_full_pipeline),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
