#!/usr/bin/env python3
"""
Compare OSNet models using ACTUAL code from osnet_clustering.py

This extracts the exact model loading and feature extraction code
that's already proven to work in Stage 4.
"""

import json
import pickle
import argparse
import time
from pathlib import Path
import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional
import sys

# Add path for imports
sys.path.insert(0, '/content/unifiedposepipeline/det_track')

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


class OSNetModel(nn.Module):
    """Copied from osnet_clustering.py"""
    
    def __init__(self, num_classes: int = 1000, feature_dim: int = 256):
        super(OSNetModel, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
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
            layers.append(ResBlock(in_channels if i == 0 else out_channels, out_channels, stride=stride_i))
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
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


def load_osnet_model(model_path: str, device: str = 'cuda') -> Tuple[Any, str]:
    """Load OSNet model - ONNX or PyTorch"""
    
    if str(model_path).endswith('.onnx'):
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime not available")
        print(f"Loading ONNX model: {model_path}")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        session = ort.InferenceSession(str(model_path), providers=providers)
        print(f"  Providers: {session.get_providers()}")
        return session, 'onnx'
    
    elif str(model_path).endswith(('.pt', '.pth')):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        print(f"Loading PyTorch model: {model_path}")
        device_obj = torch.device(device)
        model = OSNetModel(feature_dim=256)
        state_dict = torch.load(model_path, map_location=device_obj)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.to(device_obj)
        print(f"  Device: {device}")
        return model, 'pytorch'
    
    else:
        raise ValueError(f"Unsupported model format: {model_path}")


def preprocess_crops(crops: List[np.ndarray], 
                    target_size: Tuple[int, int] = (256, 128),
                    verbose: bool = False) -> Tuple[torch.Tensor, List[np.ndarray]]:
    """Copied from osnet_clustering.py"""
    resized = []
    for crop in crops:
        resized_crop = cv2.resize(crop, (target_size[1], target_size[0]))
        resized.append(resized_crop)
    
    batch_list = []
    for crop in resized:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
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
    """Copied from osnet_clustering.py"""
    
    batch_tensor, _ = preprocess_crops(crops, target_size=(256, 128), verbose=verbose)
    
    # For ONNX: Must use fixed batch size of 16 (pad if needed)
    if model_type == 'onnx':
        if len(batch_tensor) != 16:
            padding = torch.zeros((16 - len(batch_tensor), 3, 256, 128), dtype=torch.float32)
            batch_tensor = torch.cat([batch_tensor, padding], dim=0)
            original_len = len(crops)
        else:
            original_len = len(crops)
    
    features_list = []
    
    if model_type == 'onnx':
        batch = batch_tensor.numpy()
        input_name = model.get_inputs()[0].name
        feat = model.run(None, {input_name: batch})
        features_list.append(feat[0][:original_len])
    else:
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


def load_crops_from_pkl(pkl_path: str, top_n: int = 16) -> Dict:
    """Load crops from final_crops.pkl"""
    print(f"Loading crops from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        final_crops = pickle.load(f)
    
    person_ids = final_crops.get('person_ids', [])
    crops_by_person = final_crops.get('crops', {})
    
    print(f"Found {len(person_ids)} persons")
    
    crops_dict = {}
    for person_id in person_ids:
        if person_id in crops_by_person:
            crop_list = crops_by_person[person_id]
            top_crops = crop_list[:min(top_n, len(crop_list))]
            crops_dict[f'person_{person_id}'] = top_crops
            print(f"  Person {person_id}: {len(top_crops)} crops")
    
    print(f"Total persons loaded: {len(crops_dict)}")
    return crops_dict


def compute_similarity_matrix(features_dict: Dict) -> Tuple[np.ndarray, List[str]]:
    """Compute pairwise cosine similarity"""
    person_ids = sorted(features_dict.keys())
    n_persons = len(person_ids)
    
    person_avg_features = []
    for person_id in person_ids:
        avg_feature = features_dict[person_id].mean(axis=0)
        person_avg_features.append(avg_feature / (np.linalg.norm(avg_feature) + 1e-10))
    
    similarity_matrix = np.zeros((n_persons, n_persons))
    for i in range(n_persons):
        for j in range(n_persons):
            similarity_matrix[i, j] = np.dot(person_avg_features[i], person_avg_features[j])
    
    return similarity_matrix, person_ids


def main():
    parser = argparse.ArgumentParser(description='Compare OSNet models using pipeline code')
    parser.add_argument('--crops', type=str, required=True, help='Path to final_crops.pkl')
    parser.add_argument('--model1', type=str, required=True, help='Path to first model (.onnx or .pt)')
    parser.add_argument('--model2', type=str, required=True, help='Path to second model (.onnx or .pt)')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--top-n', type=int, default=16, help='Top-N crops per person')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load crops
    crops_dict = load_crops_from_pkl(args.crops, top_n=args.top_n)
    
    results = {}
    
    # Model 1
    print(f"\n{'='*70}")
    try:
        model1, type1 = load_osnet_model(args.model1, device=args.device)
        print(f"Extracting features with model 1...")
        start = time.time()
        features1 = {}
        for person_id, crops_list in crops_dict.items():
            feats = extract_osnet_features(crops_list, model1, device=args.device, model_type=type1, verbose=False)
            features1[person_id] = feats
        elapsed1 = time.time() - start
        
        sim1, persons = compute_similarity_matrix(features1)
        results['model1'] = {
            'path': str(args.model1),
            'type': type1,
            'extraction_time': elapsed1,
            'mean_similarity': float(sim1.mean()),
            'std_similarity': float(sim1.std()),
            'similarity_matrix': sim1.tolist(),
            'person_ids': persons
        }
        print(f"✓ Model 1 complete in {elapsed1:.2f}s")
        print(f"  Mean similarity: {sim1.mean():.4f} ± {sim1.std():.4f}")
    except Exception as e:
        print(f"❌ Model 1 failed: {e}")
    
    # Model 2
    print(f"\n{'='*70}")
    try:
        model2, type2 = load_osnet_model(args.model2, device=args.device)
        print(f"Extracting features with model 2...")
        start = time.time()
        features2 = {}
        for person_id, crops_list in crops_dict.items():
            feats = extract_osnet_features(crops_list, model2, device=args.device, model_type=type2, verbose=False)
            features2[person_id] = feats
        elapsed2 = time.time() - start
        
        sim2, persons = compute_similarity_matrix(features2)
        results['model2'] = {
            'path': str(args.model2),
            'type': type2,
            'extraction_time': elapsed2,
            'mean_similarity': float(sim2.mean()),
            'std_similarity': float(sim2.std()),
            'similarity_matrix': sim2.tolist(),
            'person_ids': persons
        }
        print(f"✓ Model 2 complete in {elapsed2:.2f}s")
        print(f"  Mean similarity: {sim2.mean():.4f} ± {sim2.std():.4f}")
    except Exception as e:
        print(f"❌ Model 2 failed: {e}")
    
    # Save results
    if results:
        output_file = output_dir / 'comparison.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n{'='*70}")
        print(f"✓ Saved to {output_file}")
        
        if len(results) == 2:
            print(f"\nComparison:")
            print(f"  Model 1 mean similarity: {results['model1']['mean_similarity']:.4f}")
            print(f"  Model 2 mean similarity: {results['model2']['mean_similarity']:.4f}")
            diff = results['model2']['mean_similarity'] - results['model1']['mean_similarity']
            print(f"  Difference: {diff:+.4f}")


if __name__ == '__main__':
    main()
