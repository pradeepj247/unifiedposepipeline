#!/usr/bin/env python3
"""
Find natural groupings by varying clustering threshold.
Shows which persons cluster together at different similarity levels.
"""

import pickle
import numpy as np
import sys
from pathlib import Path

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

import cv2
from typing import List, Tuple, Any
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


class OSNetModel(nn.Module):
    def __init__(self, num_classes: int = 1000, feature_dim: int = 256):
        super(OSNetModel, self).__init__()
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


def load_osnet_model(model_path: str, device: str = 'cuda'):
    if str(model_path).endswith('.onnx'):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        session = ort.InferenceSession(str(model_path), providers=providers)
        return session, 'onnx'
    elif str(model_path).endswith(('.pt', '.pth')):
        device_obj = torch.device(device)
        model = OSNetModel(feature_dim=256)
        state_dict = torch.load(model_path, map_location=device_obj)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.to(device_obj)
        return model, 'pytorch'


def preprocess_crops(crops: List[np.ndarray], target_size: Tuple[int, int] = (256, 128)):
    resized = [cv2.resize(crop, (target_size[1], target_size[0])) for crop in crops]
    batch_list = []
    for crop in resized:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        rgb = np.transpose(rgb, (2, 0, 1))
        batch_list.append(rgb)
    return torch.tensor(np.stack(batch_list), dtype=torch.float32), resized


def extract_osnet_features(crops: List[np.ndarray], model, device: str = 'cuda', model_type: str = 'onnx'):
    batch_tensor, _ = preprocess_crops(crops, target_size=(256, 128))
    
    if model_type == 'onnx':
        if len(batch_tensor) != 16:
            padding = torch.zeros((16 - len(batch_tensor), 3, 256, 128), dtype=torch.float32)
            batch_tensor = torch.cat([batch_tensor, padding], dim=0)
            original_len = len(crops)
        else:
            original_len = len(crops)
        batch = batch_tensor.numpy()
        input_name = model.get_inputs()[0].name
        feat = model.run(None, {input_name: batch})
        return feat[0][:original_len]
    else:
        device_obj = torch.device(device)
        with torch.no_grad():
            batch = batch_tensor.to(device_obj)
            feat = model(batch)
            return feat.cpu().numpy()


def load_crops_from_pkl(pkl_path: str, top_n: int = 16):
    with open(pkl_path, 'rb') as f:
        final_crops = pickle.load(f)
    person_ids = final_crops.get('person_ids', [])
    crops_by_person = final_crops.get('crops', {})
    crops_dict = {}
    for person_id in person_ids:
        if person_id in crops_by_person:
            crop_list = crops_by_person[person_id]
            top_crops = crop_list[:min(top_n, len(crop_list))]
            crops_dict[f'person_{person_id}'] = top_crops
    return crops_dict


# Load and extract
print("Loading crops...")
crops_dict = load_crops_from_pkl('/content/unifiedposepipeline/demo_data/outputs/kohli_nets/final_crops.pkl', top_n=16)
sorted_persons = sorted(crops_dict.keys())
print(f"Loaded {len(sorted_persons)} persons: {sorted_persons}\n")

print("Loading model...")
model, model_type = load_osnet_model('/content/unifiedposepipeline/models/reid/osnet_x0_25_msmt17.onnx', device='cuda')

print("Extracting features...")
all_features = []
feature_to_person = []

for person_id in sorted_persons:
    crops_list = crops_dict[person_id]
    feats = extract_osnet_features(crops_list, model, device='cuda', model_type=model_type)
    for feat in feats:
        all_features.append(feat)
        feature_to_person.append(person_id)

all_features = np.array(all_features)
print(f"Extracted {len(all_features)} total features\n")

# Normalize
features_norm = all_features / (np.linalg.norm(all_features, axis=1, keepdims=True) + 1e-10)

# Distance matrix
distances = 1.0 - np.dot(features_norm, features_norm.T)
distances = np.maximum(distances, 0)
np.fill_diagonal(distances, 0)
condensed_dist = squareform(distances)

# Hierarchical clustering
print("Computing hierarchical clustering...\n")
Z = linkage(condensed_dist, method='ward')

# Try different thresholds
print("="*70)
print("NATURAL GROUPINGS AT DIFFERENT THRESHOLDS")
print("="*70)

thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

for threshold in thresholds:
    cluster_labels = fcluster(Z, threshold, criterion='distance')
    n_clusters = len(np.unique(cluster_labels))
    
    print(f"\n>>> Threshold: {threshold} → {n_clusters} clusters\n")
    
    # Show cluster composition
    from collections import defaultdict
    clusters = defaultdict(list)
    for feature_idx, label in enumerate(cluster_labels):
        person = feature_to_person[feature_idx]
        clusters[label].append(person)
    
    # Count persons per cluster
    for cluster_id in sorted(clusters.keys()):
        persons_in_cluster = clusters[cluster_id]
        from collections import Counter
        person_counts = Counter(persons_in_cluster)
        
        # Only show cluster composition (persons, not individual features)
        persons_str = ", ".join(sorted(set(persons_in_cluster)))
        total_features = len(persons_in_cluster)
        
        print(f"  Cluster {cluster_id}: {persons_str}")
        for person, count in sorted(person_counts.items()):
            print(f"    └─ {person}: {count} features")
        print()

print("="*70)
print("INTERPRETATION:")
print("="*70)
print("- Look for thresholds where groupings make sense visually")
print("- E.g., if persons X,Y,Z cluster together = they look similar to the model")
print("- The LOWEST threshold with natural groupings = best clusters")
print()
