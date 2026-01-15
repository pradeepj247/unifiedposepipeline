#!/usr/bin/env python3
"""
Analyze hierarchical clustering dendrogram to find natural person groupings.
Shows which persons are closest to each other (merge first in the tree).
"""

import json
import pickle
import argparse
import time
from pathlib import Path
import numpy as np
import cv2
from typing import Dict, List, Tuple, Any
import sys
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform

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
    return batch_tensor, resized


def extract_osnet_features(crops: List[np.ndarray], model, device: str = 'cuda', model_type: str = 'onnx', batch_size: int = 16):
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


def main():
    parser = argparse.ArgumentParser(description='Analyze dendrogram to find natural groupings')
    parser.add_argument('--crops', type=str, required=True, help='Path to final_crops.pkl')
    parser.add_argument('--model', type=str, required=True, help='Path to OSNet model (.onnx)')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--top-n', type=int, default=16, help='Top-N crops per person')
    
    args = parser.parse_args()
    
    print("Loading crops...")
    crops_dict = load_crops_from_pkl(args.crops, top_n=args.top_n)
    
    print("Loading model...")
    model, model_type = load_osnet_model(args.model, device=args.device)
    
    print("Extracting features...")
    all_features = []
    feature_to_person = []
    person_avg_features = {}
    
    for person_id, crops_list in sorted(crops_dict.items()):
        feats = extract_osnet_features(crops_list, model, device=args.device, model_type=model_type)
        for feat_idx in range(len(feats)):
            all_features.append(feats[feat_idx])
            feature_to_person.append(person_id)
        
        avg_feat = feats.mean(axis=0)
        person_avg_features[person_id] = avg_feat / (np.linalg.norm(avg_feat) + 1e-10)
    
    all_features = np.array(all_features)
    features_norm = all_features / (np.linalg.norm(all_features, axis=1, keepdims=True) + 1e-10)
    
    print(f"\nExtracted {len(all_features)} features from {len(crops_dict)} persons\n")
    
    # Compute distance matrix and clustering
    distances = 1.0 - np.dot(features_norm, features_norm.T)
    distances = np.maximum(distances, 0)
    np.fill_diagonal(distances, 0)
    
    condensed_dist = squareform(distances)
    Z = linkage(condensed_dist, method='ward')
    
    # Get person IDs in sorted order (for indexing into dendrogram)
    sorted_persons = sorted(crops_dict.keys())
    person_indices = {person: idx for idx, person in enumerate(sorted_persons)}
    
    print("="*70)
    print("NATURAL GROUPINGS (sorted by merge distance - lower = more similar):")
    print("="*70)
    
    # Extract person-level merges from the full linkage matrix
    # Z format: [idx1, idx2, distance, sample_count]
    # Indices < n_features refer to original features
    # Indices >= n_features refer to clusters
    
    n_features = len(all_features)
    
    # Build a mapping from feature index to person
    feature_person_map = {}
    for feat_idx, person_id in enumerate(feature_to_person):
        feature_person_map[feat_idx] = person_id
    
    person_merges = []
    
    # Track which features belong to which clusters
    cluster_members = {}
    for i in range(n_features):
        cluster_members[i] = {feature_person_map[i]}
    
    # Process linkage matrix
    for merge_idx, (idx1, idx2, distance, count) in enumerate(Z):
        idx1, idx2 = int(idx1), int(idx2)
        
        # Get members of each cluster
        members1 = cluster_members.get(idx1, set())
        members2 = cluster_members.get(idx2, set())
        
        # Create new cluster
        new_cluster_idx = n_features + merge_idx
        cluster_members[new_cluster_idx] = members1 | members2
        
        # Check if this is a person-to-person (or person-cluster-to-person) merge
        persons_in_new = list(cluster_members[new_cluster_idx])
        
        if len(members1) > 0 and len(members2) > 0:
            persons1 = sorted(list(members1))
            persons2 = sorted(list(members2))
            
            # Only record merges that combine different persons (or small clusters)
            if len(persons1) <= 3 or len(persons2) <= 3:  # Include small outlier groups
                merge_info = {
                    'distance': distance,
                    'group1': persons1,
                    'group2': persons2,
                    'new_group': persons_in_new
                }
                person_merges.append(merge_info)
    
    # Sort by distance
    person_merges.sort(key=lambda x: x['distance'])
    
    # Print first 15 merges (most similar)
    print("\nClosest person pairs/groups:")
    for i, merge in enumerate(person_merges[:15]):
        print(f"\n{i+1}. Distance: {merge['distance']:.4f}")
        if len(merge['group1']) == 1 and len(merge['group2']) == 1:
            print(f"   {merge['group1'][0]} <-> {merge['group2'][0]}")
        else:
            print(f"   {merge['group1']} <-> {merge['group2']}")
            print(f"   Merged group: {merge['new_group']}")
    
    # Summary: person-to-person closest pairs (DIFFERENT persons)
    print(f"\n{'='*70}")
    print("SUMMARY - Closest DIFFERENT person pairs:")
    print(f"{'='*70}")
    
    cross_person_merges = [m for m in person_merges 
                          if len(m['group1']) == 1 and len(m['group2']) == 1 
                          and m['group1'][0] != m['group2'][0]]
    
    if cross_person_merges:
        for i, merge in enumerate(cross_person_merges[:15]):
            print(f"{i+1}. {merge['group1'][0]} <-> {merge['group2'][0]} (distance: {merge['distance']:.4f})")
    else:
        print("No cross-person merges found. All persons are fully separated.")
        print("\nShowing first merges involving person clusters:")
        for i, merge in enumerate(person_merges[:15]):
            g1_str = ', '.join(merge['group1']) if len(merge['group1']) <= 2 else f"{len(merge['group1'])} persons"
            g2_str = ', '.join(merge['group2']) if len(merge['group2']) <= 2 else f"{len(merge['group2'])} persons"
            print(f"{i+1}. [{g1_str}] <-> [{g2_str}] (distance: {merge['distance']:.4f})")


if __name__ == '__main__':
    main()
