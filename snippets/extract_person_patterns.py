#!/usr/bin/env python3
"""
Extract natural groupings from features.
Shows which persons have similar features (close pairs) vs which are unique/outliers.
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


# Main
crops_dict = load_crops_from_pkl('/content/unifiedposepipeline/demo_data/outputs/kohli_nets/final_crops.pkl', top_n=16)
print(f"Loaded {len(crops_dict)} persons\n")

model, model_type = load_osnet_model('/content/unifiedposepipeline/models/reid/osnet_x0_25_msmt17.onnx', device='cuda')

print("Extracting features per person...")
person_avg_features = {}
for person_id, crops_list in sorted(crops_dict.items()):
    feats = extract_osnet_features(crops_list, model, device='cuda', model_type=model_type)
    avg_feat = feats.mean(axis=0)
    person_avg_features[person_id] = avg_feat / (np.linalg.norm(avg_feat) + 1e-10)

sorted_persons = sorted(person_avg_features.keys())
print(f"\n{'='*70}")
print("PERSON-TO-PERSON SIMILARITY MATRIX")
print(f"{'='*70}\n")

# Print header
print(f"{'':12}", end='')
for p in sorted_persons:
    print(f"{p:>10}", end='')
print()
print("-" * (12 + len(sorted_persons) * 10))

# Print matrix
for p1 in sorted_persons:
    print(f"{p1:12}", end='')
    for p2 in sorted_persons:
        sim = np.dot(person_avg_features[p1], person_avg_features[p2])
        print(f"{sim:10.3f}", end='')
    print()

# Extract natural groupings
print(f"\n{'='*70}")
print("NATURAL GROUPINGS (persons with similar features)")
print(f"{'='*70}\n")

cross_person_sims = []
for i, p1 in enumerate(sorted_persons):
    for p2 in sorted_persons[i+1:]:
        sim = np.dot(person_avg_features[p1], person_avg_features[p2])
        cross_person_sims.append((sim, p1, p2))

cross_person_sims.sort(reverse=True)

print("Most similar person pairs (closest/most alike):")
for i, (sim, p1, p2) in enumerate(cross_person_sims[:10]):
    print(f"{i+1}. {p1} <-> {p2}: similarity = {sim:.4f}")

print("\nLeast similar person pairs (most different/outliers):")
for i, (sim, p1, p2) in enumerate(cross_person_sims[-10:]):
    print(f"{i+1}. {p1} <-> {p2}: similarity = {sim:.4f}")

# Group analysis
print(f"\n{'='*70}")
print("PATTERN ANALYSIS")
print(f"{'='*70}\n")

avg_sims = []
for p1 in sorted_persons:
    sims_to_others = []
    for p2 in sorted_persons:
        if p1 != p2:
            sim = np.dot(person_avg_features[p1], person_avg_features[p2])
            sims_to_others.append(sim)
    avg_sim = np.mean(sims_to_others)
    avg_sims.append((avg_sim, p1, sims_to_others))

avg_sims.sort(reverse=True)

print("Average similarity of each person to others (descending):")
for avg_sim, person, sims in avg_sims:
    min_sim = min(sims)
    max_sim = max(sims)
    print(f"{person}: avg={avg_sim:.4f} (range: {min_sim:.4f} - {max_sim:.4f})")

print(f"\nInterpretation:")
print(f"- HIGH average = person has GENERIC features (similar to everyone)")
print(f"- LOW average = person has UNIQUE features (different from everyone)")
print(f"- Use this to identify which persons are confusable\n")
