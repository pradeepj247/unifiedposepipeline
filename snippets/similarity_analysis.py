#!/usr/bin/env python3
"""
Analyze similarity ranges to find optimal threshold for tracklet merging.

Shows:
1. All pair similarities with statistics
2. Distribution across threshold bands
3. Recommendations based on data
"""

import pickle
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
    return torch.tensor(np.stack(batch_list), dtype=torch.float32)


def extract_osnet_features(crops: List[np.ndarray], model, device: str = 'cuda', model_type: str = 'onnx'):
    if len(crops) == 0:
        return np.array([]).reshape(0, 256)
    
    batch_tensor = preprocess_crops(crops, target_size=(256, 128))
    
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


def load_canonical_persons(npz_path: str) -> Dict:
    data = np.load(npz_path, allow_pickle=True)
    persons_list = data['persons']
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
    return person_info


def load_crops_from_pkl(pkl_path: str, top_n: int = 16) -> Dict:
    with open(pkl_path, 'rb') as f:
        final_crops = pickle.load(f)
    person_ids = final_crops.get('person_ids', [])
    crops_by_person = final_crops.get('crops', {})
    crops_dict = {}
    for person_id in person_ids:
        if person_id in crops_by_person:
            crop_list = crops_by_person[person_id]
            top_crops = crop_list[:min(top_n, len(crop_list))]
            crops_dict[person_id] = top_crops
    return crops_dict


# Main execution
print("="*70)
print("SIMILARITY THRESHOLD ANALYSIS")
print("="*70)

# Load data
print("\nLoading data...")
crops_dict = load_crops_from_pkl('/content/unifiedposepipeline/demo_data/outputs/kohli_nets/final_crops.pkl', top_n=16)
all_person_info = load_canonical_persons('/content/unifiedposepipeline/demo_data/outputs/kohli_nets/canonical_persons.npz')
person_info = {pid: all_person_info[pid] for pid in crops_dict.keys() if pid in all_person_info}
sorted_persons = sorted(person_info.keys())

print(f"✓ {len(sorted_persons)} persons loaded")

# Load model and extract features
print("Loading OSNet model...")
model, model_type = load_osnet_model('/content/unifiedposepipeline/models/reid/osnet_x0_25_msmt17.onnx', device='cuda')

print("Extracting features...")
person_features = {}
for person_id in sorted_persons:
    crops = crops_dict[person_id]
    features = extract_osnet_features(crops, model, device='cuda', model_type=model_type)
    avg_feature = features.mean(axis=0)
    person_features[person_id] = avg_feature

# Compute all pairwise similarities
print("Computing similarities...")
all_similarities = []
pair_info = []

for i, id1 in enumerate(sorted_persons):
    for id2 in sorted_persons[i+1:]:
        feat1 = person_features[id1]
        feat2 = person_features[id2]
        
        feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-10)
        feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-10)
        
        similarity = float(np.dot(feat1_norm, feat2_norm))
        all_similarities.append(similarity)
        
        info1 = person_info[id1]
        info2 = person_info[id2]
        
        # Check temporal overlap
        overlap = not (info1['max_frame'] < info2['min_frame'] or info2['max_frame'] < info1['min_frame'])
        
        pair_info.append({
            'id1': id1,
            'id2': id2,
            'similarity': similarity,
            'overlap': overlap,
            'frame_range1': (info1['min_frame'], info1['max_frame']),
            'frame_range2': (info2['min_frame'], info2['max_frame'])
        })

all_similarities = np.array(all_similarities)

# Statistics
print("\n" + "="*70)
print("OVERALL STATISTICS:")
print("="*70)
print(f"Total pairs: {len(all_similarities)}")
print(f"Min similarity: {all_similarities.min():.4f}")
print(f"Max similarity: {all_similarities.max():.4f}")
print(f"Mean similarity: {all_similarities.mean():.4f}")
print(f"Std similarity: {all_similarities.std():.4f}")
print(f"Median similarity: {np.median(all_similarities):.4f}")

# Distribution by threshold bands
print("\n" + "="*70)
print("DISTRIBUTION BY THRESHOLD BANDS:")
print("="*70)

thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
for i in range(len(thresholds) - 1):
    t1, t2 = thresholds[i], thresholds[i+1]
    count = np.sum((all_similarities >= t1) & (all_similarities < t2))
    pct = 100 * count / len(all_similarities)
    print(f"  {t1:.2f} - {t2:.2f}: {count:2d} pairs ({pct:5.1f}%)")

# Top pairs by similarity
print("\n" + "="*70)
print("TOP 15 HIGHEST SIMILARITY PAIRS (candidates for merging):")
print("="*70)

sorted_pairs = sorted(pair_info, key=lambda x: x['similarity'], reverse=True)

for rank, pair in enumerate(sorted_pairs[:15], 1):
    temporal_status = "OVERLAP" if pair['overlap'] else "GAP"
    print(f"\n{rank:2d}. person_{pair['id1']} ↔ person_{pair['id2']}")
    print(f"    Similarity: {pair['similarity']:.4f}")
    print(f"    Range 1: frames {pair['frame_range1'][0]}-{pair['frame_range1'][1]}")
    print(f"    Range 2: frames {pair['frame_range2'][0]}-{pair['frame_range2'][1]}")
    print(f"    Status: {temporal_status}")

# Bottom pairs
print("\n" + "="*70)
print("BOTTOM 5 LOWEST SIMILARITY PAIRS (definitely different):")
print("="*70)

for rank, pair in enumerate(sorted_pairs[-5:], 1):
    print(f"{rank}. person_{pair['id1']} ↔ person_{pair['id2']}: {pair['similarity']:.4f}")

print("\n" + "="*70)
print("RECOMMENDATION:")
print("="*70)
print(f"""
Based on the distribution:
- Pairs with similarity ≥ 0.75: Strong candidates for same person
- Pairs with similarity 0.60-0.75: Possible same person (verify visually)
- Pairs with similarity < 0.60: Likely different people

Current known pairs from analysis:
  ✓ person_4 ↔ person_40: 0.7875 (confirmed)
  ✓ person_29 ↔ person_65: 0.6047 (confirmed)

Verify these visually to set optimal threshold.
""")

print("="*70)
