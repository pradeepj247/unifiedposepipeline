#!/usr/bin/env python3
"""
Find tracklet recovery candidates using temporal non-overlap + ReID similarity.

Steps:
1. Load canonical_persons.npz to get frame ranges for each person
2. Load final_crops.pkl to get crops for each person
3. Extract OSNet features for each person (averaged)
4. Find all non-overlapping pairs
5. Compute similarity for ALL pairs
6. Report all pairs with similarity scores
7. Build graph: connect pairs with high similarity
8. Find connected components = person chains (same person, multiple detections)
"""

import pickle
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict

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
    return torch.tensor(np.stack(batch_list), dtype=torch.float32), resized


def extract_osnet_features(crops: List[np.ndarray], model, device: str = 'cuda', model_type: str = 'onnx'):
    if len(crops) == 0:
        return np.array([]).reshape(0, 256)
    
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


def load_canonical_persons(npz_path: str) -> Dict:
    """Load canonical_persons.npz and extract frame ranges"""
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
    """Load crops from final_crops.pkl"""
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


def check_temporal_overlap(frame_range1: Tuple[int, int], frame_range2: Tuple[int, int], overlap_tolerance: int = 0) -> bool:
    """Check if two frame ranges overlap beyond tolerance.
    
    overlap_tolerance: Allow this many frames of overlap (default 0 = strict non-overlap)
    """
    min1, max1 = frame_range1
    min2, max2 = frame_range2
    
    # Calculate actual overlap
    overlap_start = max(min1, min2)
    overlap_end = min(max1, max2)
    
    if overlap_end < overlap_start:
        # No overlap
        return False
    
    overlap_size = overlap_end - overlap_start + 1
    return overlap_size > overlap_tolerance


def find_non_overlapping_pairs(person_info: Dict, overlap_tolerance: int = 0) -> List[Tuple[int, int]]:
    """Find all pairs of persons with minimal temporal overlap.
    
    overlap_tolerance: Allow this many frames of overlap (default 0 = strict non-overlap)
    """
    person_ids = sorted(person_info.keys())
    pairs = []
    
    for i in range(len(person_ids)):
        for j in range(i+1, len(person_ids)):
            id1, id2 = person_ids[i], person_ids[j]
            range1 = (person_info[id1]['min_frame'], person_info[id1]['max_frame'])
            range2 = (person_info[id2]['min_frame'], person_info[id2]['max_frame'])
            
            if not check_temporal_overlap(range1, range2, overlap_tolerance=overlap_tolerance):
                pairs.append((id1, id2))
    
    return pairs


class UnionFind:
    """Union-Find (Disjoint Set Union) for connected components"""
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


# Main execution
print("="*70)
print("TRACKLET RECOVERY: Finding Person Chains")
print("="*70)

# Load crops FIRST to determine which persons to analyze
print("\n[1/5] Loading crops...")
crops_dict = load_crops_from_pkl('/content/unifiedposepipeline/demo_data/outputs/kohli_nets/final_crops.pkl', top_n=16)
print(f"✓ Loaded crops for {len(crops_dict)} persons")

# Load canonical persons for ALL, but only keep the ones with crops
print("\n[2/5] Loading canonical persons...")
all_person_info = load_canonical_persons('/content/unifiedposepipeline/demo_data/outputs/kohli_nets/canonical_persons.npz')
# Filter to only those with crops
person_info = {pid: all_person_info[pid] for pid in crops_dict.keys() if pid in all_person_info}
sorted_persons = sorted(person_info.keys())
print(f"✓ Loaded {len(sorted_persons)} canonical persons (filtered to those with crops)")
for pid in sorted_persons:
    info = person_info[pid]
    print(f"  person_{pid}: frames {info['min_frame']}-{info['max_frame']} ({info['num_frames']} frames)")

# Load OSNet model
print("\n[3/5] Loading OSNet model...")
model, model_type = load_osnet_model('/content/unifiedposepipeline/models/reid/osnet_x0_25_msmt17.onnx', device='cuda')
print(f"✓ Model loaded ({model_type})")

# Extract averaged features for each person
print("\n[4/5] Extracting OSNet features (averaged per person)...")
person_features = {}
for person_id in sorted_persons:
    if person_id in crops_dict:
        crops = crops_dict[person_id]
        features = extract_osnet_features(crops, model, device='cuda', model_type=model_type)
        avg_feature = features.mean(axis=0)
        person_features[person_id] = avg_feature
        print(f"  person_{person_id}: {len(crops)} crops → avg feature")

print(f"✓ Extracted {len(person_features)} averaged features")

# Find non-overlapping pairs (allow up to 20 frames of overlap)
print("\n[5/5] Finding candidate pairs and computing similarities...\n")
pairs = find_non_overlapping_pairs(person_info, overlap_tolerance=20)
print(f"Found {len(pairs)} candidate pairs (allowing up to 20 frames overlap)")
print("\n" + "="*70)
print("CANDIDATE PAIRS (including small overlaps):")
print("="*70)

# Compute similarities for all pairs
pair_similarities = {}
connections = []  # For building graph

for id1, id2 in pairs:
    if id1 not in person_features or id2 not in person_features:
        continue
    
    feat1 = person_features[id1]
    feat2 = person_features[id2]
    
    # Normalize
    feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-10)
    feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-10)
    
    # Cosine similarity
    similarity = float(np.dot(feat1_norm, feat2_norm))
    
    info1 = person_info[id1]
    info2 = person_info[id2]
    
    # Determine which comes first chronologically (compare start frames)
    if info1['min_frame'] <= info2['min_frame']:
        first_id, second_id = id1, id2
        first_info, second_info = info1, info2
    else:
        first_id, second_id = id2, id1
        first_info, second_info = info2, info1
    
    gap = second_info['min_frame'] - first_info['max_frame']
    
    # Determine connection status
    SIMILARITY_THRESHOLD = 0.60
    is_connected = similarity >= SIMILARITY_THRESHOLD
    status = "✓ CONNECTED" if is_connected else "✗ LOW"
    
    print(f"\n  person_{first_id} (frames {first_info['min_frame']}-{first_info['max_frame']}) → person_{second_id} (frames {second_info['min_frame']}-{second_info['max_frame']})")
    print(f"    Similarity: {similarity:.4f} | {status} | Gap: {gap} frames")
    
    pair_similarities[(id1, id2)] = similarity
    
    if is_connected:
        connections.append((id1, id2))

# Build connected components
print("\n" + "="*70)
print("CONNECTED COMPONENTS (Person Chains):")
print("="*70)

uf = UnionFind(sorted_persons)
for id1, id2 in connections:
    uf.union(id1, id2)

components = uf.get_components()
components.sort(key=lambda x: min(x))  # Sort by earliest person ID

print(f"\nFound {len(components)} person group(s):\n")

for group_idx, component in enumerate(components, 1):
    component.sort()
    
    # Compute avg similarity within group
    group_similarities = []
    for i in range(len(component)):
        for j in range(i+1, len(component)):
            id1, id2 = component[i], component[j]
            key = (min(id1, id2), max(id1, id2))
            if key in pair_similarities:
                group_similarities.append(pair_similarities[key])
    
    if group_similarities:
        avg_sim = np.mean(group_similarities)
    else:
        avg_sim = 0.0
    
    # Build timeline
    timeline_info = []
    for person_id in component:
        info = person_info[person_id]
        timeline_info.append((info['min_frame'], info['max_frame'], person_id))
    timeline_info.sort()
    
    timeline_str = " → ".join([f"person_{pid} ({min_f}-{max_f})" for min_f, max_f, pid in timeline_info])
    
    print(f"Group {group_idx}: {timeline_str}")
    if group_similarities:
        print(f"  Avg similarity: {avg_sim:.4f}")
    
    # Show gaps
    if len(component) > 1:
        gaps = []
        for i in range(len(timeline_info)-1):
            gap = timeline_info[i+1][0] - timeline_info[i][1]
            gaps.append(gap)
        gaps_str = ", ".join([str(g) for g in gaps])
        print(f"  Gaps: {gaps_str} frames")
    
    print()

print("="*70)
print(f"SUMMARY: {len(components)} person group(s) identified")
print(f"  - Persons in single groups (potential re-detections): {sum(1 for c in components if len(c) > 1)}")
print(f"  - Standalone persons: {sum(1 for c in components if len(c) == 1)}")
print("="*70)
