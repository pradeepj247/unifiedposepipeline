#!/usr/bin/env python3
"""
Debug why P14 and P29 aren't showing up as merge candidates in Stage 3

Data from Colab output:
- P14: tracklet_id=14, frames 103-365, 250 appearances
- P29: tracklet_id=29, frames 360-785, 425 appearances
- Gap: 360 - 365 = -5 (overlapping, should pass gap check)
- Distance: unknown (should be measured at frame 360)
"""

import numpy as np
import yaml
import re
import json
from pathlib import Path

# Load the config
config_path = '/content/unifiedposepipeline/det_track/configs/pipeline_config.yaml'

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Resolve path variables
global_vars = config.get('global', {})

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

outputs_dir = global_vars['outputs_dir']
video_file = global_vars['video_file']
current_video = Path(video_file).stem

# Construct paths
tracklets_file = f"{outputs_dir}/{current_video}/tracklets_raw.npz"
stats_file = f"{outputs_dir}/{current_video}/tracklet_stats.npz"

print(f"Loading from: {tracklets_file}")
print(f"Stats from: {stats_file}")

# Load tracklets and stats
data = np.load(tracklets_file, allow_pickle=True)
tracklets = data['tracklets']
print(f"\n✅ Loaded {len(tracklets)} tracklets")

stats_data = np.load(stats_file, allow_pickle=True)
print(f"📂 Stats file keys: {list(stats_data.files)}")
stats = stats_data['tracklet_stats']
print(f"✅ Loaded {len(stats)} stats")

# Find P14 and P29
p14_idx = None
p29_idx = None

for idx, tracklet in enumerate(tracklets):
    tid = int(tracklet['tracklet_id'])
    if tid == 14:
        p14_idx = idx
    elif tid == 29:
        p29_idx = idx

if p14_idx is None or p29_idx is None:
    print(f"\n❌ Could not find P14 (idx={p14_idx}) or P29 (idx={p29_idx})")
    exit(1)

print(f"\n✅ Found P14 at index {p14_idx}")
print(f"✅ Found P29 at index {p29_idx}")

stat_p14 = stats[p14_idx]
stat_p29 = stats[p29_idx]
tracklet_p14 = tracklets[p14_idx]
tracklet_p29 = tracklets[p29_idx]

print(f"\nP14 Stats:")
print(f"  - Frames: {stat_p14['start_frame']} - {stat_p14['end_frame']}")
print(f"  - Count: {stat_p14['count']}")
print(f"  - Mean area: {stat_p14['mean_area']:.1f}")

print(f"\nP29 Stats:")
print(f"  - Frames: {stat_p29['start_frame']} - {stat_p29['end_frame']}")
print(f"  - Count: {stat_p29['count']}")
print(f"  - Mean area: {stat_p29['mean_area']:.1f}")

# Check gap
gap = stat_p29['start_frame'] - stat_p14['end_frame']
print(f"\n📊 Gap Analysis:")
print(f"  - Gap: {gap} (P29 start {stat_p29['start_frame']} - P14 end {stat_p14['end_frame']})")

# Check temporal criterion
max_temporal_gap = config['stage3_analyze']['candidate_criteria']['max_temporal_gap']
print(f"  - Max temporal gap: {max_temporal_gap}")
print(f"  - Threshold for rejection: gap < -100 or gap > {max_temporal_gap}")
print(f"  - Would be ACCEPTED: {not (gap < -100 or gap > max_temporal_gap)}")

if gap < -100 or gap > max_temporal_gap:
    print(f"  ❌ REJECTED at temporal check!")
    exit(1)

print(f"  ✅ PASSED temporal check")

# Check spatial criterion
print(f"\n📍 Spatial Analysis:")
print(f"  - P14 frames: {tracklet_p14['frame_numbers'][:5]}...{tracklet_p14['frame_numbers'][-5:]}")
print(f"  - P29 frames: {tracklet_p29['frame_numbers'][:5]}...{tracklet_p29['frame_numbers'][-5:]}")

overlap_start = stat_p29['start_frame']  # 360
print(f"  - Overlap start: {overlap_start}")

p14_frames = tracklet_p14['frame_numbers']
p29_frames = tracklet_p29['frame_numbers']

# Check if overlap_start exists in both
in_p14 = np.isin(overlap_start, p14_frames)
in_p29 = np.isin(overlap_start, p29_frames)

print(f"  - Frame {overlap_start} in P14: {in_p14}")
print(f"  - Frame {overlap_start} in P29: {in_p29}")

if not (in_p14 and in_p29):
    print(f"  ❌ REJECTED at spatial check (frame not in both tracklets)")
    exit(1)

print(f"  ✅ PASSED spatial check (frame exists in both)")

# Get bboxes at overlap_start
p14_idx_at_overlap = np.where(p14_frames == overlap_start)[0][0]
p29_idx_at_overlap = np.where(p29_frames == overlap_start)[0][0]

p14_bbox = tracklet_p14['bboxes'][p14_idx_at_overlap]
p29_bbox = tracklet_p29['bboxes'][p29_idx_at_overlap]

print(f"\n📦 Bboxes at frame {overlap_start}:")
print(f"  - P14 bbox: {p14_bbox}")
print(f"  - P29 bbox: {p29_bbox}")

# Calculate distance
p14_center = np.array([(p14_bbox[0] + p14_bbox[2]) / 2, (p14_bbox[1] + p14_bbox[3]) / 2])
p29_center = np.array([(p29_bbox[0] + p29_bbox[2]) / 2, (p29_bbox[1] + p29_bbox[3]) / 2])
distance = np.linalg.norm(p14_center - p29_center)

print(f"  - P14 center: {p14_center}")
print(f"  - P29 center: {p29_center}")
print(f"  - Distance: {distance:.1f}")

# Check spatial criterion
max_spatial_distance = config['stage3_analyze']['candidate_criteria']['max_spatial_distance']
print(f"\n  - Max spatial distance: {max_spatial_distance}")
print(f"  - Would be ACCEPTED: {distance <= max_spatial_distance}")

if distance > max_spatial_distance:
    print(f"  ❌ REJECTED at spatial distance check!")
    exit(1)

print(f"  ✅ PASSED spatial distance check")

# Check area ratio
area_i = stat_p14['mean_area']
area_j = stat_p29['mean_area']
area_ratio = area_j / area_i if area_i > 0 else 0

area_ratio_range = config['stage3_analyze']['candidate_criteria']['area_ratio_range']
print(f"\n📏 Area Ratio Analysis:")
print(f"  - P14 mean area: {area_i:.1f}")
print(f"  - P29 mean area: {area_j:.1f}")
print(f"  - Ratio: {area_ratio:.2f}")
print(f"  - Range: {area_ratio_range}")
print(f"  - Would be ACCEPTED: {area_ratio_range[0] <= area_ratio <= area_ratio_range[1]}")

if area_ratio < area_ratio_range[0] or area_ratio > area_ratio_range[1]:
    print(f"  ❌ REJECTED at area ratio check!")
    exit(1)

print(f"  ✅ PASSED area ratio check")

print(f"\n✅✅✅ P14 and P29 SHOULD BE MERGE CANDIDATES! ✅✅✅")
print(f"\nSummary:")
print(f"  - Temporal: ✅ gap={gap}")
print(f"  - Spatial: ✅ distance={distance:.1f}")
print(f"  - Area: ✅ ratio={area_ratio:.2f}")
