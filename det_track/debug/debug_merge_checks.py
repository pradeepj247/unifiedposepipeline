#!/usr/bin/env python3
"""
Debug which specific merge check is failing between Person #2 and #53.

Tests all 6 checks:
1. Temporal order (start_frame check)
2. Temporal gap/overlap
3. Spatial proximity
4. Area ratio
5. Motion alignment
6. Jitter difference
"""

import numpy as np
import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def debug_merge_checks(output_dir):
    """Debug why Person #2 and #53 aren't merging"""
    
    output_path = Path(output_dir)
    
    print("=" * 70)
    print("🔍 DEBUGGING MERGE CHECKS: Person #2 vs #53")
    print("=" * 70)
    
    # First, load canonical persons to see which tracklets form each person
    canonical_file = output_path / 'canonical_persons.npz'
    canonical_data = np.load(canonical_file, allow_pickle=True)
    persons = canonical_data['persons']
    
    print(f"\n📊 Loading canonical persons...")
    print(f"   Found {len(persons)} canonical persons")
    
    # Find Person #2 and #53
    person_2 = None
    person_53 = None
    
    for p in persons:
        if p['person_id'] == 2:
            person_2 = p
        elif p['person_id'] == 53:
            person_53 = p
    
    if not person_2 or not person_53:
        print(f"❌ Could not find both persons in canonical_persons.npz")
        return
    
    print(f"\n👤 PERSON #2:")
    print(f"   Frames: {person_2['frame_numbers'][0]} - {person_2['frame_numbers'][-1]}")
    print(f"   Total frames: {len(person_2['frame_numbers'])}")
    print(f"   Original tracklet IDs: {person_2.get('original_tracklet_ids', 'N/A')}")
    print(f"   Num tracklets merged: {person_2.get('num_tracklets_merged', 'N/A')}")
    
    print(f"\n👤 PERSON #53:")
    print(f"   Frames: {person_53['frame_numbers'][0]} - {person_53['frame_numbers'][-1]}")
    print(f"   Total frames: {len(person_53['frame_numbers'])}")
    print(f"   Original tracklet IDs: {person_53.get('original_tracklet_ids', 'N/A')}")
    print(f"   Num tracklets merged: {person_53.get('num_tracklets_merged', 'N/A')}")
    
    # Now load tracklet stats (the raw data Stage 3b uses)
    stats_file = output_path / 'tracklet_stats.npz'
    stats_data = np.load(stats_file, allow_pickle=True)
    
    print(f"\n📦 Stats file keys: {list(stats_data.keys())}")
    
    # Get the stats array
    if 'statistics' in stats_data:
        all_stats = stats_data['statistics']
    elif 'stats' in stats_data:
        all_stats = stats_data['stats']
    else:
        print("❌ Could not find stats array")
        return
    
    print(f"   Found {len(all_stats)} tracklet statistics")
    
    # Also load the tracklets to map IDs to indices
    tracklets_file = output_path / 'tracklets_raw.npz'
    tracklets_data = np.load(tracklets_file, allow_pickle=True)
    tracklets = tracklets_data['tracklets']
    
    print(f"   Found {len(tracklets)} tracklets")
    
    # Build mapping from tracklet_id to index
    tracklet_id_to_idx = {}
    for idx, t in enumerate(tracklets):
        tracklet_id_to_idx[t['tracklet_id']] = idx
    
    # Get tracklet IDs for Person #2 and #53
    tracklet_ids_2 = person_2.get('original_tracklet_ids', [])
    tracklet_ids_53 = person_53.get('original_tracklet_ids', [])
    
    print(f"\n📋 PERSON #2 TRACKLETS (by ID):")
    indices_2 = []
    for tid in tracklet_ids_2:
        idx = tracklet_id_to_idx.get(tid)
        if idx is not None:
            indices_2.append(idx)
            stat = all_stats[idx]
            print(f"   Tracklet ID {tid} → Index [{idx}]: Frames {stat['start_frame']}-{stat['end_frame']} ({stat['duration']} frames)")
    
    print(f"\n📋 PERSON #53 TRACKLETS (by ID):")
    indices_53 = []
    for tid in tracklet_ids_53:
        idx = tracklet_id_to_idx.get(tid)
        if idx is not None:
            indices_53.append(idx)
            stat = all_stats[idx]
            print(f"   Tracklet ID {tid} → Index [{idx}]: Frames {stat['start_frame']}-{stat['end_frame']} ({stat['duration']} frames)")
    
    # Now test if the LAST tracklet of Person #2 can merge with FIRST tracklet of Person #53
    if len(indices_2) == 0 or len(indices_53) == 0:
        print(f"\n❌ Could not find tracklets for both persons")
        return
    
    # Get stats for comparison (last of #2, first of #53)
    stat_2 = all_stats[indices_2[-1]]  # Last tracklet of Person #2
    stat_53 = all_stats[indices_53[0]]  # First tracklet of Person #53
    
    print(f"\n🔬 TESTING MERGE BETWEEN:")
    print(f"   Last tracklet of Person #2: Index [{indices_2[-1]}] frames {stat_2['start_frame']}-{stat_2['end_frame']}")
    print(f"   First tracklet of Person #53: Index [{indices_53[0]}] frames {stat_53['start_frame']}-{stat_53['end_frame']}")
    
    # Load config to get merge criteria
    config_file = Path(__file__).parent.parent / 'configs' / 'pipeline_config.yaml'
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    criteria = config['stage3b_group']['grouping']['enhanced_criteria']
    
    print(f"\n📋 MERGE CRITERIA:")
    for key, val in criteria.items():
        print(f"   {key}: {val}")
    
    # Now test each check manually
    print(f"\n{'=' * 70}")
    print(f"🧪 TESTING EACH MERGE CHECK:")
    print(f"{'=' * 70}")
    
    # CHECK 1: Temporal order
    print(f"\n1️⃣  TEMPORAL ORDER (stat1 starts before stat2):")
    print(f"   stat_2['start_frame'] = {stat_2['start_frame']}")
    print(f"   stat_53['start_frame'] = {stat_53['start_frame']}")
    print(f"   {stat_2['start_frame']} < {stat_53['start_frame']}?")
    
    if stat_2['start_frame'] >= stat_53['start_frame']:
        print(f"   ❌ FAILED - stat_2 starts at or after stat_53")
    else:
        print(f"   ✅ PASSED")
    
    # CHECK 2: Temporal gap/overlap
    print(f"\n2️⃣  TEMPORAL GAP/OVERLAP:")
    gap = stat_53['start_frame'] - stat_2['end_frame']
    print(f"   gap = {stat_53['start_frame']} - {stat_2['end_frame']} = {gap}")
    
    max_overlap = criteria.get('max_overlap_frames', 50)
    max_temporal_gap = criteria['max_temporal_gap']
    
    if gap > 0:
        print(f"   Gap is positive (no overlap)")
        print(f"   {gap} <= max_temporal_gap ({max_temporal_gap})?")
        if gap > max_temporal_gap:
            print(f"   ❌ FAILED - gap too large")
        else:
            print(f"   ✅ PASSED")
    else:
        overlap_frames = abs(gap)
        print(f"   Gap is negative (overlap of {overlap_frames} frames)")
        print(f"   {overlap_frames} <= max_overlap_frames ({max_overlap})?")
        if overlap_frames > max_overlap:
            print(f"   ❌ FAILED - overlap too large")
        else:
            print(f"   ✅ PASSED")
    
    # CHECK 3: Spatial proximity
    print(f"\n3️⃣  SPATIAL PROXIMITY:")
    last_bbox_2 = np.array(stat_2['last_bbox'])
    first_bbox_53 = np.array(stat_53['first_bbox'])
    
    last_center_2 = (last_bbox_2[:2] + last_bbox_2[2:]) / 2
    first_center_53 = (first_bbox_53[:2] + first_bbox_53[2:]) / 2
    distance = np.linalg.norm(last_center_2 - first_center_53)
    
    max_spatial_distance = criteria['max_spatial_distance']
    
    print(f"   Last bbox of stat_2: {last_bbox_2}")
    print(f"   First bbox of stat_53: {first_bbox_53}")
    print(f"   Last center of stat_2: ({last_center_2[0]:.1f}, {last_center_2[1]:.1f})")
    print(f"   First center of stat_53: ({first_center_53[0]:.1f}, {first_center_53[1]:.1f})")
    print(f"   Distance: {distance:.1f} pixels")
    print(f"   {distance:.1f} <= max_spatial_distance ({max_spatial_distance})?")
    
    if distance > max_spatial_distance:
        print(f"   ❌ FAILED - bboxes too far apart")
    else:
        print(f"   ✅ PASSED")
    
    # CHECK 4: Area ratio
    print(f"\n4️⃣  AREA RATIO (size consistency):")
    mean_area_2 = stat_2['mean_area']
    mean_area_53 = stat_53['mean_area']
    area_ratio = mean_area_53 / (mean_area_2 + 1e-8)
    
    area_ratio_range = criteria['area_ratio_range']
    
    print(f"   Mean area of stat_2: {mean_area_2:.1f}")
    print(f"   Mean area of stat_53: {mean_area_53:.1f}")
    print(f"   Area ratio: {area_ratio:.3f}")
    print(f"   {area_ratio_range[0]} <= {area_ratio:.3f} <= {area_ratio_range[1]}?")
    
    if area_ratio < area_ratio_range[0] or area_ratio > area_ratio_range[1]:
        print(f"   ❌ FAILED - size mismatch")
    else:
        print(f"   ✅ PASSED")
    
    # CHECK 5: Motion alignment
    print(f"\n5️⃣  MOTION ALIGNMENT (direction similarity):")
    vel_2 = np.array(stat_2['mean_velocity'])
    vel_53 = np.array(stat_53['mean_velocity'])
    
    vel_2_mag = np.linalg.norm(vel_2)
    vel_53_mag = np.linalg.norm(vel_53)
    
    min_motion_alignment = criteria.get('min_motion_alignment', 0.6)
    
    print(f"   Mean velocity of stat_2: ({vel_2[0]:.2f}, {vel_2[1]:.2f}), mag={vel_2_mag:.2f}")
    print(f"   Mean velocity of stat_53: ({vel_53[0]:.2f}, {vel_53[1]:.2f}), mag={vel_53_mag:.2f}")
    
    if vel_2_mag > 1.0 and vel_53_mag > 1.0:
        cosine_sim = np.dot(vel_2, vel_53) / (vel_2_mag * vel_53_mag + 1e-8)
        print(f"   Cosine similarity: {cosine_sim:.3f}")
        print(f"   {cosine_sim:.3f} >= min_motion_alignment ({min_motion_alignment})?")
        
        if cosine_sim < min_motion_alignment:
            print(f"   ❌ FAILED - motion directions don't align")
        else:
            print(f"   ✅ PASSED")
    else:
        print(f"   ⏭️  SKIPPED - at least one tracklet has low motion (< 1.0 px/frame)")
    
    # CHECK 6: Jitter difference
    print(f"\n6️⃣  JITTER DIFFERENCE (smoothness consistency):")
    jitter_2 = stat_2['center_jitter']
    jitter_53 = stat_53['center_jitter']
    jitter_diff = abs(jitter_2 - jitter_53)
    
    max_jitter_difference = criteria.get('max_jitter_difference', 40.0)
    
    print(f"   Center jitter of stat_2: {jitter_2:.2f}")
    print(f"   Center jitter of stat_53: {jitter_53:.2f}")
    print(f"   Jitter difference: {jitter_diff:.2f}")
    print(f"   {jitter_diff:.2f} <= max_jitter_difference ({max_jitter_difference})?")
    
    if jitter_diff > max_jitter_difference:
        print(f"   ❌ FAILED - jitter too different")
    else:
        print(f"   ✅ PASSED")
    
    print(f"\n{'=' * 70}")
    print(f"🎯 CONCLUSION:")
    print(f"{'=' * 70}")
    print(f"If any check FAILED above, that's why the tracklets weren't merged!")
    print(f"Most likely culprits:")
    print(f"  - Spatial distance (last bbox of #2 vs first bbox of #53)")
    print(f"  - Motion alignment (if velocities point different directions)")
    print(f"  - Area ratio (if person size changes significantly)")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str,
                       default='/content/unifiedposepipeline/demo_data/outputs/kohli_nets')
    
    args = parser.parse_args()
    debug_merge_checks(args.output_dir)
