#!/usr/bin/env python3
"""
Analyze the Person #3 and #58 split with YOLOv8s.
Why did better detection lead to worse grouping?
"""

import numpy as np
import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def analyze_new_split(output_dir):
    output_path = Path(output_dir)
    
    print("=" * 70)
    print("🔍 ANALYZING PERSON #3 vs #58 SPLIT (YOLOv8s)")
    print("=" * 70)
    
    # Load canonical persons
    canonical_file = output_path / 'canonical_persons.npz'
    canonical_data = np.load(canonical_file, allow_pickle=True)
    persons = canonical_data['persons']
    
    print(f"\n📊 Total canonical persons: {len(persons)}")
    
    # Find Person #3 and #58
    person_3 = None
    person_58 = None
    
    for p in persons:
        if p['person_id'] == 3:
            person_3 = p
        elif p['person_id'] == 58:
            person_58 = p
    
    if not person_3 or not person_58:
        print(f"❌ Could not find both persons")
        print(f"\nAll persons starting from frame 0:")
        for p in persons:
            if p['frame_numbers'][0] <= 50:
                print(f"   Person {p['person_id']}: frames {p['frame_numbers'][0]}-{p['frame_numbers'][-1]}, tracklets {p.get('original_tracklet_ids', [])}")
        return
    
    print(f"\n👤 PERSON #3:")
    print(f"   Frames: {person_3['frame_numbers'][0]} - {person_3['frame_numbers'][-1]} ({len(person_3['frame_numbers'])} frames)")
    print(f"   Tracklet IDs: {person_3.get('original_tracklet_ids', [])}")
    
    print(f"\n👤 PERSON #58:")
    print(f"   Frames: {person_58['frame_numbers'][0]} - {person_58['frame_numbers'][-1]} ({len(person_58['frame_numbers'])} frames)")
    print(f"   Tracklet IDs: {person_58.get('original_tracklet_ids', [])}")
    
    # Check gap
    gap = person_58['frame_numbers'][0] - person_3['frame_numbers'][-1]
    print(f"\n⚠️  GAP BETWEEN PERSONS:")
    print(f"   Person #3 ends: frame {person_3['frame_numbers'][-1]}")
    print(f"   Person #58 starts: frame {person_58['frame_numbers'][0]}")
    print(f"   Gap: {gap} frames")
    
    if gap > 100:
        print(f"   🚨 HUGE GAP! {gap} frames missing - likely other tracklets in between")
    
    # Load tracklets to see what's in the gap
    tracklets_file = output_path / 'tracklets_raw.npz'
    tracklets_data = np.load(tracklets_file, allow_pickle=True)
    tracklets = tracklets_data['tracklets']
    
    print(f"\n🔍 TRACKLETS IN THE GAP (frames {person_3['frame_numbers'][-1]}-{person_58['frame_numbers'][0]}):")
    gap_tracklets = []
    for t in tracklets:
        start = t['frame_numbers'][0]
        end = t['frame_numbers'][-1]
        
        # Check if tracklet overlaps with gap
        if start <= person_58['frame_numbers'][0] and end >= person_3['frame_numbers'][-1]:
            gap_tracklets.append((t['tracklet_id'], start, end, len(t['frame_numbers'])))
            print(f"   Tracklet {t['tracklet_id']}: frames {start}-{end} ({len(t['frame_numbers'])} frames)")
    
    if len(gap_tracklets) == 0:
        print(f"   ❌ NO TRACKLETS in gap - person completely lost!")
    else:
        print(f"\n   Found {len(gap_tracklets)} tracklets in gap")
        print(f"   ⚠️ These should have been grouped with Person #3 or #58!")
    
    # Load stats and test merge conditions
    stats_file = output_path / 'tracklet_stats.npz'
    stats_data = np.load(stats_file, allow_pickle=True)
    all_stats = stats_data['statistics']
    
    # Build tracklet ID to index mapping
    tracklet_id_to_idx = {}
    for idx, t in enumerate(tracklets):
        tracklet_id_to_idx[t['tracklet_id']] = idx
    
    # Get last tracklet of Person #3
    person_3_tracklet_ids = person_3.get('original_tracklet_ids', [])
    if len(person_3_tracklet_ids) > 0:
        last_tid_3 = person_3_tracklet_ids[-1]
        idx_3 = tracklet_id_to_idx.get(last_tid_3)
        
        if idx_3 is not None:
            stat_3 = all_stats[idx_3]
            
            print(f"\n🔬 TESTING MERGE: Last tracklet of Person #3 (ID {last_tid_3})")
            print(f"   Frames: {stat_3['start_frame']}-{stat_3['end_frame']}")
            print(f"   Last bbox: {stat_3['last_bbox']}")
            print(f"   Mean velocity: {stat_3['mean_velocity']}")
            print(f"   Jitter: {stat_3['center_jitter']:.2f}")
            
            # Load config
            config_file = Path(__file__).parent.parent / 'configs' / 'pipeline_config.yaml'
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            criteria = config['stage3b_group']['grouping']['enhanced_criteria']
            
            # Test against first tracklet in gap
            if len(gap_tracklets) > 0:
                gap_tid = gap_tracklets[0][0]
                idx_gap = tracklet_id_to_idx.get(gap_tid)
                
                if idx_gap is not None:
                    stat_gap = all_stats[idx_gap]
                    
                    print(f"\n   vs First tracklet in gap (ID {gap_tid})")
                    print(f"   Frames: {stat_gap['start_frame']}-{stat_gap['end_frame']}")
                    print(f"   First bbox: {stat_gap['first_bbox']}")
                    print(f"   Mean velocity: {stat_gap['mean_velocity']}")
                    print(f"   Jitter: {stat_gap['center_jitter']:.2f}")
                    
                    # Test merge checks
                    gap = stat_gap['start_frame'] - stat_3['end_frame']
                    print(f"\n   📋 MERGE CHECKS:")
                    print(f"   1. Temporal gap: {gap} frames (max: {criteria['max_temporal_gap']})")
                    if gap > criteria['max_temporal_gap']:
                        print(f"      ❌ FAILED - gap too large")
                    else:
                        print(f"      ✅ PASSED")
                    
                    # Spatial distance
                    last_center_3 = (stat_3['last_bbox'][:2] + stat_3['last_bbox'][2:]) / 2
                    first_center_gap = (stat_gap['first_bbox'][:2] + stat_gap['first_bbox'][2:]) / 2
                    distance = np.linalg.norm(last_center_3 - first_center_gap)
                    
                    print(f"   2. Spatial distance: {distance:.1f} px (max: {criteria['max_spatial_distance']})")
                    if distance > criteria['max_spatial_distance']:
                        print(f"      ❌ FAILED - too far apart")
                    else:
                        print(f"      ✅ PASSED")
                    
                    # Jitter difference
                    jitter_diff = abs(stat_3['center_jitter'] - stat_gap['center_jitter'])
                    print(f"   3. Jitter difference: {jitter_diff:.2f} (max: {criteria['max_jitter_difference']})")
                    if jitter_diff > criteria['max_jitter_difference']:
                        print(f"      ❌ FAILED - jitter too different")
                    else:
                        print(f"      ✅ PASSED")
    
    print(f"\n{'='*70}")
    print(f"🎯 CONCLUSION:")
    print(f"{'='*70}")
    if gap > 100:
        print(f"Person #3 and #58 have {gap} frame gap - likely multiple tracklets in between")
        print(f"YOLOv8s may be creating MORE tracklets due to better detection")
        print(f"Need to check why gap tracklets weren't merged")
    else:
        print(f"Small gap - test merge checks above to see which one is failing")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str,
                       default='/content/unifiedposepipeline/demo_data/outputs/kohli_nets')
    
    args = parser.parse_args()
    analyze_new_split(args.output_dir)
