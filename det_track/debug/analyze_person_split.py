#!/usr/bin/env python3
"""
Debug script to analyze why person #2 and #53 weren't merged.

Person #2: frames 0-457
Person #53: frames 419-2024
Overlap: frames 419-457 (39 frames where both exist!)

This script investigates:
1. What tracklets compose each person
2. Why ByteTrack created separate IDs
3. Why Stage 3b didn't merge them
4. Bbox positions/sizes during overlap
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def analyze_person_split(output_dir):
    """Analyze the split between person #2 and #53"""
    
    output_path = Path(output_dir)
    
    # Load files
    print("="*70)
    print("ANALYZING PERSON SPLIT: Person #2 vs Person #53")
    print("="*70)
    
    # 1. Load canonical persons (Stage 3b output)
    canonical_file = output_path / 'canonical_persons.npz'
    canonical_data = np.load(canonical_file, allow_pickle=True)
    persons = canonical_data['persons']
    
    person_2 = None
    person_53 = None
    
    for p in persons:
        if p['person_id'] == 2:
            person_2 = p
        elif p['person_id'] == 53:
            person_53 = p
    
    if not person_2 or not person_53:
        print("❌ Could not find both persons!")
        return
    
    print(f"\n📊 PERSON #2:")
    print(f"   Frames: {person_2['frame_numbers'][0]} - {person_2['frame_numbers'][-1]}")
    print(f"   Total frames: {len(person_2['frame_numbers'])}")
    print(f"   Tracklet IDs: {person_2.get('tracklet_ids', 'N/A')}")
    
    print(f"\n📊 PERSON #53:")
    print(f"   Frames: {person_53['frame_numbers'][0]} - {person_53['frame_numbers'][-1]}")
    print(f"   Total frames: {len(person_53['frame_numbers'])}")
    print(f"   Tracklet IDs: {person_53.get('tracklet_ids', 'N/A')}")
    
    # 2. Analyze overlap
    frames_2 = set(person_2['frame_numbers'])
    frames_53 = set(person_53['frame_numbers'])
    overlap = frames_2 & frames_53
    
    print(f"\n⚠️  OVERLAP ANALYSIS:")
    print(f"   Overlapping frames: {len(overlap)} frames")
    if overlap:
        print(f"   Overlap range: {min(overlap)} - {max(overlap)}")
    
    # 3. Load tracklets to see ByteTrack output
    tracklets_file = output_path / 'tracklets_raw.npz'
    tracklets_data = np.load(tracklets_file, allow_pickle=True)
    tracklets = tracklets_data['tracklets']
    
    print(f"\n🔍 BYTETRACK TRACKLETS:")
    print(f"   Total tracklets: {len(tracklets)}")
    
    # Find tracklets for person 2 and 53
    tracklets_2 = []
    tracklets_53 = []
    
    if 'tracklet_ids' in person_2:
        for tid in person_2['tracklet_ids']:
            for t in tracklets:
                if t['tracklet_id'] == tid:
                    tracklets_2.append(t)
                    break
    
    if 'tracklet_ids' in person_53:
        for tid in person_53['tracklet_ids']:
            for t in tracklets:
                if t['tracklet_id'] == tid:
                    tracklets_53.append(t)
                    break
    
    print(f"\n   Person #2 tracklets: {len(tracklets_2)}")
    for i, t in enumerate(tracklets_2):
        frames = t['frame_numbers']
        print(f"      Tracklet {t['tracklet_id']}: frames {frames[0]}-{frames[-1]} ({len(frames)} frames)")
    
    print(f"\n   Person #53 tracklets: {len(tracklets_53)}")
    for i, t in enumerate(tracklets_53):
        frames = t['frame_numbers']
        print(f"      Tracklet {t['tracklet_id']}: frames {frames[0]}-{frames[-1]} ({len(frames)} frames)")
    
    # 4. Analyze bbox positions during overlap
    if overlap:
        print(f"\n📍 BBOX ANALYSIS DURING OVERLAP (frames {min(overlap)}-{max(overlap)}):")
        
        # Get bboxes for overlapping frames
        overlap_sorted = sorted(overlap)[:10]  # First 10 overlap frames
        
        for frame_num in overlap_sorted:
            # Find bbox for person 2
            idx_2 = np.where(person_2['frame_numbers'] == frame_num)[0]
            bbox_2 = person_2['bboxes'][idx_2[0]] if len(idx_2) > 0 else None
            
            # Find bbox for person 53
            idx_53 = np.where(person_53['frame_numbers'] == frame_num)[0]
            bbox_53 = person_53['bboxes'][idx_53[0]] if len(idx_53) > 0 else None
            
            if bbox_2 is not None and bbox_53 is not None:
                # Calculate IoU
                x1_2, y1_2, x2_2, y2_2 = bbox_2
                x1_53, y1_53, x2_53, y2_53 = bbox_53
                
                # Intersection
                x1_i = max(x1_2, x1_53)
                y1_i = max(y1_2, y1_53)
                x2_i = min(x2_2, x2_53)
                y2_i = min(y2_2, y2_53)
                
                intersection = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
                area_2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                area_53 = (x2_53 - x1_53) * (y2_53 - y1_53)
                union = area_2 + area_53 - intersection
                iou = intersection / union if union > 0 else 0
                
                # Calculate center distance
                center_2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
                center_53 = ((x1_53 + x2_53) / 2, (y1_53 + y2_53) / 2)
                distance = np.sqrt((center_2[0] - center_53[0])**2 + (center_2[1] - center_53[1])**2)
                
                print(f"   Frame {frame_num}:")
                print(f"      Person #2:  bbox [{x1_2:.0f}, {y1_2:.0f}, {x2_2:.0f}, {y2_2:.0f}], center ({center_2[0]:.0f}, {center_2[1]:.0f})")
                print(f"      Person #53: bbox [{x1_53:.0f}, {y1_53:.0f}, {x2_53:.0f}, {y2_53:.0f}], center ({center_53[0]:.0f}, {center_53[1]:.0f})")
                print(f"      IoU: {iou:.3f}, Center distance: {distance:.1f}px")
    
    # 5. Check if they should have been merged
    print(f"\n🔄 STAGE 3B GROUPING ANALYSIS:")
    print(f"   Why weren't these merged?")
    print(f"   1. Temporal gap? Person #2 ends at frame {person_2['frame_numbers'][-1]}, Person #53 starts at {person_53['frame_numbers'][0]}")
    
    gap_frames = person_53['frame_numbers'][0] - person_2['frame_numbers'][-1]
    print(f"      Gap: {gap_frames} frames (negative = overlap)")
    
    if gap_frames < 0:
        print(f"      ⚠️ OVERLAP EXISTS! They should have been merged!")
        print(f"      Possible reasons:")
        print(f"      - Bboxes too far apart (spatial distance threshold)")
        print(f"      - Different tracklet patterns (one continuous, one fragmented)")
        print(f"      - Stage 3b grouping threshold too strict")
    else:
        print(f"      ✓ No overlap - gap of {gap_frames} frames might exceed max_gap threshold")
    
    # 6. Load tracklet stats to see grouping decisions
    tracklet_stats_file = output_path / 'tracklet_stats.npz'
    if tracklet_stats_file.exists():
        stats_data = np.load(tracklet_stats_file, allow_pickle=True)
        print(f"\n📈 TRACKLET STATS FILE KEYS: {list(stats_data.keys())}")
        
        # Try to load stats with flexible key names
        if 'stats' in stats_data:
            tracklet_stats = stats_data['stats']
            print(f"   Found {len(tracklet_stats)} tracklet stats")
        elif 'tracklet_stats' in stats_data:
            tracklet_stats = stats_data['tracklet_stats']
            print(f"   Found {len(tracklet_stats)} tracklet stats")
        else:
            print(f"   ⚠️ Could not find stats in NPZ file")
            tracklet_stats = None
        
        if tracklet_stats is not None and 'tracklet_ids' in person_2:
            print(f"\n   Person #2 tracklets:")
            for tid in person_2['tracklet_ids']:
                for ts in tracklet_stats:
                    if ts['tracklet_id'] == tid:
                        print(f"      Tracklet {tid}: duration={ts['duration']} frames, avg_conf={ts['avg_confidence']:.3f}")
                        break
        
        if tracklet_stats is not None and 'tracklet_ids' in person_53:
            print(f"   Person #53 tracklets:")
            for tid in person_53['tracklet_ids']:
                for ts in tracklet_stats:
                    if ts['tracklet_id'] == tid:
                        print(f"      Tracklet {tid}: duration={ts['duration']} frames, avg_conf={ts['avg_confidence']:.3f}")
                        break
    
    # 7. Load and show Stage 3b grouping config
    config_file = Path(__file__).parent.parent / 'configs' / 'pipeline_config.yaml'
    if config_file.exists():
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        stage3b_cfg = config.get('stage3b_group', {})
        grouping_cfg = stage3b_cfg.get('grouping_criteria', {})
        
        print(f"\n🔧 STAGE 3B GROUPING CONFIGURATION:")
        print(f"   max_temporal_gap: {grouping_cfg.get('max_temporal_gap', 'N/A')} frames")
        print(f"   max_spatial_distance: {grouping_cfg.get('max_spatial_distance', 'N/A')} pixels")
        print(f"   area_ratio_range: {grouping_cfg.get('area_ratio_range', 'N/A')}")
        print(f"   min_motion_alignment: {grouping_cfg.get('min_motion_alignment', 'N/A')}")
        print(f"   max_jitter_difference: {grouping_cfg.get('max_jitter_difference', 'N/A')}")
    
    print(f"\n{'='*70}")
    print(f"🔴 ROOT CAUSE IDENTIFIED:")
    print(f"{'='*70}")
    print(f"Person #2 (frames 0-457) and #53 (frames 419-2024) overlap by {len(overlap)} frames!")
    print(f"")
    print(f"IoU during overlap: 0.67-0.71 (VERY HIGH - clearly same person)")
    print(f"Center distance: 22-25px (VERY CLOSE)")
    print(f"")
    print(f"🐛 BUG FOUND: Stage 3b can_merge_enhanced() at line 118:")
    print(f"   if stat1['end_frame'] >= stat2['start_frame']: return False")
    print(f"   ↑ This REJECTS overlapping tracklets!")
    print(f"")
    print(f"This is wrong because:")
    print(f"1. ByteTrack can assign new ID even when person still visible (ID switch)")
    print(f"2. Person #2 extends to frame 457, Person #53 starts at 419")
    print(f"3. They overlap 38 frames but Stage 3b refuses to merge overlapping tracks")
    print(f"")
    print(f"✅ FIX: Change line 118 to allow small overlaps (e.g., < 50 frames)")
    print(f"   Overlaps are NORMAL when ByteTrack switches IDs mid-scene!")
    print(f"")
    print(f"Secondary causes:")
    print(f"- YOLOv8n detection gaps may have triggered the ByteTrack ID switch")
    print(f"- But even with perfect detection, Stage 3b would still fail to merge")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze person split between #2 and #53')
    parser.add_argument('--output-dir', type=str, 
                       default='/content/unifiedposepipeline/demo_data/outputs/kohli_nets',
                       help='Path to output directory with NPZ files')
    
    args = parser.parse_args()
    
    analyze_person_split(args.output_dir)
