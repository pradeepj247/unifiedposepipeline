#!/usr/bin/env python3
"""
Visualize Tracklet Grouping Results

Shows 3 tables:
1. All raw tracklets (before grouping)
2. ReID merge candidates
3. Final canonical persons (after grouping/merging)

Usage:
    python visualize_grouping.py --output-dir /path/to/outputs/video_name
"""

import argparse
import json
import numpy as np
from pathlib import Path
from tabulate import tabulate


def load_tracklets(tracklets_file):
    """Load tracklets from NPZ file"""
    data = np.load(tracklets_file, allow_pickle=True)
    tracklets = data['tracklets']
    
    results = []
    for t in tracklets:
        tracklet_id = t['tracklet_id']
        frames = t['frame_numbers']
        results.append({
            'id': tracklet_id,
            'appearances': len(frames),
            'start_frame': int(frames[0]),
            'end_frame': int(frames[-1]),
            'duration': int(frames[-1] - frames[0] + 1)
        })
    
    return sorted(results, key=lambda x: x['id'])


def load_reid_candidates(candidates_file):
    """Load ReID merge candidates from JSON"""
    if not Path(candidates_file).exists():
        return []
    
    with open(candidates_file, 'r') as f:
        candidates = json.load(f)
    
    results = []
    for c in candidates:
        results.append({
            'tracklet_1': c['tracklet_1'],
            'tracklet_2': c['tracklet_2'],
            'temporal_gap': c['temporal_gap'],
            'spatial_distance': c['spatial_distance'],
            'area_ratio': c['area_ratio']
        })
    
    return results


def load_canonical_persons(persons_file, grouping_log_file):
    """Load canonical persons and their constituent tracklets"""
    persons_data = np.load(persons_file, allow_pickle=True)
    persons = persons_data['persons']
    
    with open(grouping_log_file, 'r') as f:
        grouping_log = json.load(f)
    
    results = []
    for p in persons:
        person_id = p['person_id']
        tracklet_ids = p['tracklet_ids']
        frames = p['frame_numbers']
        
        # Get grouping info from log
        group_info = grouping_log['persons'].get(str(person_id), {})
        method = group_info.get('method', 'unknown')
        
        results.append({
            'person_id': person_id,
            'tracklet_ids': tracklet_ids,
            'num_tracklets': len(tracklet_ids),
            'appearances': len(frames),
            'start_frame': int(frames[0]),
            'end_frame': int(frames[-1]),
            'duration': int(frames[-1] - frames[0] + 1),
            'method': method
        })
    
    return sorted(results, key=lambda x: x['person_id'])


def print_table1_raw_tracklets(tracklets):
    """Print Table 1: All raw tracklets"""
    print("\n" + "="*80)
    print("📊 TABLE 1: RAW TRACKLETS (Before Grouping)")
    print("="*80)
    
    table_data = []
    for t in tracklets:
        table_data.append([
            t['id'],
            t['appearances'],
            t['start_frame'],
            t['end_frame'],
            t['duration']
        ])
    
    headers = ['Tracklet ID', '# Appearances', 'Start Frame', 'End Frame', 'Duration (frames)']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print(f"\nTotal tracklets: {len(tracklets)}")


def print_table2_reid_candidates(candidates, tracklets_dict):
    """Print Table 2: ReID merge candidates"""
    print("\n" + "="*80)
    print("📊 TABLE 2: ReID MERGE CANDIDATES")
    print("="*80)
    
    if len(candidates) == 0:
        print("No ReID candidates found (Stage 4a disabled or no candidates identified)")
        return
    
    table_data = []
    for c in candidates:
        t1 = tracklets_dict.get(c['tracklet_1'], {})
        t2 = tracklets_dict.get(c['tracklet_2'], {})
        
        table_data.append([
            f"{c['tracklet_1']} → {c['tracklet_2']}",
            f"{t1.get('appearances', '?')} → {t2.get('appearances', '?')}",
            f"{t1.get('end_frame', '?')} → {t2.get('start_frame', '?')}",
            c['temporal_gap'],
            f"{c['spatial_distance']:.1f}",
            f"{c['area_ratio']:.2f}"
        ])
    
    headers = ['Tracklet Pair', '# Apps', 'Transition', 'Gap (frames)', 'Distance (px)', 'Area Ratio']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print(f"\nTotal candidates: {len(candidates)}")


def print_table3_canonical_persons(persons):
    """Print Table 3: Final canonical persons"""
    print("\n" + "="*80)
    print("📊 TABLE 3: CANONICAL PERSONS (After Grouping/Merging)")
    print("="*80)
    
    table_data = []
    for p in persons:
        tracklet_ids_str = ', '.join([str(tid) for tid in sorted(p['tracklet_ids'])])
        if len(tracklet_ids_str) > 40:
            tracklet_ids_str = tracklet_ids_str[:37] + '...'
        
        table_data.append([
            p['person_id'],
            f"({tracklet_ids_str})",
            p['num_tracklets'],
            p['appearances'],
            p['start_frame'],
            p['end_frame'],
            p['duration'],
            p['method']
        ])
    
    headers = ['Person ID', 'Tracklet IDs', '# Tracklets', '# Apps', 'Start', 'End', 'Duration', 'Method']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print(f"\nTotal canonical persons: {len(persons)}")
    
    # Print detailed breakdown of multi-tracklet persons
    multi_tracklet_persons = [p for p in persons if p['num_tracklets'] > 1]
    if multi_tracklet_persons:
        print("\n" + "-"*80)
        print("🔗 MULTI-TRACKLET PERSONS (Grouped/Merged):")
        print("-"*80)
        for p in multi_tracklet_persons:
            tracklet_ids_str = ', '.join([str(tid) for tid in sorted(p['tracklet_ids'])])
            print(f"  Person {p['person_id']}: Tracklets [{tracklet_ids_str}] | "
                  f"{p['num_tracklets']} tracklets | {p['appearances']} appearances | "
                  f"Frames {p['start_frame']}-{p['end_frame']} | Method: {p['method']}")


def main():
    parser = argparse.ArgumentParser(description='Visualize tracklet grouping results')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory containing NPZ and JSON files')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # File paths
    tracklets_file = output_dir / 'tracklets_raw.npz'
    reid_candidates_file = output_dir / 'reid_candidates.json'
    canonical_persons_file = output_dir / 'canonical_persons.npz'
    grouping_log_file = output_dir / 'grouping_log.json'
    ranking_report_file = output_dir / 'ranking_report.json'
    
    # Check files exist
    if not tracklets_file.exists():
        print(f"❌ Error: {tracklets_file} not found")
        return
    if not canonical_persons_file.exists():
        print(f"❌ Error: {canonical_persons_file} not found")
        return
    if not grouping_log_file.exists():
        print(f"❌ Error: {grouping_log_file} not found")
        return
    
    print(f"\n📂 Loading data from: {output_dir}")
    
    # Load data
    tracklets = load_tracklets(tracklets_file)
    tracklets_dict = {t['id']: t for t in tracklets}
    
    reid_candidates = load_reid_candidates(reid_candidates_file)
    
    canonical_persons = load_canonical_persons(canonical_persons_file, grouping_log_file)
    
    # Print tables
    print_table1_raw_tracklets(tracklets)
    print_table2_reid_candidates(reid_candidates, tracklets_dict)
    print_table3_canonical_persons(canonical_persons)
    
    # Print ranking info if available
    if ranking_report_file.exists():
        with open(ranking_report_file, 'r') as f:
            ranking = json.load(f)
        
        print("\n" + "="*80)
        print("🎯 PRIMARY PERSON SELECTION")
        print("="*80)
        primary_id = ranking['primary_person_id']
        primary = next((p for p in canonical_persons if p['person_id'] == primary_id), None)
        
        if primary:
            print(f"  Primary Person: {primary_id}")
            print(f"  Tracklet IDs: {sorted(primary['tracklet_ids'])}")
            print(f"  Total appearances: {primary['appearances']}")
            print(f"  Duration: {primary['duration']} frames (frame {primary['start_frame']}-{primary['end_frame']})")
            print(f"  Ranking score: {ranking['primary_score']:.4f}")
            print(f"  Selection method: {ranking['ranking_method']}")
    
    print("\n" + "="*80)
    print("✅ Visualization complete!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
