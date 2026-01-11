#!/usr/bin/env python3
"""
Stage 5b: Visualize Tracklet Grouping Results

Shows 4 tables:
1. All raw tracklets (before grouping)
2. ReID merge candidates
3. Final canonical persons (after grouping/merging)
4. Top 5 persons ranked by duration

Usage:
    python stage5b_visualize_grouping.py --config configs/pipeline_config.yaml
"""

import argparse
import json
import numpy as np
import yaml
import re
from pathlib import Path
from tabulate import tabulate


def resolve_path_variables(config):
    """Recursively resolve ${variable} in config"""
    global_vars = config.get('global', {})
    
    # First pass: resolve variables within global section itself
    def resolve_string_once(s, vars_dict):
        if not isinstance(s, str):
            return s
        return re.sub(
            r'\$\{(\w+)\}',
            lambda m: str(vars_dict.get(m.group(1), m.group(0))),
            s
        )
    
    # Resolve global variables iteratively
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
    
    def resolve_string(s):
        return re.sub(
            r'\$\{(\w+)\}',
            lambda m: str(global_vars.get(m.group(1), m.group(0))),
            s
        )
    
    def resolve_recursive(obj):
        if isinstance(obj, dict):
            return {k: resolve_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_recursive(v) for v in obj]
        elif isinstance(obj, str):
            return resolve_string(obj)
        return obj
    
    result = resolve_recursive(config)
    result['global'] = global_vars
    return result


def load_config(config_path):
    """Load and resolve YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Auto-extract current_video from video_file
    video_file = config.get('global', {}).get('video_file', '')
    if video_file:
        import os
        video_name = os.path.splitext(video_file)[0]
        config['global']['current_video'] = video_name
    
    return resolve_path_variables(config)


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
            'temporal_gap': c['gap'],
            'spatial_distance': c['distance'],
            'area_ratio': c['area_ratio'],
            'transition_1': c.get('transition_frame_1', '?'),
            'transition_2': c.get('transition_frame_2', '?')
        })
    
    return results


def load_canonical_persons(persons_file, grouping_log_file):
    """Load canonical persons and their constituent tracklets"""
    persons_data = np.load(persons_file, allow_pickle=True)
    persons = persons_data['persons']
    
    with open(grouping_log_file, 'r') as f:
        grouping_log = json.load(f)
    
    # Build dict for fast lookup
    log_dict = {log['canonical_id']: log for log in grouping_log}
    
    results = []
    for p in persons:
        person_id = p['person_id']
        tracklet_ids = p['original_tracklet_ids']
        frames = p['frame_numbers']
        
        # Get grouping info from log
        log_entry = log_dict.get(person_id, {})
        num_merged = log_entry.get('num_merged', len(tracklet_ids))
        
        results.append({
            'person_id': person_id,
            'tracklet_ids': tracklet_ids,
            'num_tracklets': len(tracklet_ids),
            'appearances': len(frames),
            'start_frame': int(frames[0]),
            'end_frame': int(frames[-1]),
            'duration': int(frames[-1] - frames[0] + 1),
            'method': 'heuristic' if num_merged > 1 else 'single'
        })
    
    return sorted(results, key=lambda x: x['person_id'])


def print_table1_raw_tracklets(tracklets):
    """Print Table 1: All raw tracklets (numbered sequentially)"""
    print("\n" + "="*80)
    print("📊 TABLE 1: RAW TRACKLETS (Before Grouping)")
    print("="*80)
    print(f"Total: {len(tracklets)} tracklets\n")
    
    table_data = []
    for idx, t in enumerate(tracklets, 1):
        table_data.append([
            idx,
            t['id'],
            t['appearances'],
            t['start_frame'],
            t['end_frame'],
            t['duration']
        ])
    
    headers = ['#', 'Tracklet ID', 'Appearances', 'Start', 'End', 'Duration']
    print(tabulate(table_data, headers=headers, tablefmt='simple'))


def print_table2_reid_candidates(candidates, tracklets_dict):
    """Print Table 2: ReID merge candidates"""
    print("\n" + "="*80)
    print("📊 TABLE 2: MERGE CANDIDATES (Temporally Disconnected)")
    print("="*80)
    print(f"Total: {len(candidates)} candidate pairs\n")
    
    if len(candidates) == 0:
        print("No merge candidates identified.")
        return
    
    table_data = []
    for idx, c in enumerate(candidates, 1):
        t1 = tracklets_dict.get(c['tracklet_1'], {})
        t2 = tracklets_dict.get(c['tracklet_2'], {})
        
        table_data.append([
            idx,
            c['tracklet_1'],
            c['tracklet_2'],
            c['temporal_gap'],
            f"{c['spatial_distance']:.0f}",
            f"{c['area_ratio']:.2f}",
            f"{t1.get('end_frame', '?')} → {t2.get('start_frame', '?')}"
        ])
    
    headers = ['#', 'Track 1', 'Track 2', 'Gap', 'Distance', 'Area Ratio', 'Transition']
    print(tabulate(table_data, headers=headers, tablefmt='simple'))


def print_table3_merge_results(canonical_persons, candidates):
    """Print Table 3: Merge results (accepted vs rejected)"""
    print("\n" + "="*80)
    print("📊 TABLE 3: MERGE RESULTS")
    print("="*80)
    
    # Find which candidates were merged
    merged_persons = [p for p in canonical_persons if p['num_tracklets'] > 1]
    
    # Build set of all merged tracklet IDs
    all_merged_ids = set()
    for p in merged_persons:
        all_merged_ids.update(p['tracklet_ids'])
    
    # Check each candidate
    accepted = []
    rejected = []
    
    for c in candidates:
        t1, t2 = c['tracklet_1'], c['tracklet_2']
        # Check if both tracklets ended up in the same person
        merged_together = False
        for p in merged_persons:
            if t1 in p['tracklet_ids'] and t2 in p['tracklet_ids']:
                merged_together = True
                accepted.append((c, p['person_id']))
                break
        
        if not merged_together:
            rejected.append(c)
    
    print(f"✅ Accepted: {len(accepted)} merges")
    print(f"❌ Rejected: {len(rejected)} candidates\n")
    
    if accepted:
        print("ACCEPTED MERGES:")
        table_data = []
        for c, person_id in accepted:
            table_data.append([
                f"{c['tracklet_1']} + {c['tracklet_2']}",
                person_id,
                c['temporal_gap'],
                f"{c['spatial_distance']:.0f}",
                f"{c['area_ratio']:.2f}"
            ])
        headers = ['Merged Pair', 'Person ID', 'Gap', 'Distance', 'Ratio']
        print(tabulate(table_data, headers=headers, tablefmt='simple'))
    
    if rejected:
        print("\n\nREJECTED CANDIDATES:")
        table_data = []
        for c in rejected:
            # Determine likely rejection reason
            reason = []
            if c['temporal_gap'] > 50:
                reason.append('gap>50')
            if c['spatial_distance'] > 300:
                reason.append('dist>300')
            if c['area_ratio'] < 0.6 or c['area_ratio'] > 1.4:
                reason.append('area')
            reason_str = ', '.join(reason) if reason else 'unknown'
            
            table_data.append([
                f"{c['tracklet_1']} + {c['tracklet_2']}",
                c['temporal_gap'],
                f"{c['spatial_distance']:.0f}",
                f"{c['area_ratio']:.2f}",
                reason_str
            ])
        headers = ['Candidate Pair', 'Gap', 'Distance', 'Ratio', 'Likely Reason']
        print(tabulate(table_data, headers=headers, tablefmt='simple'))


def print_table4_top_persons(canonical_persons, top_n=5):
    """Print Table 4: Top N persons by duration"""
    print("\n" + "="*80)
    print(f"📊 TABLE 4: TOP {top_n} PERSONS (Ranked by Duration)")
    print("="*80)
    print(f"Total canonical persons: {len(canonical_persons)}\n")
    
    # Sort by duration descending
    sorted_persons = sorted(canonical_persons, key=lambda x: x['duration'], reverse=True)
    top_persons = sorted_persons[:top_n]
    
    table_data = []
    for rank, p in enumerate(top_persons, 1):
        tracklet_ids_str = ', '.join([str(tid) for tid in sorted(p['tracklet_ids'])])
        
        table_data.append([
            rank,
            p['person_id'],
            tracklet_ids_str,
            p['num_tracklets'],
            p['appearances'],
            p['start_frame'],
            p['end_frame'],
            p['duration']
        ])
    
    headers = ['Rank', 'Person ID', 'Tracklet IDs', '# Tracks', 'Appear', 'Start', 'End', 'Duration']
    print(tabulate(table_data, headers=headers, tablefmt='simple'))
    
    # Highlight primary person
    if top_persons:
        primary = top_persons[0]
        print(f"\n🎯 Primary Person: {primary['person_id']} (Duration: {primary['duration']} frames)")


def main():
    parser = argparse.ArgumentParser(description='Stage 5b: Visualize tracklet grouping results')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to pipeline configuration YAML')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    print(f"\n{'='*70}")
    print(f"📊 STAGE 5b: VISUALIZE GROUPING RESULTS")
    print(f"{'='*70}\n")
    
    # File paths from config
    tracklets_file = Path(config['stage2_track']['output']['tracklets_file'])
    reid_candidates_file = Path(config['stage3_analyze']['output']['candidates_file'])
    canonical_persons_file = Path(config['stage5_group_canonical']['output']['canonical_persons_file'])
    grouping_log_file = Path(config['stage5_group_canonical']['output']['grouping_log_file'])
    ranking_report_file = Path(config['stage5_rank']['output']['ranking_report_file'])
    
    # Check files exist
    if not tracklets_file.exists():
        print(f"❌ Error: {tracklets_file.name} not found")
        return
    if not canonical_persons_file.exists():
        print(f"❌ Error: {canonical_persons_file.name} not found")
        return
    if not grouping_log_file.exists():
        print(f"❌ Error: {grouping_log_file.name} not found")
        return
    
    print(f"📂 Loading grouping data...")
    
    # Load data
    tracklets = load_tracklets(tracklets_file)
    tracklets_dict = {t['id']: t for t in tracklets}
    
    reid_candidates = load_reid_candidates(reid_candidates_file)
    
    canonical_persons = load_canonical_persons(canonical_persons_file, grouping_log_file)
    
    # Print tables
    print_table1_raw_tracklets(tracklets)
    print_table2_reid_candidates(reid_candidates, tracklets_dict)
    print_table3_merge_results(canonical_persons, reid_candidates)
    print_table4_top_persons(canonical_persons, top_n=5)
    
    print(f"\n{'='*70}")
    print(f"✅ VISUALIZATION COMPLETE!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
