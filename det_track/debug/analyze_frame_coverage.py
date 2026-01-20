#!/usr/bin/env python3
"""
Analyze frame coverage: Are we missing individual detections or entire tracklets?
"""

import numpy as np
import sys
from pathlib import Path

def analyze_frame_coverage(output_dir):
    output_path = Path(output_dir)
    
    print("=" * 70)
    print("🔍 ANALYZING FRAME COVERAGE FOR PERSON #2")
    print("=" * 70)
    
    # Load canonical persons
    canonical_file = output_path / 'canonical_persons.npz'
    canonical_data = np.load(canonical_file, allow_pickle=True)
    persons = canonical_data['persons']
    
    # Find Person #2
    person_2 = None
    for p in persons:
        if p['person_id'] == 2:
            person_2 = p
            break
    
    if not person_2:
        print("❌ Person #2 not found")
        return
    
    frame_numbers = person_2['frame_numbers']
    tracklet_ids = person_2.get('original_tracklet_ids', [])
    
    print(f"\n👤 PERSON #2 OVERALL:")
    print(f"   Frame range: {frame_numbers[0]} - {frame_numbers[-1]}")
    print(f"   Total frames detected: {len(frame_numbers)}")
    print(f"   Tracklets: {tracklet_ids}")
    
    # Load tracklets to see individual tracklet frame ranges
    tracklets_file = output_path / 'tracklets_raw.npz'
    tracklets_data = np.load(tracklets_file, allow_pickle=True)
    tracklets = tracklets_data['tracklets']
    
    print(f"\n📋 TRACKLET BREAKDOWN:")
    tracklet_frames = []
    for tid in tracklet_ids:
        for t in tracklets:
            if t['tracklet_id'] == tid:
                t_frames = t['frame_numbers']
                tracklet_frames.append(set(t_frames))
                print(f"   Tracklet {tid}:")
                print(f"      Frame range: {t_frames[0]} - {t_frames[-1]}")
                print(f"      Frames in tracklet: {len(t_frames)}")
                
                # Check for gaps within this tracklet
                expected_frames = t_frames[-1] - t_frames[0] + 1
                actual_frames = len(t_frames)
                gap_frames = expected_frames - actual_frames
                
                if gap_frames > 0:
                    print(f"      ⚠️  Missing {gap_frames} frames within tracklet (detection gaps)")
                else:
                    print(f"      ✅ Continuous (no internal gaps)")
                break
    
    # Check for gaps between tracklets
    print(f"\n🔍 GAPS BETWEEN TRACKLETS:")
    sorted_tracklets = []
    for tid in tracklet_ids:
        for t in tracklets:
            if t['tracklet_id'] == tid:
                sorted_tracklets.append((tid, t['frame_numbers'][0], t['frame_numbers'][-1]))
                break
    
    sorted_tracklets.sort(key=lambda x: x[1])  # Sort by start frame
    
    total_gap_frames = 0
    for i in range(len(sorted_tracklets) - 1):
        curr_id, curr_start, curr_end = sorted_tracklets[i]
        next_id, next_start, next_end = sorted_tracklets[i + 1]
        
        gap = next_start - curr_end - 1
        if gap > 0:
            print(f"   Tracklet {curr_id} → {next_id}:")
            print(f"      Frames {curr_end + 1} to {next_start - 1} ({gap} frames missing)")
            total_gap_frames += gap
        elif gap < 0:
            overlap = abs(gap) + 1
            print(f"   Tracklet {curr_id} ↔ {next_id}:")
            print(f"      Overlap of {overlap} frames ({curr_end - next_start + 1} to {curr_end})")
    
    if total_gap_frames == 0:
        print(f"   ✅ No gaps between tracklets")
    else:
        print(f"   ⚠️  Total gap frames: {total_gap_frames}")
    
    # Check full frame coverage
    print(f"\n📊 FULL FRAME COVERAGE ANALYSIS:")
    video_frames = 2027  # Total frames in video
    
    person_frame_set = set(frame_numbers)
    expected_frames = set(range(frame_numbers[0], frame_numbers[-1] + 1))
    
    missing_in_range = expected_frames - person_frame_set
    
    print(f"   Video total frames: {video_frames}")
    print(f"   Person frame range: {frame_numbers[0]} - {frame_numbers[-1]}")
    print(f"   Expected frames in range: {len(expected_frames)}")
    print(f"   Actual frames detected: {len(frame_numbers)}")
    print(f"   Missing frames in range: {len(missing_in_range)}")
    
    if len(missing_in_range) > 0:
        # Show where the missing frames are
        missing_list = sorted(list(missing_in_range))
        
        # Group consecutive missing frames
        gaps = []
        if missing_list:
            gap_start = missing_list[0]
            gap_end = missing_list[0]
            
            for i in range(1, len(missing_list)):
                if missing_list[i] == gap_end + 1:
                    gap_end = missing_list[i]
                else:
                    gaps.append((gap_start, gap_end))
                    gap_start = missing_list[i]
                    gap_end = missing_list[i]
            gaps.append((gap_start, gap_end))
        
        print(f"\n   📍 MISSING FRAME GAPS (top 10):")
        for start, end in gaps[:10]:
            gap_size = end - start + 1
            print(f"      Frames {start}-{end} ({gap_size} frames)")
    
    # Calculate what percentage is due to gaps vs scattered losses
    total_missing = video_frames - len(frame_numbers)
    frames_before_person = frame_numbers[0]
    frames_after_person = video_frames - frame_numbers[-1] - 1
    frames_missing_in_range = len(missing_in_range)
    
    print(f"\n🎯 MISSING FRAME BREAKDOWN:")
    print(f"   Total frames in video: {video_frames}")
    print(f"   Person detected in: {len(frame_numbers)} frames")
    print(f"   Missing: {total_missing} frames")
    print(f"")
    print(f"   Breakdown:")
    print(f"   - Before person appears (0-{frame_numbers[0]-1}): {frames_before_person} frames")
    print(f"   - After person disappears ({frame_numbers[-1]+1}-{video_frames-1}): {frames_after_person} frames")
    print(f"   - Missing within person's range: {frames_missing_in_range} frames")
    print(f"     • Gaps between tracklets: {total_gap_frames} frames")
    print(f"     • Scattered detection failures: {frames_missing_in_range - total_gap_frames} frames")
    
    print(f"\n{'='*70}")
    if total_gap_frames > frames_missing_in_range * 0.5:
        print(f"🎯 CONCLUSION: Missing frames are MOSTLY gaps between tracklets")
        print(f"   → Need to find and merge missing tracklets")
    else:
        print(f"🎯 CONCLUSION: Missing frames are MOSTLY scattered detection failures")
        print(f"   → YOLOv8n is missing detections, consider switching to YOLOv8s")
    print(f"{'='*70}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str,
                       default='/content/unifiedposepipeline/demo_data/outputs/kohli_nets')
    
    args = parser.parse_args()
    analyze_frame_coverage(args.output_dir)
