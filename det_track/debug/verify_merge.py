#!/usr/bin/env python3
"""
Verify that Person #2 and #53 were successfully merged.
"""

import numpy as np
import sys
from pathlib import Path

def verify_merge(output_dir):
    output_path = Path(output_dir)
    
    print("=" * 70)
    print("🔍 VERIFYING PERSON #2 + #53 MERGE")
    print("=" * 70)
    
    canonical_file = output_path / 'canonical_persons.npz'
    canonical_data = np.load(canonical_file, allow_pickle=True)
    persons = canonical_data['persons']
    
    print(f"\n📊 Total canonical persons: {len(persons)}")
    
    # Look for the merged person that covers frames 0-2024
    # It should contain tracklets [2, 53, 68]
    
    merged_person = None
    for p in persons:
        start = p['frame_numbers'][0]
        end = p['frame_numbers'][-1]
        tracklet_ids = p.get('original_tracklet_ids', [])
        
        # Check if this person covers the expected range
        if start == 0 and end >= 2020 and end <= 2030:
            merged_person = p
            print(f"\n✅ FOUND MERGED PERSON!")
            print(f"   Person ID: {p['person_id']}")
            print(f"   Frame range: {start} - {end} ({len(p['frame_numbers'])} frames)")
            print(f"   Original tracklet IDs: {tracklet_ids}")
            print(f"   Num tracklets merged: {p.get('num_tracklets_merged', 'N/A')}")
            
            # Check if it contains the expected tracklets
            expected_tracklets = {2, 53, 68}
            actual_tracklets = set(tracklet_ids)
            
            if expected_tracklets.issubset(actual_tracklets):
                print(f"\n   ✅✅✅ MERGE CONFIRMED!")
                print(f"   Contains all expected tracklets: {expected_tracklets}")
            else:
                print(f"\n   ⚠️ Partial match")
                print(f"   Expected: {expected_tracklets}")
                print(f"   Actual: {actual_tracklets}")
            break
    
    if not merged_person:
        print(f"\n❌ Could not find merged person covering frames 0-2024")
        print(f"\nAll persons covering frame 0:")
        for p in persons:
            if p['frame_numbers'][0] == 0:
                print(f"   Person {p['person_id']}: frames {p['frame_numbers'][0]}-{p['frame_numbers'][-1]}, tracklets {p.get('original_tracklet_ids', [])}")
    
    # Check merge result: Person #2 should exist (as merged person), Person #53 should NOT
    person_ids = [p['person_id'] for p in persons]
    
    print(f"\n📋 PERSON ID CHECK:")
    person_2_exists = 2 in person_ids
    person_53_exists = 53 in person_ids
    
    if person_2_exists and not person_53_exists:
        print(f"   ✅ Person #2 exists (merged person)")
        print(f"   ✅ Person #53 absorbed into Person #2")
    else:
        if not person_2_exists:
            print(f"   ⚠️ Person #2 does not exist (should be the merged result)")
        if person_53_exists:
            print(f"   ⚠️ Person #53 still exists separately (merge failed)")
    
    # Successful merge = found merged person + Person #2 exists + Person #53 gone
    merge_successful = merged_person and person_2_exists and not person_53_exists
    
    print(f"\n" + "=" * 70)
    print(f"SUMMARY: {'✅ MERGE SUCCESSFUL!' if merge_successful else '⚠️ MERGE INCOMPLETE'}")
    print(f"=" * 70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str,
                       default='/content/unifiedposepipeline/demo_data/outputs/kohli_nets')
    
    args = parser.parse_args()
    verify_merge(args.output_dir)
