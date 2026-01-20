#!/usr/bin/env python3
"""
Fix for Stage 3b: Allow merging of overlapping tracklets.

ROOT CAUSE:
  Line 118 in stage3b_group_canonical.py rejects ALL overlapping tracklets:
    if stat1['end_frame'] >= stat2['start_frame']: return False
  
  This is wrong because ByteTrack can assign new IDs while person still visible.

SOLUTION:
  Allow small overlaps (< max_overlap_frames) while still requiring temporal order.
  
EXAMPLE:
  Person #2: frames 0-457
  Person #53: frames 419-2024
  Overlap: 38 frames with IoU 0.7+ → Should merge!
"""

import sys
from pathlib import Path

# Read the file
stage3b_file = Path(__file__).parent.parent / 'stage3b_group_canonical.py'
content = stage3b_file.read_text()

# Show current problematic code
print("=" * 70)
print("CURRENT CODE (line 118):")
print("=" * 70)
print("""
def can_merge_enhanced(stat1, stat2, criteria):
    ...
    # Check 1: Temporal order (stat1 should end before stat2 starts)
    if stat1['end_frame'] >= stat2['start_frame']:
        return False
    
    gap = stat2['start_frame'] - stat1['end_frame']
    if gap > max_temporal_gap:
        return False
""")

print("\n" + "=" * 70)
print("PROPOSED FIX:")
print("=" * 70)
print("""
def can_merge_enhanced(stat1, stat2, criteria):
    ...
    # Check 1: Temporal order - allow small overlaps (ByteTrack ID switches)
    # stat1 should start before stat2
    if stat1['start_frame'] >= stat2['start_frame']:
        return False
    
    # Allow overlaps up to max_overlap_frames (negative gap = overlap)
    max_overlap = criteria.get('max_overlap_frames', 50)
    gap = stat2['start_frame'] - stat1['end_frame']  # Negative if overlap
    
    if gap > 0:  # No overlap - check temporal gap
        if gap > criteria['max_temporal_gap']:
            return False
    else:  # Overlap exists
        overlap_frames = abs(gap)
        if overlap_frames > max_overlap:
            return False
""")

print("\n" + "=" * 70)
print("CHANGES:")
print("=" * 70)
print("1. Changed first check from end_frame to start_frame")
print("   - Old: stat1 must END before stat2 STARTS (no overlap allowed)")
print("   - New: stat1 must START before stat2 STARTS (overlap OK)")
print("")
print("2. Added max_overlap_frames parameter (default 50)")
print("   - Allows ByteTrack ID switches with reasonable overlap")
print("   - Rejects unrealistic overlaps (>50 frames = likely different people)")
print("")
print("3. Split gap handling into two cases:")
print("   - Positive gap: Use existing max_temporal_gap check")
print("   - Negative gap (overlap): Use new max_overlap_frames check")
print("")
print("\n" + "=" * 70)
print("VALIDATION:")
print("=" * 70)
print("Person #2 (0-457) vs Person #53 (419-2024):")
print("  - stat1['start_frame'] = 0 < stat2['start_frame'] = 419 ✓")
print("  - gap = 419 - 457 = -38 (overlap)")
print("  - overlap_frames = 38 < max_overlap_frames = 50 ✓")
print("  - IoU = 0.7+ ✓")
print("  - Center distance = 22px ✓")
print("  → WOULD NOW MERGE! ✓✓✓")
print("")
print("\nReady to apply fix? (will modify stage3b_group_canonical.py)")
