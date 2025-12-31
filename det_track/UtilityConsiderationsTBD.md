# Utility-Based Merge Filtering - Design Document

## Problem Statement

Current pipeline processes ALL merge candidates regardless of their utility/value to the final goal: **identifying the central character occupying 75%+ of the video**.

### Issues Identified:
1. **Wasted computational effort**: Processing tiny tracklets (30-50 frames) that appear late in video
2. **No early exit**: Even when we already have 99% coverage (Person 3: 2016/2025 frames), we still run expensive ReID/grouping
3. **No ROI analysis**: Merging tracklets 86+88+95 (58 frames total, 2.9% coverage) provides minimal value

## Objective

**Primary Goal**: Identify central character with ≥75% video coverage  
**Secondary Goal**: Minimize computational cost while achieving primary goal

## Proposed Solution: Stage 3.5 - Utility Filter

Insert a new decision stage between Analysis (Stage 3) and ReID Recovery (Stage 4a).

### Decision Logic:

```
IF top_tracklet_coverage >= 75%:
    ✅ Skip to Stage 5 (Ranking)
    Reason: Already achieved goal, no recovery needed
    
ELSE:
    Filter merge candidates by utility:
    
    FOR each candidate pair (track1, track2):
        combined_frames = track1.duration + track2.duration
        combined_coverage = combined_frames / total_video_frames
        
        # Utility Criteria (ALL must pass):
        1. combined_coverage >= 0.10  (10% threshold)
        2. OR max(track1, track2) >= 100 frames (at least one is substantial)
        3. OR one tracklet is in current top 5
        
        IF criteria met:
            ✅ Accept candidate for ReID processing
        ELSE:
            ❌ Reject (low utility)
    
    IF no candidates accepted:
        ⏭️ Skip Stage 4a (ReID Recovery)
        Proceed to Stage 4b (Grouping) with existing tracklets
```

## Thresholds & Parameters

### Coverage Thresholds:
- **Early exit**: ≥75% (configurable: 70-90%)
- **Minimum utility**: ≥10% of video after merge
- **Substantial tracklet**: ≥100 frames

### Temporal Filters:
- **Late appearance penalty**: Tracklets starting after 70% of video get lower priority
- **Both-small rejection**: If both tracklets <100 frames AND neither in top 5 → reject

### Example Scenarios:

| Track 1 | Track 2 | Combined | Coverage | In Top 5? | Decision | Reason |
|---------|---------|----------|----------|-----------|----------|--------|
| 600     | 400     | 1000     | 50%      | Yes       | ✅ Accept | High utility, substantial |
| 600     | 70      | 670      | 33%      | Yes       | ✅ Accept | One tracklet substantial |
| 156     | 593     | 749      | 37%      | No        | ✅ Accept | Combined >10% threshold |
| 30      | 50      | 80       | 4%       | No        | ❌ Reject | Both small, low coverage |
| 11      | 12      | 23       | 1%       | No        | ❌ Reject | Negligible utility |

## Implementation Plan

### 1. New Function: `assess_merge_utility()`
```python
def assess_merge_utility(tracklets, candidates, total_frames, config):
    """
    Filter merge candidates by utility
    
    Returns:
        accepted_candidates: List of high-utility candidates
        rejection_reasons: Dict mapping candidate_id -> reason
        should_skip_reid: Bool (True if early exit triggered)
    """
```

### 2. Configuration Parameters:
```yaml
stage3_analyze:
  utility_filter:
    enabled: true
    early_exit_threshold: 0.75        # Skip ReID if top tracklet >= 75%
    min_combined_coverage: 0.10       # Require 10% coverage after merge
    min_substantial_frames: 100       # At least one tracklet >= 100 frames
    late_start_threshold: 0.70        # Penalize tracklets starting >70% through video
    consider_top_n: 5                 # Give priority to merges involving top 5
```

### 3. Output Enhancements:
- Log rejection reasons for each candidate
- Show utility scores in visualization
- Report computational savings (e.g., "Skipped ReID, saved 30s")

## Expected Benefits

### For kohli_nets.mp4 example:
- **Current**: Process 6 candidates, merge 2, gain 2.9% coverage
- **With utility filter**:
  - Detect Person 3 has 99.5% coverage → **Early exit at Stage 3**
  - Skip Stage 4a (ReID Recovery): Save ~2-5s
  - Skip Stage 4b (Grouping): Save ~0.5s
  - **Total savings**: ~3-5s (minimal but clean)

### For hypothetical problematic video:
- **Scenario**: Top tracklet = 400 frames (20% coverage), video = 2000 frames
- **Current**: Process 20 candidates, most are <50 frames
- **With utility filter**:
  - Reject 15 candidates (low utility)
  - Process only 5 high-value candidates
  - Potential savings: 50-70% of ReID computation time

## Trade-offs

### Pros:
- ✅ Faster pipeline when goal already achieved
- ✅ Focus effort on high-value merges
- ✅ Cleaner logic, easier to understand results
- ✅ Configurable thresholds for different use cases

### Cons:
- ❌ Might miss technically-correct merges (low-value ones)
- ❌ Adds complexity (one more decision stage)
- ❌ Requires tuning thresholds for different video types

## Decision: Implement or Not?

### Arguments FOR:
1. Aligns with stated objective (find 75%+ character)
2. Reduces waste on negligible-value operations
3. Provides early exit when goal achieved

### Arguments AGAINST:
1. Current kohli_nets.mp4 example already works (99.5% coverage)
2. Savings may be minimal for "easy" videos
3. Adds configuration complexity

### Recommendation:
**Implement as OPTIONAL feature** (disabled by default):
- Users can enable when processing large batches
- Useful for videos where tracking is fragmented
- Minimal impact when disabled (just a config check)

---

**Status**: AWAITING DECISION  
**Next Steps**: User approval before implementation
