# Implementation TODO List

## Phase 1: Fix Late-Appearance Penalty Logic

- [ ] **1.1** Read video file to get ACTUAL frame count (not 10000 default)
- [ ] **1.2** Add debug output showing actual total_frames value
- [ ] **1.3** Test with known video to verify penalty filtering removes 2 persons (87, 89)
- [ ] **1.4** Verify output shows 8 persons instead of 10

## Phase 2: Debug & Fix Stage Parsing

- [ ] **2.1** Add debug prints in run_pipeline.py stage parsing loop
- [ ] **2.2** Print each input spec and converted key for verification
- [ ] **2.3** Identify why "3b,3c,3d,4" is parsed as "5,6,7,4"
- [ ] **2.4** Verify all stages in all_stages list are correct
- [ ] **2.5** Test stage parsing with sample inputs
- [ ] **2.6** Fix parsing logic to handle shorthand correctly

## Phase 3: Fix Stage Execution Order

- [ ] **3.1** Remove any auto-enable logic that triggers Stage 3a
- [ ] **3.2** Ensure stages execute in exact requested order
- [ ] **3.3** Verify no duplicate stage execution
- [ ] **3.4** Add safeguards to prevent Stage 3a from running unexpectedly

## Phase 4: Fix Output File Handling

- [ ] **4.1** Add debug log in stage3c when saving final_crops.pkl
- [ ] **4.2** Verify file exists and has size > 0
- [ ] **4.3** Update output file list in run_pipeline.py
- [ ] **4.4** Remove "⚠️ A (not found)" errors
- [ ] **4.5** Verify final_crops.pkl shows in output file list

## Phase 5: Verify Stage 4 Execution

- [ ] **5.1** Confirm Stage 4 is in requested stage list
- [ ] **5.2** Verify Stage 4 runs after Stage 3d
- [ ] **5.3** Check for Stage 4 error handling
- [ ] **5.4** Verify WebP and HTML files are generated

## Phase 6: Validate End-to-End

- [ ] **6.1** Run full pipeline: `--stages 3b,3c,3d,4`
- [ ] **6.2** Verify execution order: 3b → 3c → 3d → 4
- [ ] **6.3** Verify person counts: 40+ → 8 → 6 (after 3d merging)
- [ ] **6.4** Verify all output files present
- [ ] **6.5** Verify no errors or warnings

---

# Implementation Notes

**Do NOT code** until user approves this analysis and TODO list.

**When implementing**:
1. One Phase at a time
2. Test after each Phase
3. Document findings in CURRENT_ISSUES_ANALYSIS.md
4. Only move to next Phase if current Phase passes tests

