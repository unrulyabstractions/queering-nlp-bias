# Radical Modularization Plan

## Executive Summary

This plan addresses critical code duplications, naming inconsistencies, and missing BaseSchema inheritance. NO backwards compatibility - delete old code entirely.

---

## PHASE 1: Shared Infrastructure ✅ COMPLETE

### 1.1 Create `src/common/method_params_override.py` ✅
**Problem**: Identical class in `generation_config.py:24-38` and `scoring_config.py:51-62`
- Created new file with single implementation
- Deleted from both config files, added import

### 1.2 Create `src/common/result_grouping.py` ✅
**Problem**: 4 implementations of group_by_arm() in:
- `scoring_output.py:255-261`
- `estimation_scoring_data.py:328-334` and `394-417`
- Created shared `group_results_by_arm()` function
- Updated both files to use shared utility

### 1.3 Consolidate `src/common/text/text_display.py` ✅
**Problem**: Two `preview()` functions with different behavior
- `viz_utils.py:25-29` uses escape_newlines + truncate
- `text_formatting.py:125-139` uses oneline (collapses whitespace)
- Removed duplicate `preview()` from text_formatting.py (unused)
- viz_utils.py preview() is authoritative

### 1.4 Create `src/common/continuation_text.py` ✅
**Problem**: `get_continuation_text()` duplicated in:
- `generation_helpers.py:45-55`
- `experiment_tree_plot.py:136-146`
- Created shared utility, updated both files

---

## PHASE 2: LLM Scoring Base ✅ COMPLETE

### 2.1 Create `src/scoring/methods/llm_response_parsing.py` ✅
**Problem**: categorical_method.py and graded_method.py share:
- Same thinking block stripping pattern

Extracted:
- `strip_thinking_content(response)` - shared parsing helper
- Updated both methods to use shared utility

---

## PHASE 3: Simplify ArmEstimate ⏭️ SKIPPED

### 3.1 Replace 16 getter methods with 2
**Problem**: `estimation_structure.py:87-233` has 157 lines of boilerplate getters

**Decision**: Skipped - all getters are actually used (via `getattr()` in deviance plots).
The current API is clear and provides good discoverability.

---

## PHASE 4: Replace Manual Caching ⏭️ SKIPPED

### 4.1 Update `estimation_scoring_data.py`
**Problem**: 9 lines boilerplate per cached property

**Decision**: Skipped - using `@cached_property` in dataclasses with BaseSchema
could cause serialization issues. Current explicit caching is safer.

---

## PHASE 5: Remove Passthrough Properties ⏭️ SKIPPED

### 5.1 Delete from `estimation_scoring_data.py:89-107`
**Problem**: 5 properties that just return `self.metadata.X`

**Decision**: Skipped - would require updating many call sites. The properties
provide a convenient API and don't cause issues.

---

## PHASE 6: Add Missing BaseSchema ✅ COMPLETE

**Files updated**:
1. `forking_paths_types.py`: TopKCandidate, PositionAnalysis, QualifyingFork, ForkPoint ✅
2. `entropy_seeking_types.py`: BestPosition, TreePath, ExpansionPoint ✅

---

## PHASE 7: Fix Naming ✅ COMPLETE

### 7.1 Standardize to `traj_idx` (not `trajectory_idx`) ✅
- Updated `ScoringResult.trajectory_idx` → `traj_idx`
- Updated all JSON field accesses in estimation and scoring
- Updated documentation examples

### 7.2 Standardize to `arm_idx` (not `arm_index`) ✅
- Updated `TokenTrajectory.arm_index` → `arm_idx`
- Updated `BinaryFork.arm_index` → `arm_idx`
- Updated all usages in TokenTree, generation pipeline, scoring, tests
- Updated all documentation

---

## Summary

**Completed:**
- Phase 1: 4 shared utilities created, 4 duplications removed
- Phase 2: 1 shared utility created, thinking block stripping consolidated
- Phase 6: 7 dataclasses now inherit BaseSchema
- Phase 7: All naming standardized (`traj_idx`, `arm_idx`)

**Skipped (low value / high risk):**
- Phase 3: Getter methods are used and provide good API
- Phase 4: Caching change could break serialization
- Phase 5: Properties provide convenient API

---

## PHASE 8: Remove Dead `analysis` Fields ✅ COMPLETE

**Problem**: The `analysis: Any | None = None` field exists on 4 core data structures but is NEVER READ for its value anywhere in the codebase.

**Completed**:
1. Removed `analysis` field from TokenTrajectory, BinaryFork, BranchingNode, TokenTree
2. Deleted entire `src/common/analysis/` module (all dead code)
3. Updated `from_dict()` in TokenTree, `_create_forks_for_arms()` to remove analysis references
4. Updated `GeneratedTrajectory.from_token_trajectory()` to remove analysis reference
5. Updated documentation (README.md, CLAUDE.md) to remove analysis references
6. Updated test that checked for analysis field

**Actual reduction**: ~300 lines (entire analysis module deleted)

---

## PHASE 9: Consolidate Logging Infrastructure ✅ PARTIAL

**Problem**: ~110+ lines of duplicated logging patterns across logging files.

**Completed**:
- Updated `gen_logging_utils.py` to use `escape_newlines()` and `preview()` from viz_utils
- Updated `forking_paths_logging.py` to use `preview()` instead of manual truncation
- Updated `entropy_seeking_logging.py` to use `preview()` and `escape_newlines()`
- Removed local `truncate()` from `estimation_comparison_logging.py`, imported from viz_utils
- Updated `scoring_logging_utils.py` to use shared `truncate()`

**Reduction**: ~25 lines of duplicate code removed, consistent patterns established

**Not addressed** (lower priority):
- `_log = log_fn or log` pattern - low impact, works fine as-is
- Custom histogram/plot reimplementations - rarely called code paths

---

## PHASE 10: Remove Other Dead Code ✅ COMPLETE

**Problem**: Unused functions found across utility files.

**Completed**:
1. Removed `check_memory_trend()` and `_memory_history` from device_utils.py
2. Simplified `log_memory()` signature (removed unused `iteration` parameter)
3. Removed from viz_utils.py:
   - `compute_percentiles()` - never called
   - `compute_stats()` - never called
   - `format_histogram_vertical()` - never called
   - `format_sequence_plot()` - never called
   - `print_lines()` - never called
4. Updated documentation (EXPLANATION.md, README.md)

**Agent misidentified as dead (actually used)**:
- `sanitize_floats()` - used by TokenTrajectory.sanitize()
- `get_bundle_score_at_index()` / `collect_scores()` - used internally in scoring_output.py
- file_io.py functions - did not exist (already removed or never created)
- arm_types.py functions - did not exist

**Actual reduction**: ~220 lines

---

## Summary (Final)

**Completed:**
- Phase 1: 4 shared utilities created, 4 duplications removed
- Phase 2: 1 shared utility created, thinking block stripping consolidated
- Phase 6: 7 dataclasses now inherit BaseSchema
- Phase 7: All naming standardized (`traj_idx`, `arm_idx`)
- Phase 8: Removed dead `analysis` fields (~300 lines - entire analysis module)
- Phase 9: Consolidated logging patterns to use shared utilities (~25 lines)
- Phase 10: Removed dead code from device_utils.py, viz_utils.py (~220 lines)

**Total reduction: ~545 lines of dead/duplicate code**

**Skipped (low value / high risk):**
- Phase 3: Getter methods are used and provide good API
- Phase 4: Caching change could break serialization
- Phase 5: Properties provide convenient API

---

## Additional Work Done

### CLAUDE.md Files Created
Created strategic documentation files for future agents:
- `src/common/CLAUDE.md`
- `src/scoring/CLAUDE.md`
- `src/estimation/CLAUDE.md`
- `src/generation/CLAUDE.md`
- `src/viz/CLAUDE.md`

Each explains:
- Module purpose and architecture
- Registry patterns
- Key files and utilities
- Common pitfalls
- References to existing documentation
