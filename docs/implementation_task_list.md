# EPModel Implementation Task List

Source spec: `/Users/davemini2/ProjectsLocal/EPModel/docs/bowen_individual_family_model_spec.md` (v1.2)

## Workflow Rules
- Update this task list whenever task status changes.
- Update the spec when behavior or parameters change.
- Add or update tests in `tests/` with every behavior change.
- Keep telemetry/docstrings/comments aligned with implemented behavior.

## Status Key
- `[ ]` Not started
- `[-]` In progress
- `[x]` Complete
- `[!]` Blocked

## Phase 0: Baseline And Guardrails
- [x] Add constants/config placeholders for generational flux settings (`INITIAL_ACTIVE_UNITS`, `NURSERY_BUFFER_UNITS`, `MATCH_INTERVAL`, `LAUNCH_BASE_P`).
- [x] Add migration-safe status enum for slot lifecycle (`ACTIVE`, `DEPARTED`, `DEAD`, `INACTIVE_BUFFER`).
- [x] Add unit tests that assert backward compatibility for existing behavior before refactor.
- [x] Expand deterministic invariant tests (telemetry cadence, interval gates, departed TX zeroing, buffer/non-buffer init behavior).
- [ ] Update docs section links if filenames or responsibilities change.

## Phase 1: Slot Lifecycle And Generational Flux
- [x] Initialize population as 8,000 active + 2,000 inactive buffer (`M=-1`, `slot_status=INACTIVE_BUFFER`) for N=10,000 profile.
- [x] Ensure births activate slots by priority: `INACTIVE_BUFFER` -> `DEAD` -> `DEPARTED`.
- [x] Ensure newborn activation creates a new identity (no resurrection of prior person state).
- [x] Ensure death transitions to `DEAD` and remains eligible for future newborn activation.
- [x] Add tests for slot transitions and birth activation priority.
- [x] Update telemetry/docs for active vs inactive accounting.

## Phase 2: Family Identity Split (FOO vs Nuclear)
- [x] Introduce `family_origin_id[N]` (immutable) and `nuclear_family_id[N]` (mutable).
- [x] Keep `family_ids` compatibility alias mapped to `nuclear_family_id` during migration.
- [x] Update all family-dependent mechanics to use `nuclear_family_id`.
- [x] Implement divorce identity flow: set `nuclear_family_id=-1` for divorced circles.
- [x] Implement remarriage flow: assign new `nuclear_family_id`, preserve `family_origin_id`.
- [x] Add tests covering marriage/divorce/remarriage identity continuity.

## Phase 3: Update Order Refactor
- [x] Reorder `update()` to match spec §5.4:
  1. `apply_income`
  2. `apply_aging`
  3. `apply_matchmaking`
  4. `apply_reproduction`
  5. `apply_launching`
  6. stress/capacity/metabolic passes
- [x] Preserve Gompertz-Makeham mortality law.
- [x] Keep two-pass matchmaking (`<=5%` primary, `>5%` secondary + penalties).
- [x] Add regression tests validating order-sensitive outcomes.

## Phase 4: Launch Calibration And Parameters
- [x] Add configurable launch baseline (`LAUNCH_BASE_P`, default 0.07).
- [x] Add stress multiplier logic for launch probability.
- [ ] Add calibration notes/tests to keep launch outcomes in realistic bands.
- [ ] Keep North America reference links in spec current.

## Phase 5: Telemetry, CSV, UI Alignment
- [x] Ensure sidebar includes `AVG AGE`, `HOMELESS %`, `BIRTHS`, `MARRIAGES`.
- [x] Ensure CSV includes `living_units`, `active_families`, `divorce_count` per log interval.
- [x] Validate homeless metric definition (`M < 100` and `S > 200` among live units).
- [x] Add tests for CSV header and telemetry fields.

## Phase 6: Consistency Cleanup
- [ ] Resolve remaining C-range mismatch to target clamp `[10, 80]`.
- [ ] Resolve C initialization mismatch to target `40 +/- 5`.
- [ ] Reconcile any section marked as implemented vs planned in spec/checklist.
- [ ] Refresh glossary/parameter table if names change in code.

## Commit Plan
- [x] Docs baseline commit (spec + task list) before code refactor.
- [x] Phase commits with tests passing at each checkpoint.
- [ ] Keep commit messages scoped by phase.

## Ready-To-Code Gate
- [x] Task list committed and pushed.
- [x] Spec v1.2 committed and pushed.
- [x] Clean plan for Phase 1 implementation prepared.
