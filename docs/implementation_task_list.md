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
- [ ] Add constants/config placeholders for generational flux settings (`INITIAL_ACTIVE_UNITS`, `NURSERY_BUFFER_UNITS`, `MATCH_INTERVAL`, `LAUNCH_BASE_P`).
- [ ] Add migration-safe status enum for slot lifecycle (`ACTIVE`, `DEPARTED`, `DEAD`, `INACTIVE_BUFFER`).
- [ ] Add unit tests that assert backward compatibility for existing behavior before refactor.
- [ ] Update docs section links if filenames or responsibilities change.

## Phase 1: Slot Lifecycle And Generational Flux
- [ ] Initialize population as 8,000 active + 2,000 inactive buffer (`M=-1`, `slot_status=INACTIVE_BUFFER`) for N=10,000 profile.
- [ ] Ensure births activate slots by priority: `INACTIVE_BUFFER` -> `DEAD` -> `DEPARTED`.
- [ ] Ensure newborn activation creates a new identity (no resurrection of prior person state).
- [ ] Ensure death transitions to `DEAD` and remains eligible for future newborn activation.
- [ ] Add tests for slot transitions and birth activation priority.
- [ ] Update telemetry/docs for active vs inactive accounting.

## Phase 2: Family Identity Split (FOO vs Nuclear)
- [ ] Introduce `family_origin_id[N]` (immutable) and `nuclear_family_id[N]` (mutable).
- [ ] Keep `family_ids` compatibility alias mapped to `nuclear_family_id` during migration.
- [ ] Update all family-dependent mechanics to use `nuclear_family_id`.
- [ ] Implement divorce identity flow: set `nuclear_family_id=-1` for divorced circles.
- [ ] Implement remarriage flow: assign new `nuclear_family_id`, preserve `family_origin_id`.
- [ ] Add tests covering marriage/divorce/remarriage identity continuity.

## Phase 3: Update Order Refactor
- [ ] Reorder `update()` to match spec §5.4:
  1. `apply_income`
  2. `apply_aging`
  3. `apply_matchmaking`
  4. `apply_reproduction`
  5. `apply_launching`
  6. stress/capacity/metabolic passes
- [ ] Preserve Gompertz-Makeham mortality law.
- [ ] Keep two-pass matchmaking (`<=5%` primary, `>5%` secondary + penalties).
- [ ] Add regression tests validating order-sensitive outcomes.

## Phase 4: Launch Calibration And Parameters
- [ ] Add configurable launch baseline (`LAUNCH_BASE_P`, default 0.07).
- [ ] Add stress multiplier logic for launch probability.
- [ ] Add calibration notes/tests to keep launch outcomes in realistic bands.
- [ ] Keep North America reference links in spec current.

## Phase 5: Telemetry, CSV, UI Alignment
- [ ] Ensure sidebar includes `AVG AGE`, `HOMELESS %`, `BIRTHS`, `MARRIAGES`.
- [ ] Ensure CSV includes `living_units`, `active_families`, `divorce_count` per log interval.
- [ ] Validate homeless metric definition (`M < 100` and `S > 200` among live units).
- [ ] Add tests for CSV header and telemetry fields.

## Phase 6: Consistency Cleanup
- [ ] Resolve remaining C-range mismatch to target clamp `[10, 80]`.
- [ ] Resolve C initialization mismatch to target `40 +/- 5`.
- [ ] Reconcile any section marked as implemented vs planned in spec/checklist.
- [ ] Refresh glossary/parameter table if names change in code.

## Commit Plan
- [ ] Docs baseline commit (spec + task list) before code refactor.
- [ ] Phase commits with tests passing at each checkpoint.
- [ ] Keep commit messages scoped by phase.

## Ready-To-Code Gate
- [ ] Task list committed and pushed.
- [ ] Spec v1.2 committed and pushed.
- [ ] Clean plan for Phase 1 implementation prepared.
