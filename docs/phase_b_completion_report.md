---
tags: [model-bt, report]
status: Phase B built — every automated exit criterion passes; one human review pending
date: 2026-10-06
plan: docs/implementation_plan_phase_b.md (approved 2026-10-06)
spec: docs/bowen_agent_model_spec_v2.md, v2.0 revision 10 (526 IDs after the plan's changes)
---

# Phase B — completion report

## Status

**Phase B is built but not yet done.** All fourteen automated exit criteria pass, and each is
shown to fail under a mutation (17 of 17 mutations, `docs/phase_b_mutation_record.md`). One
criterion cannot be a test: the owner's end-to-end read of the rendered trace (plan §7). Until
that read is done and recorded, Phase B is not declared complete.

The suite is 291 tests, all passing: `python3 -m pytest tests/`. The frozen grid engine's 37
tests are among them and pass unchanged (`M13.2`).

**What this phase shows, and what it does not.** It shows the plumbing works: the objects, the
nine-step weekly loop, the standing load, the base appraisal, the event record and its
deliveries, the M8 predicate, the scripted source, the log, and the renderer. It tests one
model claim, `M4.A.2` (gate G3), as a direction between two arms. Every number in a run comes
from invented constants, and nothing in this phase is a finding about families.

## Exit criteria

From `M13`'s Phase B *Done when* cell, as mapped in plan §2.

| # | Criterion | Status | Evidence |
|---|---|---|---|
| G1 | A scripted 40-week trace runs | done | `tests/bowen/test_phase_b_gate.py::test_m13_phase_b_scripted_40_week_trace_runs` |
| G2 | The renderer emits a trace conforming to `M16.C.2`; `M16.T.1` passes | done | `tests/bowen/test_render.py::test_m16t1_trace_renders_scripted_run`, `::test_m16t1_effects_are_reported_in_reader_units` |
| G3 | A `TRIGGER` on a dormant family-of-origin tie moves anxiety with no contact | done | `tests/bowen/test_phase_b_gate.py::test_m4a2_g3_trigger_on_dormant_tie_moves_anxiety_without_contact` — two arms; both clauses mutation-proved separately |
| G4 | `M11.D.1` engine purity | done | `tests/bowen/test_engine_purity.py::test_m11d1_engine_has_no_io` |
| G5 | `M11.D.3` config strictness | done | `tests/bowen/test_config.py::test_m11d3_config_rejects_unknown_key` (and seven sibling tests) |
| G6 | `M11.D.5` determinism | done | `tests/bowen/test_determinism.py::test_m11d5_same_seed_same_log` — in process, and across processes with different `PYTHONHASHSEED` |
| G7 | `M11.D.6` a second run is clean | done | `tests/bowen/test_determinism.py::test_m11d6_second_run_is_clean` |
| G8 | `M11.D.7` no production paths in tests | done | `tests/bowen/test_test_hygiene.py::test_m11d7_no_production_paths_in_tests`; guard in `tests/bowen/conftest.py` |
| G9 | `M16.T.2` the engine writes nothing | done | `tests/bowen/test_engine_purity.py::test_m16t2_engine_writes_nothing` |
| G10 | `M16.T.3` the sink is a pure observer | done | `tests/bowen/test_log.py::test_m16t3_sink_does_not_change_results` — vacuous in Phase B; must be rerun at the end of C and D |
| G11 | `M16.T.5` byte-identical logs, header included | done | `tests/bowen/test_log.py::test_m16t5_same_seed_same_log_with_header` |
| G12 | `M11.D.17` the register has no orphans | done | `tests/test_spec_consistency.py::test_m11d17_register_has_no_orphans` |
| G13 | `M11.D.22` the engine cannot see its arm | done | `tests/bowen/test_engine_purity.py::test_m11d22_engine_modules_do_not_import_tests_or_readouts` |
| G14 | `M13.2` the frozen engine stays green | done | `tests/test_simulator.py` (37 tests) |
| D6 | The register matches the object fields | done | `tests/bowen/test_register.py::test_m14a_register_matches_object_fields` |
| §7 | The rendered trace reads correctly end to end | **not done** | owner's read pending: `docs/review/phase_b_trace_seed7.md` and `docs/review/phase_b_trace_seed7_nadia.md` |
| §7 | The register's classifications are right | done | reviewed by the owner at plan approval, 2026-10-06 |
| `M14.1` | Spec coverage report | done | `docs/spec_coverage.md` — 526 IDs: 112 done, 27 partial, 387 not done; kept current by `tests/bowen/test_spec_coverage.py` |

## The plan's decisions, and where each landed

| | Decision | Where |
|---|---|---|
| D1 | `M4.C.1` only, in Phase B | `M13` amended; `src/bowen/engine/appraise_base.py`. Knowingly fails `M4.C.2` until Phase C |
| D2 | `M6.I.6` held back | `M4.G.2a`; `src/bowen/engine/invariants.py` keeps it as a check that raises if enabled |
| D3 | Latency in whole ticks, at least one | `M3.C.2`; enforced in `Relationship` and `Delivery` |
| D4 | `sex` and `household_id` | `M1.A.21`, `M1.A.22`; a `Sex` column in `M2.A` and `M2.A.0i` (added at step 5) |
| D5 | The reduced instance; Ravi 39 | `M2.3a`; `config/bowen/family_reduced.md` |
| D6 | Register in the spec, with a `Phase` column | `M14.A.3`, `M14.A.4`; two tests tie it to the spec and to the code |
| D7 | Keyed draws built, unused | `src/bowen/engine/draws.py`; `per_hop_fidelity` stands in for the draw |
| D8 | "M1 objects" = state and identifiers | `src/bowen/engine/objects.py`; Appendix B's split |
| D9 | The spec wins over the explainer | no Phase B content was built from the six drifted entries |

## Owner decisions taken during the build

- **Sex** for Ravi (male), Nadia and Pia (female); the rest from `M2.A`'s relationship words (step 5).
- **`M4.A.5`'s "must not swamp"** read at family and nuclear-household level; the one-tie peripheral
  exception (Sofia) is recorded in its test (step 8).
- **No change to the invented values** before the freeze (step 10).

## Corrections found while building

Each was found by a test, a mutation, or reading output, and is fixed and covered.

1. **The register missed `Relationship.id`** — caught by the register-to-code test (step 2).
2. **Triangle `bound_anxiety` had a Phase B writer it should not have** — nothing binds anxiety
   into a triangle before Phase C; moved (step 8).
3. **A one-week TRIGGER would never have fired** — it applies at step 2, after the standing
   load, so its window now starts the following week (step 9).
4. **G3's no-contact clause missed a delivered trigger** — a mutation passed; the clause was
   widened and each clause is now proved alone (step 10).
5. **Witnesses were hit harder than the person addressed** — found by reading the first
   rendered trace; a witness is now bounded by the event's own edge. A code rule, not a
   frozen constant (step 12).
6. **A mutation script left injected faults in the working tree** — zsh does not split an
   unquoted variable; caught at once (24 tests red), reverted by hand, and replaced by
   `tools/mutation_gate.py`, which restores in `finally` (step 11).

## The `learning-qa` review, 2026-10-06

A cold review of the Phase B diff against the failure-pattern catalogue (P1–P37) found eleven
issues and raised three suspicions. All were fixed in this session except one suspicion, which is
intended behaviour.

| # | Pattern | Finding | Disposition |
|---|---|---|---|
| 1 | P21/P6 | A stressor's duration had no effect, but `M1.F.6` was marked done | Owner: a stressor acts once in Phase B. `M1.F.6` marked partial; the script and the trace say so |
| 2 | P6 | A move reached a person across a cut-off tie | Refused: `InactiveTie` in visibility (`M1.B.3`); test added. `REDUCE_CUTOFF` will need its own path in Phase C |
| 3 | P24/P37 | The mutation tool counted any non-zero exit as proof; G7's mutation was caught by another guard | Exit code 1 and a failure required; each gate names its intended assertion and the tool checks the failure fired there; G7 re-mutated so its own assertion catches it. The tool was shown to reject a collection error and a wrong-assertion failure |
| 4 | P37 | The tool's clean-tree check printed only, missed ignored files and the repo root | Whole-repo snapshot, ignored and untracked included, before and after; a difference fails the run |
| 5 | P19/P2 | The renderer dropped unknown mechanisms and fields, and claimed invariants without counting them | Unknown records, mechanisms and fields raise; the invariant sentence reconciles with the records |
| 6 | P6 | The mutation record named no commit | The record is stamped with the commit and a dirty flag |
| 7 | P2 | `--view` of an unknown person gave an empty trace | Refused in the renderer and the CLI |
| 8 | P2/P6 | A failed run left a complete-looking log | The sink writes `.partial` and renames only on success |
| 9 | P28 | The write guard missed moves and deletes, and nothing fingerprinted real files | Guard extended to replace, rename, remove, unlink, rmdir, truncate and `shutil.rmtree`; a session-wide fingerprint of config, model source, spec and tests in `tests/conftest.py` |
| 10 | P31 | Coverage took one file per test name and hid tests named for no ID | Every file listed; the 56 tests named for no ID are listed in the report |
| 11 | — | (review commit count said 20; it is 18) | noted |
| s1 | — | The G4 scan missed file *reads* | `load`, `loadtxt`, `fromfile`, `read_text`, `importlib` and similar added |
| s2 | P3 | Hardening ignored multi-target withdrawals | A move counts on a tie when it goes from one member to the other, whoever else it addresses |
| s3 | — | Acute anxiety below the floor is raised to the floor, not decayed | Intended: `M1.A.7a` makes chronic anxiety the floor acute anxiety cannot go below |

## Invented constants, frozen 2026-10-06

All `[I]`. Snapshot `config/bowen/constants_frozen.md`; change log `config/bowen/constants_changes.md`
(empty). Any later change that alters an acceptance outcome must be logged and reported as post-hoc
(`M10.B.4`).

| Key | Value | Unit | Spec |
|---|---|---|---|
| `fast_tick_weeks` | 1 | week | `M3.A.1` |
| `slow_tick_fast_ticks` | 52 | fast tick | `M3.B.1` |
| `invariant_tolerance` | 1e-09 | anxiety unit | `M6.1` |
| `spouse_basic_level_tolerance` | 1 | basic-level point | `M2.A.0e` |
| `chronic_anxiety_fixation_age_years` | 12 | year of age | `M2.A.0a` |
| `per_hop_fidelity` | 0.8 | fraction kept per private hop | `M1.F.4` |
| `standing_load_gain` | 0.2 | anxiety per tick per bond-energy unit per functional-level unit | `M4.A.1` |
| `interactive_standing_fraction` | 0.5 | fraction | `M4.A.3` |
| `functional_level_floor` | 1 | functional-level point | `M4.C.1` |
| `acute_decay_rate` | 0.2 | fraction of excess per tick | `M1.A.8` |
| `route_damping` | 0.5 | gain per neutral third | `M1.F.3` |
| `hardening_run_length` | 3 | consecutive moves | `M4.G.1` |
| `bond_energy_decay_rate` | 0 | fraction per tick | `M1.B.4` |
| `tension_activation_threshold` | 5 | anxiety above the chronic floor | `M1.C.3` |
| `involvement_membership_threshold` | 0.5 | involvement units | `M1.A.12` |

The family's tie values and the script's intensities are invented too; each file says so and
explains its choices. Formulas the spec leaves open — tie tension, involvement, which trios are
triangles, the witness rule, the standing-load shape — are stated and graded `[I]` in their modules.

## Departures from the plan

- **Constants entered with their mechanisms**, not all at step 1, so no value was invented
  ahead of the code that reads it; all were in place before the freeze.
- **Module layout.** Perception sits in `appraise_base.py`. Added: `engine/state.py`,
  `engine/params.py`, `engine/event_effects.py`, `engine/recompute.py`, `scenario/params.py`,
  `scenario/assemble.py`, `scenario/header.py`, `scenario/event_kinds.py`, `run.py`, and
  `tools/mutation_gate.py` and `tools/spec_coverage.py`.
- **Test locations.** G3's test is in `test_phase_b_gate.py` (its planned name was taken by the
  unit test); `test_m14a_register_matches_object_fields` is in `tests/bowen/test_register.py`,
  because it imports model code.
- **The suite-size guard** now counts what pytest collects, not `def test_*` (a TODO item, closed).

## Open, and what Phase C must settle first

From the external review (plan §6) and the build:

- **`M6.I.6`** restated with a stock-and-flow table (review A3) — Phase C cannot assert it otherwise.
- **`M4.B.2`** scoped row by row (review A4) before the policy is planned.
- **Criteria audit** (`M11.1d`, review A1) and **phase placement** of `M11.C.17`, `.18`, `.20` (A5).
- **The cancellation premise** in explainer §17.3 and `_STATUS.md` (A2); explainer drift (D).
- **Which event kinds flip sign** by source position (`M1.F.2`), decided with `I-POSITION`.
- **The witness rule** replaced by `M4.C.9`'s.
- **`M16.T.3`** rerun at the end of C and D.
- **Before Phase D:** `M6.3` (death), the transmission-to-estimator handover (A6), sex for Leo and
  Dr Halim, Teodor/Ana's spouse tolerance, Teodor's and Sofia's family-of-origin ties (C).

## Adjacent issues, not fixed

- ~~`requirements.txt` does not list pytest; no `.github/workflows/tests.yml`.~~ *Fixed 2026-10-06:*
  pytest pinned to 8.x, and the workflow runs the suite on Python 3.11, 3.12 and 3.13, each first run
  from a fresh virtualenv with only `requirements.txt` (NumPy 2.4–2.5 there, 1.26 here; all then-282 pass).
  The 3.11 run found a real defect — a `MappingProxyType` dataclass default that 3.11 rejects — now fixed.
- A stray top-level `tests` package in the system Python's site-packages shadows `import tests…`.
- `docs/implementation_task_list.md` still tracks the frozen grid engine.
- The two stray `sim_audit.csv` copies from the frozen engine (only `src/` has one now; the repo-root
  copy CLAUDE.md mentions does not exist).
- ~~The step-11 mutation-script failure is a candidate for the learnings catalogue.~~ *Added as P37,
  with the multi-clause corollary from step 10 (`claude-standards` `c5c7dc6`).*

## The human read

Generate and read:

```
python3 -m src.bowen.run --seed 7 --log runs/phase_b_seed7.jsonl --trace docs/review/phase_b_trace_seed7.md
```

Both traces are already in `docs/review/`. Anything that reads wrong — an effect that should not
follow, a witness who should not be there, a missing line — is a finding for Phase B, not Phase C.
