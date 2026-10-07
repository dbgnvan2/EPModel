---
tags: [model-bt, plan]
status: APPROVED 2026-10-06 — all nine recommendations (D1–D9) adopted as written
date: 2026-10-06
spec: docs/bowen_agent_model_spec_v2.md, v2.0 revision 10, approved 2026-10-06
scope: Phase B only
---

# Implementation plan — Phase B

## 0. Scope and how to read this

**Phase B only**, by the owner's decision at approval of revision 10 (2026-10-06), following the external
review's recommendation 9 (`docs/REVIEW_spec_rev10_2026-09-23.md`). Phases C–E are planned after Phase B
runs, against what it exposes.

Phase B, from `M13`'s table:

> **Builds:** M1 objects; M3 clocks and update order; M4.A standing load including M4.A.5; M4.B, M4.E and
> M4.G; M1.F event record including M1.F.9; M8 the live-position predicate; `ScriptedSource`; M16.A–M16.C,
> the run log and the deterministic renderer. **No policy** — a fixed script drives it. Reduced instance per
> M2.3.
>
> **Done when:** a scripted 40-week trace runs; the renderer emits a trace conforming to M16.C.2 and M16.T.1
> passes; a `TRIGGER` on a dormant family-of-origin tie moves anxiety with no contact. M11.D.1, M11.D.3,
> M11.D.5, M11.D.6, M11.D.7, M16.T.2, M16.T.3 and M16.T.5 pass. M11.D.17 passes against the register as
> filled at plan time, and M11.D.22 passes.

**Order of this document.** §1 lists the decisions this plan needs from the owner before code starts — the
plan is not buildable until they are answered. §2 is the Phase B gate, criterion by criterion, each with its
test and the mutation that proves the test can fail. §3 lists every other requirement Phase B builds, with its
test. §4 is the package layout. §5 is the build order. §6 carries the external review's findings to the
phases they block. §7 lists what cannot be tested in code. §8 lists adjacent issues found and not fixed.
Appendix A is the `M14.A` register draft; Appendix B says which `M1` requirements Phase B satisfies.

**Conventions**, from the spec (§0.4) and the global rules: test names embed the lowercased spec ID; every
function satisfying a requirement carries a `Purpose / Spec / Tests` docstring; `docs/spec_coverage.md` is
generated at completion (`M14.1`).

---

## 1. Decisions needed before code

Each item is a gap or conflict in the approved spec that Phase B cannot build around. Each has a
recommendation. The plan below assumes the recommendation; if the owner decides otherwise, the named sections
change.

### D1. What a delivered event does in Phase B — appraisal timing

`M13` places `M4.C` (appraisal) in Phase C. The proposal's Phase B row includes "perceive–appraise–select–act",
but the spec wins (§0.2). With no appraisal, a delivered event changes no one's anxiety, so Phase B's only
anxiety effects would come from the standing load and decay, and `M16.T.1`'s "effects beside causes" would
have little to show.

**Recommendation:** pull **`M4.C.1` only** into Phase B — the base appraisal,
`intensity × conductance / functional_level`, with `route`, `source_position` and `fidelity` applied. `M4.C.2`
(the gain function) and the rest of `M4.C` stay in Phase C. This knowingly leaves Phase B failing `M4.C.2`
("a plain product is a failing implementation") until Phase C, and Phase B's tests **MUST NOT** assert any
appraisal magnitude. Same shape as `M13.4`, which pulled `M17.A.1` forward. Requires a one-line amendment to
`M13`'s Phase B *Builds* cell.

### D2. Which invariants Phase B asserts — `M6.I.6` is unsatisfiable as worded

`M4.G.2` (Phase B) requires every `M6` invariant to be asserted at the end of every fast tick. Review finding
A3: `M6.I.6` ("anxiety is conserved and redirected, never destroyed") fails on every tick, because `M1.A.8`
and `M3.D.1` step 9 decay acute anxiety toward the floor with no sink named, and `M4.C.1` creates anxiety per
event.

**Recommendation:** Phase B asserts `M6.I.1`, `M6.I.2`, `M6.I.3`, `M6.I.4`, `M6.I.5`, `M6.I.7` and `M6.I.8`
(most hold trivially in Phase B because nothing moves the quantities they govern, which the test report must
say). **`M6.I.6` is not asserted until it is restated** at the scope of review A3 — conservation of *bound*
quantities across rerouting and transfer events — with A3's stock-and-flow table. The restatement is a Phase C
prerequisite (§6). The invariant module carries `M6.I.6` as a named, disabled check that raises if enabled,
so it cannot be silently forgotten.

### D3. Latency on a weekly clock — sub-week durations

Review finding A7. `M3.A.1` fixes the fast tick at one week; `M1.B.11` and `M3.C.1` make latency a per-edge
property; explainer §4.7 and §8 give latencies "within hours" and calibration durations of "3 days" and
"2 hours".

**Recommendation:** latency is a **whole number of fast ticks, minimum 1**. An event emitted at step 8 of
tick *t* on an edge with latency *L* is delivered at step 2 of tick *t + L*. Sub-week durations are dropped
as calibration targets, and the explainer's §8 entries are restated in ticks before Phase C. `M5.D.6`'s
"next day" follow-up is a Phase C question (§6). Intra-tick scheduling is not built.

### D4. Fields the spec reads and never defines

Review finding B, "Undefined fields". Two of the three bite in Phase B:

- **Household / co-residence.** `M1.F.1b` makes the witness set a function of co-residence, and the
  visibility component (`M3.E.1`) is Phase B.
- **Sex.** Read by `M2.A.0f`, `M2.A.0g`, `M7.E.1c` and `M11.C.25` — all Phase C or later, but `Person` is
  built in Phase B.

**Recommendation:** add `household_id` and `sex` to `Person` as declared data in the family config, with no
mechanism reading `sex` in Phase B. Add both to `M1.A` as requirements. (`expectations`, the third undefined
field, is Phase D and is left.)

### D5. The reduced instance (`M2.3`) — composition and data

`M2.3` requires the nuclear four (Ravi, Marta, Nadia, Pia) plus their family-of-origin ties. The exit
condition requires a `TRIGGER` on a **dormant** family-of-origin tie. Review finding C found three data
defects in `M2.A`; one bites here.

**Recommendation:**

| | Choice |
|---|---|
| Members | Ravi, Marta, Nadia, Pia (active); Ana, Sofia, Bruno (bond-energy holders, scripted only) |
| Ties | Ravi–Marta; Ravi–Nadia; Ravi–Pia; Marta–Nadia; Marta–Pia; Marta–Ana; Ravi–Sofia; Ana–Bruno (cut off) |
| The dormant tie for the exit test | **Ana–Bruno** — cut off, no events, high bond energy (`M1.B.3`). Bruno is in the instance only to anchor that tie |
| Spouse tolerance | Ravi 38 / Marta 40 violates `M2.A.0e`'s ±1. **Change Ravi to 39.** Teodor/Ana (34/37) are outside the reduced instance and are left for Phase D |
| `structural_importance` | **Omitted from the Phase B config.** `M1.A.13a` derives it; `M2.A`'s hand-assigned labels (central, shadow, head of household, peripheral) are not tiers (review B) |
| `chronic_anxiety` | Supplied per `M2.A.0a` and held constant: the slow tick first fires at week 52, so `M1.A.7a`'s derivation never runs in a 40-week trace. The `M7.B.1`/`M1.A.7a` conflict (review B) is a Phase D item |

Whether Ana–Bruno's spike should reach Marta in 40 weeks is **not** part of the exit test: the test asserts
Ana's anxiety moves with no event on the tie, which is `M4.A.2`'s claim.

### D6. Where the `M14.A` register lives, and one column it lacks

`M14.A.3` leaves the register's rows "for the owner to fill when the implementation plan is written", and
`M11.D.17` must pass against it in Phase B. Appendix A is the draft.

**Recommendation:**

1. On approval, Appendix A's rows are entered into the spec's `M14.A` table. The spec stays the single home;
   `test_m11d17_register_has_no_orphans` parses it there.
2. **Add a `Phase` column.** `M11.D.17` fails on "a state variable with no writer", and in Phase B many
   variables have no writer *yet* (`basic_level`'s writer is the slow-tick estimator, Phase D). With a phase
   column the check is "has a writer in some phase", and a second check confirms every writer marked Phase B
   exists in code.
3. **Add `test_m14a_register_matches_object_fields`**: every field on the four `M1` classes has a register row,
   and every register row names a real field. Without it the register drifts from the code (P6), and
   `M11.D.17` checks a table that no longer describes the program.
4. **Engine-side mechanisms are owned by `engine`.** `M11.D.17`'s third clause ("reads a quantity its owner
   cannot observe") is the static form of `M4.B.2`, which binds the **policy**. `M8`'s predicate reads
   `fused_into` across a group and the visibility component reads co-residence: both are engine mechanisms,
   not agent decisions, so they may read true state. Review A4's conflicts sit in `M5.C` and `M4.D.2`
   (Phase C) and are carried there (§6).

### D7. Keyed random draws in a phase with no randomness

Phase B is scripted and, under D3, its latencies are declared. It may make no stochastic draws at all. `M3`
is Phase B scope, and `M3.D.4a`–`M3.D.4c` (counter-based, event-keyed draws) are part of `M3`.

**Recommendation:** build the draw service in Phase B — NumPy's `Philox` bit generator, keyed per `M3.D.4a`,
with the `M3.D.4b` key table as data and `M3.D.4c`'s repeated-key check — and test it directly. Any Phase B
mechanism that would otherwise need a draw (per-hop fidelity, `M1.F.4`) takes a declared value instead.
Retrofitting keyed draws after Phase C's policy exists would be far more expensive.

### D8. What "M1 objects" means in Phase B

`M1` contains 123 requirements. Many are behaviour of mechanisms `M13` places later: the `basic_level`
estimator (`M1.A.4a`, slow tick, Phase D), symptoms (`M1.A.11`, Phase D), the coach's landing (`M1.E.7`,
Phase C/D).

**Recommendation:** Phase B builds the four classes with every **state variable** `M1` names, the identifier
rules (`M1.A.20`, `M1.B.13`, `M1.C.7`), and the structural constraints that can be enforced at construction
(two-dimensional `outside_ness`, three symptom channels, three sinks, three tiers). Behaviour requirements are
satisfied in the phase that builds their mechanism. Appendix B lists the disposition of each `M1` ID.

### D9. Where the explainer and spec disagree

Review finding D lists six places where `model_explainer.md` still carries text the spec has superseded. The
proposal calls the explainer "the one to build from".

**Recommendation:** for Phase B, **the spec wins** wherever they disagree, and the six drift points are not
built from. None of the six is Phase B content, so this costs nothing now; correcting the explainer is listed
in §6.

---

## 2. The Phase B gate

Every criterion in `M13`'s Phase B *Done when* cell. Each is a test; each test is proved failing by the
named mutation before it counts as coverage (CLAUDE.md, "each must be proved failing by mutation").
Tests live under `tests/bowen/`.

| # | Criterion | Test | Mutation that must turn it red |
|---|---|---|---|
| G1 | A scripted 40-week trace runs (`M13` Phase B) | `tests/bowen/test_phase_b_gate.py::test_m13_phase_b_scripted_40_week_trace_runs` **(passing, step 10)** — runs the D5 instance under the Phase B script for 40 fast ticks and asserts 40 tick records, every scripted event delivered, and no invariant raised | Make the tick loop stop after 39 ticks; or drop one scripted event from the queue |
| G2 | The renderer emits a trace conforming to `M16.C.2` — `M16.T.1` | `tests/bowen/test_render.py::test_m16t1_trace_renders_scripted_run` — every rendered line carries time, actor, move, target and witnesses, and the `M16.A.4` effect fields in reader units | Remove the effects column from the template; or render effects from the wrong event |
| G3 | A `TRIGGER` on a dormant family-of-origin tie moves anxiety with no contact (`M4.A.2`) | `tests/bowen/test_phase_b_gate.py::test_m4a2_g3_trigger_on_dormant_tie_moves_anxiety_without_contact` **(passing, step 10; both clauses mutation-proved separately)** — **two arms**, same seed and script, differing only in a `TRIGGER` on Ana–Bruno at week 10. Asserts Ana's acute anxiety at week 11 is higher in the trigger arm, and that **no event is delivered on Ana–Bruno in either arm**. A direction between arms, per `M0.4` | Route `TRIGGER` through event delivery instead of the standing term (the no-contact clause fails); or make `TRIGGER` a no-op (the direction fails) |
| G4 | `M11.D.1` engine purity | `tests/bowen/test_engine_purity.py::test_m11d1_engine_has_no_io` — static scan of `src/bowen/engine/` for `open`, `pathlib` writes, `print`, `os`/`shutil`/`io` file calls and UI imports | Add an `open()` call to any engine module |
| G5 | `M11.D.3` config strictness | `tests/bowen/test_config.py::test_m11d3_config_rejects_unknown_key` — an unknown key raises; a malformed line raises; a missing required key raises | Make the parser skip unknown keys (the frozen engine's `_apply_config` defect, `M10.B.2`) |
| G6 | `M11.D.5` determinism | `tests/bowen/test_determinism.py::test_m11d5_same_seed_same_log` **(passing, step 11)** — two runs, one seed, byte-identical serialised logs | Iterate a `set` when appending records; or salt a key with Python `hash()` (`M3.D.4a`) |
| G7 | `M11.D.6` dirty state | `tests/bowen/test_determinism.py::test_m11d6_second_run_is_clean` **(passing, step 11)** — run A, then run B in the same process; B's log equals B's log from a fresh process | Cache the event queue or the draw cache at module level |
| G8 | `M11.D.7` no production paths in tests | `tests/bowen/test_test_hygiene.py::test_m11d7_no_production_paths_in_tests` plus an autouse guard in `tests/bowen/conftest.py` that fails a test resolving a path outside its `tmp_path` for writing | Have one test write to the repo root |
| G9 | `M16.T.2` the engine writes nothing | `tests/bowen/test_engine_purity.py::test_m16t2_engine_writes_nothing` **(passing, step 11)** — runs the engine with the working directory set to an empty temp dir and asserts it is still empty; patches `builtins.open` for write modes and asserts no call | Have the engine write an audit file, as `Simulator.__init__` does (`M16.B.2`) |
| G10 | `M16.T.3` the persistence sink is a pure observer | `tests/bowen/test_log.py::test_m16t3_sink_does_not_change_results` **(passing, step 11)** — same seed with the sink attached and detached, identical final state. **Passes vacuously in Phase B** (no policy reads the log); the spec requires it re-run at the end of C and D | Have the sink mutate a record the engine later reads |
| G11 | `M16.T.5` byte-identical logs, header included | `tests/bowen/test_log.py::test_m16t5_same_seed_same_log_with_header` **(passing, step 11)** | Put a wall-clock timestamp in the header |
| G12 | `M11.D.17` the register has no orphans | `tests/test_spec_consistency.py::test_m11d17_register_has_no_orphans` — parses `M14.A`; fails on a variable with no writer in any phase, a mechanism absent from `M3.D.1`'s order, or an agent-owned mechanism reading another person's true state | Delete one writer row; or add a mechanism not in the order |
| G13 | `M11.D.22` engine cannot see its arm | `tests/bowen/test_engine_purity.py::test_m11d22_engine_modules_do_not_import_tests_or_readouts` — import graph of `src/bowen/engine/` contains nothing under `tests/`, `src/bowen/render/` or `src/bowen/readouts/` | Import the renderer from the clock module |
| G14 | `M13.2` the frozen engine stays green | the existing suite, `python3 -m pytest tests/` — 55 tests at plan time | Any Phase B change that breaks `tests/test_simulator.py` |

**Added by D6:** `tests/test_spec_consistency.py::test_m14a_register_matches_object_fields` — fails when a
field on `Person`, `Relationship`, `Triangle` or `Family` has no register row, or a register row names no
field. Mutation: add a field to `Person` without a row.

**G3 is the only gate criterion that tests the model rather than the plumbing.** It is a direction between two
arms, it needs no invented magnitude, and its mutation targets are mechanisms, not restatements of the rule —
so it does not fall into review A1's "verifies rather than tests" class.

---

## 3. Requirements Phase B builds, and how each is tested

These are not gate criteria, but `M13` places them in Phase B and they must be satisfied there.
"Structural" means the test checks a construction-time property; "behaviour" means it runs ticks.

### M3 — clocks, order, determinism

| ID | Test | Kind |
|---|---|---|
| `M3.A.1` | `test_m3a1_fast_tick_is_one_week` — tick length in config is 1 week, graded `[I]` (`M10.C.1`) | structural |
| `M3.B.1` | `test_m3b1_slow_tick_fires_every_52_fast_ticks` — a hook fires at tick 52 and not before; it does nothing in Phase B | behaviour |
| `M3.C.1` | `test_m3c1_delivery_uses_edge_latency` — latency 1 and latency 3 edges deliver at *t+1* and *t+3* (D3) | behaviour |
| `M3.D.1` | `test_m3d1_steps_run_in_spec_order` — instrumented run records the nine step labels per tick in order | behaviour |
| `M3.D.2` | `test_m3d2_standing_load_precedes_delivery` — a tick's standing-load record precedes its delivery records | behaviour |
| `M3.D.3` | `test_m3d3_involvement_and_triangles_precede_select` | behaviour |
| `M3.D.4`, `M3.D.4a` | `test_m3d4a_draw_is_pure_function_of_seed_and_key`; `test_m3d4a_key_rejects_state_quantities`; `test_m3d4a_no_builtin_hash_in_engine` (static) | structural |
| `M3.D.4b` | `test_m3d4b_every_draw_class_declares_its_key` — the key table is data, and the draw service refuses an undeclared class | structural |
| `M3.D.4c` | `test_m3d4c_repeated_key_raises_in_debug` | behaviour |
| `M3.D.5` | covered by G6 and G11 | — |
| `M3.D.6` | `test_m3d6_no_llm_import_in_engine` (static) | structural |
| `M3.E.1` | `test_m3e1_activation_and_visibility_are_separate_components` | structural |
| `M3.E.2` | `test_m3e2_activation_regime_recorded_and_graded` — default regime in config, graded `[I]`, recorded in the log header (`M16.A.1a`) | structural |

### M4 — the Phase B steps

| ID | Test | Kind |
|---|---|---|
| `M4.A.1` | `test_m4a1_every_tie_loads_every_tick_without_events` — a tie with no events still loads its members, and the load scales with bond energy ÷ `functional_level` in direction | behaviour |
| `M4.A.2` | G3 | — |
| `M4.A.3` | `test_m4a3_reconciliation_converts_standing_to_interaction_load` — two arms; after `RECONCILIATION` the tie carries events and its standing contribution falls | behaviour |
| `M4.A.4` | `test_m4a4_institutionalize_makes_worry_edges` — ties become non-interactive with bond energy retained, and no new code path is added (asserted by the standing-load function being the only reader) | behaviour |
| `M4.A.5` | `test_m4a5_self_generated_load_derives_from_basic_level`; `test_m4a5_self_generated_load_does_not_swamp_tie_load` — a person alone loads; at every `basic_level` the self term is below the tie term on the reference instance's median tie | behaviour |
| `M4.B.1` | `test_m4b1_person_reads_addressed_and_witnessed_events` | behaviour |
| `M4.B.2` | not built in Phase B — no policy. Recorded in the register (D6) | — |
| `M4.C.1` | *(D1)* `test_m4c1_delivered_event_raises_receiver_anxiety` — direction only: an arm with the event delivered against one without; no magnitude asserted | behaviour |
| `M4.E.1` | `test_m4e1_scripted_move_becomes_full_event` — every `M1.F.1` field populated | structural |
| `M4.E.1a` | `test_m4e1a_witnesses_filled_by_visibility_not_script` — a script that names witnesses is rejected | structural |
| `M4.G.1` | `test_m4g1_three_withdrawals_register_as_distant_tie` — three scripted `DISTANCE` moves in a row move the tie to the distant state; three separated by other traffic do not | behaviour |
| `M4.G.2` | `test_m4g2_invariants_asserted_every_tick` — a deliberately broken state raises at the end of the tick it appears in (D2's set) | behaviour |

### M1.F — the event record

| ID | Test | Kind |
|---|---|---|
| `M1.F.1`, `M1.F.1a` | `test_m1f1_event_carries_all_fields` (channel included; scripted events record `SCRIPTED`) | structural |
| `M1.F.1b` | `test_m1f1b_witnesses_computed_from_household_and_conductance` (needs D4) | behaviour |
| `M1.F.2` | `test_m1f2_source_position_can_flip_sign` — under D1's `M4.C.1`, one source position gives the opposite sign of effect | behaviour |
| `M1.F.3` | `test_m1f3_route_through_neutral_third_damps` — direction between arms | behaviour |
| `M1.F.4` | `test_m1f4_fidelity_degrades_per_private_hop` | behaviour |
| `M1.F.4a` | **deferred to Phase C** — register constrains the reply's register, which is a policy choice | — |
| `M1.F.5` | witnesses appraise in Phase B (D1); the **chronic-anxiety source** clause is slow-tick, Phase D | partial |
| `M1.F.6` | `test_m1f6_scripted_stressors_are_spells` — every stressor in the script has a start and a duration; a per-tick probability stressor is rejected | structural |
| `M1.F.7` | `test_m1f7_exogenous_flag_counts_separately` | structural |
| `M1.F.8` | `test_m1f8_same_tick_batch_order_does_not_change_state` — permute delivery order within a batch, identical state. (The full `M11.D.16` is Phase C) | behaviour |
| `M1.F.9` | `test_m1f9_binder_unavailable_returns_held_anxiety_to_budget` — names its binder; the binder's held anxiety appears in the family budget, not discarded | behaviour |

### M8 — the live-position predicate

| ID | Test | Kind |
|---|---|---|
| `M8.1` | `test_m81_positions_live_counts_unfused_present_occupants` | behaviour |
| `M8.2`, `M8.3` | `test_m82_neutral_external_counts`; `test_m83_sided_external_does_not_count` | behaviour |
| `M8.4` | `test_m84_displaced_inactive_member_does_not_count` | behaviour |
| `M8.5` | `test_m85_single_implementation_called_from_all_sites` — static: one definition; the triangle-activity site calls it in Phase B, and the register lists the three later call sites | structural |
| `M8.6`–`M8.8` | **deferred** — alignment, routing and announcement act on differentiating moves (Phase C) | — |

`M13.1` (M8 lands before the policy) is met by building it here.

### M16 — the run log and renderer

| ID | Test | Kind |
|---|---|---|
| `M16.A.1`, `M16.A.1a` | `test_m16a1_header_is_self_describing` — seed, config hash, spec revision, instance id, activation/visibility identity and regime | structural |
| `M16.A.2` | `test_m16a2_event_records_emitted_and_delivered_times` | behaviour |
| `M16.A.3`, `.3a`–`.3c` | `test_m16a3_selection_record_present_for_scripted_moves` — in Phase B the selection record states `SCRIPTED` with an empty propensity vector; the fields exist so Phase C fills them without a format change | structural |
| `M16.A.4` | `test_m16a4_effects_recorded_beside_cause` | behaviour |
| `M16.A.5`, `M16.A.5a` | `test_m16a5_belief_writes_tagged_apart` — Phase B writes no beliefs; the tag and the discrepancy field exist and are exercised by a fixture record | structural |
| `M16.A.6` | covered by G6 | — |
| `M16.A.7` | `test_m16a7_header_records_constant_change_flags` (`M10.B.4`) | structural |
| `M16.A.8` | **deferred** — no mechanism with asymmetric `[I]` constants exists in Phase B | — |
| `M16.A.9`, `M16.A.10` | **deferred** — SHOULDs over the policy's channels and thresholds (Phase C) | — |
| `M16.B.1`, `M16.B.2` | G4 and G9 | — |
| `M16.B.3` | G10; the store/sink split is in the package layout (§4) | — |
| `M16.C.1` | `test_m16c1_renderer_is_deterministic` — same log, byte-identical text; no LLM import | behaviour |
| `M16.C.2` | G2 | — |
| `M16.C.3` | met by building it here | — |
| `M16.C.4` | `test_m16c4_single_agent_view` — Nadia's view holds exactly the events she sent, received or witnessed | behaviour |
| `M16.C.5` | `test_m16c5_trace_carries_header_and_framing` — the `M11.F` framing block, including `M11.F.10`, and no clinical-record layout (no "patient", "diagnosis", "case history" headings) | structural |

### Config, identifiers and the reduced instance

| ID | Test | Kind |
|---|---|---|
| `M0.3`, `M10.B.1` | `test_m10b1_no_editorial_content_in_python` — event kinds, the family and the script load from markdown | structural |
| `M10.B.2` | G5 | — |
| `M10.B.4` | `test_m10b4_constants_frozen_before_suite` — the constants file carries a frozen-at stamp, and the header's change flags (`M16.A.7`) read from it | structural |
| `M2.1` | `test_m21_family_declared_in_markdown` | structural |
| `M2.3` | `test_m23_reduced_instance_is_not_closed` — every adult in the instance has a family-of-origin tie | structural |
| `M2.A.0e` | `test_m2a0e_spouses_within_tolerance` — fails on today's `M2.A` data; passes after D5's correction | structural |
| `M2.A.0h`, `M1.A.20` | `test_m1a20_identifiers_are_declared_not_counted` | structural |
| `M1.B.13`, `M1.C.7` | `test_m1b13_tie_id_is_unordered_pair`; `test_m1c7_triangle_id_is_sorted_triple` | structural |

---

## 4. Package layout

New code lives in `src/bowen/` (CLAUDE.md, `M13.2`). The frozen `src/engine.py` is not touched.

```
src/bowen/
  engine/                 # pure: no I/O, no UI (M11.D.1, M16.B.1)
    objects.py            # Person, Relationship, Triangle, Family — state only
    identifiers.py        # M1.A.20, M1.B.13, M1.C.7
    events.py             # Event record, queue, same-tick batching (M1.F)
    draws.py              # counter-based keyed draws (M3.D.4a–c)
    activation.py         # M3.E.1 activation
    visibility.py         # M3.E.1 visibility; witness sets (M1.F.1b); hosts the M8.5 call
    live_positions.py     # M8 — the one implementation
    standing_load.py      # M4.A
    perceive.py           # M4.B
    appraise_base.py      # M4.C.1 only (D1); replaced by the Phase C appraisal module
    act.py                # M4.E
    consolidate.py        # M4.G, decay toward the floor
    invariants.py         # M6, D2's set
    tick.py               # M3.D.1 order; the slow-tick hook (M3.B.1)
    event_store.py        # in-run store, always present (M16.B.3)
    log_records.py        # record types and the emitter interface (M16.B.1)
  scenario/
    config_parse.py       # strict markdown parsing of text (M10.B.2); takes a string, not a path
    family.py             # builds the instance from parsed config
    scripted_source.py    # ScriptedSource
  io/
    load.py               # reads config files from disk; the only file reads
    sinks.py              # persistence sink, a pure observer (M16.B.3)
  render/
    trace.py              # the deterministic renderer (M16.C)
  readouts/               # empty in Phase B; M11.D.22 forbids engine imports from here

config/bowen/
  constants.md            # every [I] constant with its grade (M10.1, M10.B.4)
  family_reduced.md       # the D5 instance (M2.3)
  script_phase_b.md       # the 40-week script

tests/bowen/
  conftest.py             # M11.D.7 guard; explicit tmp paths
  test_*.py
```

The engine is a deterministic function of `(seed, config, scenario)` (`M3.D.4`): the caller parses config and
script, builds the instance, attaches an emitter, and runs. Nothing in `engine/` imports from `scenario/`,
`io/`, `render/` or `readouts/`.

---

## 5. Build order

Each step lands with its tests, written first, and the full suite green.

| Step | Builds | Depends on | Tests that land with it |
|---|---|---|---|
| 0 | Owner answers D1–D9; register rows (Appendix A) entered into spec `M14.A`; `M13`, `M1.A` and `M2.A` amended per D1, D4, D5 | — | `test_m11d17_register_has_no_orphans` (register only) |
| 1 | `scenario/config_parse.py`, `scenario/constants.py`, `io/load.py`, `config/bowen/constants.md`. **Done 2026-10-06.** The register holds the three constants the spec fixes or step 1 needs; each further `[I]` constant is entered with the mechanism that reads it, and `frozen_at` is set before the first acceptance test runs (step 10, `M10.B.4`) | 0 | G5, `M10.B.*` tests |
| 2 | `engine/identifiers.py`, `engine/objects.py` (Appendix B's structural set). **Done 2026-10-06.** The register gained a `Relationship.id` row it had missed; `test_m14a_register_matches_object_fields` is in `tests/bowen/test_register.py`, not `test_spec_consistency.py`, because it imports model code | 1 | identifier tests, `test_m14a_register_matches_object_fields` |
| 3 | `engine/draws.py`. **Done 2026-10-06.** Philox keyed by the seed, counter from BLAKE2b of the canonical key, uniforms from raw words; the ten-class table is checked row by row against the spec's `M3.D.4b` | 1 | `M3.D.4a`–`M3.D.4c` tests |
| 4 | `engine/events.py`, `engine/event_store.py`, `engine/log_records.py`, plus `scenario/event_kinds.py` and `config/bowen/event_kinds.md`. **Done 2026-10-06.** Kinds are config (`M10.B.1`) mapped onto six fixed mechanisms; the configured moves are checked against `M5.A.1` and `M5.B` | 2 | `M1.F.1`, `M1.F.6`, `M1.F.7`, `M16.A.*` structural tests |
| 5 | `scenario/family.py`, `config/bowen/family_reduced.md`. **Done 2026-10-06.** Sex column added to `M2.A` (`M2.A.0i`); two constants entered (spouse tolerance 1.0, fixation age 12, both `[I]`); tie values invented and listed in the config's header | 2 | `M2.*` tests |
| 6 | `engine/live_positions.py` (before anything that calls it, `M13.1`). **Done 2026-10-06.** A pure predicate over caller-described occupants; the adapter from model state lands with its first caller at step 8 | 2 | `M8.1`–`M8.5` tests |
| 7 | `engine/activation.py`, `engine/visibility.py`. **Done 2026-10-06.** The witness and timing rules are the project's, graded `[I]` and stated in `visibility.py`; `per_hop_fidelity` (0.8, `[I]`) stands in for the M3.D.4b draw; `activation_regime` is declared in the constants file | 2, 6 | `M3.E.*`, `M1.F.1b`, `M4.E.1a` tests |
| 8 | `engine/standing_load.py`, `engine/appraise_base.py` (perception merged in), `engine/act.py`, `engine/consolidate.py`, `engine/invariants.py`, plus `engine/params.py`, `engine/state.py`, `engine/event_effects.py`, `engine/recompute.py` and `scenario/params.py`. **Done 2026-10-06.** Nine `[I]` constants; formulas stated in each module. Register: triangle `bound_anxiety`'s writer moved to Phase C (nothing binds into a triangle in Phase B); rows added for structural events and delivered cutoffs. Owner read M4.A.5's "must not swamp" at family and nuclear-household level | 3–7 | `M4.*`, `M1.F.2`–`.4`, `.8`, `.9`, D2 invariant tests |
| 9 | `engine/tick.py` — the nine-step order, slow-tick hook. **Done 2026-10-06.** A `Source` protocol supplies each tick's scheduled events and selections; a TRIGGER applied at step 2 of week *t* spikes the standing load from week *t*+1 (M3.D.2) | 8 | `M3.A.1`, `M3.B.1`, `M3.C.1`, `M3.D.1`–`.3` tests |
| 10 | `scenario/scripted_source.py`, `config/bowen/script_phase_b.md`, `scenario/assemble.py`. **Done 2026-10-06.** Constants frozen 2026-10-06 before G1/G3 first ran: `constants_frozen.md` is the snapshot, `constants_changes.md` the log M10.B.4 requires | 4, 9 | G1, G3 |
| 11 | `io/sinks.py`, `scenario/header.py`, `run.py` (the composition root: `python3 -m src.bowen.run --seed 7 --log …`). **Done 2026-10-06.** The header hashes the parsed config, not file text, and reads the spec revision from the spec's front matter | 4 | G9, G10, G6, G7, G11 |
| 12 | `render/trace.py` | 4, 10 | G2, `M16.C.*` tests |
| 13 | Static guards | all | G4, G8, G13 |
| 14 | Mutation proof of every gate test (§2's mutation column), recorded in the completion report | all | — |
| 15 | Human read of a rendered trace (§7); `docs/spec_coverage.md` (`M14.1`); status report | all | — |

---

## 6. The external review's findings, by the phase they block

The owner approved revision 10 without first revising for the review (2026-10-06). Each finding is carried
here to the phase it blocks. Phase B's own exposure is resolved by D1–D9.

| Review | Finding | Blocks | Disposition |
|---|---|---|---|
| A1 | Criteria verify rather than test | C, D | Run `M11.1d` on paper for all 41 criteria before the Phase C plan; split `M11.C` into mechanism verification and emergent consequence |
| A2 | The cancellation premise; constant sweep only in Phase E | C | Correct explainer §17.3 and `_STATUS.md` before the Phase C plan; decide whether a dominant-constant sweep gates Phase C |
| A3 | `M6.I.6` unsatisfiable | B (via D2), C, D | D2 for Phase B. Restate `M6.I.6` with a stock-and-flow table before Phase C; `M11.C.11`, `.31`, `.37` depend on it |
| A4 | `M4.B.2` conflicts with approved gates | C | Scope `M4.B.2` row by row — which of `M5.C`'s gates, `M5.F.1`, `M4.D.2` read truth and which read belief — before the Phase C plan. D6 handles the engine-side readers in Phase B |
| A5 | Phase C criteria need Phase D machinery | C | Re-place `M11.C.17`, `.18`, `.20` (and check `.4`, `.1`, `.38`) before the Phase C plan |
| A6 | Death and birth undecided | D | Decide `M6.3` and the transmission-to-estimator handover before the Phase D plan |
| A7 | Sub-week items on a weekly clock | B (via D3), C | D3 for Phase B; `M5.D.6`'s "next day" and the multi-tick `I-POSITION` sequence before Phase C |
| B | Contradictions and stale residue | mostly C, D | Undefined fields: D4. `M2.A` tiers: D5. Others before the phase that builds them |
| C | `M2` data defects | B (via D5), D | Spouse tolerance for the reduced instance: D5. Teodor/Sofia family-of-origin ties and the fixation age before Phase D |
| D | Explainer–spec drift | C, D | D9 for Phase B. Correct the six entries before the Phase C plan |
| E | `M11.D.16` equivariance vs keyed draws; `M17.A.4` margin; no compute budget | C, E | Say "in distribution" or "byte for byte" in `M11.D.16` before Phase C; the rest at Phase E |
| F | Document form | — | Not a phase blocker; the normative-kernel extraction is a separate decision |
| G | Fidelity to the chapters | C, D | Add Bowen's two falsifiers and `L15.1`'s delayed channel as criteria; decide `L07.4` under `M11.4c`; rescope G2's six; fix G3's five ledger entries — before the phase each touches |

---

## 7. What cannot be made code-testable in Phase B

| Item | Why | Proposal |
|---|---|---|
| Whether the rendered trace is **readable** (`M16.C.3`: "how a defect the conformance check does not encode gets noticed") | `M16.T.1` checks fields, not sense | **Human review**: the owner reads the 40-week rendered trace end to end before Phase B is declared done, and records anything that reads wrong |
| Whether the register's classifications are **correct** (Appendix A) | `M11.D.17` checks structure — every variable has a writer — not whether the writer named is the right one | **Human review** of Appendix A at plan approval |
| Whether every Phase B `[I]` constant is **labelled and never described as sourced** | `M11.D.4` is a Phase C gate | Built as `test_m11d4_invented_constants_labelled` in Phase B anyway, at no extra cost, but not counted toward the Phase B gate |

`M11.E`'s four criteria (accepted as written at approval) are all Phase C or later.

---

## 8. Adjacent issues found, not fixed

- **`requirements.txt` does not list pytest**, and the system Python's pytest had drifted to a version its own
  installed plugin could not load (fixed in the environment on 2026-10-06, nothing in the repo). Phase B adds
  `tests/bowen/`; the version should be pinned.
- **No `.github/workflows/tests.yml`**, which the global rules require for a repo with a suite pushed to
  GitHub. Phase B's purity and determinism tests are exactly the kind a blank machine checks best.
- **`sim_audit.csv` in the repo root and `src/`** — two stray copies from the frozen engine (`M16.B.2`).
  Phase B does not touch the frozen engine, so they stay; CLAUDE.md says to fix the frozen engine's I/O when it
  is next touched.
- **The system Python has a stray top-level `tests` package in site-packages**, installed by some other package. It shadows `import tests…`, so `tests/bowen/test_register.py` loads `test_spec_consistency.py` by path. Found at step 2.
- **A mutation script left injected breaks in the tree at step 11.** zsh does not word-split an unquoted variable, so a shell loop's backups failed while its edits succeeded. Found at once (24 tests red), reverted by hand and checked; the mutation runs since use a Python harness that restores in `finally`. Worth a line in the learnings catalogue.
- **`docs/implementation_task_list.md` tracks the frozen v1.2 grid engine** and could be mistaken for this
  plan.
- **Spec `M10.A.1`** still lists "the ceiling on `systems_perspective`" though `M1.A.18b` is a rate (review B).
  Not Phase B content.

---

## Appendix A — `M14.A` register draft

For owner review (§7). On approval these rows go into the spec's `M14.A` table with the `Phase` column D6
adds. **Classes** are `M14.A.1`'s four: exogenous-homogeneous (EH), exogenous-heterogeneous (EX),
endogenous-decision (ED), endogenous-derived (DV). **Phase** is when the writer is built. Mechanisms owned by
`engine` may read true state (D6.4); mechanisms owned by a person are bound by `M4.B.2`.

### A.1 State variables

| Variable | Owner | Class | Written by | Phase of writer |
|---|---|---|---|---|
| `id` | Person | EX | family config (`M1.A.20`) | B |
| `role` | Person | EX | family config (`M1.A.17`) | B |
| `sex` *(D4)* | Person | EX | family config | B |
| `household_id` *(D4)* | Person | EX | family config; life-stage update | B (config); D (update) |
| `basic_level` | Person | DV | family config at `t0`; the estimator (`M1.A.4a`, `M7.A.1`) | B (initial); D |
| `functional_level` | Person | DV | `basic_level` + swing (`M1.A.5a`); swing written by consolidation and the self-directed channel | B (consolidation); C (channel) |
| `acute_anxiety` | Person | DV | standing load (`M4.A`); base appraisal (`M4.C.1`, D1); consolidation decay | B |
| `chronic_anxiety` | Person | DV | family config at `t0` (`M2.A.0a`); slow-tick derivation (`M1.A.7a`) | B (initial); D |
| `programmed_reactivity` | Person | DV | family config at `t0`; childhood fixation (`M1.A.7`) | B (initial); D |
| `outside_ness` (outward, inward) | Person | DV | appraisal and selection (`M5.F`) | C |
| `life_energy` ratio | Person | DV | derived from `basic_level` (`M1.A.10`, `M10.A.1`) | B (derivation) |
| `symptom_load` [physical, mental, social] | Person | DV | symptom accumulation (`M7.D`) | D |
| `involvement_weight` | Person | DV | step 5 recompute (`M1.A.12`) | B |
| `structural_importance` | Person | DV | derivation (`M1.A.13a`) | D |
| `sibling_position` | Person | EX | family config (`M1.A.14`) | B |
| `functional_sibling_position` | Person | DV | derivation (`M1.A.14a`) | D |
| `financially_dependent` | Person | EX | family config; life-stage update | B (config); D |
| `beliefs` | Person | DV | belief layer (`M9`) | D |
| `systems_perspective` | Person | DV | landed contact (`M1.E.7`) | C |
| reactive state, three detectors (`M1.A.19`) | Person | DV | appraisal | C |
| pseudo-self / solid-self split (`M6.I.4`, `M10.A.1a`) | Person | DV | dyadic exchange; estimator | C; D |
| `alive` | Person | DV | mortality (`M7.C.1`, `M6.3`) | D |
| `conductance` | Relationship | EX | family config (`M1.B.2`) | B |
| `bond_energy` | Relationship | DV | family config; `RECONCILIATION`, `TRIGGER`, `INSTITUTIONALIZE` handling; consolidation | B |
| `interactive` (vs worry edge) | Relationship | DV | `CUTOFF`, `RECONCILIATION`, `INSTITUTIONALIZE` (`M4.A.3`–`.4`) | B |
| `tie_state` (cut off / distant / resolved / conflict, `M1.B.3`) | Relationship | DV | consolidation hardening (`M4.G.1`) | B |
| distance-bound anxiety (`M1.D.2a`) | Relationship | DV | `DISTANCE` handling; `binder_unavailable` (`M1.F.9`) | B |
| `functioning_balance` per area, directed | Relationship | DV | pole flip (`M1.B.5`–`.7`) | C |
| `investment`, directed | Relationship | DV | appraisal (`M1.B.8`) | C |
| `areas_of_joint_activity` | Relationship | DV | functioning-balance narrowing (`M1.B.9`) | C |
| `taboo_set` | Relationship | DV | appraisal; purposeful mention (`M1.B.10`, `.10a`) | C |
| `latency` | Relationship | EX | family config (`M1.B.11`, D3) | B |
| `dyad_age` | Relationship | DV | slow tick (`M1.B.12`) | D |
| individuality–togetherness, basic and functional (`M1.A.5b`) | Relationship | DV | config (basic); anxiety (functional) | B (config); C |
| `members`, `inside_pair`, `outside` | Triangle | DV | step 6 recompute (`M1.C.3`) | B |
| `active` | Triangle | DV | step 6, via `M8` (`M1.C.3`, `M8.5`) | B |
| `bound_anxiety` | Triangle | DV | consolidation re-tally | B |
| `activation_memory` | Triangle | DV | step 6 (`M1.C.4`) | B |
| `intensity_floor` | Triangle | DV | `I-POSITION` held (`M1.C.5`) | C |
| `undifferentiation_budget` | Family | DV | family config; `binder_unavailable` return (`M1.F.9`) | B |
| sink allocations [marital conflict, spouse dysfunction, child projection] | Family | DV | sink allocation (`M1.D.1`) | C |
| `overflow` | Family | DV | sink allocation (`M1.D.3`) | C |
| `leadership_office` (occupant, sphere) | Family | DV | family config; recognition (`M1.D.4`) | B (config); C |
| `differentiation_capacity` | Family | DV | derivation (`M1.D.4a`) | C |
| `tolerance` per agent | Family | DV | `M7.D.4` | D |
| `access_vector` | Family | EX | family config (`M1.D.6`) | B |
| `ambient_anxiety` | Family | DV | societal dials (`M1.D.7`) | D |

### A.2 Phase B mechanisms

| Mechanism | Owner | Execution mode | Trigger | Reads | Writes on owner | Writes on other objects |
|---|---|---|---|---|---|---|
| Standing load (`M4.A.1`, `.5`) | engine | synchronous batch, step 1 | every tick | each tie's `bond_energy`, `interactive`; each person's `functional_level`, `basic_level` | — | `Person.acute_anxiety` |
| Deliver (`M1.F.8`) | engine | latency-delivered, step 2 | event due | event queue | — | inboxes |
| Perceive (`M4.B.1`) | engine | synchronous batch, step 3 | every tick | inbox | — | perceived set per person |
| Base appraisal (`M4.C.1`, D1) | engine | synchronous batch, step 4 | perceived event | event fields, tie `conductance`, receiver `functional_level` | — | `Person.acute_anxiety` |
| Involvement recompute (`M1.A.12`) | engine | synchronous batch, step 5 | every tick | ties, `acute_anxiety` | — | `Person.involvement_weight` |
| Triangle recompute (`M1.C.3`) | engine | synchronous batch, step 6 | every tick | tie tension, `M8` | — | `Triangle.active`, `inside_pair`, `outside`, `activation_memory` |
| Live positions (`M8`) | engine | called from step 6 and visibility | on call | group membership, `fused_into` | — | — (pure) |
| Script select (`ScriptedSource`) | engine | synchronous batch, step 7 | script entry due | script | — | selection record |
| Act (`M4.E`) | engine | synchronous batch, step 8 | selection | selection, tie `latency` | — | event queue |
| Visibility (`M3.E.1`, `M1.F.1b`) | engine | called from step 8 | event created | `household_id`, tie `conductance`, route | — | `Event.witnesses` |
| Consolidate (`M4.G.1`, decay) | engine | synchronous batch, step 9 | every tick | tie event history, `chronic_anxiety` | — | `Relationship.tie_state`, `Person.acute_anxiety`, `Triangle.bound_anxiety` |
| Invariants (`M4.G.2`, D2) | engine | synchronous batch, step 9 | every tick | all | — | — (raises) |
| `binder_unavailable` (`M1.F.9`) | engine | latency-delivered | scripted event | named binder | — | `Family.undifferentiation_budget`, the binder's object |
| Slow-tick hook (`M3.B.1`) | engine | slow tick | every 52 fast ticks | — | — | — (no-op in Phase B) |
| Same-tick batching (`M1.F.8`) | engine | **documented composite** (`M14.A.2`) | same-tick events | batch | — | — |

---

## Appendix B — `M1` disposition in Phase B

**Satisfied in Phase B (structural, at construction):** `M1.A.0` (naming only — no feeling-state fields),
`M1.A.1` (reactivity not stored), `M1.A.2` (0–100, no clip), `M1.A.5`, `M1.A.5a` (decomposition fields),
`M1.A.5b` (two balances on the tie), `M1.A.6` (no threshold on `basic_level` — enforced by a static check that
symptom code does not read it, active once symptoms exist), `M1.A.8`, `M1.A.9a` (two axes), `M1.A.11` (three
channels), `M1.A.12` (membership derived, no stored set), `M1.A.13` (three-tier enum), `M1.A.14` (static
field), `M1.A.15` (bool; no material stock), `M1.A.16` (belief container), `M1.A.17`, `M1.A.20`, `M1.B.1`,
`M1.B.2` (undirected, no distance input), `M1.B.11`, `M1.B.13`, `M1.C.1`, `M1.C.3` (topology stored apart from
the active set), `M1.C.7`, `M1.D.1` (one budget, three sinks), `M1.D.2` (no fourth sink), `M1.D.8` (via
`M2.3`).

**Satisfied in Phase B (behaviour):** `M1.B.3` (four tie states distinguishable — via `M4.G.1` and the
standing load), with `test_m1b3_four_tie_states_are_distinct`; `M1.B.4` (bond-energy decay at or near zero;
reunion with zero re-activation latency), with `test_m1b4_reunion_restores_coupling_immediately`.

**Deferred to the phase that builds the mechanism:**

- Phase C: `M1.A.3`, `.3a`–`.3d` (licence, read by the policy); `M1.A.9` (three inputs); `M1.A.18`, `.18a`–`.18d`
  and `M1.A.19` (perspective and detectors); `M1.B.5`–`M1.B.10a` (functioning balance, investment, joint
  activity, taboo set); `M1.C.2` (position value inverts with load — needs the policy's preference);
  `M1.C.3a`–`.3c`, `M1.C.4`–`M1.C.6`; `M1.D.2a` (distance binds — the `DISTANCE` move's effect),
  `M1.D.3`, `M1.D.4`, `M1.D.4a`; `M1.E.*` except the `role` field.
- Phase D: `M1.A.4`, `.4a`–`.4j` (the estimator); `M1.A.7`, `.7a` (fixation and chronic derivation);
  `M1.A.10` (life-energy use); `M1.A.11a`–`.11c` (channel assignment); `M1.A.13a`, `M1.A.14a`–`.14d`
  (derived positions); `M1.B.12` (dyad age); `M1.D.5`–`M1.D.7l` (tolerance, access use, societal dials).

`M1.A.13`'s tier field exists in Phase B, but no Phase B mechanism sets or reads it.
