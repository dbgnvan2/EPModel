# Model ergonomics — open after Parts A–C, 2026-10-22

- [ ] T-005 ⟨later⟩ **Revision 35's dispositions, for the owner's review (Part E, 2026-10-22):** `docs/spec_revision_35_dispositions.md`,
  218 rows for 213 sentences (35 stay, 32 moved, 151 rewritten; rows 219–262 added the same day for the missed
  phrases, the item below), applied as Claude's dispositions under the owner's bulk
  decision (§8 Q2). The review of 2026-10-22 found two log bullets whose quotation was not the sentence as it stood
  (`M1.A.4k`'s owner list and `M11.C.14`'s Bowen quotation), both restored verbatim, and one post-hoc label dropped from
  `M11.C.6`'s arm, restored; the E2 pass was an agent's, so the owner's read of the rewritten rows is the check.
  The "stays" rows marked unsure carry the reason in the last column. A reversed row is one edit in the spec and one
  bullet back out of `docs/spec_revision_log.md`.
- [ ] T-006 ⟨later⟩ **D4, long `Spec:` lines (Part D, 2026-10-22; the plan says record, do not split):** `engine/objects.py` 56 IDs
  (its classes `Person` 21, `Relationship` 17), `engine/life_course.py` 21, `engine/appraise.py` 18,
  `engine/iposition.py` 18, `ensemble/criteria.py` 19 (a registry, one per criterion), `policy/policy.py` 16,
  `engine/invariants.py` 15, `engine/log_records.py` 14, `engine/events.py` 12, `engine/symptoms.py` 12,
  `engine/slow_tick.py` 12, `scenario/family.py` 12. A line this long may mean the module does too much.
- [ ] T-007 ⟨later⟩ **Adjacent issues found by Part D's docstring pass, not fixed (2026-10-22):**
  - `engine/draws.py` declares the `per_hop_fidelity` draw class (`M3.D.4b`'s row) but nothing draws it;
    `visibility.fidelity_for` uses the constant. The spec's table row or the engine is out of step.
  - `engine/live_positions.py` cites `#M13.1`, a phase-plan requirement the code does not implement (the class B3
    removed elsewhere; it is a defined ID, so the test passes).
  - `engine/consolidate.py`'s module `Spec:` line omits `#M4.C.1c`, which `consolidate()` cites.
  - `engine/recompute._counter_moves`: an inner `if` on the kind is always true after the `continue` above it.
  - `engine/iposition._abort_act` re-imports `math` inside the function.
  - `objects.Role` (MEMBER/EXTERNAL) and `events.Role` (TARGET/WITNESS) share a name; a reading hazard only.
  - `scenario/header.py::resolved_config` hashes neither `kinds.components` nor `kinds.accommodating`, so a change
    to a kind's components or its `accommodates` flag leaves `config_hash` unchanged; check against `M16.A.1`.
  - `scenario/policy_source.py` implements `M1.E.8` (its comment says so) without citing it.
  - `src/bowen/__init__.py` names `docs/implementation_plan_phase_b.md` as the build order; stale since version 2.
  - `run.py` composes a `ScriptedSource` only; `PolicySource` is not reachable from the composition root.
  - `readouts/evaluation.py` line 103 emits phase narration inside a readout value ("from Phase D part II").
  - `ensemble/counterfeit.py` cites `M5.C.1` for "the gate's two thresholds" while `M5.C.1`'s first sentence is
    the general gate rule; check the ID is the one meant.
  - Comments (out of Part D's scope) still carry decision narration: `engine/observe.py` line 58, `engine/danger.py`
    line 78, `ensemble/runner.py` line 81, `ensemble/criteria.py` lines 47–48, 111–119, 146, `ensemble/criteria_d.py`
    line 430, and the `params.py` / `scenario/constants.py` field comments.
  - `engine/contact.py` said "revision-12 finding 6" where the spec (`M4.C.1c`) records revision 11; the phrase is
    gone from the docstring, but CHANGELOG may carry the same label.
