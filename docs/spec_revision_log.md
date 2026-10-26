# Spec revision log

**Status: the record of every revision of `bowen_agent_model_spec_v2.md`, 4 to 39 (revision 39 applied in part). Moved out of the spec at revision 35, 2026-10-22; nothing a run computes changed.**

The spec holds the current rules; this file holds how they got there. Until revision 34 the revision sections stood at the end of the spec, under `M17`, and the front matter's `status:` line listed every revision since 10. `docs/plan_model_ergonomics.md` Part E moved both here (the owner's answer to its §8 Q2, 2026-10-22, "as revision 35"), so a session reading the spec reads 480 fewer lines of history. The sections below are the spec's own text as it stood, tags and all, with the newest first; revisions 13 to 33 were written inline in the requirement bodies and have no section of their own, which Part E's second step records under revision 35.

The sections `### Revision 34` to `### Revision 5` and the undated section are the spec's text as it stood, moved whole; a statement inside one of them (revision 34's, that revisions 13 to 33 are "not here") was true when written, before the bodies' narratives were moved under their own revisions. Those narratives are the bullets `- \`ID\`: sentence`, each the sentence as it stood in the body, under the revision it names, in sections titled "inline narratives moved from the bodies at revision 35" where no section existed.

### Revision 39 — version 2 stage 3, applied in parts from 2026-10-25 ⟦rev39 · owner 2026-10-24 (D18–D31) and 2026-10-25 (Q1–Q14, the draft approved as a whole) · `docs/implementation_plan_version_2_stage_3.md`, `docs/spec_revision_39_draft.md`⟧

**Approved by the owner 2026-10-25** as `docs/spec_revision_39_draft.md`, through `AskUserQuestion`. Each group's text
lands in the spec in the commit that builds its code (the owner's Q6). This section lists what has landed.

- **Step 6 (2026-10-25), the batch's first commit and the first to move the engine hash** (`66dcbb1e8ba5` → the hash
  in `CHANGELOG.md`'s entry of that day). What a run computes did not change: the golden fingerprints are unchanged.
  - **T-130:** `M1.F.6a` says the thirteen event sizes were checked against Holmes and Rahe's Table 3 (p. 216) and
    agree; `config/bowen/event_sizes.md` quotes the paper's item wording (draft §6).
  - **T-014 (draft §21):** `M8`'s preamble says the rule is written for three places, the avoidance gate for `M5.C`
    not built and a version 3 candidate (D27), and `avoidance_available` is removed from
    `src/bowen/engine/live_positions.py` with its unit test; `M8.1` says the fused-occupant branch is not reached in a
    run and is *partial* (D28), and the code block and the mechanism table lose `avoidance_available` and fusion;
    `M8.5` and `M13.1` follow (D27, D28); `M11.C.7` and its `M11.5` row declare the coach's going along with a `SPLIT`
    a premise of the sided arm (D29); `M7.D.1a` says every onset takes its sizes on whichever channel, the channels
    being substitutable (`L08.3`; D30).
  - **Not a rule change, the same commit:** T-129 (docstrings in `moves.py`, `rejection.py`, `objects.py` and
    `invariants.py` restated to the code; ruff's `F` rules clean over `src/`, `tools/` and `tests/`) and T-127 (two
    status lines in `config/bowen/`).
- **Not yet applied:** group C (`M1.A.5e`, `M1.A.5f`, `M1.A.4k`, `M7.E.1f`, `M6.I.4a`; step 7), group F (`M5.F.4`,
  `M5.F.2c`, `M11.C.29`, `M11.C.49`; step 8), group A (`M4.C.1d`, `M4.D.3a`; step 9) and T-128 (`M17.E.1`; step 10).

### Revision 38 — the stage 2 milestone review's wording fixes, 2026-10-24 ⟦rev38 · owner 2026-10-24, stage 2 milestone review · `docs/cycles/2026-10-24_milestone-review_v2-stage-2.md`, `docs/todo/2026-10-24_stage_2_milestone_review.md`⟧

**Decided by the owner 2026-10-24**, in the review's triage, through `AskUserQuestion`: wording only. Nothing a run computes changed; no file under `src/bowen/` or `config/bowen/` was edited, so `engine_hash()` did not move and no record was rerun.

- **Restated to the code** (the owner's Q1: in both cases the code stands): `M1.D.2b`'s decay runs every week the tie is cut off, a contact week included, where the requirement said only weeks with no contact (MR-7); `M1.D.2c`'s reference level `R` is set when a tie enters `DISTANT` or `OPEN_CONFLICT` — a run hardening it, a cut tie reopening into it, or a declared tie at the start — so it can fall on a reopening, where the requirement named only the run (MR-8). The `reference_level` register row says the same.
- **Added:** `M11.C.48` to `M11.4`'s exceptions table, as a within-run matched comparison (the owner's Q6).
- **Placed:** `M17.A.1`'s refusal of a criterion that fixes fewer seeds than the sweep's rerun count comes when a sample needs the rerun, before it runs; such a criterion is swept while no sample needs one (the owner, 2026-10-24, MR-12). `tools/version_2_sweep.py` ran the rerun and refused after it; it now refuses first. Refusing before any run, as the requirement could be read, would have dropped `M11.C.48` and `M11.C.48[phase_d]`, which fix 100 seeds and needed no rerun, from the sweep. No sweep result changed.
- **Corrected:** the "Move physics" register row names `CONFLICT` and `M1.D.2c`, which revision 36 made bind stock, and lists `Relationship.reference_level` among its writes (MR-11; the correctness pass of 2026-10-24); `M1.F.6a` says its table was read from the scale as Wikipedia reproduces it, not yet checked against the paper (MR-14, T-130); `M10.A.1` lists the rate at which `systems_perspective` changes, not the ceiling `M1.A.18b` withdrew on 2026-08-28; `M10.A.2` loses a trailing clause that repeated its sentence; `M10.C.2`'s list regains its lead-in (MR-27); the `M11.C.48` row names its two real tests (MR-9); the pointer to this log says 4 to 38 (MR-16).
- **This log:** the tags of revisions 36 and 37 say approved by the owner 2026-10-24 (T-125), as the spec's do (MR-16).

### Revision 37 — stage 2's detection-limit rule and sample rule, 2026-10-23 ⟦rev37 · decided as recommended under the owner's instruction of 2026-10-22, approved by the owner 2026-10-24 (T-125) · `docs/implementation_plan_version_2.md` §2.2 D8, D10, D11⟧

**Decided 2026-10-23** as recommended, under the owner's instruction of 2026-10-22, on the branch `stage-2-plan-pilot`: wording in `M17` only. Nothing a run computes changed; `engine_hash()` is unchanged (`66dcbb1e8ba5`) and no record was rerun. The rules are built by `tools/version_2_sweep.py` and read from `config/sweep/sweep.md`; the gating ensemble records adopt the detection-limit rule at their next regeneration (T-018), which stage 2 does not make.

- **Restated:** `M17.A.1` (the detection limit in a sweep: a sample whose first run leaves a gating directional effect within a declared multiple of its half-width of zero is run again at a declared fixed seed count, the verdict is the second run's, UNDETERMINED if unconverged, both counts recorded; a first run at or above that count is not rerun; an equivalence limb is not read; a criterion whose own fixed seed count is below the declared count is refused; multiple and count `[I]`); `M17.E.1`'s first sentence now says *directional* `M11.C` criterion; `M17.E.1` (the sample rule: a joint draw over declared ranges, seeded by a declared seed and the criterion's key so a longer draw extends a shorter one; each sample an edit of `constants.md` in a copy; the result *n of S* with undetermined and errored samples counted apart, the errored left out of the denominator; a fraction is never written as a pass or a fail; premises and checks are not swept).
- **Constants:** none in `config/bowen/`. The sweep's own rules and ranges are in `config/sweep/sweep.md` (`sweep_samples`, `sweep_seed`, `detection_limit_multiple`, `detection_limit_seeds`, and seventeen range rows for `M11.C.16` and `M11.C.42`), each `[I]`.

### Revision 36 — version 2's group G and the Holmes–Rahe event sizes, 2026-10-22 ⟦rev36 · decided as recommended under the owner's instruction of 2026-10-22, approved by the owner 2026-10-24 (T-125) · `docs/PROPOSAL — CUTOFF ANXIETY, BRIDGING AND REPAIR.md`, `docs/PROPOSAL — RELEASE BY TIE (G2).md`, `docs/PROPOSAL — EVENT SIZES FROM THE HOLMES–RAHE SCALE.md`⟧

**Decided** by the owner on 2026-10-18 and 2026-10-19 (G1's Q1–Q11 and the event sizes' Q1–Q8, recorded in the two proposals) and, for G2 and every choice the build had to make, as recommended under the owner's instruction of 2026-10-22 ("you can run all night - proceed based on your recommendations - do not wait for my input"), each pending the owner's review and listed in the revision 36 report. Built 2026-10-22 on the branch `claude/recursing-bohr-2393a1` in one batch: the engine hash moved once and every record was regenerated once.

- **New:** `M1.D.2b` (a cut-off tie's held anxiety weighs on both members through `cutoff_threat`, is capped at `cutoff_stock_cap` on entering the cut, decays to `cutoff_floor_share` of its peak over `cutoff_decay_weeks`, is renewed to the peak by a negative contact and loses `positive_bridge_dissipation` of itself on a positive one, which alone returns the tie to its state before the cut; contact classed by `config/bowen/contact_classes.md`); `M1.D.2c` (a `DISTANT` or `OPEN_CONFLICT` tie is repaired by thirds of its reference level, recorded as the stock when the state is entered, a reopening included; `CONFLICT` binds at `conflict_binding_rate`); `M1.F.6a` (every stressor the engine raises is sized from the Holmes–Rahe ratios, `config/bowen/event_sizes.md`, at `lcu_scale` 6.9 units per LCU and the tie's reach); `M7.C.1l` (the loads `PARTNERING`, `BIRTH`, launch, the coach's retirement and `FIND_WORK` carry, 12 weeks); `M7.D.1a` (a symptom's own load, `OWN_ILLNESS` to the bearer and `FAMILY_ILLNESS` to the living family members on interactive ties, arriving the week after the onset through `RunState.arising`, on the `ENDOGENOUS` channel); `M11.C.48` (composite: the run of negative contacts standing at a tie's first cutoff against the mean run standing in that week on the seed's uncut ties, two variants, 100 seeds, mutant `cutoff-blind-to-learned-value`).
- **Restated:** `M1.B.3` (the five tie states by name, how each is entered and left); `M1.B.3c` (rejection held ⇒ `CUT_OFF`; rejection fading does not end a cutoff); `M1.D.2a` (the held anxiety also weighs and is shed by contact, by the decided rules only); `M4.C.1c` (the nodal spike reads the `CUT_OFF` state); `M4.G.1` (a run of three `CONFLICT`s hardens to `OPEN_CONFLICT`; the reference level); `M5.B.3` (`REDUCE_CUTOFF` moves none of the tie's held anxiety, D7; the third party's cost is the lost outlet, not shown at this revision: `test_m5b3_third_party_cost_is_the_lost_outlet` is a strict expected failure); `M6.3g` (grief sized by `M1.F.6a`, `grief_weight` 1.0 in place of `grief_scale` 20); `M6.4` (the tie's new sources and sinks by name; `conflict_binding`; the symptom hold's split); `M7.D.2a` (the release and the spill split by bond energy among the bearer's interactive ties to living members, G2); `M11.G.2` (onset loads left out of the stress term); `M14.A` (`cutoff_peak`, `reference_level`; the `distance_bound_anxiety`, `rejection` and `uncut_state` rows); the mechanism register (cutoff threat, bridging and repair, cutoff decay, onset loads); the phase D exit row (`M11.C.48` run and reported); §0.3's counts.
- **Removed:** `REDUCE_CUTOFF`'s transfer of the tie's stock to the third members of the triads holding it (`M5.B.3`, `M6.4`, from revision 12), and the restoration of `uncut_state` when rejection fades to nothing (`M1.B.3c`, Phase D B4): the tie now waits for a positive contact.
- **Constants** (`config/bowen/constants.md`, each `[I]`, with a `constants_changes.md` row): `conflict_binding_rate` 0.15, `cutoff_stock_cap` 1583, `cutoff_threat_weight` 0.00312, `cutoff_decay_weeks` 260, `cutoff_floor_share` 0.1, `positive_bridge_dissipation` 0.8, `tie_repair_steps` 3, `lcu_scale` 6.9, `grief_weight` 1.0; `grief_scale` retired.
- **From the build's two reviews (2026-10-22, before the hash moved):** `LAUNCH_LOAD` reaches the parents the child leaves (the first build compared against the child's new household); a tie reopening `DISTANT` or `OPEN_CONFLICT` records its stock as the reference level; `FAMILY_ILLNESS` reaches family members only; `M11.C.48`'s baseline became the run standing on the uncut ties in the cutoff's week (the span's longest run is a maximum against one point, biased against the criterion; it is reported beside the arms); a family file declaring a cut-off stock above the cap is refused; `M1.F.6a`'s anchor is the job loss (47 LCU = 324 units), not a spouse's death.
- **From the regeneration's re-sweep (2026-10-23, after the hash moved):** `M11.C.48`'s readout gave a first cutoff with no matched uncut tie a baseline of 0 and kept its run in the treatment arm; measured on the regenerated runs, 139 of 700 first cutoffs on the Phase B family (100 seeds) and 284 of 749 on the twelve-person family (20 seeds); with those left out of both arms the difference is −0.28 ± 0.15 and −0.21 ± 0.18 (2 se) against +0.43 ± 0.16 and +0.44 ± 0.10 as built. The pass recorded that day is not evidence, A11 of the G1 proposal is not met, and the row now says what such a first cutoff contributes is undecided (T-123). The fix is a readout change that moves the engine hash, left for the next move.
- **Not shown:** A6 of the G1 proposal, the third party's cost through the lost outlet (`M5.B.3`). The two-arm test found no rise in the children's anxiety after a scripted bridging over 8 seeds; no cost was written, and the test stands as a strict expected failure for the owner.

### Revision 35 — the revision history moved out of the spec, 2026-10-22 ⟦rev35 · owner decision 2026-10-22, plan_model_ergonomics.md §8 Q2⟧

**Decided 2026-10-22** by the owner: the front matter's status text and the revision sections `### Revision 34` to `### Revision 5` leave the spec for this file, and the spec takes revision 35 for it, wording only. No rule changed; `engine_hash()` is unchanged and no record was rerun. The front matter's `status:` line now reads `APPROVED` with a pointer here; its text as it stood is the last section of this file.

**Extended the same day, on the owner's instruction** (`docs/TODO_archive.md`, "narrative the history phrases miss"): eight phrases were added to `tools/spec_index_rules.json` (`earlier form`, `The text it replaces`, `is replaced`, `The form approved at`, `It replaces`, `As first built`, `revision N removed`, `rev-N form`), and the 44 sentences they match — with the `*Revision N*` prefixes outside requirement bodies, which the tool does not scan, `M11.C.14`'s withdrawn two-sided bound and `M4.C.3a`'s pre-revision-24 floor — were disposed as rows 219–262 of the record, under the sections below. Wording only; one bolded MUST NOT left the spec for the log with `M11.C.14`'s withdrawn bound (§0.3's count 1043 → 1042).

### Revision 34 — the ASR quotations paraphrased, and the stage 1 review's wording fixes, 2026-10-22 ⟦rev34 · owner decisions 2026-10-21 and 2026-10-22⟧

**Approved 2026-10-22** by the owner: the §4 texts of `docs/PROPOSAL — ASR QUOTATIONS (SP-1).md`, as written, and its eight questions, each as recommended; the stage 1 milestone review's wording fixes were decided at its triage on 2026-10-21 (D2–D6, `CHANGELOG.md`). Nothing a run computes changed, so no record was rerun. Revisions 13 to 33 are recorded in the front matter's status line and in `CHANGELOG.md`, not here.

| Item | Spec change |
|---|---|
| SP-1 | The 1979 tapes and the Kerr–Bowen interviews are ASR transcripts, cited and not quoted. The spec's quotations of them, in the proposal's 47 spec entries (SP1-S01 to SP1-S47), are replaced by the approved paraphrases, each with its citation. The KB01 interview is credited to Bowen by name; the Hijack-set interviews (KB02–KB15), including KB03 and KB04, are cited as the interview (Q5). Written-source quotations stay |
| Q3 | `M11.F.3` restated from *Family Evaluation*'s Epilogue (FE11.2, Bowen's own writing) in place of the KB13 interview; it now records three statements, the third being that Bowen declined the debate (alternative B). A requirement change, approved |
| Q8 | `M4.D`'s *Family Evaluation* quotation is kept and traced to a new finding, `family_evaluation/fe03.md` · FE03.21; the revision 9 entry below cites it |
| MR-10 | `M11.4` drops its count ("Three criteria"); the table is the list (`M11.4b`) |
| MR-12 (D4) | `M8`'s preamble, `M8.5` and `M13.1`: three mechanisms call the predicate; `M5.C`'s `avoidance_available` is defined and tested but not called. `M13.1` stays met |
| MR-13 (D6) | `M2.A.0i` records Dr Halim's declaration (revision 28); Teodor's row drops "awaiting owner review" |
| MR-15 | The front matter's closing clause, "no code until the implementation plan is also approved", is deleted: it was written before the first phase was planned, and Phases B, C and D and version 2's stage 1 have been built since |
| D3 | `M11.1`'s scope is the `M11.C` criteria |

**Inline narratives moved from the bodies at revision 35:**

- `M4.D.1a`: *Revision 34: the KB10 passage is paraphrased and the* Family Evaluation *quotation cites its finding (SP-1).*
- `M11.1`: *Revision 34:* the scope is the `M11.C` criteria.
- `M11.4`: *Revision 34:* the count this sentence gave ("Three criteria") is dropped; the table is the list, as `M11.4b` already said (MR-10)
- `M11.F.3`: *Revision 34:* restated from the written Epilogue (FE11.2) in place of the KB13 interview, which is ASR text; the third statement, declining the debate, is new (SP-1 Q3, alternative B)
- `M13.1`: *Revision 34:* three call it; `M5.C`'s site is not built (`M8`'s preamble).

**Moved from the bodies 2026-10-22, the phrases added that day (`docs/spec_revision_35_dispositions.md`, rows 219–262):**

- `M8 (preamble)`: *Revision 34:* three mechanisms call it — the witness test (`M8.6`, `visibility.py`), the coach's position (`M11.C.7`, `external.py`) and triangle activity (`M1.C.3`, `recompute.py`).

### Revision 33 — inline narratives moved from the bodies at revision 35

- `M8.5`: *Revision 33:* whether a triangle is active is decided by presence, not by this predicate's reading of a side: `recompute_triangles` counts each present occupant, and a coach who has taken a side is present, so a triangle that holds a sided coach and a recent `TRIANGLE` act is active while `coach_live` is false for it.
- `M13.2`: *Revision 33:* the frozen grid engine (`src/engine.py`, its entry point `src/main.py` and `monte_carlo.py`) and its 37 tests were retired at version 2 stage 1.

### Revision 32 — inline narratives moved from the bodies at revision 35

- `M4.C.1c`: *Revision 32, the nodal-event spike built* (the owner, 2026-10-17: "cut ties, as TRIGGER"): when a nodal event reaches a person (`M1.A.4k`'s reading of reach), each of their cut-off ties (rejection held either way, `M1.B.3c`) between two living people takes a spike of `nodal_spike_intensity` (1.0, `[I]`, the Phase B script's `TRIGGER` size) on its standing term for the event's week, the next tick, felt by both members as a `TRIGGER`'s is (`event_effects.nodal_spike`).
- `M4.C.1c`: Found missing while diagnosing `M11.C.4`: before it, a nodal event reached a cut tie's term only through `M4.C.1b`'s optimum shift
- `M8.3`: *Revision 32, built:* an external agent has taken a member's side for `triangle_activity_window` weeks after that member's `SPLIT` toward it (`external.coach_side`).
- `M8.5`: *Revision 32:* the coach's position is now decided by it: a contact lands only while the coach holds a live position, by `positions_live`, in every triangle it shares with the recipient (`external.coach_live`, read by `landing_probability`).
- `M11.C.4`: against a matched no-cutoff arm *Revision 32:* the baseline holds the tie open for the whole run, so it is the matched no-cutoff arm.
- `M11.C.6`: *Revision 32, the arm restated* (post hoc, after the trace of 2026-10-17 found the first arm picked children by birth order and run year, comparing raw act counts over spans of different length; Claude's recommendation, pending owner review): one child, in two arms of the same seed, until 20; the treatment with the parents' projection-type acts toward the child as the policy chooses, the baseline with them withheld.
- `M11.C.7`: *Revision 32, built* (plan §4.12): three ordered pairs, joint over one spouse, one spouse over sided, joint over sided, on the Phase C family under `M11.C.17`'s nodal load for 520 weeks.
- `M11.C.9`: *Revision 32, not built, out of version 1* (the owner, 2026-10-17: "Out, version 2"): no rule reads sibling position, so building this is a new mechanism, not a gate (`docs/version_1_dispositions.md`).
- `M11.C.22`: *Revision 32, mixed:* the severe limb stays premise; the mild limb is composite, the net of `M7.D.2a`'s lock-in (+7.3) and other mechanisms (about −6.8).
- `M11.C.29`: *Revision 32, mixed:* the budget limb (`M6.I.1`) is proved by `tests/bowen/test_sinks.py::test_m6i1_only_a_completed_exchange_reduces_the_budget` and reported in the ensemble; the third person's time course is written by no rule, so composite.
- `M11.C.32`: *Revision 32, post hoc:* the mover's level is declared 60 in both arms (`mover_level`), so the calm arm can reach the peak; at the fixture's 40, since revision 16, it reached it in 0 of 200 seeds and both arms read 0 (ad hoc, plan §4.12).
- `M11.C.46`: *Revision 32, built* with the unhelpful third closer to the partner (bond energies 30 with the seeker and 70 with the partner, swapped in the helpful arm).
- `M11.C.46`: *Revision 32, with the nodal-event spike:* the mutation clause is **not met, at the edge**.
- `M11.5`: *Revision 32:* its readout, the others' automatic acts toward the bearer after an onset less a no-onset run, is the learned part, so composite
- `M11.5`: *Revision 32:* the reactive-share limb is premise in the two level cells only; in the two heavier-stress cells it is composite, since the mixing weight reads functional level, not anxiety, and stress reaches it only through the stress response's bounded fall in functioning (`M1.A.5e`)

**Moved from the bodies 2026-10-22, the phrases added that day (`docs/spec_revision_35_dispositions.md`, rows 219–262):**

- `M11.E (M11.C.9 row)`: *Revision 32:* superseded in part, since the criterion was restated to a mid-scale peak that needs no effect size; it is not built, and is proposed out of version 1 because no rule reads sibling position

### Revision 31 — inline narratives moved from the bodies at revision 35

- `M6.I.4`: *Revision 31:* what an over-functioner takes in one exchange **MUST** be scaled by their own room above the stress floor, `s = clip((f − m) / (b − m), 0, 1)` with the floor `m = max(0, b − stress_floor_below_basic)` (`f` their functional level, `b` their basic level), clamped at 0 as the giver's floor and stress's are: full at or above basic, a third at 10 below for a basic level of 15 or more, none at the floor.
- `M7.E.1c`: The other four situations are not applied at `t0`, because no history before `t0` is declared to evaluate them against (accepted by the owner, plan §4.8; recorded here at revision 31)
- `M7.E.1f`: *Decided at revision 31* (the owner, 2026-10-16, after the review): the band is no longer a flat ±2%.
- `M7.E.1f`: *Built at revision 31 part B:* from 5 to 20 the acts either living parent sends the child are counted, each one, by whether it is projection-type (`OVERFUNCTION`, `PURSUE`, `TRIANGLE`, the `projection` set of `config/bowen/pattern_acts.md`) or supportive (a self-directed move, `config/bowen/event_kinds.md`).
- `M11.4`: *Revision 31:* a **one-sided** bound in place of the null: each technique, against a no-technique arm, **MUST NOT** lower the child's load by more than the equivalence bound
- `M11.C.2`: the witness path, M1.F.5 (*revision 31:* on the twelve-person family the witness path removed survives; the pass is proved by removing `M1.B.8a` with `M4.D.1g`)
- `M11.C.6`: *Restated at revision 31* (the owner, 2026-10-16, review option A): its second half, that the decline be achievable within roughly ten generations ("**It could take eight to ten generations or as few as three or four**", KS11.1; `FE08.2`), is withdrawn.
- `M11.C.6`: Under `M7.E.1f`'s band as decided at revision 31, a child falls at most 10% below the parents, and only under projection, so ten generations can fall at most from 100 to about 35.
- `M11.C.14`: *Restated at revision 31* (the owner, 2026-10-16, review option B; Bowen: "any and all 'management approaches' were equally unsuccessful", `ch04.md`): under high marital distance, **no technique lowers the child's symptom load against a no-technique arm**.
- `M11.C.14`: *Built at revision 31 part B, with the owner's baseline (2026-10-16, option a):* a scripted act is the parent's whole week's act, so a technique also removes the parents' own acts that week, toward the child and toward each other (measured: against parents under their own policy, firmness −6.4 ± 3.6 and talking it out −17.3 ± 2.9, through less witnessed marital conflict, not through the gate).
- `M11.C.21`: *Revision 31* (the owner's decisions of 2026-10-16, plan §4.11): the collapse did not reach the exchange, so `M6.I.4` gained the capacity rule (a collapsed over-functioner takes in proportion to their room above the stress floor).
- `M11.C.23`: *Revision 31: reclassified as a composite, its failure accepted as a finding* (the owner, 2026-10-16; `M11.5`).
- `M11.C.40`: *Revision 31* (the owner, 2026-10-16, option a): clause 1 gates; clause 2 is reported only and proved by `tests/bowen/test_learner.py::test_m4d6d_self_directed_channel_is_never_reinforced`.

**Moved from the bodies 2026-10-22, the phrases added that day (`docs/spec_revision_35_dispositions.md`, rows 219–262):**

- `M11.C.14`: Under high marital distance a firmness arm and a talking-it-out arm **MUST NOT** differ in the child's symptom load by more than the declared equivalence margin (`M11.4`) ⟦rev15 · owner decision 2026-10-11 · `docs/PROPOSAL — PHASE D DEFINITIONS.md`⟧
- `M11.C.14`: The two-sided comparison above is withdrawn ⟦rev31 · owner decision 2026-10-16, review⟧
- `M11.C.14`: (*revision 31:* the gate removed **MUST** turn at least one technique variant red)

### Revision 30 — inline narratives moved from the bodies at revision 35

- `M4.D.6f`: A person's own cutoff is principled when declared (`M1.B.3d`) or, since revision 30, in proportion to the danger they feel from the person cut off (`M1.B.3e`).
- `M11.5`: | M11.C.47 | Premise | `M1.B.3e`'s felt danger, read from received intrusion | new at revision 30: the rule reads danger from intrusion, and level and stress produce intrusion by the existing mechanisms, so a pass confirms the code and is never reported as a finding

### Revision 29 — inline narratives moved from the bodies at revision 35

- `M1.B.3c`: *Revision 29:* the reconnection pull's "too little" side does not act on a person who holds a principled cutoff on the tie (`M1.B.3d`).
- `M4.C.1c`: *Revision 29:* a person holding a principled cutoff (`M1.B.3d`) **MUST NOT** feel this term on that tie.
- `M7.B.1a`: *Amended at revision 29.*
- `M7.E.1c`: *Revision 29:* a declared `weakness` is the reality defect (`M1.B.8a`), and it **MUST** act from `t0` for a child present then as well as from birth, through `M1.B.8a`'s weekly shortfall rather than as a situation here.
- `M11.C.2`: *Amended at revision 29 for the twelve-person family* (the owner, 2026-10-13: "Child load is by age"). Projection-type acts **MUST** be counted per year while the child is under `projection_until_age`, and symptom load read as the child's mean over ages `load_from_age` to `load_to_age`.
- `M11.5`: From revision 29 both paths read outcomes: `M4.D.6f` learns from what followed a witnessed cutoff, and `M1.B.3d` raises the cutter's anxiety.

### Revision 28 — inline narratives moved from the bodies at revision 35

- `M2.A.0a`: Before revision 28 every agent in M2.A was past it, so all twelve carried a supplied value; the *derived* path applies only to agents born during the run.
- `M2.A.0a`: *Revision 28:* with the starting ages of `M2.A.0j`, Leo (9), Nadia (4) and Pia (1) are below the fixation age at `t0`.
- `M2.B.2`: *Revision 28:* the owner confirmed it: "Coach is tied to Coachee."
- `M3.D.1`: *Revision 28:* step 2 also delivers the life events the last slow tick queued (deaths, births, partnering, grief, lost income); step 7 is taken only by people of acting age who are not inactive (`M7.C.1f`); step 8 ends with the search for work after a lost income (`M7.C.1k`); the life course itself runs on the slow tick, after the slow tick's invariants (`M7.C.1`).
- `M6.I.4`: *Revision 28:* a living person's balance on a tie to the dead settles back as `M6.3e` states, by the same conserved transfer against the dead person's ledger
- `M6.3`: *Revision 28: the disposition.*
- `M7.E.1`: *Revision 28:* for a child, `M7.E.1f` replaces the estimator from 5 to 20; the estimator governs from 20, as for any adult.
- `M11.C.6`: *Revision 28:* limit (a) could not be met under `M7.E.1f`'s ±2% band; the owner decided at revision 31 (above)
- `M11.G.4`: *Revision 28:* children are now born in the run, but component 4's projection line is still not read; it is reported as not built

**Moved from the bodies 2026-10-22, the phrases added that day (`docs/spec_revision_35_dispositions.md`, rows 219–262):**

- `M2.A (table notes)`: **Revision 28 — the life course.**

### Revision 27 — inline narratives moved from the bodies at revision 35

- `M6.I.4`: *Revision 27:* borrowed pseudo-self settles back when the exchange stops — on a tie the pair has exchanged on (its functioning balance is not zero), whose over-functioning member carries borrowed pseudo-self and the other lent, and between whom no `OVERFUNCTION` or `UNDERFUNCTION` was sent for `exchange_window` weeks (8, `[I]`), `pseudo_self_relaxation_rate` (0.05, `[I]`) of the smaller of the two moves back from the one who borrowed to the one who lent, with the same of `functional_level`; total pseudo-self is conserved.
- `M11.C.17`: *Measured at revision 27 (ad hoc, 150 seeds):* the coached rise nearly vanished, +0.0009 ± 0.0008 (p 0.011; +0.021 ± 0.009 at revision 23); without revision 27's settling back +0.005 ± 0.002, with the symptom integrand read above the chronic floor +0.0018 ± 0.0010

### Revision 26 — inline narratives moved from the bodies at revision 35

- `M4.D.6e`: *Revision 26, built:* the cross-person term is the others' relief weighted by the actor's bond energy on its tie to each, so a child's tie to a parent outweighs a thin tie to a witness; a person with no tie to the actor carries no weight.
- `M11.C.2`: *Measured at revision 26 (ad hoc, 100 seeds):* **passes** — the child who received the most projection-type acts carries +1.73 ± 1.22 more symptom load than the one who received the fewest (p 0.003).
- `M11.C.2`: *Measured at revision 26 (ad hoc, 100 seeds):* **fails** — with projection-type acts counted in the first five years and symptom load read in the last five, the most-projected child carries no more (+0.29 ± 2.20, p 0.39). Read over the same weeks it had passed (+1.31, p 0.045), which cannot separate cause from attraction

### Revision 25 — inline narratives moved from the bodies at revision 35

- `M1.A.11c`: *Revision 25, built:* the family-focus term is written from behaviour.
- `M4.B.2a`: *Revision 25:* built as an engine-side gate on the effect of moves, not a policy read, so it reads the tie itself
- `M5.C.1`: *Revision 25, built:* while both spouses' felt contact on the family's marital tie is below `marital_distance_fraction` (0.25, `[I]`) of their optimum, a move addressed to one of the couple's children with an active symptom cannot lower their acute anxiety, and no calm transfer (`M4.C.10`) reaches them (a symptomatic spouse is not gated); engine-side, so it reads the tie's true state
- `M9.6`: *Revision 25, built:* each person holds `attribution`, a weight per family member summing to at most 1, empty at the start.
- `M9.7`: *Revision 25, built for the witness:* a witness's appraisal of a load passed between two people it is not party to is scaled by `1 + belief_appraisal_gain × believed tension` of that tie (the most tense, for several targets; a relief is not scaled) (`M9.8`; `belief_appraisal_gain` 1.0, `[I]`).
- `M11.C.14`: *Measured at revision 25 (ad hoc, 100 seeds):* **fails** — talking it out leaves the child 8.0 ± 1.9 points less symptom load than firmness, outside the equivalence bound (5.6); with the gate removed the arms differ less (−6.7 ± 1.9)

### Revision 24 — inline narratives moved from the bodies at revision 35

- `M1.A.6`: *Revision 24:* the threshold reads `functional_level` floored at `functional_level_floor`, as steepness does (`M4.C.1a`), so a person at functional level 0 has a small threshold, not none (symptom severity and channel protection divide by it).
- `M1.A.11c`: *Revision 24, built:* the expressed channel is the one with the largest `constitution_strength × [c = prior] + (1 − constitution_strength) × channel_focus[c]`, ties to the prior; `constitution_strength` is the person's own value or the constant (0.5, `[I]`), and 1 overrides any focus.
- `M1.F.9`: *Revision 24:* a symptom channel's release ends the symptom, and what it held of the family's anxiety is its hold (`symptom_bound`, `M7.D.2a`), which returns to the partners on interactive ties; its load, an integral of weeks above the usual level and not an anxiety stock, is not credited to the budget (review 2026-10-12: crediting it put 77 points into the budget on one release while the hold was 27).
- `M4.C.3a`: *Revision 24:* the floor is the person's usual level (`M1.A.5e`, revision 22): the integrator accumulates the excess above the person's usual excess, where positive.
- `M7.D.1`: *Revision 24:* every channel is checked for onset each week, not only the one being fed: load left in a channel can cross when functional level falls, and each crossing is its own onset event.
- `M7.D.2`: *Revision 24, built:* the load goes to the expressed channel (`M1.A.11c`); every channel leaks.
- `M7.D.2a`: *Revision 24, built:* each person holds `symptom_bound`, the anxiety the family's focus on their active symptom holds (a stock in `M6.4`).
- `M7.D.2c`: *Revision 24, built:* what enters a channel each week is divided by `1 + channel_protection × Σ (load in the other channels) / threshold` (`channel_protection` 1.0, `[I]`).
- `M7.D.2d`: *Revision 24, built:* past the turn (`s < 0`), `min(1, |s|)` of the hold goes back to the partners each week, and the same share of the bearer's own excess above usual spills to them, split equally: the symptom no longer holds what it held, nor the bearer's own anxiety.
- `M7.D.4`: *Revision 24, built:* each week, if a living member has an active symptom, and another member on an interactive tie to the member carrying the most active symptom load has excess above usual greater than their `tolerance` (`M1.D.5`, declared per person; none listed, never), that member sends an endogenous `INSTITUTIONALIZE` of the bearer.
- `M11.C.22`: *Restated at revision 24:* removing a member relieves the family a little whether or not they carry a symptom (measured ad hoc, 2026-10-12: −0.83 ± 0.76 points of the others' mean excess over 26 weeks, with no symptom), so each limb is read against removing the same member with no symptom.

**Moved from the bodies 2026-10-22, the phrases added that day (`docs/spec_revision_35_dispositions.md`, rows 219–262):**

- `M4.C.3a`: The integrator **MUST** accumulate `acute_anxiety − chronic_anxiety` where positive, so that **many resolved excursions and one unresolved excursion are distinguishable**.
- `M4.C.3a`: Measured before it (2026-10-12, ad hoc): above the chronic floor, the family's own traffic held everyone 15–35 points up, so in ten-year runs of the nuclear-four fixture every person crossed the threshold in the first weeks and stayed there, at 4 to 31 times it, and no symptom criterion could tell its arms apart.
- `M4.C.3a`: The integrand is still time above a floor, not peak, so resolved and unresolved excursions stay distinguishable; what changes is that a family's steady state is no longer a symptom in itself.

### Revision 23 — inline narratives moved from the bodies at revision 35

- `M1.A.4k`: *Revision 23:* the history **MUST** read pseudo-self with its sign — `functional_level` less pseudo-self, what was given way added back as well as what was borrowed left out.
- `M11.C.20`: *Restated at revision 23:* before it this was the same mutant as reading `functional_level` whole; it no longer is.

### Revision 22 — inline narratives moved from the bodies at revision 35

- `M1.A.5e`: *Revision 22, calm is relative to the person:* in the stress response and in `M1.E.7`, "loaded" and "calm" **MUST** be read against the person's own usual level — the plain mean of their excess anxiety over the weeks so far for the first `1 / usual_level_rate` weeks (`[I]`), then a running average over about a year — not a fixed band.
- `M1.E.7`: *Revision 22:* "under load" here, and the calm in which the adjustment settles, are read against the person's own usual level (`M1.A.5e`).

### Revision 21 — inline narratives moved from the bodies at revision 35

- `M1.B.5`: *Revision 21, the exchange takes two:* an `OVERFUNCTION` or `UNDERFUNCTION` delivered to a person **holding a position** toward its sender **MUST NOT** move functioning, pseudo-self or the balance: the person neither gives way to being taken over nor takes over for someone who under-functions toward them.
- `M11.C.20`: *Revision 21:* a person holding a position does not take over (`M1.B.5`), so part of the borrowed arm's scripted support can move nothing; the arm keeps the rule and reports that share (`support_blocked`, report only)

### Revision 20 — inline narratives moved from the bodies at revision 35

- `M1.A.4k`: *Revision 20:* the history **MUST** also leave out any positive `adjust_swing` — the gain taken when someone else's held stance worked (`M1.E.7`).
- `M1.E.7`: *Revision 20, it settles back:* the other's gain **MUST** be tracked apart (`adjust_swing`) and **MUST** settle back when that person is calm, by `response_relaxation_rate` a week, as the stress-moved swing does (`M1.A.5e`); while they are loaded it holds.
- `M11.C.18`: *Restated at revision 20:* it named `M1.A.4b`'s load weighting, and flattening that to an unweighted count leaves this passing (measured ad hoc, 2026-10-11); the stress response is what carries it.
- `M11.C.20`: *Restated at revision 20:* it named `M1.A.4b`'s load weighting, and flattening that to an unweighted count leaves this passing (measured ad hoc, 2026-10-11); the borrowed exclusion is what carries it

### Revision 19 — inline narratives moved from the bodies at revision 35

- `M1.A.5e`: *Revision 19:* stress **MUST NOT** push `functional_level` below `basic_level − stress_floor_below_basic` (15, `[I]`).
- `M1.E.7`: *Revision 19, amended:* the practice **MUST** be self-discovered — it counts only when the self-directed act worked.
- `M1.E.7`: *Revision 19, the system adjusts:* when a held stance works, the other **MUST** function up (`system_adjust_gain`, `[I]`).
- `M4.D.1a`: *Revision 19:* perspective **MUST** also steer within the channels: the weight of each act listed in config is scaled by `1 ± perspective_steer_gain × perspective` — away from `TRIANGLE` and `CONFLICT` ("don't triangle … not to blame", the owner), toward `DETRIANGLE` and `PREVENT_ALIGNMENT` ("know to detriangle").
- `M4.D.6`: *Revision 19, the soothing review:* now and then a person **MUST** be able to ask whether soothing helped — every `soothing_review_every` weeks, with a probability rising with `basic_level` and perspective — and, where their load weeks after soothing was no lower than when they soothed, the learned value of their avoiding and soothing acts **MUST** fall.
- `M11.C.17`: *Revision 19:* the readout is the mean rise of the people the coach is in contact with, over ten years (coaching lands slowly and reaches only them); the whole family's mean is reported
- `M11.C.20`: *Revision 19, the arms* (owner decision 2026-10-11): the basic arm's person has systems perspective, which makes the sequences differentiating; the borrowed arm's does not, and simply leans on the partner

### Revision 18 — inline narratives moved from the bodies at revision 35

- `M1.E.7`: *Revision 18, amended:* it **MUST** also rise, more slowly, by the person's own practice — each self-directed act or `WITHHOLD` made under load adds `self_practice_gain` of its distance from 1 (0.001, `[I]`, against a landed contact's 0.1).
- `M1.F.6`: *Revision 18:* an exogenous stressor **MUST** weigh on its targets for its declared duration — after its arrival, each later week adds a declared share (`stressor_persistence`, `[I]`) of the arrival's load — so a short stressor and a long one differ by the weeks they weigh (the owner: "partner mad for 2 turns vs. no job for 15 turns").
- `M4.D.1a`: *Revision 18, amended:* the mixing weight **MUST** read `systems_perspective` as well as `functional_level`: `w = f + perspective_channel_gain × perspective × (1 − f)`, with `f` as before, so a person with perspective chooses the self-directed channel more often at the same level, and one with none behaves exactly as before.

### Revision 17 — inline narratives moved from the bodies at revision 35

- `M1.A.4k`: *Revision 17, what a nodal event is:* any major event on a person — the owner: "Death, births, marriages, serious illnesses, big moves, big job changes, child moving out or back in, divorces … Any major event on a person (a node) is a nodal event."
- `M1.A.18c`: *Revision 17:* the coach-free arm may also rise, where its members' own self-directed behaviour under load shows it (`M11.C.17` rewritten)
- `M5.A.1a`: *Revision 17, cutoff read from behaviour* (the owner: "The designation of cutoff emerges from behaviour. An agent saying 'I'm not going to contact you again' is not cutoff, it's just words that need behaviour to back it up"): member A's cutoff of B **MUST** be read from what A does — B's approaches refused (by the engine's refusal, `M1.B.3c`, or answered with `CUTOFF` or `DISTANCE`) at least `pattern_min_acts` times in the window, with no act with contact from A to B — and a `CUTOFF` act alone **MUST NOT** read as cutoff.
- `M11.C.7`: *Revision 17, the direction* (owner decision P9, 2026-10-11): three arms with the coach's parameters and schedule fixed — **joint** (sessions with both spouses, siding with neither), **one spouse** (sessions with one, siding with neither) and **sided** (drawn onto one spouse's side).
- `M11.C.17`: *Rewritten at revision 17* (the owner: "A person CAN improve functioning and level of differentiation without a coach. A coach can help, but isn't required. The evidence is in the behaviour").
- `M11.5`: | M11.C.17 | Composite | … | rewritten at revision 17: no rule writes "coached rises more"; it follows from the coach raising `systems_perspective`, which opens the self-directed path the estimator reads |

**Moved from the bodies 2026-10-22, the phrases added that day (`docs/spec_revision_35_dispositions.md`, rows 219–262):**

- `M11.C.17`: *It replaces* the earlier "differentiation is not self-generating … the coach-free arm MUST NOT show upward drift".

### Revision 16 — inline narratives moved from the bodies at revision 35

- `M1.B.3c`: *Revision 16, graded rejection:* a move **MUST** be refused only while the cutter's rejection is above a declared threshold, and below it is received as usual; rejection **MUST** fade by a declared amount each week, so a cutoff that is not renewed by refusals or further cutoffs wears off.

**Moved from the bodies 2026-10-22, the phrases added that day (`docs/spec_revision_35_dispositions.md`, rows 219–262):**

- `M1.B.3c`: As first built, refusal at any rejection above zero, with only `REDUCE_CUTOFF` lowering it, left 80–88% of the Phase C family's tie-weeks cut off and nearly removed `REDUCE_CUTOFF` (`docs/implementation_plan_phase_d.md` B4, measured 2026-10-11)

### Revision 15 — inline narratives moved from the bodies at revision 35

- `M11.C.14`: *Revision 15, defined:* a **technique** is a parent's way of handling the symptomatic child, scripted as the same act toward the child each week: firmness `OVERFUNCTION`, permissiveness `UNDERFUNCTION`, punishment `CONFLICT`, talking it out `STAY-IN-CONTACT` (Bowen: when the parents are close "they could do no wrong"; when emotionally divorced "any and all 'management approaches' were equally unsuccessful", `ch04.md`).
- `M11.C.23`: *Revision 15, defined:* the **EE term** is the existing appraisal path for what reaches the patient — the impingement of `CONFLICT` received (hostility), blame-valenced acts toward the patient (critical comments), and the mother's investment in her tie to the patient weighting the impingement she delivers (over-involvement); nothing new is written.
- `M11.C.28`: *Revision 15, defined:* the **allocation-mobility term** is the decay rate of the activation memory that holds the family's focus where it last was — on a triangle (`M1.C.4`) and on the sink allocation's target (`M1.D.1`).
- `M11.5`: | M11.C.46 | Premise | `third_help` (`M1.C.1`) | new at revision 15; the rule writes the relief, so a pass confirms the code and is never reported as a finding |

### Revision 14 — inline narratives moved from the bodies at revision 35

- `M1.A.7a`: *Superseded in part at revision 14 by `M7.B.1`: after childhood it moves only with a rise in `basic_level`.*
- `M2.A.0b`: *Revision 14:* `t0` is the starting point, week 0, and is not a slow tick; the first slow tick falls at the end of week 51.
- `M4.C.1c`: *Revision 14, the reconnection pull:* while a tie is cut (`M1.B.3c`), its "too little" side **MUST** keep growing for both members, since contact is refused, and an accepted approach **MUST** relieve it.
- `M7.B.1`: *Revision 14:* after that it **MUST** change only when `basic_level` rises, falling with the rise; it **MUST NOT** drift otherwise, and **MUST NOT** rise when `basic_level` falls (the owner: "chronic anxiety stays fixed unless level of differentiation goes up, thus it's fixed for some time").
- `M7.E.4`: *Revision 14:* the decay **MUST** be driven by the two members' chronic anxiety (the owner: chronic anxiety "wears on ties, which is why the other emotional patterns act up from time to time, often triggered by other stressors"), and the bond-energy accrual is withdrawn, since `M1.B.4` holds bond energy at or near zero decay.
- `M11.C.13`: *Revision 14: moved out of Phase D; it needs a community to locate incidents in, which the model does not have, and returns when one is modelled*

### Revision 13 — inline narratives moved from the bodies at revision 35

- `M2.A.1`: *Revision 13: Dario replaces Bruno, who left the table; Phase C's `M11.C.4` ran on its own dyad fixture and is unaffected.*
- `M2.B.2`: **M2.B.2** *Amended at revision 13* (the defining line's title).
- `M5.A.1a`: *Revision 13:* the owner gave the readouts' definitions on 2026-10-10 (`docs/readouts_phase_c.md`).

**Moved from the bodies 2026-10-22, the phrases added that day (`docs/spec_revision_35_dispositions.md`, rows 219–262):**

- `M2.A (table notes)`: **Revision 13 — the owner's reference family.**

### Revision 12 — the Phase C prerequisites, 2026-10-07 ⟦rev12 · owner decision 2026-10-07⟧

**Approved 2026-10-07** by the owner: every recommendation of `docs/implementation_plan_phase_c.md` §1 and §2, as written. This section records the spec changes from §1's P1–P10 and P12. P11 (closing Phase B with the owner's read of the rendered traces) is an action, not text. §2's build decisions D1–D10 live in the plan, not here; they choose forms the spec leaves open.

| Item | Spec change |
|---|---|
| P1 | `M6.I.6` restated: transfers balance, every change is logged against a named source, sink or transfer. New `M6.4`, the stock-and-flow table. `M6.I.1` amended: the budget is reduced only by a completed differentiating exchange. `M4.G.2a`'s hold-back ends with Phase C |
| P2 | New `M4.B.2a`, `M4.B.2` row by row. `M5.F.1` amended: `outside_ness` acts through the event. `M5.C`'s system-calm row annotated |
| P2a | `M9.8`'s belief store built in Phase C |
| P3 | `M11.C.17`, `M11.C.18` and `M11.C.20` moved to Phase D. `M7.D.1` built in Phase C, channels fixed at their constitutional prior until Phase D. `M11.C.4`'s nodal event scripted in Phase C |
| P4 | `M5.D.6`'s next day is the next fast tick. New `M5.D.9`: a multi-tick sequence and one outcome per tick. Sub-week explainer durations are not calibration targets |
| P5 | Explainer §17.3 and `_STATUS.md` corrected to §17.1's form: cancellation holds only for error that does not interact with the intervention |
| P6 | Six explainer passages brought into line (capacity floor, the transition at 50, the `systems_perspective` ceiling, symptom exclusivity, §11 test 9, "unbounded") |
| P7 | `M11.D.16`'s symmetric-family test holds in distribution, not byte for byte |
| P8 | `M1.F.2`: the sign comes from the event's components relative to the receiver's optimum; no hard-coded flip |
| P9 | `M5.C`'s `outside_ness` gate has one threshold per axis. `M5.D.5`'s mutual rise is solid self. `M5.B.3` reconciled with `M5.B.3a`, and `REDUCE_CUTOFF`'s cost lands on the third party (`L22.6`). `M11.C.27` listed at `M11.4`. `M11.C.38` sweeps `chronic_anxiety`'s initial condition and is built although a SHOULD |
| P10 | New `M5.B.6`, `PROVOKE`. Bowen's two falsifiers (`L09.2`, `L16.2`), `L15.1`'s delayed channel, `L03.4`'s two-rung ladder and `L03.5`'s illness in the mover are recorded for the Phase D plan, not entered here |
| P12 | The paper material held in `TODO.md` under the stale "revision 11" label is relabelled "held for a later revision" |

**Phase C now gates 19 acceptance criteria.** Of them, by `M11.5`: 6 composite, 10 premise, 2 check, 1 mixed. Phase D gains `M11.C.17`, `M11.C.18` and `M11.C.20`.

**Inline narratives moved from the bodies at revision 35:**

- `M1.F.2`: *Revision 12:* the sign comes from the event's contact and impingement components relative to the receiver's optimum (`M4.C.1`), so no event kind carries a hard-coded flip.
- `M4.D.1a`: *Revision 12, Phase C step 7:* the self-directed channel also carries the four family moves of `M5.B` — `DETRIANGLE`, `PREVENT_ALIGNMENT`, `REDUCE_CUTOFF` and `PROVOKE` — scored by the same position gap as `I-POSITION` (`M5.F.5`); plan D3 had placed them in neither channel.
- `M4.G.2a`: *Revision 12:* `M6.I.6` is restated and `M6.4` holds the table, so from Phase C `M4.G.2` asserts every `M6` invariant, `M6.I.6` included.
- `M5.B.3`: *Revision 12, reconciling with `M5.B.3a`:* more contact on a severed tie is never penalised **as contact**, but it can carry the receiver past its optimum on the impingement side (`M4.C.1`), which is `M5.B.3a`'s bound.
- `M5.C.1`: *Revision 12:* two thresholds, one per axis of `M1.A.9a`, both `[I]`; the gate fails on either
- `M5.C.1`: *Revision 12:* the triangle clause is superseded by amended `M1.C.3` and is not a gate; the projection clause is Phase D (`M4.B.2a`)
- `M5.D.5`: *Revision 12:* both parties rise, so the gain is **solid self**, which `M6.I.5` exempts from `M6.I.4`'s zero-sum exchange.
- `M5.D.6`: *Amended at revision 12:* "the next day" is **the fast tick after `RESOLVE`** — the tick is a week (`M3.A.1`) and latency is whole ticks (`M3.C.2`).
- `M5.F.1`: *Amended at revision 12:* receivers "read" it only through the event — the actor's `outside_ness` scales the impingement component the event delivers (`M4.C.1`) — and **MUST NOT** read the field itself (`M4.B.2a`).
- `M6.I.6`: | **M6.I.6** | **Restated at revision 12.**
- `M7.D.1`: *Revision 12:* built in Phase C, because `M4.C.3`'s integrator is and six Phase C criteria read time to threshold.
- `M9.8`: *Revision 12:* this store is built in **Phase C**, written only from witnessed events by an `[I]` smoothing rule, because `M4.B.2` binds from Phase C and `M4.D.2` reads triangle position through it.
- `M11.C.38`: *Revision 12:* for `chronic_anxiety` the swept quantity is its **initial condition** (`M2.A.0a`), not a parameter, because `M1.A.7a` derives it each slow tick; and although this is a SHOULD, it is listed in Phase C's Done-when cell and is therefore built
- `M11.D.16`: so a permutation reassigns draws (external review E; amended at revision 12)
- `M14.A.4`: | `symptom_load` | Person | DV | symptom accumulation, the chronicity integrator (`M4.C.3a`, `M7.D.1`) — Phase C by revision 12's P3 | C |

**Moved from the bodies 2026-10-22, the phrases added that day (`docs/spec_revision_35_dispositions.md`, rows 219–262):**

- `M6.I.6`: *The earlier form — "anxiety is conserved and redirected, never destroyed" — failed every tick, because appraisal creates acute anxiety and decay removes it (external review A3)*

---

### Revision 11 — what is set, what is random, what emerges, 2026-10-06 ⟦rev11 · owner decision 2026-10-06⟧

**Approved 2026-10-06** by the owner: A1 (decided first), A2 with A2.1, A3 option (a), A4, A5 and A6 of `docs/proposal_rev11_emergence.md`. Every requirement this revision adds or amends carries a ⟦rev11 · …⟧ marker naming the item. The rationale is in `model_explainer.md` §19; the proposal is the working record.

**Why.** The owner's concern: a model told how families behave will behave that way and can tell us nothing. Read against `M11.1d`, 2 of the 41 criteria at revision 10 could produce a result no single rule states. The principle adopted has three layers. **Set:** two given drives — toward connection, and away from being smothered, controlled or losing autonomy — and what each act does to the state of the system. **Random:** exogenous events, selection noise, initial conditions, visibility. **Emergent:** the patterns Bowen recorded — conflict, over- and underfunctioning, distance, cutoff, projection, fixed triangles — produced by learning from felt relief, by agents that do not know the consequences of what they do.

| Item | What changed |
|---|---|
| A1 | `M4.D.6a` reversed: the automatic channel learns from felt relief over a short horizon. `M4.D.6b` (horizon swept with learning rate and temperature; the long-horizon requirement withdrawn), `M4.D.6c`, `M4.D.1a`'s note, `M4.G.3` (its condition is now met, so it binds) and `M11.C.16` (restated) follow from it |
| A2, A2.1 | `M4.C.1` re-derived as two-sided deviation from a felt contact optimum, with `M4.C.1a`–`M4.C.1c`; the standing load (`M4.A.1`) becomes the "too little" side of the same function; `M3.D.2` annotated; `M11.C.27`'s mutation target moved off the prior-state term |
| A3 | `M5.A.1a`: each core move is one act; the pattern of the same name is a readout |
| A4 | `M1.C.2`, `M1.C.3`, `M1.C.4`, `M4.D.3`, `M4.G.1`, `M7.E.1`, `M7.E.1c` amended to make their result emergent; `M5.E.0` makes the change-back ladder a learned response; `M7.D.3` and `M7.E.3` **withdrawn as mechanisms**, kept as assertions. Consequential: `M7.E.1a` becomes a shape check on the output and `M7.E.1e` is met through the mechanism, because a child's level now comes from the estimator. Criteria `M11.C.6`, `M11.C.10`, `M11.C.12` and `M11.C.15` have new mutation targets; `M11.C.42`, `M11.C.44` and `M11.C.45` are new |
| A5 | `M4.C.10`: contact with a calmer person lowers anxiety, on every tie; seeking it is learned, not a move. `M1.E.5` annotated. New `M11.C.43`. The term *social buffering* is outside the corpus and is not cited until sourced |
| A6 | `M4.B.3`: the policy holds no representation of consequences beyond the learning horizon |
| Classification | `M11.5` classifies every criterion as premise, composite, mixed or check, by reading; `M10.C.5` fills the register's premise column from it. `M13` gates the four new criteria |

**Corrections to the proposal made while applying it.** `M11.C.41`'s reactive-share limb stays a premise, not a composite as the proposal said, because amended `M4.D.3` keeps anxiety's effect on the reactive share through `M4.D.1a`'s mixing weight. `M7.E.1a` and `M7.E.1e` were not named in the proposal and had to change with `M7.E.1`. The proposal quoted `K10.7` in support of A1; the spec already flags that quotation as societal in scope, so `M4.D.6a` cites `KS04.13` instead.

**Phase B code that no longer conforms.** Phase B was built against revision 10, and these parts of it now differ from the spec. None was changed at this revision. Each is a Phase C plan item:

| Code | Revision-10 rule it implements | Revision-11 rule |
|---|---|---|
| `src/bowen/engine/appraise_base.py` | `M4.C.1`'s base product | `M4.C.1`–`M4.C.1b` |
| `src/bowen/engine/standing_load.py` | `M4.A.1` as bond energy over functional level | `M4.C.1c` |
| `src/bowen/engine/recompute.py`, the `tension_activation_threshold` test of triangle activity, and `tests/bowen/test_mechanisms.py::test_m1c3_triangles_are_inoperative_when_calm` | `M1.C.3`'s calm-system rule | `M1.C.3` amended: activity is a readout of recent `TRIANGLE` acts |
| `config/bowen/constants.md` and `config/bowen/constants_frozen.md`, the `tension_activation_threshold` row | cites `M1.C.3` as its source | the constant goes with the threshold |
| the run-log header (`src/bowen/io/load.py::load_spec_revision`, `M16.A.1`) | stamps the spec's version line at run time | a run made from this code now records revision 11 although the three rows above implement revision 10. **Until the Phase C rework, a trace or log produced from this code MUST be read as revision-10 behaviour for `M4.C.1`, `M4.A.1` and `M1.C.3`.** The committed review traces under `docs/review/` say revision 10 and are correct. ⟦rev11 · learning-qa finding 5, 2026-10-06⟧ |

**The ledger's case against the old signal, and why it does not stand against the new one** ⟦rev11 · learning-qa finding 9, 2026-10-06⟧. Several ledger entries argue against a short-horizon relief signal: `FE05.4` ("actions are taken to relieve the anxiety of the moment rather than being determined by a long-term view" — regression **defined** as that signal), `FE03.19`, `KS21.1` and `K03.2`. Each argues that **differentiated** action, and a helper's objective, are not driven by relief. Revision 11 agrees and keeps all of it: relief drives only the automatic channel, which `FE05.4` identifies as the regressed mode; the self-directed channel is never scored by relief (`M5.F.5`) and never reinforced (`M4.D.6d`); and the coach's objective stays understanding, not relief (`M1.E.2a`). `FE05.4`'s note that such a signal models "a permanently regressed family" holds only for a family running wholly on the automatic channel. `M4.D.1a`'s mixing weight by level prevents that, and `M11.C.16` measures how far each family goes toward it.

**Not decided at this revision.** The proposal marked five rule groups for review without proposing a change: `M1.A.14c`/`M1.A.14d`, `M1.B.10`, `M4.D.5a`, `M7.D.2a`/`M7.D.2d` and `M9.6`. They stand as written. **Adjacent issue, recorded for revision 12:** `M1.D.1`'s three-sink budget writes where anxiety goes; under this revision's principle it would be accounting over learned patterns.

**The material held in `TODO.md` under "Spec revision 11"** is not part of this revision. Its label predates it and is now stale; it waits for the owner to assign it to a later revision.

**Inline narratives moved from the bodies at revision 35:**

- `M1.C.2`: **Amended at revision 11: the inversion is a result, not a rule.**
- `M1.C.3`: **Amended at revision 11: as a result, not a rule.** (the second sentence of `M1.C.3`'s body, after "Triangles **MUST** be inoperative when the system is calm.")
- `M1.C.4`: **Amended at revision 11: as a result, not a rule.** (the second sentence of `M1.C.4`'s body, after "tension **MUST** preferentially reroute onto previously-used circuits.")
- `M1.E.5`: *Revision 11:* a special case of `M4.C.10`, which applies the same term on every tie.
- `M3.D.1`: *From revision 11, the "too little" side of `M4.C.1`'s function (`M4.C.1c`).*
- `M3.D.2`: *Revision 11:* `M4.C.1c` now makes the same point through the appraisal function's shape — a severed tie's "too little" deviation builds over time and does not read as relief.
- `M4.A.1`: **Amended at revision 11:** the load **MUST** be computed as the "too little" side of `M4.C.1`'s deviation function (`M4.C.1c`), with bond energy and `functional_level` entering through it; a separate standing-load formula is withdrawn.
- `M4.C.1`: **M4.C.1** — *re-derived at revision 11: anxiety on a tie rises with deviation from the person's felt contact optimum, in either direction.*
- `M4.C.1c`: *Revision 11, finding 6:* the "too little" deviation of a severed or distanced tie **MUST** build over time since contact, at an `[I]` rate, and **MUST** be spiked by a `TRIGGER` or nodal event (`M4.A.2`), rather than arriving in full at the act.
- `M4.D.1a`: *Revision 11:* the automatic channel now learns from felt relief over a short horizon (`M4.D.6a`), so the horizon argument above no longer carries weight.
- `M4.D.3`: **Amended at revision 11: through two routes, neither of them a written weight.**
- `M4.D.3a`: *Revision 11:* the slide **MUST** come through capacity, not through an anxiety weight, so that it is consistent with amended `M4.D.3`: as anxiety lowers `functional_level`, the newer, more complex acts **MUST** lose capacity — they become less available or less effective — exposing the older layer intact (`ks23.md` · KS23.2).
- `M4.D.5c`: *Revision 11:* `M4.D.6a` no longer forbids this signal; the automatic channel learns on it, and this rule is the same claim in one parent.
- `M4.D.6a`: **M4.D.6a** — *reversed at revision 11: the automatic channel's reinforcement signal **MUST** be the actor's felt change in acute anxiety over a short declared horizon.*
- `M4.D.6b`: *Amended at revision 11: the requirement that the horizon exceed `M5.D`'s reaction window is withdrawn with `M4.D.6a`'s old form.*
- `M4.D.6b`: Reinforcement operates on the swing term, short relative to it — *revision 11:* the horizon **MUST** cover the weeks over which an act's felt effect arrives, not the months over which a deferred cost arrives, because the automatic channel does not know deferred costs (`M4.B.3`) ⟦rev11 · learning-qa finding 7, 2026-10-06⟧; *as written before revision 11:* over months; the estimator operates on `basic_level`, over years.
- `M4.D.6b`: *as written before revision 11:* over months (the clause between the rev11 tag and "the estimator operates on `basic_level`, over years")
- `M4.D.6c`: **M4.D.6c** — *amended at revision 11.*
- `M4.G.1`: *Amended at revision 11:* the hardening is physics — each withdrawal lowers contact on the tie.
- `M4.G.3`: *Revision 11:* the condition is met — `M4.D.6a` now credits short-horizon relief — so this requirement binds.
- `M5.F.5`: *Revision 11:* that signal is no longer forbidden in the automatic channel, but it remains forbidden here: the self-directed channel **MUST NOT** be scored by relief.
- `M7.E.1`: **Amended at revision 11: the ordering is a result, not a rule.**
- `M7.E.1a`: **M7.E.1a** — *the transmission rule, stated; a shape check on the output from revision 11.*
- `M7.E.1a`: *As stated before revision 11:* Parents are at the **same** level (M2.A.0c), and "**they can produce children with basic levels a little higher, a little lower, or the same as their levels**" — a **three-outcome distribution centred on the parental level**, with the child's own parental triangle (M7.E.1b) deciding which offset.
- `M7.E.1c`: **M7.E.1c** — *amended at revision 11: five situations raise parental anxiety focused on a child; they do not select the target.*
- `M7.E.1e`: *Revision 11:* met through the mechanism — a child's functional history is driven by its parental triangle, which reads the marital tie — and checked on the output.
- `M10.C.5`: *Revision 11:* `M11.5` holds the classification by reading; the register's premise column **MUST** be filled from it and corrected when `M11.1d`'s mutants are run.
- `M11.C.1`: making it independent of `functional_level` **MUST** turn this red (retargeted at revision 11 from the divisor of rev-10 `M4.C.1`)
- `M11.C.16`: *Restated at revision 11; the former criterion guarded against collapse under a signal `M4.D.6a` then forbade.*
- `M11.5`: The table below is a classification **by reading**, made at revision 11 before any mutant was run.
- `M11.5`: At revision 11: 20 premise, 14 composite, 3 mixed, 7 check, 1 deferred, over 45 criteria.
- `M11.5`: Before revision 11, by the same reading, 2 of 41 were composite.
- `M11.5`: | M11.C.16 | Composite | — | restated at revision 11.
- `M11.5`: | M11.C.42 | Composite | — | new at revision 11.
- `M11.5`: | M11.C.43 | Composite | — | new at revision 11 |
- `M11.5`: | M11.C.44 | Composite | — | new at revision 11 |
- `M11.5`: | M11.C.45 | Composite | — | new at revision 11.
- `M14.A.4`: | Standing load (`M4.A.1`, `M4.A.5`, `M4.C.1c`) | … — the "too little" side from revision 11 |
- `M14.A.4`: | Base appraisal (`M4.C.1`) | … — the change in two-sided deviation from revision 11 |
- `M14.A.4`: | Triangle recompute (`M1.C.3`) | … live positions — no tension threshold from revision 11 |

**Moved from the bodies 2026-10-22, the phrases added that day (`docs/spec_revision_35_dispositions.md`, rows 219–262):**

- `M4.C.1`: *The rev-10 form, `intensity × conductance / functional_level`, is replaced; KS06.3's extraction recommended this re-derivation and it had not been applied.*
- `M4.C.1a`: Division by `functional_level`, the rev-10 form, is one admissible steepness.
- `M4.D.6a`: *The form approved at G3 forbade this signal, on the ground that it rewards the seven reactive moves and punishes the two differentiating ones by construction.
- `M4.D.6a`: The second half does not hold: `I-POSITION` and `STAY-IN-CONTACT` are in the self-directed channel, which `M4.D.6d` keeps out of reinforcement.*
- `M4.D.6b`: → user, G3 approval 2026-08-24; *revision 11 removed the clause tying this to `M11.C.16`'s second direction, which was withdrawn*
- `M4.G.3`: *The text it replaces said the condition was unsettled because `M4.D.6a` then forbade short-horizon relief.*
- `M13 (phase table, row B)`: **Revision 11:** Phase B was built and gated against revision 10's `M4.C.1`, `M4.A.1` and `M1.C.3`.

---

### Revision 10 — the method literature, 2026-09-22 ⟦rev10 · owner decision 2026-09-22⟧

**Approved 2026-10-06.** See *Approval* at the end of this section. The paragraph below is kept as written at revision 10, except that the markers no longer say *proposed*.

**Everything this revision adds was a proposal until approval.** Each requirement, table row, register entry and amendment
ends with a marker of the form ⟦rev10 · C8 · Holland 2026⟧, naming the candidate or design-lessons
item and the first author of the paper it rests on. A requirement without a marker is approved text and was
not touched, except where an approved statement had to be extended — those edits are listed below and are
marked too. At approval the owner kept the markers as provenance and removed only the word *proposed*. The rationale for every marked item lives in
`model_explainer.md` §18, on the owner's decision of 2026-09-22; the candidates and design-lessons files are
the evidence behind it (§0.2).

**What was read.** Four bodies of material, all under `papers/`:

| | File | Contributed |
|---|---|---|
| 29 model-design papers | `DESIGN_LESSONS_model_design_papers_2026-09-17.md` §1–§6 | §2 lessons, entered only where the text states a testable form |
| The SEAA addendum (Liu 2026, 2609.17331) | the same file, §7 | the four drafts `E-DR`, `E-SH`, `E-RM`, `E-RE` and the six test designs of §7.10 |
| Fourteen sweep preprints, read in full | `SWEEP_READING_REPORTS_2026-09-20.md`, from the brief in `sweep_readers_brief_2026-09-19/` | the evidence behind the candidates, cited by part and candidate, not restated |
| Their 42 candidates | `SPEC_CANDIDATES_from_preprints_2026-09-20.md` | the primary source of requirement text; each coverage status was re-checked against revision 9 before placing it |

Where a §7.8 draft and a C-number overlap, `DESIGN_LESSONS` §8.9's mapping table was followed and the item
was entered once, with both references in its marker: `E-DR` with C16, C24 and C31 (`M17.E.2`); `E-SH` with
C29 (`M17.F.1`); `E-RM` as arm (d) of C34's list (`M17.D.1`, `M17.D.2`); `E-RE` as a column on C39's readout
(`M17.B.5`). No wording was taken from the PDFs; the paraphrases are the candidates file's.

**Two placements differ from the handoff that drove this revision, both on the owner's decision of
2026-09-22.** `E-RE` is at `M16.A.9` and `M11.C.40`, gated in Phase D, not in `M17.C` — `DESIGN_LESSONS` §7.9
says it is a Phase B–D readout and the only observable `M4.D.6d` has. And the three §7.10 tests with no
C-number are entered: (c) at `M17.D.4`, (d) at `M11.D.20`, (f) at `M17.B.4`.

**Parts of three Phase E candidates were placed outside `M17`**, because they must exist during B–D: C29's
non-absorbing-bound test (`M11.D.21`), C31's margin log (`M16.A.10`) and C41's discrepancy log (`M16.A.5a`).

| | Count |
|---|---|
| New IDs | **91** — M1 4, M2 1, M3 5, M4 6, M6 1, M7 1, M9 1, M10 4, M11 25, M13 1, M14 3, M15 1, M16 9, M17 29 |
| Amendments of approved requirements, by suffixed ID | **3** — `M3.D.4a` (C1, replaces `M3.D.4`'s second sentence), `M11.4b` (the exceptions count) and `M13.4` (adaptive ensembles in Phases C and D) |
| New acceptance criteria | **7** in `M11.C` (`M11.C.35`–`M11.C.41`) and **8** in `M11.D` (`M11.D.15`–`M11.D.22`) |
| New `[I]` register entries | **18** in `M10.C.1a`, naming **25** constants; none is given a value |
| Candidates entered | **all 42** (C1–C42), the four §7.8 drafts, and all six §7.10 tests; C19 enters as a criterion form with no instance; `M11.C.41` is the owner's answer to it |
| New part headings | `M3.E`, `M14.A`, and the module `M17` with parts A–G |

**Every marked item with an ID.** Form is *new*, *amendment*, *criterion*, *register* or *Phase E*.

| Spec ID | Candidate or E-ID | Source paper | Form |
|---|---|---|---|
| `M1.A.20` | C2 | Buffalo, Pearson & Klein 2026 (2603.11084) | new |
| `M1.B.13` | C2 | Buffalo, Pearson & Klein 2026 (2603.11084) | new |
| `M1.C.7` | C2 | Buffalo, Pearson & Klein 2026 (2603.11084) | new |
| `M1.F.1b` | C7 | Li & Tao 2026 (2603.00113) | new |
| `M2.A.0h` | C2 | Buffalo, Pearson & Klein 2026 (2603.11084) | new |
| `M3.D.4a` | C1 | Buffalo, Pearson & Klein 2026 (2603.11084) | amendment |
| `M3.D.4b` | C3 | Buffalo, Pearson & Klein 2026 (2603.11084) | new |
| `M3.D.4c` | C4 | Buffalo, Pearson & Klein 2026 (2603.11084) | new |
| `M3.E.1` | C6 | Li & Tao 2026 (2603.00113) | new |
| `M3.E.2` | C6, DL §2.1 | Li & Tao 2026 (2603.00113); Axtell & Farmer 2022 | new |
| `M4.B.2` | C9 | He 2026, VISA (2607.28027) | new |
| `M4.C.9` | C8 | Holland et al. 2026 (2607.29546) | new |
| `M4.D.1e` | C11 | Buitrago López et al. 2026 (2606.12369) | new |
| `M4.D.1f` | C12 | Ye et al. 2026, TRAILS (2605.18890) | new |
| `M4.E.1a` | C7 | Li & Tao 2026 (2603.00113) | new |
| `M4.G.3` | C13 | Prasad 2026 (2607.07753) | new |
| `M6.3` | C15 | He 2026, VISA (2607.28027) | new |
| `M7.C.1e` | C15 | He 2026, VISA (2607.28027) | new |
| `M9.8` | C9 | He 2026, VISA (2607.28027) | new |
| `M10.B.4` | C23 | Zhou et al. 2026, PIMMUR (2509.18052) | new |
| `M10.B.5` | C18 | Prasad 2026 (2607.07753) | new |
| `M10.C.1a` | one per entry — C6, C8, C12, C13, C16, C19, C21, C26, C30, C32, C37, E-RE, E-SH, E-DR, DL §2.3, §2.6, §7.10(c), §7.10(e), and the owner's decision for `M11.C.41` | as each entry's marker | register |
| `M10.C.5` | C22 | Zhou et al. 2026, PIMMUR (2509.18052) | register |
| `M11.1b` | C17 | Prasad 2026 (2607.07753) | new |
| `M11.1c` | C21 | Ye et al. 2026, TRAILS (2605.18890) | new |
| `M11.1d` | C22 | Zhou et al. 2026, PIMMUR (2509.18052) | new |
| `M11.1e` | C20 | Kalluri 2026 (2603.01189) | new |
| `M11.1f` | DL §2.6 | Mou, Wei & Huang 2024 | new |
| `M11.4b` | C15, C16, C19, C26 | He 2026, VISA (2607.28027); Prasad 2026 (2607.07753); Kalluri 2026 (2603.01189); Blando et al. 2026 (2604.04543) | amendment |
| `M11.4c` | C19 | Kalluri 2026 (2603.01189) | new |
| `M11.4d` | C24 | Prasad 2026 (2607.07753) | new |
| `M11.4e` | C25 | Ye et al. 2026, TRAILS (2605.18890) | new |
| `M11.4f` | owner decision 2026-09-22 (`DESIGN_LESSONS` Q10), E-RM | owner; Liu 2026, SEAA (2609.17331) | new |
| `M11.C.35` | C8 | Holland et al. 2026 (2607.29546) | criterion |
| `M11.C.36` | C9 | He 2026, VISA (2607.28027) | criterion |
| `M11.C.37` | C15 | He 2026, VISA (2607.28027) | criterion |
| `M11.C.38` | C16 | Prasad 2026 (2607.07753) | criterion |
| `M11.C.39` | C18 | Prasad 2026 (2607.07753) | criterion |
| `M11.C.40` | E-RE | Liu 2026, SEAA (2609.17331) | criterion |
| `M11.C.41` | owner decision 2026-09-22 (answer to C19) | owner, as domain expert; `M1.C.3a` | criterion |
| `M11.D.15` | C4, DL §7.10(a) | Buffalo, Pearson & Klein 2026 (2603.11084); Liu 2026, SEAA (2609.17331) | criterion |
| `M11.D.16` | C5, DL §7.10(b) | Li & Tao 2026 (2603.00113); Liu 2026, SEAA (2609.17331) | criterion |
| `M11.D.17` | C10 | He 2026, VISA (2607.28027) | criterion |
| `M11.D.18` | C12 | Ye et al. 2026, TRAILS (2605.18890) | criterion |
| `M11.D.19` | C14 | Kurz 2025 (2512.18016) | criterion |
| `M11.D.20` | DL §7.10(d) | Liu 2026, SEAA (2609.17331) | criterion |
| `M11.D.21` | C29 | Holland et al. 2026 (2607.29546) | criterion |
| `M11.D.22` | C35 | Zhou et al. 2026, PIMMUR (2509.18052) | criterion |
| `M13.4` | C26 | Blando et al. 2026 (2604.04543) | amendment |
| `M14.A.1` | C10 | He 2026, VISA (2607.28027) | new |
| `M14.A.2` | C10 | He 2026, VISA (2607.28027) | new |
| `M14.A.3` | C10 | He 2026, VISA (2607.28027) | new |
| `M15.A.1a` | C2 | Buffalo, Pearson & Klein 2026 (2603.11084) | new |
| `M16.A.1a` | C6 | Li & Tao 2026 (2603.00113) | new |
| `M16.A.3a` | C9 | He 2026, VISA (2607.28027) | new |
| `M16.A.3b` | C11 | Buitrago López et al. 2026 (2606.12369) | new |
| `M16.A.3c` | C12 | Ye et al. 2026, TRAILS (2605.18890) | new |
| `M16.A.5a` | C41 | Kalluri 2026 (2603.01189) | new |
| `M16.A.7` | C23 | Zhou et al. 2026, PIMMUR (2509.18052) | new |
| `M16.A.8` | C20 | Kalluri 2026 (2603.01189) | new |
| `M16.A.9` | E-RE | Liu 2026, SEAA (2609.17331) | new |
| `M16.A.10` | C31 | Kurz 2025 (2512.18016) | new |
| `M17.A.1` | C26 | Blando et al. 2026 (2604.04543) | Phase E |
| `M17.A.2` | C27 | Blando et al. 2026 (2604.04543) | Phase E |
| `M17.A.3` | C25 | Ye et al. 2026, TRAILS (2605.18890) | Phase E |
| `M17.A.4` | DL §2.6 | Röchert et al. 2022 | Phase E |
| `M17.B.1` | C28 | Buffalo, Pearson & Klein 2026 (2603.11084) | Phase E |
| `M17.B.2` | C29 | Holland et al. 2026 (2607.29546) | Phase E |
| `M17.B.3` | C41 | Kalluri 2026 (2603.01189) | Phase E |
| `M17.B.4` | DL §7.10(f) | Liu 2026, SEAA (2609.17331) | Phase E |
| `M17.B.5` | C39, E-RE | Buitrago López et al. 2026 (2606.12369); Liu 2026, SEAA (2609.17331) | Phase E |
| `M17.B.6` | C40 | Sachdeva & van Nuenen 2025 (2510.10002) | Phase E |
| `M17.C.1` | C32, DL §2.6 | Li & Tao 2026 (2603.00113); Guan et al. 2026, Light Society | Phase E |
| `M17.C.2` | C30 | Sachdeva & van Nuenen 2025 (2510.10002) | Phase E |
| `M17.C.3` | C33 | Ye et al. 2026, TRAILS (2605.18890) | Phase E |
| `M17.D.1` | C34, E-RM | Sachdeva & van Nuenen 2025 (2510.10002); Liu 2026, SEAA (2609.17331) | Phase E |
| `M17.D.2` | E-RM, C34 | Liu 2026, SEAA (2609.17331); Sachdeva & van Nuenen 2025 (2510.10002) | Phase E |
| `M17.D.3` | C35 | Zhou et al. 2026, PIMMUR (2509.18052) | Phase E |
| `M17.D.4` | DL §7.10(c) | Liu 2026, SEAA (2609.17331) | Phase E |
| `M17.D.5` | DL §2.6 | Yang et al. 2024, OASIS | Phase E |
| `M17.E.1` | DL §2.3, DL §2.4 | Castellano, Fortunato & Loreto 2009 | Phase E |
| `M17.E.2` | E-DR, C16, C24, C31 | Liu 2026, SEAA (2609.17331); Prasad 2026 (2607.07753); Kurz 2025 (2512.18016) | Phase E |
| `M17.E.3` | C31 | Kurz 2025 (2512.18016) | Phase E |
| `M17.E.4` | C37 | Ye et al. 2026, TRAILS (2605.18890) | Phase E |
| `M17.E.5` | C38 | Prasad 2026 (2607.07753) | Phase E |
| `M17.E.6` | DL §2.1 | Axtell & Farmer 2022 | Phase E |
| `M17.F.1` | E-SH, C29 | Liu 2026, SEAA (2609.17331); Holland et al. 2026 (2607.29546) | Phase E |
| `M17.F.2` | DL §7.10(e), DL §2.5 | Liu 2026, SEAA (2609.17331); Axtell & Farmer 2022 | Phase E |
| `M17.G.1` | C36 | Ye et al. 2026, TRAILS (2605.18890) | Phase E |
| `M17.G.2` | C42 | Zhou et al. 2026, PIMMUR (2509.18052) | Phase E |
| `M17.G.3` | DL §2.6 | Axtell 2016 | Phase E |

**Marked edits without an ID of their own.** Front matter (revision, date); §0.1's Phase E paragraph; §0.2's
method-literature row; §0.3's counts (restated at step 7, below); `M3.D.4b`'s ten draw-class rows, every one
the sweep reader's proposal; four rows added to `M11.4`'s exceptions table (`M11.C.37`, `M11.C.38`, the
`M11.4c` form, `M17.A.1`'s `UNDETERMINED`); the Done-when cells of Phases B, C and D in `M13`, and a new
Phase E row carrying `M17` and `M11.C.8`; the `M14.A` stub table.

**Candidates judged already covered, and not entered** (candidates file §9): the engine driver surface —
reset, next, eval (`M16.B`); refactor-equivalence by draw order (superseded by C1); interval-valued initial
conditions (`M15.D.2`–`M15.D.4`); exact rational arithmetic (not adopted; byte identity plus C31 suffices); a
solo no-interaction baseline (extended by C34); information availability against awareness (the belief
layer and exogenous-flagged events — a one-line check that spell onsets enter as events, `M1.F.6`, is still
worth doing); one-component-at-a-time mechanism ablation (`M11`'s mutation protocol); readout frequency per
output (`DESIGN_LESSONS` §2.8); a single seeded generator (`M3.D.4`, now amended by C1); execution mode as an
assumption to test (§2.1, now `M17.E.6`); one-factor and factorial sweeps (§2.3, now `M17.E.1`); witness
radius with a visibility weight (per-hop fidelity); self-describing run files (`M16`); memory as digested
persistent state and event-driven interaction (`M1`, `M16.B.3`); turn order and initial event as
perturbations, and freezing a slow variable (§2.1, §2.6). The exploratory LLM-line items X1–X4 are not spec
content and were not entered.

**Design lessons noted, not specified** — no testable form is stated in the text, or the item is a cost
decision: §2.2's point that statistical-physics critical points and exponents do not carry to twelve agents;
§2.7's functional-form artefacts (a product with a zero term, equality created by rounding, accumulation with
no decay, division by a `functional_level` near zero); §2.8's snapshot-and-fork (a cost decision; under
`M3.D.4a` a fork is a plain state copy), naming of three intervention channels (`M17.D.3` declares the
channels but has no mid-run perturbation), appraisal and belief records citing event IDs, a `WITHHOLD` count
in readouts, and windowed trajectories for slow variables outside a shock; §2.9's ODD format and an
"ever varied" register column (partly met by `M10.B.4`, `M16.A.7` and `M17.E.3`); §2.10's candidate
functional shapes; §2.11's Latin-hypercube design and the history-matching question already on the
design-lessons question list; §2.12's supporting statements.

**Read at abstract level only; no requirement derived.** The eight papers of `DIGEST.md`'s 2026-09-21 entry
(2609.16395, 2609.15849, 2609.16344, 2609.21857, 2609.21636, 2609.21349, 2609.19913, 2609.21940), which were
not read in full and from which no candidate was drawn; the Light Society reference list
(`2026_Light_Society_2506.12078_model_design_refs.md`); and the catalogues `PREPRINT_SWEEP_2025-09_to_2026-09.md`
and `INDEX.md`. `SPEC_CANDIDATES_plain_language_2026-09-20.md` is explanation only and supplied no wording.

**The owner's answers of 2026-09-22, and what was done with each.** The fourteen questions this section
first listed, in their original order.

| # | Question | Answer | Done |
|---|---|---|---|
| 1 | Death: an exit from the field, or absorbed by the family? (C15, `M6.3`) | **Deferred for later review.** Guidance: relationship energy behaves like chemical bond energy and does not disappear | Guidance recorded in `M6.3` as not yet a requirement; `TODO.md` |
| 2 | Where method rationale lives | In `model_explainer.md` | Mirrored in explainer §18; every method citation in this spec now points there; §0.2 recasts the `papers/` files as evidence |
| 3 | `M17` in this document or split out | One master document | Recorded in §0.1 |
| 4 | C3's keys | Do the recommendation | The reader's six keys adopted as the recommendation; the four uncovered classes given recommended keys (`M3.D.4b`, marked *recommended rev10*) |
| 5 | C13's relief term | **Review later** | `M4.G.3` stays marked; `TODO.md` |
| 6 | C19's corpus ordering | Most mechanisms scale with differentiation of self and the stress being experienced: lower differentiation and higher stress mean more anxiety and more exaggerated patterns of functioning, such as more reactivity | New criterion `M11.C.41`, gated in Phase C: four two-arm directions over a 2×2 of level and stress. `M11.4c`'s mechanism ranking stays uninstantiated; strike it at approval unless a ranking turns up |
| 7 | The directions of `M11.C.35` and `M11.C.36` | Both confirmed | No change |
| 8 | `E-RM` within `M0.4` (`DESIGN_LESSONS` Q10) | Switching off a mechanism is a change; the owner then adopted the recommendation: such comparisons are allowed, as tests of the model only | New rule `M11.4f`: such comparisons are between two models, are reported as tests of structure, never as effects on a family, and are swept in Phase E. It reaches `M17.D.1`, `M17.D.2`, `M11.C.39` and `M11.C.40`, and — by the owner's decision — the approved ablation of `M10.C.4a`, which it leaves unchanged |
| 9 | `E-RM` and C34, one requirement or two (Q16) | **Needs review** | Left as `M17.D.1`(d) plus `M17.D.2`; `TODO.md` |
| 10 | Floors (Q15) | Keep as SHOULD | Recorded in `M17.D.4` |
| 11 | `E-DR` against the constant sweep (Q8) | Asked for a recommendation, then **approved** it | Kept separate, because they answer different questions; `M17.E.2` sweeps the same range `M17.E.1` samples |
| 12 | Ensemble size in Phases C and D | Adaptive | `M13.4` amends the fixed 1,000-seed ensembles; `M17.A.1`, `M17.A.3` and `M17.A.4` are built in Phase C, and `UNDETERMINED` does not pass a phase |
| 13 | The arm-blindness check from Phase B | Do the recommendation | `M11.D.22`, gated in Phase B |
| 14 | `CLAUDE.md` still says revision 6 and 427 | Yes, update after approval | `TODO.md`, together with the four other documents in the count check below |

**Consistency checks, run 2026-09-22 at the end of this revision.** No `M11.D` scan code exists for the v2
engine yet, but `tests/test_spec_consistency.py` implements the `M11.D.10`, `M11.D.11`, `M11.D.13` and
`M11.D.14` scans against this document and was run; the other checks were one-off Python and `grep`
scripts, not committed.

| Check | Result |
|---|---|
| New criteria numbered from `M11.C.35` (revision 9's table ended at `M11.C.34`) | pass — `M11.C.35`–`M11.C.41`; `M11.D` continues at `M11.D.15`–`M11.D.22` |
| Every ID matches `M\d+(\.[A-Za-z0-9]+)*`; no duplicates | pass — 518 defined IDs, 0 failures, 0 duplicates |
| Revision 9's IDs unchanged | pass — all 427 still defined, 0 removed, 91 added; extracted from both versions with the same pattern and diffed |
| `M11.D.11`, both directions | pass — every `M11.C` criterion's phase matches its `M13` Done-when cell (B 0, C 19, D 21, E 1) and no cell names an orphan; the project's three `M11.D.11` tests pass |
| `M11.D.13`, no subject-negated `MUST` | pass — the project's test, and a separate scan of the added text |
| Every marker resolves | pass — 136 markers; every part names a C-number heading in the candidates file, an `E-DR`/`E-SH`/`E-RM`/`E-RE` draft in `DESIGN_LESSONS` §7.8–§7.9, or a `DESIGN_LESSONS` section or §7.10 item, except three administrative forms (*owner decision 2026-09-22*, *first draft, rev10*, *register extension*) and the *reader's proposal* and *recommended rev10* tags on `M3.D.4b`'s rows |
| No quotation of eight or more words from a paper | pass — every added line compared as eight-word sequences against the `pdftotext` output of all 70 PDFs under `papers/`, including the fourteen sweep papers and SEAA, and the same comparison over the new `model_explainer.md` §18: 0 matches |
| §0.3's counts | restated above: 835 tokens, 79 named tests (revision 9: 663, 51). The project's `M11.D.14` test passes against them |
| `M14`'s coverage figures | nothing to restate — `M14` states no figures; `docs/spec_coverage.md` does not exist until implementation |
| Requirement count cited by other documents | **fails, and is left failing on this branch.** The project's requirement-count test finds `CLAUDE.md`, `README.md`, `CHANGELOG.md`, `docs/theory/_STATUS.md` and `agent_model_proposal.html` claiming 427 against the spec's 518. This revision may not edit `CLAUDE.md` or `docs/theory/`, and restating the others would present unapproved proposals as the count. Update all five at approval, with the count as approved |

**Adjacent issues found, not fixed.** `M11.2`'s note says the criteria table "now ends at 33" and `M11.3`
counts "4 of 33 criteria" and "27 unclassified"; at revision 9 the table already held 34 rows, and it now
holds 40. `M11.3`'s stressor classification has not been applied to the six new criteria. `M1.F.1a` sits
physically inside `M4.E`. Revision sections 4 and 5 are out of order below. These are approved text and
were left for the owner.

---

#### Approval — 2026-10-06

Revision 10 was approved by the owner on 2026-10-06, on these terms:

| Item | Decision |
|---|---|
| Scope of approval | All revision-10 text, plus `M11.F.10` (entered 2026-10-06 as a revision-11 proposal, approved with revision 10 because it is the owner's own decision and changes no mechanism) |
| The external review, `docs/REVIEW_spec_rev10_2026-09-23.md` | Approved without first revising for its findings, following the review's recommendation 9: *approve and build Phase B only*. Each finding is carried into the implementation plan as a blocker on the phase it affects. The spec is not changed for them at this revision |
| `M11.4c` | **Kept.** The revision-10 condition was to strike it unless a ranking of mechanisms turned up. The review found one in the ledger — `L07.4`: spouses' dynamics first, relationships outside the ego mass second, fusion intensity third — so the condition for striking is not met. Whether `L07.4` instantiates it is left to the phase that builds those mechanisms |
| `M4.G.3` | **Kept as written.** It is conditional and a SHOULD; it binds only if a term of the kind it names exists, so approving it decides nothing early |
| `M11.E` | The four human-review proposals are accepted as written |
| Still open | `M6.3` (disposition at death; blocks `M11.C.37` and Phase D) and the `E-RM`/C34 question (Phase E). Neither blocks Phase B |
| Revision-11 material in `TODO.md` | Held. Not part of this approval |
| Phase B plan, approved the same day | All nine recommendations (D1–D9) of `docs/implementation_plan_phase_b.md` adopted as written. Spec changes: `M4.C.1` pulled into Phase B (`M13`); `M1.A.21`, `M1.A.22`; `M2.3a` and Ravi's `basic_level`; `M3.C.2`; `M4.G.2a`; `M14.A.3`, `M14.A.4` and the filled register; `M11.D.17` reads "no writer in any phase" |

**Inline narratives moved from the bodies at revision 35:**

- `M10.C.1a`: **M10.C.1a** — *extended at revision 10.*
- `M11.4`: No instance exists at revision 10
- `M11.4b`: The rows added at revision 10 extend `M11.4`'s table, and the count in `M11.4`'s first sentence ("Three criteria") is superseded by the table itself, which already held six rows at revision 9; the list remains exhaustive.
- `M11.4c`: No criterion of this form is instantiated at revision 10.

**Moved from the bodies 2026-10-22, the phrases added that day (`docs/spec_revision_35_dispositions.md`, rows 219–262):**

- `M13 (phase table, row B)`: **Added at revision 10:** M11.D.17 passes against the register as filled at plan time, and M11.D.22 passes.

### Revision 9 — the corpus-fidelity sweep, 2026-08-28

**Every quoted passage in this document was checked against the primary sources.** Five cold agents on a
different model, reading the source `.txt` files and forbidden the extractions, plus a deterministic string
match by the author over all 214 quotations.

| | |
|---|---|
| Quoted passages in requirement text | **214** |
| Located verbatim in the corpus (punctuation- and OCR-proof match) | **160** |
| Legitimate self-quotes of project documents | **52** |
| Unexplained after inspection | **0** |
| Requirements resting on a verified corpus quote | 94, **all judged** |

**One fabricated quotation existed and is fixed.** `M1.A.4e` read *"`KS11.7` says so in its opening line:
**The family diagram records**, per person: birth date…"*. That phrase appears nowhere in the book. It was a
paraphrase placed inside quotation marks and attributed to the source, and it was the evidence for revision
8's demotion of the four assessment domains. Corrected, and the demotion restated to what the source carries.

**The dominant class was modal inflation** — a hedged, permissive or illustrative source becoming a mandate:

| | Source | Had become |
|---|---|---|
| `M7.D.2a` | "**can** stabilize somewhat… **or** the regression can continue to get worse" | "**MUST** fall" — one branch mandated, the other erased |
| `M16.D.2` | "Bowen **suggested**… the family **could perhaps**" | "Bowen **required**" |
| `M1.A.11b` | genes "**perhaps** having a role"; theory "**does not rule out**" | channel **MUST** be constitutional |
| `M1.D.7l` | "a new balance, **such as** 55 or 60" | a regression **MUST** be **bounded** |
| `M1.A.18b` | KB07, **hedged**: undifferentiation is roughly what sets thinking hard, and loosening it would let the thinking **change faster** (K07.3) | a **ceiling** on the attainable value |

**Scope shifts, three of them introduced at revisions 7 and 8.** A phrase restricted to "**Some of those in
this group**" applied to a whole band (`M1.A.3c`); a **societal** trend used to ground an **individual**
mechanism (`M1.A.4h`, `M4.D.1a`); a consequence presented as the quotation's when the corpus arguably cuts
the other way — the actions that can make a system more flexible "are based on a **broad and long-range
assessment**" (`M4.D.1a`; `family_evaluation/fe03.md` · FE03.21).

**Evidence framing.** "An experiment and an independent replication" were two clinical accounts, one of them
the author confirming what he had been told (`M1.E.2a`). "Two families — home movies in both" was one family
(`M7.E.1d`). "Three statements, every one about the mover's anger" was two (`M5.D.4a`).

**Attribution.** ASR transcripts attributed to Bowen personally against the project's own rule (`M5.D.7a`);
words belonging to **Andrew Solomon**, quoted by Kerr, grounding a requirement (`M4.D.5c`); a line whose
speaker is genuinely ambiguous between Bowen and Kerr (`M1.A.18b`).

**What this says about the method.** Two warm failure-pattern sweeps and eight revisions had passed over all
of it. Every one of these was found by reading the **primary** rather than the extraction — and the
mechanical existence check, which is the part a language model does worst, was done with `grep` rather than
by an agent, after two false "not found"s taught that lesson. Hedge-level findings are backlogged in
`TODO.md` rather than fixed, per the bounded-loop rule.

---

### Revision 8 — what the estimator may read, 2026-08-28

Raised by the project owner across three points, each checked and each upheld. Together they replace the
estimator's observable set.

| | What changed | Where |
|---|---|---|
| **Demoted** | `M1.A.4e` presented **genogram intake fields** as "the estimator's observables". `KS11.7`'s own opening says they are what *"the family diagram records"*. They are now a weak prior at import and **MUST NOT** set an estimate | `M1.A.4e` |
| **New** | A **three-tier** reading: a single adverse event is mostly **load**; a chronic self-maintained pattern reads **functional** level under load correction; only a position held under pressure reads **basic** level | `M1.A.4g` |
| **New** | The discriminator is **which channel was selected under discomfort, at comparable load** — `M4.D.1a`'s mixing weight read backwards out of behaviour | `M1.A.4h` |
| **New** | The **system's reaction to a symptom** is the best-controlled reading in the model: one event, N observers, one moment, so load is constant **by construction** | `M1.A.4i`, `M11.C.34` |
| **New** | The circularity boundary stated precisely: symptom **magnitude** may not feed the estimator; the **channel selected** must, because it reads policy rather than accumulated load | `M1.A.4j` |
| **Generalised** | `M1.E.2`'s three reaction detectors were scoped to the coach. They are now on **every `Person`** — without them the family's own reaction is unreadable | `M1.A.19`, `M1.E.2` |

**Three things the corpus already said, which the spec had not carried through.**

1. `KS06.8` **already corrects** `KS05.10`: "**In terms of solid self, one's personal life is where the rubber meets the road**", and it is the intimate relationship "**and not how they manage themselves in their work lives**" that determines offspring level. `M1.A.4e` cited this correction **underneath the table it invalidates**, so the table still read as the instrument.
2. The outcomes in that table are **base-rate common**. A divorce or a job loss is an event a large share of the population has; it carries almost no information about who had it, and it confounds the load on a person with their functioning.
3. Ch10 generalises the neutral third to "**any third person… no matter what the subject matter**", and `agent_model_proposal.html` §10 records the conclusion outright — "**one rule seen from four angles… Implement it once**". `M1.E.2` implemented one of the four.

**The unification, which is the reason to believe it.** "Adhering to one's convictions under relationship
pressure" and "carrying discomfort rather than discharging it" are **one** observable, not two — `M5.F.5`
says so in a sentence, and `M4.D.1a` already carries the quantity as the mixing weight. Solid self is the position a person keeps **whatever the pressure**; pseudo-self is the **negotiable** part, which pressure from others can change (1979 Tape 2). Everything needed is already state the
model holds: the accommodation stock, the taboo set, gradable concession, the solid-self fraction. **No new
machinery.**

**And the load correction applies twice over.** Channel mix is *set by* level, so reading level back out of an
uncorrected run recovers its own input. `M1.A.4b` was already the answer and had not been extended to these
readings.

**Inline narratives moved from the bodies at revision 35:**

- `M1.A.4e`: **M1.A.4e** — *intake fields, and a **weak prior only**; demoted at revision 8.*
- `M1.A.19`: **Generalised off the external agent at revision 8.**

---

### Revision 7 — the scale is a continuum, 2026-08-28

Raised by the project owner: *Bowen's quadrants are bands across a continuum, not shift points.* Checked
against the primary and upheld.

| | What changed | Where |
|---|---|---|
| **Corrected** | `M1.A.3` required *"exactly one behavioural transition on `basic_level`, at 50"*. The licence is now a **slope** — continuous in level, with the joint-decision domain lagging every other domain at every level, and no discontinuity anywhere | `M1.A.3`, `M1.A.2` |
| **Guarded** | Removing the step **MUST NOT** flatten the model: the corpus asserts real, **non-monotone** structure over the range — emotionality peaks in the middle, and a stratum looks differentiated and is not | `M1.A.3c` |
| **Generalised** | **A band edge MUST NOT be a branch point anywhere** — 25, 50, 60, 70, 75. The scheme moved three times inside the corpus and the two authors give different edges; branching on one hard-codes a coordinate the source does not hold still | `M1.A.3d` |
| **Regraded, provisionally** | The capacity floor at 25 becomes a continuous falloff, applied on the principle rather than on a fresh reading — its source has **not** been re-read, so its location stands pending that | `M1.A.4d` |
| **Withdrawn as a target** | "the transition at 50" had been listed as a corpus-stated calibration target | `M10.C.2` |
| **Rewritten** | The import rule that no rating point may land on 50 — its reason went with the step, and the problem underneath is larger without it | `M15.B.3` |
| **Word corrected** | "regime boundary" → **turning point** in the sign-flip argument. Conclusion unchanged; the picture is a roll-off, not a cliff | `M15.D.4`, `M11.F.9(c)` |

**The evidence, since this overturns a MUST cited to Bowen's own hand.** Ch16 states it as a description with
hedges — the intellect "**begins** making **a few** decisions of its own". Ch21 has **no regime change at 50
at all** and calls the lower/upper transition **explicitly gradual** (`L21.1`, already marked `[corrected]`
in the ledger). The band scheme is **present 1966, absent 1972, present again 1976**, graded `[X]` in the
explainer as "not a constant of the theory". Kerr 1988 gives different edges entirely — 0–10, >60, >70
(`FE03.5`). Bowen revised the top twice and "**subsequently dropped the term scale**" (`KS05.3`). Ch16 is the
latest of the three and is **not** outvoted — but it describes a band, and this requirement had implemented
a cliff.

**This is the same failure as `M5.D.4` at revision 4**: a source *description* firmed up into a mechanism the
source does not assert. Two instances now, both caught by reading the primary rather than the extraction, and
neither catchable by a regex. A sweep for others of this shape is logged in `TODO.md`.

**Inline narratives moved from the bodies at revision 35:**

- `M1.A.3`: **M1.A.3** — *the licence is a **slope**, not a step; corrected at revision 7.*
- `M1.A.4d`: *Regraded at revision 7, by the same argument as `M1.A.3`, and the limits of that are worth stating.*
- `M15.B.3`: *Rewritten at revision 7.* (before "The earlier form required …")
- `M15.D.4`: *Revision 7 corrects the word: these were called "regime boundaries", which implies cliffs.*

**Moved from the bodies 2026-10-22, the phrases added that day (`docs/spec_revision_35_dispositions.md`, rows 219–262):**

- `M1.A.3`: **Why this changed.**
- `M1.A.3`: The earlier form required *"exactly one behavioural transition on `basic_level`, at 50"*, which reads Ch16's description as a mechanism.
- `M1.A.3`: Ch16 is the latest of the three and is not outvoted — but it describes a band, and this requirement had implemented a cliff.
- `M15.B.3`: The earlier form required that no rating point land on "the transition at 50", reasoning that the modal rating would sit where the licence flips.
- `M15.B.3`: **`M1.A.3` no longer has a flip**, so that reasoning is gone — and the problem it named is **larger**, not smaller.

---

### Revision 6 — the run log, 2026-08-28

| | What changed | Where |
|---|---|---|
| **New module** | The run log and the readable trace. Six existing requirements already depended on a log; none of them said what one is | `M16` |
| **Exit condition made checkable** | Phase B's "the event log reads correctly" named no requirement and no test. It now names `M16.C.2` and `M16.T.1` | `M13` |
| **Engine purity** | The engine emits records; the caller persists them. The frozen engine's `sim_audit.csv` is named as the anti-pattern | `M16.B` |
| **Ordering** | `M14` was sitting after `M15`; modules are now in numeric order |  |

**Two things this settles.** A log records **effects beside causes** and the **selection rationale**, not
just the moves — without those an event log is a list, not a causal trace, and the causal trace is most of
the argument for the agent model over the grid. And the deterministic renderer is **not** the language
layer: rendering a finished log by template is reproducible and carries no interpretation, where narration
by a language model is Phase F, off by default, and never in the ensemble path (`M16.E`).

---

### Revision 4 — *Family Evaluation* (Kerr & Bowen, 1988), applied 2026-08-27

The sixth corpus source, folded in against the decisions recorded in `docs/DECISIONS — FAMILY EVALUATION.md`. **One correction, three amendments, and about twenty new requirements.**

| | What changed | Where |
|---|---|---|
| **Corrected** | `M5.D.4` was **inverted**. Ch13 says the mover's *freedom from* anger admits the peak; the spec said anger did. Settled by a re-read of the primary chapter, which mentions anger exactly once | M5.D.4, M5.D.4a, M11.C.32 |
| **Amended** | The channel prior is movable by a relational term and overridable by a constitutional one (A1, option c) | M1.A.11b, M1.A.11c |
| **Amended** | The transition at 50 keeps its licence implementation and gains an awareness **readout**; neither is reduced cognitive capacity (A2) | M1.A.3b |
| **Amended** | The dependence gate now binds the **slow** clock, not only the `I-POSITION` move (B1 tail) | M7.A.1a, M11.C.33 |
| **Reviewed, unchanged** | `M12.2` stands in full — Kerr 1988 names the locus of the first unknown, not the rule (A4, option b) | M12.2a |
| **New mechanisms** | Belief becomes a channel; cross-person reinforcement; a self-generated stress term; mutually protective channels; non-monotone lock-in; an objectivity gate on amplification; a `PREPARE` phase; a binder-unavailable event; emotional reserve; default tie deterioration | M9.6–M9.7, M4.D.6e, M4.A.5, M7.D.2c–2d, M4.C.8a, M5.D.2a, M1.F.9, M4.D.5d, M7.E.4 |
| **New constraints** | Pole assignment independent of sex; non-monotone coach contact; every criterion needs a stressor | M2.A.0g, M1.E.8, M11.3 |
| **New readout** | The eight-component family evaluation, with components 9 and 10 explicitly barred | M11.G |
| **New magnitudes** | Twelve stated bounds and shapes, admitted as **checks** and never as parameters | M10.C.4, M10.C.4a, M10.C.3a |
| **New prohibitions** | Power and punishment as mechanisms — the only `M12` entry stated by the author; no adjudication of trauma against process; no rights-as-regression readout | M12.5, M11.F.7, M11.F.8 |

**One item remains open and is not applied:** nothing. All nine decision items are resolved.

**Inline narratives moved from the bodies at revision 35:**

- `M5.D.4`: **M5.D.4** — *corrected at revision 4; the earlier form was inverted.*
- `M12.2a`: **M12.2a** — *reviewed at revision 4; **M12.2 stands in full**.*

### Revision 5 — the import question and what it exposed, 2026-08-27

Arising from a working session on whether the model can answer *"if you changed this one thing, what would happen to this family?"*

| | What changed | Where |
|---|---|---|
| **New guard** | Three clauses on speaking about a real family. (a) no output about a named family or person; (b) initialisation must carry its uncertainty; **(c) no counterfactual from parameters tuned to reproduce a known history** — a *correctness* clause, because fitting one arm breaks the cancellation `M0.4` depends on, invisibly and in one direction only | `M11.F.9` |
| **New module** | The family-diagram import contract — export format, the value/range split, four adapter traps, the envelope requirement, and fixture mode | `M15` |
| **Scope** | `M15` is Phase E but specified early, because the source application is under the project owner's control and the format is cheaper to fix before it exists than after | §0.1, `M13.3` |

**The finding worth keeping.** Initial conditions are not nuisance parameters. Invented constants cancel
between two arms because they do not interact with the intervention; **initial conditions are precisely what
the intervention acts on**, so their error crosses a regime boundary and flips the sign rather than shifting
both arms together. The model has at least five such boundaries and one of them (`M5.C.1a`) is a SAFETY
property with recorded harms. **Direction is the least robust output under mis-specified inputs, not the
most** — which is why `M15.D.2` requires an envelope over ranges and forbids a point direction.


### Undated narratives moved from the bodies at revision 35

- `M0.4`: *(Earlier revisions pointed at "four exceptions listed in M11"; no such list existed.
- `M1.D.7a`: *Their meanings were previously recorded nowhere except Pygame log strings in `src/main.py`, where the third was labelled "Climate"; `resource_pressure` is the accurate name for what Ch18 actually argues.*

**Moved from the bodies 2026-10-22, the phrases added that day (`docs/spec_revision_35_dispositions.md`, rows 219–262):**

- `M1.A.4e`: They are an **intake form**, and the earlier form of this requirement promoted them to "the estimator's observables".
- `M11.2`: *Restated 2026-08-28.
- `M11.2`: The earlier form named the range `M11.C.1–M11.C.16`, written when the table ended near 16.
- `M11.2`: The table now ends at 33, so seventeen criteria — including `M11.C.28`, `M11.C.30` and `M11.C.31`, all of them distributional claims a single run cannot establish — sat outside a bound nobody had moved.
- `M11.2`: It is stated over the whole set here so it cannot drift again.*
- `M11.2`: **A range bound over a set that grows is the shape `P29` names**, and `M11.D.9` already forbids it elsewhere in this document. It is stated over the whole set here so it cannot drift again.*
- `M11.F.9`: *(An earlier form said the bias was "unbounded".
- `M11.F.9`: That does not follow — over a bounded parameter space the difference is unidentified, not unbounded — and an unsupported strengthening is the exact error §10 records the first corpus pass making.)*
- `M11.F.9`: *Mechanism corrected 2026-08-28.* The earlier form said fitting absorbs the model's error "on one side only", which is wrong: under `M0.4` both arms share one parameter set, so nothing is absorbed asymmetrically. The real failure is non-identifiability plus loss of independence, which is a **stronger** objection, not a weaker one. The earlier form also stated as a general law that *the better the fit, the less the counterfactual can be trusted*. That holds **in the overfitting regime this model is in** — and only there. A better-specified model both fits better and extrapolates better, so the sentence is not a principle to reason from: an implementer could conclude that a deliberately poorer fit is safer, or that fitting *both* arms restores the symmetry. Neither follows.
- `M16.B.3`: *Corrected 2026-08-28. The earlier form required the whole of logging to be a pure observer, which contradicts this module's own preamble twelve lines above — an agent's delayed log is an **input** to a mechanism. The contradiction was invisible because `M16.T.3` was gated only in Phase B, which has no policy and no external agent, so it passed vacuously exactly where it was true and was never re-run where it became false.*

## The front matter's status line, as it stood at revision 34

APPROVED 2026-10-07 (revision 10 and 11 on 2026-10-06; revision 12, the Phase C prerequisites, on 2026-10-07; revision 13, the owner's reference family and pattern readouts, decided 2026-10-10; revision 14, the Phase D decisions, 2026-10-10; revision 15, the Phase D definitions and `M11.C.46`, 2026-10-11; revision 16, graded rejection in `M1.B.3c`, 2026-10-11; revision 17, nodal events, cutoff read from behaviour, `M11.C.17` and `M11.C.7`'s direction, 2026-10-11; revision 18, stress moves functioning and coaching or own practice opens the self-directed channel, 2026-10-11; revision 19, the stress floor, self-discovered perspective and perspective's steering, 2026-10-11; revision 20, the system's adjustment is a level that settles back and is not read by the estimator, and `M11.C.18` and `M11.C.20`'s mutation clauses name what carries them, 2026-10-11 and 2026-10-12; revision 21, holding a position stops the functioning exchange, 2026-10-12; revision 22, calm is relative to the person, 2026-10-12; revision 23, the estimator reads pseudo-self with its sign, 2026-10-12; revision 24, the symptom channels (Phase D B7) and `M11.C.22`'s control arm; revision 25, beliefs as a channel (B8); revision 26, the cross-person term weighted by the tie (B9); revision 27, borrowed pseudo-self settles back when the exchange stops — revisions 24 to 27 are Claude's recommendations of 2026-10-12, reviewed by the owner 2026-10-16: the owner asked for the work to proceed overnight without waiting for input; revision 28, the life course — death, income, a child's development, starting ages, acting age and the coach — applies the owner's decisions of 2026-10-13 (P13–P15), and the forms it gives them are Claude's recommendation, reviewed by the owner 2026-10-16, and the owner approved the part II plan that builds it on 2026-10-13, and it was built that day; revision 29, cutoff and focus — a held cutoff and tie-state readout for `M11.C.10`, learning from a witnessed cutoff (`M4.D.6f`), reactive and principled cutoff (`M1.B.3d`), chronic anxiety built over childhood (`M7.B.1a`), focus on a child's weakness (`M1.B.8a`) and focus steering acts (`M4.D.1g`), and `M11.C.2` read by age — applies the owner's decisions of 2026-10-13, and its forms are Claude's recommendation, reviewed by the owner 2026-10-16; its plan (§4.8) approved by the owner ("i approve proceed") and built 2026-10-14, with the review fixes in §4.8; revision 30, origin cutoff as an input (`M1.D.8a`), danger and the self-chosen principled cutoff (`M1.B.3e`), focus that locks in (`M1.B.8b`), `M11.C.10` read on ties to partners and parents at 200 seeds with a family-of-origin variant, and the danger premise `M11.C.47` — applies the owner's decisions of 2026-10-14, its forms Claude's recommendation reviewed by the owner 2026-10-16, its plan (§4.10) approved and built 2026-10-14; **revisions 24 to 30 reviewed by the owner 2026-10-16** (`docs/REVIEW_spec_revisions_24-30.md`): 81 of 85 entries approved as built, 4 changed (`M14.A`'s `channel_focus` and `tolerance` rows, `M11.C.14` restated, `M11.C.6` limit (a) restated, `M7.E.1f`'s band opened by the parents' acts); revision 31, the owner's review of revisions 24 to 30 applied — its plan (§4.11) approved by the owner 2026-10-16; part A, the documents, built 2026-10-16; part B, the code, config and records (`M11.C.14` as restated, `M7.E.1f`'s band, the criteria settings' grades), built 2026-10-16 with the owner's decisions on the version 1 premises (`M11.C.14`'s hold-back baseline, `M6.I.4`'s capacity rule for `M11.C.21`, `M11.C.23` reclassified, `M11.C.40`'s clause 2 reported) and every record regenerated once; `M11.C.21`'s reclassification and keeping the capacity rule decided by the owner 2026-10-17, the reclassification built 2026-10-17 with every record regenerated; revision 32, finishing version 1 — the version 1 status table, `M11.C.4`, `M11.C.29`, `M11.C.32` and `M11.C.22` resolved, `M11.C.7` and `M11.C.46` built, `M11.C.9` out of version 1 (for version 2), `M4.C.1c`'s nodal-event spike built, recorded mutations for `M11.C.11`, `M11.C.15` and `M11.C.39` — built 2026-10-17 under the owner's instruction "proceed using your own recommendations" and reviewed by the owner one decision at a time the same day, the nodal-event spike of `M4.C.1c` added at that review (plan §4.12); `M11.C.17` and `M11.C.6` held for a diagnosis, `M11.C.17` passing since the spike and `M11.C.6` restated, failing, accepted as a finding by the owner 2026-10-18; every record regenerated 2026-10-18; **version 1 complete**, `docs/version_1_status.md`, frozen at revision 32 and the tag `version-1`; **version 2 begins at revision 33** (version 2 plan Q9): revision 33, version 2 stage 1 — `M13.2` amended for the grid engine's retirement and `M8.5` amended so that presence decides whether a triangle is active — decided by the owner 2026-10-19 (`docs/implementation_plan_version_2.md` D1, D3) and built 2026-10-19; revision 34 — every quotation of the 1979 tapes and the Kerr–Bowen interviews paraphrased and cited, as those are ASR transcripts (`docs/PROPOSAL — ASR QUOTATIONS (SP-1).md`, its §4 texts approved and its eight questions answered by the owner 2026-10-22: `M11.F.3` restated from *Family Evaluation*'s Epilogue, `M4.D`'s *Family Evaluation* quotation traced to `FE03.21`), and the stage 1 milestone review's wording fixes to `M2.A.0i`, Teodor's row, `M8`, `M11.1`, `M11.4` and `M13.1` (owner decisions D3, D4 and D6, 2026-10-21) — built 2026-10-22; nothing a run computes changed)
