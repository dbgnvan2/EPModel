# Changelog

All notable changes to this project are recorded here. Dates are ISO.

## [Unreleased] — 2026-10-11

### Version 1 complete (2026-10-18)

- The owner accepted `M11.C.6`'s failure as a finding ("Accept M11.C.6's failure as a finding, as recommended, and
  record it so version 1 is complete"). `docs/version_1_status.md` reads V1, V2 and V3 met; V4–V7 hold. By the
  version 1 definition's §4 question 8 the spec is frozen at version 1; later revisions start version 2.

### Spec revision 32, the owner's review: the nodal-event spike, `M11.C.6` restated (2026-10-17)

- **The owner reviewed revision 32 one decision at a time** (plan §4.12, "The owner's decisions"): every decision
  approved, every proposed disposition accepted except `M11.C.17` ("Diagnose coaching") and `M11.C.6` ("Trace it
  first"), and V7 signed off with a wording fix ("the owner's family" now reads "the reference family the owner
  specified" in the family file, `_STATUS.md` and the visualization proposal). Then: "continue with your
  recommendations … keep working until version 1 is completed".
- **`M4.C.1c`'s nodal-event spike built** (the owner's form, "cut ties, as TRIGGER"): a nodal event reaching a person
  spikes each of their cut-off ties (rejection held, both members alive) by `nodal_spike_intensity` (1.0, `[I]`) for one
  week, from the next tick (`event_effects.nodal_spike`; called at tick step 2, at a symptom onset and at removal by
  tolerance). From its review: a tie only not interactive is not spiked (a narrowing, Claude's recommendation pending
  owner review); ended spikes are pruned each tick (a 40-year run was about 50% slower without it, same results);
  `canonical_state` holds the spikes still to land. Tests: `test_m4c1c_*` (unit, adversarial, wiring, one-week and
  once-per-tie, removal). An `M11.C.4` mutant removes the spike alone. Phase B gate G7's mutant now removes the pruning
  too.
- **`M11.C.17` diagnosed:** no defect; the effect dies at the estimator's evidence test (own functioning stays below
  basic level in both arms). With the spike it passes, weakly.
- **Records.** Every record regenerated once (3 h 26 min; code hash `64e190be…`, Phase C `243ebedf…`). No Phase C verdict moved; `M11.C.17`,
  `M11.C.22` and two `M11.C.7` pairs turned green, each with a red recorded mutant; `M11.C.46`'s named mutant now
  survives at the edge (clause not met, recorded). The status page reads V2 and V3 met; V1 waits on `M11.C.6`, pending
  the owner.
- **`M11.C.6` traced and restated** (post hoc, pending the owner): the first arm picked by birth order; the new arm is
  one child with the parents' projection-type acts as declared against withheld. It fails the other way (ad hoc
  +0.096 ± 0.044). Recommended to accept as a finding, with a rule question on `M7.E.1f`.

### Spec revision 32: finishing version 1 (2026-10-17)

Built under the owner's instruction "proceed using your own recommendations"; every decision is Claude's
recommendation, pending owner review (plan §4.12, `docs/version_1_dispositions.md`).

- **The version 1 status table (V1).** `tools/version_1_status.py` writes `docs/version_1_status.md` from the three
  ensemble records, the two mutation records and the dispositions file. It refuses a failing premise (or a mixed
  criterion's failing premise limb) as a finding, a disposition that cannot apply, a row after a break in the table,
  and records that are stale against the registered criteria. 11 tests in `tests/test_version_1_status.py`. It found
  `M11.C.46`, a premise approved at revision 15 that no record listed.
- **The failing premises (V2).**
  - `M11.C.4`: the baseline now holds the tie open for the whole run; held open only to the act, the policy cut the tie
    in every baseline seed, so both arms held a cutoff at the nodal event. `M4.C.1c`'s nodal-event spike is not built,
    and the spec says so.
  - `M11.C.32`: the mover is declared at level 60 in both arms (post hoc). At 40, since revision 16, the calm arm never
    reached the peak.
  - `M11.C.29` and `M11.C.22`: reclassified mixed. `M11.C.29`'s budget limb is proved by a unit test; its time-course
    limb is untestable in the arm, since no genuine `I-POSITION` completes. `M11.C.22`'s mild limb is the net of
    lock-in and other mechanisms; the earlier statement that the capacity rule caused its failure is corrected (over
    600 seeds the rule shifts the limb by −0.15 ± 0.53, ad hoc).
- **Built.** `M11.C.46` (the seeker's relief with a third closer to the seeker against one closer to the partner);
  `M11.C.7` (the coach's position: a coach is on a member's side for `triangle_activity_window` weeks after that
  member's `SPLIT` toward him, and a sided coach's contact lands nothing, by `positions_live`). The gate acts in every
  run with a coach. `Forced.withheld` accepts a `KIND>target` label.
- **Proposed out of version 1.** `M11.C.9` (no rule reads sibling position) and `M11.C.8` (Phase E).
- **Recorded mutations (V3).** `M11.C.11`, `M11.C.15`, `M11.C.39`, `M11.C.46` and `M11.C.7` in the Phase D mutation
  record; `M11.C.4`'s two in the Phase C record.
- **V7.** `docs/REVIEW_version_1_outputs.md`, with its framing checks in `tests/test_version_1_outputs.py`.
- **Reviews** (one `learning-qa` pass, one correctness pass, before the regeneration): fixed `M11.C.46`'s readouts (a
  week early, and summing other transfers), `M11.C.32`'s mover not re-initialised, a no-op mutant target, a mutant edit
  that never ran, two imprecise mutant descriptions, a silent `withheld` label, weak tests, and the status tool's
  silent paths. Minor items are in `TODO.md`.
- **Records.** Every record regenerated once after the fix (3 h 34 min on 10 cores; code hash `43cd03b8…`, Phase C
  `2fc39cf9…`). Verdicts: `M11.C.4`, `M11.C.32` and `M11.C.46` pass; every `M11.C.7` pair fails on its deviation limb
  and holds on the landed-contacts limb; the others as before, Phase C family numbers moved by the coach's gate. The
  generated status page reads V2 met and V3 met; V1 waits on the owner's review of 23 proposed dispositions.
  Suite: 909.
- **A defect caught after the first regeneration was started.** `M11.C.46`'s module-level `SEEKER, PARTNER, THIRD`
  rebound `PARTNER`, which `M11.C.22`'s and `M11.C.11`'s readouts read as the nuclear four's c2, so both changed with
  no change to their own code. Found by comparing seed 0 of every Phase C and part I criterion against HEAD; the run
  was stopped and restarted after the fix. Every criterion without a coach now matches HEAD exactly; those on the Phase
  C family differ through the coach's gate. New test: no criteria module binds a top-level name twice
  (`tests/bowen/test_ensemble_record.py::test_criteria_modules_bind_no_top_level_name_twice`).

### Spec revision 31: the owner's review of revisions 24–30 applied (2026-10-16 and 2026-10-17)

*Added 2026-10-17: these entries were missing from the changelog when revision 31 was built. Written from the commit
messages of `6c95465`, `50c6eaf`, `224aa88`, `14b9df3`, `df33d68` and `1f0064f` and from plan §4.11.*

- **The review (2026-10-16).** The owner decided all 85 entries of `docs/REVIEW_spec_revisions_24-30.md`: 81 approved as
  built, 4 changed (`M14.A`'s `channel_focus` and `tolerance` rows; `M11.C.14` restated, option B; `M11.C.6` limit (a)
  restated, option A; `M7.E.1f`'s band). Version 1's condition V4 is met.
- **Part A, documents only.** The changed entries, the text fixes and the seven inconsistencies the form listed;
  every revision 24–30 tag reads "approved by the owner 2026-10-16"; the spec's version reads revision 31.
  `tests/test_review_rev31.py` pins each change.
- **Part B, code and config.** `M11.C.14` as two technique variants, each against a baseline in which the parents hold
  back on the same weeks, with a one-sided bound mode in the runner; its gate-removed mutant survives both (clause not
  met, recorded). `M7.E.1f`'s band opened by the parents' acts: ±2% with none, to +5% with supportive acts, to −10%
  with projection, limiting the step only. The criteria settings are graded `[I]` and the loader enforces it.
- **The owner's premise decisions.** A capacity rule in `M6.I.4` (what an over-functioner takes is limited by their
  own room above the stress floor); `M11.C.23` reclassified composite, its failure accepted as a finding; `M11.C.40`'s
  clause 2 reported and proved by a unit test. On 2026-10-17 the owner kept the capacity rule, accepted `M11.C.22`'s
  failure "for now", and had `M11.C.21` reclassified composite with its failure accepted.
- **Records.** Every record regenerated (about 3.5 h on 10 cores), then again on 2026-10-17 with `M11.C.21`'s class and
  the capacity rule's no-span fix (a taker exactly at basic level takes in full when there is no span). Suite: 886.

### Platform-independent fingerprints, for the red CI (2026-10-15)

- CI had been red on every push since 2026-10-09. Only the golden fingerprints failed, and on Python 3.11 the
  diagnostic record too. The cause was the last bits of floats: Python 3.12 changed how `sum` adds floats, and the C
  maths library differs between macOS and Linux. Measured on this Mac, Python 3.11 against 3.12 on the same seeds:
  no record differs in structure; 75,496 of 197,685 phase_d floats differ, by at most 1.7e-11 relative. NumPy 2.5.4
  against 1.26.4 changed nothing.
- `tests/bowen/test_engine_golden.py` now hashes canonical JSON with floats rounded to 8 significant digits. Both
  fingerprints moved; the run itself did not change. A new test checks that a 1e-12 change leaves the fingerprint alone
  and a 1e-6 change moves it. With rounding turned off it fails.
- `tools/diagnostic_record.py` takes its mean with `math.fsum`. The record is unchanged. 29 other `sum(...) / len(...)`
  means are left as they are and listed in TODO.
- Suite: 850 tests. CI run 38763412772 is green on Python 3.11, 3.12 and 3.13, the first since 2026-10-09.

### `M11.C.2` on the twelve-person family: no single revision 30 change explains the regression (2026-10-15)

- Each revision 30 change removed alone, 150 seeds, ad hoc: unmodified +0.70 ± 1.5, p 0.16 (reproduces the record); `M4.D.1h`'s force removed +0.70 ± 0.97, p 0.13; `M1.B.8b`'s compounding removed +1.12 ± 1.7, p 0.09; `M1.B.8b`'s reinforcement removed +1.11 ± 1.6, p 0.093; `M1.D.8a` removed +0.96 ± 1.5, p 0.14; `M1.B.3e`'s danger removed +0.08 ± 1.8, p 0.46 (`symptom_load`, seeds 0–149, one-sided signed-rank with the runner's margin; ad hoc, 2026-10-15).
- No single removal restores the pass. Removing the force narrows the spread to revision 29's (±0.95 then) without
  raising the mean; removing either focus term raises the mean to about +1.1 with the wider spread. Recorded in plan
  §4.10 and TODO.

### `M11.C.10[family of origin]`'s clause restated to `M4.D.6f`, and a Phase D mutation record (2026-10-15)

- The owner: "restate the clause to M4.D.6f and add the mutant". The spec's `M11.C.10` row now requires `M4.D.6f`
  removed to turn the family-of-origin variant red; `M1.D.8a` removed is reported. The main variant keeps `M1.D.8a`.
- New `tools/phase_d_mutation_record.py` writes `docs/phase_d_mutation_record.md` from
  `docs/records_cache/phase_d_mutation.json`. Its child, `tools/phase_d_mutant_child.py`, runs `CRITERIA_D2` in a mutated
  copy; it is a separate file so the Phase C records' hash does not move. The part II record names two criteria
  `M11.C.10`, so its rows are matched to `CRITERIA_D2` in order, each id checked.
- First mutant, `witnessed-cutoff-learning-removed` (`witnessed_cutoff_gain` 0.05 → 0): **red**, FAIL,
  `cut_off_tie_weeks` −75.9 ± 132, p 0.91, 200 seeds, the same values as the ad hoc run.
- Tests: `tests/bowen/test_phase_d_mutation_record.py` (the mutant applies once, the record is current, the variant
  is red under it, and every passing part II criterion has a mutant here or is named with where its mutation is
  reported: `M11.C.15` and `M11.C.47`, both ad hoc in the plan).
- Review fixes (learning-qa and a correctness pass): the cache key also covers `apply` and `run_in_copy`, which copy the
  repository and apply the edit; an engine exception under a mutant is reported as a red and not cached, as in Phase C,
  and finished results are kept if the machine fails; the spec's `M11.C.10` row states the restated clause in its
  Mutations sentence, not only after it. The record was rerun under the new key.

### `M11.C.10[family of origin]`: learning from a witnessed cutoff carries its pass (2026-10-15)

- The three remaining candidate mutations, 200 seeds each, ad hoc: `M4.D.6f` removed (`witnessed_cutoff_gain` 0)
  gives −75.9 ± 132 tie-weeks; `M1.B.3d`'s reactive rise removed (`reactive_cutoff_gain` 0) gives +163 ± 117;
  `M1.B.3e`'s danger removed (felt danger from the declared floor only, 0 in this family) gives +102 ± 126. Unmutated:
  +162 ± 120. Only removing `M4.D.6f` puts the mean below zero, so the children learning from the renewals they witness
  carries the pass. The criterion's own signed-rank test was not run on the mutants. The spec's clause still names
  `M1.D.8a`; whether to restate it is in TODO. Recorded in the spec's `M11.C.10` row, plan §4.10 and TODO.

### `M11.C.10[family of origin]`: its pass is not carried by `M1.D.8a` (2026-10-15)

- The variant's mutation, `M1.D.8a` removed (both gains 0), does not turn it red: +240 ± 116 tie-weeks (`child_cut_weeks` +49.7 ± 44.1; 200 seeds, ad hoc, 2026-10-15), against
  +162 ± 120 with it. The intervals overlap, so removing it is not shown to change the result either way. The spec
  requires this mutation to turn the variant red, so the variant's pass has no mutation proof and is not evidence for
  `M1.D.8a`. Something else carries the parents' held cutoffs to the children's rejection; not yet traced. Recorded in
  the spec's `M11.C.10` row, plan §4.10 and TODO.

### Spec revision 30 built: origin cutoff, danger, focus lock-in (2026-10-14)

- Built plan §4.10 (575 numbered requirements, 47 acceptance criteria): the cutoff a child grows up with raises their
  chronic anxiety and pull toward closeness at 20 (`M1.D.8a`); danger read from received intrusion, its contact cost,
  and a cutoff principled in proportion to it (`M1.B.3e`); an anxious act lands with more force (`M4.D.1h`, the
  owner's "force rises with anxiety"); anxious focus moves to where it already is (`M1.B.8b`, "anxious focus
  compounds"); `M11.C.10` read on ties to partners and own parents at a fixed 200 seeds, with a family-of-origin
  variant; the danger premise `M11.C.47` on a new fixture. Seven new `[I]` constants; one family-config key
  (`dangers`).
- The owner's decisions after the reviews: felt danger is held while rejection is held; focus compounding became a
  shift among the children that adds no investment, at `focus_compound_gain` 10 ("Shift, gain 10"), after the first
  form grew without limit and a ceiling removed the lock-in.
- Review fixes, each with a test: `M11.C.47`'s arms differ in one person and read the danger felt from them (most of
  it is the child's); the cached twelve-person run's move counts were shared between arms and double-counted; the
  `origin_cutoffs` key, which the spec forbids, is removed; a dead person's declared danger stops being a floor; weak
  tests strengthened and wiring tests added.
- Records regenerated (code hash `c41fea60…`, Phase C `6a8c31f0…`):
  - `M11.C.10[family of origin]` **passes**: +162 ± 120 tie-weeks (p 0.007, 200 seeds).
  - `M11.C.10` still **fails**: +6.6 ± 130 (p 0.43, 200 seeds).
  - `M11.C.47` passes (premise): felt danger +0.050 ± 0.017, principled cutoff +0.020 ± 0.008; but the weeks of
    rejection held toward the low-level person are fewer, −16.6 ± 4.9 (reported only).
  - `M11.C.2` on the twelve-person family now **fails**: +0.70 ± 1.5 (p 0.16), from a pass at revision 29.
  - `M11.C.17` FAIL → PASS (+0.00053 ± 0.00063, p 0.013); `M11.C.44` PASS → FAIL (−0.030 ± 0.11). `M11.C.6` fails
    (+0.070 ± 0.039); `M11.C.15` passes. No mutation proof lost by a passing criterion.
  - Mutations: `M11.C.47` with the intrusion read removed reads exactly 0 in every seed (100 seeds, ad hoc), so it
    cannot pass. The `M11.C.10[family of origin]` mutation is not yet run (run 2026-10-15: it does not turn the
    variant red; see the entry above); `M11.C.10` and `M11.C.2` fail, so theirs are moot.
- The `rca` skill (review convergence audit) added at `.claude/skills/rca/SKILL.md`, from the owner's trans-summary
  project; its first audit: MIXED / INSUFFICIENT DATA, and CI red on every push since 2026-10-09 (TODO).
- Suite: 849 tests. Spec, plan §4.10, explainer §8.3, constants docs, criteria config, the danger fixture, README,
  `_STATUS.md`, CLAUDE.md, the proposal and TODO updated.

### Spec revision 29 built: cutoff and focus (2026-10-14)

- Built plan §4.8, R1–R7: reactive and principled cutoff (`M1.B.3d`), learning from what follows a witnessed cutoff
  (`M4.D.6f`), chronic anxiety as the mean of the childhood years (`M7.B.1a`), focus on a child's real or perceived
  weakness (`M1.B.8a`), focus steering projection-type acts (`M4.D.1g`), `M11.C.10` with a held cutoff and a tie-week
  readout, and `M11.C.2` read by age. Five new `[I]` constants; two optional family-config keys (`principled_cutoffs`,
  `weaknesses`).
- Review fixes, each with a test: a declared weakness acts only through an anxious parent; the principled kind ends with
  the declared rejection; focus keeps each projection kind's total weight; one learned-value key format
  (`src/bowen/engine/value_keys.py`); range checks on the new constants; skipped scripted renewals counted.
- Records regenerated (code hash `eeef3517…`):
  - `M11.C.2` on the twelve-person family now **passes**: +1.31 ± 0.95 (p 0.005). Mutations (ad hoc): the witness path off +0.27 ± 0.75, `M1.B.8a` with `M4.D.1g` off +0.45 ± 0.99; neither interval excludes zero.
  - `M11.C.10` still **fails**: −63.1 ± 200 (p 0.71). Removing `M4.D.6f`, the reactive rise, or both, moves nothing
    detectable (ad hoc, 100 seeds each).
  - `M11.C.6` fails (+0.030 ± 0.024); `M11.C.15` passes (death +1.34 ± 0.40).
  - Elsewhere: `M11.C.44` FAIL → PASS, `M11.C.17` PASS → FAIL. No mutation proof lost.
- `I-POSITION`'s ensemble tests run with focus steering off by a declared override (TODO: they rest on about three
  completions). Two part II tests tightened (`M6.3b`, `M6.3e`).
- Suite: 812 tests. Spec, plan §4.8, explainer §8.1–8.2, the part II status report, constants docs, criteria config,
  README, `_STATUS.md`, CLAUDE.md and TODO updated.

### Spec revision 29 (570 numbered requirements, 46 acceptance criteria): cutoff and focus, drafted (2026-10-13)

- Applies the owner's decisions after the diagnosis:
  - `M11.C.10` holds the cutoff for the run and reads tie states.
  - Learning from a witnessed cutoff (`M4.D.6f`).
  - Reactive and principled cutoff, as a placeholder (`M1.B.3d`).
  - Chronic anxiety built over childhood (`M7.B.1a`).
  - A parent's focus on a child's real or perceived weakness (`M1.B.8a`, `M7.E.1c` at `t0`), and focus steering acts
    (`M4.D.1g`).
  - `M11.C.2` read by age.
- The forms are Claude's recommendation, pending owner review. Plan §4.8, awaiting approval. Nothing is built.

### Why `M11.C.10` and `M11.C.2` fail on the twelve-person family (2026-10-13, ad hoc diagnosis)

- `M11.C.10`: the declared Ravi–Dario cutoff ends within about 2 years, and the uncut baseline is cut within weeks, so
  the arms differ only at the start. Two of `M7.E.3`'s three links have no mechanism: witnessed-cutoff learning and
  origin cutoff as an input to intensity. Cut-off acts number about 775 per run, mostly generation 3 cutting their own
  parents.
- `M11.C.2`: the parents spread projection evenly over the children (about 80 acts each). Chronic anxiety reflects only
  the year at 11, and the level band leaves no room. Late load follows each child's age.
- Plan §4.7; owner decisions in TODO. No code changed.

### `M11.C.15` keeps the transfer at a death (2026-10-13, owner decision)

- The owner: "keep the transfer in C.15, share at 1". `M11.C.15`'s arms now zero only grief and lost income
  (`NO_GRIEF_OR_INCOME`, `src/bowen/ensemble/criteria_d2.py`); `death_transfer_share` stays at 1 and no criterion
  overrides it.
- `M11.C.15` now passes: death +1.25 ± 0.38 (p 2.1e-08), recovery +3.11 ± 0.35.
- Its mutation, the bound anxiety discarded at the death, turns it red (death −2.03 ± 0.29; ad hoc, plan §4.7).
- `canonical_state` now includes the life course, and a new test shows a change to the life course alone changes it.
- Every record was regenerated (code hash `4dbf3f0f…`); no other verdict moved. Suite: 766 tests.
- Spec `M6.3b`, plan §4.4 and §4.7, `config/bowen/criteria.md`, `config/bowen/constants.md`, the part II status report,
  TODO and README updated.

### Phase D part II built: the life course (2026-10-13)

*`M11.C.15`'s result below is superseded by the entry above: it passes since the owner kept the transfer.*

- **The life course** (spec revision 28, plan §4, steps L0–L10): a new module, `src/bowen/engine/life_course.py`. It
  runs only when a family declares one, so Phase C, part I and their golden runs are unchanged.
  - Ages, with acting from 10; a child's chronic anxiety before 12 and development from 5 to 20.
  - Launch; partners and births from `config/bowen/partners_phase_d.md`, each new couple getting a family of its own.
  - Mortality, and the disposition at death (`M6.3a`–`3i`).
  - Grief and lost income, and `FIND_WORK` (`M7.C.1k`).
  - The coach's retirement and successor.
  - Life events are `Mechanism.LIFE_EVENT`, with no sender.
- **Twelve-person family:** the coach is tied to the coachee only. Starting ages are three settings in
  `config/bowen/family_phase_d.md`.
- **Constants:** 23 new ones in `config/bowen/constants.md`, all `[I]`.
- **Criteria** (`src/bowen/ensemble/criteria_d2.py`) and their record, `docs/phase_d2_ensemble_record.md`
  (`python3 tools/phase_d_record.py --part 2`, 52 minutes on 10 cores).
  - `M11.C.37` passes, with its mutant, as a test.
  - Four fail: `M11.C.15` (death +0.212 ± 0.35, p 0.135; recovery +3.11, holds), `M11.C.6` (+0.018 ± 0.028),
    `M11.C.10` (−12.6 ± 28) and `M11.C.2` on the twelve-person family (−0.13 ± 1.6).
  - An ad hoc diagnosis puts `M11.C.15`'s death limb on its arm design, which discards the dead person's anxiety
    (plan §4.7). It is an owner decision; no arm was changed.
- **The `M16.T.3` re-run** on the life course passes (`test_m16t3_sink_does_not_change_a_life_course_run`).
- **Records:** every record was regenerated; no Phase C or part I verdict moved.
- **Suite:** 765 tests, plus 45 ensemble-marked tests.
- **Report:** `docs/phase_d2_status_2026-10-13.md`.

### Spec revision 28 (566 numbered requirements, 46 acceptance criteria): the life course (2026-10-13)

- Applies the owner's answers to P13–P15 (`docs/implementation_plan_phase_d.md` §1b): what happens at a death
  (`M6.3a`–`M6.3i`), finding work after a lost income (`M7.C.1k`), starting ages as settings (`M2.A.0j`), the coach's
  career and single tie (`M2.A.0k`, `M2.B.2`), acting from 10, launch, partners, births and mortality
  (`M7.C.1f`–`1j`), chronic anxiety before 12 (`M7.B.1a`), and a child's development from 5 to 20 (`M7.E.1f`). The
  forms are Claude's recommendation, pending owner review. Built the same day (above).

### The Phase D ensemble record (2026-10-13)

- **`tools/phase_d_record.py`** writes `docs/phase_d_ensemble_record.md`: every Phase D part I criterion built so far
  over adaptive ensembles, stamped with the code hash, and the three not built with the reason (plan step 12). A
  default-suite test (`tests/bowen/test_phase_d_record.py`) fails when it is stale. 12 PASS, 8 FAIL, the same verdicts
  as the ad hoc runs. The mutations are not yet a record. Suite: 711 tests.

### Spec revisions 25–27: beliefs as a channel, projection, and step 10's criteria (2026-10-12) — Claude's recommendations, pending owner review

The owner asked for the work to proceed overnight without waiting for input; none of these is an owner decision.
- **Revision 25, B8 (`M9.6`, `M9.7`, `M1.A.11c`, `M5.C.1`).** Each person holds where they believe the family's
  difficulty lies, written from moves sent or delivered, by rules in a new config file `config/bowen/attribution.md`;
  a readout (`src/bowen/readouts/attribution.py`) recovers the sink from it. A witness's appraisal reads its belief
  about the tie it watches. The family's anxious focus writes the symptom channel. The marital-distance gate blocks
  improvement in a symptomatic child. New constants (`[I]`): `attribution_rate` 0.05, `focus_rate` 0.05,
  `belief_appraisal_gain` 1.0, `marital_distance_fraction` 0.25. New field: `Person.attribution`.
- **Revision 26, B9 (`M4.D.6e`).** The cross-person learning term is weighted by the actor's bond energy on each tie.
- **Revision 27 (`M6.I.4`).** Borrowed pseudo-self settles back, conserved, once a pair's exchange has stopped for
  `exchange_window` weeks (8); new `pseudo_self_relaxation_rate` 0.05 (`[I]`). This changes every run in which lending
  stops.
- **`M10.B.5`.** Learning is switched off by an override the criterion declares (`scenario(overrides=...)`).
- **B10, `M11.G`.** A family-evaluation readout (`src/bowen/readouts/evaluation.py`): components 1–8, adaptiveness
  derived against stress, belief and truth apart, the framing carried; `M11.D.20`'s ordering check passes.
- **`M11.C.23`'s suppression term**: new field `Person.symptom_suppression` (1 is none; set by a criterion only).
- **Criteria (ad hoc on the final code, not yet a record).** Pass and proved by mutation: `M11.C.26`, `.36`, `.39`,
  `.24`. Fail: `M11.C.14`, `.2` (acts counted early, load read late), `.40` (its second clause), `.43`, `.34` (reactions
  to the onset against a no-onset run), `.21`, `.23` (reversed). Not built: `M11.C.7`, `.9`, `.28`, each needing a
  mechanism first.
- **`M11.C.17`'s coached rise nearly vanished** (+0.021 at revision 23, +0.0009 ± 0.0008 now; still PASS at p 0.011).
  Probes put part of it on revision 27's settling back and part on revision 24's integrand. Recorded, not tuned.
- **Reviews' fixes:** the marital-distance gate was on almost every week (being below the contact optimum is the
  resting state); it now needs felt contact below `marital_distance_fraction` (0.25, `[I]`) and reaches only the
  couple's children, not a symptomatic spouse. Pseudo-self settled between people who never exchanged; it now settles only on a tie with a
  functioning balance. Criterion overrides are applied before assembly. `M11.C.34`, `M11.C.2`, `M11.C.40` and
  `M11.C.26` readouts were corrected (see the plan, step 10); an empty entropy window no longer reads as zero. `TRIANGLE` no longer writes focus.
- **`I-POSITION` completion fell** in the triad fixture (about 10 in 96 seeds at revision 20, 4 in 192 now);
  `test_iposition.py`'s sequence test was raised to 192 seeds (P23).
- **Status report:** `docs/phase_d_status_2026-10-13.md`, every build item and D-I criterion as done, partial or not
  done.
- **Records regenerated.** `M11.C.44` moved PASS → FAIL again, on its threshold. No other Phase C verdict moved. The
  golden fingerprints moved (behaviour). Suite: 708 tests.
- **Test fixed:** `tests/bowen/test_phase_c_criteria.py`'s C.42 check counted any event a pair member sent toward the
  third, including a symptom onset, which is not a choice; it now counts moves only.

### Spec revision 24: the symptom channels, Phase D B7 (2026-10-12) — Claude's recommendation, pending owner review

The owner asked for the work to proceed overnight without waiting for input. Every choice here is Claude's
recommendation, labelled so in the spec, and none is an owner decision.
- **`M4.C.3a`, amended.** Symptom load builds from the excess above the person's usual level (revision 22's
  yardstick), not above the chronic floor. Measured before it: in ten-year runs of the nuclear-four fixture every
  person crossed the threshold in the first weeks and stayed there.
- **Built:** the expressed channel moved by a family focus and overridden by constitution (`M1.A.11c`, `M7.D.2`);
  mutual protection between channels (`M7.D.2c`); lock-in with its turn, through a new stock `Person.symptom_bound` in
  `M6.4`'s table (`M7.D.2a`, `M7.D.2d`); a remitted symptom's hold returns to the partners (`M1.F.9`); removal fired
  by the others' tolerance, suppressed by external decision ownership (`M7.D.4`). A remitted symptom's load is no
  longer credited to the family budget (it is not an anxiety stock). The symptom threshold is floored at
  `functional_level_floor`, as steepness is (`M1.A.6`), and every channel is checked for onset, each crossing its own
  event (`M7.D.1`).
- **New constants** (`[I]`): `constitution_strength` 0.5, `channel_protection` 1.0, `lock_in_gain` 0.2,
  `lock_in_turn` 3.0. **New fields:** `Person.channel_focus`, `constitution_strength`, `symptom_bound`;
  `Family.decision_ownership_external`.
- **`M11.C.22` restated** to read each limb against removing the same member with no symptom.
- **Result (ad hoc):** `M11.C.30`, `M11.C.22`, `M11.C.31` and `M11.C.11` pass. C.30 and C.22 turn red under their own
  mutations; C.31's mutant cannot run, because discarding the held anxiety breaks the `M6.I.6` ledger at the release
  week. `M11.C.12` fails on the dyad.
- **Test fixed:** `tests/bowen/test_famdiag_export.py` still counted `CUTOFF` acts as a cutoff's support; since
  revision 17 a cutoff is read from the other's refused approaches. Seed 3's run under this engine was the first to
  read a cutoff with no `CUTOFF` act in its block.
- **Records regenerated.** `M11.C.44` moved FAIL → PASS again: it sits on its threshold (see `TODO.md`). No other
  Phase C verdict moved. The golden fingerprints moved (behaviour). Suite: 683 tests.

### Spec revision 23: the estimator reads pseudo-self with its sign (2026-10-12)

- **`M1.A.4k`.** The estimator's history reads `functional_level` less pseudo-self with its sign: what was given way is
  added back, as what was borrowed was already left out (owner decision). The exchange shifts functional level in both
  people and leaves basic level alone.
- **Why.** A trace showed the load weighting barely mattered; leaving out only the borrowed part read the coached
  people about 9 points below basic level whatever they did (`docs/implementation_plan_phase_d.md`, step 6).
- **Result (ad hoc, 150 seeds).** **`M11.C.17` passes and is proved by mutation** (CLAUDE.md's bar: removing the
  landed contact turns it red): coached +0.021 ± 0.009. No minimum meaningful effect is declared, and the effect is
  small (about 0.02 points of basic level in ten years). In a 20-seed trace, 16 of 40 coached people had risen by year
  10 against 4 of 40 without a coach. C.33, C.18 and C.20 pass (+0.013 each); C.18 and C.20 each turn red under their
  own mutation, and C.33's own mutation (the slow-clock dependence gate) was not rerun.
- **New constants:** none.
- **Records regenerated:** no Phase C verdict moved. The golden fingerprints did not move; those runs do not exercise
  the estimator (see `tests/bowen/test_engine_golden.py`). Suite: 661 tests.

### Spec revision 22: calm is relative to the person (2026-10-12)

- **`M1.A.5e`, `M1.E.7`.** Loaded and calm are read against each person's usual level: their excess anxiety averaged
  over about a year (new `usual_level_rate`, 0.02, `[I]`; new Person fields `usual_excess`, `usual_weeks`). The stress response is
  driven by the excess above it; at or below it a person is calm, and what stress moved and the system's adjustment
  settle back. Self-discovered practice counts only above it. The policy's bands and the estimator's load weighting are
  unchanged (owner decisions).
- **Why.** A trace showed the Phase C family's own traffic holds the coached pair 30–35 points above chronic (the
  others 2–25), with or without nodal events, so against a fixed band of 5 the coached pair were never calm.
- **The usual level's start.** It is the plain mean of the weeks so far for the first year, then the one-year running
  average (owner decision, after review: started from week 0 alone, the first year was over-load).
- **Result (ad hoc, `docs/implementation_plan_phase_d.md`, step 6).** The stress response now settles, and coach-free
  people's functioning no longer erodes (+2.7 over ten years). Coached people still fall a little (−2.2), through giving
  way. The estimator's evidence stays near −9, and every criterion's readout is now tiny (about +0.002 a person):
  C.33, C.18 and C.20 pass the runner's test, and `M11.C.17`'s pass (+0.003) is not counted.
- **Records regenerated:** no Phase C verdict moved. Suite: 659 tests.
- **New constants:** `usual_level_rate`.

### Spec revision 21: holding a position stops the functioning exchange (2026-10-12)

- **`M1.B.5`.** An `OVERFUNCTION` or `UNDERFUNCTION` delivered to a person holding a position toward its sender moves
  no functioning, pseudo-self or balance, either way (owner decision). Holding a position is read from behaviour: a
  self-directed act or `WITHHOLD` toward that person in the last `practice_window` weeks, and no reactive act toward
  them since — any automatic act toward them, or triangling someone in about them (owner decision, after review)
  (`src/bowen/engine/moves.py::holds_position`, `external.record_stances`). Logged as `position_held`.
- **`M11.C.20`** reports the share of its borrowed arm's scripted support that a held position blocked (report only;
  owner decision: keep the rule, show it).
- **Why.** A trace showed coached people stopped over-functioning, and others then took charge of them more, taking
  functioning from them each time (`docs/implementation_plan_phase_d.md`, step 6).
- **Result (ad hoc).** `M11.C.33`, `M11.C.18`, `M11.C.20` pass, none of them carried by this change. `M11.C.17` passes the
  runner's test (+0.01 ± 0.01, p 0.02) but is **not counted**: the interval includes zero and no coached person's basic
  level rose in the trace. With stances ending on any reactive act, coached people fall as far as coach-free ones
  (−11.4 against −11.1); with soothing alone ending them they fell less (−8.0 against −12.3). Under constant load
  nobody's basic level rises; the owner's questions on C.17's load and the estimator's form remain.
- **Records regenerated.** `M11.C.44` moved back PASS → FAIL, as it had moved FAIL → PASS at revision 20: it sits on its
  threshold (see `TODO.md`). No other Phase C verdict moved. Suite: 656 tests.
- **New constants:** none.

### Spec revision 20: the system's adjustment settles back; `M11.C.18` and `M11.C.20` restated

From the owner's decisions of 2026-10-11:
- **The system's adjustment settles back** (`M1.E.7`). The gain the other person takes when a held stance works is
  kept in their `adjust_swing` and shed by `response_relaxation_rate` each calm week, as the stress-moved part is. While
  they are loaded it holds. The estimator leaves it out (`M1.A.4k`), as it leaves out borrowed pseudo-self: the
  owner chose this after the review showed a held adjustment could otherwise become a lasting basic rise.
- **It is a level, not a stack** (2026-10-12). Built first as a stack, it grew to 20–35 points under the criteria's
  steady load and carried `M11.C.17`'s pass (the +0.66 reported for `a35a962`, and revision 19's +0.63, were that
  defect). Now the other is up by at most `system_adjust_gain`, renewed while the stance works. A first fix also took
  falls off the adjustment; the review found that recorded each fall in two ledgers and let the stack return, so it was
  removed. Every change to functioning is now in exactly one ledger.
- **`M11.C.18` and `M11.C.20`'s mutation clauses** now name the mechanism that carries each: the stress response
  (`M1.A.5e`) for C.18, and the estimator reading functioning without borrowed pseudo-self (`M1.A.4k`) for C.20.
  Flattening `M1.A.4b`'s load weighting leaves both passing; the load weighting stays proved by its unit test.
- **Result (ad hoc, not yet in a record).** `M11.C.33`, `M11.C.18` and `M11.C.20` pass, and each restated mutation
  turns its criterion red. **`M11.C.17` fails**: in ten years nobody's basic level rises in either arm. A ten-year trace
  of the coached people (`docs/implementation_plan_phase_d.md`, step 6) shows coaching changes behaviour, but under
  load in every week functioning falls in both arms and the estimator's evidence stays far below zero. Open for the
  owner (`TODO.md`).
- **Records regenerated.** One Phase C verdict moved: `M11.C.44` FAIL → PASS (`outside_inside_ratio` +0.067 ± 0.078,
  p 0.055, became +0.101 ± 0.082, p 0.012). It sat on the edge before, and both its mutants survive, so by `M11.5` it
  is not proved (`docs/spec_coverage.md` keeps it partial). The mutation record's survivors went from 55 to 57: those
  two. Suite: 651 tests.
- **New constants:** none.

### Spec revision 19: the stress floor, self-discovered perspective, the system adjusting, the soothing review

From the owner's answers of 2026-10-11, after tracing why `M11.C.17` ran the wrong way:
- **Stress floor.** Stress, and giving way to a partner, cannot push functioning below basic level − 15
  (`M1.A.5e`). Without the floor, soothing spiralled and anxiety ran away in 11 of 20 four-year runs. With the floor on
  stress alone, 2 of 20 still did. With it on both, none of 20.
- **Self-discovered perspective** (`M1.E.7`). A self-directed act builds perspective only if it worked: the person held
  their position, and their reactivity on that tie came down over 4 weeks.
- **The system adjusts** (`M1.E.7`). When a held stance works, the other person functions up.
- **Perspective steers** (`M4.D.1a`; `config/bowen/perspective_steering.md`): away from triangling and blaming, toward
  detriangling.
- **The estimator ignores borrowed functioning** (`M1.A.4k`). It reads functioning less the positive pseudo-self that
  over-functioning took.
- **The soothing review** (`M4.D.6`). Now and then a person asks whether soothing helped, more often at a higher level
  and perspective. If it did not, soothing loses its learned appeal.
- **`M11.C.20`.** Perspective only in the basic arm.
- **`M11.C.17`.** Ten years, reading the people the coach reaches.
- **Result (ad hoc, 150 seeds).** All four estimator criteria pass, and each is turned red by removing the mechanism
  that carries it. The spec's load-weighting mutation leaves C.18 and C.20 passing, which was open for the owner (restated at revision 20).
- **Records regenerated:** no Phase C verdict moved. Suite: 647 tests.
- **New constants:** six `[I]`.

### Spec revision 18 (547 numbered requirements, 46 acceptance criteria): stress, coaching and self-work

From the owner's answers of 2026-10-11 (`docs/PROPOSAL — STRESS ON FUNCTIONING, COACHING AND SELF-WORK.md`, approved):
- **Stress moves functioning** (new `M1.A.5e`). Under load, coping (a self-directed act or holding back) raises
  `functional_level` and avoiding or soothing lowers it, each scaled by basic level. The soothing acts are listed in
  `config/bowen/stress_response.md`. The moved part settles back slowly when calm (`src/bowen/engine/stress_response.py`).
- **A stressor lasts** (`M1.F.6`): it weighs on its targets for its declared duration.
- **Systems perspective opens the self-directed channel** (`M4.D.1a`).
- **Perspective grows by a person's own practice** as well as by coaching (`M1.E.7`): 0.001 per self-directed act
  under load. At 0.01 nearly everyone saturated within ten years.
- **New constants:** six `[I]`.

### Spec revision 17 and the estimator's criteria

- **Revision 17** (owner, 2026-10-11):
  - nodal events are any major event on a person (`config/bowen/nodal_events.md`; new kinds `JOB_CHANGE`, `MOVE`,
    `SERIOUS_ILLNESS`);
  - cutoff is read from behaviour (`src/bowen/readouts/patterns.py`);
  - `M11.C.17` rewritten: a coach helps and is not required;
  - `M11.C.7`'s direction: joint > one spouse > sided.
- **The estimator's criteria**, in `src/bowen/ensemble/criteria_d.py`, each with a declared three-kind nodal load. Ad hoc
  run, not yet in a record:

  | Criterion | Verdict |
  |---|---|
  | `M11.C.33` | passes, and is proved by mutation |
  | `M11.C.18` | fails: +0.05 ± 0.11 |
  | `M11.C.20` | fails, reversed: the borrowed gain rises more |
  | `M11.C.17` | fails: no coach effect |
- **The causes,** recorded for the owner in the plan's step 6 result:
  - anxiety does not move `functional_level`;
  - the coach's effect does not reach the behaviour the estimator reads.

### Phase D B5 and B6: the slow tick and the basic-level estimator

- **B5, the slow tick** (`M3.B.1`; owner decisions P5, P6, P17).
  - **When:** it runs at the end of weeks 51, 103 and so on (week 0 is the start), after the fast tick's invariants,
    with its own invariant check.
  - **What it does:**
    - chronic anxiety falls only with a rise in `basic_level`;
    - ties wear by their members' chronic anxiety, offset by `STAY-IN-CONTACT`, with bond energy untouched;
    - `dyad_age` counts years.
- **B6, the estimator** (`M1.A.4a`–`4d`, `4h`, `4k`, `M7.A.1a`; owner decision P4). Over three years of weekly history
  it takes:
  - the load-weighted mean and spread of `functional_level`;
  - the breadth in kinds of nodal event;
  - the self-directed share under load.

  `basic_level` rises only on a loaded, stable, broad window, never for a dependent person, and never falls in D-I.
- Six `[I]` constants.
- *(Superseded below by revision 17's criteria, which carry three nodal kinds.)* **No rises yet, and that is not a result.** No scenario has more than one kind of nodal event, so the breadth gate is
  never met. Without a coach, several people already show positive evidence. Whether the coach-free arm stays flat
  (`M11.C.17`) is untested until a window holds three kinds (open risk in `TODO.md`).
- **Review fixes:**
  - a refused `STAY-IN-CONTACT` no longer offsets wear;
  - the history test exercises a stressor, the channel and the week's own level;
  - the one-year contact window is tested;
  - `dyad_age` is logged;
  - contact counts in the year it arrives.
- **Records regenerated:** no Phase C verdict moved. Suite: 627 tests.

### Spec revision 16 and Phase D B3/B4: cutoff as rejection

- **B3, channel stamping.** A policy act is logged with its kind's channel, so `I-POSITION`, `STAY-IN-CONTACT` and
  the other `self` acts are `self_directed`, not `automatic`. This changes the log only.
- **B4, cutoff as rejection** (owner decisions P1 and P2; spec `M1.B.3c`):
  - **The tie stays open.** A `CUTOFF` raises the cutter's `rejection`, and the tie stays interactive.
  - **Refusal.** Moves sent to the cutter are refused while their rejection is above 0.3. A refused move does not
    reach the cutter, raises their rejection, and costs the sender anxiety.
  - **Lowering rejection.** `REDUCE_CUTOFF` lowers both members' rejection. Rejection also fades by 0.05 a week, and
    `RECONCILIATION` clears it.
  - **The pull.** A cut tie rests at no contact for both members.
  - **Declared cutoffs.** Ana–Bruno and Ravi–Dario start open, with both members rejecting.
- **Spec revision 16 (owner, 2026-10-11): graded rejection.**
  - **Why:** as first built, refusal at any rejection above zero left 80–88% of tie-weeks cut off.
  - **The values:** the refusal threshold is 0.3, not the 0.5 first proposed, because at 0.5 one cutoff refused
    nothing.
- **Measured effects** (Phase C family, seeds 7, 1 and 3, two years; `docs/implementation_plan_phase_d.md` B4):
  - **Cut tie-weeks** are near the pre-B4 level.
  - **The nuclear family runs more anxious,** because a cut tie is no longer silent.
  - **The `I-POSITION` ensemble tests** now run on the triad fixture, where preparation can happen (owner, option a).
  - **The `M11.C.29` test** now asserts that every withdrawal is legal without the hold.
  - **The dyad symmetry test** uses 200 seeds.
- Seven `[I]` constants were added and logged in `constants_changes.md`. The golden fingerprints moved (B3, B4).
- **Review:** `learning-qa` and a correctness pass found 10 items, all fixed with tests. The new `M1.B.3c` invariant
  checks each week that a tie is cut off exactly while rejection is held.
- **Records regenerated under the B4 engine.**
  - **One Phase C verdict moved:** `M11.C.32`, PASS → FAIL. Its `reached_peak` difference shrank to −0.02 ± 0.022,
    because `PREPARE` now usually fails on anxiety in both arms.
  - Every other verdict is unchanged.
  - Suite: 609 tests.

### Spec revision 15 (546 numbered requirements, 46 acceptance criteria)

The owner approved `docs/PROPOSAL — PHASE D DEFINITIONS.md`:
- **C.14:** "management technique" is a parent's weekly act toward the symptomatic child (firmness, permissiveness,
  punishment, talking it out).
- **C.23:** the expressed-emotion term (hostility, criticism and over-involvement, read through appraisal) and the
  suppression term (a factor on symptom accumulation, 0.5).
- **C.28:** the mobility term (the decay of the activation memory that holds the family's focus).
- **New `M11.C.46`:** a triangle relieves its seeker in proportion to the third's help, so the mutant
  `triangle-help-ignored` must turn it red.

### Phase D engine work, 2026-10-10/11

- **B1:** windowed event-store indexes, so a run's cost per week is flat. Seeded runs are byte-identical; a 40-year
  run of the twelve-person family takes 12.9 s, against about 110 s.
- **B2:** the batched TODO items.
  - The trace framing is read from the run.
  - The engine hash is the same on every Python version.
  - The readouts take `now`.
  - The readouts' act sets are in `config/bowen/pattern_acts.md`.
  - One shared C.44/C.45 filter.
  - Review fixes for each.

## [Unreleased] — 2026-10-10

### Phase D plan approved; spec revision 14 (545 numbered requirements)

- `docs/implementation_plan_phase_d.md`, approved by the owner with decisions P1–P5, P7 and P17 (§1a). Phase D splits
  into D-I (foundations) and D-II (the life course).
- Revision 14 applies them:
  - `M1.B.3c`: cutoff is rejection, not severing.
  - `M4.C.1c`: the reconnection pull.
  - `M5.F.2c`: the counterfeit flip is gradual, crossing at 0.5.
  - `M1.A.4k`: the estimator's form.
  - `M7.B.1`: chronic anxiety is fixed and moves only with a rise in `basic_level`.
  - `M7.E.4`: chronic anxiety wears on ties; the bond-energy accrual is withdrawn.
  - `M2.A.0b`: `t0` is week 0, not a slow tick.
  - `M11.4` gains `M11.C.14`'s null.
  - `M11.C.13` moves to Phase E.
  - The blank line splitting the criteria table is removed.

### Added — run visualization (`docs/PROPOSAL — RUN VISUALIZATION MAPPING.md`, plan V1–V7, owner-approved)

- **The pattern readouts are the owner's** (V1, V2; `docs/readouts_phase_c.md`, `src/bowen/readouts/patterns.py`).
  - A pattern is 3 or more acts in 4 weeks (`pattern_window` 8 → 4, `pattern_min_acts` 2 → 3, `pattern_pole` retired;
    logged in `constants_changes.md`).
  - No interaction is an emerging level of distance. Cutoff is refusal of approaches, not absence.
  - Pursuit reads as fusion, over/underfunctioning or projection. Projection and fusion are new readouts.
  - `current_patterns`: a pattern ends only when a new one emerges.
  - No criterion reads these readouts, so no verdict moved.
- **Spec revision 13** (V3): `M2.A` is the owner's family. Bruno and Toma leave; Viktor (Sofia's husband) and Dario (Ravi's
  brother, cut off) join. Leo is declared male. Teodor's `basic_level` is proposed at 36. `M2.A.1`'s fixture is now Iris and
  Dario. New `config/bowen/family_phase_d.md`, not yet run; the Phase B and C families are unchanged.
- **`docs/DECISIONS — CUTOFF SEMANTICS.md`** (V4): should the engine's CUTOFF become rejection? Held for Phase D
  (owner, 2026-10-10, option C).
- **`tools/famdiag_export.py`** (V5) writes a run as a fam-diag diagram (`docs/review/phase_c_famdiag_seed7.json`).
  - One fam-diag year is one 4-week block, 2001–2026.
  - The drawing rules are in `config/viz/famdiag.md`, outside the engine hash.
  - Every line traces back to the run's acts (`tests/bowen/test_famdiag_export.py`).
- **EPModel Fam-Diag** (V6): a separate copy of fam-diag (`../epmodel-fam-diag`, port 5174, private repository
  `dbgnvan2/epmodel-fam-diag`), so the owner's fam-diag stays as it is.
  - A line marked `hideAfterEnd` hides after its end year, and a dated triangle shows only inside its span.
  - Triangles keep their notes and events on load.
- **Opened in fam-diag** (V7): blocks 1, 13 and 26 match the export, and the trace agrees at blocks 9 and 26.

## [Unreleased] — 2026-10-08

**Phase C built** (steps 0–16 of `docs/implementation_plan_phase_c.md`). **Its acceptance gate does not pass**;
the per-criterion status is in `docs/phase_c_completion_report.md`.

### Added

- **Phase C closed, 2026-10-09, with its gate failing** (owner decision; report §13). Done: C.1, C.3, C.19, C.25,
  C.32, C.35, C.38. Partial: C.16, C.27, C.41. Not done, recorded as findings: C.4, C.5, C.29, C.42, C.44, C.45. §13
  lists what the next phase inherits. Open: plan §9's human reviews and Bruno's weekly fallback.
- **The approved failing-criteria decisions, carried out 2026-10-09** (results at the top of
  `docs/DECISIONS — PHASE C FAILING.md`, report §12). All test restatements, logged as post hoc:
  - **S:** every scripted act's ties are held open, in both arms, from week 0 until the act (`Forced(held_open=…)`,
    `legal_outcomes(excluded=…)`), so every act is made; the ensemble record reports each arm's acts not made. The
    approved first choice, the latest week legal in every seed, measured as week 0 for every criterion, where every
    member is on the chronic floor and no relief can show, so it was not used.
  - **Q2:** `M11.C.41`'s readout is the reactive choice less the legal set's reactive share (`reactive_over_chance`, centred
    on a chooser at random); both level limbs now pass. **`M11.C.29`** counts the weeks the third person's symptom is active.
  - **Diagnostics:** `s-scripted-weeks`, `d7-rest-state` (members with several ties rest well above the floor at every level,
    from the members' own acts) and `q3-act-effects` (a `TRIANGLE` relieves the pair; `M4.D.6e`'s cross-person
    credit counts the third's cost). Summary tables now show the minimum.
  - Verdicts: `M11.C.3` passes more strongly (pair −5.27); `M11.C.41` passes in 2 of 4 cells; C.4, C.5, C.29, C.42,
    C.44, C.45 and C.27 fail.
  - **Owner decisions on the results:** X1 a finding; `M11.C.5`'s FAIL reported; `M11.C.4`'s nodal event and
    `M11.C.27`'s reading moved to 104 weeks (one declared value, run once): no effect remains in either, and
    C.27's one passing cell no longer passed. The owner then decided `M11.C.27` is read week by week, gating on
    the week the act lands: 1 of 4 cells passes. `M11.C.41`'s readout counts the policy's draws only. The owner's account of triangling became a proposal
    (`docs/PROPOSAL — TRIANGLE RELIEF AND CREDIT.md`), built the same day (next entry).

- **The triangle change, approved and built 2026-10-09** (`docs/PROPOSAL — TRIANGLE RELIEF AND CREDIT.md`; spec
  `M1.C.1`, `M4.D.6e`, `M11.C.3`). A `TRIANGLE` relieves only its seeker, scaled by `third_help` (the third's calm
  times their alignment with the seeker rather than the partner); the partner is not relieved. Its credit leaves the
  recruited third out of the cross-person term. `M11.C.3` now reads the seeker's change and passes; `M11.C.42` and
  `.45` still fail. New named mutants `triangle-help-ignored` (survives on C.3) and `triangle-credit-cross-person`
  (no passing criterion to run on). `M11.C.27` is read week by week, and `M11.C.41`'s readout counts policy draws
  only.

- **The engine hash ignores comments, docstrings and blank lines in Python source** (`code_text` in
  `tools/ensemble_record.py`, learnings P38): fixing prose in `src/bowen/` no longer reruns every record (about an
  hour). Docstrings are found with `ast` and comments with `tokenize`, by source position, which is stable across
  the Python versions CI runs. Config `.md` files stay hashed whole: their headings decide which table a row is in.
- **`seeker-relief-removed`**, a deletion mutant that proves `M11.C.3`'s seeker readout (the transfer and roles
  mutants reach it only through the third's).
- **Report §11's qualitative claims are generated** with its numbers: whether an inversion reverses `M11.C.16`, and
  whether removing its grounds turns it red, are chosen from the mutant results.

### Fixed

- **Report §11 said no inversion reverses `M11.C.16`**, a constant sentence in `tools/c16_report.py`; since the
  triangle change `availability-level-inverted` reverses it, and `c16-grounds-removed` survives. The class goes
  back to the owner (`TODO.md`).
- **`scripted_weeks` read past its own shortened runs** once an arm read weekly readouts; the scan now stops the
  arm after scanning.
- **A record built in the run that produced its results did not match its own re-render** (2026-10-09):
  `run_cached` returned the fresh row in the child's key order rather than the cached, normalised copy.
- **`tools/c16_report.py` dropped everything after report §11**; it now replaces §11 only.
- **A scripted act was counted as made when it was placed, not when it was emitted** (batch review, 2026-10-09):
  an act toward the actor's open I-POSITION sequence's other is rewritten to `STAY-IN-CONTACT` after selection.
  `Forced.outcomes` now classifies each act from the run's records (made, not legal, owed a step, dead,
  rewritten, not reached); the scripted criteria are found by running the arms, not from typed lists.
- **The first Q2 readout scored 1/N for a chooser at random**, so it rose whenever the legal set shrank (batch
  review); replaced by `reactive_over_chance` before it was relied on.
- **A held-open tie's redraw replaced a declared absence's redraw in the same week** (latent): one redraw now
  carries both.

### Added (earlier)

- **Diagnosis of the failing Phase C criteria, 2026-10-09** (`docs/plan_phase_c_failing_criteria.md`, approved).
  `tools/diagnostic_record.py` and `tools/probes.py` write `docs/phase_c_diagnostic_record.md`; the owner's decisions
  are asked in `docs/DECISIONS — PHASE C FAILING.md`. The main finding is a test defect: a criterion's scripted act is
  skipped (not legal that week, usually because the policy already cut the tie) in 25-60% of seeds, and a skipped
  seed adds a difference of exactly 0 (C.3, C.4, C.5, C.27, C.29, C.35, C.42). Also: in the triad fixture members sit
  above their chronic floor almost every week with or without the spell (a calm-state hypothesis, D7 proposed);
  C.41's level limb conflicts with `M4.D.3a`; the learner credits `TRIANGLE` less than other acts under §8's roles;
  C.5's reaction grows over years in the seeds where the act was made. A first draft of the memo drew conclusions
  from medians over unmade seeds and was revised after the batch review.
  `M11.C.7`, `.13` and `.14` moved to Phase D. Report §3.6's account of C.29 was wrong (the third person is
  symptomatic in every week, not none) and is corrected.
- **Cached records, 2026-10-09.** The mutation and sweep records cache each result under the engine hash (engine,
  config, `RULE_KEYS` and `tools/mutant_runner.py`, the child that computes results) plus the mutant's or setting's
  own edits (`tools/record_cache.py`, `docs/records_cache/`). A record's test re-renders it from its cache and
  compares, so labels, rendering and judging change without a rerun, one new mutant reruns only itself, and an
  engine change still reruns everything. Re-rendering takes under a second, where any tool edit had forced a 35-45
  minute regeneration. The seeded cache was checked by rerunning two mutants fresh: the record and cache came out
  byte-identical. The occupancy record no longer hashes the whole mutant list, the sweep no longer hashes the
  mutation tool, and report §11 is generated by `tools/c16_report.py` with a test, so a stale number fails the suite.
  An errored row is never cached (it reruns next time), a failure of the machine running the child (a broken worker
  pool, memory, the OS) stops the run instead of becoming a cached red, and the cache is flushed after each mutant so
  a late failure loses no finished result.
- **Fifth csdp sweep fixes, 2026-10-09.** A mutant whose replacement text is broken is **broken**, for every kind,
  and never proof; an engine exception under a mutant stays a red. The occupancy record gives each span its own
  member-week count. The sweep record renders a row without readouts. The band test checks both sides and the
  comparisons. Report §11 no longer says the cited rules "do not carry it without availability": either set alone
  keeps a passing result, and only removing both removes it.
- Phase C of the agent model: revision 11's two-sided appraisal and standing load, the belief store (`M9.8`),
  move physics, the policy (two channels, `WITHHOLD`, competing urges, keyed selection), the learner, the
  `I-POSITION` state machine, the external agent and landed contact, `M6.4`'s ledger with `M6.I.6` asserted every
  tick, sink allocation, pattern readouts (`docs/readouts_phase_c.md`), the ensemble runner (paired signed-rank,
  Holm, adaptive stopping), the engineering gates, and the fixture families.
- The criteria as paired arms (`src/bowen/ensemble/criteria.py`) and three generated, hash-stamped records:
  `docs/phase_c_ensemble_record.md`, `docs/phase_c_mutation_record.md` (named, deletion, sign-inverted, a joint
  level-blind, and representation mutants, each run on a temporary copy of the repository) and
  `docs/phase_c_sweep_record.md` (plan D9). Default-suite tests fail when any record is stale.
- Constants frozen 2026-10-07 (`config/bowen/constants_frozen.md`), before any criterion ran.

### Results

- At the frozen constants, as first built: 6 criteria pass and are proved by a mutant (`M11.C.1`, `.19`, `.25`, `.32`, `.35`, `.38`);
  `.42` and `.45` pass but reverse in the sweep; `.27` and `.41` pass in some cells; `.16` is undetermined at the
  cap; `.3`, `.4`, `.5`, `.29` and `.44` fail; `.7`, `.13` and `.14` are not built.
- `M11.5` corrected from the real mutants and dated in the spec: `M11.C.1` and `.38` first proposed as joint premises (decided later the same day: they stay premise, with the nine rules listed); `.45` proposed
  as a premise, withdrawn when it stopped passing after the `TRIANGLE` decision (it stays composite).

### Changed after the freeze, and reported

- Three numerical fixes to the policy (a shift-invariant softmax, no `WITHHOLD` weight with no automatic act, the
  withheld act drawn from within-channel weights), each with a unit test proved by reverting it.
- `M11.C.25`'s readout, which could not fail, corrected after its mutant survived.

### Fixed

- From the Hermes `learning-qa-sweep` gate of 2026-10-08 (`docs/cycles/2026-10-08_phase-c-qa-gate.md`, APPROVED with
  1 medium and 3 low findings): the mutation and sweep records now hash the tools that generate them (medium);
  coverage reads `NOT_BUILT` from the criteria module, not the record's prose; the belief store's
  "no `true_counterpart`" check is an AST name check tied to the function, with its own mutation test. The fourth
  (low) finding is in `TODO.md`.
- From the second gate (`docs/cycles/2026-10-08_phase-c-qa-gate-2.md`, APPROVED), and the first gate's fourth finding:
  every record's hash now covers `tools/ensemble_record.py`, whose `RULE_KEYS` feeds every verdict (A, medium); the
  sweep staleness test imports the sweep tool's `HASHED_TOOLS` (B, low); the criteria's declared settings moved,
  unchanged, to `config/bowen/criteria.md`, and `M11.D.2`'s magic-literal scan covers `criteria.py` (4, low). All
  three records regenerated; apart from the hash they are byte-identical. The suite then held 483 tests.
- From the third gate (`docs/cycles/2026-10-08_phase-c-qa-gate-3.md`, APPROVED, three low findings): a static test
  ties the criteria's `REQUIRED` settings to what each arm reads, in both directions; the `criteria.py` docstring
  no longer restates the spell's numbers; the mutation tool loads once in the tests. The suite then held 485 tests.
- From the fourth and fifth gates: a stale suite count in this file fixed, and the tests load both tools under their
  real module names (`tools.mutation_record`, `tools.sweep_record`).

### Decided

- **`TRIANGLE`'s roles (`M1.C.1`), 2026-10-08.** The target is the third party the act recruits and stands outside;
  the inside pair is the sender and its most strained partner. The spec already read it this way in `M1.F.1b`,
  `M11.C.3`, `M11.C.42` and `M6.4`; step 5 had built an alliance reading instead. A rule change after the freeze,
  made from the spec's text and reported (`docs/phase_c_completion_report.md` §8). `M11.C.3` now passes and is
  proved, including by a mutant that restores the old reading. `M11.C.42`, `.45`, one cell of `.27` and one of `.41`
  stopped passing; they are reported, not recovered. 7 criteria now pass and are proved, 7 fail.
- **What `M11.C.42` and `.45` test, 2026-10-08**, decided from the spec's text and committed before rerunning.
  C.42's baseline arm now makes the third member unavailable for one tick, instead of scripting `STAY-IN-CONTACT`,
  and its readout counts the whole pair's triangles. C.45's readout is `TRIANGLE` selections per person-week.
  Both still fail: C.42 is +0.26 (p 0.084), C.45 +0.0020 (p 0.20). Report §9.
- From the eighth gate (APPROVED, four test-strength findings): the absence and readout tests for `M11.C.42` and
  `.45` now compare every record before the absence, check that no triangle can form during it, and check the
  pair-wide count. Tests only; no record changes.
- **`M11.C.1` and `.38` stay premise, 2026-10-08.** No new class: every level-reading rule states the direction, so a
  pass is not a finding. The spec's `M11.5` rows now list the nine rules and name the joint level-blind mutant as the
  proof; plan §3's single-rule mutant cannot prove them.
- **The ensemble precision rule, 2026-10-08.** The stopping rule compares the half-width with 0.25 of the pooled sd of
  the two arms, not the baseline arm's (`ensemble_precision`'s unit changed, logged post hoc). Decided and committed
  before rerunning. `M11.C.16` now passes, but survives its named learner mutants, so its direction does not come
  from learning; `M11.C.41`'s [light: lower level] cell now fails instead of undetermined. Report §10.
- **`M11.C.16` stays composite, 2026-10-08.** The first check ran an availability inversion that saturated at the layer
  clamp over C.16's levels, so it removed availability instead of inverting it (csdp sweep). It was replaced, and all
  nine level-reading rules now run as deletion and inversion mutants on C.16. Inverting availability now cancels C.16
  (both readouts' intervals span zero) but does not reverse it; removing it halves the result. That inversion equals
  the deletion at level 20 and below, where the lowered arm spends 40.6% of member-weeks under it over the run and
  45.2% over the entropy's window (`docs/phase_c_level_occupancy.md`, a new generated record), so the cancellation
  holds for it alone. Removing the rules C.16's criterion row cites, `M4.C.1a` and `M1.C.3a`, leaves it passing
  (`c16-cited-grounds-removed`); removing them with `M4.D.3a`'s availability turns it red (`c16-grounds-removed`). Composite rests on
  reading only availability as naming a narrower repertoire; report §11 says why that reading is contestable. A reclassification to
  premise, which read that cancellation as `M11.1d`'s flip, was withdrawn the same day by owner decision. `M5.D.3`'s hold
  capacity never acts in C.16's runs. The learner changes the result's size, not its sign, so the row's rationale and
  mutation clause are not met, and coverage keeps it partial. The mutation record now gives each mutant's mean
  difference and half-width, marks a red mutant **reversed** only when every gating readout's interval lies on
  the opposite side (report-only and equivalence readouts do not gate), and marks a mutant whose replacement
  text is broken (an unimported name) **broken**, which coverage does not count as proof; an engine exception under
  a mutant stays a red. The sweep record says **opposite sign** for a sign-only
  reversal. Report §11.
- From the tenth gate (APPROVED, four low findings): `M11.5`'s premise definition now admits a result several rules
  state redundantly; the `M11.C.38` note says no single-rule mutant reds all three pairs (four of six red none); an
  "exactly zero" claim no committed record holds was removed; this file's `.45` entry corrected.

## [Unreleased] — 2026-10-06

Spec revision 10 approved, the Phase B plan approved, and **Phase B built** (`src/bowen/`).

### Added

- **Phase B of the agent model** — objects and identifiers, keyed draws, the event record and queue, the in-run event store, log records, the reduced family and the 40-week script from markdown, the M8 predicate, activation and visibility, the standing load, the base appraisal (`M4.C.1` only), structural events, consolidation, the M6 invariants (`M6.I.6` held back), the nine-step weekly loop, the persistence sink, the log header, the deterministic renderer, and `python3 -m src.bowen.run`. 291 tests after the `learning-qa` review's fixes; every automated Phase B exit criterion passes and is mutation-proved (`docs/phase_b_mutation_record.md`). Report: `docs/phase_b_completion_report.md`; coverage: `docs/spec_coverage.md`.
- `tools/mutation_gate.py` and `tools/spec_coverage.py`.
- Constants frozen 2026-10-06 (`config/bowen/constants_frozen.md`).

### Changed

- **Spec revision 12 approved, 2026-10-07** — the Phase C prerequisites, with the Phase C plan (`docs/implementation_plan_phase_c.md`, all recommendations as written). `M6.I.6` restated with a stock-and-flow table (`M6.4`); `M4.B.2` scoped row by row (`M4.B.2a`) and `M9.8`'s belief store moved into Phase C; `M11.C.17`, `.18`, `.20` moved to Phase D and `M7.D.1` into Phase C; `M5.D.6`'s next day is the next tick, with `M5.D.9` for multi-tick sequences; `M11.D.16`'s equivariance in distribution; `PROVOKE` added (`M5.B.6`, and to `config/bowen/event_kinds.md`); the review's Phase C contradictions resolved. Explainer §17.3 and six drifted passages corrected. 542 requirements, 45 criteria; Phase C gates 19.
- **Spec revision 11 approved, 2026-10-06** — set, random and emergent layers (`docs/proposal_rev11_emergence.md`; `model_explainer.md` §19). `M4.D.6a` reversed: the automatic channel learns from felt relief. `M4.C.1` re-derived as two-sided deviation from a felt contact optimum, with the standing load as its "too little" side. Core moves are single acts and Bowen's patterns are readouts (`M5.A.1a`). Rules that wrote an outcome made emergent; `M7.D.3` and `M7.E.3` withdrawn as mechanisms. `M4.C.10` (calmer contact lowers anxiety), `M4.B.3` (no look-ahead), four new criteria (`M11.C.42`–`M11.C.45`), and `M11.5`'s premise/composite/check classification. 538 requirements, 45 criteria. A `learning-qa` review's eleven findings were fixed the same day: stale references to the old `M4.D.6a`, `M4.D.3a` brought in line with amended `M4.D.3`, mutation targets moved to `M4.C.1a`, the cutoff cost made to build over time so `M11.C.4` holds, `tools/spec_coverage.py` gaining `exclude_evidence` and two guards (293 tests). Phase B code built against revision 10's `M4.C.1`, `M4.A.1` and `M1.C.3` no longer conforms; reworking it is a Phase C item.
- **Spec v2.0 revision 10 approved by the owner, 2026-10-06** — 519 requirements over 17 modules, 41 acceptance criteria. The terms are recorded in the spec's revision-10 section under *Approval*: the external review of 2026-09-23 is carried into the implementation plan rather than fixed first (its recommendation 9), `M11.4c` and `M4.G.3` kept, `M11.E`'s proposals accepted, revision-11 material held.
- **Phase B plan approved, 2026-10-06** (`docs/implementation_plan_phase_b.md`), all nine recommendations as written. The spec now holds 525 requirements: `M1.A.21` (`sex`), `M1.A.22` (`household_id`), `M2.3a` (the Phase B instance), `M3.C.2` (latency in whole ticks), `M4.G.2a` (`M6.I.6` held back until restated), `M14.A.4` and the filled `M14.A` register. `M4.C.1` moves into Phase B. Ravi's `basic_level` is 39, within `M2.A.0e`'s tolerance of Marta's 40.
- Phase B step 5: `M2.A` gains a `Sex` column and `M2.A.0i` (526 requirements). Ravi is male and Nadia and Pia female by owner decision; the rest follow the table's relationship words; Leo and Dr Halim are not stated until Phase D.
- `test_m11d17_register_has_no_orphans` and three mutation checks added to `tests/test_spec_consistency.py`; the suite-size guard now counts tests in subdirectories.
- `M11.F.10` added: the model may not be called a "virtual family" without `M11.F`'s framing. `model_explainer.md` §1.1 gives the comparison with virtual-cell models and what a virtual family would need.

## [Unreleased] — 2026-08-28

The sixth corpus source decided and applied, plus two modules that came out of working through what the model may and may not be asked. Documentation only; no code. Spec revisions 4, 5 and 6.

### Fixed

**From a failure-pattern sweep of the batch, 2026-08-28 — fifteen findings, all verified against the files and all fixed.** Four were high:

- **`M16.B.3` required logging to be a pure observer, contradicting `M16`'s own preamble twelve lines above**, which makes an agent's delayed log an *input* to a mechanism. Worse, its test gated **only Phase B** — the phase with no policy and no external agent, where it passes vacuously — and was not re-run in Phase C where `M16.D` makes it false. Split into an **event store** (always present, causally load-bearing) and a **persistence sink** (optional, pure); `M16.T.3` re-listed in Phases C and D, and `M16.T.6` added as its converse.
- **`M11.2` bounded ensemble testing at `M11.C.16`** while the table grew to 33, leaving seventeen criteria — including three distributional claims a single run cannot establish — outside a bound nobody had moved. Restated over the whole set.
- **`M11.3` shipped with eight counterexamples in the same commit.** It required every criterion to name a stressor; 8 of the 9 criteria added alongside it name none, and `M11.C.31` *requires a calm family by construction*. Scoped to discriminating criteria, with the conservation, identifiability and null classes named as outside it.
- **Seven `M11.C` criteria and `M16.T.4` carried a phase and gated nothing**, so Phases C and D could each be declared done with them unwritten — among them `M11.C.17`, which the spec calls "the model's strongest negative test", both estimator discriminators, and the only guard stopping an agent reading another agent's private hops. All eight listed; `M13.2a` and `M11.D.11` make the correspondence mechanical.

And eleven more: `M10.C.4`'s twelve magnitudes had **no permitted home** — forbidden in config, underivable, and `M11.D.2` requires one or the other, so the violating route was also the compliant-looking one (`M10.C.4b` names a checks module; `M11.D.12` guards it). **Nine prohibitions were written `No X MUST Y`**, which under §0.3 states a permission — including `M11.F.9(c)`, the clause the document marks as correctness rather than framing (`M11.D.13` now guards the form). §0.3 claimed every MUST carries a test: 616 MUSTs against 44 tests, restated honestly. `M0.4` pointed at an exception list that did not exist (`M11.4` now is one). `M11.F.5` never existed and the gap read as a withdrawal. `M11.C.25` and `M11.C.17` are nulls that pass when underpowered (`M11.4a`). `M16.C.3`'s rationale was stale inside its own commit. `M1.F.9` never named `binder_unavailable`. And the published proposal carried revision 4's requirement count through revisions 5 and 6 while `_STATUS.md` asserted the set was consistent.

- **`M11.F.9(c)`'s stated mechanism was wrong, though its conclusion stands.** It said fitting absorbs the model's error "on one side only"; under `M0.4` both arms share one parameter set, so nothing is absorbed asymmetrically. The real failure is **loss of independence, non-exchangeability across a regime boundary, and non-identifiability** — 60–90 free constants against a single realisation, so many parameter sets reproduce the history and disagree about the counterfactual, making the bias in the difference *unbounded* rather than merely unknown. A stronger objection than the one it replaced. The line "the better the fit, the less the counterfactual can be trusted" is now scoped to the overfitting regime, where it holds, rather than stated as a general law, where it does not.
- **`M5.D.4` was inverted, and had been since it was written.** The requirement said *anger* is the gate admitting the `I-POSITION` sequence to its peak. Ch13 says the opposite — "when he is finally able to maintain his course **without getting angry** at the opposition, the opposition does a final intense emotional attack" — and mentions anger exactly once in the whole chapter. `docs/theory/ch13.md` had it right in the body and wrong in the **heading** above it ("Anger gates the escalation"), and the heading is what reached the spec. Corrected in all three places, with `M11.C.32` added to hold it. Kerr 1988 (`FE04.4`) and Bowen 1988 (`FE11.19`) both agree with the chapter, about the same person's anger.
- **A phase exit condition that could not be checked.** Phase B was done when "the event log reads correctly" — an adjective naming no requirement, no test and no artifact, in a table where every other condition names one. It now names `M16.C.2` and `M16.T.1`.
- **Module ordering.** `M14` had ended up after `M15`.

### Fixed — the corpus-fidelity sweep

**Every quoted passage in the spec was checked against the primary sources** — 214 quotations, 160 verbatim in the corpus, 52 legitimate self-quotes of project documents, 0 unexplained. Five cold agents on a different model, reading the source texts and forbidden the extractions, plus a deterministic string match by the author.

- **One fabricated quotation.** `M1.A.4e` placed a paraphrase inside quotation marks and attributed it to the source as its own opening line; it was the evidence for revision 8's demotion. Fixed, and the demotion restated to what the source carries.
- **Modal inflation, the dominant class.** `can… or` → `MUST` with one branch erased; `suggested` → `required`; `perhaps having a role` → `MUST be`; `such as 55 or 60` → a bound; and a **rate** claim hedged with *"sort of"* turned into a **ceiling**.
- **Three scope shifts, introduced at revisions 7 and 8** — a subgroup phrase applied to a whole band, a societal trend grounding an individual mechanism, and a consequence presented as a quotation's when the corpus arguably says the opposite.
- **Evidence framing.** "An experiment and an independent replication" were two clinical accounts, one of them the author confirming what he had been told. "Two families — home movies in both" was one family. "Three statements, every one about the mover's anger" was two.
- **Attribution.** An ASR transcript attributed to Bowen personally against the project's own rule; a requirement grounded on **Andrew Solomon's** words, quoted by Kerr.
- **Propagated** to `model_explainer.md` and the published proposal, both of which carried the same band-scope over-read graded `[T]`.

Hedge-level findings are in `TODO.md` with reasons rather than fixed.

### Changed — what the estimator may read

Three points from the project owner, each checked against the corpus and each upheld.

- **The four "assessment domains" were genogram intake fields, promoted to an instrument.** `KS11.7`'s own opening line says they are what *"the family diagram records"*. Occupation, health history, longevity, courtship length — these are **outcomes**, they are **base-rate common** (a divorce or a job loss carries almost no information about who had it), and they confound the **load on a person** with their functioning. Demoted to a weak prior at import; they may no longer set an estimate. `M1.A.4e`
- **The corpus already corrected this and the spec had not carried it through.** `KS06.8`: *"In terms of solid self, one's personal life is where the rubber meets the road"*, and it is the intimate relationship *"and not how they manage themselves in their work lives"* that determines offspring level. The requirement cited that correction **underneath the table it invalidates**.
- **A three-tier reading replaces the flat list** (`M1.A.4g`): a single adverse event is mostly **load**; a chronic self-maintained pattern reads **functional** level, load-corrected; only a position held under pressure reads **basic** level.
- **The discriminator is which channel was selected under discomfort, at comparable load** (`M1.A.4h`). `M4.D.1a` already carries it — the automatic channel discharges anxiety *now*, the self-directed one holds a position *through* discomfort, and the mixing weight is a function of differentiation. Reaching for an affair, a substance or distance is the first; carrying it is the second. Ch13 states the affair case as mechanism. **Adhering to convictions under pressure and carrying discomfort are one observable, not two** — `M5.F.5` says so in a sentence — and every piece is already in the model.
- **The system's reaction to a symptom is now an estimator input, and it is the best-controlled one** (`M1.A.4i`, `M11.C.34`). One symptom, N observers, one moment: load is held constant **by construction**, so the comparison across members needs no correction. It is also the only strong reading on members who are *not* symptomatic.
- **`M1.E.2`'s three reaction detectors are generalised off the coach to every person** (`M1.A.19`). Ch10 extends the neutral third to "any third person", and the proposal's §10 already recorded the conclusion — *"one rule seen from four angles… Implement it once"*. One of the four had been implemented.
- **The circularity boundary is stated precisely rather than as a ban** (`M1.A.4j`): symptom **magnitude** may not feed the estimator, because the model generates it from level; the **channel selected under discomfort** must, because it reads policy rather than accumulated load. An earlier draft of this had banned symptoms outright, which would have discarded the best signal in the model.

### Fixed — the scale is a continuum

- **`M1.A.3` implemented a cliff where the corpus describes a band.** It required *"exactly one behavioural transition on `basic_level`, at 50"*. Raised by the project owner — Bowen's quadrants are bands across a continuum, not shift points — and upheld against the primary: Ch16's own wording is graded ("**begins** making **a few** decisions of its own"), Ch21 has **no regime change at 50 at all** and calls the transition **explicitly gradual**, the band scheme is present/absent/present across 1966/1972/1976, Kerr gives different edges entirely, and Bowen "subsequently dropped the term scale". The licence is now a **slope**: continuous in level, with the joint-decision domain lagging every other domain at every point. That ordering is the mechanism; 50 is a label.
- **Removing the step must not flatten the model** (`M1.A.3c`) — the corpus asserts **non-monotone** structure over the range, and a smooth ramp would erase it.
- **A band edge may no longer be a branch point anywhere** (`M1.A.3d`) — 25, 50, 60, 70, 75 — so the error cannot recur at the other four numbers. The capacity floor at 25 is regraded to a continuous falloff (`M1.A.4d`), applied **on the principle rather than on a fresh reading**, with its location left standing pending one.
- **"Regime boundary" corrected to "turning point"** throughout the sign-flip argument. The conclusion is unchanged — mis-estimating which side of a turning point you are on still reverses the sign — but the picture is a roll-off, not a cliff, and near the turn the effect is small and noisy.

**This is the second instance of one failure**, after `M5.D.4` at revision 4: a source *description* firmed up into a mechanism the source does not assert. Both were found by reading the primary rather than the extraction, and **neither was catchable by a regex** — a sweep for others is a reading task, logged in `TODO.md`.

### Fixed — second pass

**The fix commit was swept as its own range and returned fifteen more findings**, four high. That is the
expected result, not a surprise: a fix commit is the least-reviewed code in a change, and the project's
catalogue says a fix pass introduces defects at roughly the rate it removes them. Four of the fifteen were
**the first sweep's own diagnosed failure mode recurring inside its fix**:

- **The phase-gating guard went vacuum-green on a cosmetic reformat.** It accepted a phase cell only if it
  was exactly `B`/`C`/`D`/`E`; bolding all 32 remaining cells made it check **zero** criteria and pass. One
  row was already being dropped silently. It now asserts it read every gateable row.
- **The inverted-prohibition scan was narrower than the class it named** — capital `No` plus a separately
  bolded `**MUST**` — so four instances survived, including **`M11.D.12`, the guard written for another
  finding, in the form this finding forbids.**
- **Two corrections landed in the spec only.** `model_explainer.md` still said logging is a pure observer,
  which corrected `M16.B.3` contradicts, and `_STATUS.md` still carried the repudiated "on one side only"
  mechanism — while asserting the document set was consistent.
- **`M11.4`'s "exhaustive" exception list was contradicted three times in the same document**, once by a
  table fifteen lines below it. Same shape as the `M11.3` defect it was written alongside.

Also: `M16.T.6` required disabling what `M16.B.3` forbids (now an experimental override, the pattern
`M10.C.4b` already uses); all six `M16.T` criteria named tests that do not embed their ID, against §0.4;
`M10.C.4b`'s third home cleared `M0.3` and `M10.1` but not `M10.B.1` or `M11.D.2`, which are now scoped to
engine-read constants; `M11.3`'s "discriminating" was undefined and named a table column that does not
exist; §0.3's coverage figures were measured at the parent commit and stale on arrival (now asserted by
`M11.D.14`); `M11.C.8`'s Test cell is `—`, a counterexample to §0.3's replacement claim; and
`M11.F.9(c)` said the bias was "**unbounded**", which does not follow — the difference is *unidentified*,
and an unsupported strengthening is precisely the error §10 records the first corpus pass making.

Four low findings were graded and **backlogged rather than fixed**, with reasons, in `TODO.md`.

### Added
- **`tests/test_spec_consistency.py`** — ten guards over the document set, each mutation-proved by reverting the fix it protects and confirming red. They assert what a failure-pattern sweep had to find by hand: IDs unique and machine-extractable, every cross-reference resolving in seven documents, no prohibition written in the inverted form §0.3 reads as a permission, every criterion gating the phase its own table row names, `M11.2` stated over the whole criteria set rather than a numeric range, and the requirement and suite counts in `README`/`CLAUDE.md`/`_STATUS.md`/the proposal reconciling to the artifacts they describe. They read documents only — nothing imports `src/`.
- **`M15` — the family-diagram import contract.** A Phase E capability specified early, because the diagram application that would feed it is under the project owner's control and its export format is cheaper to fix before it exists than after. Structure imports as **values**; ratings import as **ranges**; the readout is an **envelope**, never a point direction. Four adapter traps named, including a warmth rating wired to a valence-blind `investment` (which would invert the sign on exactly the conflicted families the model is for) and a genogram's single "distant" line, which covers both a rupture and a resolved low-contact tie — the discrimination `M1.B.3` calls the most consequential in this part of the theory. Fixture mode included: anonymised real topologies beat hand-built ones wherever a criterion depends on asymmetry.
- **`M16` — the run log and the readable trace.** Six requirements already depended on a log and none of them said what one is. Effects and the selection rationale are recorded beside the events, so the log is a causal trace rather than a list; belief writes are tagged apart from ground truth; the engine emits and the caller persists; logging is a pure observer. A **deterministic renderer** — template, no language model — is built in Phase B, and a per-agent **delayed view** serves the fourth form of a landed coaching contact, defaulting to the corpus's six months.
- **`M11.F.9`** — three clauses on speaking about a real family. The third is a *correctness* requirement, not a framing one: no counterfactual may be reported from parameters tuned to reproduce a known history, because the parameters are then chosen conditional on the trajectory and the fit is non-identifiable, so the counterfactual difference is not identified by the data. *(This entry originally carried the mechanism as "on one side only", which the re-sweep showed to be wrong; see Fixed.)*
- **~20 requirements from *Family Evaluation***, the largest being that the belief layer becomes a **channel** (`M9.6`, `M9.7`) rather than a parallel store — chronic anxiety runs on what *might be*, and Bowen's own correction of "projection" has the transfer running through descriptions.
- **`M10.C.4`** — twelve magnitudes stated in the corpus, admitted as **checks on an output** and never as parameters, with `M10.C.3a` clarifying that the existing prohibition is on the parameter and not on the quantity.
- **`M11.G`** — the eight-component family evaluation readout, with the source's ninth and tenth components (therapeutic focus, prognosis) explicitly barred.
- **Nine acceptance criteria**, `M11.C.25`–`M11.C.33`, and five for the log, `M16.T.1`–`M16.T.5`.
- **`docs/model_explainer.md` §17** — what the model can be asked, and the two ways the answer goes wrong. Marked as method rather than corpus.

### Changed
- **`docs/DECISIONS — FAMILY EVALUATION.md` closed.** All nine items decided 2026-08-27; the file is now the record of how each was decided.
- **`M1.A.11c`** — constitution sets the symptom channel as a *prior*; a family-focus term can shift it; a constitutional-strength term can override the shift. `model_explainer.md` §3.9 is no longer provisional.
- **`M1.A.3b`** — the transition at 50 keeps its licence implementation and gains an awareness *readout*. Neither is reduced cognitive capacity: what differs is the strength of the emotional circuits over the cognitive ones, and a low-level agent argues just as fluently.
- **`M7.A.1a`** — the financial-dependence gate now binds the slow clock, not only the `I-POSITION` move.
- **`M12.2` reviewed and left standing in full.** Kerr 1988 names the *locus* of the first unknown, not the rule; its own hedge is "determined largely by".
- **`M12.5`** — power and punishment barred as mechanisms. The only entry in that module stated as a prohibition by the author rather than inferred by the project.
- Spec is **427 requirements over 16 modules**, 34 acceptance criteria, 0 duplicate IDs, 0 dangling cross-references. Every `M…` reference in seven documents resolves, and all of that is now asserted by `tests/test_spec_consistency.py` rather than by a script run once.

### Learned
- **A summary heading is a lossy artifact, and a requirement sourced from one can invert without anything looking wrong.** The `M5.D.4` inversion above is the worked instance: the quoted sentence was right, the heading over it was not, and only the heading travelled.
- **Initial conditions are not nuisance parameters.** Invented constants cancel between two arms of a counterfactual because they do not interact with the intervention. Initial conditions are *what the intervention acts on*, so their error crosses a regime boundary and flips the sign instead of shifting both arms together. Five such boundaries are named at `M15.D.4`; one of them, `M5.C.1a`, is a SAFETY property with recorded harms. **Direction is the least robust output under mis-specified inputs, not the most.**

## 2026-08-22 — corpus alignment

The corpus-alignment batch. All 22 chapters of *Family Therapy in Clinical Practice* have now been read twice and the project's documents have been brought into line with what the source actually says.

### Added
- **`docs/model_explainer.md`** — every object, field, move, gate, invariant, clock and test in the agent model: what it is for, what it does, and which chapter it implements. 216 claims, each graded `[T]` textual / `[M]` mechanism / `[D]` direction / `[#]` stated quantity / `[I]` invented / `[X]` contested.
- **`docs/bowen_agent_model_spec_v2.md`** — the v2 specification, v2.0-draft, **awaiting approval**. 212 requirement definitions over 14 modules; 16 acceptance criteria and 10 engineering criteria, each with a named test and a mutation target; four criteria flagged as not code-testable in Phases B–D with a human-review proposal each.
- **`docs/theory/_RESOLUTIONS.md`** — the two contradictions carried out of pass 2, both resolved against the primary source.
- **`docs/theory/`** brought under version control: 22 chapter extractions, the ledger, the convergences, the corrections and the status file.
- Ledger entry **L21.11**, surprise as a third secrecy mechanism — present in the source and in neither pass.
- Spec requirement **M11.D.10**, requiring every ID to be machine-extractable.

### Changed
- **`docs/theory/_LEDGER.md`** rewritten and made self-contained: all 109 pass-1 IDs preserved and re-verdicted, 40 new pass-2 entries, all ten correction batches folded in. Withdrawn findings kept as tombstones with their reason.
- **`docs/agent_model_proposal.html`** corrected throughout rather than annotated. The undifferentiation budget has three sinks, not four. Reactivity is derived, not stored. The move repertoire is no longer claimed closed. The event record carries source position, route, fidelity and latency. Acceptance tests: eleven became fifteen. **Phase B's four-person nuclear spike is withdrawn** — cutoff with the families of origin is an *input* to intensity, so a model closed at the household cannot compute its own driving term.
- **`docs/bowen_individual_family_model_spec.md`** frozen as a v1.2 historical record of the grid engine, with a header table of the nine things the corpus contradicts in it. It is not the base for v2.
- **`CLAUDE.md`** rewritten for the pivot. Axiom 1's vectorisation mandate is scoped to the frozen grid engine — at N ≈ 12 it pushes the design the wrong way.
- **`README.md`** expanded from two lines.

### Fixed
- Two flaky tests, `test_spouse_dysfunction_asymmetric_penalty` and `test_triangle_mechanism_attaches_and_releases_circle`. Each constructed an unseeded `Simulator` and asserted on a family whose randomly drawn size did not satisfy the code path's precondition; the first was measured failing 6 times in 12 runs, with the asymmetry it names never exercised on a failing draw. Verified 36 passed across 12 consecutive full-suite runs; the suite is 37 after the working-directory guard below.
- `.gitignore`, whose last line read `.DS_Store (for Mac users)` — a literal filename, so `.DS_Store` was never ignored.
- Spec-internal ID drift: invariants were defined as `I1` and referenced as `M6.I1`, criteria as `C4` and referenced as `M11.C4`. A traceability scan would have resolved 19 of 216 references to nothing and reported a lower count rather than an error.

### Withdrawn
Five mechanisms proposed on the first reading and removed on the second, recorded rather than deleted because a merely-absent mechanism gets re-invented: the permission scalar, ally cancellation, the durable projection target queue, distance as a depletable relief stock, and "content is noise" as an argument for anything.
