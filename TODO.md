# TODO

Deferred and adjacent items. Each carries enough context to act on later without the conversation that produced it.

## Blocking the next phase

- [x] ~~**Approve or revise `docs/bowen_agent_model_spec_v2.md`.**~~ *Approved 2026-08-25; revisions 1–6 applied, the last on 2026-08-28.*
- [x] ~~**Approve spec revision 10.**~~ *Approved 2026-10-06. Terms in the spec's revision-10 section under* Approval.
- [x] ~~**Read the Phase B trace end to end and record the result** (plan §7)~~ *done 2026-10-07: the owner read both
  traces, no findings; Phase B closed (`docs/phase_b_completion_report.md`). This entry was stale until 2026-10-09.* — the last item before Phase B is declared done. `docs/review/phase_b_trace_seed7.md` and `docs/review/phase_b_trace_seed7_nadia.md`; report at `docs/phase_b_completion_report.md`.
- [x] ~~**Run the `learning-qa` review over the Phase B diff before pushing**~~ *superseded: Phase B was pushed and
  closed 2026-10-07, and every Phase C batch since has had its own sweep.*
- [x] ~~**Build Phase B.**~~ *Steps 0–15 done 2026-10-06; all 14 automated exit criteria pass, 17 of 17 mutations proved.*
- [x] ~~**Approve the Phase B implementation plan**~~ *Approved 2026-10-06, D1–D9 as written.* — drafted 2026-10-06 at `docs/implementation_plan_phase_b.md` (branch `plan-phase-b`). Its §1 holds nine decisions (D1–D9) the owner must answer before code, and Appendix A is the `M14.A` register draft for review. *Was: write the implementation plan — Phase B only* (owner decision 2026-10-06, following the external review's recommendation 9). The 2026-09-23 review's findings go in the plan as blockers on the phases they affect. The next gate, and the only thing between here and code. Every acceptance criterion mapped to the file and module that satisfies it, in dependency order, with the mutation that proves each one. Spec convention is spec → plan → build, each approved before the next. Inputs: the 33 criteria in `M11.C`, the 5 in `M16.F`, the engineering criteria in `M11.D`, and the phase table at `M13`.
- [x] ~~**Decide four criteria that cannot be made code-testable in Phases B–D** (spec §M11.E).~~ *Accepted as written by the owner, 2026-10-06, at approval of revision 10.* Each has a human-review proposal:
  - `M11.C.8` (endogenous incidence vs published rates) — needs Phase E and an editorial call on which sources are genuinely exogenous.
  - `M11.C.9` (sibling position) — the effect size is invented, so the test can only assert "a detectable difference", which is close to unfalsifiable. Decide whether it belongs in the suite or should be demoted to a readout.
  - `M11.C.11` shape — "three phases, not a step down" is a claim about curve shape; any automated version embeds an invented tolerance.
  - `M5.F.2` — the threshold separating a counterfeit move from a genuine one is invented and sets the result. **This is the model's most consequential invented constant.**

## Spec revision 30 (origin cutoff, danger, focus lock-in) — open after the build, 2026-10-14

Plan §4.10 holds the evidence.
- [x] ~~**Owner: felt danger fades once a cutoff works** (correctness review).~~ *Decided 2026-10-14, "hold felt danger while rejection is held"; built (`M1.B.3e`, `test_m1b3e_felt_danger_is_held_while_rejection_is_held`).* Refused moves are never delivered, so the
  intrusion a person receives from the one they cut off decays over about `danger_window` weeks, felt danger falls to
  0, and the cutoff turns reactive by itself: the full reactive cost and the full "too little" term return though
  nothing about the other has changed. Should felt danger be held while rejection is held, or decay more slowly?
- [x] ~~**Owner: focus compounding has no ceiling on investment** (learning-qa, medium).~~ *Decided 2026-10-14: "saturate the investment", then, after a ceiling removed the lock-in, "Shift, gain 10": the term now moves attention among the children and adds none (plan §4.10).* `M1.B.8b`'s term grows a
  parent's investment in the child they focus on in proportion to that investment, so it grows exponentially while
  pressure × share is above 5 (leak 0.1 a week). Measured (learning-qa, seed 0, ad hoc): the largest parent-to-child
  investment over 10 years on the four fixture was 3,477 at gain 2, against 27.6 at gain 0. Every reader today uses
  shares, so no result changes; a later reader of absolute investment would. An only child has share 1 and compounds
  most. A saturating form needs a new constant; not changed unasked.
- [ ] **`M11.C.10`'s input barely differs between its arms** (correctness review, 3 seeds, 12 years, ad hoc). The
  origin cutoff of the first gen-3 child was 0.79 (open) against 0.87 (held), and 0.80 against 0.92 in the
  family-of-origin variant: parents already hold some rejection on some tie most weeks, and the held cutoff adds 0.08
  to 0.12. This limits what 200 seeds can detect.
- [ ] **`M11.C.17` passes by a fragile margin, not by one mechanism** (ad hoc, 2026-10-14, the revision 29 worktree):
  switching off any one of the four revision 29 gains leaves the coached rise at about 0; switching off all four
  reproduces revision 28's +0.000881 ± 0.000828.
- [ ] **Minor, from the fix re-sweep (2026-10-14):** `update_danger`'s docstring puts the dead-person sentence in the
  middle of the paragraph on cost (move it to the end; left during the regeneration so the engine hash would not move);
  `arm_c47` reads only people with a tie to the person, so a removed tie would drop someone from both readouts
  silently (not possible on the fixture today).
- [x] ~~**CI has been red on every push since 2026-10-09.**~~ *Fixed in `bfd46cc`; CI run 38763412772 green, 850 passed on 3.11, 3.12 and 3.13 (2026-10-15).* Cause: the last bits of floats
  differ between platforms and Python versions. Python 3.12 changed how `sum` adds floats, and the C maths library
  differs between macOS and Linux. NumPy 2.5.4 against 1.26.4 changed nothing here. The golden fingerprints now hash
  floats rounded to 8 significant digits (measured margin, in the test's docstring). The diagnostic record's mean uses
  `math.fsum`; its 0.938 was on a rounding boundary and printed 0.937 on 3.11.
- [ ] **Adjacent, not fixed (2026-10-15):** `grep -rn "sum(.*) / len(" tools src/bowen | grep -v fsum` finds 29
  lines that take a mean as `sum(...) / len(...)`. Some are integer or bool counts, which are exact; the review counted
  about 18 over floats on one line each, plus 3 in mutant code in `tools/mutation_record.py`. The ones in the engine
  (`reactive.py`, `appraise.py`, `iposition.py`, `slow_tick.py`, `life_course.py`, `beliefs.py`, `learner.py`) are the
  likely source of the 3.11 against 3.12 drift in the run itself, not only in printing; in tools they can print
  differently on a rounding boundary. None fails today: no other test failed on 3.11 in CI. Moving the engine to
  `math.fsum` changes the engine hash and reruns the records, so batch it with the next engine change.
- [ ] **The phase D2 record prints `M11.C.10`'s two variants under one id** (and `M11.C.2`'s twelve-person variant as
  `M11.C.2`). The renderer uses `Criterion.id`; it should print the criterion's key. A label change: render from the
  cache, not by rerunning.
- [ ] **`M11.C.47`: the dangerous person is cut off on fewer weeks** (−16.6 ± 4.9 `held_weeks`, 150 seeds), though felt
  danger and the principled part are higher. Trace why before reading the premise as a finding about cutoff.
- [ ] **`M11.C.2` on the twelve-person family fails again at revision 30** (+0.70 ± 1.5, p 0.16; +1.31 ± 0.95 at
  revision 29). *Traced 2026-10-15, each revision 30 change removed alone:* unmodified +0.70 ± 1.5, p 0.16 (reproduces the record); `M4.D.1h`'s force removed +0.70 ± 0.97, p 0.13; `M1.B.8b`'s compounding removed +1.12 ± 1.7, p 0.09; `M1.B.8b`'s reinforcement removed +1.11 ± 1.6, p 0.093; `M1.D.8a` removed +0.96 ± 1.5, p 0.14; `M1.B.3e`'s danger removed +0.08 ± 1.8, p 0.46 (`symptom_load`, seeds 0–149, one-sided signed-rank with the runner's margin; ad hoc, 2026-10-15). No single removal brings
  the pass back. Removing the force gives revision 29's spread with revision 30's mean (the two means agree to three
  figures with different spreads; measured, not explained); removing either focus term gives back most of the mean
  with the wider spread. Next, if wanted: force and both focus terms removed together, about 10 minutes. Under the
  version 1 definition this composite can also be accepted as a finding.
- [x] ~~**Run the `M11.C.10[family of origin]` mutation**~~ *Run 2026-10-15:* `M1.D.8a` removed does not turn it
  red, +240 ± 116 tie-weeks (`child_cut_weeks` +49.7 ± 44.1; 200 seeds, ad hoc, 2026-10-15), against +162 ± 120 with it.
- [x] ~~**Find what carries `M11.C.10[family of origin]`'s pass**~~ *Run 2026-10-15:* `M4.D.6f` removed (`witnessed_cutoff_gain` 0) gives −75.9 ± 132 tie-weeks; `M1.B.3d`'s reactive rise removed (`reactive_cutoff_gain` 0) gives +163 ± 117; `M1.B.3e`'s danger removed (felt danger from the declared floor only, 0 in this family) gives +102 ± 126 (`child_cut_weeks` +48.4 ± 45.3, +37.3 ± 46.7, +18.4 ± 42.3; 200 seeds each, ad hoc, 2026-10-15). Learning
  from a witnessed cutoff carries it. The variant's pass is still not support for `M1.D.8a`.
- [x] ~~**Decide what `M11.C.10[family of origin]`'s mutation clause names.**~~ *Decided 2026-10-15:* the owner restated
  it to `M4.D.6f`; its mutant is red in `docs/phase_d_mutation_record.md` (FAIL, p 0.91, 200 seeds).
- [ ] **Phase D mutation record, three minor items** (re-review of 7c8a0fb, 2026-10-15): (1) `engine_hash` reads
  `inspect.getsource` of `apply` and `run_in_copy`, comments included, so a docstring edit there reruns the record;
  (2) `build()` does not flag an `AD_HOC` entry that now has a mutant or no longer passes (only the test checks the
  first); (3) the spec's `M11.C.10` row still opens with the revision 11 and revision 29 mutation clauses, superseded by
  the later amendments but not struck (item 3 is in revision 31's plan, approved in the review 2026-10-16).
- [x] ~~**Owner: review spec revisions 24–30**~~ *Done 2026-10-16:* all 85 entries decided in `docs/REVIEW_spec_revisions_24-30.md`, 81 approved as built, 4 changed (`M14.A` rows, `M11.C.14` option B, `M11.C.6`
  option A, and `M7.E.1f`'s band: ±2% with no parental acts, to +5% with supportive acts, to −10% with projection,
  decided the same day after Claude corrected its account of the band). V4 is met once revision 31 part B is built.
- [ ] **Spec revision 31 part B** (plan `docs/implementation_plan_phase_d.md` §4.11; plan approved 2026-10-16, part A
  built 2026-10-16). **Batch with the V2 premise fixes** (`M11.C.21`, `M11.C.23`, `M11.C.40`; owner, 2026-10-16) so every
  record regenerates once: R31.B1–B6 (the `symptoms.py` docstring, the criteria settings' grades, the config headers and
  record renderer's "pending owner review", `M11.C.14` restated with its mutation) and **R31.B7, `M7.E.1f`'s band
  opened by the parents' acts, its form approved 2026-10-16** (the share form; its known weakness is recorded in the plan).
- [ ] **Seven inconsistencies in the spec found while drafting the review form** (listed in
  `docs/REVIEW_spec_revisions_24-30.md`, "Inconsistencies found while drafting"): `channel_focus` "from belief" (line
  1630) against `M1.A.11c`; `tolerance` Family field (line 1679) against per person (`M7.D.4`); `TRIANGLE`'s channel in
  the B8 plan against `attribution.md`; `M7.E.1c` "open owner decision" (line 1084) against plan line 1355; `M11.C.2`
  settings "to be added" (line 1314) though present; criteria settings ungraded; revision 28 tags with no source. Fix
  with the review's revision, so the spec moves once. *2026-10-16:* all seven are in revision 31's plan (§4.11); the
  `TRIANGLE` one needs no spec change (the owner approved the config as built).
- [ ] **rca: record findings and commit review fixes separately** (rca audit, 2026-10-14): with fixes folded into the
  feature commit, the audit cannot run a reproducer at the parent. Have csdp write `.rca/findings.md`.
- Dismissed with reason (correctness review, 2026-10-14): intrusion is read by `tie_change` after the week's move
  effects and landed contacts, not on the state appraisal saw. Both read the same function; the spec now says which
  state.

## Spec revision 29 (cutoff and focus) — open after the build, 2026-10-14

Plan §4.8 holds the evidence.
- [x] ~~**`I-POSITION` sequence statistics rest on about three completions** (P23).~~ *Done at revision 30: the steering override is gone; the mover starts at level 60 and completes 27 sequences in 192 seeds (ad hoc, 2026-10-14). The modal-length assertion, which a few dozen completions cannot support, was removed.* `tests/bowen/test_iposition.py`'s
  `attempts_over_seeds` runs 192 triad seeds with focus steering switched off by a declared override; with it on,
  `test_m5d8_success_usually_follows_several_failures` fails (0 completions against 3, within chance). Either raise
  the seed count until completions number in the tens, or loosen the assertions to what a few completions can
  support. The tests do not describe the default model while the override stands.
- [x] ~~**A person cannot choose a principled cutoff** (`M1.B.3d`).~~ *Done at revision 30 (`M1.B.3e`): a cutoff is principled in proportion to the danger the person feels.* Principled is declared in the family config only;
  a cutoff the policy selects is always reactive. Learning that cutoff is good therefore raises reactive cutoff. The
  spec states the gap (`M4.D.6f`); closing it needs an owner decision on what a self-chosen principled cutoff is.
- [ ] **Focus steering has no measurable effect in Phase C** after the review made it a redistribution (share 0.578
  against 0.580 off, ad hoc, 20 seeds). Whether `focus_steer_gain` 1.0 is too small to matter is for the owner.
- [x] ~~**Adjacent, not fixed: the revision 28 life-course constants are not range-checked**~~ *Done at revision 30 (`tests/bowen/test_life_course.py`).* in
  `EngineParams.__post_init__` (the revision 29 ones now are). Check each for a meaningful range and add it.
- [x] ~~**`life_run`'s `acts_per_year` divides by the full run~~ *Done at revision 30: the years lived are tracked, and a child never seen at the ages read is counted.* for a child who dies before `projection_until_age`**,
  and `_pick` drops a child with no load at the ages read without counting it (learning-qa, low). Neither happens with
  the reference family over 40 years.
- [x] ~~**Owner: can an external agent receive a dead person's bound stocks?**~~ *Decided 2026-10-14: yes ("okay"); spec `M6.3c` says so.* `M6.3b` gives an external agent no share
  of the dead person's acute anxiety, but `M6.3c` sends a symptom's bound stock to its partners on interactive ties and
  discharges a triangle to all its living members, an external agent among them (found 2026-10-14, re-sweep of the
  revision 29 fixes; `test_m63b_no_share_to_an_external_agent` now isolates `M6.3b`). The code follows the spec as
  written.
- [x] ~~**A scripted act toward a dead target is counted "not legal", not "dead"**~~ *Done at revision 30: status "target dead".* (`src/bowen/ensemble/criteria.py`,
  the forced source's `selections`). In `M11.C.10`'s record, 983 of 16,000 renewals read "not legal"; on seed 2 all 12
  were after Dario died (checked ad hoc, one seed only). "dead" is used only when the actor has died. Add a status for a
  dead target; it changes the engine hash, so batch it with the next engine change.
- Dismissed with reason (correctness review, 2026-10-14): `weakness_focus` reads the usual level after this week's
  `update_usual`. The symptom channels and `find_work` read it the same way at step 8 and 9, so it is consistent.

## Phase D part II — open after the build, 2026-10-13

Plan §4.7 and `docs/phase_d2_status_2026-10-13.md` hold the evidence.
- [x] ~~**Owner: `M11.C.15`'s arm design.**~~ *Decided 2026-10-13: keep the transfer, share at 1. `M11.C.15` now passes
  (death +1.25 ± 0.38) and its mutation turns it red (ad hoc, plan §4.7).*
- [ ] **Owner: the ±2% band against `M11.C.6`'s traversal limit (a)**, and development as `M7.E.1f`'s rule against
  the estimator bounded to the band. `M11.C.6` measured +0.018 ± 0.028 under a band of about ±0.8 points.
- [x] ~~**Owner: still open.** Marta–Iris; Teodor's level 36; the `M11.C.17` gate; the plan §4.5 defaults.~~
  *Settled, corrected 2026-10-14:* the owner said on 2026-10-13 "proceed with your own recommendations" and approved
  plan §4 ("approved"), whose §4.5 lists each default as built unless the owner says otherwise. They stand: Marta–Iris
  ordinary, Teodor 36, the `M11.C.17` gate an active symptom, launch at 20, Gompertz mortality, the partner rows as
  drafted, grief 20, income loss 15, working age 16, a female successor coach. This entry was carried as open in
  error. The owner can still change any of them.
- [x] ~~**Diagnose `M11.C.10` and `M11.C.2` on the twelve-person family.**~~ *Done 2026-10-13, ad hoc (plan §4.7).*
- [ ] **Owner: `M11.C.10`.** The declared cutoff ends within about 2 years, and the uncut baseline is cut within weeks,
  so the arms run the same process. Witnessed-cutoff learning and origin cutoff as an input to intensity (`M1.D.8`)
  have no mechanism. The act count is dominated by generation 3 cutting their own parents (about 775 per run). Hold
  the cutoff? Read tie states? Build link two or three? (plan §4.7)
- [ ] **Owner: `M11.C.2` on the twelve-person family.** Projection is spread evenly (about 80 acts per child). Chronic
  anxiety takes only the year at 11, and the band fixes the level. Late load follows age (Leo 24.9, Pia 15.4). Read
  load over matched ages? Make chronic anxiety integrate childhood? Why does focus not settle on one child?
  (plan §4.7)
- [ ] **Chronic anxiety ratchets across generations.** A child's chronic anxiety is derived from the living parents'
  values and its own witnessed history and then fixed at 12 (`M7.B.1a`); nothing in the run lowers it again for later
  generations. Check over several generations whether it only rises; the 40-year run is too short to see it.
- [ ] **Starting ages as settings in the fam-diag app.** The owner asked for them as settings in the app; they are
  config settings (`M2.A.0j`) and the app was not touched (standing instruction). Deferred.
- [x] ~~**`canonical_state` omits `state.life`.**~~ *Folded in 2026-10-13 with the `M11.C.15` change.*
- [x] ~~**Prove the part II `M16.T.3` re-run by mutation**~~ *2026-10-13: `test_m16t3_canonical_state_sees_the_life_course`
  shows that a change to the life course alone changes the compared state.*
- [ ] **A mutation record for part II.** `M11.C.37`'s mutant is a test; `M11.C.15`'s ran ad hoc (plan §4.7) and is not in
  a record.

## Phase D — open from B3/B4, 2026-10-11

- [x] ~~**Owner: should the `cutoff` pattern readout count the engine's refusals?**~~ *Decided 2026-10-11 (spec
  revision 17): cutoff is read from behaviour — the other's approaches refused, by the engine or by answer, and none
  made. Built in `src/bowen/readouts/patterns.py`.*
- [x] ~~**Open risk for `M11.C.17`: coach-free rises once breadth is met.**~~ *Superseded 2026-10-11: the owner
  rewrote `M11.C.17` (revision 17). A person can rise without a coach, and the coach helps. Built with three nodal
  kinds in both arms (`src/bowen/ensemble/criteria_d.py`).*
- [x] ~~**Which mechanisms count as nodal events for the estimator**~~ *Decided 2026-10-11 (spec revision 17): any
  major event on a person; the list is `config/bowen/nodal_events.md`. Births, deaths, marriages, divorces and launches
  are added with the D-II life course.*
- [x] ~~**Decided 2026-10-11 (owner): the other's gain from "the system adjusts" settles back.**~~ *Built at spec
  revision 20 (`M1.E.7`): the gain is kept in the other's `adjust_swing` and shed by `response_relaxation_rate` each
  calm week (`src/bowen/engine/stress_response.py`;
  `tests/bowen/test_stress_response.py::test_m1e7_the_systems_adjustment_settles_back_when_calm_and_holds_under_load`).*
- [x] ~~**Decided 2026-10-11 (owner): restate `M11.C.18` and `M11.C.20`'s mutation clause.**~~ *Done at spec revision
  20: C.18 names the stress response, C.20 the estimator ignoring borrowed functioning. Measured on the revision 20
  engine in `docs/implementation_plan_phase_d.md` (step 6); recorded with the Phase D record (step 12).*
- [x] ~~**For the owner: may the estimator read the system's adjustment?**~~ *Decided 2026-10-11 (owner): (a), the
  estimator leaves it out. Built at revision 20 (`M1.A.4k`; `src/bowen/engine/slow_tick.py`;
  `tests/bowen/test_slow_tick.py::test_m1a4k_the_estimator_leaves_out_the_systems_adjustment`).*
- [x] ~~**For the owner: `M11.C.17` fails on the revision 20 engine (2026-10-12).**~~ *Resolved by revisions 21-23 (2026-10-12): C.17 passes, small (+0.021 ± 0.009). Kept below as the record of how.* Nobody's basic level rises in ten
  years in either arm. Coaching changes behaviour (perspective 0.63, self-directed weeks 0.37 against 0.11), but
  everyone is loaded in nearly every week, functioning falls in both arms (40 → 28–30) and the estimator's evidence
  stays at −11 to −15 (`docs/implementation_plan_phase_d.md`, step 6, the trace). Questions: is C.17's declared load
  (nodal events to every member, so nobody is ever calm) the right test bed; where the coached people's fall outside
  the stress swing comes from (traced 2026-10-12: being over-functioned for; revision 21 stops that for a person holding a position, and C.17 still fails, see the plan, step 6); and can the estimator (`M1.A.4k`) register improvement that shows as
  falling less rather than rising. The +0.63 (revision 19) and +0.66 (`a35a962`) passes were carried by the stacked
  system adjustment.
- [ ] **`M11.C.44` moved FAIL → PASS at revision 20 (2026-10-12), with both mutants surviving.** p went from 0.055 to
  0.012 on an engine change that does not touch the outside position (the system's adjustment as a level). A criterion
  that flips on an unrelated change is on its threshold; with `one-sided-deviation` and `deviation-inverted` both
  surviving, the pass is not evidence (`M11.5`). At revision 21 it moved back to FAIL. For the owner when Phase C's gate is
  next looked at.
  Revision 24 (2026-10-12): it moved FAIL → PASS again with B7's symptom channels. Still on its threshold. Revisions 25–27 moved it back to FAIL.
- [ ] **Review of revision 20's completion, minor (2026-10-12; the correctness pass, which read the version with
  `shed_adjustment`, since removed):**
  - At the stress floor a calm week decays `adjust_swing` while functioning cannot fall, so the estimator's `own`
    is overstated by at most `response_relaxation_rate × system_adjust_gain` (0.02). Intended, as for `stress_swing`.
  - No test pins a partial top-up (0 < `adjust_swing` < `system_adjust_gain`) in `external.self_practice`.
  - The once-per-stance gate is now mostly redundant with the level (a top-up at the gain adds nothing); it only limits
    renewal to once per `practice_window`.
- [x] ~~**For the owner: nobody in the Phase C family is ever calm (traced 2026-10-12).**~~ *Decided 2026-10-12: calm relative to the person; built at spec revision 22.* The coached people's excess anxiety
  sits at 30–35 against the calm band's edge of 5, with or without nodal events; the standing load alone would hold
  about 6–7 (`docs/implementation_plan_phase_d.md`, step 6). So the stress response never settles and C.17 cannot
  show a rise. Options: read "calm" against each person's usual level; set the bands from the engine's operating point;
  or make the family's traffic carry less anxiety.
- [x] ~~**For the owner: should the estimator also add back functioning lent? (traced 2026-10-12)**~~ *Decided 2026-10-12: yes; built at spec revision 23. C.17 then passes (+0.021 ± 0.009).* It leaves out borrowed
  functioning (positive pseudo-self, revision 19) but keeps what a person gave way, so its reading sits about 8–9
  points below basic level; leaving out the whole exchange, with its sign, puts the reading at basic level
  (`docs/implementation_plan_phase_d.md`, step 6). The load weighting is not the cause.
- [x] ~~**Revisions 25–27 (2026-10-12): for the owner's review.**~~ *Reviewed 2026-10-16 with revisions 24–30.* Attribution and its rules, belief in the witness's
  appraisal, the family-focus writer, the marital-distance gate, the tie-weighted cross-person term, and borrowed
  pseudo-self settling back. All Claude's recommendations, made overnight at the owner's request.
- [ ] **Failing criteria from steps 8–10, each the owner's question; nothing was changed to make any pass:**
  - `M11.C.14`: the gate blocks improvement but not harm, so firmness and talking it out still differ. *Decided
    2026-10-16 (review, option B):* restate it to "no technique lowers the child's load against a no-technique arm";
    the difference between techniques is reported. Revision 31.
  - *Version 1 (2026-10-16):* `M11.C.21`, `M11.C.23` and `M11.C.40` are failing **premises**, so V2 requires each to be
    fixed or restated by the owner before version 1. `M11.C.21` and `M11.C.40` need a diagnosis run first;
    `M11.C.23`'s direction is set by two `[I]` sizes, so bring the owner options. `M11.C.43` is a composite: the
    owner accepts it as a finding or not (V1).
  - `M11.C.40`: per seed, the automatic channel's entropy difference does not exceed the self-directed one's.
  - `M11.C.43`: no larger cost on losing a calming member after learned reliance.
  - `M11.C.21`: the target does not rise when both parents collapse.
  - `M11.C.23`: reversed; halving symptom accumulation outweighs the hostility.
- [ ] **`M11.C.2` fails** once projection acts are counted early and load read late (+0.29 ± 2.20, p 0.39); read over the
  same weeks it passed, which cannot separate cause from attraction. No criterion yet shows the tie-weighted
  cross-person term (revision 26) shaping a child's repertoire.
- [ ] **`M11.C.34` fails** on reactions to the onset against a no-onset run; and its declared mutant cannot bite in any
  case: no selection reads `M1.A.19`'s detectors.
- [ ] **`M11.C.17`'s coached rise nearly vanished** with revisions 24–27 (+0.021 → +0.0009). Traced 2026-10-13 (plan, step
  10): the coach's contact lands only when binders fail (an active symptom), which at revision 23 was 99% of weeks only
  because everyone was saturated; now about 40%, so landings and perspective halve. Owner's question: should the gate
  read load above the usual level instead? *Decided 2026-10-16 (review): no, it keeps reading an active symptom or
  load near threshold.* The weak pass stands and needs a recorded mutation for V3.
- [ ] **Reviews of revisions 25–27 (2026-10-12), not fixed, with reasons:**
  - **`I-POSITION` sequences now rarely complete.** In the triad fixture, completion fell from about 10 in 96 seeds
    (revision 20) to 2 in 96 after B7, and 4 in 192 now. `tests/bowen/test_iposition.py::test_m5d8_*` was raised to
    192 seeds and still rests on four sequences (P23). Why B7 lowered it is not traced.
  - The marital-distance gate blocks a symptomatic child's improvement, but the move's contact and impingement
    still land on the tie and persist. Gating them as well made `M11.C.14` no closer to its null (measured: −8.4 to
    −10.9 over 20 seeds), so it was left.
  - `update_focus` reads the sender's true excess above usual: the family's focus is engine-side attention, not a
    belief any policy reads (the spec says so).
  - A refused move never reaches its sender's own attribution.
  - `arm_c43` makes M unavailable to F, not F to M: M can still approach F in the loss window.
  - `arm_c26` and `arm_c22` return one scenario's records, so their bookkeeping covers one run of several.
  - `M11.C.36` measures appraisal only; its row says "appraisal and selection".
  - `relax_pseudo_self` runs tie by tie, so a person relaxed on two ties in one week sees the first change before
    the second: deterministic, but not `M1.F.8`'s batch rule.
- [ ] **`M11.C.7`, `M11.C.9` and `M11.C.28` are not built**: C.7 needs the coach's landing to read the coach's position
  (`M8`'s predicate exists, unused); C.9 needs sibling position to gate behaviour (`M1.A.14c`); C.28 needs the sink
  allocation to feed some behaviour, or its mobility can change no symptom.
- [ ] **Phase D record, review of 2026-10-13, minor, not fixed:** `tools/spec_coverage.py` reads only the Phase C
  ensemble record, so `docs/spec_coverage.md` gives the Phase D criteria no verdict; and the README's and the status
  report's lists of passing and failing D criteria are not checked against `docs/phase_d_ensemble_record.md` by a test.
- [ ] **B7 (spec revision 24, 2026-10-12): for the owner's review.** Every B7 choice is Claude's recommendation, made
  overnight at the owner's request: the symptom integrand read above the usual level, the channel weights, mutual
  protection, the lock-in curve and its spill past the turn, removal by tolerance, and `M11.C.22`'s control arm.
- [ ] **`M11.C.12` fails** (B7, 2026-10-12): released anxiety raises the other spouse's excess, but more anxiety does
  not select `CONFLICT` in particular (dyad, 20 seeds: 55 against 56 conflict acts between the spouses). Whether the criterion, the
  readout or the policy is wrong is the owner's question; nothing was changed to make it pass.
- [ ] **`M11.C.11`'s direction limb has no mutant** (B7): the return's disturbance exceeds the removal's. About half
  is the removed member's symptom hold, frozen while they are away (no partners to release to) and released on
  return; decide whether a removed member should keep the hold. Reunion's restored contact is not what carries the
  rest. Find what does, and declare it. *Version 1 (2026-10-16):* `M11.5` classes this limb composite, so V3 needs a
  recorded mutation for it.
- [ ] **Lock-in does not make a symptom chronic** (B7): the hold does not feed the bearer's integrator, because
  feeding it made every bearer climb to the turn and stay there. `KS23.1`'s "fosters it becoming chronic" is not
  produced.
- [ ] **Correctness pass on B7, minor (2026-10-12), not fixed:**
  - On the dyad, a remission's returned hold is split between the spouse and the bearer's parent, so half of it
    reaches the marriage.
  - The hold is one stock per person, not per channel: remitting one channel returns all of it, though another
    channel may still be active.
  - Lock-in runs in person order: a partner of two bearers gives more to the first, and one bearer's spill is
    counted in the next one's excess. Deterministic, but not symmetric between people.
  - `M11.C.30` reads mean social *load*, a stand-in for the onset hazard its row names.
  - `arm_c11` and `arm_c12` have no fast test; a window or arm error shows only in the ensemble run.
- [ ] **`Family.tolerance` is declared, never updated** (B7): `M7.D.4`'s "tolerance update" in the register is not
  built, and no fixture declares a tolerance, so removal by tolerance never fires in any run.
- [ ] **Review of revision 23, minor (2026-10-12):** `functional_level` is clamped at 100 (and at the stress floor) by the
  stress response, the I-POSITION gain and the system's adjustment, while pseudo-self is not, so a person at a bound
  can read slightly off basic level in the estimator. The exchange itself clamps its amount and is exact.
- [ ] **Adjacent, found in the review of revision 23 (2026-10-12), not fixed:** the `M1.A.5c` table says a sustained
  self-directed rise moves pseudo-self **down** ("converting toward solid"). Nothing implements that; only
  `initialise_functioning` and `functioning_shift` write pseudo-self. If it is implemented, the signed estimator reading
  (revision 23) would read each fall in pseudo-self as a rise in own functioning, on top of any rise in
  `functional_level` itself. Decide the two together.
- [ ] **Review of revision 20, minor (2026-10-11):** no full-tick test that a calm partner sheds part of the gain in the
  week it arrives (`self_practice` runs before `respond` in `tick.py`; consistent with the rule, not pinned).
- [ ] **Adjacent, not fixed (review of revision 21, 2026-10-12): two checks are written out more than once.** "Self-directed
  or a hold" (`kind == WITHHOLD or channels.get(kind) == SELF_CHANNEL`) appears in `external.py` (twice),
  `stress_response.py` and `slow_tick.py`; "held their position" is `moves.holds_position` and, separately,
  `external.self_practice`'s soothing check (which judges only the first target). One shared helper each would keep them
  from drifting apart.
- [ ] **Pre-existing unused imports in `tests/bowen/test_engineering_gates.py`:** `clamp_unit`, `dataclasses`,
  `EventStore`, `legal_outcomes`. Found while checking the B3/B4 diff. Not touched, since they predate it.

## Phase C — open after the build, 2026-10-08

Phase C steps 0–16 are built. **Its gate does not pass** (`docs/phase_c_completion_report.md`). These need a person.

- [x] ~~**Decide `TRIANGLE`'s inside pair (`M11.C.3`).**~~ *Decided 2026-10-08: the target is the recruited third,
  outside; the inside pair is the sender and its most strained partner, as `M1.F.1b`, `M11.C.3`, `M11.C.42` and
  `M6.4` already read it. C.3 now passes and is proved; C.42, C.45, one cell of C.27 and one of C.41 stopped
  passing. Recorded at spec `M1.C.1`; report §8.*
- [x] ~~**Accept or revise the `M11.5` reclassifications.**~~ *Decided 2026-10-08: `M11.C.1` and `.38` stay
  **premise**, with no new class. Every level-reading rule states the direction, so a pass is not a finding; the rule
  column now lists the nine rules, and the proof is the joint level-blind mutant. `M11.C.45`'s reclassification as a
  premise was withdrawn when it stopped passing after the `TRIANGLE` decision. Report §2.*
- [ ] **`M11.C.42` and `.45` stopped passing after the `TRIANGLE` decision.** *What each tests decided 2026-10-08
  (report §9): C.42's baseline arm and readout, and C.45's readout, brought to the spec's text. Rerun: both still
  fail. Current figures (`docs/phase_c_ensemble_record.md`, 2026-10-10): C.42 +0.08 ± 0.39 (p 0.21); C.45
  −0.0008 ± 0.0044 per person-week (p 0.69). The 2026-10-08 figures were C.42 +0.26 and C.45 +0.0020. What remains is whether the learner should produce these
  effects at all, which is a model question, not a test question.* Both were already fragile in the sweep.
  C.42's reuse is −0.17 and it fails at every sweep setting; C.45 has the right sign, without significance.
  Decide whether either criterion's arms still test what it claims, now that the act loads the recruited third.
  Report §3.8 and §8.
- [x] ~~**`M13.4`'s precision rule and `M11.C.16`.**~~ *Decided 2026-10-08 (report §10): precision against the pooled sd
  of the two arms. C.16 now passes at 150 seeds; C.41's [light: lower level] cell now fails instead of undetermined.*
- [x] ~~**Decide whether to reclassify `M11.C.16`.**~~ *Decided 2026-10-08: it stays **composite** (owner decision).
  *Superseded 2026-10-09: `availability-level-inverted` now reverses it (`repertoire_entropy` +0.0115 ± 0.0094;
  `docs/phase_c_mutation_record.md`), and the owner kept it composite.* As first written: no inversion run reversed
  it; inverting `M4.D.3a`'s availability cancelled it, and removing availability halved it; all nine level-reading rules were run. The learner changes its size, not its
  sign, so its row's rationale and mutation clause are not met, and coverage keeps it partial. Report §11.*
- [ ] **Owner: `M11.C.16`'s class rests on a reading.** Composite holds if only `M4.D.3a` states a narrower repertoire;
  if `M4.C.1a`'s KS03.2 text or `M1.C.3a`'s "stays fixed" is read as stating it, `M11.5`'s redundancy clause would make
  it a premise. Removing those two rules alone leaves C.16 passing (`c16-cited-grounds-removed`). Removing them with
  availability now also survives (`c16-grounds-removed`, −0.006 ± 0.005), while the availability inversion reverses it
  (`docs/phase_c_mutation_record.md`). Occupancy figures are in `docs/phase_c_level_occupancy.md`. Report §11.
- [ ] **The record caches are trusted, not proved.** A test re-renders each record from `docs/records_cache/` and
  compares, but nothing binds a cached result to a run: editing a value in the cache and re-rendering passes. The
  key hashes the inputs, not the result. Only a rerun checks it (learning-qa, 2026-10-09).
- [ ] **`docs/phase_c_level_occupancy.md` is checked by its hash line only**, not re-rendered from cached rows like
  the mutation and sweep records, so a hand-edited table would pass (learning-qa, 2026-10-09).
- [ ] **`tools/record_cache.py`'s infrastructure-error match is by exact class name.** Subclasses
  (`FileNotFoundError`, `BrokenPipeError`, `BrokenExecutor`) fall through to an uncached red instead of stopping the run;
  `KeyboardInterrupt` in the list is dead (it is not an `Exception`); a child that returns no row for a requested
  criterion yields `None` silently (learning-qa, 2026-10-09).
- [ ] **Diagnostic probes, minor (batch review 2026-10-09).** `c44_act_counts` re-runs `arm_spell_triad`'s scenario
  instead of calling the arm; probes sample at tick start, so the last tick is missed; `summarise` takes its columns
  from the first group and assumes every row has the same keys. None changes a current number. The
  "no rows" and "names no criterion" exits of `tools/diagnostic_record.py` have no test.
- [ ] **Record-parsing duplication.** The record JSON regex and `json.loads` appear about six times across
  `tools/mutation_record.py`, `tools/sweep_record.py`, `tools/spec_coverage.py` and the tests; one helper would serve
  them (correctness review, 2026-10-09).
- [ ] **`M11.C.16`'s coverage override is hand-written** (`tools/spec_coverage_phase_b.json`) and says "passes"; it
  will not update if the ensemble verdict changes (correctness review, 2026-10-09).
- [ ] **A child failure or timeout in `run_in_copy` aborts the whole record run** and discards earlier mutants'
  results; a timed-out child's worker pool may be orphaned. Pre-existing (correctness review, 2026-10-09).
- [ ] **`M11.C.16`'s mutation clause is not met.** The spec requires that disabling `M4.D.6` turn it red, and it does
  not. Decide whether to restate the criterion so that a learned contribution is what it tests (for example, hold
  level-reading availability fixed and compare learning on against off), or to accept it as a level result.
- [x] ~~**Owner: decide S, X1 and Q2-Q5**~~ *Approved 2026-10-09: every recommendation in
  `docs/DECISIONS — PHASE C FAILING.md`.*
- [x] ~~**Carry out the approved decisions**~~ *done 2026-10-09; results at the top of
  `docs/DECISIONS — PHASE C FAILING.md`.* S by holding each scripted act's ties open (week 0, the only week legal in
  every seed, sits on the chronic floor); X1 by D7; Q2's readout; Q3's paired measurement; Q4/Q5 rerun.
- [x] ~~**Owner decisions left by the approved steps**~~ *decided 2026-10-09* (memo, "The owner's decisions"): X1 a
  finding; `M11.C.5`'s FAIL reported; `M11.C.4` and `M11.C.27` run at 104 weeks (no effect remains).
- [x] ~~**Approve or amend `docs/PROPOSAL — TRIANGLE RELIEF AND CREDIT.md`**~~ *approved and built 2026-10-09: the
  seeker alone is relieved, as far as the third helps; a `TRIANGLE` is credited by the seeker's own relief.*
- [x] ~~**`M11.C.27`'s horizon**~~ *decided 2026-10-09: read week by week; it gates on the week the act lands.*
  1 of 4 cells passes; removing one stabilises first and destabilises from week 6; adding a third barely moves it.
- [x] ~~**`M11.C.16`'s class, again**~~ *owner decision 2026-10-09: it stays composite, although
  `availability-level-inverted` now reverses it (report §11).*
- [ ] **The triangle change is not proved by an acceptance criterion** (2026-10-09): `triangle-help-ignored`
  survives on `M11.C.3` (a direction, not a size), and `triangle-credit-cross-person` has no passing criterion.
  Both rules have unit tests. A criterion that reads the size of the seeker's relief against the third's help, or
  `M11.C.42`/`.45` passing, would prove them.
- [ ] **The trace renderer's framing says every run is scripted** ("no one in it chose anything (Phase B)"),
  which is false for a Phase C run. `tools/phase_c_trace.py` replaces the sentence and fails if it moves. Fix it in
  `src/bowen/render/trace.py` (choose the sentence from the header) with the next engine change, since that file
  is in the engine hash.
- [ ] **`code_text` (the engine hash) has two latent gaps** (re-sweep of `d7ce2b3..`, 2026-10-09): it drops a whole
  docstring line, so code sharing that line (`"""doc"""; x = 1`) is invisible to the hash; and it strips blank lines
  and trailing spaces inside ordinary multi-line strings. No current file has either pattern (checked: the fixed
  version gives every file the same text). Not fixed now because `tools/ensemble_record.py`, which holds
  `code_text`, is itself hashed, so any edit to it reruns every record (about an hour). With the next engine
  change: blank only a docstring's own span, keep the inside of strings exactly, and hash `RULE_KEYS`' values
  rather than the whole file, so the hashing code can change without changing the hash.
- [ ] **Re-sweep notes, 2026-10-09** (fix range `da57f87..996bb7a`; deferred, not fixed now):
  1. ~~`reactive_over_chance` averaged in `FALLBACK` selections~~ *done 2026-10-09: policy draws only; the
     verdicts did not change.*
  2. Three probes (`c27_deviation_terms`, `horizons`; `tools/probes.py`) call `made(criteria)` without records,
     so they count placed acts as made. Diagnostics only; pass the records at the next probe edit.
  3. Low: a scripted week the run reached but never offered would read "not reached"; the scripted-criteria
     scans stop at an arm's first `scenario` call (every arm calls it once today).
- [x] ~~**Run the approved plan for the failing criteria**~~ *D0-D6 done 2026-10-09; see the decision memo.* (`docs/plan_phase_c_failing_criteria.md`, approved 2026-10-09):
  D0 diagnostic record, D1-D6 diagnoses, one decision memo. `M11.C.7`, `.13` and `.14` moved to Phase D the same day.
- [x] ~~**The failing criteria**~~ *closed 2026-10-09: Phase C closed with its gate failing (owner decision); each
  failure is recorded as a finding with its measured cause, and what the next phase inherits, in report §13.*
- [ ] **Plan §9's human reviews:**
  - ~~`docs/readouts_phase_c.md`~~ *the owner gave the definitions, 2026-10-10 (visualization plan V1).*
  - **The 104-week Phase C trace is not reviewed** (owner, 2026-10-10: viewing the seed-7 diagram was not the trace
    review). Read `docs/review/phase_c_trace_seed7.md`, or the diagram in EPModel Fam-Diag.
  - **`M5.F.2`'s threshold is open.** The owner found the first explanation unclear (2026-10-10), so it was
    re-explained.
- [x] ~~**Owner review of the run-visualization choices**~~ *2026-10-10: the owner accepts the invented values
  ("make it up"). The tie-break and silence exception stay `[I]`.* (`docs/PROPOSAL — RUN VISUALIZATION MAPPING.md`, built
  2026-10-10). Each is `[I]`:
  - the readouts' tie-break (more acts, then the earlier first act);
  - silence not replacing a cutoff;
  - the level bands in `config/viz/famdiag.md`;
  - Viktor's and Dario's values, Teodor's `basic_level` 34 → 36 (34 broke `M2.A.0e`'s ±1 with Ana), and the Phase D
    family's tie values (`config/bowen/family_phase_d.md`).
- [x] ~~**Readout questions from the 2026-10-10 review**~~ *answered by the owner the same day and built:
  lopsided pursuit is over/underfunctioning; a parent's acts toward a child are projection, which holds until 12 weeks
  pass without a projection act (`pattern_projection_hold`).* The questions were:
  1. **Fusion usually beats over/underfunctioning when pursuit is lopsided.** Every PURSUE counts toward fusion. So
     A→B 5 times and B→A once reads as fusion (6 acts) over over/underfunctioning (5 vs 1). Pursuit reads as
     over/underfunctioning only when it is strictly one-way. Should lopsided pursuit read as over/underfunctioning?
  2. **A parent's OVERFUNCTION or PURSUE toward a child never counts for over/underfunctioning,** even in a block
     where projection does not reach 3 acts. Should those acts count when projection is not read?
- [x] ~~**`family_phase_d.md` lacks Teodor–Iris and Viktor–Dario parent ties**~~ *added 2026-10-10.* (the owner's family has Iris as Teodor's
  daughter and Dario as Viktor's son).
- [ ] **Rule 9 debt:** the readouts' act sets (`APPROACH`, `PROJECTION_ACTS`, the over/under lists in
  `src/bowen/readouts/patterns.py`) are classification rules in Python. `PROJECTION_ACTS` is tied to the sink's
  set by `test_v5_projection_reads_the_sinks_act_set`. Move them to config with the next engine change.
- [ ] **fam-diag, found in review, not fixed** (pre-existing; neither affects File > Open of an export):
  - merge import collapses two episodes of the same pattern on one pair into one line (`utils/diagramMerge.ts`);
  - `FamilyCutoffArc` ignores the timeline (`DiagramCanvas.tsx`).
- [ ] **Phase D: severed ties stay severed — the cause of the long silences** (owner: "not normal", 2026-10-10).
  On seed 7, Marta–Nadia is severed 102 of 104 weeks, from a cutoff in week 1. Evidence and design notes are in
  `docs/DECISIONS — CUTOFF SEMANTICS.md`.
- [ ] **`code_text` splits lines differently from `tokenize`** (re-sweep, 2026-10-11, latent). `tools/ensemble_record.py`
  builds `lines` with `str.splitlines`, which also splits on form feed, `\x1c`–`\x1e`, `\x85`, U+2028/9 and a bare
  `\r`; `tokenize` splits on `\n` only. A source file with one of those characters would hash differently by Python
  version. No file has one today. Fix: `io.StringIO(source).readlines()`. Do it with the next change to that file
  (it is in the ensemble record's stamp).
- [ ] **Engine CUTOFF semantics, held for Phase D** (owner, 2026-10-10) (`docs/DECISIONS — CUTOFF SEMANTICS.md`; recommendation C, Phase D).
- [ ] **The pattern readouts read `state.tick` as "now"**, but after `run_tick` it is the next week's tick. The exporter
  steps it back and restores it (`tools/famdiag_export.py:simulate`). A `now` argument on the readouts would be
  cleaner. It is an engine-hash change, so do it with the next engine change.
- [x] ~~**fam-diag branch `timeline-hides-ended-lines`**~~ *2026-10-10: the owner's fam-diag stays as it is. The work
  became a separate app, **EPModel Fam-Diag**: `/Users/davemini2/ProjectsLocal/epmodel-fam-diag`, port 5174,
  private repository `dbgnvan2/epmodel-fam-diag`. Its gate file is
  `docs/cycles/gate_2026-10-10_timeline-hides-ended-lines.md` there. The branch was removed from fam-diag, and fam-diag's
  `main` matches GitHub at `e61ae13`.*
- [ ] **What the seed-7 picture shows**, for the owner's review. Most parent–child and marital ties go silent
  for months: Marta–Nadia has no act for up to 102 weeks, Marta–Ravi for 67. No cutoff pattern forms, although there are 62
  CUTOFF acts: no one refuses 3 approaches in 4 weeks. Sofia → Ravi projection holds for 20 blocks.
- [ ] **Not built:** `M11.C.7` (needs `M8.2`/`M8.3`'s predicates and a direction), `.13` (needs a community),
  `.14` (needs `M5.C.1`'s marital-distance gate); two of `M11.1c`'s four re-encodings (rescaled state range,
  integer against float ticks); `M11.1b`'s severing mutants; plan D9's "dominant constant" (§3 names none).
- [x] ~~**Bruno falls back every week.**~~ *Recorded as a Phase C finding, not fixed (owner, 2026-10-10): Bruno left the reference family at spec revision 13, so this affects only the closed Phase C runs.* His one tie starts cut off, `REDUCE_CUTOFF` is his only legal act, and it
  carries no weight without systems perspective, so the family runs' fallback rate is 0.26–0.41 (`M11.D.18` flags
  it). Decide whether that is the intended reading of `M4.D.3b`.
- [ ] **Ninth gate's notes, 2026-10-08** (`docs/cycles/2026-10-08_phase-c-qa-gate-9.md`, APPROVED, nothing medium or
  higher): (1) `tick_of` in `tests/bowen/test_phase_c_criteria.py` does not handle `LogHeader`, which `run_tick` never
  emits; make it total or assert the record type. (2) The C.45 rate check still copies the readout's filter; share one
  predicate from `criteria.py` the next time the records are regenerated anyway, since moving it into `src` changes
  the hash. (The third note, a stale date in `CLAUDE.md`, is fixed.)
- [x] ~~**Eighth gate's low findings, 2026-10-08.**~~ *Fixed 2026-10-08, tests only: the C.45 rate check applies the
  readout's `MOVE` filter; the before-t0 check compares every record by its tick; new
  `test_m11c42_readout_counts_the_whole_pairs_triangles` and
  `test_m11c42_absence_forms_no_triangle_holding_the_absent_member`. Each proved failing by mutation.*
- [x] ~~**Fifth gate's low finding, 2026-10-08 (informational).**~~ *Fixed 2026-10-08: the tests load
  `tools/sweep_record.py` as `tools.sweep_record`, and `test_sweep_tool_shares_the_loaded_mutation_tool` asserts it.*
- [x] ~~**Third gate's low findings, 2026-10-08.**~~ *Fixed 2026-10-08:
  `test_criteria_required_matches_what_the_arms_read` ties `REQUIRED` to every arm's reads in both directions; the
  `criteria.py` docstring points at `config/bowen/criteria.md` instead of restating the spell; the mutation tool is
  loaded once (`test_sweep_tool_shares_the_loaded_mutation_tool`). Records regenerated.*
- [x] ~~**Re-gate finding A (medium, P6/P19).**~~ *Fixed 2026-10-08: every record's hash covers
  `tools/ensemble_record.py` (its `RULE_KEYS`), and each tool names its inputs in `HASHED_TOOLS`
  (`test_m134_every_record_hashes_the_ensemble_tool`). All three records regenerated.*
- [x] ~~**Re-gate finding B (low, P19).**~~ *Fixed 2026-10-08: the sweep staleness test imports the sweep tool's
  `HASHED_TOOLS`.*
- [x] ~~**Hermes gate finding 4 (low, P4).**~~ *Fixed 2026-10-08: the criteria's declared settings moved, unchanged, to
  `config/bowen/criteria.md` (parsed strictly, `test_criteria_settings_are_parsed_strictly`), and `M11.D.2`'s scan
  now covers `src/bowen/ensemble/criteria.py` (`test_m11d2_check_catches_a_literal_in_the_criteria`).*

## Spec revision 10 — deferred at the owner's review, 2026-09-22

Revision 10 (branch `spec-rev10-draft`) folds the method literature into the spec. The owner answered its
fourteen questions on 2026-09-22; these are the ones left for later. The answers table is in the spec's
revision-10 section.

- [ ] **Disposition at death (C15, spec `M6.3`).** Decide where a dead person's anxiety, bond energy,
  functioning-balance debts, positions and budget share go. Owner's guidance, not yet a requirement:
  relationship energy behaves like chemical bond energy and does not disappear when a person dies. That agrees
  with `M1.B.4` (bond energy decays at or near zero) and `M6.I.7` (no exit from the field). `M11.C.37` cannot be
  written until this is decided.
- [x] ~~**Habituation on repeated relief (C13, spec `M4.G.3`).**~~ *Kept as written at approval, 2026-10-06: it is a conditional SHOULD and binds only if such a term exists.*
- [ ] **`E-RM` and C34 — one requirement or two (DESIGN_LESSONS Q16).** Currently `M17.D.1`(d) plus `M17.D.2`.
- [x] ~~**Does disabling a mechanism count as a second parameter set under `M0.4`? (DESIGN_LESSONS Q10)**~~
  *Answered 2026-09-22: switching off a mechanism is a change to the model; such comparisons are allowed as tests of the model's structure only (the recommendation the owner adopted). Written as spec `M11.4f`.*
- [x] ~~**Does `M11.4f` reach the approved ablation of `M10.C.4a`?**~~ *Decided 2026-09-22: yes. Each run with a
  condition removed tests whether the model needs that condition; `M10.C.4a` itself is unchanged.*
- [x] ~~**Strike `M11.4c` at approval unless a mechanism ranking turns up.**~~ *Kept, 2026-10-06: the external review found one in the ledger (`L07.4`).* Answered 2026-09-22: most mechanisms
  scale with differentiation and stress, so the theory orders conditions, not mechanisms — tested as
  `M11.C.41`. The ranking form has nothing to rank.
- [x] ~~**Confirm the `E-DR` recommendation (Q11).**~~ *Approved 2026-09-22: `M17.E.1` and `M17.E.2` kept separate, sharing one sweep range.*
- [x] ~~**At approval: restate the requirement count in five documents.**~~ *Done 2026-10-06: 519.* `CLAUDE.md` (also still says revision 6),
  `README.md`, `CHANGELOG.md`, `docs/theory/_STATUS.md` and `docs/agent_model_proposal.html` claim 427;
  `tests/test_spec_consistency.py::test_requirement_counts_agree_with_the_spec` is red on the branch until they
  match the approved count.

## Held for a later revision — paper material, 2026-09-27 (labelled "revision 11" until 2026-10-07)

> **Relabelled 2026-10-07 (spec revision 12, P12).** Revision 11 became the emergence revision and revision 12 the
> Phase C prerequisites. Nothing in this section is part of either. It waits for the owner to assign it to a revision.
>
> **For the Phase D plan (revision 12, P10):** criteria for Bowen's two falsifiers of the budget (`L09.2`, `L16.2`) and
> `L15.1`'s delayed symptom after a loss; decide `L03.4`'s two-rung avoidance ladder and `L03.5`'s illness in the mover.
>
> **Phase C plan prerequisites from revision 11:** rework Phase B's base appraisal (`appraise_base.py`, now `M4.C.1`–`M4.C.1b`),
> standing load (`standing_load.py`, now `M4.C.1c`) and triangle activity (`recompute.py`'s `tension_activation_threshold`
> and `test_m1c3_triangles_are_inoperative_when_calm`, now amended `M1.C.3`). Define the pattern readouts `M5.A.1a`
> requires before any criterion reads them. Decide the five rule groups revision 11 marked for review.

The owner decided on 2026-09-27 that nothing from the 2026-09-27 paper batches enters revision 10, which is
under external review. All of it waits for revision 11. None of it is a requirement yet.

**One exception, by owner instruction on 2026-10-06:** `M11.F.10` (the "virtual family" term may not be used
without `M11.F`'s framing) was entered in the spec text directly, marked ⟦proposed rev11 · owner decision
2026-10-06⟧. **Approved with revision 10 the same day**, so it is no longer a revision-11 item. Rationale: `model_explainer.md` §1.1.

- [ ] **Method proposals from the two 2026-09-27 batches.** 36 candidates. First batch: J1–J4, SV1–SV4, PD1–PD4,
  X5–X9, in `papers/SWEEP_READING_REPORTS_2026-09-27.md`. Second batch: BB1–BB5, MM1–MM4, AD1–AD3, TM1–TM6, X10,
  in `papers/SWEEP_READING_REPORTS_2026-09-27b.md`. Plain-language versions with examples are in
  `papers/SPEC_CANDIDATES_plain_language_2026-09-27.md` and `…-27b.md`. Candidates labelled X are for the
  exploratory narrator line only and do not enter the v2 spec.
- [ ] **Owner answer: anxiety shapes beliefs (answers MM1).** Anxiety makes a person's beliefs about others more
  subjective. It also biases risk assessment upward, so the same event is read as more threatening. For revision 11:
  - belief writes (`M9.8`, and the per-person belief store) read the receiver's own anxiety, not only the events
    delivered;
  - the weight on the receiver's own state rises with anxiety;
  - the threat bias points one way, upward;
  - direction graded `[user]`, strength graded `[I]`;
  - check `M4.C.5` and FE05.10 for corpus support before raising the grade;
  - `M9.2` still binds: a threat-biased reading is not automatically false.

  MM1's probe (identical events, only the receiver's anxiety differs, plus a label-only null arm) becomes the test.
- [ ] **Owner answer: how coaching enters a family (answers SV2).**
  - **Coach.** Present from t0 in every run.
  - **Knowledge.** A family learns that coaching exists only through an outside event, which reaches one or more
    named people at a declared tick.
  - **What makes an arm.** Arms differ by that knowledge event, meaning whether it happens, when, or who receives it.
    If both runs receive the same event on the same tick, it is shared history, not an arm, and not what the
    comparison tests.
  - **After the event.** Everything is the family's own dynamics: whether to try coaching, how many sessions, and
    whether to stop once relief comes or continue. The owner expects lower-differentiated people to stop once
    relieved more often, graded `[user]`; corpus support for relief lowering motivation is ledger L14.6 and L14.2.
  - **Spread.** Once one person knows, knowledge spreads through ordinary ties, mostly to those closest and safest.
    A partner is almost always told. Spread is therefore an output of the model, not part of the arm definition. The
    exception is an arm in which the person is instructed not to tell their partner: that instruction is declared
    as part of the knowledge event.
  - **Reporting.** The headline result compares runs where the family learned about coaching with runs where it
    did not. Never tried, tried and stopped, and continued are reported underneath as a breakdown, never as the
    comparison itself.
  - **Where it sits in the spec.** Next to `M1.E.7f` (family-relative entry threshold and rejection hazard) and
    `M1.E.8` (low optimum for contact frequency).
- [ ] **Method proposals from the two 2026-10-04 batches.** Added 2026-10-04 by owner decision; held for revision 11
  on the same terms as the 2026-09-27 batches. 59 candidates, including narrator-line items.
  - **Batch a.** The five papers from the 2026-09-28 sweep: WA1–WA5, CD1–CD4, CB1–CB3, BV1–BV3, PW1–PW3, plus
    WA-X1, CD-X1 and PW-X1. In `papers/SWEEP_READING_REPORTS_2026-10-04.md`.
  - **Batch b.** 21 papers from the owner's 25-paper digest of 2026-09-15 to 09-30. The other four had already been
    read. Candidates: PF1–PF3, AQ1–AQ3, RP1–RP2, PP1–PP2, EN1, TP1, PM1–PM2, RA1–RA2, NG1–NG3, RZ1–RZ2, AN1,
    AG1–AG2, AR1, AT1–AT3, EV1–EV4, plus AQ-X1, RP-X1, TP-X1, PM-X1, NG-X1, IT-X1, AN-X1 and AT-X1. In
    `papers/SWEEP_READING_REPORTS_2026-10-04b.md`.
  - **Plain-language versions.** `papers/SPEC_CANDIDATES_plain_language_2026-10-04.md` and `…-04b.md`.
  - **X-labelled candidates.** As before, these are for the exploratory narrator line only.
  - **Interactions with earlier proposals.** TP1 should be folded into BB5. PM2 and EV4 extend the MM1 test. EV1 and
    WA5 build on the SV2 answer. AR1's wording must be reconciled with PD4's "never select".
- [ ] **Open owner questions from the 2026-10-04 batches.** Each one blocks the candidate named. The full list is at
  the end of each plain-language file.
  - **Silent mechanisms** (theory decisions):
    - **RA1.** How a move's target is chosen, and whether anxiety shifts targeting toward the sender of the latest
      upset.
    - **PM1.** How long a per-person belief persists between writes. L09.4 may point the opposite way from the memory
      literature.
    - **BV1.** What a "situation", a "decision domain" and a "demand type" are. `M1.A.3`, `M1.A.3a`, `M1.A.14d` and
      the `M1.A.4c` estimator read them, and nothing represents them.
    - **BV3.** What `M4.D.1a`'s mixing weight reads.
    - **AN1.** Whether an event in flight is delivered after the sender dies or the tie is cut off.
    - **RZ1.** Whether belief items are coupled.
    - **RZ2.** Whether confirming writes are stronger than contradicting ones.
    - **WA2.** Whether more fused families react more alike to one event.
    - **EV4.** Whether anxiety weighs more on indirect evidence.
    - **Carry-over.** Whether a withheld move carries over to the next tick.
    - **EN.** Whether any stochastic term may persist across ticks.
  - **Arm and framing decisions:**
    - **NG1.** Whether an arm may separate spouses' levels against `M2.A.0c`.
    - **AT1, AT2.** Whether model "anxiety" is described as felt, and whether the consistency-engine framing becomes
      a requirement.
  - **Corpus item found while checking, not from a paper.** FE05.17 (protecting children from one's own problems
    transmits them) is marked testable in `fe05.md`, but no `M11` criterion covers it.
- [ ] **Owner questions from the 2026-09 maturity-and-emotion conversation.** Added 2026-10-05. These four ideas
  come from Claude's reasoning in `docs/conversation_agents_maturity_emotion_2026-09.md` §4, not from a paper or
  the corpus. Each is `[I]` until corpus support is found. None is a requirement, and each needs an owner decision
  before it can become a proposal.
  - **Goals and shared resources.** Each person holds weighted goals (closeness, achievement, money, autonomy,
    relief), and a family-level ledger of time, money and attention that pursuing those goals uses up. Conflict comes
    from goals competing for the same resources. Check against `M1.A.10` (`life_energy` split between relationship
    and goal-directed activity) before adding anything; it may already be the place for this.
  - **Three cost terms on an act.** An act carries relief now, a cost to the actor later, and a cost to others
    (tie tension, others' goals blocked). Drinking is the example: relief now, cost later. Qualifications are the
    reverse: cost now, payoff later. Check against `M1.A.4g` (substance use as a chronic `functional_level`
    pattern), `M4.D.6a` (reversed at revision 11: the automatic channel now learns from felt relief) and `M7.D`.
  - **An outside world with delayed returns.** School, jobs and legal risk return outcomes after a delay, with some
    chance involved. Without this, "qualifications pay off" can never come out of a run. This would be a new
    exogenous source; check it against `M1.E` and the scripted-event source.
  - **A horizon that changes with age.** The reinforcement horizon is short in adolescence, lengthens through the
    twenties, and nodal events can shift it. At present `M4.D.6b` declares one fixed horizon. This bears on
    `M4.D.1a`'s design decision that a longer horizon cannot reach differentiation: the decision stands, and a
    horizon that changes with age would act on the automatic channel only (`M4.D.6d`). The Redish full read
    (2026-10-05) gives this no support: his discount factors are fixed and never change with age or experience, and
    he states discounting is not the cause of addiction in his model. Any support would have to come from elsewhere.
- [ ] **Method proposals from the 2026-10-05 full reads.** Added 2026-10-05 by owner request; held for revision 11 on
  the same terms as the earlier batches. Three papers that had been catalogued at abstract level since 2026-09-17:
  Redish 2004 (TD1–TD3), Park et al. 2023, Generative Agents (GA1–GA4, GA-X1, GA-X2), Argyle et al. 2022, Out of One,
  Many (OM1–OM3, OM-X1). In `papers/SWEEP_READING_REPORTS_2026-10-05.md`; plain-language version in
  `papers/SPEC_CANDIDATES_plain_language_2026-10-05.md`.
  - **Interactions with earlier proposals.** GA1 adds to PM1. GA2 is close to RZ1 and could be folded into it. TD1
    may make `M4.G.3`'s separate habituation term unnecessary, and shows `M4.D.6a`'s "converges on CUTOFF by
    construction" assumes an additive update the spec never states. *That clause was removed at revision 11, when `M4.D.6a` was reversed; TD1's point about the update form still applies to `M4.D.6`.* OM1 sits next to `M10.C.2a`.
  - **Owner questions.** TD2: is an old pattern held down or erased when it goes quiet? GA1: does belief fading run on
    time since written or time since last used? GA2: does a derived belief outlast its evidence?
  - **Catalogue gap found by the Argyle reader.** The three later silicon-sampling papers (2609.10280, 2609.15849,
    2609.16395) had not been read in full. *Closed 2026-10-05: read in batch b, below.*
- [ ] **Method proposals from the 2026-10-05 batch b (silicon sampling).** Added 2026-10-05 by owner request; held for
  revision 11 on the same terms. Total Simulated Survey Error 2609.10280 (TS1–TS4, TS-X1), Before You Poll 2609.15849
  (BP1–BP5, BP-X1, BP-X2), Silicon Sampling Country Assumptions 2609.16395 (SC1–SC5, SC-X1, SC-X2). In
  `papers/SWEEP_READING_REPORTS_2026-10-05b.md`; plain-language version in
  `papers/SPEC_CANDIDATES_plain_language_2026-10-05b.md`.
  - **Interactions with earlier proposals.** SC4 amends OM2. SC5 could be folded into SC1. BP4 could be folded into
    CB1 or OM1. TS4 extends PW3 and PD4 from filtering and selection to weighting. BP2's inert arm builds on the SV2
    answer. TS2's readout alternatives must exclude the `M11.F.6` traps.
  - **Owner answer: BP2's inert arm (answered 2026-10-05).** Coach quality, one of `M1.E.7d`'s three landing
    conditions, is a declared setting that can be set to null. The inert arm is therefore an ordinary arm, set through
    a declared channel (`M17.D.3`). It is not a model change under `M11.4f`. For revision 11:
    - coach quality is a declared per-agent attribute, and its null value is `[I]`;
    - the inert arm matches the coach arm on timing, contact schedule, tie formation and event intensity, and differs
      only in coach quality;
    - it is added as a fifth control arm, (e), in `M17.D.1`.
  - **Owner answer: BP1's dependency list (answered 2026-10-05).** The list of `M11.C` criteria an intervention result
    depends on is **derived** from `M11.1d`'s minimal rule sets, not declared by hand. For revision 11:
    - a criterion is on the list when its minimal rule set shares a rule with the mechanism chain the result runs
      through;
    - the list is computed before the run and logged with the result in `M17.G.1`;
    - BP1 therefore depends on `M11.1d`'s rule sets being complete for every criterion.
  - **At approval of revision 10 or in revision 11.** The spec's line recording 2609.15849 and 2609.16395 as "read at
    abstract level only" is out of date. Left unchanged while revision 10 is under review.

## Arising from the 2026-08-28 batch

- [ ] **Audit the spec for requirements sourced from an extraction *heading* rather than a quoted sentence.** `M5.D.4` was inverted for exactly this reason: `docs/theory/ch13.md` carried the correct quotation in its body under a heading that said the opposite, and only the heading travelled into the spec. The heading is fixed and the requirement corrected, but **nothing has checked whether there are others of the same shape.** The check is mechanical: for each requirement citing a chapter, confirm the claim appears in a *quoted* passage in the extraction, not only in a section title or a bolded gloss. Highest risk in the earliest requirements, written when the extractions were newest.
- [ ] **Two changes to the family-diagram application, both needing information the diagram does not currently hold.** Neither can be recovered afterwards, which is why `M13.3` names them as the ones to make first.
  - `M15.A.4` — export the **interval** a rater would defend, with the scale point defined, not a bare `3` on a 1–5 scale. The model divides by, differences and thresholds these values; a rank supports none of that.
  - `M15.C.2` — split the diagram's single **"distant"** line into *rupture* and *resolved low-contact*. `M1.B.3` calls telling those apart the most consequential discrimination in this part of the theory: same contact frequency, opposite bond energy. Until the app emits it, bond energy on any distant tie imports as a free range.
### Deferred from the 2026-08-28 re-sweep — graded below medium, backlogged rather than fixed

The re-sweep of the fix commit returned fifteen findings. Four high and seven medium were fixed; these four
were graded low and are recorded instead, because each fix is new unreviewed surface and the project's own
rule is to bound the loop rather than trade one defect class for another.

- [x] ~~**`test_claimed_test_count_matches_the_suite` counts definitions, not collection.**~~ *Fixed 2026-10-06: it now shells to `pytest --collect-only -q`. Phase B step 1's parametrized test was the first case where the counts differed (76 defined, 78 collected).* It walks the AST for
  `def test_*`. `@pytest.mark.parametrize` multiplies collection, `*_test.py` files fall outside the glob, and
  a `skip`/`xfail` keeps the guard green while `CLAUDE.md`'s claim — "all N tests **pass**" — is false. Today
  the three numbers agree exactly (49 defined, 49 collected, 49 passed). Fix by shelling to
  `pytest --collect-only -q`, or state the assumption in the docstring.
- [ ] **`docs/theory/` is outside the reference guard, and has a dangling reference.**
  `family_evaluation/fe03.md` cites an `M0` requirement numbered **one past the last one §0 defines** (§0 runs
  `M0.1`–`M0.4`). `fe03.md` and `fe08.md` also cite `M1.E.7c`'s numbered sub-forms as though they were IDs —
  the spec's own uses of that notation were corrected at revision 5, the extractions were not. Adding the
  directory to `CITING_DOCS` turns the suite red until all three are fixed, so do both together.
  *(The literal tokens are deliberately not written out here: `TODO.md` is inside `CITING_DOCS`, so quoting a
  dangling ID as an example makes the guard fail on the note describing it. The guard is right to be strict —
  an exclusion mechanism would be the loophole that later hides a real one.)*
- [ ] **`_REF` truncates compound references.** `M7.D.2c/2d` yields only `2c`; `M1.D.7i–l` only `7i`. Neither
  is currently dangling, so nothing is wrong today — but the scan is narrower than it reads, and
  `_STATUS.md` describes it without that qualifier.
- [ ] **Name the stressor per criterion when the tests are written.** `M11.3` now places the obligation on the
  test rather than the criterion's prose, which is honest but defers it: 27 of 33 criteria are unclassified
  as discriminating or not. Classify them when each test is written, and consider a marking column then —
  `M11.3` originally required one and the table has no such column.

- [ ] **Sweep the spec for other descriptions firmed up into mechanisms.** Two are now known — `M5.D.4`
  (anger, revision 4) and `M1.A.3` (the transition at 50, revision 7) — and both were caught by reading the
  primary, not the extraction. **Neither was catchable by a regex**, which is why this is a reading task and
  not a guard. The shape to look for: a requirement whose source is a *descriptive* passage, whose hedges
  ("begins", "a few", "tends to", "usually") the requirement dropped, and which the spec then states as a
  threshold, a switch, or an exact count. Highest risk where a single chapter is the only source.
- [ ] **Re-read `KS05.2` on the capacity floor at 25.** `M1.A.4d` was regraded to a continuous falloff at
  revision 7 **on the principle rather than on a reading** — its source says "they lack the flexibility to
  make basic change", which is a description of a group and the same grammatical shape as Ch16's band. The
  falloff is applied; the *location* stands pending the primary. It is the one number `M1.A.3d` admits as
  structural, and it is admitted provisionally.
- [ ] **Re-read the source behind `M5.F.2`'s `outside_ness` threshold.** "Below threshold the same move
  produces the opposite sign" is the same shape as the transition at 50 — a threshold on a continuous
  quantity, doing load-bearing work in the sign-flip argument at `M15.D.4`. It may be sound; it has not been
  checked with this question in mind.

### Backlogged from the corpus-fidelity sweep, 2026-08-28 — graded LOW, recorded not fixed

The sweep checked **every** quoted passage in the spec: 214 total, 160 verbatim in the corpus, 52 legitimate
self-quotes of project documents, 0 unexplained. Integrity and modal findings are fixed; these are the
hedge-level remainder, held back because each fix is new unreviewed surface.

- [ ] **Ellipses that remove a qualifier.** `M1.A.19` drops "**would probably**" and a trailing condition
  ("as long as discussion touched emotional issues") from Ch10's neutral-third sentence. `M1.A.14c` drops
  "**provided his spouse functions in a reciprocal opposite way**", which is a condition on the whole claim.
  `M1.A.5d` splices two passages ~5,700 characters apart into one quotation and drops a "per se, **but…**"
  that states the harmful case. `M11.F.3a` truncates a Bowen epigraph with no ellipsis shown.
- [ ] **Hedges trimmed off the front of quotations.** `M1.A.13a` ("**I believe** the most important cue"),
  `M1.D.7b` ("**I believe** a spectrum of problems"), `M1.A.18d` ("not observed it **fully**"),
  `M8.6a` ("**as many as** fifty or sixty" flattened to "fifty to sixty").
- [ ] **Third parties quoted by an author, unmarked.** `M7.C.1c` rests on **Sylvia Nasar's** thought, quoted
  by Kerr. (`M4.D.5c`'s Andrew Solomon attribution is fixed; this one is the same shape.)
- [ ] **A scope narrower than the requirement.** `M5.B.3a`'s band is stated of "**the alcoholic person**";
  `M1.B.10` quotes a taboo-set claim made of **spouses** and applies it to every tie; `M15.D.4`'s "they could
  do no wrong" comes from the inpatient schizophrenia project, the most severe families studied.
- [ ] **`M4.D.3a`'s four-element evolutionary ordering has three sourced slots and one unsourced** —
  conflict's position between cooperation and dominant-adaptive is nowhere given in the source.
- [ ] **`M1.A.5a` quotes a phrase absent from the book it cites** ("more mature, more responsible for self",
  cited to Kerr 2019 ch6; nearest analogue is in ch24 and says something different). Its decomposition is
  tagged as user-derived, so the model claim survives — the citation does not.
- [ ] **`M2.A.0e`'s "almost identical" reads as Kerr-2019-grounded** and is in fact from KB Interview #14.
  Kerr 2019 says spouses marry at the "**same**" basic levels — stricter than the ±1 tolerance built on it.
- [ ] **`M5.C.1a` borrows severity from an adjacent intervention class.** The three catastrophes are real and
  correctly sourced to *forcing conjoint contact*; the recorded consequence of the peace-agree/reactive
  mismatch itself is relational rupture. Careful drafting, suggestive juxtaposition.
- [ ] **`M1.A.4d`'s source sentence is gate-shaped.** Kerr 2019: "**People above 25** on the continuum can
  make basic changes." Revision 7 replaced the gate with a continuous falloff on principle; the requirement
  should say plainly that the source's own wording is the gate it forbids.

- [ ] **Run a cold sweep over this batch.** The 2026-08-28 sweep was warm — the same session that wrote the documents commissioned it. It found fifteen defects, which says the sweep works, not that the set is clean. The project's own rule is that a falling finding count from self-review is not a stopping condition, and that a pass which does not know the change's history changes the *distribution* of findings. `/csdp --cold-sweep` over `4e79697..HEAD` from a fresh session.
- [ ] **The sweep covered one failure family only.** It reports itself as not covering: theoretical fidelity (it did not open `_LEDGER.md`, the `family_evaluation/` extractions, or Ch13's primary text), whether the ~88 new requirements are individually implementable, whether the specified model is coherent, and the reasoning in the DECISIONS document. **The `M5.D.4` correction rests on one re-read of one chapter by the context that made the correction** — a second reader on the sources is the highest-value follow-up.
- [ ] **Decide whether `M16`'s renderer output format is worth fixing in the spec.** `M16.C.2` says what a rendered line must carry and points at the proposal's §4.2 table as the shape, but does not fix a format. That is deliberate for now — the format is cheaper to settle against a running Phase B than in advance — but it should be settled *before* a second consumer exists, or the two will drift.
- [ ] **`M11.F.9(c)` deserves a worked demonstration before anyone relies on it.** The claim is that fitting one arm of a counterfactual to a known history breaks the error cancellation `M0.4` depends on. It is argued, not demonstrated. Phase E could show it directly: take a synthetic family, fit one arm to its own history, and measure how far the counterfactual moves against an unfitted control. If the effect is small the requirement is over-strong; if it is large it is the most important sentence in `M11.F`.

## Known defects in the frozen grid engine

Recorded, not fixed. The engine is frozen in behaviour; these matter because the v2 spec forbids inheriting them.

- [ ] **Engine purity violated.** *(Partially contained 2026-08-22: the test suite now runs from a temp directory, so it no longer writes into the repo root. The engine still writes.)*  `Simulator.__init__` opens and writes `sim_audit.csv` in the process working directory, and `log_telemetry` appends every 10 cycles. `CLAUDE.md` says the engine stays pure — no UI, no file I/O. Now gitignored so the stray copies stop appearing as untracked, but the write itself remains.
- [ ] **`_apply_config` fails silently.** It skips any line failing its regex and any key not already in `defaults`, with no warning, so a typo in `docs/model_config.md` vanishes and the default is used. Spec `M10.B.2` forbids v2 inheriting this; the v1 engine still has it.
- [ ] **Magic literals.** Spec §14 of the frozen document claims no model parameters are hard-coded in engine source. In practice 12 keys are in markdown config, 27 are class constants, and roughly 52 distinct float literals remain inline in executable code.
- [ ] **`family_ids` and `nuclear_family_id` are the same array object**, not a copy — `sim.family_ids is sim.nuclear_family_id` is `True`. The frozen spec documents `family_ids` as a "compatibility alias", so this is intentional, but two consequences are not obviously intended and are worth checking before any further engine work: in `update_triangles` the pair `self.family_ids[circle_idx] = fid` / `self.nuclear_family_id[circle_idx] = fid` writes the same element twice, so the first line is dead; and the triangle release path at lines 324/327 restores `family_ids` and thereby silently rewrites `nuclear_family_id` too. Any future change that gives the two names different values will silently lose one. **The v2 spec must not carry an alias pair like this** — `M1.A` gives each field one name.
- [ ] **`update_m` windfall can target a dead unit.** It picks `np.random.randint(0, num_units)`, so it can multiply the resources of a dead unit or an unused nursery slot.

## Repository hygiene

- [ ] **Untracked files predating this batch need a decision.** Not staged, because they were not touched in this work: `requirements.txt`, `monte_carlo.py`, `monte_carlo_results.txt`, `data/`, `gemini.md`, `docs/background.md`, `docs/constitution.md`, `docs/logic_rules.md`, `docs/spec.md`, `docs/user_stories.md`. **`requirements.txt` is the urgent one** — the README tells people to install dependencies and the file is not in the repo.
- [x] ~~**No CI workflow.**~~ *Out of date: `.github/workflows/tests.yml` exists (see the rca entry under revision 30).* The project standard is that a repo with a test suite pushed to GitHub gets `.github/workflows/tests.yml` — checkout, install from the real dependency file, run the suite on each supported Python version. The value is the blank machine: it catches what "works on this Mac" hides, which matters more here because the repo is shared between two Macs. Blocked on `requirements.txt` being tracked.
- [ ] **No `LEARNINGS.md`.** The project convention is a repo-local fix log for repo-specific lessons, with generic patterns going to `~/.claude/standards/learnings.md`. Candidates: the ID-drift incident recorded in `M11.D.10`; the two unseeded flaky tests; and from 2026-08-28, **the `M5.D.4` heading inversion** — a requirement sourced from a summary heading rather than the sentence beneath it, which inverted silently and survived four spec revisions. The last of those may be general enough for the global catalogue rather than the repo-local log.
- [ ] Stray file `docs/model_explainer copy.textClipping` in the working tree — not mine to delete, but it is clutter.

## Documentation hygiene, deferred

- [ ] **`§n` cross-references are ambiguous across documents.** All 50 in the v2 spec resolve against `model_explainer.md`, but several sentences address other documents ("proposal §9", "the frozen spec's §5.4") and `§5.4` exists in both. A traceability scan over `§` cannot tell the targets apart. Prefix them with the document.

## Sources

- [ ] **Locate the rest of the Kerr–Bowen interview series.** The folder holds #1 of a series; Kerr closes by saying later tapes "will concentrate on some of the more specific areas of the theory". Bowen names two he wanted covered: the difference between *distance* and *differentiation*, and how much of behaviour is intellectually directed versus emotionally reactive. An interview format with Kerr probing specific concepts is the highest-value form this material could take.
- [ ] **Probe the remaining 11 validation items against the 1979 lectures.** Tape 6 especially — it is the direct counterpart to Ch15, where this project withdrew both the seven-rung severity ladder and the "hidden dependence network", and a whole lecture on family reaction to death is the natural place to test both withdrawals.

## Carried from the theory work

- [ ] **`M11.C.9`'s and sibling position's status generally.** Ch13 omits sibling position entirely, and its effect size has no source. Consider whether it earns a place in the model at all.
- [ ] **Convergence C1 needs its scope re-derived from the chapter texts**, not from the pass-1 summaries. It was the headline finding at eleven chapters; Ch22 supplies a direct counterexample, so it likely survives only as a claim about *symptom relief obtained without structural change*.
- [ ] **The ego-mass terminology arc must be re-derived.** Ch08 actively retains and defends "undifferentiated family ego mass" against the timeline's claim that it was discarded at Ch05. The arc is messier than used → discarded → revived → abandoned.
- [ ] **When the basic level is fixed is genuinely unresolved** (`model_explainer.md` §13.3). Decided for the model — slow-moving with a ratchet, not frozen — but that is a modelling decision over a real disagreement, not a resolution of it.
