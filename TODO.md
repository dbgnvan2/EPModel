# TODO

Deferred and adjacent items. Each carries enough context to act on later without the conversation that produced it.

## Blocking the next phase

- [x] ~~**Approve or revise `docs/bowen_agent_model_spec_v2.md`.**~~ *Approved 2026-08-25; revisions 1–6 applied, the last on 2026-08-28.*
- [x] ~~**Approve spec revision 10.**~~ *Approved 2026-10-06. Terms in the spec's revision-10 section under* Approval.
- [ ] **Read the Phase B trace end to end and record the result** (plan §7) — the last item before Phase B is declared done. `docs/review/phase_b_trace_seed7.md` and `docs/review/phase_b_trace_seed7_nadia.md`; report at `docs/phase_b_completion_report.md`.
- [ ] **Run the `learning-qa` review over the Phase B diff before pushing** (global working process, step 5).
- [x] ~~**Build Phase B.**~~ *Steps 0–15 done 2026-10-06; all 14 automated exit criteria pass, 17 of 17 mutations proved.*
- [x] ~~**Approve the Phase B implementation plan**~~ *Approved 2026-10-06, D1–D9 as written.* — drafted 2026-10-06 at `docs/implementation_plan_phase_b.md` (branch `plan-phase-b`). Its §1 holds nine decisions (D1–D9) the owner must answer before code, and Appendix A is the `M14.A` register draft for review. *Was: write the implementation plan — Phase B only* (owner decision 2026-10-06, following the external review's recommendation 9). The 2026-09-23 review's findings go in the plan as blockers on the phases they affect. The next gate, and the only thing between here and code. Every acceptance criterion mapped to the file and module that satisfies it, in dependency order, with the mutation that proves each one. Spec convention is spec → plan → build, each approved before the next. Inputs: the 33 criteria in `M11.C`, the 5 in `M16.F`, the engineering criteria in `M11.D`, and the phase table at `M13`.
- [x] ~~**Decide four criteria that cannot be made code-testable in Phases B–D** (spec §M11.E).~~ *Accepted as written by the owner, 2026-10-06, at approval of revision 10.* Each has a human-review proposal:
  - `M11.C.8` (endogenous incidence vs published rates) — needs Phase E and an editorial call on which sources are genuinely exogenous.
  - `M11.C.9` (sibling position) — the effect size is invented, so the test can only assert "a detectable difference", which is close to unfalsifiable. Decide whether it belongs in the suite or should be demoted to a readout.
  - `M11.C.11` shape — "three phases, not a step down" is a claim about curve shape; any automated version embeds an invented tolerance.
  - `M5.F.2` — the threshold separating a counterfeit move from a genuine one is invented and sets the result. **This is the model's most consequential invented constant.**

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
  fail. C.42 is +0.26 (p 0.084) and its sign depends on the learner's constants; C.45 is +0.0020 per person-week
  (p 0.20) and reverses at 4 of 6 sweep settings. What remains is whether the learner should produce these
  effects at all, which is a model question, not a test question.* Both were already fragile in the sweep.
  C.42's reuse is −0.17 and it fails at every sweep setting; C.45 has the right sign, without significance.
  Decide whether either criterion's arms still test what it claims, now that the act loads the recruited third.
  Report §3.8 and §8.
- [ ] **`M13.4`'s precision rule and `M11.C.16`.** The direction holds at p ≈ 1e-44, but the half-width never falls
  below 0.25 of the baseline arm's sd, so it is `UNDETERMINED` at 500 seeds. Decide whether precision should
  scale by the paired difference's sd. Report §3.4.
- [ ] **The failing criteria** — `M11.C.4` (later limb), `.5`, `.29` (the third person never accumulates symptom
  weeks), `.41` (no cell passes; the reactive share falls with level), `.42`, `.44`, `.45`, and three cells of `.27`.
  Report §3.
- [ ] **Plan §9's human reviews:** `docs/readouts_phase_c.md`, a rendered 104-week Phase C trace, and `M5.F.2`'s
  threshold.
- [ ] **Not built:** `M11.C.7` (needs `M8.2`/`M8.3`'s predicates and a direction), `.13` (needs a community),
  `.14` (needs `M5.C.1`'s marital-distance gate); two of `M11.1c`'s four re-encodings (rescaled state range,
  integer against float ticks); `M11.1b`'s severing mutants; plan D9's "dominant constant" (§3 names none).
- [ ] **Bruno falls back every week.** His one tie starts cut off, `REDUCE_CUTOFF` is his only legal act, and it
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
- [ ] **No CI workflow.** The project standard is that a repo with a test suite pushed to GitHub gets `.github/workflows/tests.yml` — checkout, install from the real dependency file, run the suite on each supported Python version. The value is the blank machine: it catches what "works on this Mac" hides, which matters more here because the repo is shared between two Macs. Blocked on `requirements.txt` being tracked.
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
