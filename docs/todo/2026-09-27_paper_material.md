# Held for a later revision — paper material, 2026-09-27 (labelled "revision 11" until 2026-10-07)

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

- [ ] T-081 ⟨later⟩ **Method proposals from the two 2026-09-27 batches.** 36 candidates. First batch: J1–J4, SV1–SV4, PD1–PD4,
  X5–X9, in `papers/SWEEP_READING_REPORTS_2026-09-27.md`. Second batch: BB1–BB5, MM1–MM4, AD1–AD3, TM1–TM6, X10,
  in `papers/SWEEP_READING_REPORTS_2026-09-27b.md`. Plain-language versions with examples are in
  `papers/SPEC_CANDIDATES_plain_language_2026-09-27.md` and `…-27b.md`. Candidates labelled X are for the
  exploratory narrator line only and do not enter the v2 spec.
- [ ] T-082 ⟨later⟩ **Owner answer: anxiety shapes beliefs (answers MM1).** Anxiety makes a person's beliefs about others more
  subjective. It also biases risk assessment upward, so the same event is read as more threatening. For revision 11:
  - belief writes (`M9.8`, and the per-person belief store) read the receiver's own anxiety, not only the events
    delivered;
  - the weight on the receiver's own state rises with anxiety;
  - the threat bias points one way, upward;
  - direction graded `[user]`, strength graded `[I]`;
  - check `M4.C.5` and FE05.10 for corpus support before raising the grade;
  - `M9.2` still binds: a threat-biased reading is not automatically false.

  MM1's probe (identical events, only the receiver's anxiety differs, plus a label-only null arm) becomes the test.
- [ ] T-083 ⟨later⟩ **Owner answer: how coaching enters a family (answers SV2).**
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
- [ ] T-084 ⟨later⟩ **Method proposals from the two 2026-10-04 batches.** Added 2026-10-04 by owner decision; held for revision 11
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
- [ ] T-085 ⟨later⟩ **Open owner questions from the 2026-10-04 batches.** Each one blocks the candidate named. The full list is at
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
- [ ] T-086 ⟨later⟩ **Owner questions from the 2026-09 maturity-and-emotion conversation.** Added 2026-10-05. These four ideas
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
- [ ] T-087 ⟨later⟩ **Method proposals from the 2026-10-05 full reads.** Added 2026-10-05 by owner request; held for revision 11 on
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
- [ ] T-088 ⟨later⟩ **Method proposals from the 2026-10-05 batch b (silicon sampling).** Added 2026-10-05 by owner request; held for
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
- [ ] T-089 ⟨later⟩ **Method proposals from the 2026-10-12 batch.** Added 2026-10-17 by owner request; held for revision 11 on the
  same terms as the earlier batches. Six papers read in full from the 2026-10-12 method-literature digest: Fu 2026,
  *"Hot-Blooded" vs "Cold-Blooded"* (2610.11951; HB1, HB2), Peng et al. 2026, *MASC* (2610.08250; MC1, MC2), Deng et
  al. 2026, *SPIN — Cognitive-Affective Personality Inference* (2610.08328; SP1, SP2, SP3), Wang et al. 2026,
  *Socio-Foundation* (2610.08967; SF1, SF2), Wang et al. 2026, *Prompts versus Rules* (2610.11015; PR1, PR2, PR3,
  PR4), and Fraile Navarro 2026, *Latent Identity Reversion* (2610.01490; IR1, IR2, IR3, narrator/LLM line only). The
  four deep-review candidates flagged in `DIGEST.md` were Hot-Blooded, MASC, SPIN and Socio-Foundation (PAIR 2610.07523
  was adjacent and not read in full); *Prompts versus Rules* and *Latent Identity Reversion* were surfaced by screening
  the wider digest. In `papers/SWEEP_READING_REPORTS_2026-10-12.md`; plain-language version in
  `papers/SPEC_CANDIDATES_plain_language_2026-10-12.md`.
  - **Scope note.** Unlike the earlier batches, none of the sixteen candidates is a theory change — all are
    inference-validity, reporting, or mutant-convention items for Phase E (`M17`). HB1/HB2 (per-readout reliability and
    the over-discreteness flag), MC1/MC2 (scripted-trajectory control arm; readouts must not reuse decision-path
    internals), SP1/SP2/SP3 (complexity-matched mutants; state-bypass collapse signature; tiny-N paired checks state
    the minimum p-value), SF1/SF2 (report dispersion as SD with extrema, never SE/CI; normalization bounds fixed and
    declared), PR1/PR2/PR3/PR4 (per-opportunity move rates; monotonicity on adjacent pairs with tie fraction; bound
    non-linearity to the input distribution; dispersion readout for artificial uniformity).
  - **Narrator-line items (do not enter the v2 spec).** IR1 (per-turn persona re-anchoring), IR2
    (represented-vs-enacted identity probe), IR3 (log the scaffold state) — same treatment as the X-series.
  - **Owner questions.** None about theory. The only owner decisions are adoption ones: whether to fold these into
    Phase E (`M17`) reporting/mutant conventions, and whether IR1–IR3 join the existing X-series protocol items.
