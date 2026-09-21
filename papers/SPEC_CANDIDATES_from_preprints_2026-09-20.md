# Candidate additions to the EPModel specification, from the fourteen full-read preprints

Date: 2026-09-20. Companion files: `SWEEP_READING_REPORTS_2026-09-20.md` (the per-paper reading reports these candidates are drawn from), `PREPRINT_SWEEP_2025-09_to_2026-09.md` (the catalogue), `DESIGN_LESSONS_model_design_papers_2026-09-17.md` (the earlier corpus of 29 papers).

## 0. Method and how to read this file

The fourteen PDFs in `Model Design papers/Sweep 2026-09/` were converted to text and read in full by five readers working from `EPMODEL_BRIEF.md`, `DESIGN_LESSONS` and a written task (report the paper; then propose 0–6 spec additions, each with a requirement in the spec's own style, a target module, the evidence and its strength, what it changes, and cost/risk). The readers had the brief and the design lessons, not the 427 requirements, so every "already covered" judgement they made has been re-checked here against `docs/bowen_agent_model_spec_v2.md` (1,677 lines) by text search. The five reports are compiled unaltered, apart from paraphrasing of paper quotations, in the companion reports file; this file is the consolidation.

Consolidation did three things: merged candidates that several readers proposed independently (noted per entry), assigned each a coverage status against the spec, and assigned a priority. Nothing here was added from memory of the papers; every candidate traces to a section of a reader report, which cites a section, table or figure of the paper.

Evidence grades, as the readers used them: **SHOWN** means an experimental or formal result in the paper; **ARGUED** means a position or reasoning in the paper; **INFERENCE** means the reader's or my own transfer to EPModel. In most entries the paper shows or argues something about its own system and the transfer to EPModel is inference; both are stated.

Coverage status, from the spec search:

- **NEW** — nothing in spec v2 addresses it.
- **PARTLY** — the spec has a neighbouring rule; the entry says what is missing.
- **COVERED** — the spec already has it; listed only so the reader can see it was checked.

Priority, my judgement:

- **P1** — design-time decisions that are cheap before Phase B and expensive to retrofit (identifiers, RNG, named engine objects, information-access rule, death disposition).
- **P2** — test, logging and register rules that belong in M11/M16 during Phases B–D.
- **P3** — Phase E content (ensemble runner, arms, readouts); Phase E is deliberately unspecified in v2, so these are inputs to writing it.
- **P4** — exploratory LLM line only; no change to v2.

What "realistic" can mean here, given the project's epistemic position (no data, no magnitudes): the candidates split into those that improve **theory fidelity** (mechanisms faithful to how interaction is structured: witnesses, information access, death, habituation, ordering) and those that improve **inference validity** (a reported direction is a property of the mechanism, not of the seed alignment, the update order, a fallback rule, a rounding artefact, or a constant tuned against the test). Each entry says which.

## 1. Summary of recommendations

Seven candidates change the design and should be settled before the implementation plan is written, because each is a foundation the code will be built on:

1. **C1 — counter-based, event-keyed random draws** (Buffalo et al.; also Holland's CRN practice and Li & Tao's per-object streams). This **corrects M3.D.4**, which specifies a single seeded generator threaded explicitly. A single stateful generator makes the per-seed paired arm difference — the project's trusted output — partly a measurement of noise re-alignment whenever an arm changes control flow, which any mechanism-deletion arm does. Formal result in the paper; strongest single finding of the sweep.
2. **C2 — stable person, tie and triangle identifiers across arms** (prerequisite for C1; births and deaths differ between arms over 40 years).
3. **C6 — activation and visibility as named, versioned engine objects** (Li & Tao), with the current "all persons every fast tick" recorded as the chosen regime rather than left implicit.
4. **C9 — information-access rule with a true-state mutant** (VISA r14, PIMMUR's god-view objection). The spec makes belief a channel (M9.7) but nowhere forbids a policy from reading another person's true state or an undelivered event, and no test would catch it.
5. **C15 — disposition of a Person at death** (VISA r8). Mortality is on the slow tick (M3.B.1, M7.C.1) and M11.C.15 tests a death, but no requirement says where a dead person's anxiety, bond energy, functioning-balance debts, triangle positions and budget share go. Without it the conservation invariants (M6.I.6, M6.I.7) are not checkable across a death.
6. **C11 — legality mask computed and logged before selection** (Buitrago López). Gates exist (33 mentions of "gate") but there is no requirement that the legal set be formed, renormalised and logged, so a never-legal move and a never-chosen move are indistinguishable in the log.
7. **C12 — fallback and tie-break provenance per decision** (TRAILS). The spec has no tie-break or fallback rule at all (0 hits); softmax selection (M4.D.1) will need one for equal scores, NaN and exhausted life energy, and a silent fallback can generate the tested behaviour.

Beyond those, the sweep supplies a set of test-design rules for M11 (monotonicity over four or more levels, matched-magnitude contingency-severing mutants, ordinal criteria, frequency-controlled asymmetry tests, representation-invariance mutants, an outcome-directive audit with a premise/mechanism register, constants frozen before tests) and a set of Phase E rules (adaptive ensemble size with UNDETERMINED, per-seed principal-strata readout, regime classification with an inconclusive class, ordering-spread and D0-variance components, structural totals held fixed across arms, three named control arms, an audit record with an "unaudited" column). Full list in §3.

## 2. Design-level candidates (P1)

### C1. Counter-based, event-keyed random draws; no stateful generator in the engine

**Proposed requirement.** Every stochastic draw in the engine MUST be computed as a pure function of the run seed and a canonical event key by a counter-based generator (Philox or Threefry); the engine MUST NOT hold a mutable generator state. Distribution sampling MUST consume a fixed number of keyed uniforms (inverse transform), never a variable number (rejection sampling). Keys MUST contain only structural identity — tick, stable object identifiers (C2), a purpose label, and a within-event index — never a state quantity or an endogenous summary.

**Module.** M3, replacing the second sentence of M3.D.4. M3.D.5 (byte-identical logs) is unchanged and still holds.

**Source and evidence.** Buffalo, Pearson & Klein 2026 (2603.11084), §3.3 Proposition 1 and Corollary 3.1: seed-matched runs with a stateful PRNG fail to produce a valid coupling whenever an arm alters the execution path, because the draw index of every later event shifts (SHOWN, formal). §2.1: rejection sampling breaks alignment for the same reason. §4.1: counter-based generator keyed by a stable event identity is the remedy (ARGUED). Appendix A: consequences of violation include the seed-pair covariance turning negative (so pairing can increase variance), one-at-a-time sensitivity sweeps and Sobol indices acquiring spurious variance, and mediation-style arms being contaminated. Independent support: Holland et al. 2026 (2607.29546) §5.2 ran their 400-cell regime map on common random numbers (fixed initial state, pairings and valences across all cells) — the practice, without the mechanism; Li & Tao 2026 (2603.00113) Action 2 asks for conclusions stable across nominally irrelevant implementation choices (ARGUED). Merges reader candidates B1.1, H2 and L2(b).

**Status.** NEW, and a **correction**: M3.D.4 as written ("a single seeded generator threaded explicitly") is the design the paper shows to be insufficient for arm comparison. Note that within one arm the current design is a valid probabilistic model; the defect is only in across-arm coupling, which is exactly where EPModel's trusted output lives (M0.4, M11 two-arm criteria, M15.D envelopes).

**What it changes.** Inference validity. Makes per-seed paired differences a comparison of the same chance events under two mechanisms. Also makes the constant sweep (design lessons §2.3) and any variance decomposition well-posed, and removes RNG state from snapshot-and-fork (design lessons Q6): a fork is a pure state copy.

**Cost and risk.** Low to moderate. NumPy ships Philox; a wrapper `draw(seed, key)` is the whole interface. Risk: the hash that turns a key into a counter MUST be a fixed, documented function — Python's built-in `hash()` on tuples is salted per process and would break M3.D.5. Risk: keys that are too fine approach independent seeds and lose the pairing (§5 of the paper); C3 addresses this.

### C2. Stable identifiers for persons, ties and triangles across arms

**Proposed requirement.** Person identifiers MUST be stable across counterfactual arms: founders receive fixed identifiers from the family definition or import (M2, M15); a person born during a run receives an identifier derived from the parents' identifiers and birth order; identifiers MUST NOT be allocated from a run-time counter. A tie identifier is the unordered pair of person identifiers; a triangle identifier the sorted triple.

**Module.** M1 (objects), M2 (reference family), M15 (import).

**Source and evidence.** Buffalo et al. §4.3, third principle: a global mutable ID counter has the same defect as a stateful PRNG — a death in one arm shifts every later identifier in the other (ARGUED, with a worked example; the paper notes birth timing may be a better key in some models). Reader candidate B1.2.

**Status.** NEW (the spec has 16 mentions of identifiers, all about requirement IDs and log-instance IDs, none about person identity across arms).

**What it changes.** Inference validity; prerequisite for C1; also lets the M16 trace renderer align two arms person by person.

**Cost and risk.** Trivial at design time; painful to retrofit. No conflicts.

### C3. Declared key composition for every draw class, with the slot/dyad choice stated

**Proposed requirement.** The spec MUST list every stochastic draw class in the engine (softmax selection, any mixing-weight noise, per-hop fidelity, per-edge latency, witness overhearing where stochastic, exogenous spell onset and duration, mortality, symptom onset, any tie-break) and for each MUST state its key, and whether the key is slot-keyed (tick, actor, purpose, index — partner identity enters only through state) or dyad-keyed (tick, actor, partner, purpose, index — a different partner is a different chance event). Deliberately coarse keys MUST be marked as such.

**Module.** M3 (a table), referenced by Phase E.

**Source and evidence.** Buffalo et al. §4.2, equations 8–11: defining the key is the same decision as defining what counts as "the same event" across two worlds; slot-keying assumes partners are exchangeable given modelled state, dyad-keying allows partner-specific residual chance; a stateful PRNG silently sets event identity to draw index (ARGUED). §4.3: omitting the tick from a key collapses repeated trials into one lifetime draw; each key sampled once and cached; never put an endogenous summary in a key. Reader candidate B1.3.

**Reader's transfer (INFERENCE, stated as a proposal not a rule).** Move selection slot-keyed on (tick, actor), so that "what A does this week" is the coupled event even when the target differs between arms; target-specific draws (fidelity per hop, receiver-side appraisal noise, witness overhearing) dyad-keyed, because a TRIANGLE move picking a different third party is exactly the paper's worker-j-versus-k case; exogenous spells keyed by (family, spell class, occurrence index) so that an arm with higher hazard can bring the k-th spell forward but not reshuffle the sequence; symptom onset keyed by (tick, person, channel) so that "more load, same or earlier symptom" is a per-seed monotone coupling — the form a directional acceptance test wants.

**Status.** NEW.

**What it changes.** A design and documentation rule; inference validity. It makes explicit an assumption that a stateful generator hides.

**Cost and risk.** Documentation plus wrapper discipline.

### C4. Placebo-arm execution-invariance test and single-query caching

**Proposed requirement.** M11 MUST include a placebo test: an arm that enables an extra mechanism at zero magnitude (extra draws, no state change) MUST reproduce the baseline trajectory byte for byte on every seed. The engine MUST query each event key at most once per run and cache the value; a debug build MUST assert on a repeated key.

**Module.** M11 (M11.D series, next to M11.D.5), M3.

**Source and evidence.** Buffalo et al. §2.3 (a mechanistically identical arm with extra draws that diverges proves a coupling violation; identical outcomes do not prove its absence) and §4.3 (sample once, cache, assert on duplicates) (ARGUED). Reader candidate B1.4.

**Status.** NEW. Note that M11.D.5 tests same-seed reproducibility of one arm; it cannot detect this defect, which only appears between arms. Design lessons §7.10(a) proposed the same bit-identical-disabled-arm test with per-mechanism substreams as the remedy; Buffalo et al. §1 classify substreams as a coarse mitigation, so the test stands and the remedy is C1.

**What it changes.** Adds a cheap test that fails loudly under any stateful-generator regression and makes the existing mutation discipline (delete the mechanism, confirm red) measure the mechanism rather than noise re-alignment.

**Cost and risk.** Trivial. Passing does not prove invariance (the paper says so).

### C5. Order-permutation invariance test (operationalises M1.F.8)

**Proposed requirement.** The acceptance suite MUST include a test that, for a fixed seed, permutes the iteration order of persons in the select step and the delivery order of events within every same-tick batch, and asserts a byte-identical final state and an order-normalised identical log. The test MUST be proved failing by replacing the commutative batch reduction with a sequential one.

**Module.** M11 (M11.D series).

**Source and evidence.** Li & Tao Action 2(3) (ARGUED: comparative conclusions must survive nominally irrelevant implementation choices); Sachdeva & van Nuenen 2025 (2510.10002) §3.3 and Fig. 2c (SHOWN in LLM debates: with everything else fixed, first-round consensus was about 40% under one speaking order and nearly 90% under the other; in three-way debates one model steered over 70% of verdicts to one answer in one ordering and the effect vanished in another). Reader candidate L2(a); the reader's point (b), per-object streams, is absorbed by C1.

**Status.** PARTLY. M1.F.8 requires batching so order cannot decide the outcome, and M16 notes that dictionary iteration and set traversal must not introduce nondeterminism; there is no test that order does not decide the outcome. Design lessons §7.10(b) already proposes the symmetric-family permutation-equivariance form of this test (Q14); C5 adds the batch-order permutation and the sequential-reduction mutation. The reader's implementation trap is real: a commutative reduction does not save M1.F.8 if a stateful generator is consumed in loop order — C1 removes that trap.

**What it changes.** Adds a test; inference validity.

**Cost and risk.** Low.

### C6. Activation and visibility as named, versioned engine objects

**Proposed requirement.** The engine MUST implement activation (which persons select a move in a given fast tick) and visibility (which persons are targets or witnesses of an event, with what latency and fidelity) as two separately named components with a declared interface; the run-log header (M16.A) MUST record the identity and version of each. The default activation, all persons every fast tick, MUST be recorded as such.

**Module.** M3 (clocks/ordering), M1.F (events), M16.A (log header).

**Source and evidence.** Li & Tao Definition 4.1: a social simulation is well defined only once it names a scheduler (who may act when) and a visibility object (who sees what) as parts of the model, and Action 1 asks that they be versioned, inspectable artefacts with logged traces (ARGUED). Sachdeva shows why order is model content (above). Buitrago López et al. 2026 (2606.12369) §3.2 shows a policy that is inspectable (a finite-state reference) against ones that are not. Reader candidate L1; the cross-paper note in the g2 report.

**Status.** PARTLY. M3.D.1 fixes the update order and every person acts every fast tick; visibility is spread across M1.F.1 (witnesses field), M1.F.3/M1.F.4 (route, fidelity), M3.C.1 (per-edge latency) and M8.5/M8.6 (witness versus new peripheral triangle). Neither is a named object with a version in the log header.

**What it changes.** Software structure and reporting; inference validity. Makes the design-lessons §2.1 alternative-regime experiment a component swap rather than a rewrite, and lets an auditor reconstruct who could act and who could see from the log alone.

**Cost and risk.** Low at design time; none of M3.D.4–6 or M16.B is touched.

### C7. Witness set produced by the visibility rule, not chosen by the sender

**Proposed requirement.** The witnesses of an Event MUST be computed by the visibility component (C6) from tie and household state (co-residence, tie conductance, route), and MUST NOT be a free field set by the sender's policy. Where the theory needs a sender to choose an audience, that choice MUST be expressed as a move (TRIANGLE addressed to a third party), not as witness selection.

**Module.** M1.F, M4.E.

**Source and evidence.** Li & Tao Definition 4.1 (visibility is a function of state) (ARGUED); transfer is INFERENCE. Reader candidate L4, flagged by the reader as "check spec".

**Status.** PARTLY. M1.F.1 carries `witnesses` as an Event field and M4.E.1 says a selected move becomes an event with witnesses, but nothing says who computes the field. M8.5 gives one predicate that decides whether a third party is a witness or a new peripheral triangle (M8.6), which is the right kind of rule; the requirement would be that this predicate, not the sender, fills the field.

**What it changes.** Theory fidelity: who overhears is a property of the household, not of the speaker's intent. Possibly already the intended implementation; the requirement costs one sentence and prevents the shortcut.

**Cost and risk.** Low.

### C8. Witness appraisal computed from the witness's own state and its ties to both parties

**Proposed requirement.** A witness's appraisal of an Event MUST be computed from the witness's own state, the witness's ties to both the sender and each target, and the intensity of the exchange between them, with a witness-specific weighting constant labelled [I]; it MUST NOT be a fidelity-scaled copy of a target's appraisal. An M11 mutant that replaces witness appraisal with a scaled copy of the target's appraisal MUST go red on a triangle-formation criterion.

**Module.** M4.C (appraisal), M1.F.5, M11.

**Source and evidence.** Holland et al. 2026 (2607.29546), equations 4–5: the witness rule uses its own constant (retribution, distinct from the participants' reciprocity constants), the mean shock over both parties, and the mean of three pairwise distances including the parties' distance from each other; §4.2 and §5 show this channel alone suffices for oscillation and polarisation (SHOWN formally in their model); transfer is INFERENCE. Reader candidate H1.

**Status.** PARTLY. M1.F.5 requires witnesses to appraise; M4.C.1 gives one appraisal formula (intensity × conductance ÷ functional_level, modulated by route, source position and fidelity) without saying which tie's conductance applies for a witness; M4.C.7 makes the witnessed form less reactive than the addressed form. Nothing ties a witness's response to its ties to **both** parties. In Bowen's account the third party's involvement depends on its ties to both members of the anxious dyad — that is what makes a triangle rather than a diluted dyad — so this is the one place the sweep adds a mechanism rather than a test.

**What it changes.** Theory fidelity; adds one appraisal path and one [I] constant.

**Cost and risk.** Moderate. Must respect per-hop fidelity and M3.D.6. Caution from the reader: the paper's pull to ±1 as absorbing sinks is a functional-form property and must not be imported with the witness rule.

### C9. Information-access rule as a testable requirement (no god-view)

**Proposed requirement.** The policy (M4) MUST compute a Person's move from only (a) that Person's own state, (b) that Person's beliefs, including its belief about any tie it is not party to, and (c) Events delivered to that Person's inbox as target or witness after per-hop fidelity; it MUST NOT read another Person's true state, the true state of a tie it is not party to, or an undelivered Event. The log MUST record the belief value used so the renderer can show believed versus true tie state. An acceptance test MUST include a mutant in which the policy reads true state in place of belief and MUST go red on at least one belief-layer criterion (M11.C.26 is a candidate).

**Module.** M4 (inputs), M1 (per-person belief about ties), M9, M11, M16.

**Source and evidence.** He 2026 (VISA, 2607.28027) consistency rule r14 (a function's decision basis must be its own attributes or authorised observations) and the generated sensing layer with a strict access-control mode (§3.3.3, App. D.3.7) (ARGUED). Zhou et al. 2026 (PIMMUR, 2509.18052) §2.3.2 and §2.4: when three LLM agents had to infer other dyads' relations from interaction instead of being handed the relationship graph, the balanced-state rate fell from 60.7% to 34.4% (n = 640), with the Unawareness and Profile principles carrying the effect (SHOWN for LLM agents; the rule-based analogue is INFERENCE). Merges reader candidates V2 and PIMMUR-P3.

**Status.** PARTLY. The belief layer exists (M9.1–M9.7); M9.7 makes appraisal read the receiver's belief about the sender and the situation. Searching the spec for "god-view", "true state" and "believed" returns nothing: there is no prohibition, no per-tie belief slot is named, and no mutant would detect a policy that reads a third-party tie's true conductance.

**What it changes.** Converts an intent into a mutation-provable constraint; theory fidelity (misperceived alliances become possible) and inference validity.

**Cost and risk.** Low for the rule and mutant; moderate if a per-person belief per tie has to be added to M1 (kept by events with per-hop fidelity). No conflict.

### C10. State and mechanism register with static checks

**Proposed requirement.** The object model (M1) MUST be accompanied by a register in which every state variable of Person, Relationship, Triangle and Family is classified (exogenous-homogeneous, exogenous-heterogeneous, endogenous-decision, endogenous-derived) with the mechanisms that write it; every mechanism MUST list what it reads, what it writes on its owner and what it writes on other objects; and the phase order (M3.D.1) MUST list each mechanism with its execution mode (synchronous batch, latency-delivered, slow tick) and trigger condition. A check MUST fail on a variable with no writer, a mechanism not placed in the phase order, and a mechanism that reads a quantity its owner cannot observe.

**Module.** M1, M3, M10, M16 (documentation), check in M11.D.

**Source and evidence.** VISA rules r1, r3, r10, r11, r14, r16 (Tables 9, 11); the AnyLogic case where the register located a reproduction barrier in a proprietary movement library (§5) (ARGUED; the paper does not show the register improves reproduction, and says its comparison with ODD is future work). Reader candidate V1, extending design lessons §2.9 (which covered constants only).

**Status.** PARTLY. M10.C.1 lists invented constants and M11.D.2/M11.D.4 check them; M11.D.8–M11.D.14 are document-scan checks of the same style; there is no register of state variables or mechanisms. The reader's rule-by-rule table (g3 report) is the basis.

**What it changes.** Documentation and a static check; inference validity. The third check (a mechanism reading what its owner cannot observe) is the static form of C9.

**Cost and risk.** Low; a table plus a scan. Note the reader's caution: VISA's four execution modes do not cover batched same-tick events (M1.F.8), which would be recorded as a documented composite.

### C11. Legality mask computed and logged before selection

**Proposed requirement.** For each person each fast tick, the policy MUST compute the set of legal moves from current tie and triangle state (CUTOFF requires a live tie; TRIANGLE requires a reachable third party; PURSUE requires a non-cutoff target; the financial-dependence gate of M5.C; the I-POSITION gates) and MUST select only over that set with renormalised weights; the mask MUST be written to the run log so that a move that was never legal is distinguishable from one that was legal and never chosen.

**Module.** M4.D, M16.A.

**Source and evidence.** Buitrago López et al. §3.2: the reference policy applies a contextual mask (has the agent read a post; are like/share/reply possible; are there follow/unfollow candidates) and renormalises before sampling (SHOWN as practice); Figs 3, 5, 7: LLM policies drove the rare actions to zero in most cells (SHOWN). Requirement wording is the reader's (INFERENCE). Reader candidate P1 (g2).

**Status.** NEW. The spec has 33 mentions of "gate", all specific gates (financial dependence, I-POSITION, anger, entry thresholds), and no use of "legal" or "permitted" for a selection set; no requirement forms the legal set, renormalises over it, or logs it. Searching for "legal" and "mask" as selection concepts returns nothing.

**What it changes.** Makes the rare-move coverage test (design lessons §2.6) interpretable and moves "what is possible right now" out of the policy into an auditable mechanism. Theory fidelity and inference validity.

**Cost and risk.** Low.

### C12. Fallback and tie-break provenance per decision

**Proposed requirement.** Each move-selection log record MUST carry a flag stating whether the outcome was chosen by the scored policy or by a fallback or tie-break rule (equal scores, non-finite score, exhausted life energy, empty legal set), and ensemble readouts MUST report the fallback rate per move and per person; a criterion whose passing ensemble has a fallback rate above a declared threshold MUST be flagged. The tie-break rule itself MUST be stated and its draw keyed (C3).

**Module.** M4.D, M16.A, M11.

**Source and evidence.** Ye et al. 2026 (TRAILS, 2605.18890) App. C.2.1, C.4, E.1.7: a parse-validity indicator is logged per decision and an unparseable response falls back to DEFECT (SHOWN as practice); that a fallback rule can silently generate the tested behaviour is INFERENCE (cf. design lessons §2.8). Reader candidate T4.

**Status.** NEW. The spec contains no tie-break or fallback rule (0 hits for tie-break; the 23 hits for "fallback|NaN" are all the substring "financially"). Softmax selection over propensities (M4.D.1) with a WITHHOLD outcome will need one.

**What it changes.** A logging rule and a test guard; inference validity.

**Cost and risk.** Low; the persistence sink remains a pure observer (M16.B.3).

### C13. Habituation on repeated relief, or a proof of non-bistability

**Proposed requirement.** Any M4 term that credits anxiety relief for a repeated identical move within a window (reassurance-seeking PURSUE, checking-type OVERFUNCTION) SHOULD decay geometrically with repetition count, with the decay constant labelled [I], or be shown by sweep not to be bistable (ignored below a threshold, absorbing above it).

**Module.** M4, M10.

**Source and evidence.** Prasad 2026 (2607.07753), Experiments and supplement A.10: a checkpoint-return bonus without habituation was bistable — ignored below a threshold, an unbounded loop above it, with no graded regime between — and a geometrically diminishing bonus produced the graded curve (SHOWN, negative result in RL). Reader candidate P6 (g4), stated as conditional on such a term existing.

**Status.** Uncertain, leaning PARTLY. M4.G.1 hardens repeated moves (three withdrawals register as a distant relationship); M4.D.6a forbids short-horizon anxiety relief as the reinforcement signal; M11.C.16 tests the two horizons. Whether any relief term is paid on repetition could not be settled by search ("relief" has 17 hits, all about symptom relief, binder relief and the forbidden proxy; "habituat" has none). Include as a functional-form guard of the design lessons §2.7 class.

**What it changes.** Theory fidelity and inference validity, if the term exists.

**Cost and risk.** Low.

### C14. Single-triangle move reachability check

**Proposed requirement.** M11 SHOULD include a static reachability test: for a three-person sub-family over one fast tick, enumerate the move assignments the M4 policy can produce from any state within the declared ranges (by linear-programming feasibility where the policy is piecewise linear, by dense sampling of the state box otherwise) and assert that every one of the nine moves and WITHHOLD is reachable; unreachable moves MUST be listed.

**Module.** M11.

**Source and evidence.** Kurz 2025 (2512.18016), Lemma 7 and Figs 3, 7, 9: whether a labelled interaction graph, or a sequence of them, can be realised by the process is a linear-programming feasibility question, solvable exactly for small n (SHOWN for bounded-confidence dynamics); the construction for EPModel is INFERENCE. Reader candidate B3.2. The reader is explicit that full enumeration does not transfer: per tick the pattern space is at least 9^12 before targets, without the ordering structure that keeps Kurz's counts small (51,505 unit interval graphs at n = 12).

**Status.** PARTLY. Design lessons §2.6 asks for rare-move coverage by ensemble sampling; this checks it at the policy level. With C11 in place the mask supplies the legal set, and the test becomes "every move is legal somewhere in the state box and, where legal, receives non-negligible propensity".

**What it changes.** Adds a test; closes the silently-filtered-behaviour-class risk at the source.

**Cost and risk.** Moderate if the policy is piecewise linear (an LP); otherwise dense sampling, which weakens "unreachable" to "not observed" and must be reported as such.

### C15. Disposition of a Person at death

**Proposed requirement.** M6 MUST name the mechanism that removes a Person at death and MUST state where that Person's acute and chronic anxiety, bond energy on each tie, functioning-balance debts, investment, triangle positions and share of the undifferentiation budget go, so that M6.I.6 (anxiety conserved) and M6.I.7 (no exit from the field) remain assertable across a death; M11 MUST include an invariant test spanning a death event. If births occur within the horizon, the creating mechanism and the initial state it assigns MUST be named likewise (and the identifier rule C2 applies).

**Module.** M1, M6, M7.C, M11.

**Source and evidence.** VISA rules r8 and r13: a type whose instance count varies must have named create and remove functions with declared external effects (ARGUED); that the spec leaves this open is my finding from the search, not the reader's inference alone. Reader candidate V3.

**Status.** NEW. Mortality is on the slow tick (M3.B.1) and governed by life stage (M7.C.1); M11.C.15 requires that a death destabilise a symptom-stabilised arrangement as recovery does; M7.C.1c scales a redistribution by the affected tie's share of investment. No requirement states the disposition of the dead person's conserved quantities. Births are mentioned (11 hits for birth/born) in the context of sibling position and the multigenerational line, not as a run-time creation mechanism.

**What it changes.** Closes a hole in the invariants. Theory fidelity (the dead remain in the emotional field through the multigenerational ledger, M1.D) and inference validity (a leak at death would fake destroyed anxiety and pass M6.I.6 everywhere else). Needs an owner decision: is death an exit, or does the ledger absorb the person?

**Cost and risk.** Small if the ledger already absorbs it; the decision is the cost.

## 3. Test-design candidates for M11 (P2)

### C16. Graded-parameter monotonicity tests over four or more levels

**Proposed requirement.** For each Person parameter the theory treats as graded (basic_level, chronic_anxiety at least), M11 SHOULD include one test that sweeps the parameter over four or more declared levels across the ensemble and asserts a monotone ordering of a pre-declared primary readout, with the ordering violated in the mutant.

**Source.** Prasad Tables 2 and 7: seven graded knobs with a pre-declared primary assay each, at four or five dose levels, 10 seeds per configuration (SHOWN in RL); transfer INFERENCE. The reader notes the paper's own monotonicity claim overstates its tables (impulsivity and addiction are non-monotone within CI; depression is step-like), which is the point: a two-arm test cannot see a threshold shape. Reader candidate P1 (g4).

**Status.** PARTLY. The spec uses "monotone" 19 times, mostly as prohibitions (M4.D.4: I-POSITION propensity not monotone in basic_level; M7.D.2d: lock-in non-monotone in severity). M11.C.6 asserts a multi-generation decline shape. No criterion asserts a graded response over four or more levels of a person parameter; M0.4 restricts criteria to two-arm directions, so this is a deliberate extension of the criterion form, and should be admitted the way M11.C.16's ceiling was (the levels are inputs, not calibrated magnitudes).

**Changes.** Theory fidelity: the corpus supplies orderings, and an ordering over four levels is a stronger use of it than a sign. Cost: about four times the ensemble runs per parameter.

### C17. Matched-magnitude contingency-severing mutants

**Proposed requirement.** For each M4 appraisal input that selection depends on, the mutation suite SHOULD include a variant that preserves the input's per-tick magnitude distribution but destroys its state-contingency (a seeded permutation across persons or ticks, keyed per C3), and the affected criterion MUST go red under it.

**Source.** Prasad Fig. 3 and Fig. 12: with the anxiety penalty held at its severe dose, shuffling, randomising or redirecting the appraisal that feeds it — all three preserving the penalty's size and frequency — collapsed risky-route avoidance from 1.00 to 0.00–0.20 with non-overlapping CIs over 10 seeds (SHOWN). Reader candidate P2 (g4).

**Status.** PARTLY. The M11 mutation protocol requires delete-or-invert; a matched-magnitude permutation is stronger because it rules out "any input of that size would do". Design lessons §2.6 has a sever-the-input example that removes the input rather than keeping its size.

**Changes.** Inference validity. Cost low.

### C18. Persistence-after-spell test with a learning-off arm

**Proposed requirement.** M11 SHOULD include a test that a relational pattern established during an exogenous anxiety spell (DISTANCE frequency, conductance loss, CUTOFF) persists after the spell ends in the arm where the automatic channel learns, and remits in the arm with learning disabled; the report MUST show the time-to-remission distribution per seed.

**Source.** Prasad Tables 4, 11, 12: after the disorder knob is removed, mania, OCD and addiction remit to zero but anxiety and PTSD phenotypes persist, because the avoidant policy never re-encounters the disconfirming evidence; exposure with response prevention reverses it (SHOWN in RL; the "resists" cells rest on 5 seeds with wide intervals, which the reader flags). Reader candidate P3 (g4).

**Status.** NEW as a test. The mechanism is present by construction: the automatic channel learns (M4.D.6) and DISTANCE/CUTOFF remove the stimulus that would disconfirm the learned response, so persistence after a spell should be an emergent property with no persistence rule written in. That makes it a good acceptance test and a clean mutation target (learning off → remits). Note M11.C.29 already uses time course (transient versus persistent symptom in a third party) as a discriminator, so the form is familiar.

**Changes.** Theory fidelity. Cost low. Requires a learning switch in M10.

### C19. Ordinal acceptance criteria

**Proposed requirement.** Where the corpus supplies an ordering of effect strengths across three or more mechanisms, a criterion MUST assert the rank order of the per-seed paired arm differences across those mechanisms, not only the sign of each; the criterion passes when the observed ordering matches the corpus ordering in a pre-declared majority of seeds, and deleting the mechanism ranked first MUST break the ordering.

**Source.** Kalluri 2026 (2603.01189) §5.1 and Fig. 3: the model reproduced only 4 of 8 meta-analytic effect sizes within their intervals but ranked the eight correctly (Spearman ρ = 0.833); the authors present the dual interval/ordinal test as a template (SHOWN as technique). Reader candidate K1.

**Status.** NEW as a criterion form. The spec says the corpus supports directions, orderings and mechanisms and almost no magnitudes (§0, line 59); ordinal criteria are the test form that uses the middle term. Needs three or more comparable arms per criterion.

**Changes.** Theory fidelity and inference validity. No magnitudes assumed.

### C20. Frequency-controlled asymmetry tests, with per-event magnitude and count logged separately

**Proposed requirement.** For every mechanism whose [I] constants are asymmetric (a loss, escalation or hardening rate larger than the corresponding gain or recovery rate), the run log MUST record per-event magnitudes and event counts separately, and the acceptance test for the asymmetry MUST hold event counts equal across the compared arms or condition on them.

**Source.** Kalluri Table 5 and §7.1: a per-event rule with negative outcomes weighted 1.5× positive produced cumulative loss/gain ratios of 0.07–0.55 that varied monotonically with failure frequency; per-event asymmetry did not produce cumulative asymmetry (SHOWN). The reader notes much of this is arithmetic (expected ratio 1.5(1−p)/p) but that repair, visibility gating and bounds also suppressed losses, undecomposed. Reader candidate K2.

**Status.** NEW. Relevant to M4.G.1 (hardening), M7.D.2 (lock-in), the change-back ladder (M5.E), and any "cutoff is easier than reconnection" readout: a cumulative total confounds rule and frequency.

**Changes.** A reporting and test-design rule; inference validity. Cost low if M16 already logs every Event.

### C21. Representation-invariance mutants

**Proposed requirement.** The acceptance suite MUST include re-encoding mutants that are numerically equivalent by construction (rescaled state ranges, changed float summation order in same-tick aggregation, altered rounding or clamping thresholds within declared tolerance, integer versus float tick counters), and every M11.C direction MUST be unchanged under them; a change is reported as an encoding artefact, not a mechanism result. Because rescaling changes bytes, this test is on directions, not on M3.D.5 byte identity.

**Source.** TRAILS §3.1 and Table 3 (representation-level perturbations) and §4.1 P3 (memory representation moved outcomes while memory content was identical) (SHOWN for LLM agents); the rounding/clamping transfer is INFERENCE, matching the design lessons §2.7 rounding artefact. Reader candidate T2.

**Status.** NEW. Complements C31 (threshold-margin log), which locates the comparisons that rounding decides.

### C22. Outcome-directive audit with sign-inversion mutants, and a premise/mechanism register

**Proposed requirement.** For each M11.C criterion the spec MUST list the minimal set of rules and [I] constants whose joint operation produces the asserted direction, and no rule in that set may name the criterion's readout or asserted outcome as its own target. Each such rule MUST be run as a sign-inverted mutant in addition to the deletion mutant, and a criterion that flips under inversion of a single rule MUST be re-stated one composition level up or documented in M10 as a programmed premise rather than a derived result.

**Source.** PIMMUR §4.4 coding rule for Minimal-Control (an instruction is a violation if removing it leaves the simulation viable but plausibly eliminates the phenomenon) and the three-arm Original/Ours/Reverse designs in which the compliant arm matched the reversed one (telephone: not different from Reverse, P = 0.079; fake news §2.4) (SHOWN for LLM prompts); transfer to rule-based directives INFERENCE. Reader candidate PIMMUR-P1.

**Status.** PARTLY. The M11 mutation protocol already says delete **or invert** and confirm red; the additions are (i) running both, (ii) the minimal-set listing, and (iii) the premise-versus-mechanism column in M10. The spec's own revision history distinguishes premises from consequences in several places (e.g. M7.D.2 "one requirement yields three observed behaviours as consequences"), so the register formalises an existing habit.

**Changes.** Both theory fidelity (separates Bowen's premises from what follows from them) and inference validity. Cost: doubles mutants for the minimal set.

### C23. Constants frozen before acceptance tests; post-hoc changes logged

**Proposed requirement.** Every [I] constant's value MUST be recorded before the acceptance suite is first run against it; any later change that alters an acceptance outcome MUST be logged with the failing criterion named, and a criterion passed only after such a change MUST be reported as post-hoc together with the sweep fraction (design lessons §2.3) over which it holds.

**Source.** PIMMUR's demand-characteristics argument and Fig. 1 (ARGUED); Minimal-Control violated in 53.3% of 576 audited LLM studies (SHOWN for that literature). Reader candidate PIMMUR-P5. Kalluri is a worked example of the failure: an asymmetry parameter calibrated to a target (Table 3) and the target then reported as a validation result (Table 5).

**Status.** NEW. M11.F.9(c) forbids counterfactuals from parameters tuned to a known history; nothing addresses tuning constants to the acceptance tests themselves.

**Changes.** Documentation rule; inference validity. Cost nil.

### C24. Readout-saturation rule

**Proposed requirement.** Every acceptance test MUST record the baseline arm's position within the readout's attainable range, and a null on a readout whose baseline sits at a bound MUST be reported as an assay limit, not as evidence about the mechanism.

**Source.** Prasad: the LavaGap ablation and the MiniWorld anxiety null, both reported as assay limits because the baseline was already at the bound (SHOWN). Reader candidate P4 (g4). Same class as the spec's own readout trap (overt emotionality peaks mid-scale) and Prasad's anxiety × depression masking (a withdrawn agent cannot express avoidance).

**Status.** PARTLY. M11.4a requires a null to carry an equivalence bound (an underpowered ensemble must not pass a null); saturation is the other way a null can be uninformative and is not covered.

### C25. Test statistic defined for degenerate arms, with multiplicity declared

**Proposed requirement.** The M11 direction test MUST be defined when one arm has zero seed-to-seed variance or discrete bounded outcomes (a rank-based paired statistic on per-seed differences), and where several readouts are tested under one criterion a multiplicity correction MUST be declared.

**Source.** TRAILS App. C.1: Mann–Whitney U chosen because some conditions have zero run-level variance; Holm correction within each metric (SHOWN as method). Reader candidate T6. Blando et al. 2026 (2604.04543) §7.4 use Welch's t-test at each step; the g1 reader's caution applies — under paired arms (C1) the test is on per-seed differences, not an unpaired test.

**Status.** NEW (0 hits for multiplicity, Holm, Bonferroni; one hit for "paired"). M11.4a covers the null side only.

## 4. Phase E candidates: ensemble runner, arms, readouts (P3)

Phase E is out of scope in v2 by decision (§0.1). These are inputs to writing it.

### C26. Adaptive ensemble size with a declared precision, and UNDETERMINED as a third outcome

**Proposed requirement.** For each directional criterion Phase E MUST add seeds in blocks until the confidence interval of the per-seed arm difference has half-width below a declared δ, up to a declared cap; the log MUST record the seed count used; a criterion whose interval has not converged at the cap is reported UNDETERMINED, not pass or fail.

**Source.** Blando et al. §4, §7.1: MultiVeStA adds simulation batches (block size 30) until the interval at every queried time point is below δ, so high-variance regions get more runs; §7.4.2: configurations whose variance was too large to converge were reported as such (SHOWN as practice); the third outcome is the reader's (INFERENCE). Reader candidate B2.1. Also M11.4a's underpowered-null concern from the other side.

**Status.** PARTLY. M11 requires ensembles for distributional criteria and M11.4a requires equivalence bounds on nulls; there is no stopping rule and no third outcome.

### C27. Time-resolved arm comparison carrying power

**Proposed requirement.** Phase E SHOULD test the arm difference at each reporting tick, report the first tick at which the pre-declared direction holds, and report the power of the test wherever the null is not rejected, so that "arms did not differ" is distinguishable from "ensemble too small".

**Source.** Blando §7.4: per-step tests with first-rejection ticks and power reported (a non-rejection at power 1.0 read as saturation) (SHOWN as practice). Reader candidate B2.2. Use a paired statistic (C25).

**Status.** PARTLY (design lessons §2.8 asks for windowed trajectories; the power-carrying null is new).

### C28. Per-seed principal-strata readout

**Proposed requirement.** Phase E SHOULD report, per criterion and per seed, the paired arm difference and classify each seed as same, converted or reversed; a directional criterion holds when converted seeds exceed reversed seeds by the pre-declared margin.

**Source.** Buffalo et al. §2.2 Table 1 and App. A: under execution invariance each event's noise places it in a principal stratum, and ordered scenarios give consistently ordered strata (ARGUED); per-seed readout INFERENCE. Reader candidate B1.5. Requires C1 to be meaningful.

**Status.** PARTLY (design lessons §2.6 per-seed paired differences; the three-way classification is new).

### C29. Per-seed trajectory regime classification, with an inconclusive class and a non-absorbing-bound check

**Proposed requirement.** Phase E MUST classify each bounded slow and fast state trajectory per seed as settled at a bound, settled at its attractor, oscillating, or inconclusive at the horizon, and report the fractions per arm; a directional criterion MUST NOT count inconclusive seeds as passes. An M11 test MUST show that no state bound (functional_level, anxiety channels, conductance) is absorbing unless the theory says it is; a run starting a Person at a bound must show the mechanism can leave it.

**Source.** Holland Fig. 6: a three-class outcome map (consensus, polarisation, inconclusive at 10,000 steps); Corollary 4.2: ±1 are sinks (SHOWN in their model); the absorbing-bound risk for EPModel is INFERENCE extending design lessons §2.7 (absorbing zero) to absorbing extremes. Reader candidate H3. Holland's N = 10 versus N = 100 result (the polarising band widens and flips under marginal constant changes at small N) is direct evidence that regime boundaries at EPModel's size need this treatment.

**Status.** NEW (one hit for transient/settling, none for absorbing bounds).

### C30. Ordering-spread as a variance component in any sequential regime

**Proposed requirement.** If Phase E runs an activation regime in which persons select or receive in sequence (design lessons §2.1), the runner MUST, per seed, run every ordering or a seeded sample of orderings of declared size, and MUST report the spread of each directional result across orderings as a variance component distinct from the across-seed spread.

**Source.** Sachdeva §3.3, Fig. 2c, Fig. 9 (numbers under C5) (SHOWN for LLM debates). Reader candidate S1.

**Status.** NEW; only applies if a sequential regime is ever run.

### C31. Threshold-margin log and per-constant invariance interval

**Proposed requirement.** For every [I] constant used as a threshold in policy or appraisal, the engine SHOULD emit per tick the signed margin of each comparison against it, and Phase E MUST report for each directional criterion the interval of each threshold constant within which every seed's trajectory is unchanged (the minimum positive and negative margins over the run); M10's register gains an "invariance interval" column.

**Source.** Kurz Lemma 3 and its breadth-first algorithm: for a fixed initial state the confidence axis splits into finitely many intervals with identical trajectories, and the switch points are the pairwise distances encountered along the way (SHOWN for bounded-confidence dynamics); the generalisation to threshold-gated rules is INFERENCE. Reader candidate B3.1.

**Status.** NEW. Turns the design-lessons §2.3 constant sweep from sampled into exact for threshold-type constants, and flags any comparison decided within numerical noise (pairs with C21).

**Cost.** One log record class; logging stays a pure observer.

### C32. Initial-condition distribution D0 with declared correlation structure and its own variance component

**Proposed requirement.** Initial state for any run family MUST be specified as a distribution over person attributes, tie states, triangle topology and beliefs that declares its correlation structure (theory-stated pairings such as spouses' matched basic_level, M2.A.0e; orderings among children), not as independent per-attribute ranges; the runner MUST draw initial states from it and MUST report initial-condition variance separately from seed variance, exogenous-spell-timing variance and constant-sweep variance.

**Source.** Li & Tao §3.2.3 (sampling attributes independently can break the joint distribution; two initialisations with identical marginals can encode different mechanisms) and Action 3 (report sensitivity to D0 as a distribution) (ARGUED). Reader candidate L3. Kurz Example 5 is supporting evidence at small n: with n = 4 and ±10% bands on inputs, freezing time, cluster count and final width spanned essentially the whole outcome space.

**Status.** PARTLY. M15.D.2–M15.D.4 require ranges and envelopes and say initial conditions are not nuisance parameters; M2.A.0e and M2.A.0g state two correlations (matched basic_level in a pair; pole independent of sex). The correlation structure as a required part of the distribution, and initial-condition variance as a named component, are missing. Must not be tuned to a known history (M11.F.9).

### C33. Structural arms hold declared structural totals fixed

**Proposed requirement.** When counterfactual arms differ in initial tie topology or tie strengths, the arm specification MUST state which structural totals are held fixed (number of ties, total conductance, total bond energy, generation structure) and the run log MUST verify them.

**Source.** TRAILS App. C.3.3: homophily varied by degree-preserving edge swaps with the degree sequence held fixed; hub assignment varied with both the degree sequence and the homophily band fixed (SHOWN as method). Reader candidate T5.

**Status.** NEW.

### C34. Three named control arms: no-interaction (two levels), endogenous-only, homogeneous-family

**Proposed requirement.** Phase E SHOULD provide (a) two no-interaction controls — all conductance zero, and ties intact with event delivery suppressed — so that drift from standing load (M3.D.1 step 1, M6.I.8) is separated from drift from events; (b) an endogenous-only arm with all exogenous spells removed and societal anxiety constant, so that persistent non-settling is attributable to the relationship mechanisms; and (c) a homogeneous-family arm in which all Persons share identical initial basic_level, chronic anxiety and tie attributes, with each M11.C criterion declaring whether it is expected to pass or fail there, and a criterion that passes where the theory says heterogeneity is required (projection onto the most vulnerable child, M2.A.2) flagged.

**Source.** (a) Sachdeva App. D, Tables 4–5: one model's verdict distribution shifted under the debate framing alone before any message was exchanged (SHOWN; the EPModel analogue — a tie exerting load without traffic — is INFERENCE). (b) Holland §6: persistent non-convergence with no external driver is stated as the substantive claim about the mechanism (ARGUED). (c) PIMMUR Profile principle; adding profile diversity alone moved balanced states by 14 pp (SHOWN for LLM agents; null-expectation use INFERENCE). Merges reader candidates S3, H4 and PIMMUR-P6.

**Status.** PARTLY (design lessons §2.6 has one no-interaction control; the two-level split, the endogenous-only arm and the homogeneous arm are new).

### C35. Arm-blindness of the engine

**Proposed requirement.** The engine MUST NOT receive an arm label, scenario name, test identifier or readout definition; arms MUST differ only through the declared channels (initial state, [I] constants, exogenous spells, mechanism switches declared in M10); the build MUST include a static check that no policy, appraisal or consolidation module imports from the test or readout modules.

**Source.** PIMMUR §2.4: when the model could see the construct under test (Unawareness violated), balanced outcomes were 1.77× more frequent (SHOWN for LLM agents); the rule-based analogue — a code path that branches on arm identity — is INFERENCE. Reader candidate PIMMUR-P2.

**Status.** PARTLY. M16.B forbids I/O and UI in the engine; M10.C.4b/M11.D.12 forbid engine code from importing the checks module and config keys from resolving to corpus magnitudes, which is the same shape of rule for one case. The general prohibition and the static import check are new. Cost trivial.

### C36. Per-result robustness audit record with claim grade

**Proposed requirement.** Every directional result reported from an ensemble MUST carry an audit record listing, per design dimension mapped to EPModel (seed; initial persons and ties; appraisal and belief rules; tick length and estimator windows; activation and same-tick aggregation; intervention timing, target and channel; topology; family composition), whether the dimension was perturbed and whether the direction held, was sensitive, or was **unaudited**; and MUST state the claim grade (exploratory, mechanism, intervention) the result is used for.

**Source.** TRAILS §3.2, §5, §6 (ARGUED) and the uneven-sensitivity finding (§4: persona format flipped a two-agent equilibrium by 76 pp in one model and about 1 pp in another; memory window and activation probability were null) (SHOWN for LLM agents). Reader candidate T1, extending design lessons §2.9 from constants to structural dimensions. The reader's mapping of the TRAILS-D taxonomy to EPModel (g5 report) is the basis for the dimension list; the representation-level half has no analogue except C21.

**Status.** NEW.

### C37. Sensitivity measured at two or more reference configurations

**Proposed requirement.** For each dimension audited under C36, sensitivity MUST be measured at no fewer than two declared reference configurations of the [I] constants (for instance a low- and a high-differentiation family), because a perturbation that is null at one configuration may flip the outcome at another.

**Source.** TRAILS §4.2 Finding Summary (same perturbation, 76 pp versus ≈1 pp across models) (SHOWN); treating the constant configuration as the analogue of model identity is INFERENCE. Reader candidate T3.

**Status.** NEW. Multiplies sweep cost by the number of reference configurations.

### C38. Pairwise additivity residual

**Proposed requirement.** For parameter pairs the theory claims interact (chronic anxiety × basic_level; functioning balance × conductance), Phase E SHOULD run a two-dimensional grid and report the maximum residual against the additive prediction from the two one-dimensional sweeps.

**Source.** Prasad Tables 13–14: mania × impulsivity residual 0.82; anxiety × depression residual 0.50, with the mechanism (a withdrawn agent cannot express avoidance) named (SHOWN). Reader candidate P5 (g4).

**Status.** PARTLY (design lessons §2.11 Latin-hypercube sweep samples the joint space but does not report interaction).

### C39. Move-transition structure and a divergence summary as readouts

**Proposed requirement.** The trace renderer SHOULD emit, per person and per arm, the first-order move-transition count matrix, and ensemble reports SHOULD compare arms on transition structure as well as on move marginals; where a criterion concerns the distribution of moves, the report SHOULD include base-2 Jensen–Shannon divergence with Laplace smoothing between arms, per person stratum, as a per-seed paired distribution rather than one pooled value.

**Source.** Buitrago López §3: JSD as the alignment metric (SHOWN as method); the transition readout is INFERENCE from the paper's own gap (its reference policy is first-order Markov in the move sequence, yet only marginals were scored). Reader candidates P2 and P3 (g2).

**Status.** NEW. Cost trivial (a 10×10 count per person); pure observer.

### C40. Inertia/conformity decomposition as a post-run diagnostic

**Proposed requirement.** The analysis layer SHOULD fit, per person, a multinomial model of the selected move on an indicator that the same move was selected last tick (inertia), the count of each move type received or witnessed in the current batch, and the count in earlier ticks (conformity, within-batch and prior), with person and tick effects; acceptance tests MAY assert directions on these coefficients (zeroing all conductance MUST drive the conformity terms to about zero; deleting the self-directed channel MUST reduce inertia for high-basic_level persons).

**Source.** Sachdeva §2.4 Eq. 2, Table 1, App. L: the two quantities are statistically separable, model-specific, and interact positively (nested-model AIC drops from 92,233 to 75,640) (SHOWN as method on LLM outputs; the coefficients themselves say nothing about humans and must not become constants). Reader candidate S2.

**Status.** PARTLY (design lessons §2.6 asks to regress selected moves on the inputs shown; the own-past versus witnessed-others split and the within-batch versus prior split are new). Rare moves will have wide intervals; report them.

### C41. Belief–truth discrepancy readout

**Proposed requirement.** For each belief in the belief layer that has a true-state counterpart, the run log MUST allow a signed discrepancy (belief − truth) per tick, and Phase E SHOULD report its distribution and trajectory per arm alongside outcome readouts, so that arms in which beliefs and outcomes move in opposite directions are identified rather than averaged away.

**Source.** Kalluri §6.3, Table 9: trust and task success decouple across scenarios; a calibration-error readout (|subjective trust − objective capability|, range 8.9–52.1) distinguishes them (SHOWN); transfer INFERENCE. Reader candidate K3.

**Status.** PARTLY. M9.5 and M11.G.4 require belief and ground truth reported separately, and M16.A.5 tags belief writes; the signed per-tick discrepancy as a readout is the addition, and with C9 it becomes the quantity the true-state mutant is expected to move.

### C42. Enumerated triad initial configurations for triangle criteria

**Proposed requirement.** Triangle-level acceptance tests SHOULD be run over an enumerated set of initial triad configurations (all sign patterns of the three ties' functioning balance and all orderings of the three conductance classes) rather than one hand-set reference triad, with per-configuration outcomes reported.

**Source.** PIMMUR §4.6.2: all 2^6 = 64 initial directed signed states of a three-agent system enumerated, 10 runs each (SHOWN as method); applicability to EPModel triads INFERENCE. Reader candidate PIMMUR-P4. Kurz's transition graphs over n = 3 (Fig. 3) are the same idea for a deterministic process.

**Status.** NEW. Feasible for triads (tens of configurations × seeds), not for the twelve-person family, which stays on M15 ranges.

## 5. Exploratory LLM line only (P4; no change to v2)

All four papers on this line support M3.D.6 (no LLM in the decision path) and add nothing that argues for relaxing it. Buitrago López: mean JSD 0.212 between LLM action distributions and the intended finite-state policy across nine configurations, no prompt best across models, 135–1,337× slower than the reference policy; even when handed the exact probabilities, one model still inverted the rare/common actions (SHOWN, single run per configuration, no variance). Sachdeva: relative model behaviours persist under prompt edits, so prompting does not remove model identity as a factor (SHOWN). Wang et al. 2026 (2608.06485): personas differ at baseline (across-persona SD ≈ 0.7) but respond to life events alike (response SD 0.19), with median changes about ten times smaller than human meta-analytic bands, 13 of 27 definite-prior cells reversed, retirement reversed by every model, a pull toward agreeableness, and self-reported change nearly uncorrelated with behavioural choices (ρ ≤ 0.105) (SHOWN; horizon three turns). Li et al. 2026 (2608.24912): a benevolence bias across 18 models that a deliberately antisocial persona cannot push below the human baseline on prosociality, social desirability or harm aversion, though it can on emotional softening and fairness optimism; only an output-level contrastive calibration against a neutral-persona call moved the distribution, and that needs logits and a human reference (SHOWN on survey items).

Four protocol items for the exploratory notes, none for M1–M16:

- **X1. Change-claim protocol** (Wang §6): any claim that an LLM persona's maturity or reactivity changed MUST be reported against a no-event retest floor for the same persona and model, under at least one paraphrase of the stressor, and with a closed-action behavioural readout beside any questionnaire readout.
- **X2. Heterogeneity floor** (Wang RQ4): report between-persona SD of the response and treat collapse relative to baseline SD as failure of the persona layer before interpreting any mean shift.
- **X3. Persona-range ceiling test** (Li, malicious-persona result): before an LLM persona represents a low-maturity agent, run an antisocial rewrite of the same persona on a closed-choice battery and report whether it crosses the neutral-persona baseline on prosociality and harm aversion; if not, declare that region of the scale unreachable for that model.
- **X4. Neutral-persona contrast over the closed move set** (Li Eq. 1): if an LLM ever proposes a distribution over the nine moves (design lessons §3.6), obtain a matched neutral-persona distribution per call and report the persona-specific component with α declared [I], keeping the sample draw in the seeded engine; without a human reference α cannot be calibrated, so this is a diagnostic, not a fix.

The reader's cost estimate for the record (INFERENCE from Buitrago López's timings): one EPModel run is 12 × ~2,080 ≈ 25,000 decisions; at the observed LLM rates of 0.09–0.94 s per decision that is 39 minutes to 6.5 hours per run before ensembles, against about 17 s for a finite-state policy.

## 6. Interactions with existing rules

**M3.D.4 (single seeded generator).** C1 replaces the second sentence. The first sentence (the engine is a deterministic function of seed, config and scenario) and M3.D.5 stand. The design-lessons file's §2.8 note on snapshot-and-fork and Q6 should be re-read in the light of C1: with keyed draws a fork needs no generator state.

**M3.D.5 (byte-identical logs).** Unaffected by C1–C5. C21 (re-encoding mutants) deliberately changes bytes and tests directions; the requirement text must say so to avoid a false conflict.

**M3.D.6 (no LLM in the decision path).** Every LLM-line paper supports it; C40 and C39 are post-run analyses outside the engine.

**M11.F.9 (no fitting to known histories).** C23 is its twin for fitting to the acceptance tests; C32 must not tune the initial distribution to a history.

**M16.B (engine purity, pure-observer logging).** C11, C12, C31, C39 add log record classes; all are observer-side. C35 adds a static check of the same kind as M10.C.4b.

**M0.4 (two-arm direction criteria).** C16 (four-level monotonicity) and C19 (ordinal over three or more arms) extend the criterion form; both use inputs, not calibrated magnitudes, and can be admitted the way M11.C.16's declared ceiling was.

**M1.F.8 (batched simultaneous events).** C5 tests it; C10's register has to record batching as a composite execution mode, since the VISA modes do not include it.

## 7. Coverage findings from the spec search (for the owner's reference)

Searched `bowen_agent_model_spec_v2.md` on 2026-09-20. Zero hits: god-view, true state, believed (as an appraisal input), tie-break, fallback (all "fallback|NaN" hits are the substring of "financially"), legal or permitted as a selection set, burn-in or settling window (one hit for "transient", in M11.C.29), snapshot or fork, Philox, counter-based, stream, permutation, multiplicity correction, saturation, habituation, absorbing bound. One hit for "paired". Present: gate (33), witness (15), belief (21, including M9.1–M9.7), mortality and death (12, none stating disposition), monotone (19, mostly prohibitions), mutation protocol with delete-or-invert (M11, line 1014), equivalence bound on nulls (M11.4a), single seeded generator (M3.D.4), byte identity (M3.D.5, M11.D.5, M16.T.5), engine-import guard for the checks module (M10.C.4b, M11.D.12).

## 8. Candidate table

| ID | Candidate | Module | Status | Priority | Fidelity / validity | Sources (reader IDs) |
|---|---|---|---|---|---|---|
| C1 | Counter-based event-keyed draws | M3.D.4 | NEW, corrects | P1 | validity | Buffalo B1.1; Holland H2; Li&Tao L2b |
| C2 | Stable identifiers across arms | M1, M2, M15 | NEW | P1 | validity | Buffalo B1.2 |
| C3 | Declared key per draw class | M3 | NEW | P1 | validity | Buffalo B1.3 |
| C4 | Placebo arm; single-query caching | M11.D, M3 | NEW | P1 | validity | Buffalo B1.4 |
| C5 | Order-permutation invariance test | M11.D | PARTLY (M1.F.8) | P1 | validity | Li&Tao L2a; Sachdeva |
| C6 | Named activation and visibility objects | M3, M1.F, M16.A | PARTLY | P1 | validity | Li&Tao L1 |
| C7 | Witnesses computed by visibility rule | M1.F, M4.E | PARTLY (M8.5) | P1 | fidelity | Li&Tao L4 |
| C8 | Witness appraisal from ties to both parties | M4.C, M11 | PARTLY (M4.C.1, M4.C.7) | P1 | fidelity | Holland H1 |
| C9 | Information-access rule, true-state mutant | M4, M1, M9, M11 | PARTLY (M9.7) | P1 | both | VISA V2; PIMMUR P3 |
| C10 | State/mechanism register, static checks | M1, M3, M10, M11.D | PARTLY | P1 | validity | VISA V1 |
| C11 | Legality mask, logged | M4.D, M16.A | NEW | P1 | both | Buitrago López P1 |
| C12 | Fallback/tie-break provenance | M4.D, M16.A, M11 | NEW | P1 | validity | TRAILS T4 |
| C13 | Habituation on repeated relief | M4, M10 | uncertain | P1 | both | Prasad P6 |
| C14 | Single-triangle reachability | M11 | PARTLY | P2 | both | Kurz B3.2 |
| C15 | Disposition at death | M1, M6, M7.C, M11 | NEW | P1 | both | VISA V3 |
| C16 | Four-level monotonicity tests | M11 | PARTLY | P2 | fidelity | Prasad P1 |
| C17 | Matched-magnitude severing mutants | M11 | PARTLY | P2 | validity | Prasad P2 |
| C18 | Persistence-after-spell, learning-off arm | M11, M10 | NEW | P2 | fidelity | Prasad P3 |
| C19 | Ordinal criteria | M11 | NEW | P2 | both | Kalluri K1 |
| C20 | Frequency-controlled asymmetry tests | M16, M11 | NEW | P2 | validity | Kalluri K2 |
| C21 | Representation-invariance mutants | M11, M6 | NEW | P2 | validity | TRAILS T2 |
| C22 | Outcome-directive audit; premise register | M11, M10 | PARTLY (invert exists) | P2 | both | PIMMUR P1 |
| C23 | Constants frozen before tests | M10, M11, M16 | NEW | P2 | validity | PIMMUR P5; Kalluri |
| C24 | Readout-saturation rule | M11, M16 | PARTLY (M11.4a) | P2 | validity | Prasad P4 |
| C25 | Degenerate-arm statistic; multiplicity | M11 | NEW | P2 | validity | TRAILS T6; Blando |
| C26 | Adaptive ensemble size; UNDETERMINED | Phase E, M11, M16 | PARTLY | P3 | validity | Blando B2.1 |
| C27 | Time-resolved comparison with power | Phase E | PARTLY | P3 | validity | Blando B2.2 |
| C28 | Per-seed principal strata | Phase E | PARTLY | P3 | validity | Buffalo B1.5 |
| C29 | Regime classification; non-absorbing bounds | Phase E, M11, M6 | NEW | P3 | validity | Holland H3 |
| C30 | Ordering-spread variance component | Phase E | NEW (conditional) | P3 | validity | Sachdeva S1 |
| C31 | Threshold-margin log; invariance interval | M16, Phase E, M10 | NEW | P3 | validity | Kurz B3.1 |
| C32 | D0 with correlation structure and variance | M15, M10, Phase E | PARTLY (M15.D) | P3 | both | Li&Tao L3; Kurz |
| C33 | Structural totals fixed across arms | M15, Phase E | NEW | P3 | validity | TRAILS T5 |
| C34 | Three control arms | Phase E, M11 | PARTLY | P3 | both | Sachdeva S3; Holland H4; PIMMUR P6 |
| C35 | Arm-blindness of the engine | M16.B, M3, M11 | PARTLY (M10.C.4b) | P3 | validity | PIMMUR P2 |
| C36 | Robustness audit record, claim grade | M16, Phase E | NEW | P3 | validity | TRAILS T1 |
| C37 | Sensitivity at ≥2 reference configurations | Phase E | NEW | P3 | validity | TRAILS T3 |
| C38 | Pairwise additivity residual | Phase E | PARTLY | P3 | validity | Prasad P5 |
| C39 | Transition matrix and JSD readouts | M16, Phase E | NEW | P3 | validity | Buitrago López P2, P3 |
| C40 | Inertia/conformity diagnostic | M11, M16 | PARTLY | P3 | validity | Sachdeva S2 |
| C41 | Belief–truth discrepancy readout | M16, Phase E | PARTLY (M9.5) | P3 | validity | Kalluri K3 |
| C42 | Enumerated triad configurations | M11, Phase E | NEW | P3 | validity | PIMMUR P4; Kurz |
| X1–X4 | LLM-line protocols | exploratory notes | n/a | P4 | validity | Wang W1, W2; Li L1, L2 |

## 9. What the readers judged already covered (checked; no change proposed)

Engine driver surface reset/next/eval (Blando) — M16.B. Refactor-equivalence by draw order (Blando) — superseded by C1. Interval-valued initial conditions (Kurz Example 5) — M15.D.2–4. Exact rational arithmetic (Kurz) — not adopted; byte identity plus C31 suffices. Solo baseline as no-interaction control (Sachdeva) — design lessons §2.6, extended by C34. Information availability versus awareness (Li & Tao) — belief layer and exogenous-flagged Events, provided spell onsets enter as Events (worth a one-line check in M1.F.6). Mechanism ablation one component at a time (Li & Tao) — M11 mutation protocol. Readout frequency per output (VISA T6b) — design lessons §2.8. Single seeded RNG threaded through all draws (VISA D.3.8) — M3.D.4, now to be replaced by C1. Execution mode as an assumption to test (VISA T7) — design lessons §2.1. OFAT and factorial sweeps (Kalluri) — design lessons §2.3. Witness radius with visibility weight (Kalluri) — witnesses with per-hop fidelity. Self-describing run files, single-script regeneration (Prasad) — M16. Memory as digested persistent state; observation outside agent memory; event-driven rather than aggregate-driven interaction (PIMMUR) — M1, M16.B.3, the v2 event design. Turn order and initial event as perturbations; freezing a slow variable to isolate structure (TRAILS) — design lessons §2.1, §2.6.
