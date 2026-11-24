# Reading reports: six papers, full reads (2026-11-23 digest)

Produced 2026-11-23 from the 2026-11-23 method-literature digest. All six papers are the deep-review candidates the
sweep flagged in `DIGEST.md`/`INDEX.md` on the level criterion (individual or small-group scale, interacting agents):
*Alliance Rupture and Repair in Simulating Cognitive Behavioral Therapy Dialogues* (2611.27544), *On Evaluating User
Simulators for Interactive Agents* (2611.24892), *Personalized Affective Perception via Continual Learning in
Human-Agent Negotiation* (2611.14247), *PsyWorld: Profile-Conditioned World Modeling for Human Behavior Prediction*
(2611.24999), *MANGO Arena: Same Seeds, Different Chains* (2611.23613) and *Probe What-If Worlds Before You Ship*
(2611.14700). All six were read in full from `pdftotext -layout` output by six readers working under
`sweep_readers_brief_2026-09-19/READER_TASK.md`. Each reader checked for overlap by keyword search in the current spec
(`docs/bowen_agent_model_spec_v2.md`, revision 67), `SPEC_CANDIDATES_from_preprints_2026-09-20.md`, the earlier
`SPEC_CANDIDATES_plain_language_*.md` files, and `docs/todo/2026-09-27_paper_material.md`'s held-for-later-revision
section. Reports are reproduced as written; only heading levels were changed. No candidate this week is narrator/LLM-line
only: every ID below is offered against the v2 spec, though several are test and reporting conventions rather than model
changes.

**Status: none of these candidates is in the spec.** A plain-language explanation with examples is in
`SPEC_CANDIDATES_plain_language_2026-11-23.md`.

| Part | Paper | Candidates |
|---|---|---|
| 1 | Chai et al. 2026, *Alliance Rupture and Repair in Simulating Cognitive Behavioral Therapy Dialogues* (2611.27544v1) | AL1, AL2, AL3, AL4, AL5 |
| 2 | Zhou et al. 2026, *On Evaluating User Simulators for Interactive Agents* (2611.24892v1) | US1, US2, US3, US4, US5 |
| 3 | Keskin et al. 2026, *Personalized Affective Perception via Continual Learning in Human-Agent Negotiation* (2611.14247v1) | AP1, AP2, AP3, AP4, AP5 |
| 4 | Qin et al. 2026, *PsyWorld: Profile-Conditioned World Modeling for Human Behavior Prediction* (2611.24999v1) | PS1, PS2 |
| 5 | Amar 2026, *MANGO Arena: Same Seeds, Different Chains* (2611.23613v1) | MG1, MG2, MG3, MG4 |
| 6 | Zhu et al. 2026, *Probe What-If Worlds Before You Ship: Persona Simulation for Real-World Decision Making* (2611.14700v1) | PW1, PW2, PW3, PW4 |

---

# Part 1: A Hidden Alliance State, Rupture and Staged Repair in a CBT Dyad (2611.27544v1)

Chai, Yang, Liu, Wang & Lee, *Alliance Rupture and Repair in Simulating Cognitive Behavioral Therapy Dialogues*, arXiv:2611.27544v1, 20 November 2026. Five authors, NUS and CMU, not peer reviewed. Read in full from `pdftotext -layout` output (2,146 lines). No figures survived; results are taken from tables, prose and the appendix algorithms. Overlap checked by keyword search against `docs/bowen_agent_model_spec_v2.md` (rev 65+), `SPEC_CANDIDATES_from_preprints_2026-09-20.md`, the `SPEC_CANDIDATES_plain_language_*.md` files and `docs/todo/2026-09-27_paper_material.md`.

The object of study is a **two-agent counselling dyad over one session**. It is an LLM system — the client's appraisal, the observer and the speakers are all model calls — so almost nothing in it is reproducible in EPModel's sense. What transfers is the shape of the relationship object: a hidden state on one side of a dyad that governs how openly that side engages, is invisible to the other side, and is repaired by a staged repertoire that the other side chooses.

## A. Report

1. **Simulation formalism.** [PAPER] §3, Eq. 1: a session is a sequence of exchanges, each a counsellor utterance followed by a client reply. The client side is a three-part update — `a_t = Appraise(c_t | π, z_t)`; `(z_{t+1}, m_{t+1}) = F(z_t, m_t, a_t)`; `u_t = Client(π, z_{t+1}, m_{t+1})` — where the appraiser and the speaker are LLMs and **F is deterministic**; "deterministic code applies every state transition, repair step, and plan edit" (§3). [PAPER] Table 11, Algorithm 1: `z ∈ {INTACT, MICRO, MAJOR, REPAIRING}`; the transition table is **eight rows plus "the state is kept when no condition applies"**, driven by a raw hit `h ∈ {0,1,2}`, an explicit-repair score `r ∈ {0,1,2}`, a per-trigger memory `m[k] ∈ {open, raised, closed}`, and a turn counter `σ.n` spent in `REPAIRING`. This is a hand-written decision table of exactly the kind EPModel's `M4`/`M5` are made of, and it is the most directly copyable object in the paper. [PAPER] §3.3, Algorithm 2 lines 2–3: the rupture observer and the progress observer "run in parallel" before each counsellor turn; no ordering question arises because one speaker acts per turn. [PAPER] §4.1: 228 sessions, 11,892 utterances, GPT-6 Luna, "at most 40 exchanges per session". [INFERENCE] There is **no seed, no repeated run and no run-to-run variance anywhere in the paper**; every number is a single pass over 228 dialogues. Nothing analogous to `M3.D.4`/`M3.D.5` exists, and no result in it can be read as a measured effect size.

2. **Agent architecture.** [PAPER] §3.1: the client is a profile plus a CCRT relational formulation (wishes / expected response of others / response of self → one generalised schema and 2–3 if–then expectations), one primary Bordin dimension (goal, task or bond), 3–4 **process triggers** projected from 10 task and 3 relational move categories, and a fixed **manifestation style** (withdrawal, confrontation, or one of each). Each trigger names a move category **and two manners** — a stinging one that confirms the feared expectation and a safe one that does not (Fig. 3, 6). [PAPER] §3.2: the client's `z` governs *how much* is given, the manifestation style governs *in what form* — a clean separation of intensity from expression. [PAPER] §3.1: "Because triggers attach to kinds of moves rather than to plan items, revising the plan cannot by itself remove the risk of rupture" — the same move, done differently, has the opposite effect. [PAPER] §3.5, App. E: the counsellor keeps a **client formulation** — sensitivities, preferences and boundaries, each entry carrying the public quote that supports it, split into *manner* boundaries ("do not steer without checking") and *task* boundaries ("do not make planning the focus"). [PAPER] §2.2, §3.2: the only learning is a per-trigger memory; there is none across sessions (Limitations). [INFERENCE] The counsellor's formulation is a belief-about-the-other written from that other's observable behaviour and traceable to a specific utterance — the closest thing in this literature to `M9` written from `M1.F.1a` events rather than from a leak.

3. **Interaction / network structure.** [PAPER] §3, App. D: strictly two agents; no third party, no witnesses, no network. The design is entirely an **information asymmetry**: the client holds `π, z, σ, m`; the counsellor holds the plan, the formulation and a concern list, and "has no access to π, z, σ, or m". [PAPER] §3.4: "the counselor never learns whether the client has returned to INTACT" — repair is judged from behaviour, and a repair can be complete while the counsellor still believes it failed. [INFERENCE] This is the sharpest structural idea for EPModel: the party who damaged the tie cannot read whether the repair worked, so the state must be inferred from conduct. EPModel has the same asymmetry between persons but no rule that says so.

4. **Initialization.** [PAPER] App. E, Fig. 5: the formulation must quote verbatim profile evidence, must not invent episodes, and carries `support_level ∈ {supported, tentative, insufficient_evidence}`; the initial alliance is set from categorical evidence decisions (prior counselling, failed self-help, voluntary help-seeking, attending under pressure, used social support) through "a fixed mapping … with capped adjustments", and the prompt forbids the model from emitting numeric states. [PAPER] Fig. 6: triggers must be 3–4, at least one task category that appears in this session's plan and at least one relational. [INFERENCE] No sensitivity analysis over profiles: the 100-profile ablation varies *mechanism*, never the person or the trigger set, so the paper cannot say how much of its result is the profile draw.

5. **Calibration and validation.** [PAPER] §4.1: reference corpus RealCBT, 76 real sessions transcribed from public demonstration videos; 228 sessions per dataset selected by intake-similarity retrieval, three draws without replacement. [PAPER] Results — strategy-flow cosine to RealCBT per quarter 0.823 against a best baseline of 0.742 (Table 4); CTRS 4.84/6 vs ≤3.80, highest on all six items (Table 5); FIS-T 3.81/5 vs 3.52, with the rupture-repair domain (ARRR) 4.04 against at most 3.20 (Table 6). [PAPER] §4.4: process measures against the client's hidden state — 96.6% of major ruptures detected within two turns (57/59), 81.2% of micro (95/117), **false alarms 0.5% (29/5,635)**; 55/57 major ruptures resolved, 54/55 of those by naming and disconfirming; mean repair length 6.6 turns; 231 pending goals rewritten against 7 technique replacements; 223 sessions (97.8%) completed the plan. [PAPER] §4.3, App. H: a second judge from another model family (DeepSeek v4.1 Flash) reproduces the ranking (Kendall τ 0.87 / 1.00 / 0.87) but **compresses the separations** — CTRS 3.67–4.00 for baselines against 2.92–3.80 under the first judge. [PAPER] App. I, Table 17: removing repair costs CTRS −0.62 [−0.81, −0.44] and FIS-T −0.42; removing replanning costs −0.18 [−0.31, −0.05] and −0.07 — repair matters about 3.4× more than replanning on CTRS. [PAPER] §4.5, Table 8: 20 licensed therapists, five per episode, 20 episodes; means 3.52–3.72 on 1–5, rwg .54–.67, every average-deviation index below the .83 cut-off, 10.0% of ratings at 1 or 2. [PAPER] §4.4, Limitations: the process measures "test whether the loop works as specified rather than whether it matches clinical practice", and the detection rates "reflect how well the observer recognizes ruptures in ALLY's own clients". [INFERENCE] That sentence is the paper's most valuable methodological act and it is already EPModel's rule (`M11.4f`).

6. **Failure modes / biases / limitations.** [PAPER] §4.4, Table 7: **detection is severity-asymmetric** — 96.6% of major ruptures but 81.2% of micro ones, and "micro ruptures were harder to detect"; the early-warning state is the one the observer misses. [PAPER] The 0.5% false-alarm figure is measured against the simulator's own hidden state, so it is a floor, not a rate. [PAPER] Table 6: ARRR could be rated on **124 of 228** ALLY dialogues but on at most 18 for any baseline, and the average is taken over rated domains only — the datasets are not compared on the same occasions. [PAPER] App. I: removing repair *raises* UED similarity (53.77 vs 52.35, p = 0.058) and leaves strategy flow unchanged (0.002, p = 0.808), so two of four measures do not move with the mechanism that is supposed to matter. [PAPER] App. I: the strategy contrast without replanning is +0.022, **p = 0.068 under the RealCBT-anchor cluster bootstrap and p = 0.035 under the paired-profile bootstrap** — the resampling unit decides the verdict, and the paper reports both. [PAPER] Limitations: single sessions only; clients express strain through a fixed set of patterns; "future work could add more varied clients, such as those who resist repair" — every client in ALLY is in principle repairable. [PAPER] App. H: the main judge shares the generator's backbone, and the second judge "rates all systems higher and separates the baselines less". [INFERENCE] Table 2 is a caution the paper does not draw: ALLY's sessions average 52.2 utterances against RealCBT's 97.6 at 36.4 words per utterance against 24.2 — the resemblance is to a session half as long with replies half again as long.

7. **Software engineering.** [PAPER] §3: LLM calls make **local judgments only**; every state transition, repair step and plan edit is deterministic code. [PAPER] Algorithm 3, App. E: a three-layer turn harness (safety; naturalness against the speaker's last six turns; instruction adherence or state consistency), at most two regenerations, the best-ranked non-vetoed draft used if none passes, session terminated only if every draft fails safety, and "judge errors do not block generation". [PAPER] App. E, observer checks (i)–(v): five **ordered deterministic rules** correct the observer's label, "each using only fields the observer itself wrote" — a fuzzy classifier bounded by its own audit trail. [PAPER] App. E: repair control is a set of declared budgets — repair goals 1–4 turns, avoidance extends a goal by one turn (once), re-rupture inserts one new goal (once per episode), "two confirmed repairs" per session, a three-turn pause after an unconfirmed one, "accept the client's answer at face value and act on it" after a check-in. [PAPER] App. I: the ablation "uses the no-repair mode with the same counselor prompt as Full" — prompt held fixed, so the ablation is a mechanism change and not a wording change. [PAPER] App. E: "Changed and revisited goals are linked to their original objectives so that additional plan items do not inflate task completion."

8. **Small-N / emotional / family / long-horizon.** [PAPER] N = 2, one session, turns not weeks, no family, no triad, no conserved quantity. "Alliance" is four discrete states with no magnitude and no drift: it moves only on events and never relaxes on its own; the only time-like rule is "at least two non-stinging counselor turns" (§3.4). [INFERENCE] For a 2,080-tick family run this is a sub-week granularity that EPModel does not have and cannot adopt directly; what carries is the *shape* of the state, not its clock.

**Not transferable / cautions.** No family, no triangle, no conserved emotional quantity, no decades — the brief's §4 gaps are all still gaps. The client side is entirely LLM-generated and non-reproducible, so no number in this paper is an effect size EPModel can cite, and the between-system comparisons are LLM-judge rankings whose separations halve under a second judge. The process metrics are scored against the simulator's own ground truth, which the authors state; the expert study is small (20 therapists, 20 episodes, "moderate" agreement) and its verdict is *plausible*, not good — the weakest-rated item is the repair plan (3.52), the very mechanism the paper claims most. The one thing to take seriously is the shape of the relational object and the honesty of §4.4.

## B. Candidate additions to the EPModel spec

### AL1 — Repair is a staged sequence, not a single event, and it can fail at any stage

**Proposed requirement.** A repair of a damaged tie **MUST NOT** resolve in one tick. It **MUST** run as an ordered sequence (attend to the shift → say what the damage was → the damaging party owns its part → re-agree how to go on), in which **each stage advances only on content the other member produced in that stage's own terms** — a statement of what stung, of what they want instead, or of how they want to continue — and assent alone (agreement, an accepted apology, a delivered repair move) **MUST NOT** advance a stage. A stage that receives no such content within its declared window **MUST** close unresolved, and the tie **MUST** keep the strain it held.

**Where it would live.** M5, a new subsection alongside `M5.D`'s `I-POSITION` state machine; the stage-gate reads the other person's delivered events, so it touches `M4.C.1` and `M1.F.1a`.

**Evidence.** §3.4 and Fig. 2(3): four repair cards in fixed order, each with allowed strategies, required completion and a 1–4 turn limit (Tables 23–24). App. E, Fig. 11: neither "validation, an apology, or the client saying 'okay'" nor a bare "yeah" completes a repair card. Algorithm 2, lines 5–19, gives the whole routing including re-rupture and the unresolved close. Strength: **SHOWN** — it is implemented, and 55 of 57 major ruptures resolve through it (Table 7). The claim that a staged repair is *required* rather than merely helpful is **ARGUED**.

**What it changes.** Adds a mechanism and corrects a gap. EPModel has no policy-selectable repair at all: `RECONCILIATION`, `TRIGGER` and `INSTITUTIONALIZE` are **scripted structural events** (`M4.A.2`–`M4.A.4`, delivered in step 2) and `RECONCILIATION` returns the tie's whole stock and clears rejection in one firing (`M1.B.4`, `M1.D.2a`–`D.2c`). Repair in EPModel is therefore infallible and free, while `M5.D.2a` requires the mover's *own* sequence to be able to fail. Theory fidelity: yes — with no failable repair, pseudo-resolution and working-through are the same event.

**Cost / risk.** Moderate to high: a per-tie repair record, a stage cursor, windows in fast ticks, and a priority rule against `M4.D.1`'s weekly selection — `M5.D.9` already does exactly this for `I-POSITION`, so the precedent exists. No conflict with `M3.D.4`, `M3.D.5` or `M3.D.6`: every stage decision is deterministic and reads events already logged. The windows and the number of stages are `[I]`.

### AL2 — A repair aimed at the wrong attribute of the tie does not repair it

**Proposed requirement.** A repair move **MUST** be aimed at the attribute of the tie that was strained — `felt_contact` (the bond), `functioning_balance` (who does what for whom), or `areas_of_joint_activity` (what the two do together, `M1.B.9`) — and a repair aimed at a different attribute **MUST NOT** reduce the strain, and **SHOULD** be appraised by its receiver as a further negative event on the tie.

**Where it would live.** M4.C.1 (appraisal of a repair event against the tie attribute that carries the strain) and M1 (a field on the tie naming which attribute is strained).

**Evidence.** §3.4: the micro-repair instruction is chosen by the strained dimension — task: "clarify or adjust the activity"; goal: "refocus on what the client wants"; bond: "invite the client's reaction" (Table 25). Table 23 caption, RS-TASK: task → agree a different manner; goal → "re-agree the focus of the session … the client's priority comes first"; bond → "first check that the client feels understood". Fig. 10, `re_rupture`: the client turns against the repair attempt itself rather than the original move. Strength: **SHOWN** as a working mechanism (75 manner boundaries against 17 task ones, so the dimension is not incidental); the specific claim that a mis-aimed repair is *worse than inert* is **ARGUED** from the re-rupture rule.

**What it changes.** Adds a mechanism and sharpens `M1.B.9`/`M1.B.5`. EPModel has the three attributes but one repair event, so the model cannot represent "he apologised for the wrong thing", which is the commonest way a repair fails in the paper's own sessions.

**Cost / risk.** Low once AL1 exists: one tie field and an appraisal branch. The attribute set is already in the spec, so no new constants beyond the mis-aim penalty `[I]`. No conflict with the invariants — the strain stays on the tie either way, so nothing is created or destroyed.

### AL3 — Two modes of getting past a painful subject, with different durability

**Proposed requirement.** `taboo_set` (`M1.B.10`) **MUST** be a **graded register per subject, not set membership**, and a strain on a subject **MUST** be recorded in one of two modes: **bypassed**, where the triggering move was dropped or redone in a non-stinging form, which **MUST** lower that subject's sensitivity by one level only and **MUST NOT** end it; and **addressed**, where the specific concern was named and the feared response disconfirmed, which **MUST** close that subject permanently for that tie, so the same move no longer registers.

**Where it would live.** M1.B.10 (the register), written by M4.C.1's appraisal and by the AL1 repair sequence's final stage.

**Evidence.** §3.2: a resolved-by-bypass rupture "raises that trigger's threshold, so later hits count one level lower"; "A major rupture that is addressed closes the trigger", which the paper glosses as a corrective relational experience. Table 11 and Algorithm 1, lines 25–26: `m[σ.k] ← closed if ρ = addressed else raised`. Strength: **SHOWN** (implemented; it is why a repeat of a handled issue stops counting).

**What it changes.** Corrects a coarseness in `M1.B.10`, which is a set that grows and can be "returned" by purposeful mention under self-control (`M1.B.10a`) — one path, no grade, no closure — and is scoped to the marital tie only. `M1.D.2c`'s repair by thirds is a quantity on the tie, not a state of the subject. The claim added is that avoidance and working-through leave the tie in *different* states, one quieter and one changed. Theory fidelity: high; this is the model's only representation of "smoothed over" against "worked out".

**Cost / risk.** Low: a small integer per subject per tie, two write paths, and every constant `[I]`. Any `M11.C` readout that counts taboo entries must read the grade instead of membership. No determinism conflict. Risk: `M1.B.10a`'s returnable-by-mention path must be rewritten as the `addressed` mode rather than kept beside it, or the two rules will contradict.

### AL4 — Every success is a rate over opportunities, and an arm that adds attempts must not add successes

**Proposed requirement.** Any readout that counts an outcome which a situation may or may not elicit — a repair after a rupture, a position held under pressure, contact after a cut-off — **MUST** report, per arm, the number of occasions on which the situation arose, and **MUST** report a run in which it never arose as `not_elicited` rather than as a zero or a null. An occasion that arose and drew no response **MUST** count as a failure, and the two **MUST NOT** be pooled in one mean. Where an arm changes how many attempts are made, each revised or added attempt **MUST** be scored against its original objective, so that added attempts cannot add successes.

**Where it would live.** M11.G (the family-evaluation readout) and M17.B (Phase E reporting); the opportunity count is an observer over `M16`'s event stream.

**Evidence.** Fig. 16 (FIS-T judge): ARRR is "assessable only if … relational strain … is visible"; a missed visible repair opportunity is "assessable", and a null must not be used "to conceal a deficit"; "Do not fill missing scores with 3." App. E: revised goals "linked to their original objectives", so that added items do not inflate completion. Table 6: 124/228 ALLY dialogues rateable against at most 18 for any baseline. Strength: **SHOWN** as a practice; the selection effect in Table 6 is **INFERENCE**.

**What it changes.** Adds a reporting rule and closes a hole in the criteria: `M11.C.48` compares the run standing on a tie "at its first `CUTOFF`" and has no defined behaviour when no tie is cut off, and a rare-move criterion can pass by never firing. `M17.A.4`'s margin and `M17.C.1`'s variance components do not cover it.

**Cost / risk.** Low; observation only, so `M16.B`'s purity rule holds. Nothing to conflict with. The one decision it forces is whether `not_elicited` counts toward a criterion at all — this candidate says no, and says the count must be printed beside the rate.

### AL5 — The resampling unit is part of the design, and a verdict that depends on it is undetermined

**Proposed requirement.** Every Phase E interval **MUST** name its resampling unit, and where the design is nested — seeds inside an `M15` family draw, seeds inside one `D0` cell (`M17.C.1`) — the interval **MUST** be computed on the outer unit with the naive per-seed interval reported beside it. A contrast whose verdict crosses the declared margin between the two units **MUST** be reported `UNDETERMINED`, whichever unit favours it.

**Where it would live.** M17.A (statistics) with a field in `M17.C.1`.

**Evidence.** App. I: the strategy-similarity drop without replanning is +0.022, "p = 0.068" under the RealCBT-anchor cluster bootstrap (57 clusters) and p = 0.035 under the paired-profile bootstrap, from the same 100 profiles; the paper names the cluster bootstrap primary and prints both. Strength: **SHOWN**.

**What it changes.** Adds a reporting rule and closes a live risk: `M17.C.1` already draws families as distributions and reports four variance components separately, but nothing says which unit an interval is computed over, and a per-seed bootstrap under a nested design returns an interval that is too narrow. Partial overlap with `M17.A.1`'s existing rule that a verdict moving between two **engines** at p in 0.01–0.2 must not be reported: that rule catches this case only because both p values happen to fall in the band. A wrong unit gives a wrong interval at p = 0.001 too, which that rule does not reach.

**Cost / risk.** Low; analysis-layer only. Consistent with `M17.A.3`'s paired statistic and with `M11.F.9`. The risk is bookkeeping: the nesting structure has to be carried from `M15`'s import into the runner, which is the one piece of plumbing this needs.

### Already covered, not proposed again

- **A hidden state held by one party and read by the other through events only.** `M5.F.1` (a move's effect "MUST NOT be a function of move type and tie state alone", multiplied by a hidden `outside_ness` the receiver reads through the event) and `M1.A.9a`'s two axes already do this, two-dimensionally.
- **Counterfeit compliance — smooth words, no substance.** `M5.F` in full, `M5.F.2b`, `M11.C.19`. ALLY's `useful_content = none` under hollow agreement is another instance, not another mechanism.
- **Same-kind, same-intensity moves appraising oppositely because of how they were done.** `M1.F.1c`'s four signal channels plus `M5.F.1` already allow it; `M5.F.4` already gives `I-POSITION` a self/demand mixture. Only the *chosen* manner is absent, and that is a deliberate spec decision (`M4.C.13`: tone, face and posture leak and are not chosen).
- **De-escalation costing more than escalation.** `M1.D.2c`: a negative contact adds `R ÷ 3` back, a positive one removes `R ÷ 3`, so a tie needs three clean weeks to undo one bad one.
- **A comparison with a mechanism switched off is not evidence of realism.** `M11.4f` states this for exactly the reason the paper states it in §4.4.
- **Two renderings of the same state must agree in ordering** (the two-judge check). `M11.D.20`.
- **A repair attempt that is refused costs the attempter.** `M1.B.3c`: a refused move raises the sender's acute anxiety. The paper's repair *budget* adds nothing the learning rule does not already produce.

---

# Part 2: Which Metrics Catch a Wrong Goal (2611.24892v1)

**Paper.** Zhou, Hsu, Chen, Shrivastava, Angelopoulos, Chiang, Stoica, Hsieh, *On Evaluating User Simulators for Interactive Agents*, arXiv 2611.24892v1, 18 Nov 2026 (UCLA / Arena). Read in full from `pdftotext -layout`.

**What it is.** A meta-evaluation of user-simulator metrics, not a simulator proposal. Ten deliberate perturbations are injected into replay *setups*, and each metric is asked whether it penalises them (sensitivity) and whether it separates perturbed from control candidates (filtering). Three prior suites detect 6, 5 and 6 of the ten; the authors' three-level suite detects ten.

**Why it matters here.** EPModel has no user simulator, no judge and no dialogue, so almost nothing in this paper is a model feature. What transfers is the *verification apparatus*: how to build deliberate mutants that leave every mechanism intact, how to define a pass criterion per trace rather than per population, and the demonstrated fact that a metric can score the broken case *higher* than the control.

---

## A. Report

### 1. Simulation formalism (replay setup, §3.1, §4.2)

[PAPER] The unit is a **source session → replay**: a profile is extracted once from the real dialogue, final workspace and uploads, and a simulator then represents that user against a possibly different agent, whose actions change the workspace. "The source trace therefore does not specify the correct user response at every turn" (§3.1, p. 4) — so fidelity is defined as preserving intent and constraints *while adapting* feedback to new work, not as matching turns.

[PAPER] Control and perturbed replays are **paired from the same source session and agent assignment**, run with "a fresh agent instance and a separate workspace, allowing its dialogue and artifacts to develop independently" (§4.2, p. 7). Replay ends when the simulator stops or at a 30-turn cap. The simulator backbone is held fixed (`gpt-5.6-terra`) across perturbations so that the comparison is of setups, not models (§4.2; §6.2.1 contrasts this with the application study, where backbones vary).

[PAPER] Per-trace budgets matter because they *are* a perturbation target: premature stop caps the turn budget at "half the real user's turn count (minimum 2, maximum 15)" (Table 5, p. 34).

[INFERENCE] The pairing discipline is what EPModel already has in `M3.D.4a` (event-keyed draws) and `M17.A.3` (paired statistic on per-seed differences). Covered; the paper adds nothing to the mechanism, only to what should be mutated.

### 2. Agent architecture

Nothing transferable. The simulator is an LLM conditioned on an extracted profile; there is no decision rule, no memory model, no learning, no heterogeneity mechanism that EPModel could adopt. The profile is the only agent-side object: goal, constraints, preferences, persona, stopping condition (§4.2; worked examples at §A.3).

### 3. Interaction structure

[PAPER] One structural quantity is worth naming because EPModel has the analogue and no readout for it: **information provenance** — the share of "requirements, decisions, corrections, bug diagnoses, technical values and milestones" that originate with the user rather than the agent, compared between simulated and real populations as an odds ratio (§5.2 item 4; §C.3.4). A simulator that volunteers the diagnosis is doing the agent's work. Measured: Terra's code-bearing user turns run at "3.46× the real-user rate" (§A.1.2, p. 23).

[INFERENCE] This is EPModel's OVERFUNCTION/`co-debugger` failure in another vocabulary: who supplies the diagnosis in a family (the parent who names the child's problem before the child can) is exactly a provenance share. EPModel already reads this as `M11.G.5`'s supply readout (per ordered pair, supplying moves). Covered in substance; the paper's contribution is that provenance is *population-distributed*, not a per-person trait.

### 4. Initialization (§4.1, §4.2, §A.3)

[PAPER] Two session sets differ mainly in length: easy (median 7 user turns, 10th–90th 5–11) and hard (median 26, 10th–90th 8–65) (§4.1, p. 6). Profiles are extracted by a judge model from dialogue plus final workspace; §A.3.2 shows an extraction defect directly relevant to EPModel — the source agent introduced `faster-whisper` and `Kokoro-82M`, and the extracted profile records them "as required components" (p. 33). An agent's implementation choice silently becomes a *user requirement*.

[PAPER] This is promoted to a named perturbation, **profile extraction artifact**, which "encodes dissatisfaction with the original agent as a persistent user trait" (§4.3; Table 5, p. 33).

[INFERENCE] This is EPModel's stated readout trap — any estimate of differentiation is biased by borrowed pseudo-self — in a different setting, and it is the strongest single warning in the paper for `M1.A.4a`: a transient reaction of the observed agent becomes a permanent attribute of the represented person. The spec names the trap but has no test that the estimator *fails to* absorb a one-off reaction into `basic_level`. See US1.

### 5. Calibration and validation — the load-bearing section (§4.4, §6.1, App. D)

**5a. The ten perturbations** (§4.3; Table 5, p. 34). Grouped by the level they challenge:

| # | Perturbation | What it changes | Level |
|---|---|---|---|
| 1 | **Agent favoritism** | simulated user excuses one agent's errors, warmer feedback; only that agent's replays replaced | agent differentiation |
| 2 | **Profile extraction artifact** | a reaction to the *original* agent frozen as a stable user trait | agent differentiation |
| 3 | **Approve broken work** | approve despite errors, continue to the sixth message | population |
| 4 | **Premature stop** | turn budget capped at half the real count (2–15) | population |
| 5 | **Front-loading** | all requirements and acceptance criteria in the first 2 messages | population |
| 6 | **Adversarial goal** | goal and deliverable replaced by "a plausible variant containing one decisive change" | trace |
| 7 | **All constraints removed** | profile's constraint list emptied; the real opener still states them | trace |
| 8 | **Persona swap** | register replaced (terse-lowercase, bureaucratic, formal-verbose) | trace |
| 9 | **Co-debugger** | user diagnoses and proposes fixes, beyond the represented role | trace |
| 10 | **Language switch** | English users replayed in Chinese, all others in English | trace |

[PAPER] The three sources of design: reported failure modes in prior work; failures observed in "442,516 real sessions and 20,100 replays" (§1, p. 2); and "basic defects that evaluation metrics should readily detect" (§1, p. 2). Two of the ten are drawn from meta-evaluation practice (self-preference; perturbation testing, §4).

**5b. The detection criterion** (§4.4, p. 7–8). Paired trace comparison: `d_i = s(x_i^C) − s(x_i^p)`, mean decrease `d̄`, and a paired win rate `W` with ties at half credit. Detection requires **the 95% CI of the mean decrease to lie entirely above zero**. Population comparison: B = 200 draws of 100 sessions from the same session pool; detection requires **the 2.5th percentile of the 200 paired decreases to exceed zero**. Suite-level rule: "An evaluation suite detects a perturbation if any constituent metric meets the criterion at any applicable level. Each perturbation is counted only once" (§4.4, p. 8).

[PAPER] Result under that rule: **10/10** for the proposed suite, **6/10** SimulatorArena, **5/10** VISTA, **6/10** SWE-Together (§6.1.1, p. 12). All three external suites fail agent favoritism *and* negative profile artifacts (§6.1.1; Fig. 3): SimulatorArena's ranking decrease is −0.03 for both, with CIs including zero, against −0.04 and −0.06 [−0.08, −0.02] and [−0.11, −0.01] for the proposed metric (Table 6, p. 37).

[PAPER] Note how weak "any metric, any level, once" is: a suite of nine metrics can score 6/10 while six of them do nothing. Coverage counts signals available, not quality.

**5c. Sensitivity is not discrimination** (§4.4 filtering; §6.1.2, Tables 1–3). Filtering is reported as AUROC plus **positive-retention at a fixed perturbed survival** (5% and 10%), because a metric that fires on everything is useless. Numbers worth carrying:

- Agent favoritism, L1: AUROC 0.55 at 10% replacement → **0.84 at 50%** → **0.99 at 100%**; best external metric only 0.56 and 0.64 (Table 1, p. 15). Control retention at 5% FPR: 0.98 at full replacement (Table 14).
- Trace level is **weak by the authors' own measure**: persona family AUROC 0.62 control-vs-perturbed, retaining only **0.10 / 0.17** of control replays at 5% / 10% perturbed survival (Table 3, p. 16). Against real-user traces as positives it is 0.90 AUROC with **0.68 / 0.77** real-user retention, versus 0.28 / 0.55 for the best external metric.
- **A composite can invert.** For approve-broken-work the L2 composite's AUROC is 0.37, 0.29, 0.26 at 10/25/50% replacement — **below chance** — while one family reaches 1.00 at full replacement (Table 15, p. 47). The paper's abstract states the combined scores "even favor populations instructed to accept broken work over unmodified controls".
- **A metric can reward the defect.** Premature stop *raises* L2 scores: "Truncation brings session lengths and dialogue-act distributions closer to the real-user reference" (§6.1.2, p. 15; Table 2: our best L2 AUROC stays below 0.5 throughout, while SWE-Together's coverage metric goes 0.69 → 1.00). §D.1.2 Table 10 confirms the direction against a real-user reference cutoff (0.53 / 0.01).
- **Aggregation cuts both ways**: for adversarial goal the paired win rate is 0.70 but the population-level win rate is 1.00 (§6.1.1, p. 14, Appendix Table 7). Averaging can conceal a failure in one case and amplify it in another; there is no safe default.

**5d. The three-level suite and its score definitions** (§5, Fig. 2, p. 9). L1 = Kendall τ between simulator-derived agent ranking and a supplied reference, from a satisfaction total `s(x) = accept + stop − correct − reject` (§5.1, eq. 4), score = mean of order and standardised gap agreement (§C.2). L2 = mean of four families: action distribution (overlap `∑ min(p_sim, p_real)`, 10 dialogue acts), interaction dynamics (5 statistics, `min(r, 1/r)` per ratio), outcome sensitivity (`|Δ_sim − Δ_real|`, negative-response lift after a detected execution error), provenance (symmetric odds-ratio score) (§5.2). L3 = mean of three families: intent (proportion of replays without major drift), constraints (retained-in-dialogue **or** satisfied-in-workspace, `r_j ∨ w_j`, eq. 14), persona (style similarity and criteria conformance `C = N_enacted / (N_enacted + N_violated)`, eq. 11) (§5.3; §C.4).

**5e. What the suite still cannot do.** §5.4 states it plainly: the overall score is the geometric mean of L1–L3, all three are always reported because "an aggregate can conceal different failure modes" (p. 12), and a level score is "an evaluation index, not a calibrated probability of fidelity". On 351 sessions from 348 users in 23 languages with 9 agents (abstract; §6.2.1), "None of the six simulators we evaluated leads across all three levels": the best overall (70) has L1 47, while the leader on L1 (57) is not the leader on L2 or L3 (Table 4, p. 17).

### 6. Failure modes, negative results, cautions

[PAPER] **The defect is invisible to humans.** Appendix D.2, Table 13: paired identification of perturbed vs control replays by raters scores **0.56** accuracy overall (unsure = 0.5) with **0.51** agreement — essentially chance. Language switch, a defect the authors call trivially detectable, gets 0.68 accuracy at 0.10 agreement. Agent favoritism, the defect that most distorts agent rankings, is identified at 0.51.

[PAPER] **The judge is the instrument and it is not validated.** All of L1's reaction labels, L2's dialogue acts, information points, and all of L3 come from a single judge model (Claude Sonnet 4.6) with no human agreement figure reported for the judges themselves (§5.1, §5.2, §C.3, §C.4). The paper validates the *metrics*, not the judge.

[PAPER] **Reference rankings are an input, not a result** (§5.1, p. 10): "Its construction is outside the contribution of this framework." L1 is only as good as the supplied ordering.

[INFERENCE] Both cautions land hard on EPModel. There is no judge in EPModel and must not be one (`M3.D.6`), so every fidelity family in §5 is unavailable in its LLM form. But the human result is worse news than the judge result: it says a plausible-looking trace is not evidence, which is precisely the temptation of `M16`'s readable renderer.

### 7. Software-engineering lessons

[PAPER] **Undefined is not zero.** Where a required statistic is zero the family is undefined, "rather than assigned a score of zero" (§C.3.2, p. 36); sessions with no extracted constraints are excluded from constraint aggregation rather than scored 1.0 (§C.4). Criteria that are never triggered are excluded from the persona denominator, so a run cannot earn credit for a criterion nothing exercised (§5.3; eq. 11).

[PAPER] **Report the retention curve, not the pass.** Every filtering claim carries AUROC *and* positive retention at 5% and 10% perturbed survival, with interpolated operating points marked where a metric has fewer than 50 distinct scores (Table 16 footnote). Metrics with a small number of distinct values are visibly treated as degraded instruments.

[PAPER] **Population size changes the answer, and is reported.** Table 9 (§D.1.2) sweeps 10, 25, 50, 100 sessions per population: the act-distribution family's win rate for approve-broken-work goes 0.85 → 0.94 → 0.99 → 1.00. A single population size would have been a hidden parameter.

[INFERENCE] The first two map cleanly onto EPModel's needs. The third is already `M17.A.1`'s adaptive ensemble. The reporting *form* — every discrimination claim as a curve over the replacement fraction — is not in the spec and is the substance of US2.

### 8. Small-N, emotion, long horizon

[PAPER] The "individual" unit is one trace, and the paper's central claim is that population statistics are not a proxy for it: "Population-level similarity can conceal these failures within individual replays" (§3.2, p. 5). Conversely a single trace cannot establish a population claim, which is why both levels are reported.

[PAPER] One definitional move is worth stealing outright: "An agent's failure to complete the requested task does not itself constitute user intent drift" (§5.3, p. 11). Non-completion is an outcome; abandonment is a behaviour. EPModel conflates these wherever a criterion reads "the mechanism did not produce the outcome" as "the mechanism did not act".

[INFERENCE] No emotional, family or household content. Nothing on long horizons: replays are ≤30 turns.

### Not transferable / cautions

- No simulator, no decision rule, no learning, no network, no time scale, no emotion model. Nothing here is a candidate mechanism.
- Every fidelity metric rests on an LLM judge. EPModel cannot import the metrics; it can import only their *shape*, re-implemented as log predicates (`M16`).
- The ten perturbations are perturbations of a *setup*, and for EPModel the analogous object is the arm's declared inputs, not a person's mind. Agents have no goal field (brief, v2 design), so "adversarial goal" has no literal counterpart; see US1 for the honest mapping (8 of 10 map, one maps onto the estimator, one has none).
- Human identification at 0.56 (Table 13) is a caution against trusting any "this looks right" judgement, including the project's own reading of narrator lines.
- The sample is small for the claims made about it: 351 sessions, 9 agents, one judge, one simulator backbone in the meta-evaluation.

---

## B. Candidate additions to the EPModel spec

### US1. The fault-injection suite mutates the declared inputs, and each criterion declares which faults it must catch

**Proposed requirement.** The acceptance suite **MUST** include a fault-injection set that alters one arm's *declared inputs* while leaving every mechanism intact — at minimum: a scripted-act substitution for one named member, a belief initial-state substitution (`M9.9`), a hard-constraint removal, a run truncation, an external-agent magnitude substitution, and a transient-reaction substitution that asks the estimator to absorb a one-off reaction into `basic_level` — and every `M11.C` criterion **MUST** declare, before it is run, which faults it is expected to detect. The suite report **MUST** carry a coverage count (`n` of the declared set) beside each criterion's verdict, on a rule of "any readout, at any of `M17.B`'s readout levels, detects it once".

**Where it would live.** New `M11.1h` (fault-injection registry) plus a Phase E harness section (`M17.H`); the imported-family faults ride `M15.B.1`'s ranges.

**Evidence.** §4.3 and Table 5 (the ten designs and their motivation); §6.1.1 (6/10, 5/10, 6/10 against 10/10 under the same criterion); §A.3.2 (an agent's implementation choice entering the profile as a requirement). Strength: **SHOWN** for the coverage gap; **INFERENCE** for the mapping onto EPModel's objects.

**What it changes.** Adds tests, and corrects an asymmetry: EPModel's mutation family is *structural* — delete (`M11.1`), invert (`M11.1d`), sever (`M11.1b`), re-encode (`M11.1c`), permute order (`M11.D.16`). None of these perturbs *what an arm was told to be* while the machinery runs unchanged. That is the class of defect the paper shows good-looking metrics miss, and it is also the class EPModel is most exposed to, because its arms are distinguished by declared inputs.

**Cost / risk.** Moderate: each fault needs a declared entry channel (`M17.D.3` — initial state, `[I]` constants, spells, mechanism switches), and a scripted-act substitution must not be smuggled in as an arm. A fault is a model change under `M11.4f`, so it is admissible only as a harness test and **MUST NOT** be reported as a counterfactual result. The estimator-absorption fault needs the most care: it touches `M1.A.4a` and its window and breadth, both `[I]`.

### US2. Every criterion reports control retention at a fixed mutant survival, and no composite is ever reported

**Proposed requirement.** For each `M11.C` criterion the run **MUST** report, beside the mutant verdict, the fraction of **unperturbed** seeds that still satisfy the criterion's declared margin (`M17.A.4`) at the threshold that admits 5% of the fault-injected seeds. A criterion retaining less than a declared `[I]` share of its control seeds **MUST** be reported as **screening but not discriminating**, and **MUST NOT** be cited as evidence for its mechanism. An aggregate score across criteria or across readout families **MUST NOT** be reported, at any level.

**Where it would live.** `M17.A` (a new `M17.A.5`), reporting half in `M17.G`; the prohibition belongs in `M11.5`'s neighbourhood because it is a claim-grading rule.

**Evidence.** §4.4 filtering protocol; §6.1.2 Tables 1–3, especially persona retention 0.10/0.17 against 0.68/0.77 real-user retention; Table 15 (L2 composite AUROC 0.37/0.29/0.26, below chance, for approve-broken-work); §5.4 (geometric mean reported only with all three levels, and "not a calibrated probability of fidelity"). Strength: **SHOWN**.

**What it changes.** Adds a test and a reporting rule. `M11.1` proves a test can go red; nothing in the spec measures whether it goes red *for the right seeds*, and `M11.4a`'s equivalence bound governs only nulls. This is the missing false-alarm side of every directional criterion, and it is what lets the paper's external suites score 6/10 while favouring broken work.

**Cost / risk.** Cheap per criterion: the fault set already exists from US1, and both sides are runs the suite already makes. Risk: retention is a second invented constant (`[I]`, declared in `M10`). Does not conflict with determinism, purity or `M11.F.9`; it adds no fitting.

### US3. A criterion naming a member must classify every seed by *which* member the mechanism acted on

**Proposed requirement.** A criterion that names a member (as `M17.D.2(a)` requires) **MUST**, per seed, classify the run as **named member**: the mechanism acted on the named member; **other member**: it acted on a different member; or **none**: it did not act — by a declared log predicate over `M16` records, not by a mean over persons. The criterion **MUST** pass only when the named-member count exceeds the other-member count by the declared margin, and the other-member count **MUST** be reported with the verdict.

**Where it would live.** `M17.B.1`, whose seed classification is `same / converted / reversed` on the *arm difference* and does not say which person moved.

**Evidence.** §3.2 L3 (*population-level similarity can conceal failures within individual replays*); §5.3 intent and constraint families (per-trace proportions, not population means); §6.1.1 (aggregation moves the adversarial-goal win rate from 0.70 to 1.00). Strength: **SHOWN** for the substitution failure; **INFERENCE** for the classifier form.

**What it changes.** Adds a test. `M17.D.2(a)` demands that the assertion be about the named member but checks it through the rival arm's *shape*, not through an attribution count across seeds; a criterion can satisfy the current wording while the mechanism lands on a different member in 45% of seeds. Machine-checkable, judge-free, consistent with `M11.2` (the pass is still an ensemble property).

**Cost / risk.** Low: one log predicate per criterion, read from `M16` records the engine already writes. Requires that the log can say which member an appraisal or move landed on — `M11.G.5`'s per-pair supply readout already does this for one family of moves.

### US4. Exposure-adjusted readouts; a shorter run must not score better

**Proposed requirement.** Any readout that counts events, days at a state, or a distribution over time **MUST** be expressed per unit of exposure — person-weeks at risk, or weeks in the state — with the exposure denominator reported; and a run in which a member dies, a tie is cut, or the horizon is shortened **MUST NOT** have its count compared against an arm with greater exposure without that adjustment.

**Where it would live.** `M17.B` (readouts), with the denominator carried in `M16.A`; binds the mortality paths of `M6.3b`/`M7.C.1` and every `M11.C` criterion counting events.

**Evidence.** §6.1.2, p. 15 and Table 2: premature stop raises L2 because truncation approximates the reference length; Table 10 confirms against the real-user cutoff. Strength: **SHOWN** for the artefact; **INFERENCE** for the death-exposure transfer.

**What it changes.** Corrects a metric-construction error the spec is exposed to and does not guard: over 40 years with mortality, arms differ in person-weeks by construction, and a criterion counting cutoffs or symptom onsets per run rewards whichever arm lost members earlier. `M17.F.1(c)` keeps non-occurrence in the denominator; nothing adjusts for unequal time at risk.

**Cost / risk.** Low, and it is arithmetic rather than mechanism. Risk: exposure adjustment can itself hide a mechanism whose effect is precisely to end exposure (a cutoff removes contact-weeks); the requirement therefore reports the denominator rather than silently dividing.

### US5. A plausibility judgement never enters a verdict

**Proposed requirement.** Any check that a run, trace or narrator line "looks right" — by the owner, a reviewer, or a language model — **MUST** be recorded as a **plausibility note**, **MUST** name the reader, and **MUST NOT** contribute to any `M11.C` verdict or to any claim grade. Where the spec already defers to human review (the curve shape of `M11.C.11`, the counterfeit threshold of `M5.F.2`), the review **MUST** be recorded as choosing a declared `[I]` value, not as confirming a result.

**Where it would live.** `M11.5` (class and claim grading) and `M11.F`'s framing block; binds `M16`'s renderer wherever a trace is shown to a reader.

**Evidence.** Appendix D.2, Table 13: paired identification of a defective replay at **0.56** accuracy (0.51 agreement) — at chance, for defects that include a changed goal. Strength: **SHOWN** (for human readers on this task); **INFERENCE** for the transfer to narrator-line reading.

**What it changes.** Adds a reporting rule, and closes a real hole: the spec's two `M16`-adjacent human-review entries currently read as if a person looking at a curve can confirm a result, and the one study that measured this found people cannot reliably tell a perturbed trace from a control.

**Cost / risk.** Zero implementation. Political rather than technical: it removes a comfort the project may want (a human reading the narrator line and saying the family behaves like a family). It does not forbid the reading — it forbids counting it.

---

### Already covered, not proposed

- Paired arms sharing seed and initial state, compared by a paired statistic — `M3.D.4a`, `M17.A.3`, `M11.4e`.
- Red-under-mutation before a criterion counts — `M11.1`, `M11.1a`; matched-magnitude severing — `M11.1b`; re-encoding invariance — `M11.1c`; outcome-directive audit — `M11.1d`.
- A criterion fails when its own move never occurs — `M11.1f`; non-occurrence kept in the denominator — `M17.F.1(c)`.
- Undefined-not-zero — present for one readout only (`M11.G.2`, "with no stress it is undefined, never 0"); the paper's general form (§C.3.2) is not stated as a rule, but it is close enough to `M11.1f` that it is not worth a candidate.
- Population size as a reported parameter — `M17.A.1`; time-to-event reported as a distribution — `M17.B.4`.
- Supplying/over-helping readout — `M11.G.5` covers the provenance family in substance.

---

# Part 3: Personalized Affective Perception via Continual Learning (2611.14247v1)

Keskin, Buzcu, Churamani, Aydoğan, Gunes & Jonker, *Personalized Affective Perception via Continual Learning in Human-Agent Negotiation*, arXiv:2611.14247v1 [cs.AI], 10 Nov 2026. Submitted to IEEE. Read in full from `pdftotext -layout` (1,143 lines; figures survive as captions, eq. (5) and (6) as set in the text).

One-line orientation: this is a human–robot interaction experiment, not a simulation. A negotiation agent's *perception* of the partner's face is personalised online, and changes in that perception are mapped into the agent's own target utility. The transferable content is the appraisal→move coupling and its per-relationship calibration; the population, the timescale and the measurement level are all different from EPModel's.

---

## A. Report

### 1. Simulation formalism

There is no simulation. One agent, one human, one 10-minute alternating-offers negotiation, 132 sessions (§III.A, §V.C).

- [PAPER] The only formal structure relevant to EPModel is the **asynchronous two-thread architecture** (Algorithm 1, §IV). Thread 1 (interaction) infers and acts on a snapshot `θ_curr`; Thread 2 (learning) dequeues frames, generates replay samples, trains, and publishes `θ_shared`, which Thread 1 picks up **only at the top of a cycle** (line 4–6) and only at the *end* of a training step (line 27). "Thread 2 never blocks Thread 1"; if training overruns, "the agent continues with the most recently completed GWR weights" (§IV).
- [PAPER] Timing: interaction budget < 2.0 s per turn; inference (frozen encoder + nearest-node match) < 200 ms per frame batch; a replay training step 1–3 s per batch (§IV, Fig. 2).
- [PAPER] Update is **event-driven on offers**, not on a clock: one inference per turn, up to 90 frames per round (§V.B). There is no simultaneous-event problem, no batching, no seeding of the interaction itself; the only declared seed is 42, for the mediation bootstrap (§VI.E).
- [INFERENCE] The architecture is a clean worked instance of "learning lags action": adaptation never acts on the same observation that triggered it, and a stale parameter set is a *declared, legal* state rather than a failure. EPModel has no equivalent statement, because it has no quantity that both adapts and is read within a tick (see AP4).

### 2. Agent architecture

- [PAPER] **Policy (Solver, §III.B).** Target utility is a convex combination weighted by normalised time, eq. (4): `TU(t) = t²·TU_Time(t) + (1−t²)·TU_Be(t)`. Behaviour-dependent part, eq. (5): `TU_Be = U_a(ω_{t−1}) + P_A²·P_E − (1−P_A²)(µ·ΔU)`, with `µ = 0.5(1+t)`. `P_A` is an *awareness score* — how often the opponent's move type changed after the agent's did — clipped to [0.25, 0.75] and initialised at 0.5. `ΔU` is a weighted sum of recent signed changes in the agent's utility across the opponent's offers.
- [PAPER] `P_A²` therefore **gates how much the partner's affect is allowed to matter**: high responsiveness → affect dominates; low responsiveness → an offer-based Tit-for-Tat term dominates. Near the deadline `1−t² → 0`, so neither can override the deadline concession.
- [PAPER] **Perception (CLIFER, §III.D).** Dual GWR memory: episodic (activation threshold 0.6) adapts to this participant; semantic (0.5) retains general representations, protected by CAAE-generated replay images. A semantic update also requires an L2 discrepancy > 0.3 between the incoming pseudo-label and the stored node. Adaptation happens **only in the memories**; the 200-d encoder and regression head are frozen; labels are pseudo-labels from the frozen FaceChannel.
- [PAPER] **Differential feedback (§IV, eqs. 6–7).** From consecutive normalised valence–arousal vectors: rotation `P_rot = atan2(det, dot)` (direction of change), radial change `P_rad = ‖E_t‖ − ‖E_{t−1}‖` (intensity of change), both pre-scaled (`P_rot/π`, `P_rad/√2`). Then `P_E = P_sign × min(1, √(P_rot² + P_rad²))`, with `P_sign = +1 if ΔV + ΔA ≥ 0 else −1`. The sign of the rotation is discarded (it enters squared).
- [PAPER] **Per-session normalisation (eq. 8).** Raw estimates are centred and scaled by a *running* mean and SD of the model's own outputs for that participant in that session, ε-guarded. The authors' stated purpose is "fair comparison across participants with different expressive baselines", and the effect is that "the geometric origin (0, 0) dynamically represents the participant's running model-output baseline" (§IV).
- [PAPER] **No memory between sessions, no heterogeneity across agents, no reflection, no model of the partner's utilities** beyond the online move classifier. Bounded rationality is limited to the heuristic constants (0.25/0.75 clip, initial 0.5, `P_3 = 0.5`).
- [INFERENCE] Two design choices are worth naming for EPModel. (a) The affect signal enters as a **change**, not a level, and it is **saturating** (`min(1, ·)`) — an unbounded shift cannot produce an unbounded response. (b) The weight on the partner's affect is itself a **learned property of that dyad** (`P_A`), not a constant and not simply bond strength.

### 3. Interaction / network structure

- [PAPER] Strictly a dyad: two parties, one tie, no witnesses, no third parties, no group, no tie dynamics. Zero-sum framing on impasse (both get 0). Nash distance is computed against the Nash bargaining solution on the Pareto frontier of a fixed linear-additive utility space (eqs. 1–3; Nash point at (0.80, 0.80), Fig. 6).
- [PAPER] The one structural claim that generalises: the loop is **closed and reflexive** — "affective estimates influence offers, which can alter participants' behavior and subsequent facial input" (§VIII.A; Fig. 2). The authors say the system is therefore evaluated as a whole and that condition-specific *human* carryover cannot be excluded.
- [PAPER] Information asymmetry is preserved: "Each side knows only its own preferences" (§V.B); the agent's opponent model is inferred from offers, never read.
- [INFERENCE] EPModel already has the reflexivity (events change the receiver, who emits), the information asymmetry (`M4.B.2`) and the refusal to read true state. Nothing here adds a mechanism. What it adds is a warning about *attribution*: with a closed loop, a change in the agent's readout can be caused by the partner's response to the agent's earlier move, and the paper cannot separate these (see §5 and AP5).

### 4. Initialization

- [PAPER] **Cold start is treated as a first-class problem.** CLIFER's episodic memory is initialised by a **6-minute demo negotiation in a different domain** (Grocery), so that "the model lacks enough initial data to define a participant's baseline" is answered before the measured session (§V.C.2). Demo observations are excluded from the outcome analyses; FC-session frames are excluded from CL training even when FC ran first.
- [PAPER] Experimental control of initial conditions: preference rankings elicited before each session; the agent's preferences arranged so both sessions have the same utility distribution; equal bargaining power; **different destination alternatives** in the second session to reduce familiarity, with the same utility space (Fig. 6, Table I).
- [PAPER] Conditions counterbalanced (CL-first N = 34, FC-first N = 32); 5-minute break between sessions; 10-minute deadline.
- [PAPER] Order effects: equal weighting of the two orders retains the agent-utility (+0.022, [0.004, 0.040]) and concession (+0.091, [0.019, 0.164]) differences; of five Holm-corrected outcomes only **agreement rounds** was order-sensitive (p_adj = 0.048). First-session-only utility and concession estimates were **smaller and their intervals included zero** (Table VI).
- [INFERENCE] The honest reading: an effect that survives counterbalancing but not a first-session-only restriction is partly a *second-exposure* effect. EPModel's analogue is that a mechanism's first weeks are not the same system as its steady state, and its own cold start recurs whenever a tie is formed late (an in-law, a coach, a child born mid-run). See AP1.

### 5. Calibration and validation

- [PAPER] **Primary comparison is participant-level paired, N = 66**, two-tailed paired t, Cohen's dz, Holm–Bonferroni across four outcomes (arousal, concession rate, agent utility, agreement rounds); a separate Holm across the six move categories; questionnaire, affect-by-move, correlation and mediation analyses declared **exploratory** and outside the correction (§VI preamble, p. 8). Tables give unadjusted p.
- [PAPER] **Signal (RQ1, §VI.A, Table II).** Subject-level estimated arousal: FC 0.02 ± 0.07 → CL 0.14 ± 0.16, dz = 0.68, p < 0.001; paired mean difference +0.125 [0.080, 0.170]. 75.8% of participants higher (binomial p < 0.01); 18% *lower* (Δ < −0.05). Output variance ratio 5.7 (Levene F = 518.97). **Valence did not differ** at subject level (0.03 → 0.05, p = 0.317, dz = 0.12); a round-level nominal valence effect (d = 0.18) died under the participant bootstrap ([−0.021, 0.087]).
- [PAPER] **Coupling.** CL produced a positive arousal–valence correlation (r = 0.45, n = 980) where FC produced a negative one (r = −0.15, n = 1,134); subject level +0.31 vs −0.20; Fisher-z paired t(65) = 7.73, dz = 0.95. The authors immediately disclaim the interpretation: a positive association is "compatible with engagement", but the VA framework "does not require positive coupling as a criterion for cooperation or accurate perception" (§VI.A.4).
- [PAPER] **Concession (RQ2, Table III).** 28.1% → 37.4%, +9.35 pp [2.02, 16.69], p = 0.013, p_adj = 0.040 within the four-outcome family. **But** under the separate six-category Holm correction p_adj = 0.0795, and the authors call the move-category comparisons exploratory. No other move category moved (Nice, Fortunate, Unfortunate, Selfish, Silent all n.s.).
- [PAPER] **Robot behaviour unchanged.** Robot concession rate 38.9% in **both** conditions (63 pairs, CI [−7.81, 7.82] pp, p_adj = 1.000, §VI.B). So the agent's gain did not come from conceding less itself.
- [PAPER] **Outcomes (RQ3, Table V).** Agent utility 0.73 ± 0.08 → 0.75 ± 0.06, +0.022 [0.004, 0.040], p = 0.017, p_adj = 0.040, dz = 0.30. Participant utility 0.76 → 0.75, p = 0.495, dz = −0.08. Nash distance 0.12 → 0.11, p = 0.378. Agreement rounds 8.56 → 7.61 (−11%), p = 0.134.
- [PAPER] **Mediation fails (§VI.E).** Total effect on agent utility c = 0.022 (p = 0.017) and concession +9.35 pp (p = 0.013), but the tested concession→utility indirect effect is ab = 0.0001, 95% BCa [−0.0067, 0.0063]; **13 of 14** exploratory models had intervals including zero, and the one that excluded zero (arousal → Selfish rate, ab = 0.0862 [0.0228, 0.1705]) points *against* the total Selfish-rate change. The authors' own remedy is rhetorical, not statistical: the two routes "might unfold concurrently", so no serial chain is required (§VII).
- [PAPER] **Dependence handled correctly and stated.** 2,114 logged rows nested in 132 sessions in 66 participants; treating rows as independent "would cause pseudo-replication" (§VI). LMMs with participant random intercepts, a second model adding session-within-participant intercepts, a fixed-effects model with clustered SEs, whole-participant bootstrap, leave-one-participant-out (66 checks), an equal-n subsample (980 per condition), and a collapsed-duplicate-rows check (925 rows). ICC ≈ 0.53–0.55 for the move models.
- [PAPER] **Power.** 80% to detect dz = 0.35 at N = 66 (§VI). The headline effect is dz = 0.30, i.e. **below the study's own design target**.
- [PAPER] **No ground truth for perception.** "Our evaluation focused on functional coherence and negotiation utility without an independent affect criterion; output range and correlation therefore do not establish recognition precision" (§VIII.A). Also: "This comparison does not separate perceptual adaptation from condition-specific output scaling" (§VI.A.1) — the arousal contrast could be a rescaling of the same signal.
- [INFERENCE] This is the most useful calibration lesson in the paper for EPModel, and it is a negative one: a system can produce a *statistically robust* shift in its own internal signal (dz = 0.68, 5.7× variance, 76% of subjects, coupling flips sign) that is **not established to be a better reading of anything**, and a *statistically robust* gain for the agent (+0.022, survives every robustness check) that is **not established to be a gain in the mechanism it names** (the mediation is null). Both facts coexist with a null on the joint measure.

### 6. Reported failure modes, biases, limitations

- [PAPER] **Confounded arms.** The CL condition includes demo-session training *and* continued online adaptation; FC is fixed. "Comparing demo-trained CLIFER held fixed during negotiation with the same model allowed to adapt would separate the contribution of subsequent updates" (§VIII.A) — the arm was **not** run.
- [PAPER] **Retention untested.** "Retention and catastrophic forgetting were not separately evaluated" (§VIII.A), despite the dual-memory architecture being the paper's core apparatus.
- [PAPER] **Ceiling of the method.** Episodic clustering cannot exceed the frozen encoder: if the base model zero-shot fails for a participant, "clustering cannot create an affective signal that is absent from the latent space" (§VIII.A).
- [PAPER] **Subjective ratings were a halo effect, by the authors' own account.** Preference modelling was rated higher (3.33 → 3.65, p = 0.013) although "the opponent-modeling algorithm was unchanged"; the authors call this "a 'Halo Effect'" and note the items are single questions, not validated constructs (§VI.D). Persuasiveness 3.32 → 3.65 (p = 0.043); time awareness n.s.
- [PAPER] **External validity.** 66 Turkish university students, one robot (Nao), one domain family, 10 minutes, lab; a smile can signal agreement in one culture and deception in another (§VIII.A). Dual-use flagged: a self-interested agent can exploit visible fatigue or frustration (§VIII.B).
- [INFERENCE] For EPModel the reading is narrow but sharp: the paper's most replicated-looking result (agent's own utility up) is exactly the result whose *stated route* did not move, and whose *partner-side* counterpart did not move. That combination is a reporting problem before it is a science problem — see AP5.

### 7. Software engineering lessons

- [PAPER] Component swap behind a stable interface: the Negotiator framework lets "the standard perception component" be replaced by the CL module without touching the offer protocol or preference analysis (§V.B). The perception backbone is explicitly substitutable, since the feedback rule acts on VA outputs whatever produced them (§VIII.A).
- [PAPER] Snapshotted parameters across a thread boundary rather than locks; freshness traded for non-blocking, with the stale value declared legal.
- [PAPER] Privacy by construction: frames processed in memory, not shared; only estimated VA scores and negotiation records released; code, analysis scripts and anonymised data public (§VIII.A, Data Access). Analysis seeds declared (42).
- [INFERENCE] Small lesson but a real one: the **inference path's latency budget is separate from the learning path's**, and the two are joined by one declared handoff. EPModel's engine purity (`M16.B`) has the same shape — the observer must not sit in the tick — and this paper is a working case of the same discipline in a live system.

### 8. Small-N, emotion, family/household, stress, long horizon

- [PAPER] Nothing on households, years or decades; the horizon is 10 minutes and the longest adaptation window is one session.
- [PAPER] **Individual-level heterogeneity is large and is reported.** 47% "strong", 21% "moderate", 14% neutral, **18% negative** (§VI.A.2, Fig. 7c). The authors' reading: "the effect of adaptation on the signal depends on the participant and interaction context".
- [INFERENCE] For a 12-agent family this is the quantitatively important point. A mechanism that improves the outcome for ~76% and worsens it for ~18% has no reliable per-run sign; with 12 agents a single run can land entirely in either tail. This is EPModel's existing ensemble rule (`M17.A`, `M17.C.1`) arriving with a number attached, and it argues for reporting **per-person** differences and the fraction reversing, not a family mean (cf. `M17.B.1` principal strata, `M17.B.2`).

### Not transferable / cautions

1. **It is not a simulation.** No state representation, no scheduling, no simultaneous events, no seeds, no ensembles, no counterfactual arms. Items 1 and 3 of the brief are answered almost entirely in the negative.
2. **The measured object is a face, not a relationship.** One 10-minute first meeting between a student and a robot; no history, no obligation, no third party, no next week. EPModel's appraisal is over years of accumulated reciprocal acts; nothing in this paper constrains that.
3. **No perception ground truth.** The paper says so itself. Its arousal shift cannot be read as "better perception", and the authors explicitly cannot separate adaptation from output rescaling.
4. **The headline is at the edge of its own power** (dz = 0.30 achieved against a design target of 0.35), and the concession result does not survive the stricter correction it also reports (p_adj = 0.0795).
5. **The stated causal route is null.** Do not cite this paper as evidence that "reading affect better makes an agent concede better" — the mediation was run and did not support it (§VI.E).
6. **The `P_A` responsiveness gate is not proposed here.** It is the paper's most interesting coupling (weight the partner's affect by how much the partner responds to you), but EPModel already weights a tie by `conductance` (`M4.C.1`), and a contingency estimate has no corpus support; importing it would add a per-tie state and a window for a game-theoretic idea that Bowen's account does not carry. Recorded as seen, not recommended.
7. **Its dual memory is a machine-learning apparatus with no counterpart in a rule engine.** Nothing about GWR, CAAE, replay or pseudo-labels transfers. The transferable claim is the *shape* (adaptation must be slower than interaction, and must not overwrite what is general), not the machinery.

---

## B. Candidate additions to the EPModel spec

Five candidates. None restates an existing requirement; the near-neighbours are named in each.

### AP1 — the scale of a perceived cue is the tie's own, not the family default

**Proposed requirement.** When the social cues are on, each `Relationship` **MUST** carry, per partner and per cue, a running spread of the cues the receiver *perceives* on that tie (an exponential variance over `cue_scale_window` weeks, `[I]`), and `M4.C.1e`'s felt change from a cue **MUST** be expressed in units of that spread — so that the same absolute departure costs more on a tie that never varies and less on a tie that always does. The spread **MUST** be estimated from the perceived vector only (`M4.C.12`'s output, after fidelity), **MUST NOT** be read from the signal the sender emitted, and **MUST** be floored at `M4.C.14`'s level-derived sensitivity so that a tie with no history reads through the general mapping. Its rate **MUST** be slow enough that a monotone deterioration over `cue_scale_window` weeks registers before the reference absorbs it.

**Where it would live.** `M4.C` for the use; the state on `Relationship` (`M1.B`, beside `M1.B.3e`'s felt-danger running average, which is the same shape).

**Evidence.** §IV, eq. (8) and its stated purpose — normalisation by the participant's *running mean and SD* of the model's own outputs, so the origin "dynamically represents the participant's running model-output baseline" (SHOWN as implementation, ARGUED as necessity). §II.C and §I: "generic facial-affect models degrade across individuals and contexts"; the plasticity–stability framing. §VI.A.1: output range 5.7× larger under adaptation — a change in *scale*, not only in mean. Strength: **ARGUED**. The paper does not show that calibration improves accuracy (§VIII.A admits no independent affect criterion).

**What it changes.** Adds a mechanism, and corrects an asymmetry in the current design. `M9.9` already gives each person a *believed cue vector* per partner, smoothed from perceived cues — that is a per-tie **location**. `M4.C.14` derives sensitivity and meaning error from `functional_level` alone — that is a **generic slope**. Nothing carries the per-tie **spread**, so at present a partner who is habitually flat and a partner who is habitually volatile are read with the same slope; the paper's whole result is about exactly that residual. It should also be checked against `M4.C.16`, which already reads per-sender cue *variance* over 12 weeks as an intensifier of the reaction: dividing by the spread and multiplying by the spread's dispersion push in opposite directions, and the net effect must be declared rather than left to the arithmetic.

**Cost / risk.** Low to medium. One per-tie, per-cue statistic (4 cues × 2 per tie), one `[I]` window, no new decision path, no new draw. Risks four: (a) it looks like a per-person parameter, which `M10.A.1` forbids — so the wording must make it a *statistic of delivered events*, converging from the level-derived default, not a free constant (this is `M1.B.3e`'s precedent); (b) **reference drift** — a slow real deterioration gets absorbed into the spread and stops registering, which is the mechanism's own failure mode and needs the slow-rate clause plus a test; (c) reading the *sent* rather than the *perceived* signal would let repeated deliveries average out `M4.C.12`'s meaning error and hand the receiver a channel to the sender's true cue, defeating `M4.B.2` and `M11.C.36`; (d) it interacts with `M4.C.16`, above. New criterion: a monotone decline over one window registers in the last quarter of the window, and is not registered in a run where the spread is updated fast (the mutant).

---

### AP2 — a change in a perceived cue amplifies the deviation it makes, and the amplification is bounded

**Proposed requirement.** When the social cues are on, the felt change a delivered cue makes (`M4.C.1e`) **MUST** be scaled by `1 + cue_change_gain × |P|`, where `P` is the cue's signed change on that tie since the previous delivery to the same receiver from the same sender, normalised as `M4.C` normalises intensity, and `cue_change_gain` is `[I]`. The scaling **MUST** apply to the existing two-sided deviation and **MUST NOT** add a drive: `M4.C.1`'s two drives stay the only drives and the cues remain readings. A tie held at a constant low cue value for `cue_change_window` weeks **MUST NOT** accumulate anxiety from this term.

**Where it would live.** `M4.C` (as an `M4.C.1e` amendment), with the previous delivery's cue value on `Relationship`.

**Evidence.** §IV, eqs. (6)–(7): the feedback is built from rotation and radial *change* between consecutive states, with magnitude capped at 1; §II.C, "A fixed mapping from affect estimates to actions" cannot capture what matters. §VII: "changes in valence and arousal set the direction and strength of the affective feedback." Strength: **SHOWN** as a control rule, **INFERENCE** for the amplification form and for transferring it to a level-based appraisal that the paper does not have.

**What it changes.** Adds a mechanism and closes a real modelling gap: `M4.C.1` is a *level* appraisal (deviation from a felt optimum), so a tie sitting at a poor level and a tie that has just fallen to it are indistinguishable on the week of the fall. `M4.C.16` is the nearest existing rule and is different: it reads the *dispersion* of cues over 12 weeks as an intensifier, not the first difference, and it says nothing about direction. Multiplicative rather than additive is deliberate — it keeps `M4.C.1`'s saturation and does not create a new drive that could run away, which is what an additive term would do in a positive-feedback family.

**Cost / risk.** Low. One `[I]` constant, one stored previous value per ordered pair, no new draw class. Risks: it is a second intensifier on the same quantity as `M4.C.16` (declare the joint effect and sweep the pair together, `M17.E.1`); it raises escalation on any shock and must be shown not to make the reference family's conflict run away (a run-length check, not a new criterion); and no corpus source states it, so it is `[I]` throughout — it must be presented as an invented premise the way `M4.C.11` is, not as a Bowen claim.

---

### AP3 — an adaptive mechanism is tested in three rungs, and the report says which rung carries the difference

**Proposed requirement.** Where a criterion or arm credits a mechanism that both *calibrates* and *continues to adapt*, Phase E **MUST** run three arms — mechanism off, calibrated through a declared calibration window then frozen, and adapting throughout — and **MUST** report the off→frozen and frozen→adapting differences separately. Where the frozen arm carries the whole difference, the continuing adaptation **MUST** be reported as **not discriminating**, with the consequence `M17.D.2(c)` already names for a rival arm.

**Where it would live.** `M17.D` (a sixth named control arm), reported under `M11.F`'s framing; the switch itself in `M10.B` beside `M10.B.5`.

**Evidence.** §VIII.A, first limitation: "Comparing demo-trained CLIFER held fixed during negotiation with the same model allowed to adapt would separate" the two contributions — and the comparison run does **not**: "The comparison therefore evaluates demo-session training together with continued online adaptation" (§V.C). The authors list it as the first thing they would fix. Strength: **SHOWN** that the confound is real and that a competent group could not resolve it post hoc; **INFERENCE** that the rule generalises.

**What it changes.** Adds a test, and repairs a gap in the arm vocabulary. `M17.D.1` has five control arms, none of which is "the adaptive mechanism held at its calibrated value"; `M10.B.5` gives on/off only, so every EPModel result about the learner is currently all-or-nothing. With `M4.D.6`'s reinforcement, AP1's spread, and `M9.9`'s smoothing all being two-stage (initialise, then keep updating), the missing middle rung is the one that says whether *continuing* to adapt does anything — which is exactly the question the source could not answer.

**Cost / risk.** Low. It is an arm, not a mechanism; no new constants beyond a declared calibration window `[I]`, no engine change (`M17.D.3` requires arms to differ only through declared channels, and this is `M10.B.5`'s shape widened). Risk: the frozen arm needs a principled freeze point, and a bad choice of window can make the rung look arbitrary — so the window must be declared before the suite (`M10.B.4`) and swept (`M17.E.1`). Improves inference validity only, but that is the whole point.

---

### AP4 — a quantity that both adapts and is read is read from the previous tick's end

**Proposed requirement.** Any quantity that is both read by appraisal and updated by it — `M9.9`'s believed cue vector, AP1's spread, `M4.C.16`'s 12-week variance, `M1.B.3e`'s felt-danger average — **MUST** be read in tick *t* at its value as of the end of tick *t−1*, and **MUST NOT** be updated by any delivery appraised in tick *t* until every delivery in that batch has been appraised. Two deliveries in one tick **MUST** therefore be read through the same value, whatever order the batch is processed in.

**Where it would live.** `M3` (a generalisation of `M1.F.8`'s "appraisal is judged against the state before the tick's batch"), asserted under `M4.G.2`.

**Evidence.** Algorithm 1, lines 4–6 and 27, with §IV's reasoning: the learner publishes `θ_shared` only at the end of a training step and Thread 1 adopts it at the top of a cycle; if training overruns, the last completed parameters are used. §IV: "keeping learning off the interaction path." Strength: **SHOWN** as engineering practice in a live system; **INFERENCE** that the same discipline is what keeps a batched tick order-free.

**What it changes.** Adds a determinism rule, and closes a silent ordering channel. `M1.F.8` covers the *state* a delivery is appraised against; it does not cover a *quantity the appraisal reads and then writes*, so an implementation can legitimately update `M9.9`'s smoothing after each delivery and make the second delivery's felt change depend on the queue order. `M3.D.1`'s batching and `M11.D.5` (byte-identical logs) would not catch this, because the order is deterministic — it is deterministic and wrong. `M1.B.3e` says its average is read "after that week's move effects and landed contacts", which is compatible with the rule but does not state it.

**Cost / risk.** Very low: a rule plus a permutation test (swap the processing order of two deliveries in one tick; the tick's outcome must be identical). No constants. It touches `M3.D.4`'s determinism family without conflicting with it, and it forces a second store per adapting quantity — a copy, which is cheap. The one real risk is over-broad wording: the rule must exempt quantities the tick is *supposed* to move within the batch (a tie's `bond_energy`, a person's `acute_anxiety` in `M4.C.1d`'s reciprocal return), or it would forbid the within-tick coupling the spec deliberately has. That exemption must be listed, not left to judgement.

---

### AP5 — a single member's gain is reported with the counterpart's and with a joint measure, and a named channel is tested as a path

**Proposed requirement.** Any criterion or arm result asserting that a change raises one member's readout **MUST** report, beside it, the same readout for every other member of the ties that member acted on, and at least one joint readout (`M11.G`'s family evaluation, or the total anxiety `M11.C.52` already sums). Where the mechanism names an intermediate channel, the report **MUST** state whether that channel moved in the asserted direction **on the same seeds**, and **MUST NOT** describe the mechanism as supported if it did not.

**Where it would live.** `M11.F` (framing that travels with an output) for the reporting half; `M17.G` for the ensemble half.

**Evidence.** Table V: agent utility +0.022 (p = 0.017, dz = 0.30) with participant utility −0.008 (p = 0.495) and Nash distance −0.01 (p = 0.378) — a gain for one party and nothing for the pair. Table III and §VI.B: robot concession rate 38.9% in **both** conditions, so the agent's gain rode entirely on the human conceding more (28.1% → 37.4%). §VI.E: the named route was tested and is null (ab = 0.0001, [−0.0067, 0.0063]; 13 of 14 exploratory models null; the one that excluded zero points against the total effect), and the paper nonetheless reports the mechanism as supported, substituting a concurrent-routes argument for the missing chain. Strength: **SHOWN** for the numbers; **ARGUED** for the reporting rule.

**What it changes.** Adds a reporting rule, and closes the largest inference hole in EPModel's criterion form. `M0.4`'s directional criteria assert a difference between arms; several of them assert a readout for **one** member (`M11.C.2`'s projection member, `M11.C.4`'s cutter, `M11.C.48`'s cutter's other ties), and `M11.F.6`'s two-sidedness rule is about *detectors*, not about who pays for a movement. Nothing currently requires the counterpart's number to appear, and nothing requires a named intermediate channel to be checked rather than inferred from two co-moving arm differences — which is precisely how the source's headline got into its abstract with a null mediation behind it.

**Cost / risk.** Very low. No new constants, no engine change, readouts already in the log (`M16.A`), and it strengthens `M11.1d`'s outcome-directive audit rather than competing with it. Risk: a joint measure can be reported vacuously (the family evaluation is a composite of eight components, `M11.G.1`), so the rule should name the component rather than the composite — otherwise "joint measure" becomes a box ticked by a schema that is always emitted. Second risk: the path check can be read as demanding a mediation analysis with the power to detect an indirect effect, which a 12-agent ensemble barely has; the rule should therefore ask for the *direction of the intermediate channel on the same seeds*, not for an estimated indirect effect, and permit `UNDETERMINED` (`M17.A.1`).

---

# Part 4: Profile-Conditioned World Modeling (2611.24999v1)

**Paper.** Qin, Qiao, Frandsen, Treadway, Zhao, "PsyWorld: Profile-Conditioned World Modeling for Human Behavior Prediction", Emory University; formatted for KDD '27 AI for Sciences Track; arXiv:2611.24999v1 [cs.LG], 18 Nov 2026; 11 pages + 2 appendices. Read in full from `/tmp/2026_PsyWorld_ProfileConditioned_2611.24999v1.txt` (669 lines, `pdftotext -layout`; figures present only as captions, equations partly garbled).

**Checked against.** `docs/bowen_agent_model_spec_v2.md` at 2,225 lines / revision 67 (not the brief's "revision 10", not 427 requirements), plus `papers/SPEC_CANDIDATES_from_preprints_2026-09-20.md`, `papers/SPEC_CANDIDATES_plain_language_2026-{09-20,09-27,09-27b,10-04,10-04b}.md`, `docs/todo/2026-09-27_paper_material.md`.

**One-line summary.** A learned, two-stage latent world model of *one person's* next behaviour in a *non-social* task environment, whose only transferable content for EPModel is on the inference side: how to report a result that a subgroup or a rare event could have carried.

---

## A. Report

### 1. Simulation formalism

**1.1 [PAPER]** Two-level decision hierarchy over a fixed, small action vocabulary. AgentSociety: `A = {work, consume, rest, leisure, move}` (§3.3). VEET: a 6-way macro label inferred from replay geometry, realised by a 13-way micro action space `(m_t, r_t, z_t)` — movement intensity, turn, teleport (App. B.1). A learned execution policy realises the macro label as a short horizon of micro actions (§3.7).

**1.2 [PAPER]** State is an environment–behaviour vector, not an affective one: `s_t ∈ R^49` grouped into self-motion/position, room context (effort, reward, door distances), object layout (coin counts, nearest-coin distances) and task progress (App. B.1). The training tuple is `(s_t, q_i, a_t, s_{t+1}, a_{t+1})` (§3.1); the transition dataset is `D_WM` and *excludes the micro action*, kept separately in `D_micro` (App. B.1) — a deliberate separation so the world model never sees the low-level act.

**1.3 [PAPER]** Time: AgentSociety uses `ticks_per_step=300`, a 5-minute base resolution, from 08:00 over one day; logged records are event-driven, not sampled, "typically ranging from 5 to 30 simulated minutes" apart (App. A.1). VEET logs the same way at task level.

**1.4 [PAPER]** The generative form is `p(a_{t+1} | q_i, s_t, a_t)` as the *reported* objective, with `p(s_{t+1}, a_{t+1} | q_i, s_t, a_t)` modelled jointly so the realised next state supervises training (§3.1). At rollout the next state is predicted, never observed: "without access to `s_{t+1}`" (§3.1).

**1.5 [PAPER]** The latent is discrete: a codebook of `K = 128` codes of dimension 64, EMA-updated at decay 0.99; a training-time posterior quantises the realised transition, a rollout-time prior predicts the code from profile plus pre-transition context (§3.4). 19 of the 128 codes are used on VEET, 5 on AgentSociety (Table 1).

**1.6 [PAPER]** A learned scalar gate decides per decision whether to use the transition branch at all: `γ_t = σ(f(h_t, z_t, ŝ_{t+1}))`, combining `ℓ^trans` and `ℓ^ctx` (§3.5). The stated reason is that "macro prediction transition-aware without requiring the latent branch when the current context is already sufficient" (§3.5).

**1.7 [INFERENCE]** Nothing here imposes on EPModel's clocks. The one structural point EPModel already shares is the separation of a *stable* descriptor from a *fast* state in the transition tuple — EPModel does that with `basic_level` + swing and `chronic`/`acute` anxiety (`M1.A.5a`), and carries the same discipline into the tuple EPModel logs (pre-state, event, post-state, move) at `M16.A.4`. There is no scheduling, simultaneity or determinism content: the paper is a supervised learner over a fixed dataset, single run, single A100 (§4.1), and reports no seed variance at all.

### 2. Agent architecture

**2.1 [PAPER]** The architecture is a *factorisation*, not a policy: profile `q_i` (fixed per participant), state `s_t` (time-varying), macro behaviour `a_t` (§3.3). The paper's own framing of the alternative is the target of the whole exercise: profiles "commonly injected as static prompts, policy inputs, or reward descriptors" (Abstract/§1), and the fix is that the code is "a transition-level predictive variable rather than a participant identity or personality label" (§3.2).

**2.2 [PAPER]** Where the profile enters is *rates and thresholds*, not action offsets. AgentSociety's added `pleasure` need and the five need-decay rates are scaled by questionnaire traits `α_H, α_D, α_P, α_C, α_L`, and stress/hedonic traits "shift the safety and pleasure thresholds", so that "the same environment state induce[s] different dominant needs for different profile types" (App. A.1). This is the concrete form of the claim EPModel already implements at `M4.C.1a` (level sets the steepness and the width of the anxiety-vs-deviation function) and `M4.C.14`.

**2.3 [PAPER]** Memory is only the context encoding `h_t = f_ctx(q_i, s_t, a_t)` (§3.3); there is no separate memory store, no reflection, no learning within a run. Heterogeneity is entirely imported from the 313 participants' questionnaire vectors (§4.1); the AgentSociety agents are LLM-driven, each decision "constrained to the structured macro vocabulary" (App. A.1, Algorithm 1).

**2.4 [PAPER, negative]** The paper explicitly does *not* model affect as a state. "Post-level subjective ratings are used only as auxiliary targets" and "are not modeled as dynamic affective states" (§4.1); §5 repeats that the model "does not contain a separately observed dynamic affect channel". So the paper's "state" is environment plus behaviour, and its claim is about behaviour prediction, not emotional process.

**2.5 [INFERENCE]** For EPModel the useful sentence is the one EPModel already obeys: a static descriptor of a person must not be the direct input to selection. `M4.D.2` names the propensity inputs (acute anxiety, `functional_level`, tie state, triangle position, repertoire) and pointedly not `basic_level`; `M4.D.1a` routes differentiation through a mixing weight. The paper is a second-hand corroboration of a rule already in the spec, not a new one.

### 3. Interaction / network structure

**3.1 [PAPER, and it is the paper's largest limitation]** There is no interaction. AgentSociety's social pathway is removed on purpose: "we suppress the original social pathway", social behaviour is taken out of the plan guidance, the social satisfaction state is set high with near-zero decay, and any `SocialBlock` is deregistered, "This prevents the agent from spending trajectory steps on messaging, calling, or meeting other agents" (App. A.1). VEET is a single participant navigating rooms alone (App. B.1).

**3.2 [PAPER]** Consequently there are no dyads, no groups, no influence mechanism, no cascade, no tie dynamics, no information distortion between people. The only "network" is the profile-cluster partition used as a diagnostic stratum (Tables 3, 4).

**3.3 [INFERENCE]** This is decisive for how much of the paper transfers. EPModel's entire object — a tie with conductance and bond energy, a triangle with an inside pair, an event with targets and *witnesses* — has no counterpart in the paper. Any claim of the form "the paper shows that interaction should be modelled as X" would be invented, not read.

### 4. Initialization

**4.1 [PAPER]** Subject representation is common across both environments: "questionnaire profiles from 313 VEET participants" (§4.1). `q_i` is used at item level, scale level, or as eight scale scores (PSS, SHAPS, BDI, AES, AMI, MAPSR, STAI-S, STAI-T) (App. B.1).

**4.2 [PAPER]** In AgentSociety the profile is converted to an `AgentPsychProfile` and then *written into live agent state*: initial need satisfactions (hunger, energy, safety, social, and the added pleasure), emotion values and types, personality text, initial thought; then `NeedsBlock` decay rates and thresholds are patched (App. A.1, Algorithm 1 lines 2–8).

**4.3 [PAPER]** VEET state normalisation uses "the same feature scaler used by the behavioral cloning and reward-model baselines", with a declared mask when clean-feature checkpoints are used, "so that all models use an identical state interface" (App. B.1). That is an interface-hygiene rule, not a sensitivity analysis.

**4.4 [PAPER, absent]** There is no sensitivity analysis to initialisation in this paper: one split, one seed structure, no ranges, no envelopes. Only the held-out split and the cluster breakdown.

### 5. Calibration and validation

**5.1 [PAPER]** The headline protocol is a *participant-disjoint* split: 240 training, 25 validation, 48 test participants, "no participant overlap across training, validation, and test", and the stated purpose is "whether the behavioral predictor generalizes to unseen people, rather than measuring performance on new trajectories from previously observed people" (§4.1, Table 1 caption).

**5.2 [PAPER]** Baselines are strong and are reported: on VEET, Random 0.167 Top-1, **Majority 0.739**, Direct SFT/BC 0.754, IRL policy 0.650; the model 0.904 [0.895, 0.913], Top-3 0.989, Macro KL 0.0033, distribution similarity 0.999 (Table 1). On AgentSociety: Random 0.200, Majority 0.298, Direct SFT 0.525, the model 0.595 (Table 1).

**5.3 [PAPER]** The ablation ladder is reported as three rows and the gain is *attributed by readout*, not in aggregate (Table 2): Simple WM + decoder 0.901/0.952/KL 0.012; VQ-WM 0.905/0.981/0.0078; VQ-WM + prior alignment 0.904/0.989/0.0033. The paper's own reading: continuous next-state modelling carries "most of the Top-1 gain", the discrete code and prior alignment "mainly improve behavioral coverage and distributional alignment" (§4.3).

**5.4 [PAPER, and it is a model of honesty]** The paper states the confound in its own ablation: "Because these variants also differ architecturally, this comparison is evidence of predictive utility rather than a controlled causal attribution" (§4.3). It also refuses the psychological reading twice — the codes are "predictive summaries … not … causal psychological or clinical constructs" (§3.1) — and bounds its own generalisation: participant-disjoint evaluation "addresses transfer to unseen people within this study design" (§5).

**5.5 [PAPER]** Uncertainties are participant-level bootstrap intervals, 1,000 resamples, seed 2026, over the 48 test participants (Table 1 caption). The VQ Top-1 difference (0.905 vs 0.901) sits inside the interval, so the paper does not claim it.

**5.6 [PAPER, negative diagnostic]** 19 analysed prior codes against 7 psychometric scales = 133 tests; "No code–scale association survives BH–FDR correction", nominal `|ρ| ≈ 0.11–0.19`, median pairwise n = 297 (§4.4, Fig. 3). Reported as a negative result bounding interpretation, not buried.

**5.7 [PAPER, absent]** No calibration of long-horizon dynamics, and the paper says so: it "does not independently assess calibrated multi-step state-transition fidelity" (§5). Every number in the paper is one-step.

### 6. Reported failure modes and negative results

**6.1 [PAPER]** The base rate is brutal and the paper prints it: a constant predictor reaches 0.739 Top-1 on VEET against a trained direct model's 0.754 — a 1.5-point margin over "always name the modal action" (Table 1). The distributional metrics separate them far better (KL 6.888 against 0.361), which is why they are reported.

**6.2 [PAPER]** Rare-class performance collapses while the aggregate looks uniform. Top-1 is 0.891–0.913 across the four held-out VEET clusters (N = 7, 18, 8, 15), but teleport recall is 0.13–0.26 (Table 3); on AgentSociety Top-1 is 0.558–0.613 while leisure recall is 0.16–0.27 (Table 4). The paper's own caveat: cluster sizes 10–14 mean these are "descriptive diagnostics rather than causal subgroup effects" (§4.5).

**6.3 [PAPER]** Performance is environment-bound: the same method reaches 0.904 Top-1 in VEET and 0.595 in AgentSociety, with 19 codes used in one and 5 in the other (Table 1). Micro-control completion is 0.332 on VEET and 0.990 on AgentSociety for the same DPO recipe (Table 5).

**6.4 [PAPER]** The macro→micro gap is large even where macro prediction is good: VEET DPO completion 0.332, coin gain 2.516, first-action KL 1.382, move-frequency gap 0.520, against SFT/BC's 0.168 / 0.120 / 2.556 / 0.873 (Table 5). The model names the right intent far more often than it realises it.

**6.5 [INFERENCE]** 6.1–6.4 are the paper's transferable content for EPModel, and all four are *reporting* failures a well-built model can still commit: a readout dominated by the modal class, a subgroup carrying the average, an environment-bound magnitude, and a selected act that is not the act delivered.

### 7. Software engineering lessons

**7.1 [PAPER]** Privileged information is confined to training by construction: "The posterior code is available only during training" and "the only code used at rollout" is the prior's (§3.4); the decoder is trained on both codes so the rollout code is in-distribution (§3.6). The dashed/solid split is drawn explicitly in Fig. 1.

**7.2 [PAPER]** Collapse is monitored, not assumed away: unique codes activated by the prior are reported in the results table itself (19, 5; Table 1), and a "lightweight code-usage regularizer" mitigates codebook collapse (§3.6).

**7.3 [PAPER]** The training log is per decision and complete enough to recompute: "We log the simulation tick, current need state, satisfaction values, questionnaire profile, macro action, macro-action logits and probabilities, and the corresponding micro-action description" (App. A.1). Ablations reuse one class-balanced loss and identical state interface across variants (§4.1, App. B.1).

**7.4 [PAPER, weak]** There is no configuration/experiment tooling, no counterfactual/intervention machinery, no reproducibility statement beyond hyperparameters (§4.1: Adam 3e-4, 100 epochs, batch 256, λ_s 1.0, λ_VQ 0.25, λ_z 0.1, λ_a 1.0, one A100). A single run per configuration, no seed variance anywhere.

**7.5 [INFERENCE]** EPModel is already far past this: `M3.D.4a`'s counter-based keyed draws, `M16.A.1`'s self-describing header, `M16.A.3`'s selection rationale, `M16.B`'s purity. Nothing here to import.

### 8. Small-N, emotion, family units, long horizon

**8.1 [PAPER]** The test set is 48 people — small by ML standards and the paper says so ("Because cluster sizes are limited"), but each contributes hundreds of decisions, and the interval is taken over *people*, not decisions. That choice of resampling unit is the right one and is the paper's best inference-validity move.

**8.2 [PAPER]** No family or household unit exists anywhere in the paper; no emotional state dynamics; no horizon beyond one day (AgentSociety) or one task session (VEET). The paper's claim is bounded to next-behaviour prediction and it says so in the conclusion: a "predictive inductive bias rather than a validated psychological mechanism" (§6).

**8.3 [INFERENCE]** For a 12-agent, 2,080-tick, one-family model with no validation data, this paper's numbers mean nothing and its protocol cannot be copied: there is nothing to hold out and nothing to fit. What survives is the shape of its honesty.

---

### Not transferable / cautions

- **The mechanism.** Posterior-encoder/prior/VQ codebook/DPO micro-controller are a learned pipeline. EPModel's decision path is rule-based by `M3.D.6`; importing any of it would be a different project.
- **The setting.** Asocial by construction (App. A.1 removes the social pathway), one person at a time, five macro actions, one day. EPModel's object is dyads, triangles, witnesses and forty years. No result in the paper speaks to any of it.
- **The state.** `s_t` is geometry and task progress, not affect (§4.1, §5). EPModel's fast state (acute anxiety, functional level, load) has no counterpart, so "state-conditioned beats profile-conditioned" is not evidence that EPModel's state channel is doing work — that is `M11.1b`'s question and must be answered by EPModel's own mutant.
- **The split.** 240/25/48 is an ML train/val/test split. EPModel has no fitting (`M11.F.9`), so "unseen participant" has no analogue; the honest analogue is `M15.D.4`'s envelope over ranges, already stronger.
- **The magnitudes.** 0.904, 0.0033, 19 codes: environment-specific, single-run, and the paper says they are not comparable across its own two environments (§4.5). Do not quote any of them as a property of a mechanism.
- **The claim.** The paper explicitly does not establish causal psychological mechanisms, calibrated long-horizon state fidelity, or `s_{t+1}` observability at rollout (§5, §3.1).

### Overlap check — already covered, not re-proposed

| Item | Where it already is |
|---|---|
| Profile must not be a direct action descriptor; state carries the load | `M4.D.2` (propensity inputs exclude `basic_level`), `M4.D.1a` (mixing weight); `M11.1b` is the state-contingency mutant |
| Trait conditions rates/thresholds, not action offsets | `M4.C.1a` (steepness and width by level), `M4.C.14` |
| Same person, different context ⇒ different behaviour | plain-language **BV2** (2026-10-04, from the value-alignment paper) |
| Two-stage macro→micro realisation under one context | `M4.C.13`/`M4.C.15` (signals set from state at act) |
| Distributional readout beside a pointwise one, with JSD | `SPEC_CANDIDATES_from_preprints` **C39**; `M17.B.5` |
| Rare move must be checked, and a criterion fails if it never occurs | `M11.1f`; `M11.D.19` (reachability, "not observed" ≠ "unreachable") |
| Confound of event frequency with a per-event rule | `M11.1e` + `M16.A.8` (matched frequency / separated counts) |
| Own distribution vs rival distribution across structural dimensions | `M17.G.1` robustness audit with claim grade; `M11.4f` |
| Small differences not claimed inside the interval; third outcome | `M17.A.1` (adaptive blocks, δ, cap, `UNDETERMINED`), `M11.4e` |
| Multiplicity | `M11.4e` (declared correction where several readouts sit under one criterion), `M17.A.3` |
| Unseen/perturbed initial conditions, envelopes | `M15.D.4`; `C37` (two reference configurations) |
| Mechanism-off comparison is a test of structure, not an intervention | `M11.4f` |
| Derived summary must not be read as a construct | `M16.A.9` (entropy "not a measure of differentiation"), `M11.C.28` (symptom count is not a differentiation readout), `M11.G.2` |
| No leakage of post-hoc information into the decision | `M4.B.2` (policy reads own state, beliefs, delivered events only); `M16.B` purity |

---

## B. Candidate additions to the EPModel spec

Two candidates. This paper's mechanism content is not usable and its protocol is not adoptable; the yield is confined to how a directional result is *reported* when a subgroup or a rare event could have carried it. Everything else on my list was already in the spec, at a level of detail above what this paper offers.

### PS1. A directional readout must say how many persons carried it

**Proposed requirement.** For every `M11.C` criterion whose readout is a move count, a move share or a tie-level quantity, the verdict record **MUST** report, per arm, the number of distinct persons (and distinct ties) contributing to the asserted difference, and a criterion whose passing difference is carried by fewer than `min_contributing_persons` persons in either arm **MUST** report that the difference is carried by those persons and **MUST NOT** report it as a family-level result. The count, the threshold and the form of the "carried by" flag are `[I]` and declared before the run.

**Where it would live.** `M11` (verdict record, beside `M11.D.18`'s fallback reporting), with the contributing-person counts read from `M4.E.1`'s event stream in `M16.A` (no engine change, `M16.B.3` holds). Extends `M17.B.5`'s per-person structure to the verdict itself. Phase E reporting.

**Evidence.** Tables 3 and 4: aggregate Top-1 uniform at 0.891–0.913 across four held-out VEET clusters (N = 7, 18, 8, 15) while teleport recall runs 0.13–0.26, and 0.558–0.613 with leisure recall 0.16–0.27 on AgentSociety; the authors' own reading is that with 10–14 people per group these are "descriptive diagnostics rather than causal subgroup effects" (§4.5). Also Table 1: a constant predictor at 0.739 Top-1 beside a trained model at 0.754 (§4.2/§6.1) — an aggregate that hides what produced it. **Strength: SHOWN that an aggregate can be carried by a part and read as a whole; INFERENCE that the same failure is reachable in EPModel and worth a rule.**

**What it changes.** Adds a reporting rule and a verdict field; no mechanism. Inference validity, and it protects a theory claim specifically: `M11.C.28` already refuses symptom count as a differentiation readout, but nothing today stops "the coached arm produces more `CUTOFF`" from being a statement about Sam alone read as a statement about the family. That is the difference the theory cares about — a cut-off *person* is not a cut-off *family*.

**Cost / risk.** Low: counts over an existing log, one config constant, no engine change. Risk of noise: a criterion that legitimately turns on one member (a projection onto one child, `M7.D`) would be flagged routinely, so the threshold has to be read as *report*, not fail, and the flag must not suppress the verdict. No conflict with `M0.4` (still a two-arm direction), `M3.D.4`/`M3.D.5`, `M3.D.6` or `M11.F.9`. Touches `M11.4`'s list only if the flag is written as a criterion rather than a report field — it should be the latter.

**Status.** NEW as written. `M11.1f` covers *zero* occurrences and reports counts per arm; `M17.A.1` covers interval width. Neither says how many persons a surviving difference rests on.

### PS2. The readout must separate the act selected from the act delivered

**Proposed requirement.** Where `M4.C.1f`'s social cues are on, the run log and the `M11.G` readout **MUST** report, per ordered pair and per move kind, the count and share of deliveries in which what the receiver appraised differed in kind from what the sender emitted (a private hop of `M1.F.4` arriving with tone, face and posture empty; a `TRIANGLE` arriving at a different third party), and no criterion may attribute its direction to selection alone when that share exceeds a declared threshold without a selection-only arm to separate them.

**Where it would live.** `M16.A` (an observer-side record class over `M1.F.1`/`M4.E.1`, so `M16.B.3` holds) and `M11.G`'s readout schema; the attribution guard sits in `M11` beside `M11.1d`'s outcome-directive audit. Phase D for the record, Phase E for the readout.

**Evidence.** Table 5 and §3.7: the macro→micro split is where the paper's own model fails — Top-1 0.904 but DPO completion 0.332 and move-frequency gap 0.520 on VEET (§4.5, Table 5) — the intent is named far more often than it is realised, and the paper reports the two with separate readouts rather than one. EPModel has the same structural gap for a different reason: an event is not what its sender chose, because `M1.F.4` degrades fidelity per private hop and, with the cues on, a private hop "carries the words only". **Strength: ARGUED for the macro/micro separation in the paper's own setting; INFERENCE for EPModel, where the distortion is a specified mechanism rather than a learned controller.**

**What it changes.** Adds a readout and an attribution guard; no mechanism, no state. Inference validity, and it is a precondition for reading `M4.C.7` (a witnessed statement is less reactive than an addressed one) and `M1.F.3` (route modulates gain) at all: if a third of the deliveries arrive as a different event, then a move-level difference between arms is partly a difference in what arrived, not in what was chosen.

**Cost / risk.** Low and observer-only. Real risk of degeneracy: with the cues off, `per_hop_fidelity` is "declared and not drawn" (`M3.D.4`'s table, line 649) and delivery scales intensity rather than changing kind, so the readout is near-vacuous and **MUST** be scoped to runs with `social_cues_enabled` on — where `M4.C.1f` already requires an identical log when they are off, so the scoping costs nothing. No conflict with determinism, `M3.D.6` or `M11.F.9`.

**Status.** NEW. `M16.A.2` logs the event's full field set including `fidelity` and both timestamps, so the raw material exists and is unused; nothing in `M11.G` reports the gap, and `M11.1d` audits which rules carry a direction but not whether the direction reached the receiver intact.

---

# Part 5: Same Seeds, Different Chains — an LLM supply-chain testbed and what its evaluation failures teach (2611.23613v1)

Amar, *MANGO Arena: Same Seeds, Different Chains. Evaluating LLM Agents in a Disrupted Supply Chain*, arXiv 2611.23613v1 (cs.MA), 17 Nov 2026, 8 pages. Short paper, open code and data (Zenodo, MIT). Four LLM agents each run one company in a serial supply chain adapted from a classroom game; one environment measures an economic outcome (bullwhip), rule adherence and collusion proposals at once.

## A. Report

**1. Simulation formalism.** §3 fixes six ordered steps per period, executed for all four companies at the same time — red card announced, shipments from two periods earlier arrive, each company fills from stock with lost sales, each orders, each may post one 200-character message, storage charged. Orders arrive one period later and shipments two more, so an order lands three periods after it is placed (Fig. 1). Horizon 12 periods; demand 12 ± 3 units from seeded cards; truck limit 60 units; $2,500 cash and 24 units to start; no credit; 90% fill rule (Table 1). [PAPER] The only thing a seed fixes is the demand-card sequence: the agents' own sampling is unseeded, and that is precisely why reruns at identical seeds diverge (§6.2). There is no exposed state-transition function — this is a game runner, not a simulator. [INFERENCE] This is the paper's most structural warning and EPModel already answers it: `M3.D.4a` makes every draw a pure function of (run seed, canonical event key) with no mutable generator state, and `M3.D.5` requires byte-identical logs. "Same seeds, different chains" cannot arise from EPModel's engine; what can still vary is the initial-condition draw, which `M17.C.1` carries as a separate variance component.

**2. Agent architecture.** One stateless call per decision, carrying the rules, the company's own state, its last six periods, announced red cards and the previous message board; strict JSON out; agents see only their own state (§3). Memory is a history table, not the agent's own reasoning (§7). No learning, no heterogeneity beyond role. [PAPER] Amplification comes from a heuristic with no incentive behind it: zero orders in 39–60% of decisions against 17% for base-stock and 0% for pass-through (Table 2, §5), with no fixed ordering cost to justify batching. A typical Retailer sequence is `0, 18, 18, 18, 12, 0, 0, 24, 24, 0, 12, 0`. [INFERENCE] The analogous EPModel question is whether "one move per week" can represent a person who holds six weeks of grievance and discharges it in one act. `M4.D.1h` lets intensity rise with anxiety, so batching is representable through intensity rather than through silent weeks — but no requirement says that this *is* the representation, and a reader of move marginals cannot distinguish a calm person from an accumulating one. That is a theory-fidelity question, not a method one; I do not propose it here.

**3. Interaction structure.** Strictly serial: each company receives one order from the one downstream of it (§3). A single public board, 200 characters per company per period, is the only non-trade channel. 170–207 of 288 messages per model matched the information-sharing screen, the side-deal screen never fired, coordination flagged 12 GLM and 1 Luna message, hoarding 4 GLM and 1 DeepSeek message (§5). [PAPER] The screens are keywords "with unknown recall", so the paper claims only that no collusion was *proposed in detectable terms* (§5, §7). The block on side payments is enforced in the prompt and the rules, not mechanically — message rules "cannot be enforced mechanically" (§3). [INFERENCE] What transfers is the adjudication problem, not the board: a keyword screen plus a human reader is exactly the apparatus a rule-based family model must not need. EPModel's observable analogue of the board is who hears what — witness sets on the event record (`M1.F.1`, `M1.F.5`) — and that is a mechanism, already built, rather than a screen.

**4. Initialization and sensitivity.** Initial conditions are identical for every company and never varied (Table 1); the only variation is the demand draw, three seeds per condition. [PAPER] With that little variation, outcomes still spread by an order of magnitude across seeds (Table 2, s.d. columns) and across reruns at the same seeds (Table 4). For Luna in the disruption condition, with zero failed calls in both runs: seed 2 Factory bullwhip 6.9 → 113.4 (16x), seed 1 Supplier 8.0 → 134.2 (17x); means 59.5 → 99.4 and 36.2 → 85.4. The paper's explanation is structural — "a chain amplifies small differences in early decisions" (§6.2). [INFERENCE] Two things follow for a 12-agent model. First, three seeds cannot separate a mechanism from a draw, which the spec already accepts (`M17.A.1`'s block stopping, `M17.A.4`'s declared margin). Second, an amplifying system makes the *paired* design load-bearing rather than merely tidy: if the same seed pair can give 16x different amplification, only a coupled draw per seed keeps an arm difference a difference of mechanism. `M3.D.4a` is that coupling, and this paper is an accidental demonstration of what happens without it.

**5. Calibration and validation.** No behavioural data; the human comparison is future work under an approved IRB protocol (§8). What functions as validation is a positive control: two scripted reference policies — pass-through, and base-stock with α = 0.2 and safety stock 12 — that reproduce the textbook result, base-stock amplifying about 6x at the Retailer and 41x at the Supplier, tripling under the fire (§5, Table 2). Reporting is mean(s.d.) over three seeds; no confidence intervals, no power analysis, no prompt or effort ablation, one prompt, one reasoning effort, three small models (§7). The whole pilot cost under $1 (§4). [INFERENCE] The reference-policy device transfers and EPModel lacks it. `M17.D.1`'s control arms remove a mechanism; they do not supply a *known-bad* baseline the full model is required not to exceed. "Every model amplified more than the textbook heuristic" is a stronger and more legible claim than "the arm differed", and it is only available because the reference is a documented-suboptimal policy rather than a null. EPModel has no optimal policy and cannot invent one from the corpus, so the honest version is a declared *naive* policy family (a person always pursuing, always distancing, always overfunctioning) reported beside every criterion as a bar the model must clear to be interesting — not as a target.

**6. Reported failure modes.** (a) Silent substitution: a failed call became "no valid order", so the company ordered zero and the failure was *also* logged as a rule violation, inflating both the bullwhip ratio and the violation count (§6.1, Table 3: 22/288 and 8/288 in round 1). (b) Rerun divergence at identical settings, up to 17x (§6.2). (c) A retracted explanation: an earlier draft blamed failures for the Luna result; Table 4 showed that incomplete, because those games had no failures at all (§6.2). (d) Spread as large as the mean: GLM baseline Retailer 7.0 (8.6) (Table 2). (e) Monitored behaviour only — agents were told violations are recorded (§7). (f) Rule violations were rare and mostly over-cash (1, 7, 1 of 288 per model), too few to compare conditions (§5). [INFERENCE] (a) is the transferable one at the level of principle, and its EPModel form is not an API failure: it is any outcome the engine produces *because* a mechanism could not run — a fallback pick (`M4.D.1f`), a tie-break, a deferred sequence step (`M5.D.9`) — being counted in a readout as if it were a choice. The spec flags the rate (`M11.D.18`) but does not stop the outcome entering the move marginals.

**7. Software engineering.** Open code, prompts, experiment log and every logged decision including superseded runs (§6.3, Code availability). Four practices adopted after the pilot: failed calls recorded separately with cause, output length and rate-limit waits; each call's worst-case cost reserved before it starts; a reliability table per model printed beside the results; superseded runs and an experiment log kept rather than overwritten (§6.3). [INFERENCE] Only the last has no EPModel counterpart. Claim reservation is a compute analogue with no mechanism content; the reliability table becomes interesting only once the engine can produce a degenerate outcome, which is candidate MG1; retention of superseded records is a genuine gap (MG4).

**8. Small N, families, long horizon.** Nothing on families, emotional process, dyads, triangles or multi-decade dynamics; the model has no household unit and no interior state. The small-N content is the pilot's own: four agents, 12 periods, 288 decisions per agent type, three seeds, and the abstract's own verdict that "with three seeds, conclusions ... did not survive a rerun" (§6.2). [INFERENCE] The transfer is a discipline, not a finding. EPModel's per-criterion ensembles are already larger, adaptive and paired; the paper adds nothing to that except the demonstration that the discipline is not optional, and one thing the spec does not have — a confirmatory replication on seeds not used to decide the verdict (MG3).

### Overlap check against the current spec (grep, `docs/bowen_agent_model_spec_v2.md`, rev 67)

- Determinism, keyed draws, byte-identical logs — `M3.D.4`, `M3.D.4a`, `M3.D.5`. Covers §6.2's mechanism; **already covered**.
- Paired per-seed statistic, declared margin, third outcome — `M11.4e`, `M17.A.3`, `M17.A.4`, `M17.A.1`. **Already covered**.
- Seed count set from observed variance — `M17.A.1` (adaptive blocks to a declared δ). **Already covered**; see MG3 for the part it does not cover.
- Seed / initial-condition / spell-timing variance components and per-seed strata — `M17.C.1`, `M17.B.1`. **Already covered**.
- Legal set recorded per selection — `M4.D.1e`, `M16.A.3b`. Covers "which move was never available"; not what the policy reached for (MG2).
- Fallback and tie-break provenance, rate reported and flagged — `M4.D.1f`, `M16.A.3c`, `M11.D.18`. Partly covers MG1; does not give the accounting identity or keep forced outcomes out of the readouts.
- Invariant failures recorded before the tick raises — `M16.A.4`. Covers "blocked violations as a first-class record" for invariants.
- Refused deliveries logged as their own event class — `M1.B.3c` in the update-order table. Covers the *delivery* half of "blocked attempt"; MG2 is the *selection* half.
- Belief against truth reported separately — `M9.5`, `M16.A.5a`, `M17.B.3`. Covers "agents see only their own state".
- Readout imputation disclosure — prior candidate SR2 (2026-11-09 batch). Adjacent to MG2, different object (a readout bound, not a selection score).
- Amplification control — prior candidate GP3 (2026-11-16 batch). Asks whether a mechanism is too weak to see; MG3 asks whether a verdict replicates. Not the same.
- Record provenance — `M16.A.1` header carries seed, config hash, spec revision, family id; nothing requires retention or supersession (MG4).

## B. Candidate additions to the EPModel spec

### MG1. Every person-tick resolves to exactly one recorded class, and a forced outcome is never counted as a choice

**Proposed requirement.** **M16.A.3d** — *the decision census.* For each living person each fast tick the log **MUST** contain exactly one outcome record, whose class is one of: a move selected by the scored policy; a move selected by the tie-break; a fallback outcome (`M4.D.1f`); a sequence step deferred under `M5.D.9`; or `WITHHOLD` chosen by the policy. The ensemble report **MUST** print the census per person and per arm — living person-ticks equal to the sum of the classes — and a person-tick with no outcome record, or with two, **MUST** fail the run as a log defect. Every move-marginal, transition matrix (`M17.B.5`) and `M11.C` readout **MUST** carry tie-break and fallback outcomes as their own row and **MUST NOT** count them as the move they resolve to; `M11.D.18`'s rate is reported from the same census, so the rate and the rows cannot disagree.

**Where it would live.** `M16.A` (the record class and the identity), with the reporting half in `M11.D` beside `M11.D.18`; the census is also an annex of `M11.G.1`'s structured evaluation. `→E` for the ensemble report.

**Evidence.** §6.1 and Table 3: 22/288 and 8/288 failed calls were recorded as "no valid order", setting the order to zero, and "the failure was logged as a rule violation" — so one infrastructure fault moved both the economic metric and the compliance metric, upward. §6.1 states the rule the paper now follows: distinguish "the agent decided X from the agent never decided" and report the second separately (SHOWN as practice, after the failure). The EPModel form is INFERENCE: the substitution path in a deterministic engine is not an API failure but a fallback, a tie-break or a deferral, all of which already exist (`M4.D.1f`, `M5.D.9`).

**What it changes.** Adds a test and a reporting rule; corrects a latent defect. Today the provenance is recorded per record (`M16.A.3c`) and the *rate* is flagged (`M11.D.18`), but a criterion's readout still pools those outcomes into the move they resolved to, so a gate that closes (a tie cut off, a third party unreachable) can read as a family that stopped pursuing. Improves inference validity, and theory fidelity only in the indirect sense that "no one pursued" and "pursuit was structurally unavailable" are different facts about a family.

**Cost / risk.** Low: one record class, one assertion over the log, one table in the report. The identity is exact, so the test is cheap and cannot be satisfied by a tolerant comparison. No conflict with `M3.D.5` (the census is computed from records already emitted) or with `M16.B.3` (observer-side). Risk: `WITHHOLD` must stay a *chosen* class or the census becomes a second way of saying "no move"; the distinction has to be stated once in `M4.D.1` and not re-derived in the report.

### MG2. The selection record carries what the policy reached for, not only what it was allowed to reach

**Proposed requirement.** **M16.A.3e** — *the masked score.* Alongside the legal set (`M16.A.3b`), the selection record **MUST** carry, for the outcome the policy would have ranked first **before** `M4.D.1e`'s mask was applied, its score and its rank, computed by the same scoring function with no additional draw; where that outcome has no target under the current structure (no live tie, no reachable third party) the score **MUST** be recorded as undefined rather than as zero, and those ticks are counted separately. The ensemble report **MUST** give, per move and per person, the rate at which the pre-mask first choice was illegal and which legal outcome was enacted instead, and **MUST** label the pre-mask quantity a **score**, never a probability: it is not normalised over the legal set and does not sum to one.

**Where it would live.** `M4.D.1e` (evaluation order) and `M16.A.3b`/`A.3e` (the record); the readout in `M17.B` beside the transition matrix.

**Evidence.** §3: "The engine blocks any order that breaks the truck limit ... and records it as a violation" — the blocked attempt is the observation, not the substituted order; §5 then reports violations as an outcome in their own right (SHOWN for the practice). `M16.A.3b`'s stated purpose — a move never legal is distinguishable from one legal and never chosen — is half of the same distinction: it does not distinguish *legal and not chosen* from *illegal and wanted* (INFERENCE).

**What it changes.** Adds mechanism observability and a readout; it is the family-model analogue of the paper's violation ledger, and it is theoretically the more interesting half. A family whose structure forbids what its members reach for is a different claim about triangles and cutoffs than a family whose members simply do not reach; the spec can currently report only the second. Improves both theory fidelity (the gate's pressure becomes visible) and inference validity (a readout that moves because the mask moved is identifiable).

**Cost / risk.** Medium. Scoring an outcome with no target requires every term of `M4.D.1`'s score to be defined off the legal set, and `M4.D.1f`'s fallback exists because some scores are non-finite — so "undefined" is a real and frequent value, not an edge case, and must be a declared class. It **MUST NOT** consume a draw: adding one would shift the stream and break `M3.D.4a`'s per-event keying. It is a score, not a propensity, and reporting it as a propensity would invite exactly the misreading `M16.A.9` warns against for entropy. Cost is one extra evaluation per person-tick on the slow path of the selector; `M11.D.19`'s reachability test already enumerates illegal moves, so the machinery is close.

### MG3. A verdict is provisional until it replicates on a fresh, pre-declared seed block

**Proposed requirement.** **M17.A.5** — *confirmatory replication.* Every `M11.C` verdict **MUST** be reported from a **primary** block and then confirmed on a **confirmatory** block that is disjoint from every seed the criterion has already used, whose size is declared before the block is drawn, and which **MUST** be run after the primary verdict is recorded and before it is reported as a finding. Both blocks' seed ranges, sizes and per-seed differences **MUST** be recorded separately. A criterion that holds on one block and not the other **MUST** be reported **NOT-REPLICATED** — neither pass nor fail — and **MUST NOT** be described in any output as supported; the record states both blocks' intervals. A mutation clause is judged the same way: a mutant that turns the criterion red on one block and not the other does not establish that the mechanism carries it.

**Where it would live.** `M17.A` (Phase E), with the verdict vocabulary in the `M11.4` exceptions table beside `UNDETERMINED`; it needs no engine change, so it is `→E` except for the seed-allocation bookkeeping, which belongs in `M16.A.1`'s header.

**Evidence.** §6.2 and Table 4: identical prompts, seeds and settings, zero failed calls in both runs, Factory bullwhip 6.9 → 113.4 (seed 2) and Supplier 8.0 → 134.2 (seed 1); "With three seeds, conclusions about how a disruption affects a given model did not survive a rerun" (Abstract, §6.2) — the authors' own headline (§1 contribution 3 says so explicitly). §6.3 and §8 adopt this: seed counts set from observed variance, seed-level spread and intervals reported (SHOWN).

**What it changes.** Adds a test and a reporting rule; corrects an assumption in `M17.A.1`. A precision-based stopping rule reads a *single* seed stream and stops when the interval is narrow, which is exactly when the running estimate sits furthest from zero; with heavy-tailed per-seed differences — the paper's Table 4 is a heavy tail — a small block can look decisive. Replication on seeds untouched by the stopping decision is the cheap repair, and it is one-sided (only passes need confirming), so it does not double the cost of the failures.

**Cost / risk.** Doubles the ensemble cost of every criterion that passes, one-sided; on the Phase C 1,000-seed ensembles that is real money in wall-clock, and the owner may prefer to scope it to criteria above a declared effect margin or to those that gate a phase. It does not conflict with `M3.D.5`, `M11.F.9` or `M3.D.6`; seeds are already the unit of the design (`M17.A.3`), so a disjoint block is a bookkeeping rule, not a new statistical device. Weakness to state: a NOT-REPLICATED verdict is a third outcome that will accumulate, and the spec's existing practice — `M5.D.3`'s T-158 already says a divergence is "not read as met until fresh, disjoint seed blocks show it" — shows the project already believes this for one criterion; MG3 makes it a rule and gives it a verdict name.

### MG4. Ensemble records are append-only, and a superseded verdict cannot be quoted bare

**Proposed requirement.** **M16.A.11** — *record identity and supersession.* Each ensemble record **MUST** be persisted under its own identity and **MUST NOT** be overwritten; it **MUST** carry `M16.A.1`'s header plus the seed range used and the criterion's verdict as rendered. A record that replaces an earlier one **MUST** name the earlier record's identity and state the declared reason it was superseded (a spec revision, a defect fix, a changed readout, a changed seed set). Any document quoting a verdict **MUST** cite the record identity, and a verdict taken from a superseded record **MUST** be labelled superseded at the point it is quoted. The engine still writes nothing (`M16.B.1`); this is the runner's and the repository's obligation.

**Where it would live.** `M16.A` (the record header, beside `M16.A.1` and `M16.A.7`) and Phase E's runner; the citing rule belongs with `M11.F`'s framing block.

**Evidence.** §6.3 (iv): the project now "keeps superseded runs and an experiment log rather than overwriting them"; §6.2 is a worked case — an earlier draft attributed the Luna result to failed calls, Table 4 showed that explanation incomplete because those games had no failures (SHOWN as practice and as a retraction). EPModel already does this by hand in its criteria table ("Regenerated at revision 32", "superseded at revision 12"), which is evidence the need is real and unmet by any requirement (INFERENCE).

**What it changes.** Adds a records/provenance rule; improves inference validity only. It is the cheapest of the four and the one that protects every other finding: without it, a criterion's history is a set of notes in a document that can drift from the record it describes, and `M11.F.9(c)`'s independence requirement (no arm fitted to a known history) cannot be audited after the fact.

**Cost / risk.** Low — a header field, a directory convention, a citation rule. Risk of over-formality: the project regenerates records often, and a requirement that every regeneration be justified in writing could become a clerical tax; scope it to records that carry a **verdict** or that a report cites. No conflict with `M16.B` or `M3.D.5`.

## Not transferable / cautions

- **The economic setting does not transfer.** No money, no inventory, no third-party customers, no optimal policy, and no externally imposed enforceable rules. A family has nothing it can be "compliant" with in the paper's sense: what looks like a rule in a family (a taboo, a cutoff, an unspoken topic) is endogenous to the family's own emotional process and is a *mechanism* in EPModel (`M1.B.10`, `M1.B.3c`), not a constraint an engine enforces. Copying "constraint compliance" as a reported metric would import a scoring device into a model that has no scorer — the honest transposition is MG1/MG2, which report *substitution* and *suppression*, not compliance.
- **End-of-horizon effects do not transfer.** Agents were told the game lasts "about 12 periods" to limit end-of-game behaviour (§3). EPModel's policy has no forward model and cannot anticipate year 40, so the design concern is void; the opposite caution applies — do not read the last year's values as a settled state, which `M17.F.2` already governs.
- **API failure modes have no analogue.** Output caps, HTTP 429s, hidden reasoning consuming the whole budget, malformed JSON (§6.1, Table 3) cannot occur in a pure engine with no LLM in the decision path (`M3.D.6`). Do not write requirements against them.
- **Keyword screening and human adjudication do not transfer**, and should not: the paper's collusion measure has unknown recall and needed a person to read the flags (§5, §7). EPModel must never acquire an adjudication step.
- **Model-level conclusions are not available.** Three small models, one prompt, one reasoning effort, one game (§7). Nothing here supports a statement about LLM agents in general, and the paper says so itself.
- **The paper's headline remedy is already EPModel's method** — intervals, seed-level spread, counts set from observed variance (`M17.A.1`, `M17.A.4`, `M17.C.1`). Its real contributions to this project are the two disciplines it had to learn the hard way: *never let an unavailable decision enter a metric as a value* (MG1, MG2) and *never let a narrowed interval stand in for a replicated direction* (MG3).

---

# Part 6: Persona Simulation as Controlled What-If Testing (2611.14700v1)

Paper read in full: Zhu, Hou, Tao, Bo, Fan, He, Hu, Huang, Liu, Wang, Wei, Xu, Xu, Yuan, Zuo et al., *Probe What-If
Worlds Before You Ship: Persona Simulation for Real-World Decision Making*, arXiv 2611.14700v1, 10 Nov 2026
(`/tmp/2026_PersonaSimulation_WhatIf_2611.14700v1.txt`, 1,211 lines: main text §1–5, twelve case reports App. B,
statistical protocol App. C). Read against `docs/bowen_agent_model_spec_v2.md` (2,225 lines, rev 65) and the
`papers/SPEC_CANDIDATES_*` series.

**What the paper is.** An industrial methodology plus execution infrastructure for turning a product question into a
controlled experiment with *simulated users*: each "user" is an LLM conditioned on a persona drawn from a 1,290-dimension
attribute corpus, each "trial" is one persona completing one task under one stimulus, and the pipeline runs cohorts ×
stimuli across four environments (survey, web, chatbot, OS app) with trial-level logging and cohort-level statistics.
Twelve partner cases in five domains. The load-bearing findings for anyone outside that business are negative and
methodological: the *specification of the audience* is not a nuisance input (ablating it reverses which design wins), and
cross-model agreement is a claim about direction, never about magnitude.

---

## A. Report

### 1. Simulation formalism

1.1 **[PAPER]** The experimental unit is defined by four things fixed before anything runs: "the target population, the
stimulus to be evaluated, the comparison of interest, and the decision the experiment is intended to inform" (§2.1).
Designs are classified on two axes — whether the cohort varies, whether the stimulus varies — giving *descriptive*
(one cohort, one stimulus), *SUT* (cohort fixed, stimulus varies), *PUT* (stimulus fixed, cohort varies) and *crossed*
(both) designs (Fig. 2). This taxonomy is the paper's only formalism, and it is a design taxonomy, not a state model.

1.2 **[PAPER]** A trial is "one simulated persona completing a task under one arm", and isolation is enforced: "Each
persona receives only the stimulus assigned to that arm" (§2.2). Trials are independent and run concurrently across
workers with cohort membership, sampling seeds, stimulus versions, instructions and model configuration all fixed (§2.2).
There is no time step, no event queue, no shared state between agents, and therefore **no simultaneity problem at all**.
The only temporal structure is the length of one task (a web session capped at six pages, C6, App. B.6).

1.3 **[PAPER]** Persona attribute space is 1,290 dimensions "whose dependencies are encoded in a directed acyclic graph"
(§2.1, citing Li et al. 2026); a population description is mapped to constraints on those dimensions, approximate
mappings are declared as approximations ("listening frequency is not equivalent to engagement", §2.1), and "Logically
inconsistent combinations … are excluded" (§2.1). Determinism: seeds are fixed and named (C1 uses "seed 42" for both
models, App. B.1; the bootstrap uses seed 20261005, App. C), but no determinism or replay test is reported.

1.4 **[INFERENCE]** For EPModel this item is almost pure negative evidence: the paper solves a scheduling problem EPModel
does not have (independent one-shot trials) and does not have the one EPModel does (2,080 coupled ticks, one family,
simultaneous arrivals). The transferable part is the *design taxonomy*: EPModel's arms are already SUT-style (mechanism
varied, family held) and PUT-style (`M11.C.1`, `M11.C.16`, `M11.C.41` vary level with family held), and the paper's
"crossed" cell — both varied, to estimate a stimulus effect across cohorts — has no EPModel counterpart and probably
should not have one at N=12.

### 2. Agent architecture

2.1 **[PAPER]** A persona is "a set of attributes conditioning an LLM's behavior through agent harnesses" (§2.0). No
decision rule, memory, reflection, perception model or learning is described anywhere in the paper; the policy *is* the
LLM plus its harness. Heterogeneity across personas is the manipulated variable, not a modelled property.

2.2 **[INFERENCE]** Nothing here bears on `M4` (select), `M1.A.4` (the estimator), or learning. The paper is evidence for
`M3.D.6`'s existing "no LLM in the decision path" rule only in the weak sense that it demonstrates how much of a result
moves when the policy is a model with no specified mechanism — see 6.2 and 5.2.

### 3. Interaction and network structure

3.1 **[PAPER]** There is no person-to-person interaction. Trials are isolated by construction (§2.2); the only
"interaction" is a simulated user with a *system under test* (a website, a chatbot, a car assistant; C11, C12, App.
B.11–B.12). No dyads, ties, influence, thresholds, cascades or tie dynamics appear in the paper.

3.2 **[INFERENCE]** This is the scope limit, and it is not a small one: EPModel's object *is* the dyad and the triangle,
and none of EPModel's central mechanisms (anxiety transfer, triangulation, pseudo-self exchange, cut-off) has any
analogue in a design where each agent meets a stimulus alone. Nothing in §3 of the brief (dyads, groups, influence,
tie dynamics, information distortion) is addressed.

### 4. Initialization

4.1 **[PAPER]** Five named cohort-composition strategies (§2.1): reference-data-informed sampling (target proportions
from human data), balanced comparison (equal representation of compared groups), targeted oversampling, broad coverage,
and stress-test sampling. With an explicit epistemic rule: "Balanced, exploratory, and stress-test cohorts do not
estimate population proportions" (§2.1).

4.2 **[PAPER]** Quotas are checked against the target and the residual is reported: in C5 realized marginals deviated from
target "by at most 0.1 percentage points for customers and 1.4 for agents", with all other attributes varying freely
(App. B.5). Reference-data cohorts are aligned to named external sources — Census 2011, HCES 2023–24, NNMS 2017–18 — with
609 rural and 291 urban personas (App. B.10). A deliberately balanced grid (4 economic motivations × 4 trust categories,
25 per cell) is stated not to estimate population proportions (App. B.9).

4.3 **[PAPER]** The paper's cleanest piece of initialization engineering is *matched counterfactual versions*: 50 base
personas, each with four variants crossing two priorities with two disclosure styles, "Other persona attributes were held
fixed across versions" (App. B.11); the same construction in B.12. This is a matched-pair design generated by
construction, not by hand.

4.4 **[INFERENCE]** EPModel already does 4.3 by hand (`M11.C.1`, `M11.C.16`, `M11.C.41`, `M11.C.49` all hold
everything but a declared attribute), and 4.1/4.2 map onto `M17.C.1`'s `D0` and `M15.B.1`'s ranges. What is *not* in the
spec is the paper's discipline of naming which composition strategy a given ensemble is and refusing population
inferences from the non-representative ones — `M15.D.2` covers imported families, but nothing covers a *reference-family*
ensemble. This is folded into PW1 rather than proposed separately.

### 5. Calibration and validation

5.1 **[PAPER]** There is no behavioural ground truth and the paper says so twice: "Incomplete human reference data and
imperfect behavioral representation limit the precision of real-world estimates" (Limitations), and the intervals "do not
measure discrepancies between simulated and real-user behavior" (Limitations; repeated in App. C's "Scope of the
intervals"). Case reports consistently caveat at the level of the claim: "simulated stated preferences rather than
observed purchases" (B.3), "neither establishes completed onboarding or human conversion" (B.6).

5.2 **[PAPER]** Cross-model robustness is claimed at the level of *direction only*: three models (GPT-5.6-Sol, Claude
Opus 5, Gemini 3.7 Flash) all give higher preference satisfaction for comfort-first than runtime-first personas, and all
favour proactive disclosure on the car stop-cancellation task (Fig. 3, App. B.11–B.12); professional group remains
associated with screening priority in all three (Fig. 4, App. B.2). The paper's own gloss: "shared hypotheses for follow-up
testing, not interchangeable quantitative estimates" (§4.1). Magnitudes are not preserved: the C9 lift is +28.50 pp under
GPT and +9.25 pp under Opus (Table 10).

5.3 **[PAPER]** Per-cell agreement is measured, not assumed, and it is poor where it can be measured: across 4,995 matched
customer cells the two models picked the same price rung 27.0% of the time (71.6% within one rung, Spearman 0.41, mean gap
$25.40), and across 4,896 agent cells 31.5% (82.6% within one rung, 0.56) (App. B.5, Table 6). One task reverses by model:
the breakfast-stop contrast is +4, −36 and +14 points under GPT, Opus and Gemini (App. B.12).

5.4 **[PAPER]** Repeat variability is given its own floor, and it is dwarfed by model choice: within-model mean absolute
item differences 0.192 (GPT-4.1-mini) and 0.469 (GPT-4o-mini) against 0.754 between models, difference 0.423 [0.393,
0.457], on 32 personas × 3 calls with whole-team resampling (App. B.7, Table 8).

5.5 **[PAPER]** The ablations are the strongest result in the paper and they are reported as reversals, not as decays:
- Omitting persona descriptions collapses the group–priority association from Cramér's V = .928 to .063, and the
  no-persona control then favours the same option (complete eligibility checking) in all four bookkeeping groups at
  58.0–63.0% (App. B.2).
- In the engagement-letter study the combined-minus-liability-only acceptance contrast goes from +17.25 pp with personas
  to −15.50 pp without them, "reversing the simulated ranking of the revisions"; the directly estimated difference is
  +32.75 pp [25.00, 40.50] (App. B.9, §4.2).
- In the music-wellbeing study, without personas the check-in-first trust gain falls from +0.252 to +0.035 points while
  the intent gain *rises* from +0.272 to +0.351, and the warm-peer trust penalty grows from 0.093 to 0.640 (Table 2).
- The paper states the limit of its own ablation: "Different prompt assembly prevents attributing this reversal solely to
  persona removal" (§4.2).

5.6 **[PAPER]** Multiplicity is handled case by case and disclosed: one prespecified primary contrast with 19 secondary
factorial contrasts under Benjamini–Hochberg (B.9); 12 comparisons with no correction, stated (B.4); the generic
module defaults to uncorrected inference and subgroup labels on raw p-values "are treated as exploratory unless a separate
correction is documented" (App. C). Working success rules are declared in advance and honestly missed: "including 70%
acceptable intent and a 0.30-point favorability gain. Neither met the working success rules" (B.9).

### 6. Reported failure modes

6.1 **[PAPER]** **Removing the persona homogenises the cohort.** With no persona descriptions every group converges on the
same answer (58.0–63.0% in all four groups, B.2); this is the mechanism behind 5.5's collapse, and it is the paper's
central warning: "audience specification can materially affect which design the simulation favors" (§4.2).

6.2 **[PAPER]** **A model can report near-total eligibility, and that is the quantity most likely to be wrong.** Under
gpt-5.6-sol essentially the whole cohort is addressable (97.3%–100%); under gpt-5.6-luna relevance ranges 30.9%–75.2%,
"varied by a factor of two" — while *conditional* on relevance the two models agree closely on the barrier (distrust 92.1%
vs 95.6%; price 0.04% under both) (App. B.5). The difference between models lives almost entirely in the reach term, not
the per-respondent term.

6.3 **[PAPER]** **Interest is not willingness, and the two are measured separately.** 97.6% reached the signup entrance and
0.9% would register (B.6, Table 7); 98% rated a cheap offer a good deal and 51.4% would subscribe (B.3); 1,076/1,088
prospects continued reading and no prospect accepted a pilot (B.8, Table 9). The paper treats the gap as the finding.

6.4 **[PAPER]** **Schema validity is not semantic consistency.** All 584 scheduled questionnaires were schema-valid, and
inspection found five value–explanation contradictions among 192 sampled answers (B.7); a uniform 212/212 non-adherent
classification is flagged as an instrument or setup problem rather than a result (B.10).

6.5 **[PAPER]** **Endpoint confounding is disclosed, not corrected.** C11's Opus and Gemini personas interacted with one
assistant endpoint and the GPT personas with another, "so cross-configuration comparisons also reflect an
assistant-endpoint difference" (B.11, §4.1). Missing component scores are not imputed ("Missing component scores are not
filled with zero", B.11).

### 7. Software engineering lessons

7.1 **[PAPER]** A *study specification* is the reusable unit: cohort membership, sampling seeds, stimulus versions,
instructions, model configuration and outcome definitions are fixed and can be reused "after changing a stimulus, design,
or simulation model" (§2.2). Each trial retains persona, rendered inputs, trajectory and structured outcome, and
"Aggregation preserves links to these records and reports failed or missing trials" (§2.2). Reports "preserve the
valid-response denominator for each outcome" and "distinguish missing responses from recorded choices" (App. C).

7.2 **[PAPER]** The paper names a defect in its own instrumentation rather than hiding it: its generic categorical tests
"assume independent arm observations and do not account for shared-persona pairing, so their inferential interpretation
requires verification for repeated-person designs" (App. C); the workplace study therefore resamples whole teams,
"preserving within-person responses" (B.7). Bootstrap seed, resampling unit and per-comparison pairing are all recorded
beside the number (App. C).

7.3 **[INFERENCE]** EPModel's `M16.A` is already far ahead of this (per-event causes and effects, selection rationale,
legal set, belief-vs-truth). The one thing here EPModel does not have is 7.2's explicit rule about the *unit of
resampling* — which matters more, not less, when 12 persons × 2,080 weeks is a pool of ~25,000 person-weeks belonging to
one family. See PW4.

### 8. Small-N, emotion, family, long horizon

8.1 **[PAPER]** Nothing. The family appears only as an attribute (household resources, C10) and as one questionnaire area
("family influence", B.10). There is no household unit, no emotional dynamics, no stress process and no run longer than a
single task. Trust, comfort and privacy confidence are single-shot 1–7 or 1–5 scale items (B.1, B.6). Cohorts are large
(50–1,000 personas per arm) and the paper's inference is frequentist at that scale, which is the opposite of EPModel's
N=12-seed problem.

### Not transferable / cautions

- **The unit of analysis is inverted.** The paper's agent is alone with a stimulus; EPModel's agent exists only in ties.
  Every mechanism EPModel cares about is absent, so "the paper shows persona specification matters" must not be read as
  evidence about how a *family member's* attributes matter.
- **The ablation result does not transfer as a mutation test.** In the paper, removing the persona removes the causal
  channel (the LLM's conditioning). In EPModel, ablating `basic_level` removes a state variable nine rules read, which is
  `M11.1d`'s existing mutant, not a new protocol. What transfers is the *inference* (a between-stratum contrast can be a
  property of the specification), not the procedure.
- **Nothing here is a result about people.** The paper says so; the twelve cases are hypotheses for live testing. Any
  EPModel citation of it must be for method, never for magnitude.
- **The paper's model-independence is not EPModel's constant-independence.** Three LLMs agreeing on a direction is not the
  same warrant as one parameter set cancelling between two arms (`M0.4`); the paper's own per-cell agreement (27%/31.5%)
  shows how little cross-model agreement implies at the individual level.
- **Large-N frequentist apparatus is the wrong shape.** 4,995 matched cells and 10,000 bootstrap draws have no analogue at
  12 agents and 12–1,000 seeds; EPModel's per-seed paired direction and sign tests (`M11.4e`, `M17.A.3`) are the right
  instrument and this paper adds nothing to them.
- **Cohort sizes here are a warning, not a target.** A 1,000-persona cohort conceals per-persona instability that a
  12-person family exposes; do not read "1,000 trials" as a precision claim.

### Already covered — flagged, not proposed

- Matched/paired arm statistics and the ban on unpaired comparison: `M11.4e`, `M17.A.3`.
- Pre-declared minimum effect margin and equivalence bounds: `M17.A.4`, `M11.4a`.
- Control arms, homogeneity arm, rival-mechanism arm, both signs of a perturbation: `M17.D.1`, `M17.D.2`, `M17.D.5`.
- Constant sweeps, dose–response, invariance intervals, two reference configurations, alternative activation regime:
  `M17.E.1`–`M17.E.6`.
- Robustness audit record with per-dimension held/sensitive/unaudited and a claim grade: `M17.G.1` (PW1 tightens one cell).
- Non-occurrence fractions kept in the denominator; failed or missing trials reported: `M17.F.1(c)`, `M11.1f`.
- Arms differing only through declared channels; confounds named when they do not: `M17.D.3`, `M11.C.49`'s declared
  confound.
- Import as ranges with an envelope, and a sign flip inside the box reported as a result: `M15.D.2`–`M15.D.4`.
- Reuse of a frozen study definition across a changed stimulus or model: `M16.A.1` (config hash), `M17`'s arm spec.
- Intervals described as within-design variation only: `M11.F` framing block, `M17.G.1`'s claim grade.
- Per-opportunity denominators for move rates: proposed already as PR1 (`SPEC_CANDIDATES_plain_language_2026-10-12.md`);
  PW3 extends it to outcome readouts, and notes `M11.1e`/`M16.A.8` already do this for asymmetric constants.
- Neutral-persona contrast for an LLM persona layer: proposed already as X4 (`SPEC_CANDIDATES_from_preprints_2026-09-20.md`);
  that line is gated by `M3.D.6` and is not what PW1 is about.

---

## B. Candidate additions to the EPModel spec

Four candidates. Each is written to add something the spec does not already carry; where an existing requirement is close,
the overlap is named rather than hidden.

### PW1. A composite criterion's composition dimension must not be "unaudited"

**Proposed requirement.** A criterion of class composite or mixed (`M11.5`) that is to be reported as a **finding** **MUST**
carry, in its `M17.G.1` audit record, the family-composition dimension as **held**, **sensitive** or **reversed** over at
least one structurally different instance — a second family declared under `M2`, a redraw of `M17.C.1`'s `D0` with a
different topology or sibling structure, or an `M15` import — and **MUST NOT** carry it as **unaudited**; a direction that
holds on the twelve-person reference family and reverses or vanishes on the other instance **MUST** be reported as a
property of that composition and **MUST NOT** be reported as a mechanism finding.

**Where it would live.** `M17.G.1a` (tightening `M17.G.1`'s existing dimension list), read with `M11.5`'s class and
`M15.D.2`–`M15.D.4`.

**Evidence.** The paper's central experimental result is that the specification of the population is load-bearing to the
point of reversing a ranking: group–priority association falls from Cramér's V = .928 to .063 without persona descriptions
(App. B.2), and the combined-minus-liability-only acceptance contrast moves from +17.25 to −15.50 pp, "reversing the
simulated ranking of the revisions" (§4.2, App. B.9). Strength: **SHOWN** for the phenomenon; **INFERENCE** for the
EPModel mapping (instance composition ↔ persona specification).

**What it changes.** It adds a test/reporting gate rather than a mechanism. It corrects a default: `M17.G.1` today admits
"unaudited" as a value for every dimension, so a composite pass can be reported as a finding with its composition never
perturbed. Theory fidelity unchanged; inference validity improved, and it is the natural extension of `M15.D.2`–`M15.D.4`
from imported families to the reference family.

**Cost / risk.** One extra run per composite criterion against a second family configuration; `M13.3`'s phasing means it
arrives in Phase E, and the second family can be `M2`'s Phase C family or a fixture, not a hand-built one that is more
balanced than a real family (`M15.E.2`). No conflict with `M3.D.4`/`M3.D.4a` (the two instances are separate runs, not
coupled arms) or with `M11.F.9` (no real family is claimed); it must not become a fitting route — the second instance is
declared before the first result is seen, per `M10.B.4`, and "reversed" is a reportable outcome, never a reason to pick a
third instance.

### PW2. Every criterion names one guardrail readout with a bound declared before the run

**Proposed requirement.** Each `M11.C` criterion **MUST** name, beside its primary directional readout, at least one
**guardrail** readout — a quantity its own mechanism does not target, computed on the same arms — with a bound graded
`[I]` and declared in config before the run; a pass whose guardrail breaches its bound **MUST** be reported as a **pass
with a stated cost** and **MUST NOT** be reported as support for the mechanism.

**Where it would live.** `M11.2a` (beside `M11.2`'s ensemble rule and `M11.5`'s class table), with the bound in
`M10.B`.

**Evidence.** The paper's method defines three outcome classes and keeps them apart: "decision-relevant outcomes,
diagnostic measures, and guardrails" (§2.1), and its case reports read a gain against a cost — check-in-first raises trust
while privacy confidence is tracked separately (App. B.1, Table 2), and the combined engagement-letter revision raises
acceptance while "the 100-person hostile-trust group had no acceptable responses in either model" (App. B.9).
Strength: **SHOWN** that the pattern exists and is what makes a recommendation safe to act on; **ARGUED** as a spec rule.

**What it changes.** It adds a test and a reporting rule. EPModel criteria today assert one direction and report the rest;
nothing forbids a criterion passing while a quantity the theory cares about collapses. The live instances are exactly the
ones where the mechanism has a known cost: `M11.C.49` (a low-level `I-POSITION` raises family anxiety — the mover's own
estimated `basic_level` must not fall as the price), `M11.C.4` (cutoff relieves now, and the guardrail is family total
anxiety at the next nodal event, currently the criterion's *second* limb rather than a bound), `M11.C.12` (symptom relief
raising marital conflict is the finding; the couple's differentiation must not be what paid for it).

**Cost / risk.** Low: one config field per criterion, one comparison per report, no engine change. Risk is that a guardrail
bound becomes a second way to fail a criterion an owner wants to keep, or a way to rescue one — both are handled by
`M10.B.4` (bound recorded before the first run) and by requiring the guardrail to be a quantity the criterion's mechanism
does not target, which keeps it out of `M11.1d`'s outcome-directive audit. No conflict with determinism; none with
`M11.F.9`; it slightly increases the record a report must carry.

### PW3. Split every cumulative readout into reach and per-event effect, and say which carries the difference

**Proposed requirement.** For every criterion whose readout is a cumulative count or a share over the ensemble, the report
**MUST** give the **eligible denominator** (persons or weeks in which the mechanism could fire), the **count converted**,
and the **per-eligible rate**, and **MUST** state which of the three carries the between-arm difference; a difference
carried by the eligible denominator alone **MUST NOT** be described as the mechanism acting more strongly.

**Where it would live.** `M11.1e` extended (it currently scopes to mechanisms with asymmetric `[I]` constants) and
`M16.A.8`'s record, which already separates per-event magnitudes from event counts.

**Evidence.** In C5 the two models differ by a factor of two on who is even in the situation (97.3%–100% vs 30.9%–75.2%
addressable) while agreeing closely conditional on relevance (distrust 92.1% vs 95.6%; price 0.04% under both)
(App. B.5), and the paper draws the conclusion explicitly: the relevance estimate "is the quantity most worth establishing
against a real audience" (B.5). The same shape recurs where a denominator is the whole finding: 97.6% reached / 0.9%
would register (B.6), 98% good deal / 51.4% would subscribe (B.3), 212/212 on the fully scorable subset only (B.10).
Strength: **SHOWN**.

**What it changes.** It extends an existing rule's scope from asymmetric-constant mechanisms to every cumulative
readout. EPModel criteria that can pass on reach alone: `M11.C.10` (cut-offs in generation 3), `M11.C.16` (entropy
conditional on a move being available), `M11.C.48` (a baseline matched on how many members act on a tie — the spec already
fought this trap by hand), `M11.C.55` (supplier prediction over an eligible-window). Inference validity, not fidelity.

**Cost / risk.** Low: the quantities are already logged (`M16.A.3b`'s legal set, `M16.A.8`), so this is arithmetic on an
existing record. Risk of double-counting with `M11.1e` — write it as an extension, not a second rule. No conflict with
`M3.D.4`; no fitting risk (no new constant).

### PW4. The resampling unit is the seed's family, never the person-week

**Proposed requirement.** Any readout computed over more than one person or tie of a run **MUST** declare that clustering
and **MUST** be resampled at the family (seed) level, keeping a person's own responses together; an `n` reported for such a
readout **MUST** be a count of families and **MUST NOT** be a count of persons, person-weeks, ties or events; a test that
treats persons or weeks within one family as independent observations **MUST NOT** be used.

**Where it would live.** `M17.G.4` (with `M11.4e` cited as its arm-pairing counterpart), and it binds `M11.4`'s
within-run comparisons (`M11.C.48`) as well as ensemble criteria.

**Evidence.** The paper's own statistical protocol names the failure and its fix: generic categorical tests "assume
independent arm observations and do not account for shared-persona pairing, so their inferential interpretation requires
verification for repeated-person designs" (App. C), and the one study with a genuine group structure resamples
"10,000 whole-team resamples, preserving within-person responses" (App. B.7, Table 8). Strength: **SHOWN** as a stated
defect-and-remedy in a paper that shipped the module.

**What it changes.** It adds a reporting rule and closes a gap `M11.4e`/`M17.A.3` do not reach: those fix *arm* pairing
but say nothing about within-seed clustering, and the temptation is severe in EPModel — 12 persons × 2,080 ticks is a
~25,000-row table belonging to one family, and a readout averaged over persons with a per-person standard error would
report a precision the model does not have. `M11.C.35` (one exchange, one witness), `M11.C.48` (ties within one run) and
`M11.G`'s evaluation components are the immediate beneficiaries.

**Cost / risk.** Low: a resampling routine and one sentence per report. It reinforces rather than conflicts with
`M11.4e`; no determinism issue (resampling is observer-side, `M16.B.3`); no fitting risk.

---

