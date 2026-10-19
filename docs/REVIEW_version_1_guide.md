# Review guide: version 1 of the agent model

**Status: FOR OWNER REVIEW, drafted 2026-10-18 by Claude.** Nothing in it is decided, except the open questions on
`M11.C.46`'s mutation clause, the nodal spike's narrowing and the four pending disposition notes, which the owner
answered on 2026-10-19 (`docs/version_2_plan.md` §5, Q10). Version 1 was completed on
2026-10-18 with conditions V1–V7 met (`docs/version_1_definition.md`, `docs/version_1_status.md`); the spec is frozen
at v2.0 revision 32.

**What this is.** A guide for the owner's complete review of version 1, module by module (`M0` to `M17` of
`docs/bowen_agent_model_spec_v2.md`), in an order meant for reading beside one run: the seed-7 trace
(`docs/review/phase_c_trace_seed7.md`) and its fam-diag diagram (`docs/review/phase_c_famdiag_seed7.json`). It follows
the form of `docs/REVIEW_spec_revisions_24-30.md`, which the owner completed entry by entry on 2026-10-16. That form
reviewed revisions; this one reviews the model as it stands.

**How to use it.** Read an entry, look at the trace rows it names, then tick one box. *Approve* keeps the module as
built. *Change* says what to change: Claude drafts the revision and its plan, and nothing is rebuilt before the plan
is approved, as in the revisions 24–30 review. *Question* asks something Claude answers before the entry is decided. A
change here is input to version 2: `TODO.md` records that no spec revision or code is made before the owner answers
`docs/version_2_plan.md` §5.

**Each entry gives:**
- what the module requires, in plain words, with the requirement IDs to look up;
- the explainer section that explains it (`docs/model_explainer.md`);
- the config files in `config/bowen/` that feed it;
- the `M11.C` criteria that test it, each with its line from `docs/version_1_status.md` copied as generated (class,
  record, verdict, status, evidence, and the disposition where there is one);
- how many of the module's IDs a test names, from `docs/spec_coverage.md` (see "Coverage counts" below);
- one to three places in the trace where the behaviour can be seen, by week, with the row's "What it did" text quoted
  exactly; where the trace shows nothing, the entry says so;
- points to check, then the boxes and a notes line.

**What this guide does not do.** It makes no claim of its own about the corpus. The sources behind each module, with
their evidence grades (`[T]` `[M]` `[D]` `[#]` `[I]` `[X]`) and attribution grades (`[B]` `[K]` `[K-ext]`), are in
the explainer section each entry names. One run is a sample, not a result (`M11.2`; explainer §10): no criterion's
verdict can be read from this trace, and every number in it comes from invented constants. Where an entry says something about this
run (a count, a pattern), it is a fact about seed 7 only.

**How it was drafted.** Trace rows are quoted from the trace as committed at `b773efd` and placed by line number;
criterion lines are copied from `docs/version_1_status.md`; the coverage counts are read from `docs/spec_coverage.md`;
every spec ID cited was checked against the IDs the spec defines. A script in Claude's scratch space did the copying
and the checks; it is not part of the repository.

## Inconsistencies found while drafting

Disagreements inside the documents, listed so the review does not stall on them. None is fixed here.

1. `docs/version_1_status.md`, row `M11.C.9`: the Evidence column says "out of version 1, Claude's recommendation
   2026-10-17 pending owner review", while the Disposition column is tagged "*(owner 2026-10-17)*" and `TODO.md`
   records every disposition accepted on 2026-10-17 except `M11.C.17` and `M11.C.6`. The Evidence text looks stale.
2. `docs/spec_coverage.md` is titled "Spec coverage — after Phase C", but it names Phase D tests (for example
   `M7.D.4`'s).
3. `docs/SPEC_REVIEW_CHECKLIST.md`, Part 1, describes the spec as "(v2.0, revision 27)"; the spec is at revision 32.
4. The same checklist calls `config/bowen/family_phase_d.md` "the owner's reference family". V7's review
   (`docs/REVIEW_version_1_outputs.md`) changed "the owner's family" to "the reference family the owner specified" in
   four documents so that no reader takes it for a real family; the checklist's wording was not among them.

---

## Reading the trace and the diagram

### The run

- **Seed 7, 104 weeks (weeks 0–103), the Phase C family** (`config/bowen/family_phase_c.md`): Ravi (52) and Marta
  (50); their daughters Nadia (17) and Pia (14), who live with them; Ana (78), Marta's mother; Sofia (76), Ravi's
  mother; Bruno (74), Ana's brother, declared cut off from her at the start; and Dr Halim, the external agent, with
  thin ties to Ravi and Marta. This is **not** the twelve-person reference family of `M2.A`
  (`config/bowen/family_phase_d.md`); see entry 2.
- **It is `M11.C.1`'s baseline arm** (`tools/phase_c_trace.py`): the family at its own levels, under the policy and the
  learner, with the declared spell, a `JOB_LOSS` of intensity 120 to each parent every 4 weeks from week 4
  (`config/bowen/criteria.md`, the `spell` rows). That is why `JOB_LOSS` rows come in pairs every fourth week, 25 times.
- **Dr Halim acts only in session weeks**, every 8 weeks (`session_interval_weeks` = 8, `config/bowen/constants.md`):
  weeks 0, 8, 16, …, 96.
- The trace's header records the seed, config hash, spec revision, activation and visibility components, the freeze
  date and the 78 constants changed since the freeze (`M16.A.1`, `M16.A.1a`, `M16.A.7`).

### Reading a row

Columns: Week · Who · Move · Toward · Witnesses · What it did. "What it did" lists each effect beside the act that
caused it (`M16.A.4`, `M16.C.2`). The phrases, and the requirement each one shows:

| Phrase in "What it did" | Requirement |
|---|---|
| "arrives week N" | delivery after the tie's latency (`M3.C.2`); Ravi–Sofia has latency 2 |
| "anxiety X +n", and "(witness)" | change in acute anxiety at delivery; a witness's own appraisal (`M1.F.5`, `M4.C.9`) |
| "calmer sender takes some anxiety" | contact with a calmer person (`M4.C.10`) |
| "refused by X (rejection +n); the sender bears the refusal" | a cutoff is rejection held by the cutter (`M1.B.3c`) |
| "X rejects the other on A–B (rejection +n; the tie stays open)" | `CUTOFF` raises rejection; the tie is not severed (`M1.B.3c`) |
| "impingement removed at once" | `CUTOFF`'s immediate relief (`M4.C.1c`) |
| "n of the sender's anxiety bound into A–B" | `DISTANCE` binds anxiety into the tie (`M1.D.2a`) |
| "X left outside: anxiety …" | the `TRIANGLE` transfer to the outsider (`M1.C.1`) |
| "X takes n of functioning from the other", "functioning balance flips" | the functioning exchange (`M1.B.5`, `M6.I.4`) |
| "X holds a position: no functioning changes hands" | holding a position stops the exchange (`M1.B.5`, revision 21) |
| "held back MOVE: computed, not emitted" | `WITHHOLD` (`M4.D.1b`) |
| "the nodal event re-opens the cut tie A–B: its standing load spikes next week" | the nodal-event spike (`M4.C.1c`, revision 32) |
| "symptom onset, … channel" | an endogenous onset (`M7.D.1`, `M1.F.7`) |
| "assertion form (no systems perspective, or angry)" | `I-POSITION`'s assertion form (`M5.F.4`, `M5.D.4a`) |
| "the contact lands with X: systems perspective +n" | a landed external-agent contact (`M1.E.7`) |
| "bound anxiety released to the third party" | `REDUCE_CUTOFF`'s cost lands on a third party (`M5.B.3`) |
| system rows: "triangle active", "inside pair", "tie now distant", "rejection has faded", "slow tick (yearly)" | `M1.C.3` and `M3.D.1` step 6; `M4.G.1`; `M1.B.3c`; `M3.B.1` and `M7.E.4` |

**What the trace does not show.** Its header says: "The standing load, the decay toward each person's floor, the
relaxation of felt contact, symptom accumulation, a symptom's hold on the family's anxiety (lock-in), investment, the
reactive detectors, outside-ness, attribution, the family's focus and borrowed pseudo-self settling back run every
week for everyone and are not listed line by line." The rendered trace also does not print why a move was chosen:
the log records the selection rationale (`M16.A.3`) but the renderer shows only the move and its effects. Beliefs,
the family's budget and its sinks are not printed. The moves of week 103 show "—" because the run ends before they
arrive.

### Opening the diagram

- Open `docs/review/phase_c_famdiag_seed7.json` in **EPModel Fam-Diag** at http://localhost:5174 with **File > Open**.
  If the app is not running, it starts with the command below (from this Mac's `.claude/launch.json`):

  ```bash
  npm --prefix /Users/davemini2/ProjectsLocal/epmodel-fam-diag/src/frontend run dev
  ```

  The app is a copy of fam-diag whose timeline hides a line after its end year when the line asks it to; the owner's
  own fam-diag is not changed (`tools/famdiag_export.py`).
- **It is the same run as the trace** (`tools/famdiag_export.py`; its page note reads "Run: M11.C.1 baseline, seed 7").
- **Weeks and years.** Each year on the timeline is one block of 4 weeks: 2001 is weeks 1–4 and 2026 is weeks 101–104
  (the diagram's page note). **The diagram counts weeks from 1 and the trace from 0**: trace week N is the diagram's
  "Week N+1". Within a year the four weeks are dated 1 January, 1 April, 1 July and 1 October. Example: the trace's
  week 0 row "Bruno CUTOFF → Ana" is the diagram's event "Week 1: CUTOFF by Bruno to Ana", dated 2001-01-01, on the
  Ana–Bruno cutoff line.
- **What is drawn** (the export as committed at `b773efd`): 13 people (the 8 simulated, and Dario, Iris, Leo, Teodor
  and Viktor from the reference family, greyed and marked "Not simulated in this run"); 15 pattern lines (8 distance,
  6 over/underfunctioning drawn as fusion, 1 cutoff); 3 fixed triangles, each with Marta outside; and 712 events, the
  run's acts placed on the line they fall under or on the sender. The lines are the declared pattern readouts
  (`docs/readouts_phase_c.md`, `config/bowen/pattern_acts.md`), read at the end of each block; a line ends when a new
  pattern replaces it on that tie. The drawing rules are `config/viz/famdiag.md`.
- A copy saved from the app can differ from the committed export. The entries below describe the committed export.

### Coverage counts

Each entry gives the counts `docs/spec_coverage.md` (generated by `tools/spec_coverage.py`) reports for the module's
IDs. That page counts an ID **done** when a test names it. **"Not done" there means no test names the ID; it does not
mean the requirement is unbuilt.** For example, `M1.A.4a` (basic level is an estimator over functioning history) is
built in `src/bowen/engine/slow_tick.py` and tested under `M1.A.4k`, and the page lists `M1.A.4a` as not done. Read
the lists as "requirements no test names", which is worth reviewing in itself. The `M11.C` criteria are left out of
the counts; their status is `docs/version_1_status.md`'s.

---

## Part 1 — What you are looking at

### 1. `M0` — Status, scope and the parameter rule

**What it requires.** The spec covers Phases B, C and D: the objects, the event loop, the policy, the slow clock and
the reference family. Phase E (`M17`) and the family-diagram import (`M15`) are drafted, not built, except `M17.A.1`'s
stopping rule, which Phases C and D use (`M13.4`). Every numeric constant carries a grade, and the grade matches the
explainer's (`M0.1`); an `[I]` constant is never described as sourced (`M0.2`); every constant comes from config or a
derivation from `basic_level`, never a literal in engine code (`M0.3`); acceptance tests assert a direction of
difference between two arms, with the exceptions listed at `M11.4` and nowhere else (`M0.4`). IDs are never renumbered
(§0.4).

**Explainer:** "How to read an entry" (the grades); §15.3 (magnitudes the corpus supplies); §17.1 (why the
trustworthy output is a difference).

**Config:** `constants.md` (every constant with its grade).

**Criteria:** none in `M11.C`. The engineering tests for this module are `M11.D.2` (no magic literals), `M11.D.4`
(invented constants labelled) and `M11.D.8` (spec references resolve); they are in the suite, which V5 and V6 cover.

**Coverage:** 4 IDs: 2 done, 2 partial, 0 not done.

**Where to see it in the trace:**
- The framing block, trace lines 8–9: "Every number in it comes from invented constants (`[I]`); only differences
  between two runs that share them carry meaning (`M0.4`)."
- The header row "spec revision | 2.0, revision 32" (trace line 16).

**Points to check:** the trace prints effects to one decimal. Each size comes from `[I]` constants, so read a row for
its direction and its mechanism, not its size.

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

### 2. `M2` — The reference family, and the family this run uses

**What it requires.** One invented, tunable reference family, declared in markdown config and never in source
(`M2.1`); three generations, at least one cut-off tie and at least one family-of-origin tie per adult (`M2.2`). `M2.A`
lists its twelve people (Dr Halim is external, not a family member); `M2.B` declares only the ties that carry traffic,
including Dr Halim's tie to whoever is coached (`M2.B.2`). Every value is `[I]`, and the whole of `M11` carries the
burden of testing it. The spec's revision 28 note says the Phase B and Phase C instances keep the values the table had
before; they are the closed phases' records.

**Explainer:** no section of its own; §8.1 covers the life course the twelve-person family lives through.

**Config:** `family_phase_d.md` (the twelve, with starting ages and earners), `partners_phase_d.md` (partners and
births), `family_phase_c.md` (this run's family), `family_reduced.md` (Phase B), `criterion_roles.md` (who plays each
role in a criterion), and `fixtures/` (`dyad.md`, `triad.md`, `nuclear_four.md`, `danger.md`) for the criteria that
run on small fixtures.

**Criteria:** none tests `M2` as such. The criteria that run on the twelve-person family are listed in entries 21
and 22:
`M11.C.2[twelve-person family]` (pass), `M11.C.6` (fail accepted as a finding),
`M11.C.10` (fail accepted as a finding), `M11.C.10[family of origin]` (pass) and
`M11.C.15` (pass).

**Coverage:** 20 IDs: 11 done, 1 partial, 8 not done. No test names: `M2.2`, `M2.A.0`, `M2.A.0b`, `M2.A.0f`, `M2.A.0d`, `M2.A.0g`, `M2.A.1`, `M2.A.2`.

**Where to see it in the trace:**
- The header row "family instance | phase_c" (trace line 17).
- Week 0, Bruno CUTOFF → Ana (trace line 28). What it did: "arrives week 1; anxiety Ana +1.1, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open)"
  - This is the declared cut-off tie Ana–Bruno, the instance's one cut-off tie (`M2.2`). Bruno is in this instance only;
    the diagram says of him "In this run's family only; left the reference family at spec revision 13".
- Week 0, Dr Halim STAY-IN-CONTACT → Marta; witnesses Nadia, Pia, Ravi (trace line 29). What it did: "arrives week 1; anxiety Dr Halim +0.1, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Dr Halim +0.3, Marta -0.3"
  - Dr Halim's ties are to Ravi and Marta (`M2.B.2`).

**Points to check:** this trace runs the Phase C instance, not `M2.A`'s twelve. There is no rendered trace of the
twelve-person family in `docs/review/`, so the life course (entry 21) and transmission (entry 22) cannot be read
beside a run here.

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

### 3. `M16` — The run log and the readable trace

**What it requires.** A log that describes itself: a header with the seed, config hash, spec revision, activation and
visibility components, and which `[I]` constants changed after the suite first ran (`M16.A.1`, `M16.A.1a`, `M16.A.7`).
Every event with its full field set, and the selection rationale, the legal set and how the outcome was decided
(`M16.A.2`, `M16.A.3`–`M16.A.3c`); effects recorded beside their cause (`M16.A.4`); belief writes tagged apart from
true state (`M16.A.5`). The engine emits records through an interface and writes no file (`M16.B.1`). A deterministic
renderer, with a single-agent view (`M16.C.1`, `M16.C.4`); the trace carries the header and `M11.F`'s framing
(`M16.C.5`). A delayed view of one agent's own events (`M16.D`). Language-model narration is Phase F and off by
default (`M16.E`).

**Explainer:** §17.6 (reading a run: the log and the trace).

**Config:** none in `config/bowen/`. The diagram's drawing rules are `config/viz/famdiag.md`.

**Criteria:** none in `M11.C`. The tests are `M16.F`'s, `M16.T.1`–`M16.T.6`, in the suite.

**Coverage:** 34 IDs: 23 done, 3 partial, 8 not done. No test names: `M16.A.3c`, `M16.A.8`, `M16.A.9`, `M16.A.10`, `M16.D.1`, `M16.E.1`, `M16.E.2`, `M16.E.3`.

**Where to see it in the trace:**
- The header table, trace lines 12–21.
- Week 0, Bruno CUTOFF → Ana (trace line 28). What it did: "arrives week 1; anxiety Ana +1.1, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open)"
  - The first event row: each effect is beside the act that caused it.
- Week 103, Ana OVERFUNCTION → Bruno (trace line 921). What it did: "—"
  - The last week's moves show "—": they arrive after the run ends. The trace's last line reads "104 weeks. The
    invariants were asserted at the end of every week."

**Points to check:** the trace cannot show why a move was chosen, because the renderer does not print the selection
rationale the log holds (`M16.A.3`); `M16.C.2` does not ask for it. A reader who wants the "why" needs another view.

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

### 4. `M11.F` — Framing that travels with any output

**What it requires.** No output is presented as carrying Bowen's authority (`M11.F.1`); a claim that the family level
informs a societal level is stated as analogy, not derivation (`M11.F.2`); the objection to the model's foundation is
stated accurately, with the distinction that answers it (`M11.F.3`, `M11.F.3a`); no claim to reproduce measured
differentiation (`M11.F.4`); no detector, readout or modulation is one-sided (`M11.F.6`); the model has no opinion on
trauma (`M11.F.7`); societal readouts do not inherit an editorial symptom list (`M11.F.8`); the model is never
presented as speaking about a real family (`M11.F.9`) and never given the name `M11.F.10` restricts without that requirement's framing
(`M11.F.10`).

**Explainer:** §14 (a note on framing), §1.1 (the name `M11.F.10` restricts, and its framing), §17 (what the model can be asked).

**Config:** none.

**Criteria:** none in `M11.C`. Condition V7 was signed off by the owner on 2026-10-17
(`docs/REVIEW_version_1_outputs.md`); `tests/test_version_1_outputs.py` checks that each output keeps its framing.

**Coverage:** 11 IDs: 0 done, 2 partial, 9 not done. No test names: `M11.F.1`, `M11.F.2`, `M11.F.3`, `M11.F.3a`, `M11.F.5`, `M11.F.6`, `M11.F.4`, `M11.F.7`, `M11.F.8`.

**Where to see it in the trace:**
- The framing block, trace lines 3–10, including "It does not concern any real family or person, and nothing here
  is advice (`M11.F.9`)."
- The diagram's page note begins "Simulated run — not a real family (spec M11.F.9)."

**Points to check:** V7 is already signed off; this entry is for anything the full review changes.

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

## Part 2 — One week of the run

### 5. `M3` — Clocks, ordering and determinism

**What it requires.** The fast tick is one week (`M3.A.1`); the slow tick is one year and fires every 52 fast ticks
(`M3.B.1`). Delivery latency is a property of the tie, a whole number of weeks, at least one (`M3.C.1`, `M3.C.2`).
Each week runs in a fixed order: standing load, deliver, perceive, appraise, recompute involvement, recompute active
triangles, select, act, consolidate (`M3.D.1`); the standing load comes before delivery (`M3.D.2`). The engine is a
deterministic function of seed, config and scenario, with counter-based draws keyed to the event they decide, so two
arms face the same chance events (`M3.D.4`–`M3.D.4c`); same seed, byte-identical logs (`M3.D.5`); no language model in
the decision path (`M3.D.6`). Activation and visibility are separate, named components, and every person selecting
every week is the declared `[I]` regime (`M3.E.1`, `M3.E.2`).

**Explainer:** §8 (clocks, latencies and windows), §18.1 (random draws and identity), §18.2 (order, activation and
visibility).

**Config:** `constants.md` (`slow_tick_fast_ticks`); `family_phase_c.md` (each tie's latency).

**Criteria:** none in `M11.C`. The engineering tests are `M11.D.5` (same seed, same log), `M11.D.15` (placebo arm) and
`M11.D.16` (order does not decide the outcome), in the suite.

**Coverage:** 15 IDs: 15 done, 0 partial, 0 not done.

**Where to see it in the trace:**
- Week 0, Sofia OVERFUNCTION → Ravi; witnesses Marta, Nadia, Pia (trace line 34). What it did: "arrives week 2; anxiety Marta +0.8 (witness), Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +2.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.5, Sofia +0.5; Sofia takes 2.0 of functioning from the other"
  - Ravi–Sofia's latency is 2, so Sofia's act of week 0 arrives in week 2.
- Week 1, system row (trace line 35): "Dr Halim–Marta–Ravi triangle active"
  - The first week's acts set the triangles that step 6 reads in week 1.
- Week 51, system row (trace line 482): "slow tick (yearly): 10 ties wore (conductance); every dyad a year older"
  - The first slow tick, at the end of week 51; the second is at week 103 (trace line 928). The header names the
    activation and visibility components: "synchronous_activation v1 (synchronous)" and "household_conductance_visibility
    v1" (trace lines 18–19).

**Points to check:** `TODO.md` records that `canonical_state` leaves out `state.stressors`, so the determinism tests
cannot see a stressor still weighing after the last compared tick (listed under "Open owner questions").

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

### 6. `M1.F` — The event

**What it requires.** Every event carries its sender, targets, witnesses, kind, intensity, timestamp, duration,
whether it is exogenous, the sender's position, its route and its fidelity (`M1.F.1`), and the channel that selected
it (`M1.F.1a`). Witnesses are computed by the visibility component from tie and household state, never chosen by the
sender (`M1.F.1b`). The sender's position can change the sign of an effect (`M1.F.2`); the route changes the gain and
fidelity falls with each private hop (`M1.F.3`, `M1.F.4`). Witnesses appraise events not addressed to them, and that
witnessing is the source of chronic anxiety (`M1.F.5`). Stressors come as spells with a start and a duration, never as
per-week draws (`M1.F.6`); endogenous events are never sampled from incidence data (`M1.F.7`); simultaneous events are
batched so order within a week cannot decide the outcome (`M1.F.8`). `binder_unavailable` is a second event type: a
binder becomes unavailable and what it held is released (`M1.F.9`).

**Explainer:** §2.6 (the event), §9.1–§9.4 (the record, illness as output, routing, the witness path).

**Config:** `event_kinds.md` (every event kind and its mechanism), `nodal_events.md` (which kinds are nodal),
`criteria.md` (the declared spell).

**Criteria:**
- `M11.C.31` — removing a binder raises anxiety with no stressor, by what it held. *Check; Phase D part I record; verdict PASS:* **pass**. Evidence: `budget_rise` +24 ± 1.2.

**Coverage:** 12 IDs: 8 done, 3 partial, 1 not done. No test names: `M1.F.4a`.

**Where to see it in the trace:**
- Week 0, Marta UNDERFUNCTION → Pia; witnesses Nadia, Ravi (trace line 30). What it did: "arrives week 1; anxiety Marta +0.5, Nadia +0.6 (witness), Pia -0.1, Ravi +1.3 (witness); Pia takes 2.0 of functioning from the other"
  - Nadia and Ravi, who share the household, are the witnesses: the visibility component computed them (`M1.F.1b`);
    the sender did not choose them.
- Week 4, JOB_LOSS from outside → Ravi; witnesses Marta, Nadia, Pia (trace line 64). What it did: "a 1-week spell, one week; the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +1.6 (witness), Nadia +1.5 (witness), Pia +1.2 (witness), Ravi +3.7"
  - An exogenous spell: one week, from the declared schedule.
- Week 5, Ravi SYMPTOM_ONSET → Dr Halim, Marta, Nadia, Pia, Sofia (trace line 84). What it did: "arrives week 6, 7; the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); symptom onset, physical channel; anxiety Dr Halim +0.5, Marta +2.6, Nadia +2.4, Pia +2.0; anxiety Sofia +1.3"
  - An endogenous event. It reaches Sofia a week later than the others, by the Ravi–Sofia latency.

**Points to check:** no `binder_unavailable` event occurs in this run; `M11.C.31` tests it in its own arm.

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

### 7. `M4.A`–`M4.C` — Standing load, perception and appraisal

**What it requires.** Each week every person takes a load from every tie, whether or not anything happened on it, as
the "too little" side of the appraisal function (`M4.A.1`, `M4.C.1c`), before any event is delivered (`M6.I.8`). A
`TRIGGER` can spike it with no contact (`M4.A.2`); from revision 32, a nodal event that reaches a person spikes each of
their cut-off ties the same way (`M4.C.1c`). A second, self-generated load derives from `basic_level` (`M4.A.5`). A
person perceives only its own state, its beliefs and the events delivered to it (`M4.B.1`, `M4.B.2`), and the policy
knows no consequences beyond its short horizon (`M4.B.3`). Appraisal: anxiety on a tie rises with the deviation from
the person's felt contact optimum, in either direction (`M4.C.1`); level sets the steepness and the band tolerated
(`M4.C.1a`); anxiety moves the optimum toward closeness (`M4.C.1b`); above a threshold, content is defended against
rather than absorbed (`M4.C.2`); a witness appraises from its own position (`M4.C.9`); contact with a calmer person
lowers anxiety (`M4.C.10`). Removing impingement is immediate and the "too little" side builds over time, so `CUTOFF`
relieves at once and costs later (`M4.C.1c`).

**Explainer:** §7.2 (the standing load), §3.4 (acute anxiety), §19.2 (the drives and the appraisal), §19.5 (calmer
contact).

**Config:** `constants.md` (the appraisal constants, `nodal_spike_intensity`), `nodal_events.md`.

**Criteria:**
- `M11.C.4` — `CUTOFF` relieves the actor now and raises family anxiety at the next nodal event. *Premise; Phase C record; verdict PASS:* **pass**. Evidence: `actor_relief_now` -2.81 ± 0.39; `family_anxiety_at_nodal` +7.93 ± 1.3; proved by `cutoff-later-costs-removed` red (Phase C mutation record).
- `M11.C.35` — a witness appraises from its own position. *Check; Phase C record; verdict PASS:* **pass**. Evidence: `witness_appraisal` +0.485 ± 0.028; proved by `witness-position-blind` red (Phase C mutation record), `witness-reach-inverted` reversed (Phase C mutation record).
- `M11.C.41[light: lower level]` — lower level exacerbates the pattern, under light stress. *Mixed; Phase C record; verdict PASS:* **pass**. Evidence: `mean_acute` +99.2 ± 11; `reactive_over_chance` +0.0645 ± 0.0031; proved by `steepness-level-independent` red (Phase C mutation record), `mixing-weight-level-independent` red (Phase C mutation record), `level-blind` red (Phase C mutation record), `steepness-inverted` red (Phase C mutation record), `mixing-weight-inverted` red (Phase C mutation record).
- `M11.C.41[heavy: lower level]` — lower level exacerbates the pattern, under heavy stress. *Mixed; Phase C record; verdict PASS:* **pass**. Evidence: `mean_acute` +90.3 ± 9.9; `reactive_over_chance` +0.0645 ± 0.0028; proved by `mixing-weight-level-independent` red (Phase C mutation record), `level-blind` red (Phase C mutation record), `steepness-inverted` red (Phase C mutation record), `mixing-weight-inverted` red (Phase C mutation record).
- `M11.C.41[higher level: heavier stress]` — heavier stress exacerbates the pattern, at the higher level. *Mixed; Phase C record; verdict FAIL:* **fail accepted as a finding**. Evidence: `mean_acute` +14.4 ± 6.3; `reactive_over_chance` +0.00122 ± 0.0023. Disposition: Heavier stress raises mean acute anxiety but not the reactive choice beyond the legal set's reactive share. The owner chose the composite reading (2026-10-17): the mixing weight reads functional level, not anxiety, so heavier stress reaches the reactive share only through the stress response's bounded fall in functioning, and no single rule writes the rise; `M11.5` is amended. *(owner 2026-10-17)*
- `M11.C.41[lower level: heavier stress]` — heavier stress exacerbates the pattern, at the lower level. *Mixed; Phase C record; verdict FAIL:* **fail accepted as a finding**. Evidence: `mean_acute` +5.58 ± 13; `reactive_over_chance` +0.00114 ± 0.0022. Disposition: As the higher-level cell: acute anxiety rises, the reactive choice beyond chance does not. The same reading (the owner, 2026-10-17). *(owner 2026-10-17)*
- `M11.C.43` — learned reliance on a calming tie raises the cost of losing it. *Composite; Phase D part I record; verdict FAIL:* **fail accepted as a finding**. Evidence: `loss_rise` -0.105 ± 1.1. Disposition: After learned reliance, losing a calming member costs no more (spec row, revision 27). *(owner 2026-10-17)*

**Coverage:** 24 IDs: 21 done, 0 partial, 3 not done. No test names: `M4.B.2a`, `M4.C.3`, `M4.C.8a`.

**Where to see it in the trace:**
- Week 4, JOB_LOSS from outside → Ravi; witnesses Marta, Nadia, Pia (trace line 64). What it did: "a 1-week spell, one week; the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +1.6 (witness), Nadia +1.5 (witness), Pia +1.2 (witness), Ravi +3.7"
  - The revision 32 spike: the job loss reaches Marta and Nadia, so their cut tie's standing load spikes.
- Week 12, Ana CUTOFF → Marta; witnesses Nadia, Pia, Ravi (trace line 140). What it did: "arrives week 13; anxiety Ana +0.3, Marta +1.6, Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +1.5 (witness); calmer sender takes some anxiety: Ana +1.1, Marta -1.1; Ana rejects the other on Ana–Marta (rejection +0.61; the tie stays open); impingement removed at once, anxiety Ana -4.4"
  - `CUTOFF`'s immediate relief to Ana. The later cost is not printed line by line (the standing load is one of the
    terms the header says run every week).
- Week 0, Dr Halim STAY-IN-CONTACT → Marta; witnesses Nadia, Pia, Ravi (trace line 29). What it did: "arrives week 1; anxiety Dr Halim +0.1, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Dr Halim +0.3, Marta -0.3"
  - Contact with a calmer sender lowers Marta's anxiety (`M4.C.10`).

**Points to check:** 60 rows carry the nodal-event spike. Its narrowing to ties where rejection is held (so a retired
coach's or a removed member's tie is not spiked) is Claude's recommendation, kept by the owner 2026-10-19 (open questions, item 5).

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

### 8. `M4.D`–`M4.G` — Selection, action and consolidation

**What it requires.** Each person resolves exactly one outcome a week, by softmax: an emitted move or a `WITHHOLD`,
which is computed, not emitted, and still changes the tie (`M4.D.1`, `M4.D.1b`). The legal set is formed first
(`M4.D.1e`); tie-break and fallback are declared and flagged (`M4.D.1f`). Selection runs over two channels with
different objectives, automatic and self-directed, mixed by a weight that reads functional level (`M4.D.1a`). Focus
steers whom anxious acts land on (`M4.D.1g`); an anxious act lands with more force (`M4.D.1h`); rising anxiety slides
selection down a complexity ordering (`M4.D.3`, `M4.D.3a`); the default is the fused default, not an absence of a move
(`M4.D.5`). The automatic channel learns from the actor's felt change in anxiety over a short declared horizon, plus
others' relief weighted by the actor's tie to them and what a witnessed cutoff led to (`M4.D.6`, `M4.D.6a`,
`M4.D.6e`, `M4.D.6f`); the self-directed channel, `basic_level` and `systems_perspective` are never reinforced
(`M4.D.6d`). The selected move becomes an event whose witnesses come from the visibility component (`M4.E.1`,
`M4.E.1a`). Consolidation hardens ties, so three withdrawals in a row read as a distant tie (`M4.G.1`), and every
invariant is asserted at the end of every week (`M4.G.2`).

**Explainer:** §5.2 (the policy, two channels), §18.4 (the policy's bookkeeping), §19.3 (learning), §19.4 (acts and
patterns).

**Config:** `policy.md` (tie-break and fallback), `perspective_steering.md`, `stress_response.md`, `event_kinds.md`,
`constants.md` (`policy_temperature`, `learning_rate`, `focus_steer_gain`, `force_anxiety_gain`).

**Criteria:**
- `M11.C.16` — repertoire concentration is a result, and it depends on level. *Composite; Phase C record; verdict PASS:* **pass**. Evidence: `repertoire_entropy` -0.0314 ± 0.0061; proved by `level-blind` red (Phase C mutation record), `availability-level-inverted` red (Phase C mutation record), `availability-and-learner-removed` red (Phase C mutation record), `c16-grounds-removed` red (Phase C mutation record).
- `M11.C.34` — the system's reaction to a symptom discriminates level. *Mixed; Phase D part I record; verdict FAIL:* **fail accepted as a finding**. Evidence: `automatic_reaction` +0.0867 ± 0.13. Disposition: The others' automatic acts toward the bearer after an onset, less a no-onset run, do not differ by level, and no selection reads `M1.A.19`'s detectors, so its declared mutant cannot bite (spec row, revision 27). The owner chose the composite reading (2026-10-17): the readout is the reaction toward the bearer, which `M11.5` calls learned. *(owner 2026-10-17)*
- `M11.C.39` — a pattern learned under a spell outlasts it. *Composite; Phase D part I record; verdict PASS:* **pass**. Evidence: `pattern_after_spell` +6.63 ± 1.3; proved by `learning-off-in-both-arms` red (Phase D mutation record).
- `M11.C.40` — reinforcement narrows the automatic channel only. *Premise; Phase D part I record; verdict PASS:* **pass**. Evidence: `automatic_entropy_fall` +0.0564 ± 0.0077.

**Coverage:** 32 IDs: 23 done, 2 partial, 7 not done. No test names: `M4.D.1c`, `M4.D.5`, `M4.D.5a`, `M4.D.5b`, `M4.D.5c`, `M4.D.5d`, `M4.D.6c`.

**Where to see it in the trace:**
- Week 0, Ana WITHHOLD → Bruno (trace line 27). What it did: "held back CONFLICT: computed, not emitted; attention on the tie rises"
  - A `WITHHOLD`: Ana's automatic `CONFLICT` toward Bruno is computed and held back.
- Week 27, system row (trace line 278): "Pia–Ravi tie now distant"
  - Consolidation (`M4.G.1`): the Pia–Ravi tie now reads as distant.
- Week 7, Bruno CUTOFF → Ana (trace line 96). What it did: "arrives week 8; anxiety Ana +0.9, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.07; the tie stays open)"
  - Bruno sends `CUTOFF` to Ana in 20 of the 104 weeks (weeks 0, 1, 7, 8, 9, 11, 17, 22, 29, 31, 35, 36, 46, 55, 57, 64,
    65, 66, 72, 73).

**Points to check:** the trace cannot say whether a repeated act such as Bruno's is learned, because the learner's
values and the propensities are not printed (entry 3).

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

### 9. `M5.A`–`M5.C` — The repertoire and its gates

**What it requires.** The core repertoire is exactly nine moves: `PURSUE`, `DISTANCE`, `CONFLICT`, `OVERFUNCTION`,
`UNDERFUNCTION`, `TRIANGLE`, `CUTOFF`, `I-POSITION`, `STAY-IN-CONTACT` (`M5.A.1`). Each is one act on one week; the
pattern of the same name is a readout, never a move (`M5.A.1a`); nothing is added without a citation (`M5.A.2`). The
corpus adds `DETRIANGLE`, `PREVENT_ALIGNMENT`, `REDUCE_CUTOFF` (which lowers the actor's rejection, and whose passing
cost lands on a third party), the moves toward an external agent `SPLIT`, `FRAME_AMBIGUITY` and `DISPLACE`, and
`PROVOKE` (`M5.B.1`–`M5.B.6`); the external agent has a strict subset (`M5.B.5`). Gates are evaluated before selection
and can remove a move from the draw (`M5.C.1`), including the live-issue gate (`M5.C.1a`) and, from revision 25, the
marital-distance gate on moves toward a symptomatic child; `I-POSITION` is not gated on mid-band family anxiety
(`M5.C.2`).

**Explainer:** §5.1 (the nine), §5.3 (move gates), §5.7 (moves the nine do not cover), §5.8 (the external agent's
repertoire); the pattern readouts are defined in `docs/readouts_phase_c.md`.

**Config:** `event_kinds.md`, `pattern_acts.md` (which acts each pattern readout counts), `perspective_steering.md`,
`constants.md` (`marital_distance_fraction`).

**Criteria:**
- `M11.C.14[firmness]` — a management technique has no independent effect while marital distance is high (firmness). *Premise; Phase D part I record; verdict PASS:* **pass**. Evidence: `child_symptom_load` +10.5 ± 3.2.
- `M11.C.14[talking it out]` — the same (talking it out). *Premise; Phase D part I record; verdict PASS:* **pass**. Evidence: `child_symptom_load` -3.6 ± 2.3.

**Coverage:** 13 IDs: 4 done, 4 partial, 5 not done. No test names: `M5.A.2`, `M5.B.3a`, `M5.B.5`, `M5.C.1a`, `M5.C.2`.

**Where to see it in the trace:**
- Week 73, Ravi REDUCE_CUTOFF → Pia; witnesses Marta, Nadia (trace line 665). What it did: "arrives week 74; anxiety Marta +2.5 (witness), Nadia +0.8 (witness), Pia +0.6, Ravi +0.4; Ravi's rejection -0.25; bound anxiety released to the third party: Marta +226.4"
  - The anxiety released to Marta, 226.4, equals the total the trace shows bound into Pia–Ravi by `DISTANCE` in weeks
    0–72 (the "bound into Pia–Ravi" figures sum to 226.4).
- Week 17, Marta SPLIT → Dr Halim; witnesses Nadia, Pia, Ravi (trace line 188). What it did: "arrives week 18; anxiety Dr Halim +0.2, Marta +0.3, Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +2.2 (witness)"
  - A family move toward the external agent.
- Week 32, Nadia DETRIANGLE → Ravi; witnesses Marta, Pia (trace line 319). What it did: "arrives week 33; anxiety Marta +2.1 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +1.7"

**Points to check:**
- In this run the counts of each outcome, Dr Halim's included, are: `DISTANCE` 118, `CONFLICT` 103, `WITHHOLD` 89,
  `CUTOFF` 88, `UNDERFUNCTION` 87, `OVERFUNCTION` 81, `PURSUE` 68, `TRIANGLE` 50, `STAY-IN-CONTACT` 27, `SPLIT` 8,
  `FRAME_AMBIGUITY` 6, `DISPLACE` 4, `DETRIANGLE` 4, `REDUCE_CUTOFF` 3, `PROVOKE` 3, `I-POSITION` 2.
  `PREVENT_ALIGNMENT` does not occur.
- `M5.B.3` says the passing cost lands on a third party; it gives no size. Is the whole store bound over 73 weeks
  landing on one person in one week the intended form? (A second instance: week 89, trace line 806, releases 23.1 to
  Ravi, the amount bound into Marta–Nadia at week 47.)

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

### 10. `M5.D`–`M5.F` — `I-POSITION`, the reaction ladder and counterfeit moves

**What it requires.** `I-POSITION` is a state machine, not a one-week move: `PREPARE` → `DEFINE` → `OPPOSITION` →
(`ABORT` | `HOLD`) → `PEAK` → `RESOLVE` → `FOLLOW_UP` (`M5.D.1`, `M5.D.2`). `PREPARE` can fail (`M5.D.2a`); aborting at
the first opposition is the usual outcome (`M5.D.3`); the mover's freedom from anger admits the sequence to `PEAK`, and
the mover's anger degrades the move to the assertion form (`M5.D.4`, `M5.D.4a`). Only a sequence that reaches
`FOLLOW_UP` counts as completed; it raises functional level and the triangle's floor, and never writes `basic_level`
(`M5.D.7`); success usually follows several failures (`M5.D.8`). The system's reaction ("you are wrong", "change back",
"or else") is learned by the others from the life-energy debit the move imposes, not written (`M5.E.0`, `M5.E.1`,
`M5.E.7`); no reaction means the move did not land (`M5.E.3`); success leaves the tie more solid (`M5.E.8`). A move's
effect is multiplied by the actor's hidden two-axis outside-ness and can change sign, gradually (`M5.F.1`, `M5.F.2`,
`M5.F.2c`); asserting a differentiated state is evidence against it (`M5.F.2a`); the detector reports which axis failed
(`M5.F.2b`); an `I-POSITION` chosen with little systems perspective executes as the assertion form (`M5.F.4`).

**Explainer:** §5.4 (`I-POSITION` is a state machine), §5.5 (the reaction ladder), §5.6 (counterfeit moves), §3.6
(outside-ness).

**Config:** `constants.md` (`hold_gain`, `defence_threshold` and the sequence's other constants),
`criterion_roles.md` (`M11.C.5`'s mover, target and third person).

**Criteria:**
- `M11.C.5` — the system pushes back on an `I-POSITION`: a change-back reaction, with symptom in a third person. *Composite; Phase C record; verdict FAIL:* **fail accepted as a finding**. Evidence: `target_reaction` +2.98 ± 23; `third_person_symptom_load` +8.79 ± 13. Disposition: No settled change-back reaction over 60 weeks. The ladder is learned (`M5.E.0`); the model's later consequences of one act wash out within weeks to months (`docs/phase_c_completion_report.md` §13). *(owner 2026-10-17)*
- `M11.C.19` — the two counterfeits are told apart by which axis failed. *Check; Phase C record; verdict PASS:* **pass**. Evidence: `outward_failed` +1 ± 0; `inward_failed` -1 ± 0; proved by `axes-collapsed` red (Phase C mutation record), `axes-swapped` reversed (Phase C mutation record).
- `M11.C.29` — a genuine `I-POSITION` and distance-in-disguise differ by the third person's time course. *Mixed; Phase C record; verdict FAIL:* **fail accepted as a finding**. Evidence: `third_symptom_weeks` -1.24 ± 2.9. Disposition: Mixed since revision 32: the budget limb (`M6.I.1`) is proved by `tests/bowen/test_sinks.py::test_m6i1_only_a_completed_exchange_reduces_the_budget` and reported; the third person's time course is composite. **Its precondition is not met:** no genuine `I-POSITION` completes in the arm (0 in 100 seeds; 37 run as the assertion form because the mover is angry at the act, 58 fail in PREPARE on the mover's own excess, 20.7 against `defence_threshold` 15, with the declared spell landing on the mover the same week; ad hoc, plan §4.12). So the failure says nothing about time course; what it does show about the model is that a lone genuine `I-POSITION` by a level-40 mover under this load does not complete. The owner may prefer a redesigned arm to an acceptance. *(owner 2026-10-17)*
- `M11.C.32` — the mover's anger degrades the `I-POSITION`. *Premise; Phase C record; verdict PASS:* **pass**. Evidence: `assertion_form` +1.42 ± 0.26; `reached_peak` -0.25 ± 0.098; proved by `anger-gate-inverted` reversed (Phase C mutation record), `anger-gate-removed` red (Phase C mutation record).

**Coverage:** 32 IDs: 19 done, 3 partial, 10 not done. No test names: `M5.D.7b`, `M5.E.0`, `M5.E.1`, `M5.E.2`, `M5.E.4`, `M5.E.5`, `M5.E.6`, `M5.E.7a`, `M5.E.9`, `M5.F.2c`.

**Where to see it in the trace:**
- Week 43, Sofia I-POSITION → Ravi; witnesses Marta, Nadia, Pia (trace line 412). What it did: "arrives week 45; anxiety Marta +1.2 (witness), Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +4.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; assertion form (no systems perspective, or angry): extra impingement; claiming the position raises the sender's outward axis"
  - Sofia's first `I-POSITION` executes as the assertion form.
- Week 63, Sofia I-POSITION → Ravi; witnesses Marta, Nadia, Pia (trace line 581). What it did: "arrives week 65; anxiety Marta +1.1 (witness), Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +2.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; assertion form (no systems perspective, or angry): extra impingement; claiming the position raises the sender's outward axis"
  - Her second, twenty weeks later, the same.

**Points to check:** no row in the trace reports an `I-POSITION` sequence stage (the renderer has a row for each stage,
`src/bowen/render/trace.py`), and no row reports a completed differentiating exchange. This fits the open item that a
genuine `I-POSITION` at level 40 almost never completes (open questions). Sofia's basic level is 41.

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

## Part 3 — The objects the moves change

### 11. `M1.A` — The person

**What it requires.** Two state variables from which reactive behaviour is derived, a basic level and an anxiety
level; reactivity is never stored (`M1.A.1`). `basic_level` runs 0–100 as a continuum, with no band edge used as a
branch point (`M1.A.2`, `M1.A.3`, `M1.A.3d`). It is derived by an estimator over functional-level history and never
written by a move (`M1.A.4`, `M1.A.4a`, `M1.A.4k`). `functional_level` is distinct: basic level plus a pseudo-self
swing, moved up or down by what the person does under stress (`M1.A.5`, `M1.A.5a`, `M1.A.5e`); symptom thresholds read
it, not basic level (`M1.A.6`). Programmed reactivity is fixed in childhood from witnessed history, and acute anxiety
decays toward the chronic floor (`M1.A.7`, `M1.A.8`). Outside-ness has two axes, outward and inward impingement
(`M1.A.9`, `M1.A.9a`); life energy is zero-sum between relationships and goals (`M1.A.10`). Three symptom channels,
physical, mental and social, with a constitutional prior that the family's focus can move, at a declared strength
(`M1.A.11`–`M1.A.11c`). The rest: involvement weight, structural importance, sibling position as birth-order data,
financial dependence as a gate, beliefs, role, systems perspective, the reactive detectors, stable identifiers, sex and
household (`M1.A.12`–`M1.A.22`).

**Explainer:** §2.1 (`Person`), §3.1–§3.13 (every `Person` field).

**Config:** `family_phase_c.md` (each person's levels, chronic anxiety and `channel_prior`), `constants.md`
(`constitution_strength`, `focus_rate`), `stress_response.md`.

**Criteria:**
- `M11.C.1` — a lower basic level reaches symptom threshold sooner under the same stressors. *Premise; Phase C record; verdict PASS:* **pass**. Evidence: `time_to_threshold` -2.58 ± 0.24; proved by `level-blind` red (Phase C mutation record).
- `M11.C.38[-0 vs -5]` — basic level orders time to threshold (levels lowered by 0 and 5). *Premise; Phase C record; verdict PASS:* **pass**. Evidence: `time_to_threshold` -1.58 ± 0.28; proved by `level-blind` red (Phase C mutation record).
- `M11.C.38[-5 vs -10]` — the same (lowered by 5 and 10). *Premise; Phase C record; verdict PASS:* **pass**. Evidence: `time_to_threshold` -1.09 ± 0.18; proved by `level-blind` red (Phase C mutation record), `steepness-inverted` red (Phase C mutation record).
- `M11.C.38[-10 vs -15]` — the same (lowered by 10 and 15). *Premise; Phase C record; verdict PASS:* **pass**. Evidence: `time_to_threshold` -1 ± 0.12; proved by `level-blind` red (Phase C mutation record).

**Coverage:** 57 IDs: 22 done, 1 partial, 34 not done. No test names: `M1.A.0`, `M1.A.3`, `M1.A.3c`, `M1.A.3d`, `M1.A.3b`, `M1.A.3a`, `M1.A.4`, `M1.A.4a`, `M1.A.4c`, `M1.A.4e`, `M1.A.4g`, `M1.A.4i`, `M1.A.4j`, `M1.A.4f`, and 20 more.

**Where to see it in the trace:**
- Week 5, Ravi SYMPTOM_ONSET → Dr Halim, Marta, Nadia, Pia, Sofia (trace line 84). What it did: "arrives week 6, 7; the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); symptom onset, physical channel; anxiety Dr Halim +0.5, Marta +2.6, Nadia +2.4, Pia +2.0; anxiety Sofia +1.3"
  - Ravi's `channel_prior` is physical.
- Week 16, Nadia SYMPTOM_ONSET → Marta, Ravi; witnesses Pia (trace line 183). What it did: "arrives week 17; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); symptom onset, mental channel; anxiety Marta +3.7, Pia +0.8 (witness), Ravi +4.2"
  - Nadia's `channel_prior` is mental.
- Week 63, Sofia I-POSITION → Ravi; witnesses Marta, Nadia, Pia (trace line 581). What it did: "arrives week 65; anxiety Marta +1.1 (witness), Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +2.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; assertion form (no systems perspective, or angry): extra impingement; claiming the position raises the sender's outward axis"
  - "claiming the position raises the sender's outward axis": one of outside-ness's two axes (`M1.A.9a`).

**Points to check:** all ten symptom onsets in this run are in the person's declared `channel_prior` (Ravi twice,
Marta, Ana and Dr Halim physical; Pia twice and Bruno social; Nadia twice mental). Whether the family's focus ever
moves a channel (`M1.A.11c`) cannot be read from one run; `constitution_strength` (0.5) and `focus_rate` (0.05) are
both `[I]`.

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

### 12. `M1.B` — The relationship

**What it requires.** A tie connects exactly two people and holds their coupling state (`M1.B.1`); conductance is
undirected and not a function of contact frequency or time apart (`M1.B.2`); bond energy tells apart cut off, distant,
resolved low-contact and open conflict (`M1.B.3`). A cutoff is rejection held by the cutter, not severing: the tie
stays open, the other's moves are refused while rejection is above a threshold, the sender bears the refusal, and
rejection fades each week unless renewed (`M1.B.3c`). Cutoffs are reactive or principled (`M1.B.3d`); felt danger is
read from the intrusion a person receives, and a cutoff toward danger is principled in proportion (`M1.B.3e`). Bond
energy barely decays (`M1.B.4`). The functioning balance is directed and bistable, flips on a relative comparison, and
costs more to reverse one way than the other (`M1.B.5`–`M1.B.7`); from revision 21, a person holding a position stops
the exchange. Investment is directed and blind to valence, a parent's focus can settle on a child's weakness, and focus
locks in (`M1.B.8`–`M1.B.8b`). Areas of joint activity, the taboo set, latency, dyad age and a stable tie identifier
(`M1.B.9`–`M1.B.13`).

**Explainer:** §2.2 (the tie), §4.1–§4.9 (every tie field), §8.2 (cutoff and focus), §8.3 (origin cutoff, danger,
force and focus that locks in).

**Config:** `family_phase_c.md` (the ties), `constants.md` (the `rejection_*`, `danger_*` and `focus_*` constants,
`practice_window`), `fixtures/danger.md`.

**Criteria:**
- `M11.C.25` — pole assignment is independent of sex. *Premise; Phase C record; verdict PASS:* **pass**. Evidence: `first_dominant` +0 ± 0; proved by `sex-term-in-pole` red (Phase C mutation record).
- `M11.C.47` — low level and stress breed felt danger, and the cutoff toward it is principled. *Premise; Phase D part II record; verdict PASS:* **pass**. Evidence: `felt_danger` +0.0652 ± 0.021; `principled_cutoff` +0.0332 ± 0.014.

**Coverage:** 19 IDs: 13 done, 1 partial, 5 not done. No test names: `M1.B.9`, `M1.B.10`, `M1.B.10a`, `M1.B.11`, `M1.B.12`.

**Where to see it in the trace:**
- Week 1, Ana OVERFUNCTION → Bruno (trace line 41). What it did: "refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.3"
  - Bruno holds rejection on the tie, so Ana's move is refused and the cost is hers.
- Week 20, system row (trace line 217): "Ana–Marta: rejection has faded; the tie is no longer cut off (ordinary)"
  - Ana's rejection of Marta, raised by her `CUTOFF` in week 12 (trace line 140), has faded.
- Week 12, Ravi OVERFUNCTION → Pia; witnesses Marta, Nadia (trace line 145). What it did: "arrives week 13; anxiety Marta +5.0 (witness), Nadia +1.2 (witness), Pia +4.2, Ravi +1.3; Pia holds a position: no functioning changes hands"
  - Pia held back toward Ravi the same week (trace line 144); holding a position is read from such behaviour
    (`M1.B.5`, revision 21).

**Points to check:** on Ana–Bruno both sides hold rejection for much of the run: 74 of Bruno's 95 emitted moves are
refused by Ana, and 23 of Ana's 43 rows toward Bruno are refused by him.

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

### 13. `M1.C` — The triangle

**What it requires.** A triangle holds three members, the inside pair, the outsider and its bound anxiety. A `TRIANGLE`
act's target is the third person it recruits and leaves outside; the inside pair is the sender and the member the
sender is most strained with (`M1.C.1`). The outside position's value inverts with load, as a result of the appraisal
and learning, not as a rule (`M1.C.2`). The stored topology is kept apart from the active set; whether a triangle is
active is a readout of recent `TRIANGLE` acts, with no calm-system threshold written (`M1.C.3`). A triangle's routing
capacity depends on its members' functional level (`M1.C.3a`); the coach does not act on triangles as a mechanism of
change (`M1.C.3b`). Activation memory, a permanent intensity floor lowered by a held `I-POSITION`, sibling conflict
instantiating the parental triangle, and a stable identifier (`M1.C.4`–`M1.C.7`).

**Explainer:** §2.3 (`Triangle`), §6.1 (triangle fields), §13.1 (three live positions).

**Config:** `constants.md` (`triangle_activity_window`), `pattern_acts.md`.

**Criteria:**
- `M11.C.3` — a `TRIANGLE` relieves its seeker and costs the third, the same week. *Premise; Phase C record; verdict PASS:* **pass**. Evidence: `seeker_anxiety` -1.31 ± 0.11; `third_anxiety` +1.14 ± 0.21; proved by `triangle-transfer-removed` red (Phase C mutation record), `seeker-relief-removed` red (Phase C mutation record), `triangle-roles-swapped` red (Phase C mutation record).
- `M11.C.27[stable,add_third]` — the twosome 2×2: adding a third destabilises a stable pair. *Composite; Phase C record; verdict FAIL:* **fail accepted as a finding**. Evidence: `pair_deviation` +0.0142 ± 0.034. Disposition: Adding a third barely moves the pair's contact in any week: a `TRIANGLE` changes anxiety, not contact (`docs/phase_c_completion_report.md` §3 item 5). *(owner 2026-10-17)*
- `M11.C.27[stable,remove_one]` — the twosome 2×2: removing one destabilises a stable pair. *Composite; Phase C record; verdict FAIL:* **fail accepted as a finding**. Evidence: `pair_deviation` -0.0681 ± 0.046. Disposition: Removing one steadies either couple for about four weeks and unsettles it from week 6; the stable couple steadies first, so the week the act lands reads the wrong sign and weeks 6–12 read the claimed one (reported readouts in the record). *(owner 2026-10-17)*
- `M11.C.27[unstable,add_third]` — the twosome 2×2: adding a third stabilises an unstable pair. *Composite; Phase C record; verdict FAIL:* **fail accepted as a finding**. Evidence: `pair_deviation` +0.00454 ± 0.032. Disposition: As for the stable couple: adding a third barely moves the pair's contact. *(owner 2026-10-17)*
- `M11.C.27[unstable,remove_one]` — the twosome 2×2: removing one stabilises an unstable pair. *Composite; Phase C record; verdict PASS:* **pass**. Evidence: `pair_deviation` -0.132 ± 0.07; proved by `one-sided-deviation` reversed (Phase C mutation record), `deviation-inverted` red (Phase C mutation record).
- `M11.C.42` — a triangle that relieved is reused. *Composite; Phase C record; verdict FAIL:* **fail accepted as a finding**. Evidence: `triangle_reuse` +0.05 ± 0.3. Disposition: A `TRIANGLE` relieves its seeker less than a `DISTANCE` (−1.29 against −4.97 one week after, at Phase C close), so it is not learned and not reused (`docs/phase_c_completion_report.md` §3 item 8, §13). *(owner 2026-10-17)*
- `M11.C.44` — position value inverts with load. *Composite; Phase C record; verdict FAIL:* **fail accepted as a finding**. Evidence: `outside_inside_ratio` +0.00101 ± 0.096. Disposition: No calm state near the floor: the calm arm is the triad's resting state, 8–9 points above the floor, and the spell adds about 2, so the outside position's value does not invert with load (Phase C close). It has moved between pass and fail on unrelated revisions with both mutants surviving, so a pass would not be evidence either (`TODO.md`). *(owner 2026-10-17)*
- `M11.C.45` — triangles are quiet when the system is calm. *Composite; Phase C record; verdict FAIL:* **fail accepted as a finding**. Evidence: `triangle_rate` +0.000722 ± 0.0051. Disposition: The causes of `M11.C.42` and `M11.C.44` together: triangling is not learned, and there is no calm state near the floor. *(owner 2026-10-17)*
- `M11.C.46` — a triangle relieves its seeker in proportion to the third's help. *Premise; Phase D part I record; verdict PASS:* **pass**. Evidence: `seeker_relief` +1.45 ± 0.11.

**Coverage:** 10 IDs: 4 done, 1 partial, 5 not done. No test names: `M1.C.2`, `M1.C.3b`, `M1.C.3c`, `M1.C.4`, `M1.C.6`.

**Where to see it in the trace:**
- Week 0, Nadia TRIANGLE → Marta; witnesses Pia, Ravi (trace line 31). What it did: "arrives week 1; anxiety Marta -0.3, Nadia +0.5, Pia +0.6 (witness), Ravi +1.3 (witness); Marta left outside: anxiety Marta +0.6, Nadia -0.4 (of which 0.2 the outsider's own)"
  - Nadia recruits Marta and leaves her outside; the week 1 system row below names the inside pair.
- Week 9, Ravi TRIANGLE → Pia; witnesses Marta, Nadia (trace line 117). What it did: "arrives week 10; anxiety Marta +4.4 (witness), Nadia +1.2 (witness), Pia +1.9, Ravi +1.2; Pia left outside: anxiety Pia +9.7, Ravi -6.4 (of which 3.2 the outsider's own)"
  - A larger transfer: Pia, left outside, takes 9.7.
- Week 1, system row (trace line 38): "Marta–Nadia–Ravi: inside pair Nadia–Ravi"

**Points to check:** in the diagram all three fixed triangles have Marta outside: Marta–Nadia–Ravi (blocks 11–26),
Dr Halim–Marta–Ravi (blocks 13–26) and Marta–Pia–Ravi (block 26).

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

### 14. `M1.D` — The family

**What it requires.** One undifferentiation budget and exactly three sinks: marital conflict, spouse dysfunction and
child projection (`M1.D.1`). Emotional distance runs outside the budget and is never a fourth sink (`M1.D.2`); distance
binds anxiety into the tie, where it persists, reads as the distance itself, and returns if distancing is prevented
(`M1.D.2a`). An overflow goes to conflict with the families of origin (`M1.D.3`). A leadership office whose vacancy
stalls the system (`M1.D.4`, `M1.D.4a`); a per-member tolerance for disturbing behaviour (`M1.D.5`); an access vector
that is not a gate (`M1.D.6`); ambient anxiety driven by three named societal dials (`M1.D.7`–`M1.D.7l`). The family is
not the boundary of the system: every adult has a family-of-origin tie (`M1.D.8`), and a parent's cutoff is an input to
a child's later intensity (`M1.D.8a`).

**Explainer:** §2.4 (`Family`), §6.2 (family fields), §6.3 (the access vector), §6.4 (the societal dials), §7 I1–I2
(the budget and distance).

**Config:** `family_phase_c.md` (`undifferentiation_budget` 90), `family_phase_d.md`, `constants.md` (the
`origin_cutoff_*` constants).

**Criteria:**
- `M11.C.13` — help relocates incidents; it does not reduce them. *Premise; no record; verdict not built:* **out of scope**. Evidence: needs incidents located in a community; the model has none. Disposition: It needs incidents located in a community, and the model has no community (version 1 definition §4 question 4). *(owner 2026-10-15)*
- `M11.C.28` — sink mobility is protective. *Composite; no record; verdict not built:* **out of scope**. Evidence: the sink allocation feeds no behaviour, so pinning its mobility can change no symptom. Disposition: The sink allocation feeds no behaviour, so pinning its mobility can change no symptom (§4 question 4). *(owner 2026-10-15)*
- The budget's one reduction, by a completed differentiating exchange, is `M11.C.29`'s budget limb (entry 10).

**Coverage:** 22 IDs: 5 done, 0 partial, 17 not done. No test names: `M1.D.4`, `M1.D.4a`, `M1.D.5`, `M1.D.6`, `M1.D.7`, `M1.D.7a`, `M1.D.7a1`, `M1.D.7b`, `M1.D.7c`, `M1.D.7d`, `M1.D.7i`, `M1.D.7j`, `M1.D.7k`, `M1.D.7l`, and 3 more.

**Where to see it in the trace:**
- Week 10, Marta DISTANCE → Pia; witnesses Nadia, Ravi (trace line 121). What it did: "arrives week 11; anxiety Marta +1.2, Nadia +0.9 (witness), Pia +2.7, Ravi +3.1 (witness); 26.6 of the sender's anxiety bound into Marta–Pia"
  - Distance binds the sender's anxiety into the tie; it is not destroyed.
- Week 2, Sofia DISTANCE → Ravi; witnesses Marta, Nadia, Pia (trace line 55). What it did: "arrives week 4; anxiety Marta +0.8 (witness), Nadia +0.8 (witness), Pia +0.6 (witness), Ravi +2.8, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; 1.2 of the sender's anxiety bound into Ravi–Sofia"

**Points to check:** the budget, the sinks and the dials are not printed in the trace. The coverage list above includes
the leadership office, the access vector and the societal dials.

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

### 15. `M1.E` — External agents (the coach)

**What it requires.** An external agent is a `Person` with `role = EXTERNAL`, a restricted repertoire and real ties
(`M1.E.1`), with a reactive state of its own (`M1.E.2`); its objective is to understand, not to help (`M1.E.2a`). It can
absorb responsibility and certify a defect (`M1.E.3`, `M1.E.4`); its proximity carries a burden transfer (`M1.E.5`);
its presence changes the configuration (`M1.E.6`). Systems perspective rises only on a landed contact, which needs
coincidence with the recipient's current state, not only the right content (`M1.E.7`, `M1.E.7c`, `M1.E.7d`). The coach
tie stays thin and accumulates no investment, and a coach seen as taking sides becomes a lender of pseudo-self
(`M1.E.7e`). The entry threshold is relative to the family (`M1.E.7f`); contact frequency has a low optimum
(`M1.E.8`).

**Explainer:** §2.5 (external agents), §5.8 (the external agent's repertoire), §10.2 (the coach's own state is a
detector).

**Config:** `family_phase_c.md` (Dr Halim: basic level 60, chronic anxiety 30, ties of conductance 0.3 and bond energy
10 to Ravi and Marta), `constants.md` (`session_interval_weeks`, the `coach_*` constants).

**Criteria:**
- `M11.C.17` — differentiation is earned in behaviour; a coach helps and is not required. *Composite; Phase D part I record; verdict PASS:* **pass**. Evidence: `coached_rise` +0.000888 ± 0.0013; proved by `landed-contact-changes-nothing` red (Phase D mutation record). Disposition: Failed before the nodal-event spike; the owner asked for a diagnosis ("Diagnose coaching"), which found no defect, the effect dying at the estimator's evidence test. With the spike (revision 32) it passes, weakly; a recorded mutant (landed contact changes nothing) must turn it red (V3). *(Claude's recommendation 2026-10-18, pending owner review)*

**Coverage:** 13 IDs: 5 done, 1 partial, 7 not done. No test names: `M1.E.2`, `M1.E.2a`, `M1.E.3`, `M1.E.4`, `M1.E.5`, `M1.E.7d`, `M1.E.7f`.

**Where to see it in the trace:**
- Week 80, Dr Halim TRIANGLE → Marta; witnesses Nadia, Pia, Ravi (trace line 725). What it did: "arrives week 81; anxiety Dr Halim +0.1, Nadia +0.9 (witness), Pia +0.8 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Dr Halim +0.7, Marta -0.7; Marta left outside: anxiety Dr Halim -0.3, Marta +0.4 (of which 0.1 the outsider's own); the contact lands with Marta: systems perspective +0.10"
  - The run's one landed contact.
- Week 88, Dr Halim OVERFUNCTION → Ravi; witnesses Marta, Nadia, Pia (trace line 796). What it did: "arrives week 89; anxiety Dr Halim +0.1, Marta +1.5 (witness), Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +0.4; calmer sender takes some anxiety: Dr Halim +0.7, Ravi -0.7; Dr Halim takes 2.1 of functioning from the other"
- Week 32, Dr Halim CUTOFF → Marta; witnesses Nadia, Pia, Ravi (trace line 317). What it did: "arrives week 33; anxiety Dr Halim +0.1, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +1.2 (witness); calmer sender takes some anxiety: Dr Halim +0.8, Marta -0.8; Dr Halim rejects the other on Dr Halim–Marta (rejection +0.51; the tie stays open)"

**Points to check:**
- Dr Halim's 13 session acts are 5 `STAY-IN-CONTACT`, 4 `TRIANGLE`, 2 `CUTOFF` and 2 `OVERFUNCTION`. `TODO.md`
  (revision 32, `M11.C.17`'s diagnosis) records that the coach's automatic-channel acts land with full effect, though
  `M1.E.7c`(3) reads non-participation as the coach's default.
- In week 88 and week 96 (trace line 862) Dr Halim's `OVERFUNCTION` moves functioning on the coach tie. Is that
  consistent with `M1.E.7e`? `TODO.md` records that a sided coach's lending of pseudo-self is not built.
- Dr Halim has a symptom onset at week 77 (trace line 704): the external agent carries symptom load as a member does.
  Is that intended?

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

### 16. `M8` — The live-position predicate

**What it requires.** `positions_live(group)` counts only occupants not fused into another member of the group
(`M8.1`). An external agent in the outside position counts (`M8.2`); one who has taken a side does not. Built at
revision 32: an external agent has taken a member's side for `triangle_activity_window` weeks after that member's
`SPLIT` toward it (`M8.3`). A present but emotionally inactive member does not count (`M8.4`). The same predicate
decides whether a group can avoid its issue, whether a third party is a witness or a new triangle, whether the coach is
outside, and whether a triangle is active (`M8.5`). A third party's effect is set by position, not by what it knows;
knowledge is a hazard on alignment, not a gate (`M8.6`–`M8.7`); announcing an act is free, announcing the programme
carries a penalty (`M8.8`).

**Explainer:** §13.1 (three live positions), §13.2 (alignment is the gate, not knowledge), §13.4.

**Config:** `constants.md` (`triangle_activity_window`).

**Criteria:**
- `M11.C.7[joint over one spouse]` — position, not skill: coaching both spouses together against one spouse. *Mixed; Phase D part I record; verdict PASS:* **pass**. Evidence: `deviation_fall` +0.0429 ± 0.052; proved by `landed-contact-changes-nothing` red (Phase D mutation record). Disposition: The owner accepted this pair's failure (2026-10-17). With the nodal-event spike (revision 32) it passes, at the edge (p 0.05); a recorded mutant (landed contact changes nothing) must turn it red (V3). *(Claude's recommendation 2026-10-18, pending owner review)*
- `M11.C.7[joint over sided]` — position, not skill: coaching both spouses together against a coach who has taken a side. *Mixed; Phase D part I record; verdict PASS:* **pass**. Evidence: `deviation_fall` +0.0519 ± 0.047; `landed_contacts` +10.2 ± 0.56; proved by `landed-contact-changes-nothing` red (Phase D mutation record), `coach-position-blind` red (Phase D mutation record). Disposition: The owner accepted this pair's failure (2026-10-17). With the nodal-event spike (revision 32) it passes; recorded mutants (the position-blind gate, landed contact changes nothing) must turn it red (V3). *(Claude's recommendation 2026-10-18, pending owner review)*
- `M11.C.7[one spouse over sided]` — position, not skill: coaching one spouse against a coach who has taken a side. *Mixed; Phase D part I record; verdict FAIL:* **fail accepted as a finding**. Evidence: `deviation_fall` +0.00906 ± 0.054; `landed_contacts` +9.72 ± 0.62. Disposition: The gate's limb holds (a sided coach's contact never lands); the deviation limb does not, for the reason given for the first pair. *(owner 2026-10-17)*

**Coverage:** 10 IDs: 4 done, 1 partial, 5 not done. No test names: `M8.6`, `M8.6a`, `M8.6b`, `M8.7`, `M8.8`.

**Where to see it in the trace:**
- Week 1, system row (trace line 36): "Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi"
  - In week 1, before any `SPLIT`, the coach is in a triangle's inside pair.
- Week 17, Marta SPLIT → Dr Halim; witnesses Nadia, Pia, Ravi (trace line 188). What it did: "arrives week 18; anxiety Dr Halim +0.2, Marta +0.3, Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +2.2 (witness)"
  - Marta's `SPLIT` toward the coach; under `M8.3` as built, the coach is on her side for `triangle_activity_window` weeks
    after it.

**Points to check:** `TODO.md` lists, adjacent and not fixed, that `recompute.py`'s triangle activity treats the coach
as an unfused occupant, though `M8.5` says the same predicate decides triangle activity. The diagram draws
Dr Halim–Marta–Ravi as a fixed triangle (blocks 13–26). Read beside `M1.C.3b`.

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

### 17. `M9` — The belief layer

**What it requires.** A family-level and per-person belief store that may differ from the truth (`M9.1`); a belief's
truth and its emotional function are independent (`M9.2`); institutional acts write the who-is-sick belief with
hysteresis that denial cannot reverse (`M9.3`); shock waves travel by several hops on the ordinary tie graph plus
belief (`M9.4`); readouts report truth and belief separately (`M9.5`). As built: each person holds a weight per member
for where the family's difficulty lies, moved by the moves they send, receive or witness, from which a readout outside
the engine names the active sink (`M9.6`); a witness's reaction to load between two others is scaled by how tense the
witness believes that tie is (`M9.7`); each person holds beliefs about ties it is not party to (`M9.8`).

**Explainer:** §9.5 (the belief layer), §3.13 (`beliefs`).

**Config:** `attribution.md` (where each act places the difficulty), `constants.md` (`attribution_rate`,
`belief_appraisal_gain`).

**Criteria:**
- `M11.C.26` — the active sink is recoverable from the spouses' beliefs. *Premise; Phase D part I record; verdict PASS:* **pass**. Evidence: `belief_to_sink` +0.68 ± 0.033.
- `M11.C.36` — appraisal reads belief, not truth. *Check; Phase D part I record; verdict PASS:* **pass**. Evidence: `witness_response` +3.07 ± 0.011.

**Coverage:** 8 IDs: 3 done, 0 partial, 5 not done. No test names: `M9.1`, `M9.2`, `M9.3`, `M9.4`, `M9.5`.

**Where to see it in the trace:**
- Week 11, Ravi CONFLICT → Marta; witnesses Nadia, Pia (trace line 136). What it did: "arrives week 12; anxiety Marta +7.8, Nadia +2.2 (witness), Pia +2.3 (witness), Ravi +1.1"
  - Under `M9.7` as built, Nadia's and Pia's witness figures include the scaling by their belief about Marta–Ravi; the
    trace prints the result, not the belief.

**Points to check:** attribution is one of the terms the header says run every week and are not listed, so no belief
value appears in the trace.

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

### 18. `M6` — Invariants

**What it requires.** Asserted at the end of every week; a violation raises, never warns (`M4.G.2`). `M6.I.1`: the
budget has three sinks and is conserved, reduced only by a completed differentiating exchange. `M6.I.2`: distance runs
outside the budget. `M6.I.3`: life energy is zero-sum per person. `M6.I.4`: dyadic exchange conserves pseudo-self.
`M6.I.5`: solid self is exempt from fusion. `M6.I.6`: every change to an anxiety stock is logged against a source, a
sink or a transfer in `M6.4`'s table, and every transfer balances to `M6.1`'s tolerance. `M6.I.7`: there is no exit
from the field. `M6.I.8`: the standing load runs before delivery. Disposition at death (`M6.3`) is in entry 21.

**Explainer:** §7 (I1–I7) and §7.2 (the standing load).

**Config:** `constants.md` (`invariant_tolerance`, 1e-9).

**Criteria:**
- `M11.C.21` — reciprocity inverts when the over-functioners' capacity falls. *Composite; Phase D part I record; verdict FAIL:* **fail accepted as a finding**. Evidence: `target_rise` -0.0906 ± 0.1. Disposition: With both projecting parents over-functioning throughout in both arms, the child has lent down to its own floor in both, so the capacity rule has nothing to change: +0.15 ± 0.19 (plan §4.11). *(owner 2026-10-17)*
- `M11.C.24` — dissolve a discrepant pair and the discrepancy reverses. *Premise; Phase D part I record; verdict PASS:* **pass**. Evidence: `under_change` +10.6 ± 0.56; `over_change` -11.5 ± 0.66.
- `M11.C.37` — anxiety is conserved across a death. *Check; a unit test; verdict PASS:* **pass, proved by a test**. Evidence: proved by `tests/bowen/test_life_course.py::test_m11c37_anxiety_conserved_across_death`.
- The conservation limbs of `M11.C.11` (entry 20) and `M11.C.31` (entry 6) test `M6.I.6` too.

**Coverage:** 11 IDs: 6 done, 4 partial, 1 not done. No test names: `M6.2`.

**Where to see it in the trace:**
- The last line: "104 weeks. The invariants were asserted at the end of every week." (trace line 930).
- Week 8, Ravi OVERFUNCTION → Pia; witnesses Marta, Nadia (trace line 110). What it did: "arrives week 9; anxiety Marta +4.1 (witness), Nadia +1.3 (witness), Pia +4.1, Ravi +1.1; Pia–Ravi functioning balance flips; Ravi takes 0.8 of functioning from the other"
  - The balance flips and functioning changes hands. The row prints the gain; `M6.I.4` requires the same amount to
    leave the other side.
- Week 89, Marta REDUCE_CUTOFF → Nadia; witnesses Pia, Ravi (trace line 806). What it did: "arrives week 90; anxiety Marta +0.7, Nadia +0.5, Pia +0.6 (witness), Ravi +1.0 (witness); Marta's rejection -0.25; Nadia's rejection -0.24; bound anxiety released to the third party: Ravi +23.1"
  - The 23.1 released to Ravi equals the 23.1 bound into Marta–Nadia at week 47 (trace line 441): what distance bound
    comes back (`M1.D.2a`, `M6.I.6`).

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

## Part 4 — The slower processes

### 19. `M7.A`–`M7.B` — Basic level and chronic anxiety on the slow tick

**What it requires.** `basic_level` is never advanced by a move. Once a year it is re-estimated from the person's
functional-level history, with the situations weighted by the load they carried and nodal events in the window, and it
cannot rise while the person is financially dependent (`M7.A.1`, `M7.A.1a`, with the estimator's form at `M1.A.4k`).
Differentiation gained in a peripheral system does not transfer to the nuclear family automatically; the family of
origin is the privileged peripheral system (`M7.A.2`–`M7.A.2b`). Chronic anxiety is fixed in childhood from witnessed
history and after that falls only with a rise in basic level (`M7.B.1`, `M7.B.1a`).

**Explainer:** §3.1 (basic level), §3.2a (how basic level is obtained), §3.3 (chronic anxiety), §8 (clocks).

**Config:** `constants.md` (`estimator_window`, `estimator_breadth`, `estimator_rate`, `estimator_sd_weight`),
`nodal_events.md`.

**Criteria:**
- `M11.C.18` — the estimator tells a cutoff-heavy arm from an `I-POSITION`-heavy arm at the same functioning. *Check; Phase D part I record; verdict PASS:* **pass**. Evidence: `rise` +0.0254 ± 0.003.
- `M11.C.20` — the estimator separates a borrowed gain from a basic one. *Check; Phase D part I record; verdict PASS:* **pass**. Evidence: `rise` +0.026 ± 0.0029.
- `M11.C.33` — the dependence gate binds the slow clock. *Premise; Phase D part I record; verdict PASS:* **pass**. Evidence: `rise` +0.0262 ± 0.0031.

**Coverage:** 7 IDs: 4 done, 0 partial, 3 not done. No test names: `M7.A.2`, `M7.A.2a`, `M7.A.2b`.

**Where to see it in the trace:**
- Week 51, system row (trace line 482): "slow tick (yearly): 10 ties wore (conductance); every dyad a year older"
- Week 103, system row (trace line 928): "slow tick (yearly): 10 ties wore (conductance); every dyad a year older"
  - No person rows follow either slow-tick row, so no adult's basic level rose in this run (the renderer prints a row
    for each person field the slow tick changes, `src/bowen/render/trace.py`; people under `development_end_age_years`
    are skipped by the estimator, `src/bowen/engine/slow_tick.py`).

**Points to check:** `TODO.md`'s diagnosis of `M11.C.17` found the coaching effect dying at the estimator's evidence
test (own functioning stays below basic level in both arms) and names `M1.A.4k`'s evidence rule as the owner's lever.

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

### 20. `M7.D` — Symptoms

**What it requires.** Symptom load integrates time spent above the person's usual anxiety, not peaks (`M4.C.3`,
`M4.C.3a`, revision 24), and on crossing a threshold read from functional level it emits an endogenous onset (`M7.D.1`,
`M1.A.6`). The three channels can stand in for each other: load goes to the expressed channel and every channel leaks
(`M7.D.2`); load carried in one channel protects against the other two (`M7.D.2c`). Lock-in: while a symptom is mild
the people tied to the bearer move part of their anxiety into the bearer's hold, so the family calms; past the turn the
hold spills back (`M7.D.2a`, `M7.D.2d`). A removal has opposite effects on different members, by position (`M7.D.2b`).
Removal (`INSTITUTIONALIZE`) fires off another member's tolerance, not the bearer's severity, and is suppressed when
decisions are held by an outside agent (`M7.D.4`). `M7.D.3` is withdrawn as a mechanism; its result is a criterion.

**Explainer:** §3.9 (`symptom_load`), §9.2 (illness is an output before it is an input).

**Config:** `constants.md` (`lock_in_gain`, `lock_in_turn`, `channel_protection`, `constitution_strength`,
`focus_rate`), `attribution.md` (which acts focus which channel), `family_phase_c.md` (`channel_prior`).

**Criteria:**
- `M11.C.11` — a removal runs in three phases, the return is the larger disturbance, and anxiety is conserved. *Mixed; Phase D part I record; verdict PASS:* **pass**. Evidence: `disturbance` +3.37 ± 1.6; proved by `removed-member-hold-released-while-away` red (Phase D mutation record).
- `M11.C.12` — curing a spouse's symptom without changing the deficit raises marital conflict. *Composite; Phase D part I record; verdict FAIL:* **fail accepted as a finding**. Evidence: `marital_conflict` +0.02 ± 0.41. Disposition: Released anxiety raises the other spouse's excess, but more anxiety does not select `CONFLICT` in particular (`TODO.md`, B7). *(owner 2026-10-17)*
- `M11.C.22` — symptom lock-in, and its turn (a mild and a severe limb). *Mixed; Phase D part I record; verdict PASS:* **pass**. Evidence: `mild_removal_effect` +1.15 ± 0.67; `severe_removal_effect` -1.37 ± 0.83; proved by `lock-in-removed` red (Phase D mutation record). Disposition: The owner accepted the mild limb's failure (2026-10-17). With the nodal-event spike (revision 32) both limbs pass; the mild limb's composite reading stands, and a recorded mutant (lock-in removed) must turn it red (V3). *(Claude's recommendation 2026-10-18, pending owner review)*
- `M11.C.23` — expressed emotion dominates medication. *Composite; Phase D part I record; verdict FAIL:* **fail accepted as a finding**. Evidence: `relapses` -0.687 ± 0.087. Disposition: Hostility delivers 41 ± 10 more appraisal load than talking it out, but higher anxiety decays faster (−33 ± 10) and the child gets relief by cutting off and distancing, so symptom load is no higher; medication's effect outweighs the hostility term (plan §4.11, traced ad hoc 2026-10-16). *(owner 2026-10-16)*
- `M11.C.30` — carrying one symptom protects against a second. *Premise; Phase D part I record; verdict PASS:* **pass**. Evidence: `social_load` -31.9 ± 3.

**Coverage:** 8 IDs: 6 done, 0 partial, 2 not done. No test names: `M7.D.2b`, `M7.D.3`.

**Where to see it in the trace:**
- Week 15, Pia SYMPTOM_ONSET → Marta, Ravi; witnesses Nadia (trace line 171). What it did: "arrives week 16; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); symptom onset, social channel; anxiety Marta +3.7, Nadia +0.6 (witness), Ravi +4.1"
- Week 16, Bruno SYMPTOM_ONSET → Ana (trace line 182). What it did: "arrives week 17; the nodal event re-opens the cut tie Ana–Bruno: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); symptom onset, social channel; anxiety Ana +1.8"
  - Bruno's onset reaches Ana only: his one tie in this family.

**Points to check:**
- Ten onsets in this run: Ravi (weeks 5 and 90), Marta (6), Pia (15 and 75), Bruno (16), Nadia (16 and 76), Ana (75)
  and Dr Halim (77). Lock-in is not printed line by line.
- No `INSTITUTIONALIZE` row appears. `family_phase_c.md` declares no tolerance for anyone, and `M7.D.4` as built never
  fires for a member with no tolerance listed (`src/bowen/engine/symptoms.py`).

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

### 21. `M7.C` and `M6.3` — The life course, and death

**What it requires.** Life stage governs childhood, launch, partnering and mortality (`M7.C.1`). A new tie
redistributes a conserved investment budget rather than adding a stressor (`M7.C.1a`–`M7.C.1d`). A person acts from
age 10 until death (`M7.C.1f`); a child launches at 20 (`M7.C.1g`); partners and births come from settings and are
never invented by the engine (`M7.C.1h`, `M7.C.1i`); each person's age at death is drawn once from a declared hazard
(`M7.C.1j`); `FIND_WORK` follows a lost income (`M7.C.1k`). A death goes through `M6.3`'s mechanism: the dead person's
anxiety goes to the living in proportion to their ties; tie- and triangle-bound anxiety is discharged; investment is
redistributed; functioning balances settle back; sinks that named the dead return to the family; grief and lost income
reach the living as lasting stressors; and the rest is not written as a rule (`M6.3a`–`M6.3i`).

**Explainer:** §8.1 (the life course).

**Config:** `family_phase_d.md`, `partners_phase_d.md`, `constants.md` (`acting_age_years`, `launch_age_years`,
`mortality_a`, `mortality_b`, the `grief_*` and `income_*` constants, `coach_retirement_age_years`).

**Criteria:**
- `M11.C.15` — death destabilises as recovery does. *Composite; Phase D part II record; verdict PASS:* **pass**. Evidence: `death_rise` +2.6 ± 0.76; `recovery_rise` +5.25 ± 0.52; proved by `death-bound-anxiety-discarded` red (Phase D mutation record).
- `M11.C.37`, anxiety conserved across a death, is in entry 18.

**Coverage:** 22 IDs: 16 done, 0 partial, 6 not done. No test names: `M6.3`, `M6.3i`, `M7.C.1b`, `M7.C.1c`, `M7.C.1d`, `M7.C.1e`.

**Where to see it in the trace:** nothing. In 104 weeks of the Phase C family no one dies, launches, partners or is
born, and no `FIND_WORK` row appears; the slow-tick rows say only "every dyad a year older". The life course is seen
in the twelve-person family's records (`docs/phase_d2_ensemble_record.md`); no trace of that family is rendered.

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

### 22. `M7.E` — Generational transmission and tie wear

**What it requires.** The generational ordering (the primary projection object lower than the parents, minimally
involved siblings about the same, those outside the process better) is a result checked on the output and never used
to assign a child's level (`M7.E.1`, `M7.E.1a`). Siblings differ because they occupy different parental triangles
(`M7.E.1b`). Five situations raise parental anxiety focused on a child but do not select the target (`M7.E.1c`);
projection can begin with no signal from the child (`M7.E.1d`); transmission reads the parents' functioning on the
marital tie (`M7.E.1e`). A child's basic level develops from 5 to 20, within a band of the parents' (`M7.E.1f`). The
generational rate is stochastic, never a fixed decrement (`M7.E.2`). Ties wear by default, faster at a lower level
(`M7.E.4`). `M7.E.3` is withdrawn as a mechanism.

**Explainer:** §6.2 (the family's sinks), §8 (the slow tick), §8.1 (development from 5 to 20), §8.3 (focus that locks
in).

**Config:** `constants.md` (the `development_*` constants, `tie_wear_rate`, `wear_offset_per_act`,
`focus_compound_gain`, `focus_reinforce_gain`), `family_phase_d.md`, `pattern_acts.md` (the projection-type acts:
`OVERFUNCTION`, `PURSUE` and `TRIANGLE` from a parent toward a child).

**Criteria:**
- `M11.C.2` — symptoms concentrate on the child who received the most projection-type acts. *Composite; Phase D part I record; verdict FAIL:* **fail accepted as a finding**. Evidence: `symptom_load` +1.48 ± 2.8. Disposition: On the Phase C family over ten years, with projection-type acts counted in the first five years and symptom load read in the last five, the most-projected child carries no more; read over the same weeks it passed, which cannot separate cause from attraction (spec row, revision 26). Read by age on the twelve-person family it passes (`M11.C.2[twelve-person family]`, proved red by removing focus on weakness). *(owner 2026-10-17)*
- `M11.C.2[twelve-person family]` — the same, read by age on the twelve-person family. *Composite; Phase D part II record; verdict PASS:* **pass**. Evidence: `symptom_load` +3.42 ± 1.8; proved by `witness-path-removed` red (Phase D mutation record), `weakness-focus-and-steering-removed` red (Phase D mutation record).
- `M11.C.6` — transmission is multigenerational: the projection line's level declines. *Composite; Phase D part II record; verdict FAIL:* **fail accepted as a finding**. Evidence: `development` +0.0957 ± 0.044. Disposition: Traced 2026-10-17 at the owner's request ("Trace it first"): the first arm picked children by raw act counts over spans of different length, so it picked by birth order and run year, and the band never bound (0 binds in 180 child-runs). Restated at revision 32 (post hoc) as a paired arm on one child (Nadia): the parents' projection-type acts toward her as declared, against the same seed with them withheld. It still fails, the other way: she develops slightly more with them (ad hoc, 150 seeds). Recommended: accept as a finding: projection-type acts do not lower a child's functioning in the model, so `M7.E.1f`'s band, which they open and do not push, never constrains development. Whether projection should push development down is a rule decision ("the acts open room, they do not push"). *(owner 2026-10-18)*
- `M11.C.9` — the sibling-position effect peaks mid-scale. *Premise; no record; verdict not built:* **out of scope**. Evidence: no rule reads sibling position, so building it means a new mechanism, not a gate (`M1.A.14c`); out of version 1, Claude's recommendation 2026-10-17 pending owner review (`docs/version_1_dispositions.md`). Disposition: No rule reads sibling position (`objects.py` documents it as never read by the propensity vector; `functional_sibling_position` is never written; no fixture has ranked siblings joined by a tie). The decision of 2026-10-15 to include it rested on "one gate that is partly built"; building it is a new mechanism whose profiles the spec leaves open (`M1.A.14c` gives one example) and which could write a pattern (§19). Recommended for version 2. *(owner 2026-10-17)*
- `M11.C.10` — cutoff begets cutoff in the next generation. *Composite; Phase D part II record; verdict FAIL:* **fail accepted as a finding**. Evidence: `cut_off_tie_weeks` +4.51 ± 1.4e+02. Disposition: The arms' input barely differs: the parents already hold some rejection on some tie most weeks, and the held cutoff adds 0.08 to 0.12 to the first grandchild's origin cutoff (3 seeds, 12 years, ad hoc 2026-10-15). The family-of-origin variant passes and is proved red by removing `M4.D.6f`. *(owner 2026-10-17)*
- `M11.C.10[family of origin]` — the same, with both parents' cutoffs from their own parents held. *Composite; Phase D part II record; verdict PASS:* **pass**. Evidence: `cut_off_tie_weeks` +238 ± 1.5e+02; proved by `witnessed-cutoff-learning-removed` red (Phase D mutation record).

**Coverage:** 10 IDs: 3 done, 0 partial, 7 not done. No test names: `M7.E.1`, `M7.E.1a`, `M7.E.1b`, `M7.E.1d`, `M7.E.1e`, `M7.E.2`, `M7.E.3`.

**Where to see it in the trace:**
- Week 51, system row (trace line 482): "slow tick (yearly): 10 ties wore (conductance); every dyad a year older"
  - Tie wear (`M7.E.4`).
- Week 9, Marta OVERFUNCTION → Nadia; witnesses Pia, Ravi (trace line 114). What it did: "arrives week 10; anxiety Marta +1.2, Nadia +4.1, Pia +1.1 (witness), Ravi +3.8 (witness); Marta takes 0.5 of functioning from the other"
  - A projection-type act from a parent toward a child. In the diagram, Marta–Nadia carries an over/underfunctioning
    line from block 6 to block 26.

**Points to check:** the generational processes need decades and the twelve-person family; two years of the Phase C
family show only their inputs. The rule question `TODO.md` keeps open for `M11.C.6` (should projection push a child's
development down, `M7.E.1f`?) is in the open questions below.

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

## Part 5 — Parameters, tests, readouts and scope

### 23. `M10` — Parameters

**What it requires.** Every constant appears in the register with a grade and resolves from config or a derivation
from `basic_level` (`M10.1`); derivation is preferred, and agency is the solid-self fraction, not a parameter
(`M10.A.1`, `M10.A.1a`, `M10.A.2`). Editorial content lives in config, not Python (`M10.B.1`); config parsing fails on
an unknown key or a malformed line (`M10.B.2`); the estimator's window and breadth are declared, with a sensitivity
analysis (`M10.B.3`); constants are frozen before the suite runs, every later change is logged, and a result that
passes only after one is reported as post hoc (`M10.B.4`); learning can be switched off in tests only (`M10.B.5`). The
`[I]` register (`M10.C.1`, `M10.C.1a`); the quantities usable as calibration targets, including one external target
(`M10.C.2`, `M10.C.2a`, `M10.C.2b`); bounds and shapes admitted as checks on an output, never as parameters
(`M10.C.4`–`M10.C.4b`); a premise column (`M10.C.5`); no per-generation or societal rate as a parameter (`M10.C.3`,
`M10.C.3a`).

**Explainer:** "How to read an entry" (the grades), §15.3 (magnitudes the corpus supplies).

**Config:** `constants.md`, `constants_changes.md` (every change after the freeze, and the criterion it altered),
`constants_frozen.md` (the Phase C snapshot), `constants_frozen_phase_b.md`.

**Criteria:** none in `M11.C`. The engineering tests are `M11.D.2`, `M11.D.3`, `M11.D.4` and `M11.D.12`, and
`tests/bowen/test_phase_b_gate.py::test_m10b4_constants_frozen_before_suite`.

**Coverage:** 20 IDs: 5 done, 2 partial, 13 not done. No test names: `M10.A.1`, `M10.A.1a`, `M10.A.2`, `M10.B.3`, `M10.C.2`, `M10.C.2a`, `M10.C.2b`, `M10.C.4`, `M10.C.4b`, `M10.C.4a`, `M10.C.5`, `M10.C.3`, `M10.C.3a`.

**Where to see it in the trace:** the header rows "constants frozen | 2026-10-07" and "constants changed since freeze",
which lists 78 constants (trace lines 20–21).

**Points to check:** `docs/version_2_plan.md` proposes a constant sweep; this entry is where to note any constant to
look at first.

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

### 24. `M11` — Acceptance and engineering criteria

**What it requires.** Each criterion has a named test, asserts a direction of difference between two arms, and counts
as coverage only once its mutation has been run and turned it red (`M11`, `M11.1`); mutations are real behaviour
changes, with matched-magnitude, representation-invariance and outcome-directive variants (`M11.1a`–`M11.1d`);
asymmetric rules are tested at matched frequency (`M11.1e`); a criterion fails when the move it depends on never
occurs (`M11.1f`). Ensemble tests only, never one run (`M11.2`); a criterion that tells two states apart runs under a
stressor (`M11.3`); the exceptions to two-arm directions are listed (`M11.4`–`M11.4f`). Every criterion carries a
class: a **premise** passing confirms that the code does what its rule says and is never reported as a finding; a
**composite** follows from several rules and is the model's evidence; a **check** tests an instrument (`M11.5`).
`M11.D` holds the engineering criteria; `M11.E` lists four criteria that cannot be made code-testable in Phases B–D.

**Explainer:** §11 (the acceptance tests), §11.3 (what revision 32 changed), §18.6 (test design), §19 (set, random,
emergent).

**Config:** `criteria.md` (every number a criterion sets for its arms), `criterion_roles.md`, `fixtures/`.

**Criteria, all together** (`docs/version_1_status.md`). Rows by status:
fail accepted as a finding 19; out of scope 4; pass 36; pass, proved by a test 1. V1, V2 and V3 are met.
Rows whose disposition was Claude's recommendation pending owner review, all accepted by the owner 2026-10-19
(version 2 plan Q10):
`M11.C.7[joint over one spouse]` (entry 16), `M11.C.7[joint over sided]` (entry 16), `M11.C.17` (entry 15), `M11.C.22` (entry 20).
The criteria `M11.E` holds out are `M11.C.8`, `M11.C.9`, the shape clause of `M11.C.11`, and `M5.F.2`'s counterfeit
threshold, which `TODO.md` calls "the model's most consequential invented constant".

**Coverage:** 39 IDs: 28 done, 2 partial, 9 not done. No test names: `M11.1`, `M11.1b`, `M11.1e`, `M11.4`, `M11.4b`, `M11.4c`, `M11.4f`, `M11.3`, `M11.D.12`.

**Where to see it in the trace:**
- Week 4, JOB_LOSS from outside → Marta; witnesses Nadia, Pia, Ravi (trace line 65). What it did: "a 1-week spell, one week; the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +3.2, Nadia +1.5 (witness), Pia +1.2 (witness), Ravi +1.8 (witness)"
  - This run is one seed of `M11.C.1`'s baseline arm, under its declared spell. The criterion compares it with a
    treatment arm (levels lowered by 10, `criteria.md`) over many seeds; one arm of one seed decides nothing.

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

### 25. `M11.G` — The family-evaluation readout

**What it requires.** The model emits a structured family evaluation with a given schema (`M11.G.1`). Adaptiveness is
derived as reactivity compared against stress and is undefined, never 0, with no stress (`M11.G.2`). The source's
ninth and tenth components, therapeutic focus and prognosis, are never emitted (`M11.G.3`). The evaluation carries
`M11.F`'s framing and reports belief and truth separately (`M11.G.4`).

**Explainer:** §10 (readouts, including the three that lie).

**Config:** none of its own; the beliefs it reports are written using `attribution.md`.

**Criteria:** none in `M11.C`; the tests are in `tests/bowen/test_evaluation.py`, and the readout is
`src/bowen/readouts/evaluation.py`.

**Coverage:** 4 IDs: 4 done, 0 partial, 0 not done.

**Where to see it in the trace:** nothing. The evaluation is a separate readout and is not rendered for this run.

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

### 26. `M12` — Out of scope, and never to be built

**What it requires.** Mechanisms proposed and withdrawn are named so they are not re-invented, and are never built
(`M12.1`). Two questions the spec records as unanswered in the corpus are labelled `[I]` wherever the model decides
them (`M12.2`, `M12.2a`; the explainer gives the sources). No anniversary effect (`M12.3`); no metabolic column
(`M12.4`); `power` and `punishment` do not exist as mechanisms (`M12.5`).

**Explainer:** §12 (what is deliberately not in the model).

**Config:** none.

**Criteria:** none.

**Coverage:** 6 IDs: 0 done, 0 partial, 6 not done. No test names: `M12.1`, `M12.2`, `M12.2a`, `M12.3`, `M12.4`, `M12.5`.

**Where to see it in the trace:** nothing; a trace cannot show an absence. The check is the code and this review.

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

### 27. `M13` — Implementation order

**What it requires.** `M8` lands in Phase B, before the policy (`M13.1`); the frozen grid engine and its tests stay
green, and v2 lives in `src/bowen/` (`M13.2`); every criterion gates the phase its row names (`M13.2a`); `M15`'s
importer is Phase E and is not built in B–D (`M13.3`); Phases C and D stop their ensembles adaptively under `M17.A.1`
(`M13.4`).

**Explainer:** none of its own; §18.7 covers `M13.4`'s stopping rule.

**Config:** `script_phase_b.md` (the Phase B script).

**Criteria:** none in `M11.C`; `M11.D.11` checks that every criterion gates its phase.

**Coverage:** 5 IDs: 4 done, 1 partial, 0 not done.

**Where to see it in the trace:** nothing.

**Points to check:** `docs/version_2_plan.md` §5 asks whether to retire the grid engine, which `M13.2` keeps green.

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

### 28. `M14` — Spec coverage and the state and mechanism register

**What it requires.** `docs/spec_coverage.md` is generated as part of a completion report, every ID marked done,
partial or not done, each done with the test that proves it (`M14.1`), and "implementation complete" is not accepted
without it (`M14.2`). The register classifies every state variable of `Person`, `Relationship`, `Triangle` and `Family`,
lists what each mechanism reads and writes, and places each mechanism in `M3.D.1`'s order, with the phase that builds
it (`M14.A.1`–`M14.A.4`).

**Explainer:** §18.5 (structure and lifecycle).

**Config:** none.

**Criteria:** none in `M11.C`; `M11.D.17` checks that the register has no orphans.

**Coverage:** 6 IDs: 4 done, 1 partial, 1 not done. No test names: `M14.2`.

**Where to see it in the trace:** nothing directly; the mechanisms the header says run every week (trace line 23) are
register mechanisms.

**Points to check:** `docs/spec_coverage.md`'s title still reads "after Phase C", and its "not done" means no test
names the ID (see "Coverage counts").

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

### 29. `M17` — Phase E: the ensemble runner, arms and readouts (first draft)

**What it requires.** Built, and used by Phases C and D (`M13.4`): seeds are added in blocks until the interval on the
per-seed arm difference is narrower than a declared precision, up to a seed cap, and a criterion not converged at the
cap is UNDETERMINED (`M17.A.1`); a paired statistic (`M17.A.3`); a minimum effect margin declared before the run
(`M17.A.4`); and the engine cannot see which arm it is in (`M17.D.3`, checked by `M11.D.22`). The rest is a first
draft for Phase E, not built: per-seed readouts and regime classes (`M17.B`), initial conditions as a declared
distribution (`M17.C`), named control and rival-mechanism arms (`M17.D`), constant sweeps and sensitivity (`M17.E`),
shock-and-settle protocols (`M17.F`), and an audit record per result (`M17.G`).

**Explainer:** §18.7 (Phase E, the ensemble runner), §17.4 (run estimates as ranges).

**Config:** `constants.md` (`ensemble_precision` and the other ensemble settings), `criteria.md` (seeds where a
criterion fixes them).

**Criteria:**
- `M11.C.8` — endogenous incidence lands near published rates. *Unclassified; no record; verdict not built:* **out of scope**. Evidence: no record lists it. Disposition: A Phase E criterion (`→E`): endogenous incidence against published rates needs the editorial call on which sources are exogenous, accepted as not code-testable in Phases B–D on 2026-10-06 (`M11.E`). Version 1 excludes Phase E (§3). *(owner 2026-10-17)*

**Coverage:** 29 IDs: 4 done, 0 partial, 25 not done. No test names: `M17.A.2`, `M17.B.1`, `M17.B.2`, `M17.B.3`, `M17.B.4`, `M17.B.5`, `M17.B.6`, `M17.C.1`, `M17.C.2`, `M17.C.3`, `M17.D.1`, `M17.D.2`, `M17.D.4`, `M17.D.5`, and 11 more.

**Where to see it in the trace:** nothing; the runner works across seeds.

**Points to check:** the open item on criteria near the runner's detection limit (open questions) would change
`M17.A.1` for every criterion.

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

### 30. `M15` — Family-diagram import (Phase E, not built)

**What it requires.** A contract on the diagram application: structural fields exported as values; every rating as an
interval, declared basic or functional, with its rater and date; no differentiation score; the identified patient not
exportable into basic level or a sink (`M15.A`). Structure imports as values and ratings as ranges (`M15.B`); four
adapter traps are named (`M15.C`); an imported family runs as ranges and reports an envelope, never a point direction
(`M15.D`); an anonymised fixture mode (`M15.E`).

**Explainer:** §17.5 (import, and what it does not change), §17.3 (initial conditions are not nuisance parameters).

**Config:** none.

**Criteria:** none.

**Coverage:** 25 IDs: 0 done, 0 partial, 25 not done. No test names: `M15.A.1`, `M15.A.1a`, `M15.A.2`, `M15.A.3`, `M15.A.4`, `M15.A.5`, `M15.A.6`, `M15.A.7`, `M15.A.8`, `M15.B.1`, `M15.B.2`, `M15.B.3`, `M15.B.4`, `M15.B.5`, and 11 more.

**Where to see it in the trace:** nothing. The fam-diag export used in this review goes the other way, model to
diagram (`tools/famdiag_export.py`), and is not `M15`.

**Points to check:** `docs/version_2_plan.md` §5 asks about `M15`.

- [ ] Approve  [ ] Change: ____  [ ] Question: ____
- Notes: ____

---

## Open owner questions from `TODO.md` ("Revision 32", open items)

The unticked items in `TODO.md`'s section "Revision 32 (finishing version 1) — open after the build, 2026-10-17", in
its order, plus the rule question its ticked `M11.C.6` entry keeps open. Items marked *Owner* are addressed to the
owner there; the others are recorded open with no owner tag. `TODO.md` holds the full text.

1. **A genuine `I-POSITION` at level 40 almost never completes** since revision 16 (0 of 200 seeds reach the peak in
   `M11.C.32`'s calm arm at level 40). `hold_gain` and `defence_threshold` may be sized for the calmer model before
   cutoff became rejection. A rule decision for every `I-POSITION`. (Entry 10.)
   - Answer: ____
2. **Criteria near the runner's detection limit.** `M11.C.22`'s mild limb has about a 78% pass chance at the
   runner's stop; others may be similar. A general rule (a fixed seed count when the effect is within a set multiple
   of the precision) would change `M17.A.1` for all criteria. (Entry 29.)
   - Answer: ____
3. **`M1.E.7e`'s lending of pseudo-self by a sided coach is not built**: a sided coach's contact lands nothing.
   (Entry 15.)
   - Answer: ____
4. *Owner:* **`M11.C.46`'s mutation clause is not met at the edge** (2026-10-18). With the nodal spike,
   `triangle-help-ignored` survives (+0.16 ± 0.23, p 0.041) against +1.45 unmutated. Options: accept, as for
   `M11.C.14`'s clause; or restate the readout as relief per unit of the seeker's excess (post hoc). A premise, so V3
   does not apply. (Entry 13.)
   - Answer (owner, 2026-10-19, version 2 plan Q10): accept, as for `M11.C.14`'s clause. Recorded as a note in `docs/version_1_dispositions.md`.
5. *Owner:* **the nodal spike's narrowing** (2026-10-17): a cut tie is one holding rejection, not every tie that is not
   interactive, so a retired coach's or a removed member's tie is not spiked. Claude's recommendation, pending owner
   review. (Entry 7.)
   - Answer (owner, 2026-10-19, version 2 plan Q10): keep the narrowing. After G1, a cut tie is one in state `CUT_OFF`.
6. **Adjacent, not fixed: the Phase B mutation record cannot be regenerated.** Gate G2's `expect` fragment was removed
   from `tests/bowen/test_render.py` at Phase C step 1, so `tools/mutation_gate.py` stops at G2;
   `docs/phase_b_mutation_record.md` stays the record of Phase B's close. Gate G7's mutant was extended at revision 32
   and proved on a copy, not in the record.
   - Answer: ____
7. **Adjacent, not fixed: `canonical_state` leaves out `state.stressors`**, so the determinism tests cannot see a
   stressor still weighing after the last compared tick. (Entry 5.)
   - Answer: ____
8. **`M11.C.29` could be redesigned** (optional, for version 2): its genuine `I-POSITION` never completes in the arm,
   so its time-course limb is untested. (Entry 10.)
   - Answer: ____
9. **Minor items from the revision 32 reviews, not fixed:** `SIDE_TAKING` and `FAMILY_TO_EXTERNAL` are
   classifications kept in code, which the working rules (rule 9) put in config; how long a coach stays on a side
   borrows `triangle_activity_window`, so tuning `M1.C.3` changes `M11.C.7` too; `arm_c7` scripts the `SPLIT` at
   `session − 1`, inside the window only while the coach tie's latency is 2 or less (it is 1); the status tool counts
   a red on any gating readout of a mixed criterion without knowing which limb a mutant reached (no passing mixed
   criterion depends on this today); and, adjacent, `recompute.py`'s triangle activity treats the coach as an unfused
   occupant though `M8.5` says the same predicate decides triangle activity (entry 16).
   - Answer: ____
10. **The rule question kept open by `M11.C.6`** (accepted as a finding by the owner 2026-10-18): should projection push
    a child's development down (`M7.E.1f`)? Today "the acts open room, they do not push". (Entry 22.)
    - Answer: ____

---

## Every criterion, and the entry that reviews it

From `docs/version_1_status.md`; each row is reviewed once, in the entry named.

| Criterion | Class | Status | Reviewed in |
|---|---|---|---|
| `M11.C.1` | premise | pass | entry 11 |
| `M11.C.2` | composite | fail accepted as a finding | entry 22 |
| `M11.C.2[twelve-person family]` | composite | pass | entry 22 |
| `M11.C.3` | premise | pass | entry 13 |
| `M11.C.4` | premise | pass | entry 7 |
| `M11.C.5` | composite | fail accepted as a finding | entry 10 |
| `M11.C.6` | composite | fail accepted as a finding | entry 22 |
| `M11.C.7[joint over one spouse]` | mixed | pass | entry 16 |
| `M11.C.7[joint over sided]` | mixed | pass | entry 16 |
| `M11.C.7[one spouse over sided]` | mixed | fail accepted as a finding | entry 16 |
| `M11.C.8` | unclassified | out of scope | entry 29 |
| `M11.C.9` | premise | out of scope | entry 22 |
| `M11.C.10` | composite | fail accepted as a finding | entry 22 |
| `M11.C.10[family of origin]` | composite | pass | entry 22 |
| `M11.C.11` | mixed | pass | entry 20 |
| `M11.C.12` | composite | fail accepted as a finding | entry 20 |
| `M11.C.13` | premise | out of scope | entry 14 |
| `M11.C.14[firmness]` | premise | pass | entry 9 |
| `M11.C.14[talking it out]` | premise | pass | entry 9 |
| `M11.C.15` | composite | pass | entry 21 |
| `M11.C.16` | composite | pass | entry 8 |
| `M11.C.17` | composite | pass | entry 15 |
| `M11.C.18` | check | pass | entry 19 |
| `M11.C.19` | check | pass | entry 10 |
| `M11.C.20` | check | pass | entry 19 |
| `M11.C.21` | composite | fail accepted as a finding | entry 18 |
| `M11.C.22` | mixed | pass | entry 20 |
| `M11.C.23` | composite | fail accepted as a finding | entry 20 |
| `M11.C.24` | premise | pass | entry 18 |
| `M11.C.25` | premise | pass | entry 12 |
| `M11.C.26` | premise | pass | entry 17 |
| `M11.C.27[stable,add_third]` | composite | fail accepted as a finding | entry 13 |
| `M11.C.27[stable,remove_one]` | composite | fail accepted as a finding | entry 13 |
| `M11.C.27[unstable,add_third]` | composite | fail accepted as a finding | entry 13 |
| `M11.C.27[unstable,remove_one]` | composite | pass | entry 13 |
| `M11.C.28` | composite | out of scope | entry 14 |
| `M11.C.29` | mixed | fail accepted as a finding | entry 10 |
| `M11.C.30` | premise | pass | entry 20 |
| `M11.C.31` | check | pass | entry 6 |
| `M11.C.32` | premise | pass | entry 10 |
| `M11.C.33` | premise | pass | entry 19 |
| `M11.C.34` | mixed | fail accepted as a finding | entry 8 |
| `M11.C.35` | check | pass | entry 7 |
| `M11.C.36` | check | pass | entry 17 |
| `M11.C.37` | check | pass, proved by a test | entry 18 |
| `M11.C.38[-0 vs -5]` | premise | pass | entry 11 |
| `M11.C.38[-10 vs -15]` | premise | pass | entry 11 |
| `M11.C.38[-5 vs -10]` | premise | pass | entry 11 |
| `M11.C.39` | composite | pass | entry 8 |
| `M11.C.40` | premise | pass | entry 8 |
| `M11.C.41[heavy: lower level]` | mixed | pass | entry 7 |
| `M11.C.41[higher level: heavier stress]` | mixed | fail accepted as a finding | entry 7 |
| `M11.C.41[light: lower level]` | mixed | pass | entry 7 |
| `M11.C.41[lower level: heavier stress]` | mixed | fail accepted as a finding | entry 7 |
| `M11.C.42` | composite | fail accepted as a finding | entry 13 |
| `M11.C.43` | composite | fail accepted as a finding | entry 7 |
| `M11.C.44` | composite | fail accepted as a finding | entry 13 |
| `M11.C.45` | composite | fail accepted as a finding | entry 13 |
| `M11.C.46` | premise | pass | entry 13 |
| `M11.C.47` | premise | pass | entry 12 |

---

**Owner sign-off:** [ ] Review complete. Date: ____
