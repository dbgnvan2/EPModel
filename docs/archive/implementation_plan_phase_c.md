---
tags: [model-bt, plan]
status: APPROVED 2026-10-07 — all recommendations (P1–P12, D1–D10) adopted as written; §1's spec changes applied as revision 12
date: 2026-10-07
spec: docs/bowen_agent_model_spec_v2.md, v2.0 revision 12, approved 2026-10-07
scope: Phase C only
---

# Implementation plan — Phase C

## 0. Scope and how to read this

Phase C, from `M13`'s table as it stands at revision 11:

> **Builds:** M4.C appraisal including M4.C.8a; M4.D policy; M5 the full repertoire, gates and the
> `I-POSITION` state machine including `PREPARE` (M5.D.2a) and the corrected anger gate (M5.D.4); M6
> invariants M6.I.1–M6.I.8; M16.D's delayed view. **Added at revision 11:** `M4.C.1`–`M4.C.1c` as
> re-derived, replacing Phase B's base appraisal and standing-load formula and removing Phase B's calm-system
> triangle threshold; `M4.C.10`; `M4.B.3`; `M5.A.1a`'s pattern readouts. By `M13.4`, also `M17.A.1`,
> `M17.A.3` and `M17.A.4`.
>
> **Done when:** M11.C.1, .3, .4, .5, .7, .13, .14, .16, .25, .27, .29, .32, .17, .18, .19, .20 pass over
> ensembles (adaptive, `M13.4`), each mutation-proved; M11.D.2, .4, .8 pass; M16.T.3, .4, .6 pass;
> M11.C.35, .38, .41 pass; M11.D.15, .16, .18, .19, .21 pass; M11.C.42, .44, .45 pass.

That is **22 acceptance criteria, 8 engineering criteria and 3 log criteria**.

**Two things are true before any Phase C code.** Phase B is not closed: the owner's read of the two rendered
traces in `docs/review/` is still open (`docs/phase_b_completion_report.md`, "The human read"). And the
external review and the Phase B report route nine items to "before the Phase C plan". This plan cannot settle
them, because each changes approved text. §1 puts each one to the owner with a recommendation. The rest of the
plan assumes the recommendations; where the owner decides otherwise, the sections named change.

**Order.** §1 prerequisites (spec and document changes, owner decisions). §2 build decisions (functional forms
and structure the spec leaves open). §3 the Phase C gate, criterion by criterion. §4 the other requirements
Phase C builds. §5 package layout. §6 build order. §7 compute budget and how ensemble tests run. §8 the
review's findings by disposition. §9 what cannot be made code-testable. §10 adjacent issues. Appendix A
lists the register rows Phase C adds.

**Conventions** are the Phase B plan's: test names embed the lowercased spec ID; every function satisfying a
requirement carries `Purpose / Spec / Tests`; `docs/spec_coverage.md` is regenerated at completion; every gate
test is proved failing by mutation with `tools/mutation_gate.py` before it counts.

**What revision 11 changes about Phase C.** Revision 11 moves Bowen's patterns out of the rules and into what
learning produces. Phase C is where that is first built: the policy, the learner and the move physics all
arrive here. The plan follows `M11.5`'s classes throughout. A **premise** criterion passing confirms the code
renders the spec. A **composite** criterion passing or failing is a result. Of the 22 Phase C criteria, `M11.5`
classes 6 as composite (C.5, C.16, C.27, C.42, C.44, C.45), 11 as premise, 4 as check and 1 as mixed (C.41).
After P3 moves C.17, C.18 and C.20 to Phase D, Phase C gates 19: 6 composite, 10 premise, 2 check, 1 mixed.
About a third of the gate can say something about the theory; the rest confirms the code. §3 carries the class
per row.

---

## 1. Prerequisites — to settle before code

Each is a change to approved text or an owner decision. **Recommendation:** apply the approved items together
as spec **revision 12**, a small revision limited to Phase C's prerequisites. The paper material held in
`TODO.md` under the stale "revision 11" label would then move to a later revision; that is P12.

### P1. Restate `M6.I.6` with a stock-and-flow table (review A3) — **blocks C.13, C.29, C.31's Phase D use, and `M4.G.2`'s full assertion**

`M6.I.6` says anxiety is "conserved and redirected, never destroyed". Appraisal creates acute anxiety and
decay removes it every tick, so as worded it fails every tick, and `M4.G.2a` holds it back. Revision 11 adds
two more terms that move anxiety between people (`M4.C.10`, calmer contact) or create it (`M4.C.1c`'s
"too little" side), and Kerr's `KS03.1` says the outsider of a triangle **generates** anxiety as well as
absorbing it.

**Recommendation.** Restate `M6.I.6` at the scope of convergence C3 and explainer I6: **conservation of bound
quantities across transfer events**. Every change to an anxiety stock is logged against a named source or
sink, and the invariant asserts that transfers balance. Proposed table:

| Stock | Sources (create) | Sinks (remove) | Transfers (must balance, to `M6.1`'s tolerance) |
|---|---|---|---|
| `Person.acute_anxiety` | appraisal (`M4.C.1`), including the standing "too little" term (`M4.C.1c`); competing urges (`M4.D.1d`); a triangle outsider's own positional anxiety (`KS03.1`) | decay toward the chronic floor (`M1.A.8`) | `TRIANGLE` pair → outsider (`M1.C.1`); calmer contact, the calmer person's share (`M4.C.10`); binding into a tie by `DISTANCE` (`M1.D.2a`) |
| `Triangle.bound_anxiety` | — | discharge on resolution (`M1.C.3a`), logged | from and to its members |
| tie-bound anxiety (`M1.D.2a`) | — | — | from the person who distanced; back to members on `RECONCILIATION` (`M4.A.3`) |
| `Family.undifferentiation_budget` | — | reduction by a completed differentiating exchange (`M11.C.29`, `M6.I.1`) | between the three sinks (`M1.D.1`); back to the family on `binder_unavailable` (`M1.F.9`) |

**Test.** `test_m6i6_transfers_balance_every_tick` asserts the transfer column each tick; a separate test
asserts every source and sink is logged by name, so an unlogged change fails. This also answers the review's
question about the budget and anxiety: the two are separate stocks, and the only stated relation between them
is the transfers listed.

**Also needed for P1:** `M11.C.29` says differentiation *reduces* the budget while `M6.I.1` says the budget is
conserved. Recommendation: amend `M6.I.1` to "conserved across its three sinks, reduced only by a completed
differentiating exchange (`M5.D.7`)". The reduction is a logged sink.

### P2. Scope `M4.B.2` row by row (review A4)

`M4.B.2` limits the policy to its own state, its beliefs and delivered events. Several approved requirements
read true state. **Recommendation, row by row:**

| Requirement | Reads | Recommendation |
|---|---|---|
| `M5.F.1` receivers "read" the actor's `outside_ness` | another person's hidden state | **Conforms if** `outside_ness` acts through the event: the actor's state sets the delivered event's contact and impingement components (§2 D2), and the receiver appraises the event. No receiver reads the field. Amend `M5.F.1`'s wording to say so |
| `M5.C` "system calm" | family-wide tension | Revision 11 already removed this for triangles (`M1.C.3`). For projection, the gate is Phase D content. Strike the row from Phase C's gate table |
| `M5.C` "marital distance high" | the marital tie's state | Read by the spouses as their own tie (conforms). Read by anyone else through their belief (P2a below) |
| `M5.C` "decision ownership held externally" | external agent state | Phase D (`M7.D.4`). Not built in C |
| `M8.1` `positions_live` | `fused_into` across the group | Engine-side predicate, not a policy read. Conforms if the policy never calls it; `M11.D.17` checks this statically |
| `M4.D.2` the person's position in the active triangle | true tension on ties the person is not party to | Through belief (P2a) |

**P2a — a minimal belief about ties one is not party to, pulled into Phase C.** `M9.8` (each person holds a
belief about each tie its policy reads and is not party to, written only from delivered and witnessed events)
is Phase D, but `M4.B.2` is in force from Phase C and two Phase C readers need it. Recommendation: build
`M9.8`'s store in Phase C, written only from witnessed events by an `[I]` smoothing rule; the rest of `M9`
stays in D. `M11.C.36` stays in Phase D.

### P3. Phase placement (review A5) — **changes `M13` and six criterion rows**

Several Phase C criteria read machinery `M13` puts in Phase D:

| Criterion | Needs | Recommendation |
|---|---|---|
| C.17 no drift in `basic_level` without a coach | the `M1.A.4a` estimator on the slow tick; `M7.E.4`'s default deterioration; multi-year runs | **Move to D** |
| C.18, C.20 the estimator discriminates | the estimator; nodal events in its window | **Move to D** |
| C.1, C.38 time to symptom threshold | symptom onset | **Keep in C.** `M4.C.3` (the chronicity integrator) is already Phase C. Pull **`M7.D.1`** (accumulate; on crossing threshold emit an endogenous event) into C. The three channels and their substitution (`M7.D.2`) stay D; in C a person's channel is its constitutional prior (`M1.A.11b`) |
| C.5, C.13, C.29 a symptom in a third person; incident location; a third party's time course | symptom onset and location | **Keep in C** with `M7.D.1` pulled in, as above |
| C.4 the cost lands at the next nodal event | a nodal event | **Keep in C.** The nodal event is scripted, through `ScriptedSource`, in both arms |

With this, Phase C gates 19 acceptance criteria, not 22, and Phase D gains C.17, C.18 and C.20.

### P4. Sub-week items and the multi-tick `I-POSITION` (review A7)

- **`M5.D.6`'s "next day".** Recommendation: restate as "the fast tick after `RESOLVE`". The tick is a week
  (`M3.A.1`); latency is already whole ticks (`M3.C.2`).
- **One outcome per tick and a sequence that spans ticks (`M4.D.1`).** Recommendation: while a person is in an
  `I-POSITION` sequence, a tick on which a sequence step is due resolves to that step; on any other tick the
  person selects normally, and acts on the sequence's tie count as `STAY-IN-CONTACT`. A step that collides
  with a forced outcome (the fallback, `M4.D.1f`) is deferred one tick and logged.
- **Explainer durations under a week** ("within hours", "3 days", "2 hour") are dropped as calibration targets.

### P5. The cancellation premise, and a sweep in Phase C (review A2)

Explainer §17.3 says an invented constant "does not interact with the intervention". The turning-point table
beside it puts a sign reversal on an invented constant. **Recommendation:** correct §17.3 and `_STATUS.md` to
§17.1's careful form (cancellation holds for error that does not interact with the intervention). For the
sweep, see D9.

### P6. Explainer drift (review D)

Six explainer passages carry superseded text (the capacity floor at 25, the transition at 50, the ceiling on
`systems_perspective`, symptom exclusivity, §11's test 9, "unbounded" bias). Revision 11 added pointers for its
own changes; these six are older. **Recommendation:** correct them in the same commit as revision 12. Doc-only.

### P7. `M11.D.16`'s equivariance (review E)

Under keyed draws (`M3.D.4a`) permuting person identifiers reassigns draws, so equivariance holds in
distribution, not byte for byte. **Recommendation:** amend `M11.D.16`'s second test to "in distribution over
an ensemble", and keep the first (batch-order permutation) byte for byte.

### P8. Which event kinds flip sign by source position (`M1.F.2`)

The Phase B report left this for the `I-POSITION` work. **Recommendation:** under revision 11 the sign comes
from the event's contact and impingement components relative to the receiver's optimum (§2 D2), so no kind
needs a hard-coded flip. `M1.F.2` is then met when the same kind, from a sender in a different position,
moves the receiver on opposite sides of its optimum. A test asserts one such pair exists.

### P9. Review B contradictions that reach Phase C

| Item | Recommendation |
|---|---|
| `M5.C`'s "`outside_ness` below threshold" is undefined for a two-axis state (`M1.A.9a`) | Two thresholds, one per axis, both `[I]`; the gate fails on either |
| `M6.I.4` (pseudo-self zero-sum) vs `M5.D.5` (both rise on a held peak) | The rise on a held peak is solid self, which `M6.I.5` exempts; state this in `M5.D.5` |
| `M5.B.3` vs `M5.B.3a` (`REDUCE_CUTOFF`: no optimum vs not monotonically good) | Revision 11's `M4.C.1` answers it: more contact on a severed tie is never penalised as such (`M5.B.3`), but it can overshoot the receiver's optimum on the impingement side (`M5.B.3a`). Also adopt `L22.6`: the transient cost lands on a third party whose distancing is being stripped (review G1) |
| `M11.C.38` is a SHOULD in a gate | Gate it as written; a SHOULD that is listed in a Done-when cell is built |
| `M11.4`'s exceptions omit C.27's 2×2 | Add C.27 to `M11.4`: four two-arm directions, one per cell, so no new form |
| `M7.B.1` vs `M1.A.7a`; `M11.C.38` sweeps `chronic_anxiety` as a parameter | `M11.C.38` sweeps the initial condition for `chronic_anxiety` (`M2.A.0a`), not a parameter; say so in its row |

### P10. Chapter findings that reach Phase C (review G1)

| Finding | Recommendation |
|---|---|
| `PROVOKE` — raising an issue deliberately; `M5.C.1a` and the live-issue gate need it | Add as `M5.B.6`, with explainer §5.7's citation (`M5.A.2` allows it) |
| `L22.6` — where `REDUCE_CUTOFF`'s cost lands | In P9 above |
| `L03.4` — two-rung avoidance ladder | Phase D (it needs the symptom-bearer and the therapist) |
| Bowen's two falsifiers (`L09.2`, `L16.2`); `L15.1`'s delayed channel | Phase D criteria. Recorded here so the Phase D plan picks them up |

### P11. Close Phase B

The owner reads `docs/review/phase_b_trace_seed7.md` and `docs/review/phase_b_trace_seed7_nadia.md` and
records anything that reads wrong. A finding there is fixed in Phase B, before step 1 below.

### P12. Labels

The emergence revision took the number 11. **Recommendation:** these prerequisites become revision 12, and the
paper material held in `TODO.md` becomes "held for a later revision", assigned when the owner reviews it.

---

## 2. Build decisions

Functional forms and structure the spec leaves open. Every constant introduced is `[I]`, declared in
`config/bowen/constants.md` with its grade, and frozen before the acceptance suite runs (`M10.B.4`).

### D1. The Phase C family, and fixture families

The Phase B instance has four active people and three who act only by script. Phase C has a policy, so every
person selects. **Recommendation:**

- **The Phase C instance:** the Phase B seven, all active, plus **Dr Halim** as the external agent
  (`M2.A.0`, `role = EXTERNAL`), whose contact is scripted in timing and policy-driven in content. Declared in
  `config/bowen/family_phase_c.md`.
- **Fixture families for criteria that need a controlled structure:** a dyad, a triad, and a nuclear four,
  declared in `config/bowen/fixtures/`. Each criterion names its family in its test. C.27's 2×2, C.42's
  triangle reuse and `M11.D.19`'s reachability use the triad; C.25 uses dyads with sex drawn per seed.

Arms differ only through `M17.D.3`'s declared channels (initial state, constants, exogenous spells, declared
mechanism switches). A fixture is an initial state.

### D2. The contact state behind revision 11's appraisal (`M4.C.1`–`M4.C.1c`)

**Recommendation.** Each person holds, for each of its ties:

- `felt_contact` ∈ [0, 1] — how much contact the person is receiving on the tie. Raised by delivered events
  by each kind's **contact component**; decays each tick with no contact at an `[I]` rate. A severed or
  non-interactive tie decays toward zero, which gives `M4.C.1c`'s build-up over time and `M11.C.4`'s
  relief-now, cost-later shape.
- `felt_impingement` ∈ [0, 1] — how much the person is being acted on. Raised by each kind's **impingement
  component**; decays at an `[I]` rate.
- `contact_optimum` — derived each tick from the tie's `bond_energy` and the person's `functional_level`, and
  moved toward closeness by acute anxiety (`M4.C.1b`). Never stored as a free parameter.

Deviation = (optimum − contact)⁺ on the "too little" side and (impingement − band)⁺ on the "too much" side.
Appraisal adds `steepness(functional_level) × deviation change × conductance`, with `route`, `source_position`
and `fidelity` as in Phase B. Steepness falls and the band widens with `functional_level` (`M4.C.1a`). The
standing load is the "too little" term evaluated every tick, before delivery (`M4.C.1c`, `M3.D.2`).

**Each event kind declares its two components** in `config/bowen/event_kinds.md` (`[I]`). `CONFLICT` carries
both (`KS04.13`). A sender's `outside_ness` scales the impingement component it delivers (P2, `M5.F.1`).

**Phase B's G3** (a `TRIGGER` on a dormant tie moves anxiety with no contact) must still pass: a `TRIGGER`
spikes the "too little" deviation on the cut-off tie (`M4.A.2`).

### D3. The policy

**Recommendation.** Per person per tick, in `M3.D.1` step 7:

1. **Legal set** (`M4.D.1e`): structural preconditions plus `M5.C`'s removing gates, logged.
2. **Two channels** (`M4.D.1a`). The mixing weight is a function of `functional_level` only, `[I]` in form.
   The automatic channel holds the seven reactive acts; the self-directed channel holds `I-POSITION` and
   `STAY-IN-CONTACT`; `WITHHOLD` is an outcome of the automatic channel computed and not emitted
   (`M4.D.1b`).
3. **Automatic channel scores** are learned values only, conditioned on a coarse state: the person's acute
   anxiety band (3 bands, `[I]` edges) and the target tie (`M4.D.3`(b)). `TRIANGLE` values are kept **per
   triangle**, which `M11.C.42` needs. The older-layer capacity limit (`M4.D.3a`) multiplies the newer acts'
   availability by a function of `functional_level`; it is a gate, not a score.
4. **Self-directed channel scores** come from `M5.F.5`'s objective (distance of the person's two-axis
   `outside_ness` from low on both axes), gated by `systems_perspective` (`M4.D.3b`), non-monotone in level
   (`M4.D.4`). Never learned (`M4.D.6d`), never scored by relief.
5. **Selection** by softmax at an `[I]` temperature, with `M4.D.1f`'s tie-break and fallback, each draw keyed
   per `M3.D.4b`. Competing urges raise anxiety by the entropy of the automatic channel's distribution
   (`M4.D.1d`).

**Initial learned values: equal for all legal acts.** Any initial ordering would write a preference in.

**The policy module may import only** its own person's state, its belief store (P2a), its inbox and its
learned values. `M4.B.3` (no look-ahead) is enforced by the import guard of §3 E-row.

### D4. The learner (`M4.D.6`, as reversed)

**Recommendation.**

- **Signal:** the actor's acute-anxiety change over the horizon `H` ticks after the act, negated, plus
  `M4.D.6e`'s cross-person term (the change in the target's and witnesses' anxiety, weighted by an `[I]`
  factor). `H` is short: 1–4 weekly ticks (`M4.D.6b` as amended, "weeks, not months").
- **Credit:** each act within the last `H` ticks receives the signal, discounted by age at an `[I]` rate.
  Nothing outside `H` is credited (`M4.B.3`).
- **Update:** value ← value + α (signal − value), α `[I]`. This is a declared update form; TODO's note on
  Redish TD1 (the "converges by construction" argument assumed an additive form the spec never stated) is
  answered by stating it here.
- **Habituation** (`M4.G.3`, now binding): the relief credited to a repeated identical act on the same tie
  decays geometrically with the repetition count, `[I]`.

Learning rate α, horizon `H` and temperature are swept in Phase C (D9), because `M4.D.6b` names them as the
learner's three constants.

### D5. Move physics

Each act's effect on state is set, declared per kind in config: its contact and impingement components (D2),
and these specific effects:

| Act | Effect, beyond its components |
|---|---|
| `TRIANGLE` | moves tension from the pair to the third (`M1.C.1`); the outsider also generates positional anxiety (`KS03.1`, P1) |
| `DISTANCE` | binds anxiety into the tie (`M1.D.2a`); lowers contact |
| `CUTOFF` | sets the tie non-interactive (Phase B's path); impingement removed at once; contact decays (D2) |
| `OVERFUNCTION` / `UNDERFUNCTION` | pushes `functioning_balance` toward a pole; the flip is relative and immediate (`M1.B.5`–`M1.B.7`); conserved pseudo-self transfer (`M6.I.4`) |
| `PURSUE` | raises contact for the target and impingement by the sender's outward `outside_ness` |
| `CONFLICT` | both components (`KS04.13`) |
| `STAY-IN-CONTACT` | contact without impingement; offsets deterioration where built |
| `REDUCE_CUTOFF` | restores interactivity with a floor; cost to the third party whose distancing is stripped (P9, `L22.6`) |
| `I-POSITION` | the `M5.D` state machine; `M1.C.5`'s permanent decrement on completion |

### D6. Pattern readouts (`M5.A.1a`)

`M5.A.1a` requires each pattern readout's definition declared before a criterion reads it. **Recommendation,
all `[I]`, over a rolling window W (`[I]`) of the tie's or triangle's acts:**

| Pattern | Definition |
|---|---|
| conflict | both members emit `CONFLICT` toward each other, each at least k times in W |
| over/underfunctioning | `functioning_balance` held at a pole for W, with the pole's sides matching who emits `OVERFUNCTION` and `UNDERFUNCTION` |
| distance | `DISTANCE` acts on the tie exceed approach acts, and `felt_contact` below the optimum for both, over W |
| cutoff | the tie non-interactive for W |
| fixed triangle | the same outsider in the same triangle across ≥ m consecutive activations |
| projection | Phase D (it needs a child-focused readout) |

`docs/readouts_phase_c.md` holds the definitions. A test fails if a criterion reads a pattern not declared
there.

### D7. Ensemble tests, outside the default suite

Criteria run over ensembles (`M11.2`), adaptively (`M13.4`). They cannot run in the default `pytest` pass
(§7). **Recommendation:**

- An ensemble runner in `src/bowen/ensemble/`, outside the engine package, that runs paired arms on keyed
  seeds, in parallel processes.
- Criterion tests carry a `pytest` marker `ensemble` and are excluded from the default run by `pytest.ini`.
  `python3 -m pytest -m ensemble` runs them. A committed record, `docs/phase_c_ensemble_record.md`,
  generated by a tool, holds each verdict with seeds used, the interval and the margin; a default-suite test
  fails if the record is older than the code or constants it ran against (by hash).
- CI runs the default suite on every push. The ensemble suite runs on demand; the record test in the default
  suite is what CI checks.

### D8. The paired statistic (`M11.4e`, `M17.A.3`)

**Recommendation:** Wilcoxon signed-rank on the per-seed differences, implemented in NumPy (no new
dependency), with a test against published table values. Multiplicity: Holm across the readouts of one
criterion. Margins (`M17.A.4`) are relative to the seed-to-seed spread of the baseline arm, as the review
recommends (E), so they are unit-free. `UNDETERMINED` at the cap does not pass (`M13.4`).

### D9. The constant sweep in Phase C (review A2)

**Recommendation.** For each **composite** criterion, run the criterion at three settings (low, central,
high, each declared) of each of: the learner's α, `H` and temperature, and the criterion's dominant constant
(named in its row, §3). The gate requires the direction to hold at the central setting. The other settings are
reported in the ensemble record, and a direction that reverses at any setting is flagged in the verdict. A
premise criterion is run at the central setting only. Phase E's full sweep (`M17.E.1`) is unchanged.

### D10. Phase B code that revision 11 replaced

Reworked first (§6 step 1): `appraise_base.py` (→ D2), `standing_load.py` (→ the "too little" term),
`recompute.py`'s `tension_activation_threshold` and its test (→ triangle activity as a readout of recent
`TRIANGLE` acts), and the config rows citing it. `spec_coverage_phase_b.json`'s three `partial` overrides
are removed when the reworked tests pass.

---

## 3. The Phase C gate

After P3. Each criterion test is ensemble-marked (D7), runs both arms under a named stressor where `M11.3`
requires it, and is proved failing by its mutation. `M11.1c`'s representation-invariance mutants and
`M11.1d`'s sign-inverted mutants run for every row (§6 step 15). Class is from `M11.5`.

| ID | Class | Family · arms · stressor | Test | Mutation that must turn it red |
|---|---|---|---|---|
| M11.C.1 | Premise | Phase C instance; lower vs higher `basic_level`; declared spell | `test_m11c1_lower_c_reaches_threshold_sooner` | `M4.C.1a`'s steepness made independent of `functional_level` |
| M11.C.3 | Premise | triad; a `TRIANGLE` vs none at one tick | `test_m11c3_triangle_relieves_pair_costs_third` | `M1.C.1`'s transfer removed |
| M11.C.4 | Premise | a dyad fixture modelled on `M2.A.1`'s pair (comparable contact, different bond energy); `CUTOFF` vs none; scripted nodal event | `test_m11c4_cutoff_trades_now_against_later` | `M4.C.1c`'s accrual removed (contact does not decay on a severed tie) |
| M11.C.5 | Composite | Phase C instance; a scripted `I-POSITION` sequence vs none | `test_m11c5_change_back_reaction_shape` | `M5.E.7`'s life-energy debit removed |
| M11.C.7 | Premise | Phase C instance with Dr Halim; two topologies, coach constants fixed | `test_m11c7_topology_not_coach_skill` | `M8.2`/`M8.3` predicate made position-blind |
| M11.C.13 | Premise | Phase C instance; help vs none | `test_m11c13_help_relocates_not_reduces` | `M6.I.6`'s transfer made a sink (help destroys anxiety) |
| M11.C.14 | Premise (null) | nuclear four; technique vs none, marital distance high | `test_m11c14_technique_null_under_marital_distance` | `M5.C.1`'s marital-distance gate removed. Carries `M11.4a`'s equivalence bound and power |
| M11.C.16 | Composite | Phase C instance; lower vs higher `basic_level`; spell | `test_m11c16_repertoire_concentration_depends_on_level` | `M4.D.6` disabled in both arms |
| M11.C.19 | Check | dyad; forceful declarer vs compliant accommodator | `test_m11c19_counterfeit_axis_is_identified` | `M5.F.2b`'s two axes collapsed to their mean |
| M11.C.25 | Premise (null) | dyads, sex drawn per seed | `test_m11c25_dominant_pole_independent_of_sex` | a sex term added to pole assignment. Carries `M11.4a`'s bound and power |
| M11.C.27 | Composite | triad; four cells | `test_m11c27_twosome_two_by_two_all_cells` | `M4.C.1` made one-sided |
| M11.C.29 | Premise | triad; genuine `I-POSITION` vs distance-in-disguise | `test_m11c29_relief_and_differentiation_differ_in_time_course` | `M6.I.1`'s budget reduction made a reallocation |
| M11.C.32 | Premise | dyad; angry vs calm mover | `test_m11c32_mover_anger_stalls_and_degrades` | `M5.D.4`'s gate inverted |
| M11.C.35 | Check | triad; C–A conductance high vs low | `test_m11c35_witness_appraisal_depends_on_both_ties` | witness appraisal replaced by a fidelity-scaled copy |
| M11.C.38 | Premise | Phase C instance; four `basic_level` levels | `test_m11c38_graded_parameter_orders_primary_readout` | `M4.C.1a`'s steepness made level-independent |
| M11.C.41 | Mixed | Phase C instance; 2×2 of level and stress | `test_m11c41_level_and_stress_each_exacerbate_the_pattern` | `M4.C.1a`'s steepness, or `M4.D.1a`'s mixing weight, made level-independent |
| M11.C.42 | Composite | triad; third member available vs absent at one tick; spell | `test_m11c42_relieving_triangle_is_reused` | `M4.D.6` disabled in both arms |
| M11.C.44 | Composite | triad; calm vs spell | `test_m11c44_position_value_inverts_with_load` | `M4.C.1` made one-sided |
| M11.C.45 | Composite | triad; calm vs spell | `test_m11c45_triangles_quiet_when_calm` | `M1.C.1`'s relief made independent of the pair's tension |

**Moved to Phase D by P3:** C.17, C.18, C.20.

**Engineering and log criteria:**

| ID | Test | Mutation |
|---|---|---|
| M11.D.2 | `test_m11d2_no_magic_literals_in_engine` — AST scan of `src/bowen/engine/` and `src/bowen/policy/` for numeric literals outside an allow-list (0, 1, −1, indices) | add `0.3` to the policy |
| M11.D.4 | `test_m11d4_invented_constants_labelled` (built in Phase B, not gated there) | relabel one `[I]` constant `[T]` |
| M11.D.8 | `test_m11d8_spec_references_resolve` — every `Spec:` reference resolves; **exact count** (`M11.D.9`) | add a docstring citing `M99.Z.1`; or narrow the scan to one package |
| M11.D.15 | `test_m11d15_placebo_arm_is_byte_identical` | let a zero-magnitude mechanism consume an unkeyed draw |
| M11.D.16 | `test_m11d16_batch_order_permutation_is_invariant` (byte for byte); `test_m11d16_symmetric_family_is_permutation_equivariant` (in distribution, P7) | replace the batch reduction with a sequential one |
| M11.D.18 | `test_m11d18_fallback_rate_is_reported_and_flagged` | drop the fallback count from the ensemble record |
| M11.D.19 | `test_m11d19_every_move_is_reachable_in_a_triad` — dense sampling of the triad's state box; reports "not observed", never "unreachable" | gate one act out everywhere |
| M11.D.21 | `test_m11d21_state_bounds_are_not_absorbing` — start each bounded state at its bound, assert it leaves | clamp `felt_contact` at 0 once it reaches 0 |
| M16.T.3 | `test_m16t3_sink_does_not_change_results` — rerun now that a policy exists | let the sink mutate a record the learner reads |
| M16.T.4 | `test_m16t4_delayed_view_is_scoped_and_lagged` | return events newer than the delay |
| M16.T.6 | `test_m16t6_event_store_is_load_bearing` | the store emptied by an override, and the two runs compared |

**The `M4.B.3` guard** (no look-ahead, revision 11): `test_m4b3_policy_reads_no_future_quantity` — a static
check that the policy package reads no attribute indexed by future time and imports nothing from the ensemble
runner or readouts. Mutation: give the policy a call to a readout.

---

## 4. Other requirements Phase C builds

Not gate criteria, but `M13` places them in Phase C. Each carries a unit test named for its ID.

| Area | IDs | Test (summary) |
|---|---|---|
| Appraisal | `M4.C.1`–`.1c` (D2), `M4.C.2` gain, `M4.C.3`/`.3a` integrator, `M4.C.4` perspective falls with anxiety, `M4.C.5` inward readout, `M4.C.6` self-attribution, `M4.C.7` route on both sides, `M4.C.8`/`.8a` attention amplifies unless objective, `M4.C.9` witness, `M4.C.10` calmer contact | one direction per ID, e.g. `test_m4c2_content_defended_above_threshold` |
| Policy | `M4.D.1`–`.1f`, `M4.D.2`, `M4.D.3`(amended), `M4.D.3a`(capacity), `M4.D.3b`, `M4.D.4`, `M4.D.5`–`.5d`, `M4.D.6`–`.6e`, `M4.B.3`, `M4.G.3` | e.g. `test_m4d1b_withhold_changes_tie_state`, `test_m4d4_iposition_not_monotone_in_level`, `test_m4d6d_self_directed_channel_never_reinforced` |
| Moves | `M5.A.1`, `M5.A.1a`, `M5.B.1`–`.5`, `M5.B.6` (P10), `M5.C.1`, `.1a`, `.2`, `M5.D.1`–`.8`, `M5.E.0`–`.9`, `M5.F.1`–`.5` | e.g. `test_m5d3_abort_is_the_usual_first_outcome`, `test_m5d8_success_usually_follows_failures`, `test_m5e0_ladder_is_not_scripted` (static: no rung sequence in code) |
| Objects written in C | `M1.A.9`/`.9a` (`outside_ness`), `M1.A.18`–`.18d`, `M1.A.19`, `M1.B.5`–`.10a`, `M1.C.5`, `M1.D.1`, `.3`, `.4a`, `M1.E.1`–`.8`, `M1.A.5b`'s functional togetherness | per register row (Appendix A) |
| Invariants | `M6.I.1`–`M6.I.8` asserted every tick, `M6.I.6` as restated (P1); `M6.1`, `M6.2` | `test_m6i6_transfers_balance_every_tick`, one per invariant |
| Symptoms (P3) | `M7.D.1` | `test_m7d1_threshold_crossing_emits_endogenous_event` |
| Belief (P2a) | `M9.8` store only | `test_m98_belief_written_only_from_delivered_events` *(built at step 4 under this name: the coverage tool maps `test_m98_` to `M9.8`, and the store is written from target as well as witness deliveries, as `M9.8` says)* |
| Log | `M16.D.1`, `M16.D.2` | `M16.T.4`, and the delayed-view arm `M16.D.2` asks for |
| Stopping | `M17.A.1`, `.3`, `.4`; `M11.4a`, `.4d`, `.4e` | `test_m17a1_undetermined_at_cap_does_not_pass`, `test_m114e_signed_rank_matches_table` |

**Not built in Phase C:** the slow-clock content (`M7.A`–`M7.C`, `M7.D.2`–`.4`, `M7.E`), the rest of `M9`,
`M11.G`, the twelve-person family, and everything `M13` puts in D or E.

---

## 5. Package layout

```
src/bowen/
  engine/          Phase B, reworked per D10; appraise.py replaces appraise_base.py
    contact.py       D2 — felt contact, impingement, optimum, the two-sided function
    moves.py         D5 — each act's physics
    triangles.py     M1.C, revision-11 activity readout
    iposition.py     M5.D state machine and M5.E success state
    invariants.py    extended: M6.I.1–M6.I.8, the P1 ledger
    symptoms.py      M7.D.1 (P3)
    external.py      M1.E
  policy/          D3, D4 — legal set, gates, channels, learner; imports engine state types only
  beliefs/         P2a — the M9.8 store. *Built at step 4 as `engine/beliefs.py` instead: the engine writes it at step 3, and the purity guard (`M11.D.22`) keeps the engine from importing outside itself*
  readouts/        D6 pattern readouts; M5.F.2b counterfeit detector; M1.A.4h channel readout
  ensemble/        D7, D8 — paired runs, keyed seeds, signed-rank, adaptive stopping, record
config/bowen/
  family_phase_c.md, fixtures/{dyad,triad,nuclear_four}.md, event_kinds.md (components), constants.md
tests/bowen/
  unit tests per area; test_phase_c_gate.py (ensemble-marked criteria); test_ensemble_record.py
tools/
  mutation_gate_phase_c.json; ensemble_record.py
docs/
  readouts_phase_c.md; phase_c_ensemble_record.md; phase_c_mutation_record.md; phase_c_completion_report.md
```

`M11.D.22`'s guard extends to `policy/`: it may not import `readouts/`, `ensemble/` or tests.

---

## 6. Build order

Each step ends with the default suite green. Ensemble criteria are written at step 14, after the constants
freeze, so no constant is tuned against a criterion.

| Step | Work | Depends on |
|---|---|---|
| 0 | P1–P12 decided; revision 12 applied; explainer corrected (P5, P6); Phase B closed (P11) | — |
| 1 | D10 rework: contact state (D2), the two-sided appraisal and standing load, triangle activity as a readout; Phase B tests and G3 updated; register rows | 0 |
| 2 | Appraisal: `M4.C.2`–`.10`, `M1.A.19` reactive state, `M1.B.8` investment, `M4.C.3` integrator and `M7.D.1` | 1 |
| 3 | `outside_ness` (two axes), act identity `M5.F`, counterfeit detector | 2 |
| 4 | Belief store (P2a) | 2 |
| 5 | Move physics (D5), including pole flip, distance binding, `REDUCE_CUTOFF`, `M5.B` moves, `PROVOKE` | 3 |
| 6 | Policy skeleton (D3): legal set, gates, fallback, `WITHHOLD`, channels, competing urges, keyed draws; `M4.B.3` guard | 4, 5 |
| 7 | Learner (D4), habituation | 6 |
| 8 | `I-POSITION` state machine, `M1.C.5`, `M5.E` success state (ladder learned, `M5.E.0`) | 7 |
| 9 | External agent `M1.E`, landed contact, `systems_perspective`; `M16.D` delayed view; `M16.T.4`, `.T.6` | 8 |
| 10 | Invariants `M6.I.1`–`.8` with P1's ledger; `M1.D.1`/`.3` sink allocation | 5 |
| 11 | Pattern readouts (D6) and `docs/readouts_phase_c.md` | 7 |
| 12 | Ensemble runner, statistic, adaptive stopping, record, `M11.D.18`; `pytest` marker; compute measured (§7) | 7 |
| 13 | Engineering gates `M11.D.2`, `.4`, `.8`, `.15`, `.16`, `.19`, `.21`; `M16.T.3` rerun | 6–12 |
| 14 | Constants frozen (`M10.B.4`); criterion tests written; ensemble run; record committed | 13 |
| 15 | Mutation proofs: each row's mutation; `M11.1c` representation mutants and `M11.1d` sign-inverted mutants for every criterion; `M11.1b` severing mutants where cheap | 14 |
| 16 | D9 sweep for composites; `M11.5` corrected from the real mutants; coverage; completion report; `learning-qa` | 15 |

---

## 7. Compute budget

**Measured 2026-10-07:** a 40-week scripted Phase B run, seven people, takes about 0.4 s wall time
end to end (`python3 -m src.bowen.run --seed 7`, process start included). Phase C adds a policy and a learner
per person per tick; the cost per tick is not yet known.

**Estimate, to be replaced by a measurement at step 12.** If a 104-week Phase C run costs 1 s, one criterion
at a seed cap of 500 is 1,000 runs, about 17 minutes on one core. 19 criteria at the central setting is about
5 hours on one core; D9's sweep multiplies the composites by up to 12 settings. On the machine's cores in
parallel, the central run is about an hour. **This is why the ensemble suite sits outside the default run
(D7).** If step 12 measures a cost much higher than 1 s per run, the plan comes back to the owner before
step 14, with the cap and the sweep as the levers.

---

## 8. The external review's findings, by disposition

| Review | Finding | Disposition in this plan |
|---|---|---|
| A1 | Criteria verify rather than test | Done at revision 11 (`M11.5`); §3 carries the class; step 16 corrects it from real mutants |
| A2 | Cancellation premise; no sweep before Phase E | P5, D9 |
| A3 | `M6.I.6` unsatisfiable | P1 |
| A4 | `M4.B.2` conflicts | P2, P2a |
| A5 | Phase C needs Phase D machinery | P3 |
| A6 | Death and birth | Phase D (unchanged) |
| A7 | Sub-week items | P4 |
| B | Contradictions | P9 for Phase C's; the rest at Phase D |
| C | `M2` data | Phase D (the twelve-person family) |
| D | Explainer drift | P6 |
| E | `M11.D.16`; margins; budget | P7, D8, §7 |
| F | Document form | Not a phase blocker |
| G | Chapter fidelity | P10 for Phase C's; Bowen's falsifiers and `L15.1` recorded for Phase D |

---

## 9. What cannot be made code-testable in Phase C

| Item | Why | Proposal |
|---|---|---|
| Whether the pattern readouts (D6) **capture the patterns Bowen named** | a definition can be applied correctly and still name the wrong thing | **Human review** of `docs/readouts_phase_c.md` before step 14, and of one rendered trace per pattern |
| Whether learned behaviour **reads as Bowen's account** | criteria test directions, not plausibility | **Human review** of a rendered 104-week trace of the Phase C instance, as in Phase B |
| `M11.E`'s `M5.F.2` threshold | the counterfeit threshold sets the result | test the sign flip across it; human review of the threshold, as `M11.E` says |
| `M11.C.11`'s curve shape | Phase D | unchanged |
| Whether a premise passing **says anything about families** | by definition it does not | reports carry `M11.5`'s class beside every verdict |

---

## 10. Adjacent issues found, not fixed

- **The default suite will not exercise the criteria** (D7). A record test closes the gap partially: it shows
  the record is current, not that it is right. The ensemble suite should run before each phase gate is
  declared and before any spec revision that touches a criterion.
- **`M11.1c` (representation-invariance mutants) is a MUST for every criterion** and has not been costed. It
  multiplies step 15's work by the number of re-encodings. Recommendation at step 15: three re-encodings
  (rescaled state range, float summation order, integer vs float tick counter), reported per criterion.
- **`M1.D.1`'s three-sink budget** writes where anxiety goes; recorded for revision 12 at revision 11, and P1's
  ledger leans on it. If revision 12 takes up the sink question, P1's table changes.
- **The five rule groups revision 11 marked for review** (`M1.A.14c`/`.14d`, `M1.B.10`, `M4.D.5a`,
  `M7.D.2a`/`.2d`, `M9.6`) stand as written. `M1.B.10` (taboo set) and `M4.D.5a` (accommodation stock) are built
  in Phase C as written.
- **`requirements.txt`** lists `pandas`, `networkx`, `pygame-ce` and `matplotlib`, and `numpy` twice; nothing in
  `src/bowen/` uses the first four. Not a Phase C change.

---

## Appendix A — register rows Phase C adds or changes (`M14.A`)

New state, all derived (DV):

| Variable | Owner | Written by | Phase |
|---|---|---|---|
| `felt_contact` (per person, per tie) | Relationship | delivery (D2); decay, step 9 | C |
| `felt_impingement` (per person, per tie) | Relationship | delivery (D2); decay, step 9 | C |
| `contact_optimum` (per person, per tie) | Relationship | derived each tick from `bond_energy`, `functional_level`, `acute_anxiety` (`M4.C.1b`) | C |
| `tie_bound_anxiety` | Relationship | `DISTANCE` (`M1.D.2a`); `RECONCILIATION` (`M4.A.3`) | C |
| `bound_anxiety` | Triangle | `TRIANGLE` transfer (`M1.C.1`) | C |
| `learned_values` | Person | learner (`M4.D.6`), step 9 | C |
| `iposition_state` | Person | `M5.D` state machine | C |
| `chronicity_integral` | Person | `M4.C.3a` integrator, step 4 | C |
| `tie_beliefs` | Person | witnessed events (`M9.8`, P2a) | C |

Changed: `acute_anxiety`'s writers (the standing load and base appraisal become the two-sided appraisal);
`Triangle.active` (a readout of recent `TRIANGLE` acts, not a threshold). New mechanisms: policy selection
(replaces scripted selection for active persons), learner, move physics, `I-POSITION` steps, external-agent
contact, invariants ledger. Each is placed in `M3.D.1`'s order and checked by `M11.D.17`.
