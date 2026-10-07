---
status: APPROVED 2026-10-06 and applied as spec revision 11 — this file is the working record
date: 2026-10-06
against: docs/bowen_agent_model_spec_v2.md, revision 10 (approved 2026-10-06)
decided: A1 first; then A2 with A2.1, A3 option (a), A4, A5, A6 — owner, 2026-10-06
superseded_by: the spec's revision-11 section, M11.5 (classification) and model_explainer.md §19
---

# Revision 11 proposal — what is set, what is random, what emerges

> **Applied 2026-10-06.** Where this draft and the spec differ, the spec wins. Three corrections were made while
> applying it, listed in the spec's revision-11 section: `M11.C.41`'s reactive-share limb stays a premise;
> `M7.E.1a` and `M7.E.1e` changed with `M7.E.1`; `M4.D.6a` cites `KS04.13`, not the societal `K10.7`. The
> criteria added in §4 as C.42 and C.43 became `M11.C.42`–`M11.C.45` in the spec; §4's counts are superseded by
> `M11.5`.

## 0. Why this exists

The owner's concern, 2026-10-06: if the model is told how families behave, it will behave that way, and
running it tells us nothing. The spec already describes the model as a "consistency engine for one theory"
(`model_explainer.md` §1) that "cannot show that real families behave that way". This proposal goes
further. It asks which of the model's results come from mechanisms that do not name them. Only those
results can be findings.

The spec already has a tool for this: `M11.1d`'s outcome-directive audit and `M10.C.5`'s premise column.
Neither has been run. §4 below is that audit, done **on paper**. It is provisional until the mutants are
actually run (`M11.1`).

---

## 1. The design principle (owner, 2026-10-06)

Three layers, plus one rule about what agents know.

**Layer 1 — what simply *is* part of being human (set).** Every person wants connection: to feel
connected, appreciated and loved. Every person also does not want to be smothered, bossed, threatened, or
to lose their autonomy. Both hold at once. Except at the highest levels of differentiation, a person is
pulled toward relationship and pushed away from too much of it. These drives are inputs. They are not
learned and they are the same for everyone. The person's level sets how strongly the drives act and how
much deviation the person can tolerate.

**Layer 2 — actions that manage the balance (set as actions, chosen by learning).** Approach, withdraw,
push back, give way, take charge, turn to a third person, hold a position, stay in contact. What each action
*does* to the state of the system is set. *When* a person chooses it is learned.

**Layer 3 — the patterns Bowen recorded (emerge).** Conflict, over- and underfunctioning, distance, cutoff,
projection onto a child, fixed triangles. These are not chosen as patterns. They are what repeated choices
produce over time, together with the consequences those choices create.

**The knowledge rule.** Agents do not know any of this. An agent does not know that distance relieves now
and costs later, or that projection calms the parent and then produces trouble in the child. It feels its
own state change and repeats what made it feel better. A cost that arrives later is felt as anxiety. It is
not traced back to the act that caused it, unless it arrives inside the learning horizon. A second round of
learning comes from those consequences: the new anxiety drives new choices, and so on.

**Examples the principle has to produce without a rule written for them:**

| Action | Relief it gives | Tension it creates later | Where the later tension comes from |
|---|---|---|---|
| `DISTANCE` / `CUTOFF` | less tension on the tie, at once | anxiety from **lack** of the relationship | Layer 1: the connection drive is not met. `M4.A.1`'s standing load already partly says this |
| Projection (a triangle with a child) | the parents calm down | the child's functioning falls; over time, conflict with the child | the child is learning (`M4.D.6e`) and has its own Layer 1 drives |
| `TRIANGLE` | the pair calms down | the outsider's anxiety, plus the outsider's attempts to get back in | `KS03.1`, `KS03.4` |
| Pursuit / closeness under anxiety | the pursuer's fear of rejection eases | the other's "too close" side is triggered, and they withdraw | Layer 1, two-sided |

**Support in the corpus.** This is close to what Kerr writes. All four entries below are Kerr 2019:
attribution `[K]`, evidence `[T]`. Whether Kerr is reporting Bowen in each place has not been checked.

- `KS03.2` — "human beings have a **profound need for emotional closeness but are adverse to too much of it**… Threats of too much distance **and** too much closeness activate the stress response."
- `KS06.3` — threats to adequate contact and to sufficient distance "appear to be the **fundamental triggers of anxiety**". The extraction's own note says "`M4.C.1` should be re-derived from this". **That was never done.** `M4.C.1` still raises anxiety from event intensity.
- `KS04.14` — "**Each of the three patterns solves the dilemma** posed by the intense need for closeness and equally intense aversion to too much of it. Each pattern solves it in a different way." This puts the patterns in Layer 3: solutions that substitute for one another, not primitives.
- `KS06.5` — "automatic intensification of the togetherness force is the default mode" as anxiety rises.

---

## 2. Proposed amendments

### A1 — Reverse `M4.D.6a`: the automatic channel learns from felt relief  **(decided)**

**Replace `M4.D.6a` with:**

> **M4.D.6a** — *the automatic channel's reinforcement signal **MUST** be the actor's felt change in
> anxiety over a short declared horizon.* The signal is the change in the actor's own acute anxiety,
> together with `M4.D.6e`'s cross-person term, measured over the horizon of `M4.D.6b`. It is short by
> design. The automatic channel's objective is to discharge anxiety now (`M4.D.1a`). "Cause and effect laws
> designed to relieve the anxiety of the moment, and the more we do that, the more we promote the thing we're
> trying to fix" (`kb/kb10.md` · K10.7) describes learning of this kind. Whether this learning drives a
> family's repertoire onto `CUTOFF`, or onto any single move, is a result to be measured (`M11.C.16`). No
> rule prevents it.
> ⟦rev11 · owner decision 2026-10-06; reverses the rev-2 form approved as G3, 2026-08-24⟧

**Consequential changes:**

| Requirement | Change |
|---|---|
| `M4.D.6b` | Keep "declared in config with its horizon, `[I]`". Delete "longer than the reaction window of M5.D". The horizon still belongs to `functional_level`'s timescale. `M17.E.1` **MUST** sweep it |
| `M4.D.6c` | Rewrite. A degenerate policy is no longer a failure to guard against. It is a possible result, and `M11.C.16` reports it |
| `M4.D.6d` | **Unchanged.** Only the automatic channel learns. `I-POSITION` and `STAY-IN-CONTACT` are in the self-directed channel, so short-horizon learning cannot punish them out of existence. That was `M4.D.6a`'s original worry, and `M4.D.6d` already prevents it |
| `M4.D.1a` | Keep the two-channel design. Delete the flagged "Consequence" paragraph's reliance on a long horizon. Differentiation is still not reachable by the automatic learner, because the target is not in that channel's objective |
| `M4.D.5c` | Now consistent with `M4.D.6a`. No change |
| `M4.G.3` | Its condition ("if any `M4` term credits anxiety relief to a repeated identical move") is now **met**. The habituation `SHOULD`, or the bistability sweep, now applies. Move it from the open questions to a live requirement |
| `M11.C.16` | Restate. See §4, C.16 |

**Risk.** The repertoire might collapse because of a property of the learner (learning rate, exploration
temperature), not because of any family process. `M17.E.1`'s sweep and `M17.D.2`'s rival arm are how to
tell the two apart. Both learning constants **MUST** be in the sweep.

### A2 — Two-sided appraisal: anxiety from deviation in either direction  **(proposed)**

Re-derive `M4.C.1` from `KS03.2` and `KS06.3`, as the extraction recommended.

> **M4.C.1** — *anxiety on a tie **MUST** rise with the tie's deviation from the person's felt contact
> optimum, in either direction.* Each person, on each tie, has a felt optimum between too little contact
> (rejection, isolation) and too much (intrusion, control, smothering). Acute anxiety rises with the
> distance from that optimum on **either** side. The person's level sets **how steeply** anxiety rises with
> deviation (lower level, steeper) and **how wide** the tolerated band is (lower level, narrower). Rising
> anxiety moves the optimum toward closeness (`KS06.5`). Incoming events change the tie's state: an event
> addressed to the person moves the state, and the deviation is what is appraised. `route`, `source_position`
> and `fidelity` keep their present roles. Both drives are Layer 1 and are never learned.

**What changes as a result:**

- **`CUTOFF` creates its own anxiety.** No extra rule is needed. Severing the tie moves it to the far
  "too little" side. This replaces part of what `M4.A.1`'s standing load now does by a separate rule.
  **Open question A2.1:** fold the standing load into the "too little" side of this function, so that one
  rule does both jobs, or keep both rules. I recommend folding them together. `M3.D.2`'s reason, that
  severing a tie must not register as relief, is then met by the function's shape.
- **Conflict touches both sides at once.** "Poking at the other gets a reaction and pushes the other away at
  the same time" (`KS04.13`). Under A2 this follows directly, and it gives `M1.A.9a`'s two impingement axes
  a source.
- **The prior-state term behind `M11.C.27` could be deleted.** A stable pair sits near its optimum, and
  adding a third person pulls one member away from it. An unstable pair sits far from its optimum, and a
  third person takes up the excess. The 2×2 result would then be composite. See §4.

Grades: the two-sided shape is `[T]` (`KS03.2`, `KS06.3`, `[K]`). The link to level is `[D]` (`KS03.2`:
"well-differentiated people manage the closeness-distance dilemma more effectively"). Every functional form
and constant is `[I]`.

### A3 — Acts, not patterns, in the move list  **(decision needed)**

`M5.A.1`'s moves include `CONFLICT`, `OVERFUNCTION`, `UNDERFUNCTION` and `DISTANCE`. Those are the names of
Bowen's patterns. Under §1 the patterns are Layer 3, so choosing one should not be an atomic act.

- **Option (a), recommended.** Keep the names and redefine each as a **single act**: one critical push, one
  instance of taking charge, one instance of giving way, one withdrawal. The **pattern** is a readout over the
  tie's history: sustained, reciprocal acts that `M4.G.1` hardens. Over/underfunctioning is then two people
  each learning their side of a reciprocity (`M1.B.5`'s pole, `M6.I.4`'s transfer). Small diff, same effect.
- **Option (b).** Replace `M5.A.1` with an explicit act vocabulary (approach, withdraw, push, give way, take
  charge, turn to a third, hold position, stay in contact) and derive every pattern name as a readout. This is
  cleaner, but `M5.A.2` requires a citation for every new act, and it touches every gate in `M5.C`.

Projection is already not a move, which is consistent with §1.

### A4 — Remove rules that write an outcome; keep them as tests  **(proposed; one decision per row)**

These rules write a Layer 3 result directly. Under §1 each becomes an `M11.C` assertion, and the rule
is deleted or reduced to its physics. The aim is that a criterion stops being a premise (§4).

| Rule | What it writes now | What would replace it | Criterion that becomes composite |
|---|---|---|---|
| `M1.C.2` | outside position unfavoured when calm, favoured under tension | nothing. A2 makes the inside position costly under tension and preferable when calm; learning does the rest | new: assert the inversion |
| `M1.C.3` (2nd sentence) | triangles inoperative when calm | nothing. With no tension there is no relief to learn from | new: assert low triangle use in calm arms |
| `M1.C.4` | tension reroutes onto previously-used circuits | `M4.D.6` per-triangle learned value | new: a triangle once relieving is reused (the owner's example) |
| `M4.D.3` | rising anxiety raises the weight on reactive moves | learned value: reactive moves pay off more when anxiety is high. **Keep `M4.D.3a`**, because it is a capacity limit (the newer layer is lost under load), not a preference | `M11.C.41`'s reactive-share limb |
| `M4.G.1` | three withdrawals register as a distant relationship | keep the physics (contact falls with each withdrawal); the *pattern* label is a readout | — |
| `M7.D.3` | curing a symptom without changing the deficit raises tension | nothing. Removing the symptom releases bound anxiety (`M6.I.1`); where it lands depends on learned patterns | `M11.C.12`, `M11.C.15` |
| `M7.E.3` | cutoff forms a positive feedback loop across generations | the child witnesses cutoff (`M1.A.7`), learns it (`M4.D.6`), and enters the next marriage with cutoff from the family of origin as an input (`M1.D.8`) | `M11.C.10` |
| `M7.E.1` | the primary projection object ends lower than the parents | the child's `basic_level` is set by the estimator (`M1.A.4a`) over the child's own functional history. Partly blocked: childhood is not simulated for generations 1–2 (`M2.A.0a`) | `M11.C.6` |
| `M7.E.1c` | the projection target is picked from a closed list of five situations | keep the five situations as **conditions that raise parental anxiety at the child's birth** (timing, defect, firstborn, sex, last-born). Who becomes the target then follows from where that anxiety is relieved (`M4.D.6e`) | `M11.C.2` |
| `M5.E.1`–`M5.E.6` | the change-back ladder and its damped-oscillation shape | keep `M5.E.7`'s cause (the mover withdraws energy the others were receiving). The others learn that pressure restores what they lost, and the ladder becomes their learned response | `M11.C.5` |

**Not proposed for removal.** These rules are Bowen's premises, entered deliberately as inputs:
`M4.C.1`'s divisor by `functional_level`, the triangle transfer `M1.C.1`, conservation `M6`, the dependence
gate `M7.A.1a`, the coach gate `M1.E.7`, and the sex-free pole rule `M2.A.0g`. Results that follow from
them are premises, and §4 labels them that way. That is acceptable, provided no report calls them findings.

**Adjacent issue found, not proposed here.** `M1.D.1`'s undifferentiation budget with exactly three sinks
is itself structure that writes where anxiety goes. Under §1, conflict, spouse dysfunction and child
projection are where learned patterns put anxiety, and the budget would be accounting over the result.
That is a larger change than this revision should carry. It is recorded for revision 12.

### A5 — Social buffering: learned, from one physics rule  **(proposed)**

"Social buffering" is not in the six corpus sources. It comes from the stress-physiology literature, and
any requirement citing it needs a source added to `_EXTERNAL_MEASURES.md` or a ledger entry first. The
closest thing in the spec is `M1.E.5`: an external agent's proximity carries a burden-transfer term with a
negative sign. That is a buffering rule written for the coach only.

Proposed: generalise the physics and let the behaviour be learned.

> **Physics (set).** Contact with a less anxious person on a connected tie lowers the receiver's acute
> anxiety, scaled by conductance and by the gap between them. The same rule already moves anxiety upward
> from a more anxious person. This makes it two-way. `[I]` in form and constants; source pending.
>
> **Behaviour (learned).** Approaching that person relieves anxiety, so `M4.D.6` reinforces it. Nothing
> names buffering as a move.

Under §1, a learned buffer is a person lending self (`M1.A.5d`; borrowing is "not a bad thing"). The
consequence is a composite criterion that extends `M11.C.31`: **the more a person has learned to rely on a
buffer, the more anxiety is released when the buffer becomes unavailable.** Two arms differing only in
learning history; no rule names the dependence.

### A6 — Agents do not know consequences  **(proposed)**

`M4.B.2` already limits what the policy may read to own state, own beliefs and delivered events. Add:

> **M4.B.3** — *the policy **MUST NOT** contain any representation of a move's consequences beyond the
> horizon of `M4.D.6b`.* No look-ahead, no model of the family, no term that values a move by its effect on
> anyone's later state. A deferred cost reaches the policy only as felt anxiety when it arrives. It is
> credited to whatever move fell inside the horizon at that time. `M11.D.22`'s import guard **SHOULD** be
> extended to fail if the policy module reads any quantity indexed by future time.

The self-directed channel is outside this rule. Its objective is `M5.F.5`'s position, not a forecast.

---

## 3. Rule classification — `M1`–`M9`

**Classes:**

- **S — set.** What a thing is or what an act does. Physics, fixed in every run.
- **G — gate.** A set capacity limit or legality condition.
- **R — random.** A seeded draw: exogenous events, selection noise, initial conditions, visibility.
- **P — written policy.** A rule that says when or how often someone chooses something.
- **O — written outcome.** A rule that writes a Layer 3 result directly.
- **D — readout or estimator.** Measures; does not drive.
- **E — engineering.** Identity, ordering, determinism, record fields; no behavioural content.

Under §1, **P** rules should become learned and **O** rules should become tests. **S, G, R, D, E** stay.
The last column gives the proposal's action.

### 3.1 `M1` — core objects

| IDs | Class | Note | Rev 11 |
|---|---|---|---|
| `M1.A.0`, `.1`, `.2`, `.3`, `.3a`–`.3d` | S | the scale and the intellect's licence as a slope | keep |
| `M1.A.4`, `.4a`–`.4j`, `.4f` | D | `basic_level` is an estimator over `functional_level` history | keep. This is what makes A4's `M7.E.1` change possible |
| `M1.A.5`, `.5a`, `.5b`, `.5c`, `.5d` | S | functional vs basic; relationship-level togetherness balance; pseudo-self sign | keep. `.5b` is where A2's optimum lives |
| `M1.A.6` | S | symptoms read `functional_level` | keep |
| `M1.A.7`, `.7a` | S | programmed reactivity from witnessed childhood; chronic anxiety derived | keep. This is the route by which a child learns from what it witnesses |
| `M1.A.8` | S | acute anxiety, decaying to the floor | keep |
| `M1.A.9`, `.9a` | S | two-axis `outside_ness` | keep. A2 gives the axes a source |
| `M1.A.10` | S | life energy, relationship vs goal | keep |
| `M1.A.11`, `.11a` | S | three substitutable symptom channels | keep |
| `M1.A.11b` | R | channel prior from constitutional data | keep |
| `M1.A.11c` | S | relational term moves the channel prior | keep |
| `M1.A.12` | D | membership as a threshold over involvement | keep |
| `M1.A.13`, `.13a` | D | structural importance, derived | keep |
| `M1.A.14` | S | birth order as static data | keep |
| `M1.A.14a`, `.14b` | D | functional sibling position derived from functioning | keep |
| `M1.A.14c`, `.14d` | P | sibling-position effects gated by level and suppressed by anxiety | **review.** Could become learned: position-typical acts pay off at mid-scale. `M11.C.9` depends on this |
| `M1.A.15` | G | financial dependence | keep |
| `M1.A.16`, `.17`, `.20`, `.21`, `.22` | E | beliefs field, role, identifiers, sex, household | keep |
| `M1.A.18`, `.18a`–`.18d` | S/D | systems perspective, two-sided readout, per-person | keep |
| `M1.A.19` | S | drifting reactive state with three detectors | keep |
| `M1.B.1`, `.13` | E | the tie as the unit; identifier | keep |
| `M1.B.2`, `.3`, `.4` | S | conductance, bond energy, near-zero decay | keep |
| `M1.B.5`, `.6`, `.7`, `.12` | S | bistable functioning balance; relative flip; asymmetric reversal; dyad age | keep. Over/underfunctioning emerges on top of these (A3) |
| `M1.B.8`, `.9` | S | valence-blind investment; areas of joint activity | keep |
| `M1.B.10`, `.10a` | P | taboo set grows by default | **review.** "Each party learns what makes the other anxious" is a learning claim written as a growth rule. Could be learned avoidance of topics |
| `M1.B.11` | S | latency | keep |
| `M1.C.1` | S | triangle holds pair, outsider, bound anxiety | keep (the transfer is physics) |
| `M1.C.2` | P | outside position value inverts with load | **A4 — learn** |
| `M1.C.3` | E + P | persistent topology stored (E); inoperative when calm (P) | **A4 — learn the second sentence** |
| `M1.C.3a` | S | routing capacity a function of members' level | keep |
| `M1.C.3b`, `.3c` | G/E | coach may not act on triangles; two detriangle operations | keep |
| `M1.C.4` | P | reroute onto previously-used circuits | **A4 — learn** (the owner's example) |
| `M1.C.5` | S | permanent intensity decrement after an I-position | keep |
| `M1.C.6` | S | sibling conflict instantiates the parental triangle | keep |
| `M1.C.7` | E | identifier | keep |
| `M1.D.1` | S/O | budget with three sinks | keep for now. **Revision 12** (§2, adjacent issue) |
| `M1.D.2`, `.2a` | S | distance binds anxiety into the tie; not a fourth sink | keep |
| `M1.D.3` | S | overflow to the families of origin | keep |
| `M1.D.4`, `.4a` | S/D | leadership office; differentiation capacity | keep |
| `M1.D.5` | S | tolerance for disturbing behaviour | keep |
| `M1.D.6` | S | access vector | keep |
| `M1.D.7`, `.7a`–`.7l` | R/S | societal dials as exogenous drivers; damping by level | keep |
| `M1.D.8` | S | at least one family-of-origin tie per adult | keep |
| `M1.E.1`–`.7f`, `.8` | S/G | external agent physics; `M1.E.7` gate on systems perspective | keep. `M1.E.5` is generalised by A5 |
| `M1.F.1`, `.1a`, `.1b` | E/R | event fields; channel field; witnesses computed by visibility | keep |
| `M1.F.2`, `.3`, `.4`, `.4a`, `.5` | S | source position, route, fidelity, register, witnessing | keep |
| `M1.F.6`, `.7` | R | exogenous spells; endogenous events never drawn from incidence | keep |
| `M1.F.8` | E | batched simultaneous events | keep |
| `M1.F.9` | S | `binder_unavailable` | keep; A5 extends its test |

### 3.2 `M2` — the reference family

| IDs | Class | Note | Rev 11 |
|---|---|---|---|
| `M2.1`, `M2.2`, `M2.3`, `M2.3a`, `M2.A.0`, `.0b`, `.0h`, `.0i`, `M2.B.1`, `M2.B.2` | E | declarations | keep |
| `M2.A.0a` | R/S | chronic anxiety for generations 1–2 supplied as an initial condition | keep. It limits A4's `M7.E.1` change |
| `M2.A.0c`, `.0e`, `.0f` | S | spouses matched on level; sibling-position complementarity | keep |
| `M2.A.0d` | S | fusion depends on life stage | keep |
| `M2.A.0g` | S | pole assignment independent of sex | keep (deliberate premise) |
| `M2.A.1` | E | fixture for `M11.C.4` | keep |
| `M2.A.2` | O-as-test | Nadia becomes the target **as an outcome** | keep. This is already the §1 form; A4's `M7.E.1c` change makes it true |

### 3.3 `M3` — clocks, ordering, determinism

All **E** (`M3.A.1`–`M3.E.2`), except `M3.D.2`, which is **S**: the standing load runs before events, so
severing a tie does not register as relief. Under A2 that becomes a consequence of the appraisal shape
(open question A2.1). `M3.E.2`'s synchronous activation is **R**/`[I]`.

### 3.4 `M4` — the fast tick

| IDs | Class | Note | Rev 11 |
|---|---|---|---|
| `M4.A.1` | S | standing load from every tie | **A2.1 — fold into the "too little" side** |
| `M4.A.2`–`.4` | S | trigger, reconciliation, institutionalise | keep |
| `M4.A.5` | S | self-generated stress from level | keep |
| `M4.B.1` | S | read addressed and witnessed events | keep |
| `M4.B.2` | G | what the policy may read | keep; A6 adds `M4.B.3` |
| `M4.C.1` | S | appraisal by intensity × conductance / level | **A2 — re-derive as two-sided deviation** |
| `M4.C.2`, `.3`, `.3a` | S | gain function; chronicity integrator; time above floor | keep |
| `M4.C.4`–`.9` | S | systems perspective falls with anxiety; attention amplifies; self-attribution; route on both sides; witness appraisal | keep |
| `M4.D.1` | R | softmax selection, the source of a first act "by chance" | keep. The temperature **MUST** be in `M17.E.1`'s sweep |
| `M4.D.1a` | S | two channels; mixing weight from level | keep; edit its Consequence paragraph (A1) |
| `M4.D.1b`, `.1c` | S | `WITHHOLD` distinct, insufficient alone | keep |
| `M4.D.1d` | S | competing urges raise anxiety | keep. Under A2 it follows from a person close to both edges of the band |
| `M4.D.1e`, `.1f` | G/E | legal set; tie-break and fallback | keep |
| `M4.D.2` | S | inputs to propensity, including learned repertoire | keep |
| `M4.D.3` | P | anxiety raises the weight on reactive moves | **A4 — learn** |
| `M4.D.3a` | G | complexity ordering; anxiety slides selection down it | keep, restated as a capacity limit |
| `M4.D.3b` | G | engagement with a loaded tie gated by systems perspective | keep |
| `M4.D.4` | P | I-position propensity not monotone in level | keep. It constrains the self-directed channel, which does not learn |
| `M4.D.5` | S | the fused default | keep |
| `M4.D.5a` | P | accommodation stock grows monotonically | **review.** Under A1 accommodation is a learned act that relieves now; the stock could be a readout of its history. Monotone growth would then be a result to test |
| `M4.D.5b`, `.5d` | S | reporting baseline drifts; emotional reserve gates onset | keep |
| `M4.D.5c` | S | short-horizon decision rule | keep; now consistent with A1 |
| `M4.D.6` | S | reinforcement exists | keep |
| `M4.D.6a` | S | the signal | **A1 — reversed** |
| `M4.D.6b`, `.6c` | S/E | horizon; rationale | **A1 — edit** |
| `M4.D.6d`, `.6e` | S | automatic channel only; cross-person signal | keep |
| `M4.E.1`, `.1a`, `M1.F.1a` | E | move to event | keep |
| `M4.G.1` | S + P | hardening (S); "registers as distant" label (P) | **A4 — keep physics, label as readout** |
| `M4.G.2`, `.2a` | E | invariants asserted | keep |
| `M4.G.3` | S | habituation on repeated relief | **A1 — now live** |

### 3.5 `M5` — moves

| IDs | Class | Note | Rev 11 |
|---|---|---|---|
| `M5.A.1`, `.2` | S | the repertoire | **A3** |
| `M5.B.1`–`.5` | S | added moves; external agent subset | keep |
| `M5.C.1`, `.1a`, `.2` | G | gates | keep |
| `M5.D.1`–`.8` | S | `I-POSITION` state machine | keep. The self-directed channel is not learned (`M4.D.6d`) |
| `M5.E.1`–`.6` | O | change-back ladder and its shape | **A4 — learned response to `M5.E.7`'s debit** |
| `M5.E.7`, `.7a` | S | the cause: life-energy debit | keep |
| `M5.E.8`, `.9` | S | success state; three timescales | keep |
| `M5.F.1`–`.5` | S | act identity; counterfeit sign; two-axis detector | keep |

### 3.6 `M6`–`M9`

| IDs | Class | Note | Rev 11 |
|---|---|---|---|
| `M6.1`–`.3` | S | conservation tolerance; disposition at death | keep |
| `M7.A.1`, `.1a`, `.2`, `.2a`, `.2b` | S/G | `basic_level` on the slow clock; dependence gate; transfer re-earned per tie | keep |
| `M7.B.1` | S | chronic anxiety fixed in childhood | keep |
| `M7.C.1`, `.1a`–`.1e` | S | life stage; investment redistribution on a new tie | keep |
| `M7.D.1`, `.2`, `.2b`, `.4` | S | symptom accumulation; substitution; removal effects by position; tolerance | keep |
| `M7.D.2a`, `.2d` | O | lock-in and its turn | **review.** Could emerge from learning: the family learns patterns that route anxiety to the symptomatic member, so removal disrupts them. Keep for now; flag `M11.C.22` |
| `M7.D.2c` | S | channels mutually protective | keep (premise) |
| `M7.D.3` | O | curing raises tension | **A4 — delete; test only** |
| `M7.E.1` | O | projection object lower | **A4** |
| `M7.E.1a`, `.1b`, `.1d`, `.1e` | S | three-outcome transmission; different triangles; signal-free initiation; marital-tie functioning | keep |
| `M7.E.1c` | P | target from a closed list | **A4 — conditions, not a selector** |
| `M7.E.2` | R | stochastic generational rate | keep |
| `M7.E.3` | O | cutoff feedback loop | **A4 — delete; test only** |
| `M7.E.4` | S | ties deteriorate by default | keep |
| `M8.1`–`.8` | S/D | live-position predicate; alignment | keep |
| `M9.1`–`.5`, `.7`, `.8` | S/E | belief store and channel | keep |
| `M9.6` | O | sink identifiable from belief configuration | **review.** As written, the attribution is written per sink, so `M11.C.26` is identifiability by construction. It would be composite if beliefs were formed from witnessed events and the attribution pattern emerged |

**Tally of the `P` and `O` rules this proposal acts on:** A1 changes 1 rule and edits 4. A2 re-derives 1 and
folds a second into it. A4 converts 10 rule groups. "Review" is marked on 5 more. Every other rule in
`M1`–`M9` stays.

---

## 4. Criterion classification — the 41 `M11.C` criteria

**Classes:**

- **Premise.** Deleting or inverting a single rule flips the result, and that rule states the result or
  nearly states it. Passing confirms the code matches the spec. It is not a finding.
- **Composite.** No single rule states the result. It needs several rules acting together, usually through
  learning or through time. Passing or failing is informative.
- **Check.** An instrument, conservation or architecture test. It verifies the machinery, not a claim about
  families.

**This is a classification by reading.** `M11.1d` requires the mutants to be run. A criterion marked
composite here can still turn out to be a premise when the mutants run.

| ID | Short | Now | Single rule it rests on now | Under rev 11 | What would make it composite |
|---|---|---|---|---|---|
| C.1 | lower level reaches symptom sooner | Premise | `/ functional_level` in `M4.C.1` | Premise | Nothing proposed. It is Bowen's central premise, entered as an input |
| C.2 | symptoms concentrate on the projection target | Premise | `M7.E.1c` selector | **Composite** | A4: target selection follows from where parental anxiety at birth is relieved (`M4.D.6e`) |
| C.3 | triangling relieves the pair, costs the third | Premise | `M1.C.1` transfer | Premise | — (physics of the act) |
| C.4 | cutoff relieves now, costs at the next nodal event | Premise | `M4.A.1` standing load | Premise | Under A2 the cost comes from the two-sided function, still one rule |
| C.5 | change-back reaction shape | Premise | `M5.E.1`–`.6` | **Composite** | A4: the ladder is the others' learned response to the debit |
| C.6 | multigenerational decline; spread widens | Premise | `M7.E.1` | **Composite (partial)** | A4: child level from the estimator. Limited by `M2.A.0a` for generations 1–2 |
| C.7 | position, not coach skill | Premise | `M8` predicate | Premise | — |
| C.8 | incidence near published rates | — | deferred to Phase E | — | It is the only criterion against outside data |
| C.9 | position effect peaks mid-scale | Premise | `M1.A.14c` gate | Premise (review) | `M1.A.14c` learned |
| C.10 | cutoff begets cutoff | Premise | `M7.E.3` | **Composite** | A4: witnessed and learned cutoff, plus family-of-origin cutoff as input to the next marriage |
| C.11 | removal: three phases; the return is larger | Mixed | `M6.I.6` (conservation limb) | Mixed | The return limb is already composite: no rule says the return is larger |
| C.12 | curing a symptom raises conflict | **Premise — the rule is the criterion** | `M7.D.3` | **Composite** | A4: delete `M7.D.3` |
| C.13 | help relocates incidents | Premise | `M6.I.6` | Premise | — |
| C.14 | technique null under marital distance | Premise | `M5.C.1` gate | Premise | — |
| C.15 | death destabilises like recovery | Premise | `M7.D.3` | **Composite** | A4: delete `M7.D.3` |
| C.16 | repertoire does not collapse onto relief-seeking | Guard on the learner | `M4.D.6a` | **Composite — restated** | Restate: *lower-level families concentrate their automatic repertoire more than higher-level families*. Report the collapse fraction per seed as a readout; it is no longer a pass/fail guard |
| C.17 | no drift in `basic_level` without a coach | Premise | `M1.E.7` gate | Premise | — (deliberate) |
| C.18 | estimator separates cutoff from I-position | Check | `M1.A.4b` | Check | — |
| C.19 | counterfeit axis identified | Check | `M5.F.2b` | Check | — |
| C.20 | estimator separates borrowed from basic | Check | `M1.A.4b` | Check | — |
| C.21 | reciprocity inverts when overfunctioners collapse | Premise | `M6.I.4` transfer | Premise | Possibly composite under A3(a), if the reciprocity is learned |
| C.22 | symptom lock-in is non-monotone | Premise | `M7.D.2a`, `.2d` | Premise (review) | `M7.D.2a` learned (§3.6) |
| C.23 | expressed emotion dominates medication | Premise | the EE term | Premise | — |
| C.24 | whisper of nature | Premise | `M6.I.4` transfer | Premise | as C.21 |
| C.25 | pole independent of sex | Premise (structural null) | `M2.A.0g` | Premise | — (deliberate) |
| C.26 | sink recoverable from beliefs | Premise | `M9.6` attribution write | Premise (review) | `M9.6` formed from witnessed events |
| C.27 | twosome 2×2, all four cells | Premise | the prior-state term | **Composite** | A2: stable pairs sit near the optimum, unstable pairs far from it; delete the prior-state term |
| C.28 | sink mobility is protective | Composite | arm input; outcome through the `M4.C.3` integrator | Composite | already |
| C.29 | relief vs differentiation by the third person's time course | Premise | `M6.I.1` budget reduction | Premise | — |
| C.30 | channels mutually protective | Premise | `M7.D.2c` | Premise | — |
| C.31 | binder removal releases anxiety | Check (conservation) | `M1.F.9` | Check + **new composite (A5)** | A5: learned reliance raises the released amount |
| C.32 | mover's anger stalls the move | Premise | `M5.D.4` gate | Premise | — |
| C.33 | dependence gate blocks `basic_level` rise | Premise | `M7.A.1a` | Premise | — |
| C.34 | system reaction to a symptom discriminates level | Premise | `M4.D.1a` mixing weight | Mixed | The share of automatic responses is premise; *which* automatic responses appear becomes learned under A1 |
| C.35 | witness appraises from its own position | Check | `M4.C.9` | Check | — |
| C.36 | policy reads belief, not truth | Check | `M4.B.2` | Check | — |
| C.37 | anxiety conserved across a death | Check | `M6.3` | Check | — |
| C.38 | graded parameter orders its readout | Premise | same divisor as C.1 | Premise | — |
| C.39 | a learned pattern outlasts the spell | **Composite** | none: "No persistence rule is written" | Composite | already |
| C.40 | reinforcement narrows the automatic channel only | Premise + check | the learning rule itself | Premise + check | — |
| C.41 | lower level and higher stress each worsen the pattern | Premise | divisor + `M4.D.3` | **Mixed** | A4: the reactive-share limb becomes composite when `M4.D.3` is learned; the anxiety limb stays premise |

**Counts, by reading:**

| | Now | Under rev 11 |
|---|---|---|
| Premise | 30 | 20 |
| Composite | 2 (C.28, C.39) | 10 (C.2, C.5, C.6, C.10, C.12, C.15, C.16, C.27, C.28, C.39) |
| Mixed | 1 (C.11) | 3 (C.11, C.34, C.41) |
| Check | 7 (C.18, C.19, C.20, C.31, C.35, C.36, C.37) | 7 |
| Deferred | 1 (C.8) | 1 |
| **Total** | **41** | **41**, plus two new composites (C.42, C.43 below) |

C.16 counts as a premise now, because its result rests on `M4.D.6a`. C.40 counts as a premise in both
columns. C.31 counts as a check in both; A5's extension of it is the new C.43.

**What this means for the owner's question.** As written, 2 of 41 criteria could tell us something we did
not put in. Revision 11 as proposed raises that to 10, plus 2 new ones, and 3 more become partly composite.
These are the criteria that bear on Bowen's patterns: projection, cutoff across generations, the reaction to
change, symptom relief. Premises stay
useful as checks that the code renders Bowen's inputs correctly. Reports **MUST** label them premises
(`M10.C.5`), never findings.

**Two new criteria proposed, both composite:**

- **C.42 — a relieving triangle is reused.** Two arms, identical seeds. In one, a first `TRIANGLE` happens
  to relieve the pair; in the other, the third person is unavailable at that tick. Later use of that
  triangle **MUST** be higher in the first arm. No rule names reuse (`M1.C.4` deleted under A4).
  Mutation: disable `M4.D.6` per-triangle learning.
- **C.43 — a learned buffer's removal costs more.** This is A5's extension of `M11.C.31`.

---

## 5. Order of work, if approved

1. Owner decisions on A2, A2.1, A3, each A4 row, A5 and A6.
2. Apply the approved amendments to the spec as revision 11, with ⟦rev11 · …⟧ markers.
3. Add a **premise / composite / check** column to the `M11` table, so that `M10.C.5` has its column.
4. Rewrite the Phase C plan. A2 changes `M4.C.1`, the first mechanism Phase C builds on.
5. Run the `M11.1d` audit as real mutants when each criterion's test is written, and correct §4 where the
   reading was wrong.

**What does not change:** two-arm differencing (`M0.4`), mutation proof, the constant sweep, the rival arm,
and `M11.F`'s framing. The model is still a consistency engine. The change is that more of what it reports
will be derived rather than entered.
