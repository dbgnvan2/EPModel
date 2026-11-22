# Proposal — conviction in holding an I-position, and the coach acting only in sessions

**Status:** draft, pending the owner's approval (2026-11-21). Nothing is built. Drafted by Claude from the owner's
decisions of 2026-11-21 on T-259, T-262 and T-265, checked against the corpus the same day. If approved, §2 and §3
are built together as one spec revision (67), so the records are regenerated once.

## 1. Why

- **T-259.** At revision 66 no `I-POSITION` sequence completes on the triad fixture (192 seeds, mover at level 60;
  27 at revision 30). Whether a mover holds is one comparison (`M5.D.3`, `iposition.py`): excess anxiety against
  `hold_gain × functional_level × efficacy`. T-264 found that revision 66 holds functional level far below basic
  level (about 27 against 40 on the triad), so the capacity to hold fell with it. The model has nothing for
  conviction: no requirement names it.
- **T-262.** A coach's `I-POSITION` sequence makes its `DEFINE` step at week 124, which is not a session week, so the
  coach acts outside a session.
- **T-265.** The LLM experiment's committed runs do not replay on revision 66's engine.

## 2. The coach acts only in sessions (T-262)

**The owner, 2026-11-21:** "the coach can't act outside of a session. they can continue the I position if need in
the next session."

**The corpus.** In coaching one family member, the sessions teach and supervise and the person does the work
between them; supervisory sessions run from every two weeks to every few months, depending on the work done between
appointments (*Family Therapy in Clinical Practice* Ch. 10 ¶111–112, [B,T]). Papero: progress comes from the
individual's own efforts (*Bowen Family Systems Theory* Ch. 4 ¶20, [T]).

**Rule (new, `M5.D.9b`).** An external agent's sequence step **MUST** fall in a session week. A step that is due
between sessions waits for the agent's next session; the wait is not a deferral under `M5.D.9` and does not end the
sequence. The windows counted in fast ticks (`opposition_window`, `hold_window`, `stall_limit`) count only the
agent's session ticks. A family member's sequence is unchanged.

## 3. Conviction (T-259)

### 3.1 The owner's account, 2026-11-21

- A person taking an I-position will often meet resistance. If they hold it, others usually come around, but not
  always: others may fight hard or distance, and with more reactivity the mover may give up. "It depends on how
  important the position is."
- Conviction is both A (it rests on the basic level) and B (each position has its own importance).
- "The lower level of differentiation, the lower level of conviction in general."
- "There are some things that a person will draw a line at. A burglar won't hurt a child; people don't steal, unless
  it's to save a life. It's very individual."
- "It also depends on the importance of the relationship — partners might lie to friends or children, for example."
- "More intense relationships — more subjectivity — lower level of differentiation — higher levels of stressors —
  less conviction."

### 3.2 The corpus

| Claim | Source | Grade |
|---|---|---|
| Differentiation starts when one member states their own convictions and acts on them; becoming sure may take months or longer. The family opposes in three steps ("you are wrong", "change back", consequences). Defending, counterattacking or going silent slips the mover back; staying on a calm course brings a peak, then a fall, and the family pulls up to the new level. | *FTCP* Ch. 18 ¶54–55 | [B,T] |
| The basic self is made of beliefs and convictions; it is not changed by coercion, pressure or the wish for approval. The pseudo-self is negotiable in the relationship system. | *FTCP* Ch. 21 ¶15; Ch. 10 ¶39 | [B,T] |
| Conviction rises with level: people at 50–75 have fairly well-defined beliefs on most essential issues, but under enough stress they can compromise principle and make feeling decisions rather than risk others' displeasure. People at 25–50 largely lack their own convictions. | *FTCP* Ch. 9 ¶37; Ch. 10 ¶42 | [B,T] |
| The pseudo-self is most vulnerable in emotionally intense relationships; a marriage is the example. As differentiation falls, subjectively based attitudes have more influence. | *FE* Ch. 4 ¶94–95; Ch. 3 ¶149 | [K,T] |
| Functional level falls as anxiety in a person's most important relationships rises. | *FE* Ch. 4 ¶65 | [K,T] |
| One must believe in a position enough to take an action stand. | Tape 4 ¶28 (ASR, paraphrased) | [B] |

No passage found says that the other person develops a symptom in response to an I-position; the owner's "or get
sick" is the owner's [user] claim until a source is found. The corpus gives no magnitudes: every constant below is [I].

### 3.3 The rule (new, `M5.D.3b`)

When a sequence reaches `DEFINE`, the mover's **conviction** in that position is set once, and holds for the
sequence:

```
conviction  = (basic_level + importance) × (1 − tie_discount × intensity)
capacity    = hold_gain × conviction × efficacy        (replaces functional_level in M5.D.3's capacity)
```

- **A — `basic_level`.** The solid self, which pressure does not change, so a mover whose functioning has been dragged
  down keeps what they believe (*FTCP* Ch. 21 ¶15). Direction [T]; form [I].
- **B — `importance`.** How much this position matters to the mover, drawn per sequence: a person-level draw, not a
  rule written per outcome. Each person is given a **line tendency** at the start of a run (a random initial
  condition, from a declared range): with that probability a position is a **line**, whose importance is large
  (`line_importance`), so even a low-level mover holds it against ordinary opposition; it is not infinite, so
  enough stress still breaks it ("unless it's to save a life"). Any other position draws its importance from
  `[0, importance_max]`. Direction [D, user]; form and sizes [I].
- **The relationship — `intensity`.** The intensity of the tie to the sequence's target, read from an existing tie
  quantity chosen at the plan step (candidates: `functional_togetherness`, `bond_energy`). The more intense the
  tie, the less conviction (*FE* Ch. 4 ¶94–95). Direction [K,T]; form [I].
- **Stress** is already in the comparison: a mover holds while excess anxiety stays within capacity, so more stress
  means fewer holds (*FTCP* Ch. 9 ¶37). No second stress term is added (O2).
- **Preparation** stays as it is: `efficacy` lowers the capacity of an unprepared sequence (`M5.D.2a`); "months to
  become sure" is `PREPARE`.

**What is set, random and emergent.** Set: the rule above. Random: line tendency, each position's importance.
Emergent: whether a sequence holds or aborts, who completes one, how often a family comes around. No rule writes
any of these.

**New constants, all [I]:** `importance_max`, `line_importance`, `line_tendency_range`, `tie_discount`; `hold_gain`
is re-set, because its units change from per point of functional level to per point of conviction.

### 3.4 Predictions, registered before the build

Each is a direction between two arms. P2–P4 are **premises** (`M11.5`): they confirm the code, and a pass must never
be reported as a finding. P1 and P5 are measurements.

| # | Arms | Expected direction | Class |
|---|---|---|---|
| P1 | T-259's triad fixture, revision 66 against revision 67 | revision 67 completes more sequences than 0 | measurement |
| P2 | mover's basic level 30 against 60 | the higher level holds more often | premise |
| P3 | the same mover toward a more against a less intense tie | the more intense tie aborts more often | premise |
| P4 | a scripted load against none | the load lowers the hold share | premise |
| P5 | line tendency 0 against the declared range, movers at level 30 | the range completes some sequences at level 30 | measurement |

`M5.D.3`'s claim that abort is the usual first outcome is measured again on fresh seed blocks; it is not tuned for.

## 4. For the owner

- **O1. The extra acts within a sequence.** At revision 66, an extra reactive act toward the sequence's other person
  is made as `STAY-IN-CONTACT` (`M5.D.9`), which turns an automatic act into a self act and opens a self-practice
  window, against `M4.D.3c`. Recommended: keep the extra act automatic (it is part of the opposition the mover has
  to hold against), and stop converting it.
- **O2. Stress.** Recommended: stress enters only through excess anxiety, as now; a second term would count it twice.
- **O3. Intensity.** Which tie quantity is "the importance of the relationship" — settled at the plan step against
  the spec's definitions.

## 5. T-265: the LLM experiment

A live language-model run does not repeat, even at temperature 0. The saved responses are the record, and replay only
checks that the engine gives the same behaviour from them, so any engine change that alters a prompt breaks it.
Recommended: freeze the experiment as a revision-64 result. Keep the runs, label each with its engine, and change
`test_lx16_report_is_current` and `test_lx25_committed_pilot_replays_fixed` to check the report against that recorded
engine rather than the current one. A later live run, if any, goes on a tagged version, with several samples per
seed and A0 on the same engine, and is compared as a direction with its spread.

## 6. Cost

§2 and §3 move the engine hash, so every engine-keyed record is regenerated once, on Mac Mini1, after both are built
(about the size of T-258's run). §5 changes tests only and does not move the hash.
