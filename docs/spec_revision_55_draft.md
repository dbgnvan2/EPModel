# Spec revision 55 — draft (T-202, social cues and signal channels)

> **Status: DRAFT 2026-11-07, for owner review.** This draft is T-202's step 2 (`docs/plan_t202_social_cues.md`, approved
> by the owner 2026-11-07 with D1–D6 each as recommended). It turns `SC-1` to `SC-11` of `docs/proposal_social_cues.md`
> (decided 2026-11-07, Q1–Q11 each as recommended) into requirement text. `SC-12` is not in it (plan D5; T-204).
> This page edits nothing under `src/bowen/` or `config/bowen/`, and nothing in the spec itself. The text is applied with
> the code it describes, in one engine-hash batch (plan steps 3–5), and the spec's revision log records where the
> applied text departs from this page. The engine hash (`0b363cb3186c`) did not move.

Grades follow `model_explainer.md`'s legend: evidence `[T]` textual, `[M]` mechanism, `[D]` direction, `[I]`
invented, `[X]` contested; attribution `[B]` Bowen, `[K]` Kerr, `[K-ext]` Kerr's extension. Kerr's 2019 book (`KS`)
and Kerr and Bowen 1988 (`FE`) are written sources, so short quotation is allowed; every `KS` and `FE` row is `[K]`.
Only `ch18` is Bowen's own text (`[B]`).

---

## 1. What this revision changes

| ID | New or amended | From | Class of its criterion |
|---|---|---|---|
| `M1.F.1` | amended: the field list adds `signals` | `SC-3` | check |
| `M1.F.1c` | new: the four signal channels on an Event | `SC-3` | check |
| `M1.F.4` | one sentence: a private hop carries the words only | Q9 | check |
| `M4.C.1e` | new: an event is read as a cue vector over four cues | `SC-1` | premise |
| `M4.C.1f` | new: the reduction, and the switch that is off by default | `SC-2` | check |
| `M4.C.1a` | one sentence: the level terms of `M4.C.14` | `SC-7` | — |
| `M4.C.5` | one sentence: the inward reading includes the cue reading | `SC-1`, `SC-6` | — |
| `M4.C.6` | amended: self-attribution by signal channel when the cues are on | `SC-8` | composite (`M11.C.52`) |
| `M4.C.9` | one sentence: a witness reads all four channels | Q9 | check |
| `M4.C.11` | new: anxiety raises the receiver's weight on the nonverbal channels | `SC-5` | premise |
| `M4.C.12` | new: real or imagined — the meaning error | `SC-6` | premise + composite (`M11.C.51`) |
| `M4.C.13` | new: words are chosen; tone, face and posture leak | `SC-4` | premise + composite (`M11.C.50`) |
| `M4.C.14` | new: differentiation on three terms, none reaching zero | `SC-7` | check + premise |
| `M4.C.15` | new: attention sent falls with the sender's stressor load (a proxy for `life_energy`) | `SC-9`, plan D3 | composite (`M11.C.53`) |
| `M9.9` | new: each person's believed cue vector per partner | `SC-10` | composite (`M11.C.54`) |
| `M7.E.1d` | one sentence: the belief travels as leak | `SC-10` | — |
| `M1.A.14b` | one sentence: no role field, and the role readout | `SC-11` | check + composite (`M11.C.55`) |
| `M11.G.4` | new: the supply readout | `SC-11` | — |
| `M11.C.50`–`M11.C.55` | six new criterion rows | `SC-4`, `SC-6`, `SC-8`, `SC-9`, `SC-10`, `SC-11` | composite |
| `M11.5` | six new rows | | |
| `M10.C.1a` | the new `[I]` constants | | |

The unit premises and checks (`SC-1`, `SC-2`, `SC-3`, `SC-4`(i), `SC-5`, `SC-6`(i), `SC-7`) are tests in
`tests/bowen/test_social_cues.py`, each proved red by its mutant through `tools/mutation_gate.py`; they are not `M11.C`
rows, as `M4.C.1d`'s and `M4.C.10`'s tests are not. The plan's §3 maps each to its test and mutant.

---

## 2. The decisions this revision rests on

- **Q1** (the owner, 2026-11-07): the four cues are readings that move `M4.C.1`'s two drives, not new drives. So
  `M4.C.1`'s sentence "These two drives are given … and **MUST NOT** be learned" is unchanged, and `M4.C.1e` says the
  cues add no drive.
- **Q2**: disapproval acts on both sides, impingement larger. **Q3**: another's distress acts on impingement, `[X]`.
  **Q4**: `M4.C.11` kept at `[I]`. **Q5**: sensitivity and meaning accuracy are two terms; the imagined part of a cue is
  meaning only. **Q8**: the cues are engine-only (the LLM agents are not in the spec). **Q9**: a private hop drops the
  nonverbal channels, a witness reads all four. **Q11**: built behind a switch off by default.
- **Plan D1**: the deviation form. The word channel carries no cue of its own: what the words do is the move kind's two
  components in `event_kinds.md`, unchanged. The cue reading adds to those components what the other three channels
  and the reading itself add. So with the nonverbal channels empty and no reading error, the felt change equals
  today's, and no move kind needs a cue vector.
- **Plan D2**: `M11.C.51` compares a belief, not two levels; `M11.C.52` is probed before the records are regenerated;
  `M11.C.54` gates on functional level only; `M11.C.55` compares two predictors.
- **Plan D3**: `M4.C.15` is a proxy while `M6.I.3` is not built. **Plan D5**: no age term (`SC-12` not built; T-204).

---

## 3. The requirement text

Each block below is the text to be inserted, as it would read in the spec.

### 3.1 `M1.F.1` — the field list (amended)

> **M1.F.1** An `Event` **MUST** carry: sender, targets, witnesses, move type, intensity, timestamp, `duration`,
> `exogenous`, `source_position`, `route`, `fidelity`, and, when the social cues are on (`M4.C.1f`), `signals`
> (`M1.F.1c`).

### 3.2 `M1.F.1c` — new, after `M1.F.1b`

> **M1.F.1c** — *a move carries four signal channels.* When the social cues are on (`M4.C.1f`), a move event **MUST**
> carry `signals`: four **signal channels** — words, tone, face and posture — each a cue vector over attention,
> approval, expectations and distress, each cue in [−1, 1], the threat pole negative (`M4.C.1e`). The engine sets
> `signals` at act from the sender's state (`M4.C.13`, `M4.C.15`, `M9.9`); no policy and no script sets it. A move
> without `signals` **MUST** be refused while the cues are on. When the cues are off, `signals` **MUST** be absent and
> **MUST NOT** be written to the run log, so a run with the cues off is the run of the engine before this revision
> (`M4.C.1f`). Non-move events carry no signals. "The signals monitored are primarily auditory and visual ones and
> include body postures, tone of voice, facial expressions, and actual word content" (`family_evaluation/fe03.md` ·
> FE03.3, `[K]` `[T]`); Bowen's clinical profile has a child and mother reading each other's "facial expressions,
> tone of voice, body posture, and footsteps" (`ch18.md` line 42, `[B]` `[T]`; footsteps folded into posture). The
> words *signal channel* are kept apart from `M4.D.1a`'s channels of selection.

### 3.3 `M1.F.4` — one sentence added

> When the social cues are on, a private hop carries the words only: a reported message arrives with its tone, face
> and posture empty, because those are what is "monitored" in presence (`FE03.3`); the form is `[I]` (Q9, the owner,
> 2026-11-07).

### 3.4 `M4.C.1e` — new, after `M4.C.1d`

> **M4.C.1e** — *an event is read as a cue vector; the cues move the two drives and are not drives.* When the social
> cues are on (`M4.C.1f`), a move delivered to its target **MUST** also be read as a **perceived cue vector** over four
> cues — attention, approval, expectations, distress — each in [−1, 1], the threat pole negative: lack of attention,
> disapproval, an expectation placed on the receiver, upset in the sender. The perceived vector is the receiver's
> weighted reading of the four signal channels (`M4.C.11`), after detection and the meaning error (`M4.C.12`,
> `M4.C.14`). Each cue **MUST** move the receiver's felt contact and felt impingement through two weights held in
> config, scaled as the move's own components are (`conductance`, `route`, `fidelity`, intensity), and added to the
> move kind's two components (`M4.C.1`): a cue at its threat pole lowers felt contact by its contact weight and raises
> felt impingement by its impingement weight; at its other pole it does the reverse. The weights' signs are directions
> from the sources and their sizes are `[I]`:
>
> | Cue | Felt contact | Felt impingement | Basis |
> |---|---|---|---|
> | attention | yes | no | `KS10.1`: inadequate attention triggers "rejection, panic, and emotional isolation"; `FE03.3`'s too-little set; `[K]` `[D]` |
> | approval | yes, smaller | yes, larger | `M4.C.5`'s reading of criticism; `FE03.3`'s too-little set includes feeling unloved and rejected; `[K]` `[D]`; the split `[I]` (Q2) |
> | expectations | no | yes | `KS08.4`: the expectation in the voice; `M1.A.9a`'s outward axis; `[K]` `[D]` |
> | distress | no | yes | `KS10.1`'s "upset or equanimity in the other"; the side `[X]`: the corpus does not say which side another's distress acts on (Q3) |
>
> The four cues are readings of a received signal. They **MUST NOT** become drives, set points or goals: `M4.C.1`'s two
> drives remain the only drives, and the cue reading **MUST NOT** write any person field the policy reads other than
> through felt contact, felt impingement and acute anxiety (`M9.9`'s belief is the one store the reading writes). The
> list is open — "and certainly other cues as well" (`KS10.1`) — and a cue added later is a reading, not a drive.
> Sources for the list: `KS01.7`, `KS07.6`, `KS09.4`, `KS10.1`, `KS23.1` (`[T]` `[K]`): people are sensitive to
> "approval, attention, expectations, and distress", and "the less 'self' people have, the more they are governed by
> reactions to such real or imagined cues" (`kerr_book/ks01.md` · KS01.7). Kerr calls them "the most common threats
> people experience" (`KS10.1`), not fundamental; the fundamental triggers stay `M4.C.1`'s two (`KS03.2`, `KS06.3`).
> A held expectation that goes unmet reads on the too-little side: a received attention or approval cue below the
> receiver's belief about that partner (`M9.9`) lowers felt contact by the shortfall times an `[I]` weight
> (`KS10.1`'s "expectations met or unmet"; `KS08.4`, the mother's anxiety at the daughter's silence; `[K]` `[D]`).

### 3.5 `M4.C.1f` — new, after `M4.C.1e`

> **M4.C.1f** — *the reduction, and the switch.* The social cues (`M1.F.1c`, `M4.C.1e`, `M4.C.11`–`M4.C.15`, `M9.9`)
> **MUST** be behind one config switch, `social_cues_enabled`, which is **off** by default. With it off, a run **MUST**
> produce the same log as the engine before this revision, record for record (the golden fingerprints,
> `tests/bowen/test_engine_golden.py`, unchanged), and **MUST** make no draw of the `cue_perception` class
> (`M3.D.4a`). With it on, with the three nonverbal channels empty, no meaning error and full detection, each move's
> felt change **MUST** equal the change with it off, for every move kind. This holds by construction under the
> deviation form (plan D1): the word channel carries no cue, so with nothing on the other channels the cue reading adds
> nothing. Turning the switch on by default is an owner decision, and it moves every record. `[I]`, an engineering rule.
> The `M11.C` criteria that test the cues (`M11.C.50`–`M11.C.55`) run both their arms with the switch on.

### 3.6 `M4.C.1a` — one sentence added

> When the social cues are on, level also sets the two terms of `M4.C.14`, sensitivity and meaning error, by the same
> rule: derived from `functional_level`, never parameterised per person (`M10.A.1`).

### 3.7 `M4.C.5` — one sentence added

> When the social cues are on, the impingement a delivered move adds includes its cue reading (`M4.C.1e`), so reading a
> neutral event as disapproving (`M4.C.12`) is read here as criticism heard: the perception side of the inward axis.

### 3.8 `M4.C.6` — amended (the sentence after the first is replaced while the cues are on)

> When the social cues are on, the part of a received move that an agent with `systems_perspective` attributes to its
> own prior emission **MUST** be the **nonverbal** part — the felt change that tone, face and posture carry — in
> proportion to the agent's own recent nonverbal leak on that tie, against the other's, over `reappraisal_window`;
> the words part (the move kind's components) is not reattributed. What he took to be a tense expression on his
> wife's face "reflected his own facial expression and tone of voice, which in turn were intensified by what he saw in
> her" (`kerr_book/ks17.md` · KS17.3, `[K]` `[T]` `[M]`). The share's form is `[I]`. With the cues off, the rule is as
> it stands. Criterion: `M11.C.52`.

### 3.9 `M4.C.9` — one sentence added

> When the social cues are on, a witness reads all four signal channels from its own position, through `M4.C.11`,
> `M4.C.12` and `M4.C.14` (Q9; `[I]` in form).

### 3.10 `M4.C.11` — new

> **M4.C.11** — *anxiety raises the weight on the nonverbal channels.* When the social cues are on, the weight a
> receiver gives tone, face and posture against the words **MUST** rise with the receiver's acute anxiety above its
> chronic level, from a base weight to a cap below one. Direction and size `[I]`: no extraction entry says that anxiety
> moves weight from words to the nonverbal channels; the nearest is `KS01.7`'s "the less 'self' … the more they are
> governed by reactions to such real or imagined cues" (`[K]`), which is about level, not acute anxiety. Kept by the
> owner, `[I]`, as a premise (Q4, 2026-11-07).

### 3.11 `M4.C.12` — new

> **M4.C.12** — *real or imagined: the error is on meaning, never on presence.* When the social cues are on, the cue a
> receiver perceives on a detected signal channel **MUST** be the cue sent plus an error toward each cue's threat pole,
> with a spread drawn per delivery from the `cue_perception` draw class (`M3.D.4a`). The bias **MUST** grow with the
> receiver's acute anxiety above its chronic level, with the receiver's belief that the sender sends that cue's threat
> pole (`M9.9`, `M9.7`), and as `functional_level` falls (`M4.C.14`). The error changes what a signal that is there is
> read to mean — a neutral face read as disapproval, a tired voice read as an expectation; it **MUST NOT** create a
> signal on a channel where none was detected (the owner, 2026-11-07: "meaning only"). The source is
> "the perception (or misperception)" (`FE03.3`) and "real or imagined" cues (`KS01.7`, `KS07.6`), `[K]` `[T]`; the
> form is `[I]`. `M4.C.1c`'s too-little side already carries the reading of an absence, and is unchanged. Criterion:
> `M11.C.51`.

### 3.12 `M4.C.13` — new

> **M4.C.13** — *words are chosen; tone, face and posture leak.* When the social cues are on, the sender's tone, face
> and posture **MUST** carry a leak toward the threat poles of expectations and distress that grows with the sender's
> acute anxiety above its chronic level, saturating at `defence_threshold` (`M4.C.2`); the sender does not choose it,
> and no policy reads or sets it. A more differentiated sender leaks less only because its acute anxiety is lower; no
> separate level term is added (`M10.A.1`). The leak is the channel along which `M4.C.10`'s calmer-sender effect
> travels. Source: the mother's phone message, where "the anxiety in her voice and her message amounted to an
> expectation that the daughter call her", and the changed message, near-identical in words with a lighthearted
> delivery, "erased a decades-old emotional cue" (`kerr_book/ks08.md` · KS08.4, `[K]` `[T]` `[M]`). The leak's size, its
> split between the two cues and the three channels' shares are `[I]`. Criterion: `M11.C.50`.

### 3.13 `M4.C.14` — new

> **M4.C.14** — *differentiation changes sensitivity and meaning error by degree, and removes neither.* When the social
> cues are on, two terms **MUST** be derived from the receiver's `functional_level` (`M10.A.1`), beside `M4.C.1a`'s
> steepness:
>
> - **sensitivity** — how readily a nonverbal signal is detected — **MUST** rise as level falls. It is accurate about
>   one thing: that a signal is being sent. It **MUST NOT** add meaning.
> - **meaning error** — the spread and the threat bias of `M4.C.12` — **MUST** rise as level falls.
>
> Neither term, nor the steepness, **MUST** reach zero at any level: the sensitivities "are part of all people"
> (`KS01.7`). Sources: people with more self have "less intense reactions, more accurate perceptions" (`KS09.4`,
> `[K]` `[T]`); the impaired child "read" the mother's signals and the mother read hers equally well (`ch18.md` line 42,
> `[B]` `[T]`). The two hold together once sensitivity and meaning accuracy are two terms (the owner, 2026-11-07, Q5:
> lower differentiation means more sensitivity to a signal, which is accurate that a signal is being sent, not about
> what it means): `[D]` for the two directions, from the two sources together; every form and size `[I]`.

### 3.14 `M4.C.15` — new (a proxy, plan D3)

> **M4.C.15** — *attention is finite.* When the social cues are on, the attention cue on a sender's tone, face and
> posture **MUST** fall toward its threat pole with the load of the lasting exogenous stressors on the sender this week
> (`M1.F.6`), so that an outside preoccupation is read by the people around it as less attention. Source: a partner's
> preoccupation with work leaves the other "feeling inadequately attended to", and an outside stressor becomes an
> inside one (`kerr_book/ks07.md` · KS07.6, `[K]` `[T]`). **This is a proxy.** `M1.A.10`'s `life_energy`, zero-sum
> between relationship-seeking and goal-directed activity, is not built (`M6.I.3`); when it is built, this rule's bound
> **MUST** become the relationship share of `life_energy`, and the proxy is retired. Form and size `[I]`. Criterion:
> `M11.C.53`.

### 3.15 `M9.9` — new, after `M9.8`

> **M9.9** — *expectations are beliefs: each person holds a believed cue vector for each partner.* When the social
> cues are on, each person **MUST** hold, for each tie it is party to, a belief about the cues that partner sends it —
> its attention, its approval, its expectations of the person, its distress — written only from the cues the person
> perceives on that tie (`M4.C.12`, after fidelity), by an `[I]` smoothing rule, and starting neutral unless a scenario
> or a criterion's arm sets it. The belief **MUST** be permitted to be false (`M9.2`) and **MUST** drive appraisal
> (`M9.7`) in two ways: it biases the reading toward what is believed (`M4.C.12`), and it is the held expectation an
> unmet cue is read against (`M4.C.1e`). A person's belief that a partner is in distress or failing **MUST** also be
> sent toward that partner as expectation and distress leak on the tie (`M4.C.13`), beside the leak of its own acute
> anxiety. This holds for every tie and every person; it is not a parent's rule. Sources: `KS08.4` (the expectation in
> the voice, and the mother's anxiety at the daughter's silence), `KS10.1` ("expectations met or unmet"), both `[K]`
> `[T]`, and `M7.E.1d`'s sources; the form is `[I]`. Criterion: `M11.C.54`.

### 3.16 `M7.E.1d` — one sentence added

> When the social cues are on, a parent's belief about a child travels to the child as expectation and distress leak
> (`M9.9`, `M4.C.13`), with no signal from the child needed to start it; the child reads it, and what follows is
> learned, not written.

### 3.17 `M1.A.14b` — one sentence added

> No config key and no `Person` field **MAY** name a role — caretaker, scapegoat, helper, parentified child, or any
> other word in `config/lint/role_labels.md`; a test scans for them. Who supplies attention, approval and help to whom
> is a readout computed after the run (`M11.G.4`), and the policy **MUST NOT** read it.

### 3.18 `M11.G.4` — new, after `M11.G.3`

> **M11.G.4** — *the supply readout.* The model **MUST** be able to report, after a run, who supplied attention,
> approval and help to whom: per ordered pair, the count of moves from the kinds a readout config lists as supplying
> (`STAY-IN-CONTACT`, `OVERFUNCTION` and those the owner adds), and, when the social cues are on, the mean perceived
> attention and approval cues each person received from each other. A role name ("the one who worries", "the
> caretaker") may label this readout in a report, never the person, and nothing in the engine reads it
> (`M1.A.14b`). Which kinds count as supplying is editorial and `[I]`.

### 3.19 `M11.C.50`–`M11.C.55` — six new rows, after `M11.C.49`

Every value a row names (weeks, levels, seeds, the belief set in an arm) goes in `config/bowen/criteria.md`, declared
`[I]` before the row first runs. Every row runs both arms with `social_cues_enabled` on.

> | **M11.C.50** | **A calmer delivery is met with more contact.** `KS08.4`'s two arms on identical seeds: one sender's moves toward one receiver scripted the same in both arms over a window, the sender's acute anxiety held high in one arm and low in the other by the scenario. Over the weeks after the window, the receiver's self-selected moves with positive contact toward the sender **MUST** be more frequent in the low arm. | `tests/bowen/test_criteria_sc.py::test_m11c50` | C | the leak removed (`cue-leak-removed`) **MUST** turn it red; Phase C mutation record |
> | **M11.C.51** | **A believed disapproval is read into neutral moves, and the receiver distances or fights more.** Same family, same levels, identical seeds; the receiver's starting belief that one partner disapproves of it (`M9.9`) set toward the threat pole in one arm and neutral in the other. Over the run, the receiver's share of `DISTANCE` and `CONFLICT` among its moves toward that partner **MUST** be higher in the first arm. (Plan D2(a): two levels are not compared, because level also moves steepness, band and gain.) | `::test_m11c51` | C | the belief term of `M4.C.12`'s bias removed (`cue-belief-bias-removed`) **MUST** turn it red |
> | **M11.C.52** | **Self-attribution by channel damps an escalation.** Two members of one tie both held anxious at the start; in one arm one of them has `systems_perspective`. The tie's escalation — the sum of both members' acute anxiety over the exchange's weeks — **MUST** be lower in that arm. Probed before the records are regenerated (plan D2(b)): `M4.C.6`'s whole-event rule may produce the direction alone. | `::test_m11c52` | C | the attribution moved to the words part (`cue-attribution-reads-words`) **MUST** turn it red |
> | **M11.C.53** | **A preoccupied spouse is felt as less attentive.** A lasting work stressor on one spouse in one arm and none in the other, identical seeds. Over the stressor's weeks, the other spouse's too-little deviation on the marital tie **MUST** be higher in the first arm, and its `PURSUE` plus `CONFLICT` rate toward the stressed spouse **MUST** be higher. | `::test_m11c53` | C | attention not lowered by the load (`cue-attention-load-removed`) **MUST** turn it red |
> | **M11.C.54** | **A parent's false belief about a child lowers that child's functioning.** On the Phase C family, identical seeds, the children's own starting states unchanged: one parent's starting belief about one child (`M9.9`) set toward the distress and expectation threat poles in one arm and neutral in the other. At the end of the run that child's `functional_level` **MUST** be lower in the first arm. The child's share of supplying moves toward that parent (`M11.G.4`) is reported, not compared (plan D2(c)). | `::test_m11c54` | C | the belief not sent as leak (`cue-belief-leak-removed`) **MUST** turn it red |
> | **M11.C.55** | **Who supplies a parent is learned, not given by birth order.** On the twelve-person family over its seeds, the child who becomes the main supplier to a parent (most supplying moves toward that parent, `M11.G.4`, over a declared age window) **MUST** be predicted more often by which child had the higher felt relief from supplying that parent before the window than by which child was born first. Probed before the records are regenerated, because the relief predictor is read from the same run (the lesson of T-194). | `::test_m11c55` | D | supplying moves weighted by birth order (`supply-by-birth-order`) **MUST** turn it red; Phase D mutation record |

### 3.20 `M11.5` — six new rows

> | M11.C.50 | Composite | — | needs `M4.C.13`'s leak, the receiver's reading and learning from relief; the direction is not written by any rule |
> | M11.C.51 | Composite | — | needs `M9.9`, `M4.C.12` and learning; the share of `DISTANCE` and `CONFLICT` is chosen by the policy |
> | M11.C.52 | Composite | — | needs `M4.C.6` by channel, `M4.C.13` on both sides and the exchange over weeks; if its named mutant survives the probe, the row comes back to the owner before it counts |
> | M11.C.53 | Composite | — | needs `M4.C.15` (a proxy), the reading and learning; its arms also differ in the stressed spouse's own anxiety and moves, so only its named mutant shows whether `M4.C.15` carries it |
> | M11.C.54 | Composite | — | needs `M9.9`'s leak, the child's reading and `M7.E`'s development; the projection is not written |
> | M11.C.55 | Composite | — | needs learning from relief (`M4.D.6`); a within-run predictor, so its probe decides whether it is informative |

### 3.21 `M10.C.1a` — added to the list

> - the social cues' switch, `social_cues_enabled` (off), the four cues' contact and impingement weights, the unmet
>   expectation weight, the leak gain with its split between expectations and distress and the three nonverbal
>   channels' shares, the nonverbal weight's base, anxiety gain and cap, the sensitivity's floor and level gain, the
>   meaning error's bias floor, level gain, anxiety gain and belief gain and its spread's floor and level gain, the
>   attention load gain and load scale, the belief smoothing rate and the belief leak gain (`M4.C.1e`, `M4.C.1f`,
>   `M4.C.11`–`M4.C.15`, `M9.9`), all in `config/bowen/social_cues.md`, set before the cues' criteria first run
>   (`M10.B.4`)

---

## 4. `config/bowen/social_cues.md` — the starting values (applied at plan step 3 and 4)

All `[I]`, invented here and declared before any cue criterion runs. Parsed strictly, as `constants.md` is.

| Name | Value | What it sets | Rule |
|---|---|---|---|
| `social_cues_enabled` | false | the switch | `M4.C.1f` |
| attention weights (contact, impingement) | 0.10, 0 | | `M4.C.1e` |
| approval weights | 0.04, 0.08 | | `M4.C.1e`, Q2 |
| expectations weights | 0, 0.08 | | `M4.C.1e` |
| distress weights | 0, 0.06 | `[X]` side | `M4.C.1e`, Q3 |
| `unmet_expectation_weight` | 0.05 | contact lost per unit shortfall | `M4.C.1e` |
| `cue_leak_gain` | 0.6 | leak at `defence_threshold`'s excess | `M4.C.13` |
| leak split (expectations, distress) | 0.5, 0.5 | | `M4.C.13` |
| channel shares (tone, face, posture) | 1.0, 1.0, 0.5 | | `M4.C.13` |
| `nonverbal_weight_base`, `_anxiety_gain`, `_cap` | 0.5, 0.3, 0.9 | | `M4.C.11` |
| `sensitivity_floor`, `sensitivity_level_gain` | 0.3, 0.6 | detection as level falls | `M4.C.14` |
| `bias_floor`, `bias_level_gain`, `bias_anxiety_gain`, `bias_belief_gain` | 0.02, 0.15, 0.15, 0.3 | | `M4.C.12`, `M4.C.14` |
| `spread_floor`, `spread_level_gain` | 0.05, 0.2 | | `M4.C.12`, `M4.C.14` |
| `attention_load_gain`, `attention_load_scale` | 0.5, set at step 4 from the stressor sizes in use | | `M4.C.15` |
| `belief_cue_rate` | 0.1 | per perceived cue | `M9.9` |
| `belief_leak_gain` | 0.5 | | `M9.9` |

No value is presented as sourced. A value tuned to make a criterion pass would be a fit (`M11.F.9`), so the values are
set before the criteria run and changed only by an owner decision logged in `constants_changes.md` (`M10.B.4`).

---

## 5. The §19 check: does any new rule write an outcome?

| Rule | What it sets | What it leaves to emerge |
|---|---|---|
| `M4.C.1e` | how a perceived cue moves felt contact and impingement | every move, distance, conflict |
| `M4.C.11`, `M4.C.12`, `M4.C.14` | how a signal is read, by state and level | what the reader does about it |
| `M4.C.13`, `M4.C.15` | what anxiety and load put on the nonverbal channels | how others respond |
| `M9.9` | that a belief is held, updated, and leaks | whether it is confirmed; which child carries a projection |
| `M1.A.14b`, `M11.G.4` | that no role is stored; a readout | who becomes the supplier |

No rule names a caretaker, a projection target, a cutoff or a conflict. The six composites test outcomes that no rule
states; each must still be proved red by its named mutant, and three (`M11.C.52`, `M11.C.53`, `M11.C.55`) carry a
stated risk that the mutant survives.

## 6. The attribution check

Every `KS`, `FE` and `KB` source above is `[K]`. Bowen is cited once, for `ch18`, as `[B]`. `KS09.4`'s supervision
questions are Kerr's report and are not used. The owner's file's unfound quotations are not used (Q10).

## 7. Open questions for the owner

- **R55-Q1 — the unmet expectation term.** The draft keeps the proposal's "an unmet held expectation reads on the
  too-little side" (`M4.C.1e`, last sentence), reading the held expectation as `M9.9`'s belief. Recommended: keep it.
  The other choice is to leave it out, so that the belief acts only through the bias and the leak; `M11.C.54` would
  then rest on the leak alone.
- **R55-Q2 — the starting values (§4).** Recommended: accept them as starting values, to be declared `[I]` and frozen
  before the criteria first run. They are set without having run anything.

## 8. Documents that change when this revision is applied

The spec (the text above, its status line, `M10.C.1a`), `docs/spec_revision_log.md`, `docs/model_explainer.md`,
`docs/readouts_phase_c.md` (`M11.G.4`), `config/bowen/social_cues.md` (new), `config/bowen/criteria.md`,
`config/bowen/constants_changes.md`, `config/lint/role_labels.md` (new), the regenerated records and the generated
indexes, and the documents the plan's §6 lists.
