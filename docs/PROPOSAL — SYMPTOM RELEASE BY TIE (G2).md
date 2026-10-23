# Proposal — symptom release by tie (version 2, G2)

> **Status: DRAFT for the owner's review, 2026-10-23. Version 2, group G; nothing built, no spec revision.** It
> answers the two questions `docs/version_2_plan.md` §3 (stage 3, group G) records under G2: when a symptom gives back
> what it held, or spills the bearer's own excess past the turn, should the shares follow the ties rather than be
> equal, and should any of it reach an external agent? Nothing in the corpus gives a weighting; every form here is
> Claude's recommendation, graded `[I]`, and no new constant is proposed. It changes what a run computes, so it moves
> the engine hash; by G1's Q9 (decided 2026-10-19) it is built in one batch with G1 and the Holmes–Rahe sizes, so
> every record regenerates once. Questions are Q1–Q5 in §8. Before the batch is built, the Holmes–Rahe Q3 is asked
> again (`TODO.md` T-013, owner 2026-10-21).

## 1. What the owner has decided so far

Nothing on G2 itself. What bears on it:

| Date | Decision | Where |
|---|---|---|
| 2026-10-19 | Group G (G1, G2, the Holmes–Rahe sizes) is in version 2 and built before the stage 2 pilot, since it reaches every run that binds anxiety. | `version_2_plan.md` §5 Q2 |
| 2026-10-19 | G1, G2 and the Holmes–Rahe sizes are built in one batch, one revision, one regeneration. | G1's Q9; Holmes–Rahe Q7 |
| 2026-10-19 | G1's Q1: a cut-off tie's bound stock weighs on both members, **scaled by bond energy**. | G1 §3.1 |
| 2026-10-14 | `M6.3c`: a dead person's `symptom_bound` goes back to the partners as when a symptom ends, and an external agent who is a symptom partner receives its share; only `M6.3b`'s share of the dead person's own acute anxiety excludes them ("okay"). | `M6.3c` |
| 2026-10-21 | Before group G's build starts, the Holmes–Rahe Q3 is asked again (D10, the stage 1 review's Q14). | T-013 |

## 2. What the code does today

Checked 2026-10-23 against `main` at `63409c1`.

- **The hold.** Each person has `symptom_bound`, the anxiety the family's focus on their active symptom holds
  (`M7.D.2a`, a stock in `M6.4`). Each week, for a bearer whose most loaded active channel has severity `x`,
  `s(x) = lock_in_gain × x × (1 − x / lock_in_turn)` (0.2 and 3.0, both `[I]`). While `s > 0`, **each partner on an
  interactive tie** moves `s ×` their own excess above usual into the hold (`symptoms.lock_in`). What each gives is set
  by what they have above their usual level; there is no weighting on this side, and none is needed.
- **The release.** Past the turn (`s < 0`), `min(1, |s|)` of the hold goes back each week (`release_bound`), and the
  same share of the bearer's own excess above usual spills to the partners (`lock_in`, the spill branch). When the
  symptom ends (re-arm) or a `binder_unavailable` names its channel (`M1.F.9`), the whole hold goes back. In every
  case the amount is **divided by the number of partners** (`symptoms.py:147` and `:198`): `each = amount / len(partners)`.
- **Who the partners are.** `_partners` returns every living person on an interactive tie to the bearer, whatever
  their role. In the Phase C family (`config/bowen/family_phase_c.md`) Marta's five are Ravi (spouse, conductance
  1.0, bond energy 60), Nadia and Pia (1.0, 50), Ana (0.6, 40) and Dr Halim, the coach (professional, 0.3, 10).
  So the coach takes a fifth of what Marta's symptom gives back, and gives into her hold whatever he has above his
  usual level.
- **The seed-7 trace** (`docs/review/phase_c_trace_seed7.md`, the `M11.C.1` baseline arm). Weeks 12–14 already show
  the pattern in small amounts: "Marta's symptom gives back 1.3 (Ana +0.3, Dr Halim +0.3, Nadia +0.3, Pia +0.3,
  Ravi +0.3)". In week 74, after `REDUCE_CUTOFF` handed Marta 226.4 of tie-bound anxiety and pushed her symptom past
  the turn: "Marta's symptom gives back 95.3 (… Dr Halim +19.1 …)" and "Marta's symptom is past its turn and holds
  their anxiety no longer (Marta −222.4, Ana +44.5, Dr Halim +44.5, Nadia +44.5, Pia +44.5, Ravi +44.5)": 63.6 to
  each of five, the coach among them.
- **How the rest of the model treats the same question.** Where a person's stock disperses to their ties, the
  model already weights by bond energy and keeps external agents out: `M6.3b` splits a dead person's acute anxiety
  among the living **family members** in proportion to bond energy, and an external agent takes no share. The
  learner's cross-person term weights the others' relief by the actor's bond energy on each tie (`M4.D.6e`,
  `learner.cross_person_relief`). The witnessed-cutoff record reads family members only (`learner._family_ties`).
  The coach tie accumulates no investment (`M1.E.7e`; `act.py:90`, `reactive.py:60`). The one place that goes the
  other way is `M6.3c`, decided 2026-10-14: at a death the dead person's hold goes back as when a symptom ends,
  external agent included, which is the equal split this proposal changes.
- **The ledger.** `lock_in` is a transfer in `M6.4`; `invariants.py` checks that what leaves the hold equals what
  the partners receive, per person. A weighting changes who receives, not the sum, so the invariant is unchanged.

## 3. What the corpus says, and does not

- **The hold is the family's.** Lock-in comes from `KS23.1` `[T]` `[M]`: "a family can stabilize somewhat around the
  presence of a symptom". `FE03.9` `[K]` `[T]` and `FE07.15` `[K]` `[T]` give the turn: past a point the symptom
  "add[s] to everyone's anxiety", and "if the dysfunctional person leaves the family, the family's anxiety level may
  actually decrease". "Everyone" and "the family" are the family. No passage names an outside helper as a recipient
  of what a symptom was holding.
- **Effects are keyed to position, not uniform.** `M7.D.2b` (`KS20.8` `[T]`, `KS22.8` `[T]` `[D]`): when Gary Gilmore
  was removed the family relaxed and only his mother wanted him back. `KS10.3` `[T]` `[M]`: who absorbs is set by the
  active patterns, not by proximity to the stressor. These support a release that falls unequally, by the tie, over an
  equal split. They do not say what the weight is.
- **The professional can absorb.** `L05.2` `[B]`, 1961: function is absorbed by whoever will carry it, including the
  professional, and proximity of help is a load-bearing input with a negative sign (`M1.E.5`). This is about
  function and helplessness, not about anxiety a symptom holds; it is the one entry that bears on the external agent,
  and it argues for keeping the coach's burden-transfer term (`M1.E.5`), not for the coach taking a share of a
  symptom's release. `M1.E.7e` (`KS18.5`): a coach who is drawn in becomes a lender of pseudo-self, and the tie must
  stay thin.
- **No magnitude anywhere.** Bond energy, conductance and investment are all the project's constructs. The weight is
  `[I]` whichever is chosen; the corpus supports only the direction (unequal, by involvement) and the boundary (the
  family).

## 4. The options

The same rule applies to the three releases: the weekly share past the turn, the bearer's spill past the turn, and
the whole hold on re-arm or `binder_unavailable`. `A` is what is being given back and the recipients are the living
people on interactive ties to the bearer.

| | Weight | For | Against |
|---|---|---|---|
| (a) | equal shares (today) | simplest | treats a thin professional tie like a spouse; no corpus support for uniformity, and `M7.D.2b` against it |
| (b) | **bond energy** of each tie, normalised | the rule the model already uses for a stock dispersing to ties (`M6.3b`, `M4.D.6e`, G1's Q1); undirected, fixed at import, never zero on a family tie in the Phase C file | `[I]`, as all of them |
| (c) | conductance of each tie, normalised | the tie's measure of how much passes along it (`M1.B.2`) | conductance scales what an act transmits, not where a stock settles; using it here gives the model two different "how much of a stock lands here" rules |
| (d) | each partner's investment in the bearer (`M1.B.8`), normalised | investment is "share of thought occupied by the target", the nearest thing to the family's focus; the coach's is always zero (`M1.E.7e`), so the external agent falls out without a role check | directed and accumulating, often zero on a tie that is interactive; a partner with none would receive nothing, and a fallback rule is needed from the first week |
| (e) | a per-contributor ledger: each gets back what they put in | the most literal "gives back what it held" | new state on the person (a dict), no rule for the bearer's own spill (it has no contributors) or for a contributor who dies or is removed, and a readout of the hold's composition no criterion reads |

**Recommended: (b)**, with external agents excluded (§5). On Marta's five ties conductance and bond energy rank the
same way, so (b) and (c) differ little there (they do not rank the same way across the whole family: Ana–Bruno is 0.8
and 55 against Marta–Nadia's 1.0 and 50); they differ for an imported family (`M15`), where the two are
set independently.

What the week-74 release would have been under each, from the trace's 317.7 (95.3 given back and 222.4 spilled) and
the ties in `family_phase_c.md`; arithmetic only, not a run:

| Recipient | bond energy | conductance | (a) equal | (b) bond energy | (b) + coach excluded | (c) conductance | (c) + coach excluded |
|---|---|---|---|---|---|---|---|
| Ravi | 60 | 1.0 | 63.5 | 90.8 | 95.3 | 81.5 | 88.2 |
| Nadia | 50 | 1.0 | 63.5 | 75.6 | 79.4 | 81.5 | 88.2 |
| Pia | 50 | 1.0 | 63.5 | 75.6 | 79.4 | 81.5 | 88.2 |
| Ana | 40 | 0.6 | 63.5 | 60.5 | 63.5 | 48.9 | 52.9 |
| Dr Halim | 10 | 0.3 | 63.5 | 15.1 | 0 | 24.4 | 0 |

(The trace prints 63.6 per head from the two rounded rows; 317.7 ÷ 5 is 63.5.)

## 5. The rules

All `[I]`. No new constant.

### 5.1 Release by bond energy (Q1)

When a symptom gives back any part of its hold, and when the bearer's own excess spills past the turn, the amount
**MUST** be divided among the recipients in proportion to the bond energy of the bearer's tie to each. If the
recipients' bond energies sum to zero, the shares are equal (Q4). If there is no recipient, the hold stays with the
bearer, as today.

Amends `M7.D.2a` ("split equally" → "in proportion to bond energy, among living family members on interactive ties"),
`M7.D.2d` (the same for the weekly share and the spill), `M6.4`'s `Person.symptom_bound` row, and `M6.3c` (Q3).

### 5.2 The family's hold holds the family's anxiety (Q2)

The recipients **MUST** be the living family members (`role = MEMBER`) on interactive ties to the bearer. An external
agent takes no share of a release and no share of a spill. On the binding side, an external agent's excess above
usual **MUST NOT** enter a family member's hold either: `lock_in`'s `s > 0` branch skips them. Both sides change
together so that nothing an external agent holds can be absorbed by a symptom and never returned.

A bearer whose only interactive tie is to an external agent keeps their hold until a family tie is interactive again,
which is `M7.D.2a`'s existing rule for a bearer with no partners.

### 5.3 What does not change

- What each partner gives while `s > 0`: `s ×` their own excess above usual (Q5). The hold is filled by what the
  partners have; a weight on this side would make a partner give anxiety in proportion to a tie rather than to their
  state, and nothing in the corpus asks for that.
- The curve `s(x)`, its two constants, the weekly share `min(1, |s|)`, re-arm, `binder_unavailable`, removal.
- The `M6.4` invariant and the `lock_in` record; the trace renderer, which prints the shares it is given.

## 6. Acceptance: tests and what is reported

Unit tests in `tests/bowen/test_symptoms.py`, each proved failing by mutation before it counts (the mutant is the
equal split, or the external agent back in the partner set):

1. **The closer tie receives more.** A bearer with two family partners at bond energies 60 and 10 gives back a hold
   of 7 as 6 and 1; the ledger balances. Mutant (a) gives 3.5 and 3.5.
2. **The spill follows the same shares.** Past the turn, the bearer's spill lands 6:1 on the same two partners.
3. **An external agent takes no share and gives none.** With the coach on an interactive tie and above his usual
   level, the family member's hold does not grow by his excess, and a release leaves his anxiety unchanged.
4. **Zero bond energy falls back to equal shares** (Q4).
5. `test_m7d2d_past_the_turn_the_symptom_gives_back_what_it_held` asserts every interactive partner rises; it is
   restated over family members. `test_m7d2a_a_symptom_ending_returns_what_it_held` sums the release and is
   unchanged. `M6.3c`'s death test gains the same two assertions.

Reported, before and after, in the batch's regeneration (every record, since the engine hash moves): the criteria
that read a symptom's release — `M11.C.12` (the cure arm; fails at version 1), `M11.C.22` (both limbs), and `M11.C.15`
through `M6.3c` — and the seed-7 trace. Each changed result is reported as a new result under a
new rule, not as version 1's finding withdrawn (`version_2_plan.md` §3). `M11.C.12` is not expected to turn green
from this alone and must not be read that way: the rule moves where the released anxiety lands, and the criterion is
earned only if marital `CONFLICT` rises from it.

**Emergence check** (`model_explainer.md` §19). The rule writes a landing, not a pattern: it says whose anxiety rises
when a symptom lets go, in proportion to the tie, and nothing about what they then do. No criterion's outcome is
written by it.

## 7. Conflicts stated

- **`M6.3c` (owner, 2026-10-14):** an external agent who is a symptom partner receives its share at a death. Q2 as
  recommended reverses that clause for the symptom's hold, so that one rule governs the hold wherever it empties.
  `M6.3b`'s exclusion of the external agent from the dead person's own anxiety stands and is the precedent. Q3 asks.
- **`M7.D.2a` / `M7.D.2d` "split equally":** the project's wording, `[I]`, with no source; amended, not contradicted.
- **The three stated corpus passages** give the boundary (the family) and the direction (unequal). The weight is
  invented, and its grade says so.

## 8. Questions for the owner

1. **Q1. The weight:** bond energy (recommended, §4 (b)); conductance (c); each partner's investment in the bearer
   (d); a per-contributor ledger (e); or keep equal shares (a).
2. **Q2. The external agent:** excluded from the release, the spill and the binding side (recommended, §5.2);
   excluded from the release and the spill only; or included as today.
3. **Q3. `M6.3c`'s death clause** follows the same rule, so a dead bearer's hold goes back by bond energy to family
   members only (recommended); or stays as decided 2026-10-14.
4. **Q4. All recipients at zero bond energy:** equal shares (recommended); or the hold stays with the bearer.
5. **Q5. The binding side** stays as it is, each partner giving `s ×` their own excess (recommended); or is also
   weighted by the tie.

## 9. Order of work, once decided

1. The Holmes–Rahe Q3 asked again (T-013), before anything is written.
2. One spec revision for G1, G2 and the Holmes–Rahe sizes: here `M7.D.2a`, `M7.D.2d`, `M6.4`, `M6.3c`;
   `docs/model_explainer.md`'s lock-in paragraph; `config/bowen/constants_changes.md` records that G2 adds no
   constant.
3. Build, tests first (§6), in `src/bowen/engine/symptoms.py` (`release_bound`, `lock_in`, `_partners` or a family-only
   variant), and `event_effects.py` and `life_course.py`, where a named or a dead bearer's hold is returned.
4. One regeneration of every record for the batch, then the report of §6.
