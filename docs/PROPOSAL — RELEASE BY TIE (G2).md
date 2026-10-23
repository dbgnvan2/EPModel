# Proposal — release by tie (version 2, G2)

> **Status: DECIDED as recommended under the owner's instruction of 2026-10-22, pending the owner's review (Q1–Q2 in
> §8); drafted the same day. Version 2; nothing built, no spec revision.** It turns the owner's G2 question,
> recorded in `docs/version_2_plan.md` (stage 3, group G, 2026-10-18), into a rule, tests and no new criterion. The
> owner's words are quoted in §1. Every form and number here is Claude's recommendation, graded `[I]`. It changes what
> a run computes, so it moves the engine hash; by G1's Q9 it is built in one batch with G1 and the Holmes–Rahe event
> sizes, with one regeneration.

## 1. What the owner asked (2026-10-18)

From `docs/version_2_plan.md`, group G, verbatim:

> **G2. Releases split equally, ignoring the tie.** `release_bound` and the lock-in spill divide by the number of
> interactive partners (`symptoms.py:147`, `symptoms.py:198`). The coach, whose tie to Marta has conductance 0.3 and
> bond energy 10 (`config/bowen/family_phase_c.md`), took the same 63.6 as Nadia. The learner's cross-person term
> already weights by bond energy (`M4.D.6e`), and `M1.E.7e` requires the coach's tie to stay thin. Weight by bond
> energy or conductance? Should a family member's symptom release reach an external agent at all?

The observation behind it (the same page): in the seed-7 chart's week 73, lock-in past its turn "released her
symptom-bound stock (95.3) and spilled her excess above usual (222.4) to her five interactive partners in equal shares,
the coach among them."

## 2. What the code does today

Checked 2026-10-22 against the working tree (`src/bowen/engine/symptoms.py`).

- **Who receives.** `_partners` (line 131) lists every living person on an interactive tie to the bearer, with no
  regard to the tie's bond energy, conductance or the partner's role. The coach, on a tie of bond energy 10, counts as
  Nadia does on a tie of bond energy 50.
- **Three releases, all equal splits.**
  1. `release_bound` (lines 136–150): the hold goes back when the symptom ends, when a `binder_unavailable` names
     its channel (`event_effects.py:117–125`, `M1.F.9`), and past the turn (`lock_in`, line 194, a share of it each
     week, `M7.D.2d`). `each = amount / len(partners)` (line 147).
  2. The lock-in spill (lines 196–203): past the turn, the same share of the bearer's own excess above usual goes to
     the partners, `each = spill / len(partners)` (line 198).
  3. The intake is not a split: below the turn each partner gives `s ×` their own excess above usual (lines 180–186),
     so what a partner gives already depends on that partner, and nothing on the tie.
- **What the split ignores.** The tie's `bond_energy` (`M1.B.2`, how much the tie matters) and `conductance`
  (`M1.B.2`, how much passes along it). Every other by-person distribution in the engine reads the tie:
  - a dead person's anxiety goes to the living family members in proportion to bond energy, and an external agent
    takes no share (`M6.3b`, `life_course.die`, `_member`);
  - the learner's cross-person relief is weighted by bond energy (`M4.D.6e`, `learner.py:101`);
  - a nodal event's reach through a tie is bond energy over 100 (`M1.A.4k`), and the Holmes–Rahe proposal's E1 scales
    an event's arrival by the same (`docs/PROPOSAL — EVENT SIZES FROM THE HOLMES–RAHE SCALE.md`, §3).
- **The coach.** `M1.E.7e` requires the coach tie to stay thin and never accumulate investment. Today a symptom's
  release reaches the coach as it reaches a child, so a coached family's release loads the coach, who then acts on it
  as any person does (`M1.E.4`, `external.py`). `M6.3b` already excludes external agents from a death's transfer.

## 3. The rule

One rule, in three places, no new stock and no new mechanism: `lock_in`'s records keep their name.

### 3.1 A release is split by bond energy, among family members (Q1, Q2)

When a bearer's hold goes back to the partners (the three releases in §2, items 1 and 2), each receiving partner's
share is their tie's `bond_energy` divided by the sum of bond energy over the receiving ties. The receiving ties are
the bearer's interactive ties to living **family members** (`role = MEMBER`); an external agent on an interactive tie
receives nothing. If the receiving ties' bond energy sums to zero, the split is equal among them (the same fallback
`M6.3b` uses when no tie carries weight). If there is no receiving tie, nothing moves and the hold stays with the
bearer, as today when `_partners` is empty.

- **Grades.** The direction, that what a tie carries is set by the tie and not by the number of ties, is `[D]`
  (`KS10.4` `[K]`, the shock wave reaching four nuclear families through their ties to the dead; `FE07.4` `[K]`, the
  projection loop's reinforcement arriving through the parent–child tie, which `M4.D.6e` already builds as a
  bond-energy weight). The weight being bond energy rather than conductance, the exclusion of external agents, and the
  zero-sum fallback are `[I]`.
- **Why bond energy, not conductance** (the owner's first question). Conductance (`M1.B.2`) is a rate: how fast an
  event passes along the tie, read by delivery latency and the standing load (`M4.C.1c`). Bond energy is how much the
  tie matters, and is what every other distribution in the engine reads (§2). A release is not an event travelling
  the tie at a speed; it is the family's focus letting go of what it held, and where it lands is a matter of who the
  bearer is bound to. Using the same weight as `M4.D.6e` and `M6.3b` also keeps one meaning for "the tie's weight"
  across the engine. Conductance already does its work on the way in: the partners' excess that the hold took was
  built from events that arrived through their ties at their conductance.
- **Why family members only** (the owner's second question). `M1.E.7e` says the coach tie stays thin and the coach
  must not become a lender of pseudo-self; a release that loads the coach by the count of ties makes a coached family
  calmer per release than an uncoached one, with the coach as a sink, which is the artefact `M11.C.17`'s no-drift arm
  must not be able to exploit. `M6.3b` already draws the line at `role = MEMBER` for a death. On the phase_c family
  the coach's tie is bond energy 10 against 50 and 60 on the family ties, so weighting alone would cut the coach's
  share from a fifth to under a twentieth; the exclusion takes it to nothing and keeps the coach's load what the
  coach's own exchanges make it.
- **The intake is unchanged.** Below the turn each partner gives `s ×` their own excess (§2, item 3). It is already
  per partner and reads the partner's own state; no weight is added. A weighted intake would make the coach give less
  than a child does, which the rule above does not need, and `M7.D.2a`'s "the family calms around the symptom" is the
  partners' own excess being held, not a share of a pool.

### 3.2 What does not change

- `symptom_bound` is the same stock; `lock_in` and `binder_unavailable` records log the same fields; `M6.4`'s table
  gains no row. The ledger's balance check (`M6.I.6`) holds as before: the amounts sum to what left the hold.
- The coach is still a partner for everything else: the moves, the position gate, the learner's witness term. Only
  the symptom releases skip the coach.
- `M11.C.12` reads marital conflict after a remission on the dyad, where the two spouses are each other's only
  partner: the split is one share either way, so its arms do not move by this rule. `M11.C.22` reads family relief on
  the nuclear-four fixture, where every tie is a family tie of equal bond energy (its fixture file): the split is equal
  there too. `M11.C.17` (coached arm) and the phase_c runs (`M11.C.1`, the trace, the chart) are where the rule shows.

### 3.3 The §19 check

`model_explainer.md` §19: patterns emerge; a rule that writes one of them is a defect. This rule changes where a
release lands and by how much per receiver. It does not name a symptom bearer, a projection target, a triangle or a
tie state, and no outcome in §19's list (conflict, over/underfunctioning, distance, cutoff, projection, fixed
triangles) is written by it. A release reaching a child in proportion to the child's bond is not a projection rule:
projection is the parents' focus choosing a child (`M7.E.1c`), and this rule only distributes what the family already
held once that focus lets go.

## 4. Criteria

No new criterion. The rule is a premise (`M11.5`): it states where a release lands, so a test that reads the shares
confirms the code and is never a finding. The ensemble criteria it may move are listed in §3.2; the before/after
report (§6) says which did.

## 5. Acceptance criteria and tests

Test names carry the spec IDs assigned at the revision; `<id>` marks where. All in `tests/bowen/test_symptoms.py`
unless named.

| # | Criterion | Test | Mutation that must make it fail |
|---|---|---|---|
| B1 | On the phase_c family, a release from Marta's hold gives Nadia (bond 50) five times the coach's would-be share, and the coach receives nothing; the amounts sum to what left the hold. | `::test_<id>_release_is_split_by_bond_energy_among_members` | the split restored to `amount / len(partners)` |
| B2 | The lock-in spill past the turn is split the same way: the coach receives none of the bearer's excess. | `::test_<id>_spill_is_split_by_bond_energy_among_members` | the spill restored to `spill / len(partners)` |
| B3 | A `binder_unavailable` naming the channel returns the whole hold by the same split (the third site). | `::test_<id>_binder_unavailable_release_uses_the_same_split` | `event_effects` calling an equal split |
| B4 | With the family ties' bond energy at zero the split is equal among the members; with no member on an interactive tie the hold stays with the bearer. | `::test_<id>_zero_bond_falls_back_to_equal_and_no_member_keeps_the_hold` | the fallback removed (division by zero) |
| B5 | The ledger balances on every release (`M6.I.6`): what left the hold equals what the members gained. | `::test_m7d2a_the_family_calms_around_a_mild_symptom` and `::test_m7d2d_past_the_turn_the_symptom_gives_back_what_it_held` (existing, extended with the coach present) | a share dropped on the floor |
| B6 | Dirty state (P8): a release after a member has died reads the living members only, and the dead member's tie takes no share. | `::test_<id>_release_skips_the_dead` | the `alive` test removed |
| B7 | The intake below the turn is unchanged: a partner at their usual level gives nothing, and what a partner gives reads that partner's excess, not the tie. | `::test_m7d2a_the_family_calms_around_a_mild_symptom` (existing, asserted on two partners of unequal bond) | an intake weight added |

B1–B4 and B6 are unit tests of a set mechanism with invented constants; none is a finding (`M11.5`).

## 6. Order of work, once approved

1. Tests B1–B7 first; B1 is the highest-impact rule and is written before the code.
2. `symptoms.py`: one helper, `_release_shares(state, pid)`, returning `(partner, weight)` over the receiving ties;
   `release_bound` and the spill use it. `_partners` stays for the intake.
3. Spec revision (with G1 and the event sizes, one revision): `M7.D.2a`'s last sentences and `M7.D.2d`'s "split
   equally" become the rule in §3.1; `M1.F.9`'s symptom branch names it; `M1.E.7e` gains a sentence that symptom
   releases skip the coach; the `M14.A` register's `symptom_bound` row names the split.
4. `docs/model_explainer.md` §7's lock-in paragraph: the split, graded as in §3.1.
5. Before/after report on `M11.C.12`, `M11.C.22`, `M11.C.17` and `M11.C.1`, from the regenerated records, in the
   build's changes page (G1 §6, step 5).

## 7. Conflicts with the spec and the corpus, stated

- **`M7.D.2a` and `M7.D.2d`** say "split equally". Both sentences are the build's wording of 2026-10-12 (revision
  24), graded `[I]`; no source states a split. Changing them changes an invented form, not a sourced one.
- **`M1.F.9`** returns a binder's hold "to its partners": unchanged in direction, narrowed to family members.
- **No corpus statement** names who receives a family's released anxiety. The direction (§3.1, `[D]`) is read from
  case material in which what a tie carries is set by the tie; the proposal does not claim more.

## 8. Questions for the owner

1. **Q1. Weight by bond energy or conductance?** Recommended: bond energy, for the reasons in §3.1. **Decided as
   recommended under the owner's instruction of 2026-10-22, pending the owner's review.**
2. **Q2. Should a family member's symptom release reach an external agent at all?** Recommended: no; family members
   only, as `M6.3b` already does for a death. **Decided as recommended under the owner's instruction of 2026-10-22,
   pending the owner's review.**
