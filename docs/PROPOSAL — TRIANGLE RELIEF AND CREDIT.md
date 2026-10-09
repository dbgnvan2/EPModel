# Proposal — what a `TRIANGLE` relieves, and what the seeker learns from it

> Written 2026-10-09, from the owner's answer to Q3 of `docs/DECISIONS — PHASE C FAILING.md`. **A proposal for
> approval; no code until approved.** It changes the spec (`M1.C.1`, `M4.D.6e`), so it changes the engine and needs
> one full rerun of the records (about an hour).

## The owner's account (2026-10-09)

1. Either or both people whose anxiety rose in an exchange can triangle in a third person, and usually not the same
   one: A may seek C while B seeks D.
2. Usually one of A and B is more anxious, and that one seeks a third.
3. If A seeks C, A's anxiety goes down, provided C is a positive experience.
4. C may be on B's side, or may be very anxious themselves; then C does not help A.

## How the model differs today

Code: `triangle_transfer` in `src/bowen/engine/moves.py`. Credit: `M4.D.6e` via the learner.

| Owner's account | Model now |
|---|---|
| Either of the pair may triangle, each toward their own third | Either may; each `TRIANGLE` names its own target. **Matches.** |
| The more anxious one seeks a third | Selection rises with the seeker's anxiety through the policy, not by rule. **Matches in direction**; no rule is needed (rules that write outcomes are defects, `model_explainer.md` §19). |
| A's anxiety goes down if C is a positive experience | A **and the partner B** are both relieved, by a fixed fraction of each one's excess, scaled by the three people's levels (`routing_capacity`). C's response plays no part. |
| If C is on B's side, or very anxious, A gets no help | C's alignment and C's own anxiety do not affect A's relief. |
| What A learns from is A's own relief | A is credited with A's own change **plus** 0.5 × the others' mean change (`M4.D.6e`), and the others include C's rise (+8.9 in `q3-act-effects`). That term is why triangling is not reinforced (`M11.C.42`, `.45`). `M4.D.6e` was written for projection (a child learns that calming the mother is the reward), not for recruiting a third. |

## Proposed changes

**P1 — the seeker's relief depends on C's response (`M1.C.1`).** The amount moved from A is scaled by how far C
can take it: lower when C's own excess anxiety is high, and lower when C is aligned with B rather than with A. A
proposed form, `[I]`:

    help(C) = (1 − C's excess / SCALE_MAX) × (A–C bond / (A–C bond + B–C bond))

Here a bond is the tie's `bond_energy`, and "on B's side" means C's tie to B is stronger than to A. Every input is
state the model already holds; no new constant.

**P2 — what the seeker learns from (`M4.D.6e`).** A `TRIANGLE`'s reinforcement signal is the seeker's own felt
change only. The recruited third's change is left out of the cross-person term for this act. The cross-person term
stays for every other act, where `M4.D.6e`'s projection account applies.

**Question for the owner — B's relief.** When A seeks C, does B's anxiety also go down? The twosome's tension
eases, but your account names only A. Today B is relieved too, and `M11.C.3` asserts that a triangle "relieves the
pair". Options:
- (a) Only A is relieved. Then `M11.C.3` is restated to "relieves the seeker and costs the third".
- (b) A is relieved by C's help, and B by a smaller share, as the strain between A and B eases.
- (c) Both are relieved as now, scaled by C's help.

## Acceptance criteria and tests (if approved)

| ID | Criterion | Test |
|---|---|---|
| P1.1 | A's relief falls as C's excess rises, others equal | `tests/bowen/test_moves.py::test_m1c1_anxious_third_helps_less` |
| P1.2 | A's relief falls as C's bond to B rises against C's bond to A | `tests/bowen/test_moves.py::test_m1c1_third_aligned_with_partner_helps_less` |
| P1.3 | B's relief follows the owner's answer to the question above | `tests/bowen/test_moves.py::test_m1c1_partner_relief_as_decided` |
| P2.1 | A `TRIANGLE`'s signal excludes the recruited third's change; other acts keep the cross-person term | `tests/bowen/test_learner.py::test_m4d6e_triangle_credit_excludes_recruited_third` |
| P2.2 | Adversarial: a `TRIANGLE` that relieves A but loads C heavily is still reinforced; one where C is anxious and aligned with B is not | `tests/bowen/test_learner.py::test_m4d6e_triangle_reinforced_only_when_third_helps` |
| — | Mutation: `triangle-help-ignored` (P1 removed) and `triangle-credit-cross-person` (P2 removed) are named mutants; each must turn at least one of `M11.C.3`, `.42`, `.45` red | `docs/phase_c_mutation_record.md` |

Order: decide B's relief → amend the spec (`M1.C.1`, `M4.D.6e`, perhaps `M11.C.3`) → tests first → code → one
rerun of every record → report. This is a change after the constants freeze, and it is logged as one.

## What it would not do

- It sets no outcome. Whether triangling is learned, and with whom, stays a result of relief and learning.
- It adds no constant: P1 reads state the model already holds.
