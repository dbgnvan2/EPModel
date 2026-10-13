# EPModel v2 — the reference family (spec `M2.A`, revision 13)

> The owner's family, 2026-10-10 (`docs/PROPOSAL — RUN VISUALIZATION MAPPING.md` §0, D5):
> - grandparents (4): Teodor and Ana, Marta's parents; Sofia and Viktor, Ravi's parents;
> - parents (2): Ravi and Marta;
> - the parents' siblings (2): Iris, Marta's sister; Dario, Ravi's brother, **cut off from Ravi at the start**;
> - children (3): Leo, Nadia and Pia (Leo was launched before revision 28; at 9 he is at home);
> - the coach: Dr Halim (`role = external`, `M2.A.0`: not counted in the family).
>
> **Every value here is invented** (`M2`). Values in the spec's `M2.A` table are copied from it. The
> rest were chosen on 2026-10-10 and are drafted for the owner's review:
> - Viktor and Dario's rows, which the table now carries at `[I]`.
> - Every tie value.
> - Households.
> - Each `channel_prior`.
> - Dr Halim's values, which are Phase C's (`family_phase_c.md`).
>
> Teodor's `basic_level` is 36, not the table's former 34: 34 broke `M2.A.0e`'s ±1 pairing with Ana (37).
> Proposed at revision 13, as Ravi's 38 was corrected to 39; awaiting owner review.
>
> Ravi–Dario mirrors the Ana–Bruno tie it replaces: high bond energy, cut off (`M1.B.3`). From Phase D B4 a cut-off
> tie stays interactive and both members start rejecting the other (`M1.B.3c`); which brother cut off the other is
> not stated.
> Iris and Dario have comparable, low contact and differ in bond energy (`M2.A.1`).
> Leo's sex was declared male by the owner on 2026-10-10 (`M2.A.0i`).
>
> **Spec revision 28 (2026-10-13), the life course.** The owner's starting ages are three settings, below (`M2.A.0j`):
> the youngest of generation 3 starts at 1, generation 2 at 30, generation 1 at 55. Each person's `age_gap` is their
> gap to their generation's anchor: the youngest for generation 3, the eldest for generations 1 and 2 (Claude's
> reading). The gaps are the ages the family had before (Teodor 81 → 55, and so on). The children are below the
> fixation age, so their chronic anxiety is derived from their parents' (`M7.B.1a`), and their basic level is their
> parents' mean until it develops (`M7.E.1f`): both are `—`. Leo is at home, no longer launched, and his ties to his
> parents are now Nadia's and Pia's. Dr Halim is female and 35 (the coach's starting age, a constant) and is tied
> to the coachee only, Ravi (`M2.B.2`, owner 2026-10-13); the successor coach's sex is declared here. `earner` marks
> who brings income into their household (`M6.3h`): the adults. Partners and their children are in
> `partners_phase_d.md`. Changing a starting-age setting changes only ages (`M2.A.0j`).

instance_id: phase_d
grade: [I]
nuclear_household: ravi_marta
undifferentiation_budget: 90
start_age_g1: 55
start_age_g2: 30
start_age_youngest_g3: 1
partners: partners_phase_d.md
coachee: ravi
successor_coach_sex: female

## People

| id | name | generation | age_gap | sex | household | basic_level | functional_level | chronic_anxiety | sibling_rank | sibship_size | financially_dependent | earner | role | channel_prior |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `teodor` | Teodor | 1 | 0 | male | teodor_ana | 36 | 36 | 42 | 1 | 2 | no | yes | member | physical |
| `ana` | Ana | 1 | -3 | female | teodor_ana | 37 | 37 | 38 | 1 | 2 | no | yes | member | physical |
| `sofia` | Sofia | 1 | -5 | female | sofia_viktor | 41 | 41 | 35 | 1 | 1 | no | yes | member | physical |
| `viktor` | Viktor | 1 | -3 | male | sofia_viktor | 40 | 40 | 36 | 2 | 2 | no | yes | member | physical |
| `ravi` | Ravi | 2 | 0 | male | ravi_marta | 39 | 39 | 44 | 1 | 3 | no | yes | member | physical |
| `marta` | Marta | 2 | -2 | female | ravi_marta | 40 | 40 | 46 | 1 | 2 | no | yes | member | physical |
| `iris` | Iris | 2 | -5 | female | iris | 43 | 43 | 33 | 2 | 2 | no | yes | member | social |
| `dario` | Dario | 2 | -3 | male | dario | 36 | 36 | 50 | 2 | 3 | no | yes | member | mental |
| `leo` | Leo | 3 | 8 | male | ravi_marta | — | — | — | 1 | 3 | yes | no | member | social |
| `nadia` | Nadia | 3 | 3 | female | ravi_marta | — | — | — | 2 | 3 | yes | no | member | mental |
| `pia` | Pia | 3 | 0 | female | ravi_marta | — | — | — | 3 | 3 | yes | no | member | social |
| `halim` | Dr Halim | — | — | female | halim | 60 | 60 | 30 | — | — | no | yes | external | physical |

## Ties

| a | b | relation | conductance | bond_energy | latency | tie_state | interactive |
|---|---|---|---|---|---|---|---|
| `ravi` | `marta` | spouse | 1.0 | 60 | 1 | ordinary | yes |
| `ravi` | `leo` | parent_of | 1.0 | 50 | 1 | ordinary | yes |
| `ravi` | `nadia` | parent_of | 1.0 | 50 | 1 | ordinary | yes |
| `ravi` | `pia` | parent_of | 1.0 | 50 | 1 | ordinary | yes |
| `marta` | `leo` | parent_of | 1.0 | 50 | 1 | ordinary | yes |
| `marta` | `nadia` | parent_of | 1.0 | 50 | 1 | ordinary | yes |
| `marta` | `pia` | parent_of | 1.0 | 50 | 1 | ordinary | yes |
| `teodor` | `ana` | spouse | 1.0 | 55 | 1 | ordinary | yes |
| `ana` | `marta` | parent_of | 0.6 | 40 | 1 | ordinary | yes |
| `teodor` | `marta` | parent_of | 0.5 | 35 | 1 | ordinary | yes |
| `ana` | `iris` | parent_of | 0.5 | 35 | 1 | ordinary | yes |
| `teodor` | `iris` | parent_of | 0.4 | 30 | 1 | ordinary | yes |
| `sofia` | `viktor` | spouse | 1.0 | 55 | 1 | ordinary | yes |
| `sofia` | `ravi` | parent_of | 0.6 | 40 | 2 | ordinary | yes |
| `viktor` | `ravi` | parent_of | 0.5 | 35 | 2 | ordinary | yes |
| `sofia` | `dario` | parent_of | 0.6 | 40 | 2 | ordinary | yes |
| `viktor` | `dario` | parent_of | 0.5 | 35 | 2 | ordinary | yes |
| `marta` | `iris` | sibling | 0.4 | 35 | 2 | ordinary | yes |
| `ravi` | `dario` | sibling | 0.8 | 55 | 1 | cut_off | yes |
| `halim` | `ravi` | professional | 0.3 | 10 | 1 | ordinary | yes |
