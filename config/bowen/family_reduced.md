# EPModel v2 — the Phase B family (spec `M2.3`, `M2.3a`)

> Ravi, Marta, Nadia and Pia, plus Ana, Sofia and Bruno, who hold bond energy and
> act only by script. People's values come from the spec's `M2.A` table (Ravi's
> `basic_level` is 39, per `M2.A.0e`); sex per `M2.A.0i`.
>
> **Every value here is invented** (`M2`). Values not in `M2.A` were chosen at
> Phase B step 5 and may be changed before the constants are frozen (`M10.B.4`):
>
> - `functional_level` equals `basic_level` at `t0` (swing zero, `M1.A.5a`).
> - Households: the nuclear four share one; Ana, Sofia and Bruno each have their own
>   (Sofia "lives 200 miles away", `M2.A`; Ana's household with Teodor is outside this instance).
> - Ties: conductance, bond energy (0–100) and latency (whole weeks, `M3.C.2`).
>   Ana–Bruno is cut off: no events, high bond energy (`M1.B.3`), so not interactive.
>   Ravi–Sofia has latency 2 for distance; conductance does not depend on distance (`M1.B.2`).
> - The undifferentiation budget starts at 0. The spec gives no relation between the
>   budget and anxiety (external review A3); in Phase B only `binder_unavailable` writes it.
>
> Relations: `parent_of` reads "a is the parent of b". A person's family-of-origin
> ties are those to a parent or a sibling (`M1.D.8`).

instance_id: phase_b_reduced
grade: [I]
nuclear_household: ravi_marta
undifferentiation_budget: 0

## People

| id | name | generation | age | sex | household | basic_level | functional_level | chronic_anxiety | sibling_rank | sibship_size | financially_dependent | role |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ravi` | Ravi | 2 | 52 | male | ravi_marta | 39 | 39 | 44 | 1 | 3 | no | member |
| `marta` | Marta | 2 | 50 | female | ravi_marta | 40 | 40 | 46 | 1 | 2 | no | member |
| `nadia` | Nadia | 3 | 17 | female | ravi_marta | 41 | 41 | 52 | 2 | 3 | yes | member |
| `pia` | Pia | 3 | 14 | female | ravi_marta | 44 | 44 | 37 | 3 | 3 | yes | member |
| `ana` | Ana | 1 | 78 | female | ana | 37 | 37 | 38 | 1 | 2 | no | member |
| `sofia` | Sofia | 1 | 76 | female | sofia | 41 | 41 | 35 | 1 | 1 | no | member |
| `bruno` | Bruno | 1 | 74 | male | bruno | 29 | 29 | 55 | 2 | 2 | no | member |

## Ties

| a | b | relation | conductance | bond_energy | latency | tie_state | interactive |
|---|---|---|---|---|---|---|---|
| `ravi` | `marta` | spouse | 1.0 | 60 | 1 | ordinary | yes |
| `ravi` | `nadia` | parent_of | 1.0 | 50 | 1 | ordinary | yes |
| `ravi` | `pia` | parent_of | 1.0 | 50 | 1 | ordinary | yes |
| `marta` | `nadia` | parent_of | 1.0 | 50 | 1 | ordinary | yes |
| `marta` | `pia` | parent_of | 1.0 | 50 | 1 | ordinary | yes |
| `ana` | `marta` | parent_of | 0.6 | 40 | 1 | ordinary | yes |
| `sofia` | `ravi` | parent_of | 0.6 | 40 | 2 | ordinary | yes |
| `ana` | `bruno` | sibling | 0.8 | 55 | 1 | cut_off | no |
