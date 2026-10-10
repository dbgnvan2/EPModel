# EPModel v2 — the reference family (spec `M2.A`, revision 13)

> The owner's family, 2026-10-10 (`docs/PROPOSAL — RUN VISUALIZATION MAPPING.md` §0, D5):
> - grandparents (4): Teodor and Ana, Marta's parents; Sofia and Viktor, Ravi's parents;
> - parents (2): Ravi and Marta;
> - the parents' siblings (2): Iris, Marta's sister; Dario, Ravi's brother, **cut off from Ravi at the start**;
> - children (3): Leo (launched), Nadia and Pia;
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
> Ravi–Dario mirrors the Ana–Bruno tie it replaces: high bond energy, cut off, not interactive (`M1.B.3`).
> Iris and Dario have comparable, low contact and differ in bond energy (`M2.A.1`).
> Leo's sex was declared male by the owner on 2026-10-10 (`M2.A.0i`).
>
> Not yet run: Phase D builds what this family needs. Until then it is read by the run visualization only
> to draw the people a run does not simulate (`tools/famdiag_export.py`).

instance_id: phase_d
grade: [I]
nuclear_household: ravi_marta
undifferentiation_budget: 90

## People

| id | name | generation | age | sex | household | basic_level | functional_level | chronic_anxiety | sibling_rank | sibship_size | financially_dependent | role | channel_prior |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `teodor` | Teodor | 1 | 81 | male | teodor_ana | 36 | 36 | 42 | 1 | 2 | no | member | physical |
| `ana` | Ana | 1 | 78 | female | teodor_ana | 37 | 37 | 38 | 1 | 2 | no | member | physical |
| `sofia` | Sofia | 1 | 76 | female | sofia_viktor | 41 | 41 | 35 | 1 | 1 | no | member | physical |
| `viktor` | Viktor | 1 | 78 | male | sofia_viktor | 40 | 40 | 36 | 2 | 2 | no | member | physical |
| `ravi` | Ravi | 2 | 52 | male | ravi_marta | 39 | 39 | 44 | 1 | 3 | no | member | physical |
| `marta` | Marta | 2 | 50 | female | ravi_marta | 40 | 40 | 46 | 1 | 2 | no | member | physical |
| `iris` | Iris | 2 | 47 | female | iris | 43 | 43 | 33 | 2 | 2 | no | member | social |
| `dario` | Dario | 2 | 49 | male | dario | 36 | 36 | 50 | 2 | 3 | no | member | mental |
| `leo` | Leo | 3 | 22 | male | leo | 36 | 36 | 48 | 1 | 3 | no | member | social |
| `nadia` | Nadia | 3 | 17 | female | ravi_marta | 41 | 41 | 52 | 2 | 3 | yes | member | mental |
| `pia` | Pia | 3 | 14 | female | ravi_marta | 44 | 44 | 37 | 3 | 3 | yes | member | social |
| `halim` | Dr Halim | 2 | 55 | male | halim | 60 | 60 | 30 | — | — | no | external | physical |

## Ties

| a | b | relation | conductance | bond_energy | latency | tie_state | interactive |
|---|---|---|---|---|---|---|---|
| `ravi` | `marta` | spouse | 1.0 | 60 | 1 | ordinary | yes |
| `ravi` | `leo` | parent_of | 0.7 | 45 | 1 | ordinary | yes |
| `ravi` | `nadia` | parent_of | 1.0 | 50 | 1 | ordinary | yes |
| `ravi` | `pia` | parent_of | 1.0 | 50 | 1 | ordinary | yes |
| `marta` | `leo` | parent_of | 0.7 | 45 | 1 | ordinary | yes |
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
| `ravi` | `dario` | sibling | 0.8 | 55 | 1 | cut_off | no |
| `halim` | `ravi` | professional | 0.3 | 10 | 1 | ordinary | yes |
| `halim` | `marta` | professional | 0.3 | 10 | 1 | ordinary | yes |
