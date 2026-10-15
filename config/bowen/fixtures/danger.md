# EPModel v2 — fixture: danger (spec revision 30, `M11.C.47`)

> Two spouses and a child, each spouse with a parent outside the household, as the triad. `M11.C.47`'s arms set the
> spouses' level and the stressors: the owner (2026-10-14), "this might need a seed with a low level of differentiation
> and higher stressors." Only `f` moves: the arms differ in one person. The arm sets `f`'s level after loading, which a
> config would refuse for spouses (`M2.A.0e`); that is deliberate, the comparison is of one person.
>
> **Every value is invented, `[I]`.** A fixture is an initial state (`M17.D.3`). No one is declared dangerous: danger is
> read from what each person receives (`M1.B.3e`).

instance_id: fixture_danger
grade: [I]
nuclear_household: home
undifferentiation_budget: 60

## People

| id | name | generation | age | sex | household | basic_level | functional_level | chronic_anxiety | sibling_rank | sibship_size | financially_dependent | role | channel_prior |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `f` | F | 2 | 38 | male | home | 40 | 40 | 45 | — | — | no | member | physical |
| `m` | M | 2 | 37 | female | home | 40 | 40 | 45 | — | — | no | member | physical |
| `c` | C | 3 | 12 | female | home | 40 | 40 | 45 | — | — | yes | member | mental |
| `pf` | F's parent | 1 | 66 | male | pf | 40 | 40 | 40 | — | — | no | member | physical |
| `pm` | M's parent | 1 | 65 | female | pm | 40 | 40 | 40 | — | — | no | member | physical |

## Ties

| a | b | relation | conductance | bond_energy | latency | tie_state | interactive |
|---|---|---|---|---|---|---|---|
| `f` | `m` | spouse | 1.0 | 60 | 1 | ordinary | yes |
| `f` | `c` | parent_of | 1.0 | 50 | 1 | ordinary | yes |
| `m` | `c` | parent_of | 1.0 | 50 | 1 | ordinary | yes |
| `pf` | `f` | parent_of | 0.4 | 40 | 1 | ordinary | yes |
| `pm` | `m` | parent_of | 0.4 | 40 | 1 | ordinary | yes |
