# EPModel v2 — fixture: a dyad (Phase C plan D1)

> Two spouses of comparable level (`M2.A.0e`), the pair `M11.C.4`, `M11.C.19` and `M11.C.25` read.
>
> **Every value is invented, `[I]`.** A fixture is an initial state (`M17.D.3`): criteria that need a
> controlled structure name their fixture in their test. Each nuclear adult has one parent outside the
> household, because a closed nuclear family cannot compute its own driving term (`M2.3`, `M1.D.8`); the
> parents' ties are thin and they are otherwise quiet.

instance_id: fixture_dyad
grade: [I]
nuclear_household: home
undifferentiation_budget: 60

## People

| id | name | generation | age | sex | household | basic_level | functional_level | chronic_anxiety | sibling_rank | sibship_size | financially_dependent | role | channel_prior |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `a` | A | 2 | 40 | female | home | 40 | 40 | 45 | — | — | no | member | physical |
| `b` | B | 2 | 40 | male | home | 40 | 40 | 45 | — | — | no | member | physical |
| `pa` | A's parent | 1 | 68 | female | pa | 40 | 40 | 40 | — | — | no | member | physical |
| `pb` | B's parent | 1 | 68 | male | pb | 40 | 40 | 40 | — | — | no | member | physical |

## Ties

| a | b | relation | conductance | bond_energy | latency | tie_state | interactive |
|---|---|---|---|---|---|---|---|
| `a` | `b` | spouse | 1.0 | 60 | 1 | ordinary | yes |
| `pa` | `a` | parent_of | 0.4 | 40 | 1 | ordinary | yes |
| `pb` | `b` | parent_of | 0.4 | 40 | 1 | ordinary | yes |
