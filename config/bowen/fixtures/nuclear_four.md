# EPModel v2 — fixture: a nuclear four (Phase C plan D1)

> Two parents and two children, as `M11.C.14` needs.
>
> **Every value is invented, `[I]`.** A fixture is an initial state (`M17.D.3`): criteria that need a
> controlled structure name their fixture in their test. Each nuclear adult has one parent outside the
> household, because a closed nuclear family cannot compute its own driving term (`M2.3`, `M1.D.8`); the
> parents' ties are thin and they are otherwise quiet.

instance_id: fixture_nuclear_four
grade: [I]
nuclear_household: home
undifferentiation_budget: 60

## People

| id | name | generation | age | sex | household | basic_level | functional_level | chronic_anxiety | sibling_rank | sibship_size | financially_dependent | role | channel_prior |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `f` | F | 2 | 42 | male | home | 40 | 40 | 45 | — | — | no | member | physical |
| `m` | M | 2 | 41 | female | home | 40 | 40 | 45 | — | — | no | member | physical |
| `c1` | C1 | 3 | 16 | female | home | 40 | 40 | 45 | — | — | yes | member | mental |
| `c2` | C2 | 3 | 14 | male | home | 42 | 42 | 40 | — | — | yes | member | social |
| `pf` | F's parent | 1 | 70 | male | pf | 40 | 40 | 40 | — | — | no | member | physical |
| `pm` | M's parent | 1 | 69 | female | pm | 40 | 40 | 40 | — | — | no | member | physical |

## Ties

| a | b | relation | conductance | bond_energy | latency | tie_state | interactive |
|---|---|---|---|---|---|---|---|
| `f` | `m` | spouse | 1.0 | 60 | 1 | ordinary | yes |
| `f` | `c1` | parent_of | 1.0 | 50 | 1 | ordinary | yes |
| `m` | `c1` | parent_of | 1.0 | 50 | 1 | ordinary | yes |
| `f` | `c2` | parent_of | 1.0 | 50 | 1 | ordinary | yes |
| `m` | `c2` | parent_of | 1.0 | 50 | 1 | ordinary | yes |
| `pf` | `f` | parent_of | 0.4 | 40 | 1 | ordinary | yes |
| `pm` | `m` | parent_of | 0.4 | 40 | 1 | ordinary | yes |
