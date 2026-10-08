# EPModel v2 — the Phase C family (plan D1)

> The Phase B seven (`family_reduced.md`, unchanged values) plus **Dr Halim**, the external agent
> (`M1.E.1`: a `Person` with `role = external`, the restricted repertoire of `M5.B.5`, and real ties).
> Under the policy every person selects; Dr Halim selects only in the weeks a session is scheduled.
>
> **Every value here is invented** (`M2`). Dr Halim's: basic level 60, so that his efficacy is high
> enough for contact to land at all (`M1.E.7d`'s coach quality); his own household; chronic anxiety 30;
> no family-of-origin tie (he is not a family member, and `M1.D.8` binds family adults only). His ties to
> Ravi and Marta are thin — conductance 0.3, bond energy 10 — because the coach tie must stay thin
> (`M1.E.7e`). All `[I]`.
>
> The undifferentiation budget is 90, `[I]` (Phase C step 10): the quantity the three sinks absorb
> (`M1.D.1`). The spec gives no magnitude; it starts unallocated and moves into the sinks as the family acts.
>
> Relations: as in `family_reduced.md`, plus `professional` for an external agent's ties.

instance_id: phase_c
grade: [I]
nuclear_household: ravi_marta
undifferentiation_budget: 90

## People

| id | name | generation | age | sex | household | basic_level | functional_level | chronic_anxiety | sibling_rank | sibship_size | financially_dependent | role | channel_prior |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ravi` | Ravi | 2 | 52 | male | ravi_marta | 39 | 39 | 44 | 1 | 3 | no | member | physical |
| `marta` | Marta | 2 | 50 | female | ravi_marta | 40 | 40 | 46 | 1 | 2 | no | member | physical |
| `nadia` | Nadia | 3 | 17 | female | ravi_marta | 41 | 41 | 52 | 2 | 3 | yes | member | mental |
| `pia` | Pia | 3 | 14 | female | ravi_marta | 44 | 44 | 37 | 3 | 3 | yes | member | social |
| `ana` | Ana | 1 | 78 | female | ana | 37 | 37 | 38 | 1 | 2 | no | member | physical |
| `sofia` | Sofia | 1 | 76 | female | sofia | 41 | 41 | 35 | 1 | 1 | no | member | physical |
| `bruno` | Bruno | 1 | 74 | male | bruno | 29 | 29 | 55 | 2 | 2 | no | member | social |
| `halim` | Dr Halim | 2 | 55 | male | halim | 60 | 60 | 30 | — | — | no | external | physical |

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
| `halim` | `ravi` | professional | 0.3 | 10 | 1 | ordinary | yes |
| `halim` | `marta` | professional | 0.3 | 10 | 1 | ordinary | yes |
