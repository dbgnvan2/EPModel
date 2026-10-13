# EPModel v2 — the reference family's partners and their children (spec `M7.C.1h`, `M7.C.1i`, revision 28)

> The owner, 2026-10-13: "Provide settings for partners." Partners are declared here and never invented by the engine
> (`M7.C.1h`). **Every value is invented** (`M2`), drafted by Claude for the owner to edit:
> - who pairs and at what age: Leo at 28, Nadia at 27, Pia at 29 (years 19, 23 and 28 of a run that starts with them
>   at 9, 4 and 1);
> - the partner's age gap, sex, chronic anxiety, sibling position and channel prior; their basic level is not
>   declared, because it is drawn within `M2.A.0e`'s ±1 of the member's level at pairing;
> - each partner's family of origin, a holder who keeps only bond energy and never acts (`M1.D.8`, in `M2.3`'s form);
> - two children per couple, born two and five years after pairing; `draw` means the sex or channel is drawn from
>   the run's stream.
>
> The tie values below are the starting values of the ties the life course adds: the spouse tie, each parent's tie to
> a newborn, a newborn's tie to each older sibling, and the partner's tie to their family of origin. They mirror the
> reference family's own (`family_phase_d.md`).

instance_id: phase_d_partners
grade: [I]
spouse_conductance: 1.0
spouse_bond_energy: 60
parent_conductance: 1.0
parent_bond_energy: 50
sibling_conductance: 0.4
sibling_bond_energy: 35
origin_conductance: 0.5
origin_bond_energy: 35
tie_latency: 1

## Partners

| id | name | partner_of | pair_age | age_gap | sex | chronic_anxiety | sibling_rank | sibship_size | channel_prior | origin_id | origin_sex | children_at_years | children_sexes | children_channels |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `sara` | Sara | `leo` | 28 | -1 | female | 44 | 2 | 2 | physical | `sara_parent` | female | 2; 5 | draw; draw | draw; draw |
| `omar` | Omar | `nadia` | 27 | 2 | male | 46 | 1 | 3 | social | `omar_parent` | male | 2; 5 | draw; draw | draw; draw |
| `theo` | Theo | `pia` | 29 | 1 | male | 40 | 2 | 3 | mental | `theo_parent` | female | 2; 5 | draw; draw | draw; draw |
