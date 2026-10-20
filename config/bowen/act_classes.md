# EPModel v2 — which acts belong to each act class the engine and the policy read

> Read by `src/bowen/io/load.py` into `EventKinds` (`side_taking`, `family_to_external`, `external_repertoire`), as
> `nodal_events.md` is read into `EventKinds.nodal`; the engine reads them from the run's kinds and the policy from the
> kinds it is given, so neither holds a set of act names (version 2 stage 1, S1.3; rule 9). Each act must be a declared
> move, and each class must appear once; a misspelt act fails at load (`M10.B.2`).
>
> The two policy classes are the spec's own lists: `M5.B.4` names the family→external moves and `M5.B.5` the external
> agent's repertoire (`M1.E.3`'s and `M1.E.4`'s acts are Phase D and not yet moves). `side_taking` is revision 32's
> choice for `M8.3`, approved by the owner 2026-10-17: `SPLIT`, consulting the professional without telling the
> others (Ch19 · L19.3), is the move that draws an external agent onto its sender's side.

| class | acts | used by |
|---|---|---|
| side_taking | SPLIT | `external.coach_side`: the member an external agent has been drawn in on (`M8.3`) |
| family_to_external | SPLIT, FRAME_AMBIGUITY, DISPLACE | `policy.legal_outcomes`: the moves a family member makes toward an external agent, and only toward one (`M5.B.4`) |
| external_repertoire | STAY-IN-CONTACT, I-POSITION, OVERFUNCTION, TRIANGLE, CUTOFF | `policy.legal_outcomes`: the only moves an external agent makes (`M5.B.5`) |
