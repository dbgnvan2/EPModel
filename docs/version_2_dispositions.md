# Version 2: dispositions

**Status: begun 2026-10-29, pending owner review (`docs/version_2_definition_plan.md` §3); on 2026-10-30 each of the five rows W1 waits on carries Claude's recommendation, made under the owner's instruction of that night to proceed without stopping (`docs/version_2_definition_plan.md` §8).** Every row below is Claude's recommendation,
pending the owner's review; none is the owner's until its `decided` cell says `owner` and a date.

`docs/version_1_dispositions.md` is frozen at the tag `version-1`. This file holds the dispositions made in version 2.
`tools/version_2_status.py` reads both and writes `docs/version_2_status.md`. It is parsed strictly: a row naming a
key no record has, an unknown disposition, a verdict other than PASS or FAIL, an `accepted finding` on a PASS, or a
`decided` cell in neither form below raises.

- `criterion` is a record row's key, a variant named as its record names it.
- `disposition` is one of:
  - `accepted finding`: a failing **composite** or **mixed** criterion accepted as a finding about the model (W1). As
    in version 1, the tool refuses it on a failing premise, a check, or a mixed criterion's failing premise limb (W2).
  - `verdict reviewed`: the owner has reviewed a verdict that is not the one the tag records (T-122). It changes no
    other status: a pass still needs its mutation (W3).
  - `note`: carried beside the status without changing it.
- `verdict` is the verdict the disposition was decided on. The disposition applies only while the row's verdict is
  that one; after a regeneration that moves it, the page shows the row as not applied, for the owner to decide again.
- `decided` is either `owner YYYY-MM-DD` or `Claude's recommendation YYYY-MM-DD, pending owner review`. Any row the
  owner made is the owner's review of the verdict it names.
- `reason` gives why, and where it is traced. It does not repeat the record's verdict numbers, which the status page
  shows from the records; a number in a reason is a diagnosis, marked ad hoc or with its source.

| criterion | disposition | verdict | decided | reason |
|---|---|---|---|---|
| `M11.C.49` | accepted finding | FAIL | Claude's recommendation 2026-10-29, pending owner review | New at revision 39 (group F). The premise limb holds: the target feels a lower-level mover's I-position as more impingement. The composite limbs do not resolve: the family's summed excess and the F–M tie's `CONFLICT` share after the act do not differ by the mover's level at the settings fixed before the row first ran, which are not tuned to it (`M10.B.4`; T-137, 150 seeds on Mac Mini1, `docs/review/stage_3_step_8_criteria.md`). So in the model the demand share an I-position carries at a lower level reaches the target and does not spread to the family within six weeks. Its sweep ranges are not declared (T-144); W4 does not need them while it fails. |
| `M11.C.48` | accepted finding | FAIL | Claude's recommendation 2026-10-30, pending owner review | Fails since revision 43 (T-123), when a first cutoff with no uncut tie in its week stopped counting with a baseline of 0. The pass recorded at revision 36 rested on that bias (`docs/readouts_phase_c.md`, "Negative runs"). No rule writes the run: `CUTOFF`'s value is learned from felt relief (`M4.D.6`). So in the model a tie is not cut off after a longer run of negative contacts than its family's uncut ties stand on in the same week. The settings are those fixed before the row first ran (`M10.B.4`), not tuned to it. |
| `M11.C.48[phase_d]` | accepted finding | FAIL | Claude's recommendation 2026-10-30, pending owner review | The same criterion on the twelve-person family over 40 years, with the same change and the same reading as `M11.C.48`; a larger share of its first cutoffs is unmatched than on the Phase B family (284 of 749 against 139 of 700 at the re-sweep of 2026-10-23, T-123's record; the record's `unmatched_cutoffs` now). |
| `M11.C.34` | verdict reviewed | PASS | Claude's recommendation 2026-10-30, pending owner review | Moved FAIL → PASS at revision 41 (T-131 (b), T-133 (b), T-145 (a)); no rule writes the others' reaction toward the bearer, and the composite reading is the owner's (2026-10-17). Its declared mutant, `M4.D.1a`'s mixing weight made blind to the responder's level, turns it red (T-162, revision 43), so the pass rests on the mechanism the row names. Recommended as the model's result under revision 41's rules, with its weakness stated: the sweep holds the direction in 1 of 8 samples, so it holds at the settings fixed before the row first ran and not across the declared ranges (the sweep is a measurement, not a verdict, `M11.5`). |
| `M11.C.21` | note | PASS | Claude's recommendation 2026-10-30, pending owner review | Not recommended for review as a final pass. The pass rests on the arm's design (T-161): both parents' scripted `OVERFUNCTION` toward C1 goes on after their collapse, which neither of the row's two instances has (both parents collapsed; both parents died). Recommended instead: T-161 (b), the scripted lending stops at `t0` in **both** arms, so what the parents do after `t0` is their own policy's and the arms differ by the drop alone. Measured on this engine with form (b), ad hoc (100 seeds, 2026-10-30, `tools/stage_3_targeted_run.py` in a scratch copy, no record): FAIL, `target_rise` −0.72 ± 1.09, p 0.94; so after that batch the row is expected to need an accepted-finding decision. Version 1's disposition (an accepted finding, owner 2026-10-17) was decided on a FAIL under the arm as it is. |
