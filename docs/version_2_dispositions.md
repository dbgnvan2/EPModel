# Version 2: dispositions

**Status: begun 2026-10-29 (`docs/version_2_definition_plan.md` §3); on 2026-10-30 each of the five rows W1 waits on carried Claude's recommendation, made under the owner's instruction of that night to proceed without stopping, and on 2026-10-31 the owner decided each as recommended (`docs/version_2_definition_plan.md` §8). After spec revision 44 (2026-10-31) the owner decided `M11.C.21` and `M11.C.17` again on their FAIL, each an accepted finding, through `AskUserQuestion`.** A row is
the owner's when its `decided` cell says `owner` and a date; every row below is.

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
- `decided` is either `owner YYYY-MM-DD` or `Claude's recommendation YYYY-MM-DD, pending owner review`. Any `accepted
  finding` or `verdict reviewed` row the owner made is the owner's review of the verdict it names; a `note` is not
  (2026-10-31: `M11.C.21`'s note records that the owner did not review its pass).
- `reason` gives why, and where it is traced. It does not repeat the record's verdict numbers, which the status page
  shows from the records; a number in a reason is a diagnosis, marked ad hoc or with its source.

| criterion | disposition | verdict | decided | reason |
|---|---|---|---|---|
| `M11.C.49` | accepted finding | FAIL | owner 2026-10-31 | New at revision 39 (group F). The premise limb holds: the target feels a lower-level mover's I-position as more impingement. The composite limbs do not resolve: the family's summed excess and the F–M tie's `CONFLICT` share after the act do not differ by the mover's level at the settings fixed before the row first ran, which are not tuned to it (`M10.B.4`; T-137, 150 seeds on Mac Mini1, `docs/review/stage_3_step_8_criteria.md`). So in the model the demand share an I-position carries at a lower level reaches the target and does not spread to the family within six weeks. Its sweep ranges are not declared (T-144); W4 does not need them while it fails. |
| `M11.C.48` | accepted finding | FAIL | owner 2026-10-31 | Fails since revision 43 (T-123), when a first cutoff with no uncut tie in its week stopped counting with a baseline of 0. The pass recorded at revision 36 rested on that bias (`docs/readouts_phase_c.md`, "Negative runs"). No rule writes the run: `CUTOFF`'s value is learned from felt relief (`M4.D.6`). So in the model a tie is not cut off after a longer run of negative contacts than its family's uncut ties stand on in the same week. The settings are those fixed before the row first ran (`M10.B.4`), not tuned to it. |
| `M11.C.48[phase_d]` | accepted finding | FAIL | owner 2026-10-31 | The same criterion on the twelve-person family over 40 years, with the same change and the same reading as `M11.C.48`; a larger share of its first cutoffs is unmatched than on the Phase B family (284 of 749 against 139 of 700 at the re-sweep of 2026-10-23, T-123's record; the record's `unmatched_cutoffs` now). |
| `M11.C.34` | verdict reviewed | PASS | owner 2026-10-31 | Moved FAIL → PASS at revision 41 (T-131 (b), T-133 (b), T-145 (a)); no rule writes the others' reaction toward the bearer, and the composite reading is the owner's (2026-10-17). Its declared mutant, `M4.D.1a`'s mixing weight made blind to the responder's level, turns it red (T-162, revision 43), so the pass rests on the mechanism the row names. Recommended as the model's result under revision 41's rules, with its weakness stated: the sweep holds the direction in 1 of 8 samples, so it holds at the settings fixed before the row first ran and not across the declared ranges (the sweep is a measurement, not a verdict, `M11.5`). |
| `M11.C.21` | accepted finding | FAIL | owner 2026-10-31 | Fails at spec revision 44, once both parents' scripted `OVERFUNCTION` toward C1 ends at `t0` in both arms (T-161 (b), D4); the pass of revisions 41–43 rested on that lending going on after the collapse, and the owner's `note` of 2026-10-31 was decided on that pass. No rule writes the run: after `t0` the parents act on their own policy, and `M6.I.4a` returns lent functioning on a cause it does not choose. So in the model, when both projecting parents' functioning drops, the target's level does not rise once the parents act on their own. The settings are those fixed before the row first ran (`M10.B.4`), not tuned to it. Version 1's disposition (an accepted finding, owner 2026-10-17) was decided on a FAIL under the earlier arm. |
| `M11.C.17` | accepted finding | FAIL | owner 2026-10-31 | Fails at spec revision 44, once `M5.D.5`'s pull-up no longer applies when the mover is the coach (T-165, D5): that pull-up carried the pass of revision 43 (the two arms' coached rise is equal in 136 of 150 seeds, `docs/phase_d_ensemble_record.md`'s `n_nonzero`). No rule writes "coached does not rise more": what remains of the coach's effect, landed contact raising `systems_perspective` and opening the self-directed path the estimator reads (`M11.5`), gives no measurable rise in ten years. So in the model a coach's contact, without the pull-up, does not raise the coached people's basic level within ten years. Why the path is flat in most seeds is not traced (T-169). |
