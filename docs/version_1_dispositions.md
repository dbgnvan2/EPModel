# Version 1: dispositions of the criteria that do not pass

**Status: in progress, 2026-10-17.** Rows decided by the owner say so in `decided`. Every other row is Claude's
recommendation, made under the owner's instruction of 2026-10-17 ("proceed using your own recommendations"), and is
**pending owner review**: it does not count as a final status under V1 until the owner confirms it.

`tools/version_1_status.py` reads the table below and writes `docs/version_1_status.md`. This file holds the editorial
part (what the owner decided and why); the tool holds none. It is parsed strictly: a row naming a criterion no record or
spec row knows, an unknown disposition, or a `decided` cell in neither form below raises.

- `disposition` is one of:
  - `accepted finding`: a failing **composite** or **mixed** criterion accepted as a finding about the model (V1;
    `docs/version_1_definition.md` §4 question 2). The tool refuses it on a premise or check: a failing premise is a
    defect, not a finding, and must pass or be restated or withdrawn (V2, §4 question 3).
  - `out of scope`: not in version 1 (§4 question 4).
  - `withdrawn`: withdrawn by the owner.
  - `note`: carried beside the status without changing it.
- `decided` is either `owner YYYY-MM-DD` or `Claude's recommendation YYYY-MM-DD, pending owner review`.
- `reason` gives why, and where it is traced. It does not repeat the record's own verdict numbers, which the status
  page's Evidence column shows from the records; a number in a reason is a diagnosis, marked ad hoc or with its source.

| criterion | disposition | decided | reason |
|---|---|---|---|
| `M11.C.13` | out of scope | owner 2026-10-15 | It needs incidents located in a community, and the model has no community (version 1 definition §4 question 4). |
| `M11.C.28` | out of scope | owner 2026-10-15 | The sink allocation feeds no behaviour, so pinning its mobility can change no symptom (§4 question 4). |
| `M11.C.21` | accepted finding | owner 2026-10-17 | With both projecting parents over-functioning throughout in both arms, the child has lent down to its own floor in both, so the capacity rule has nothing to change: +0.15 ± 0.19 (plan §4.11). |
| `M11.C.23` | accepted finding | owner 2026-10-16 | Hostility delivers 41 ± 10 more appraisal load than talking it out, but higher anxiety decays faster (−33 ± 10) and the child gets relief by cutting off and distancing, so symptom load is no higher; medication's effect outweighs the hostility term (plan §4.11, traced ad hoc 2026-10-16). |
| `M11.C.22` | accepted finding | Claude's recommendation 2026-10-17, pending owner review | Mixed since revision 32: the severe limb (premise) passes; the mild limb is the net of `M7.D.2a`'s lock-in (+7.3) and other mechanisms (`REDUCE_CUTOFF` releases −2.4, distance binding −1.7, decay −1.5), so it is composite. The owner kept the capacity rule and accepted this failure "for now" (2026-10-17); over 600 seeds the rule shifts the mild limb by −0.15 ± 0.53, which does not show it causes the failure, and the mild limb passes there (+1.00 ± 0.40). The record's 150 seeds stop near the detection limit (ad hoc, 2026-10-17; plan §4.12). |
| `M11.C.29` | accepted finding | Claude's recommendation 2026-10-17, pending owner review | Mixed since revision 32: the budget limb (`M6.I.1`) is proved by `tests/bowen/test_sinks.py::test_m6i1_only_a_completed_exchange_reduces_the_budget` and reported; the third person's time course is composite. **Its precondition is not met:** no genuine `I-POSITION` completes in the arm (0 in 100 seeds; 37 run as the assertion form because the mover is angry at the act, 58 fail in PREPARE on the mover's own excess, 20.7 against `defence_threshold` 15, with the declared spell landing on the mover the same week; ad hoc, plan §4.12). So the failure says nothing about time course; what it does show about the model is that a lone genuine `I-POSITION` by a level-40 mover under this load does not complete. The owner may prefer a redesigned arm to an acceptance. |
| `M11.C.9` | out of scope | Claude's recommendation 2026-10-17, pending owner review | No rule reads sibling position (`objects.py` documents it as never read by the propensity vector; `functional_sibling_position` is never written; no fixture has ranked siblings joined by a tie). The decision of 2026-10-15 to include it rested on "one gate that is partly built"; building it is a new mechanism whose profiles the spec leaves open (`M1.A.14c` gives one example) and which could write a pattern (§19). Recommended for version 2. |
| `M11.C.8` | out of scope | Claude's recommendation 2026-10-17, pending owner review | A Phase E criterion (`→E`): endogenous incidence against published rates needs the editorial call on which sources are exogenous, accepted as not code-testable in Phases B–D on 2026-10-06 (`M11.E`). Version 1 excludes Phase E (§3). |
| `M11.C.5` | accepted finding | Claude's recommendation 2026-10-17, pending owner review | No settled change-back reaction over 60 weeks. The ladder is learned (`M5.E.0`); the model's later consequences of one act wash out within weeks to months (`docs/phase_c_completion_report.md` §13). |
| `M11.C.42` | accepted finding | Claude's recommendation 2026-10-17, pending owner review | A `TRIANGLE` relieves its seeker less than a `DISTANCE` (−1.29 against −4.97 one week after, at Phase C close), so it is not learned and not reused (`docs/phase_c_completion_report.md` §3 item 8, §13). |
| `M11.C.44` | accepted finding | Claude's recommendation 2026-10-17, pending owner review | No calm state near the floor: the calm arm is the triad's resting state, 8–9 points above the floor, and the spell adds about 2, so the outside position's value does not invert with load (Phase C close). It has moved between pass and fail on unrelated revisions with both mutants surviving, so a pass would not be evidence either (`TODO.md`). |
| `M11.C.45` | accepted finding | Claude's recommendation 2026-10-17, pending owner review | The causes of `M11.C.42` and `M11.C.44` together: triangling is not learned, and there is no calm state near the floor. |
| `M11.C.27[stable,add_third]` | accepted finding | Claude's recommendation 2026-10-17, pending owner review | Adding a third barely moves the pair's contact in any week: a `TRIANGLE` changes anxiety, not contact (`docs/phase_c_completion_report.md` §3 item 5). |
| `M11.C.27[unstable,add_third]` | accepted finding | Claude's recommendation 2026-10-17, pending owner review | As for the stable couple: adding a third barely moves the pair's contact. |
| `M11.C.27[stable,remove_one]` | accepted finding | Claude's recommendation 2026-10-17, pending owner review | Removing one steadies either couple for about four weeks and unsettles it from week 6; the stable couple steadies first, so the week the act lands reads the wrong sign and weeks 6–12 read the claimed one (reported readouts in the record). |
| `M11.C.41[higher level: heavier stress]` | accepted finding | Claude's recommendation 2026-10-17, pending owner review | Heavier stress raises mean acute anxiety but not the reactive choice beyond the legal set's reactive share. **For the owner:** `M11.5` classes this limb premise, through the mixing weight; the version 1 definition (§2) listed these cells for acceptance. Confirm one reading. |
| `M11.C.41[lower level: heavier stress]` | accepted finding | Claude's recommendation 2026-10-17, pending owner review | As the higher-level cell: acute anxiety rises, the reactive choice beyond chance does not. The same class question. |
| `M11.C.2` | accepted finding | Claude's recommendation 2026-10-17, pending owner review | On the four-person fixture, with projection-type acts counted in the first five years and symptom load read in the last five, the most-projected child carries no more; read over the same weeks it passed, which cannot separate cause from attraction (spec row, revision 26). Read by age on the twelve-person family it passes (`M11.C.2[twelve-person family]`, proved red by removing focus on weakness). |
| `M11.C.12` | accepted finding | Claude's recommendation 2026-10-17, pending owner review | Released anxiety raises the other spouse's excess, but more anxiety does not select `CONFLICT` in particular (`TODO.md`, B7). |
| `M11.C.17` | accepted finding | Claude's recommendation 2026-10-17, pending owner review | Fails since revision 31: the capacity rule turned its weak pass red (with the rule removed the same seeds give +0.00053 ± 0.00063, ad hoc 2026-10-16), and the owner kept the rule (2026-10-17). Since revision 32 the coach also holds no position for the weeks after a spouse's own `SPLIT` toward him (`M8.3`), which lowers the coached arm's landed contacts (8.2 to 5.5 over 12 seeds, ad hoc, the review of 2026-10-17). The coach's contact lands only when binders fail. |
| `M11.C.34` | accepted finding | Claude's recommendation 2026-10-17, pending owner review | The others' automatic acts toward the bearer after an onset, less a no-onset run, do not differ by level, and no selection reads `M1.A.19`'s detectors, so its declared mutant cannot bite (spec row, revision 27). **For the owner:** `M11.5` classes the automatic share premise; the version 1 definition (§2) listed it for acceptance. |
| `M11.C.43` | accepted finding | Claude's recommendation 2026-10-17, pending owner review | After learned reliance, losing a calming member costs no more (spec row, revision 27). |
| `M11.C.6` | accepted finding | Claude's recommendation 2026-10-17, pending owner review | The most-projected child does not develop less than the others under revision 31's band. The cause is not traced. |
| `M11.C.10` | accepted finding | Claude's recommendation 2026-10-17, pending owner review | The arms' input barely differs: the parents already hold some rejection on some tie most weeks, and the held cutoff adds 0.08 to 0.12 to the first grandchild's origin cutoff (3 seeds, 12 years, ad hoc 2026-10-15). The family-of-origin variant passes and is proved red by removing `M4.D.6f`. |
| `M11.C.7[joint over one spouse]` | accepted finding | Claude's recommendation 2026-10-17, pending owner review | The couple's reactivity on their own tie falls no more with a joint coach than with one who sees one spouse. Coaching barely moves the marital deviation in any arm, the weakness that makes `M11.C.17` fail: the coach's contact lands 12 or 13 times in ten years (seed 0, ad hoc), and only when binders fail. The gate's limb is not tested in this pair. |
| `M11.C.7[one spouse over sided]` | accepted finding | Claude's recommendation 2026-10-17, pending owner review | The gate's limb holds (a sided coach's contact never lands); the deviation limb does not, for the reason given for the first pair. |
| `M11.C.7[joint over sided]` | accepted finding | Claude's recommendation 2026-10-17, pending owner review | The gate's limb holds; the deviation limb points the stated way and is not significant, for the reason given for the first pair. |

## Premise limbs of mixed criteria

From `M11.5`'s class table. A failing mixed criterion whose premise limb named here does not hold is a failing premise
under V2, and the tool refuses to accept it as a finding. `M11.C.41` and `M11.C.34` are not listed: `M11.5` calls
their failing limbs premise, but the version 1 definition (§2) listed them for acceptance; which reading holds is the
owner's (see their rows above). `M11.C.11`'s premise limb is the conservation ledger, asserted every tick, not a
readout; `M11.C.29`'s is reported, not gating.

| criterion | premise readouts |
|---|---|
| `M11.C.7` | `landed_contacts` |
| `M11.C.22` | `severe_removal_effect` |
