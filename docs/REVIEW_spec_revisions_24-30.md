# Review form: spec revisions 24–30

**Status: REVIEWED BY THE OWNER 2026-10-16.** All 85 entries decided, in chat, revision by revision, on Claude's recommendations: 81 approved as built, 4 changed (revision 24 `M14.A` register rows; revision 25 `M11.C.14`, option B; revision 28 `M11.C.6`, option A, and `M7.E.1f`'s band, decided after the review). Revision 28 was decided by the owner's "continue to finish the review process", read as "as recommended"; the entries say so. The changes, the text fixes noted on approved entries and the inconsistencies below go into spec revision 31 (`docs/implementation_plan_phase_d.md` §4.11), which waits for the owner's approval of its plan. *Drafted as:* FOR OWNER REVIEW, 2026-10-15. Under the version 1 definition
(`docs/version_1_definition.md`, condition V4 and question 6), the owner's review of revisions 24–30 is a condition of
version 1.

**What this is.** Revisions 24–30 of `docs/bowen_agent_model_spec_v2.md` (2026-10-12 to 2026-10-14) were built before
the owner reviewed them. Each change is tagged inline in the spec (⟦rev24 · …⟧ to ⟦rev30 · …⟧), and the reasoning
is in `docs/implementation_plan_phase_d.md`. This form gathers them, one entry per requirement added or changed.

**How to use it.** Tick one box per entry. *Approve* keeps the rule as built. *Change* says what to change: Claude
drafts the revision and the plan for it, and nothing is rebuilt before the plan is approved. *Reject* removes the rule
in a later revision. Where the owner already decided the substance and only the form is Claude's, the entry says so,
and the review is of the form. Revision 24–27 entries are Claude's recommendations throughout; revisions 28–30 apply
the owner's decisions in Claude's forms.

**Each entry gives:** what the rule does; whose decision it is; its constants (`[I]` = invented, no source value);
and what the plan or a record says it changed in the runs, or "not measured". Measurements are ad hoc runs unless a
record is named. Verdicts quoted from the plan are as of that revision; the current verdicts are in the ensemble
records (`docs/phase_c_ensemble_record.md`, `docs/phase_d_ensemble_record.md`, `docs/phase_d2_ensemble_record.md`).

Drafted 2026-10-15 by three agents working from the spec, the plan and the constants files, assembled and
number-checked by Claude (every interval quoted was found in the plan, the spec or a record). Line numbers are the
spec's at commit `c74ca86`.

## Inconsistencies found while drafting

These are disagreements inside the documents, not decisions. They are listed so the review does not stall on them;
each is in `TODO.md`.

1. `channel_focus`: the field register (spec line 1630) says it is "written by B8 from belief"; `M1.A.11c` (line 249)
   says it is engine-side attention, not a belief any policy reads.
2. `tolerance`: a Family field in the register (line 1679), but declared per person in `M7.D.4` (line 1076).
3. `TRIANGLE`'s symptom channel: the B8 plan maps it to the mind channel; `config/bowen/attribution.md`, which is what
   was built, gives it none.
4. `M7.E.1c`: the spec (line 1084) calls the other four situations at `t0` "an open owner decision"; plan §4.8 (line
   1355) records them as accepted by the owner and not applied.
5. `M11.C.2` (spec line 1314) says its ages and `renew_every` "are to be added" to the criteria config; they are there.
6. The criteria settings (the `M11.C.2` ages, `renew_every`, `seeds`, the `M11.C.47` fixture values) carry no grade.
7. Some revision 28 tags name no source (the `M2.A` table rows and note, `M3.D.1`, `M6.4`, `M11.C.6`, `M11.G.4`, the
   register rows); those entries say "no source named".


## Revisions 24 to 27 (all 2026-10-12)

All four revisions are Claude's recommendations of 2026-10-12, pending owner review. The spec's status line says the owner asked for the work to proceed overnight without waiting for input (`docs/bowen_agent_model_spec_v2.md` line 4; the plan's B7 and B8 design sections say the same). None of the tagged passages in revisions 24 to 27 quotes the owner. All results below are ad hoc runs, not records; the plan says so (`docs/implementation_plan_phase_d.md`, "Step 7 result", "Step 8 result", "Every Phase D criterion on the final code of revision 27").

---

## Revision 24 — Symptom channels (Phase D B7) and `M11.C.22`'s control arm

Revision 24 builds the three symptom channels (body, mind, conduct): which channel a person's symptoms show in, how load in one channel protects against the others, how a symptom calms the family up to a point and then unsettles it ("lock-in" and its "turn"), and when the family removes a symptomatic member. It also changes what counts toward symptom load: time above the person's usual anxiety, not above their chronic floor, because before this change every person in ten-year runs was symptomatic from the first weeks. Source: Claude's recommendation 2026-10-12, pending owner review (plan "B7 design", `docs/implementation_plan_phase_d.md` line 732).

**`M1.A.6`** (spec line 224) — The symptom threshold is read from functional level with a floor under it, so a person at functional level 0 has a small threshold rather than none. Amendment.
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: `functional_level_floor` = 1.0 [I] (an existing constant, now also used here; `config/bowen/constants.md` line 97; no new row in `constants_changes.md`).
- Effect measured: not measured.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M1.A.11c`, revision 24 part** (spec line 249) — The channel a symptom shows in is chosen by weighing the person's constitutional channel against what the family focuses on in them; a constitution strength of 1 means family focus never moves it. Ties go to the constitutional channel. Amendment (built form of an existing requirement). Until revision 25, a scenario set the family focus.
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: `constitution_strength` = 0.5 [I] (a person's own value overrides it).
- Effect measured: not measured.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M1.F.9`** (spec line 524) — When a symptom channel is released, the symptom ends, and the anxiety it was holding (`symptom_bound`) goes back to the people on live ties to the bearer. The symptom's accumulated load is not added to the family budget, because it is a count of weeks above usual, not a store of anxiety. Amendment.
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: none.
- Effect measured: the spec states the reason, from a review on 2026-10-12: crediting the load put 77 points into the budget on one release while the hold was 27 (spec line 524).
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M4.C.3a`** (spec line 714) — Symptom load now builds from anxiety above the person's usual level, not above their chronic floor. It still counts time above a floor, not peaks. A rise that lasts about a year becomes usual, so a symptom with nothing holding it goes away. Amendment.
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: none new (the usual level is revision 22's).
- Effect measured: before the change, in ten-year runs of the nuclear-four fixture, every person crossed the threshold in the first weeks and stayed at 4 to 31 times it (spec line 714; plan "B7 design", "What was found first"). The plan's `M11.C.17` trace attributes about three quarters of the fall in the coached rise to revision 24 (plan, "Why `M11.C.17`'s coached rise collapsed": landed coach contacts 19.3 at revision 23 against 8.0 at revision 24).
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M6.4`, stock table** (spec lines 982–983) — Adds a new anxiety stock, `Person.symptom_bound`, with where it fills from and where it returns, and notes that a symptom channel's hold returns to the partners rather than to the family budget. Amendment (new row and an edited row).
- Source: the tags carry no source text (lines 982–983 are tagged only "rev24"); the revision as a whole is Claude's recommendation 2026-10-12, pending owner review.
- Constants: none.
- Effect measured: not measured.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M7.D.1`** (spec line 1058) — Every channel is checked for a new symptom each week, not only the channel being fed, because load left in a channel can cross the threshold when functional level falls. Each crossing is its own onset. Amendment.
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: none.
- Effect measured: not measured.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M7.D.2`** (spec line 1060) — Load goes to whichever channel is currently expressed (`M1.A.11c`), and every channel leaks. Amendment (built form).
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: none stated on this line.
- Effect measured: not measured.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M7.D.2a`** (spec line 1062) — Lock-in: while a symptom is mild, the people on live ties to the bearer move part of their anxiety into the bearer's hold, so the family calms around the symptom. When the symptom ends, or a `binder_unavailable` names its channel, the whole hold goes back to them in equal shares. While the bearer is removed, the hold neither fills nor empties. Not built: the hold does not feed the bearer's own symptom load, so lock-in does not by itself make a symptom chronic. Amendment (built form).
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: `lock_in_gain` = 0.2 [I]; `lock_in_turn` = 3.0 [I]. The plan's draft had 0.05 and 2; the built values were set when the code was written, before any run (plan, "As built, four changes from the draft above").
- Effect measured: see `M11.C.22` below. The plan also records that feeding the hold into the bearer's load made every bearer's symptom climb to the turn and stay there, which is why it was left out (plan, same section).
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M7.D.2c`** (spec line 1068) — Load already carried in one channel reduces what enters the other two, so a person with one symptom is protected against a second. Amendment (built form).
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: `channel_protection` = 1.0 [I].
- Effect measured: `M11.C.30` passes, social-channel load −20.8 ± 1.7 (50 seeds), and fails with the cross-channel term zeroed (plan, "Step 7 result" table); −21.4 ± 1.6 on the final revision 27 code (plan, "Every Phase D criterion on the final code of revision 27"). `M11.C.30`'s spec row carries no revision 24 tag.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M7.D.2d`** (spec line 1070) — Past the turn (a severe symptom), part of the hold goes back to the partners each week, and the same share of the bearer's own excess anxiety spills to them in equal shares. Both are logged transfers. Amendment (built form).
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: `lock_in_turn` = 3.0 [I] (shared with `M7.D.2a`).
- Effect measured: the plan records that returning only what had been held was too little for a severe symptom to unsettle anything (plan, "As built, four changes from the draft above"). Criterion result under `M11.C.22` below.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M7.D.4`** (spec line 1076) — Each week, if someone has an active symptom and another member on a live tie to the most loaded bearer is more anxious than their own tolerance, that member removes the bearer (`INSTITUTIONALIZE`). It reads the others' anxiety, not the bearer's severity. A family whose decisions are held by an outside agent never does this. A person with no declared tolerance never fires it. Amendment (built form).
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: none (tolerance is declared per person; none is declared).
- Effect measured: not measured. No Phase D I criterion depends on it; it is pinned by `tests/bowen/test_symptoms.py::test_m7d4_removal_fires_off_the_others_tolerance_not_the_bearers_severity` (spec line 1076). The plan's draft used the partners' mean anxiety; the build uses any one partner's own tolerance (plan, "As built").
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M11.C.11`** (spec line 1323) — Records how the removal-and-return criterion is covered: the conservation part is the ledger check run every tick; the "return is larger than removal" part has no declared mutation, and about half of it is the removed member's symptom hold, frozen while away and released on return. Amendment (coverage note).
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: none.
- Effect measured: +2.35 with the hold, +1.21 without (spec line 1323; plan "Step 7 result", 150 seeds: +2.35 ± 0.86 and +1.21 ± 0.73).
- Verdict at revision 27 (plan table; the current verdict is in the ensemble record): PASS, +2.63 ± 0.92.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M11.C.12`** (spec line 1324) — Records the measured result: curing a spouse's symptom raises the other spouse's anxiety, but marital `CONFLICT` does not rise. Amendment (result note).
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: none.
- Effect measured: on the dyad, `CONFLICT` 55 against 56, while `PURSUE` (63 → 75) and `UNDERFUNCTION` (52 → 65) rose; on the nuclear four, −0.16 ± 0.35 (plan, "Step 7 result", "What it shows"). Nothing was changed to make it pass (same section).
- Verdict at revision 27 (plan table; the current verdict is in the ensemble record): FAIL, −0.04 ± 0.24.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M11.C.22`** (spec line 1334) — Restates the lock-in test so each limb is read against removing the same member with no symptom, because removing anyone relieves this family a little. The mild limb must raise the others' anxiety on removal more than that control; the severe limb must lower it more. Amendment (criterion restated).
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: none (severities held at 1.5 and 4.5 in the run, plan "Step 7 result").
- Effect measured: control −0.83 ± 0.76; without the restatement the mild limb fails (−0.09) (spec line 1334). Step 7: mild +0.33 ± 0.36 (p 0.027), severe −0.80 ± 0.44, each limb failing under its own mutant (plan "Step 7 result"). The plan calls the mild limb weak: its interval reaches below zero.
- Verdict at revision 27 (plan table; the current verdict is in the ensemble record): PASS, mild +0.81 ± 0.38, severe −1.03 ± 0.44.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M14.A` register rows** (spec lines 1630, 1631, 1632, 1679, 1680) — Adds the new state fields to the register: `channel_focus`, `constitution_strength`, `symptom_bound` (Person); `tolerance` (marked "declared, not updated") and `decision_ownership_external` (Family). New rows.
- Source: the tags carry no source text; the revision as a whole is Claude's recommendation 2026-10-12, pending owner review.
- Constants: none.
- Effect measured: not applicable.
- Note: line 1630 says `channel_focus` is "written by B8 from belief", while `M1.A.11c` (line 249) says B8 writes it "as engine-side attention ... not as a belief any policy reads". Unclear from the spec which wording is meant; they disagree.
- Note: `tolerance` is listed on the Family object (line 1679), while `M7.D.4` (line 1076) says it is declared per person.
- [ ] Approve  [x] Change: line 1630 to say `channel_focus` is written from the moves delivered (engine-side attention, as `M1.A.11c` and `src/bowen/engine/attribution.py` do), and the stale docstring at `src/bowen/engine/symptoms.py:26` with it; line 1679 to say `tolerance` is on Family, one declared value per member (`src/bowen/engine/objects.py:325`)  [ ] Reject — owner 2026-10-16, as recommended

---

## Revision 25 — Beliefs as a channel (Phase D B8)

Revision 25 gives each person a belief about where the family's difficulty lies, written from the moves they send, receive or witness; lets a witness's reaction to a quarrel scale with how tense they believe that tie is; writes the family's focus on a person's symptom channel from what anxious family members actually do to them; and adds a marital-distance gate that stops moves toward a symptomatic child from helping while the parents are distant. It also records the built `M11.G` evaluation readout (plan "B10 as built"). Source: Claude's recommendation 2026-10-12, pending owner review (plan "B8 design", `docs/implementation_plan_phase_d.md` line 847).

**`M1.A.11c`, revision 25 part** (spec line 249) — The family's focus on each of a person's channels fades a little each week, and grows when an anxious family member sends them a move of a kind tied to that channel: caretaking to the body, worry and talk-about to the mind, blame to conduct (mapping in `config/bowen/attribution.md`). Amendment.
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: `focus_rate` = 0.05 [I].
- Effect measured: not measured.
- Note: the plan's draft maps `TRIANGLE` to the mind; `config/bowen/attribution.md` gives `TRIANGLE` no focus (review 2026-10-12). The config is what is built.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M5.C.1`, marital-distance row** (spec line 865) — While both spouses' felt contact on their tie is below a quarter of their optimum, a move toward one of their children with an active symptom cannot lower that child's anxiety, and calming contact does not reach the child. A symptomatic spouse is not gated. Amendment (built form).
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: `marital_distance_fraction` = 0.25 [I].
- Effect measured: in ordinary 208-week runs the gate is on 26–32 weeks in the nuclear four and 18–70 in the Phase C family (3 seeds each; plan "B8 design", last bullet of "What it shows"). As first built it was on almost every week, which is why the fraction was added.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M9.6`** (spec line 1148) — Each person holds a weight per family member for where they believe the difficulty lies, empty at the start. Each move they send, receive or witness points at someone by its kind (from config), and the weights move toward that. A readout outside the engine reads the spouses' largest weights to name the active sink. No rule writes the three patterns. Amendment (built form).
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: `attribution_rate` = 0.05 [I].
- Effect measured: see `M11.C.26` below.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M9.7`, built for the witness** (spec line 1151) — A witness's reaction to load passed between two other people is scaled up by how tense the witness believes that tie is (a relief is not scaled). An event addressed to the person reads their own tie. Amendment (built form).
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: `belief_appraisal_gain` = 1.0 [I].
- Effect measured: see `M11.C.36` below.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M11.C.14`** (spec line 1326) — Records the measured result of the "management technique has no effect under marital distance" test. Amendment (result note).
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: none.
- Effect measured: talking it out leaves the child 8.0 ± 1.9 less symptom load than firmness, outside the bound of 5.6; with the gate removed −6.7 ± 1.9 (spec line 1326; plan "Step 8 result"). The plan says the null needs either a gate that also blocks harm or a reading of "zero independent effect" that allows it, and that this is the owner's choice.
- Current verdict (plan): FAIL.
- [x] Approve (the result note)  [x] Change: restate the criterion (option B): under high marital distance, no technique lowers the child's symptom load against a no-technique arm; the difference between techniques is reported, not tested. Needs a third arm and a rerun of this criterion  [ ] Reject — owner 2026-10-16, as recommended

**`M11.C.26`** (spec line 1338) — Records the measured result: the spouses' beliefs recover the driven sink. The reverse direction is reported, not tested, because here it is the same comparison. Amendment (result note).
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: none.
- Effect measured: +0.76 ± 0.04 (150 seeds); randomising the attribution write gives +0.01 ± 0.02 (plan "Step 8 result"; spec line 1338). The plan classes it as a premise: it confirms the code, not a finding.
- Current verdict (plan): PASS.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M11.C.36`** (spec line 1348) — Records the measured result: a witness who believes a tie is tense reacts more to the same events. Selection is not read. Amendment (result note).
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: none.
- Effect measured: +3.07 ± 0.01 (50 seeds); reading the tie's true state gives 0.0 (spec line 1348; plan "Step 8 result"). The plan classes it as a check.
- Current verdict (plan): PASS.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M11.G.2`** (spec line 1522) — The evaluation readout reports adaptiveness as the members' automatic acts per unit of outside stress, next to each figure alone; with no stress it is undefined, never 0. Built in `src/bowen/readouts/evaluation.py`. Amendment (built form; the plan files it under "B10 as built", not B8).
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: none.
- Effect measured: not measured (tests named in the plan: `tests/bowen/test_evaluation.py`).
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M11.G.4`** (spec line 1526) — The evaluation carries the framing block, and reports the sink allocation (truth) and the spouses' attributions (belief) separately. The projection line was reported as not computable at the time. Amendment (built form). A later revision 28 note on the same line is outside this form.
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: none.
- Effect measured: not measured.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M14.A` register row `attribution`** (spec line 1633) — Adds the `attribution` field (Person) to the register. New row.
- Source: tag carries no source text; the revision is Claude's recommendation 2026-10-12, pending owner review.
- Constants: none.
- Effect measured: not applicable.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

---

## Revision 26 — The cross-person learning term weighted by the tie (Phase D B9)

Revision 26 changes how a person learns from another person's calming after their act: the others' relief is now weighted by how strong the actor's tie to each of them is, so a child learns more from a parent's calming than from a distant witness's, without any rule naming a parent. It also records the `M11.C.2` result. Source: Claude's recommendation 2026-10-12, pending owner review (plan "B9 design", `docs/implementation_plan_phase_d.md` line 923).

**`M4.D.6e`** (spec line 797) — The relief of others that reinforces an act is weighted by the actor's bond on its tie to each of them; someone with no tie to the actor counts for nothing. Before, it was a plain mean. Amendment.
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: none.
- Effect measured: removing this term does not turn `M11.C.2`'s same-window reading red (+2.31 ± 1.35) (spec line 1314). The plan says the term is tested on its own (`test_learner.py`), but no criterion yet shows it shaping a child's repertoire (plan "Step 9 result").
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M11.C.2`** (spec line 1314, two revision 26 notes) — Records two measurements. Read over the same weeks, the most-projected child carries more symptom load, and removing the witness path turns it red. Read with acts counted in the first five years and load in the last five, so cause can be separated from attraction, it fails. Amendment (result notes). A later revision 29 amendment on the same line is outside this form.
- Source: Claude's recommendation 2026-10-12, pending owner review (both notes).
- Constants: none.
- Effect measured: same window +1.73 ± 1.22 (p 0.003), later +1.31 (p 0.045); early acts against later load +0.29 ± 2.20 (p 0.39) (spec line 1314; plan "Step 9 result").
- Current verdict (plan "Step 9 result" and final table): FAIL.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

---

## Revision 27 — Borrowed pseudo-self settles back when the exchange stops

Revision 27 adds one mechanism: when a pair who have been over- and under-functioning for each other stop doing so for eight weeks, the pseudo-self one borrowed from the other moves back a little each week, with functional level, and the total is conserved. Without it `M11.C.24` could not pass. It also records the measured results of the step 10 criteria that run on existing mechanisms. Source: Claude's recommendation 2026-10-12, pending owner review (plan "Step 10 (part) ... revision 27", `docs/implementation_plan_phase_d.md` line 946).

**`M6.I.4`** (spec line 965) — On a tie the pair has exchanged on, where one carries borrowed pseudo-self and the other lent it, and no over- or under-functioning has been sent between them for 8 weeks, 5% of the smaller amount moves back each week from borrower to lender, with the same change in functional level. Total pseudo-self is conserved. Amendment. A later revision 28 note on the same row is outside this form.
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: `exchange_window` = 8 [I]; `pseudo_self_relaxation_rate` = 0.05 [I].
- Effect measured: removing it turns `M11.C.24` red (B −0.13 ± 0.26) (plan, step 10 table). Without it, `M11.C.17`'s coached rise is +0.005 ± 0.002 against +0.0009 ± 0.0008 with it (spec line 1329). The plan says it changes every run in which pseudo-self is lent and the lending stops. As first built it paired any opposite-signed tie; the balance condition was added at review (plan, same section).
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M11.C.17`** (spec line 1329) — Records that the coached rise in basic level nearly vanished by revision 27, with two probes. Amendment (result note).
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: none.
- Effect measured: +0.0009 ± 0.0008 (p 0.011), against +0.021 ± 0.009 at revision 23; without revision 27's settling back +0.005 ± 0.002; with the symptom integrand above the chronic floor +0.0018 ± 0.0010 (spec line 1329). The plan's trace attributes the fall to the coach's contact landing only when a symptom is active or near (about three quarters from revision 24, the rest from 25–27, chiefly 27), and leaves to the owner whether "pain is necessary" should read an active symptom or load above usual (plan, "Why `M11.C.17`'s coached rise collapsed").
- Current verdict (plan final table): PASS, weak.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended; "pain is necessary" (`M1.E.7`) keeps reading an active symptom or load near threshold, not load above usual

**`M11.C.24`** (spec line 1331) — Records the result: after the discrepant pair's tie is dissolved, the under-functioner rises and the over-functioner falls. Amendment (result note).
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: none.
- Effect measured: +11.9 ± 0.4 and −12.6 ± 0.5 (150 seeds); zeroing the transfer or removing the settling turns it red (spec line 1331; plan step 10 table).
- Current verdict (plan): PASS.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M11.C.21`** (spec line 1333) — Records the result: when both parents' functional level drops by 10, the projection target's does not rise. Amendment (result note).
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: none.
- Effect measured: −0.52 ± 1.32 (100 seeds) (spec line 1333; plan step 10 table).
- Current verdict (plan): FAIL.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M11.C.23`** (spec line 1335) — Records the result: high expressed emotion with suppression ("medication") relapses less, not more, than low expressed emotion without it; halving accumulation outweighs the hostility. Amendment (result note).
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: none new (the suppression factor 0.5 [I] is revision 15's).
- Effect measured: −0.29 ± 0.08 (150 seeds) (spec line 1335; plan step 10 table).
- Current verdict (plan): FAIL, reversed.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M11.C.34`** (spec line 1345) — Records the result: the others' automatic acts toward a new symptom bearer do not differ between a low-level and a high-level family. No choice of act reads `M1.A.19`'s detectors, so the declared mutant cannot bite. Amendment (result note).
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: none.
- Effect measured: −0.04 ± 0.10 (150 seeds) (spec line 1345; plan step 10 table). The plan notes it first passed (−0.18) on a readout that counted all acts with no onset fired; the review found this.
- Current verdict (plan): FAIL.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M11.C.39`** (spec line 1351) — Records the result: after an anxiety spell, distance and cutoff acts persist longer when the automatic channel learns. Amendment (result note).
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: none.
- Effect measured: +6.0 ± 1.1 `DISTANCE`/`CUTOFF` acts in the 52 calm weeks; time to remission +2.7 ± 2.6 weeks (reported); learning off in both arms turns it red (spec line 1351; plan step 10 table).
- Current verdict (plan): PASS.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M11.C.40`** (spec line 1352) — Records the result: learning narrows the automatic channel (clause 1 holds), but the second clause, that the self-directed channel differs less between arms than the automatic one, does not hold. Amendment (result note).
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: none.
- Effect measured: clause 1 +0.027 ± 0.006; clause 2 −0.0005 ± 0.041 (p 0.49 in the plan) (150 seeds) (spec line 1352; plan step 10 table).
- Current verdict (plan): FAIL.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M11.C.43`** (spec line 1355) — Records the result: losing a member who had been a calming presence during a spell does not raise the person's anxiety more than in the arm where that member was never available. Amendment (result note).
- Source: Claude's recommendation 2026-10-12, pending owner review.
- Constants: none.
- Effect measured: −0.54 ± 0.71 (150 seeds) (spec line 1355; plan step 10 table).
- Current verdict (plan): FAIL.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

## Revision 28 — the life course

Revision 28 makes the twelve-person family live through time: people age, children start acting at 10, children's chronic anxiety and basic level develop from their parents', partners and births come from settings, people die and their anxiety, grief and lost income reach the living, and the coach retires and is replaced. It applies the owner's decisions of 2026-10-13 (P13–P15); the forms the spec gives them are Claude's recommendation, pending owner review (spec status line, line 4). The owner approved the part II plan that builds it on 2026-10-13 (`docs/implementation_plan_phase_d.md` §4, line 1085), and it was built that day.

Where an entry says "owner decided the substance", the decision itself is already made; what is under review is the form Claude gave it.

---

#### The reference family

**`M2.A` table and note** (spec lines 552–557) — Leo, Nadia and Pia's rows now say their level and chronic anxiety are derived from their parents, and Leo launches at 20 instead of being already launched. Dr Halim is female, 35, retires at 68, and her level of 60 now follows the config. The note at line 557 says the Phase B and C instances keep their old values. Amendment.
- Source: owner decision 2026-10-13 · P15 (Dr Halim's age and sex, line 555); the other rows carry a plain ⟦rev28⟧ tag with no source named.
- Constants: none in the table (see `M2.A.0j`).
- Effect measured: not measured.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended)

**`M2.A.0a`** (spec line 563) — Only people already past age 12 at the start get a supplied chronic anxiety. With the new starting ages, the three children (9, 4 and 1) are under 12, so theirs is derived instead (`M7.B.1a`). Amendment.
- Source: Claude's recommendation 2026-10-13, pending owner review.
- Constants: `chronic_anxiety_fixation_age_years` = 12 [I] (existing, `config/bowen/constants.md` line 87).
- Effect measured: not measured.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended)

**`M2.A.0j`** (spec lines 583–587) — Starting ages are three settings (youngest grandchild 1, parents' generation 30, grandparents 55); each person has an age gap to their generation's anchor, and the loader computes each age. The built note lists the resulting ages and three consequences: the children's chronic anxiety is derived, their levels start at the parents' mean, and Leo is back in the household. New. **Owner decided the substance** (the three ages, "settings in app"); Claude's are the anchors for generations 1 and 2 (the eldest, "Claude's reading", line 583), deferring the app display, and the second consequence (line 586: the children's former levels 36, 41, 44 replaced by the parents' mean 39.5).
- Source: owner decision 2026-10-13 · P15 (line 583); Claude's recommendation 2026-10-13, pending owner review (line 586, the children's levels).
- Constants: `start_age_g1` = 55, `start_age_g2` = 30, `start_age_youngest_g3` = 1 (family settings, `config/bowen/family_phase_d.md` lines 42–44; the file is graded [I] at line 39); `coach_start_age_years` = 35 [I] (the owner's value; `constants.md` line 176).
- Effect measured: not measured.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended)

**`M2.A.0k`** (spec line 589) — Dr Halim retires at 68. Her tie to the coachee goes quiet but stays, and a new coach enters at 35 with her level, chronic anxiety and channel prior, a fresh reactive state and a new tie. What the coachee learned is kept. New. **Owner decided the substance** ("retires at 68 and new coach starts over"); Claude's are the details of what the successor inherits and what is kept.
- Source: owner decision 2026-10-13 · P15; Claude's recommendation 2026-10-13, pending owner review.
- Constants: `coach_retirement_age_years` = 68 [I] (the owner's value), `coach_start_age_years` = 35 [I] (`constants.md` lines 176–177); `successor_coach_sex: female` (`family_phase_d.md` line 47; plan §4.5 default "female, as Dr Halim").
- Effect measured: not measured.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended)

**`M2.B.2`** (spec line 599) — The coach has one tie, to the person being coached, instead of ties to both spouses. `M11.C.7`'s joint arm still ties the coach to each spouse, because it is a test arm. Amendment.
- Source: owner decision 2026-10-13 · P15 ("Coach is tied to Coachee").
- Constants: none (`coachee: ravi`, `family_phase_d.md` line 46).
- Effect measured: not measured.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended)

#### Clocks and determinism

**`M3.D.1`** (spec line 625) — The weekly update order now includes the life course: life events queued at the yearly tick arrive in step 2, only people of acting age choose moves in step 7, the search for work runs at the end of step 8, and the life course itself runs on the yearly tick. Amendment.
- Source: plain ⟦rev28⟧ tag, no source named; unclear from the spec whether this is the owner's or Claude's (line 625).
- Constants: none.
- Effect measured: not measured.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended); the tag becomes "Claude's recommendation 2026-10-13, approved by the owner 2026-10-16" (inconsistency 7)

**`M3.D.4b`** draw table, "Life course" row (spec line 651) — The random draws of the life course (a partner's level, a newborn's sex and channel, each week's chance of finding work) get their own keys, per person, so two runs with the same seed draw the same values. New row.
- Source: Claude's recommendation 2026-10-13, pending owner review.
- Constants: none.
- Effect measured: not measured.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended)

#### Conservation

**`M6.I.4`** (spec line 965) — When one partner of a tie dies, the living person's borrowed or lent pseudo-self settles back as `M6.3e` describes, by the same conserved transfer. Amendment.
- Source: Claude's recommendation 2026-10-13, pending owner review.
- Constants: `pseudo_self_relaxation_rate` = 0.05 [I] (revision 27's, `constants_changes.md` line 146).
- Effect measured: not measured.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended)

**`M6.4`** stock-and-flow table (spec lines 979–982) — Adds the new sources and transfers: grief and lost income as sources of acute anxiety; a person's entry at birth or partnering as a logged source; the dead person's anxiety to the living; a dead member's triangle and tie holdings to the living; a new family's budget at pairing. Amendment.
- Source: plain ⟦rev28⟧ tags, no source named.
- Constants: none.
- Effect measured: not measured (`test_m64_entry_and_new_budget_are_logged_sources` is listed for L7, plan line 1118, and §4.7 line 1172 says every listed test exists; that is a test, not a measured effect).
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended); the tag becomes "Claude's recommendation 2026-10-13, approved by the owner 2026-10-16" (inconsistency 7)

#### Disposition at death (`M6.3`)

**`M6.3`, revision 28 note** (spec line 989, with line 1009) — Records the owner's account of death (anxiety redistributed like bond energy, death as a stressor, lost income, grief over 1–3 years, "Trying equals finding, but at 20% less income") and states that `M6.3a`–`M6.3i` apply it in that order in the week of the death. Line 1009 says `M11.C.37` checks conservation across `M6.3b`–`M6.3c` only; grief and lost income are outside it. New. **Owner decided the substance**; the forms in `M6.3a`–`M6.3i` are Claude's reading (line 989).
- Source: owner decision 2026-10-13 · P13; Claude's recommendation 2026-10-13, pending owner review.
- Constants: see each clause.
- Effect measured: `M11.C.37` PASS (a test, with its mutant), conserved to `M6.1`'s tolerance (plan §4.7, line 1199).
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended)

The clauses `M6.3a`–`M6.3i` carry no tag of their own except `M6.3b`; the line-989 tag covers them. Each is listed so it can be decided separately.

**`M6.3a`** (spec line 991) — A death is an outside event. The dead person stops acting and perceiving; their ties stay but go quiet; bond energy on those ties stays where it is. What moves is anxiety. New.
- Source: as line 989 (owner P13 substance; Claude's form).
- Constants: none.
- Effect measured: not measured.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended)

**`M6.3b`** (spec line 993) — The dead person's whole acute anxiety is split among living family members tied to them, in proportion to bond energy; the coach takes no share. Fallbacks: the household, then all living family members. No criterion overrides the full transfer. New.
- Source: owner decision 2026-10-13 (the in-line tag on `M11.C.15` keeping the transfer); otherwise as line 989.
- Constants: `death_transfer_share` = 1.0 [I] (`constants.md` line 180).
- Effect measured: `M11.C.15` PASS, 100 seeds: death +1.25 ± 0.38 (p 2.1e-08), recovery +3.11 ± 0.35; its mutation turns the death arm red (−2.03 ± 0.29). With the share at 0 (first design) death read +0.212 ± 0.35, FAIL (plan §4.7, lines 1200, 1208–1220).
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended)

**`M6.3c`** (spec line 995) — What the dead person's ties, triangles and symptom held goes to the living members (triangles discharge equally and dissolve; symptom hold returns to the partners). New. Its last sentence, on external agents, is an owner decision of 2026-10-14, outside revision 28.
- Source: as line 989; last sentence owner decision 2026-10-14.
- Constants: none.
- Effect measured: covered by `M11.C.37` PASS (plan line 1199).
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended)

**`M6.3d`** (spec line 997) — Each living person's investment in the dead is spread over their other active ties, in proportion to what each already holds. New.
- Source: as line 989.
- Constants: none.
- Effect measured: not measured.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended)

**`M6.3e`** (spec line 999) — Borrowed or lent pseudo-self on a tie to the dead settles back to zero at revision 27's rate: an overfunctioner loses what they borrowed, an underfunctioner recovers what they lent. New.
- Source: as line 989.
- Constants: `pseudo_self_relaxation_rate` = 0.05 [I].
- Effect measured: not measured.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended)

**`M6.3f`** (spec line 1001) — The family's budget held against the dead (as dysfunctional spouse or projected child) returns to the family. When both spouses are dead the family stops taking allocations and its budget is kept and logged. New.
- Source: as line 989.
- Constants: none.
- Effect measured: not measured.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended)

**`M6.3g`** (spec line 1003) — Each living family member actively tied to the dead gets a grief stressor, larger and longer on stronger ties: one year on the weakest tie, three on the strongest. New. **Owner decided** "1–3 years"; the sizes are Claude's.
- Source: as line 989.
- Constants: `grief_scale` = 20.0 [I], `grief_min_weeks` = 52 [I], `grief_extra_weeks` = 104 [I] (`constants.md` lines 168–170).
- Effect measured: not measured.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended)

**`M6.3h`** (spec line 1005) — If the dead person earned income, everyone in their household gets a lasting stressor. When one of them finds work it drops to 20% and stays; that person becomes an earner. This amends `M1.F.6` for this stressor. New. **Owner decided** "at 20% less income"; the size is Claude's.
- Source: as line 989.
- Constants: `income_loss_intensity` = 15.0 [I], `income_shortfall_share` = 0.2 [I] (the owner's 20%) (`constants.md` lines 171–172).
- Effect measured: not measured.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended)

**`M6.3i`** (spec line 1007) — The owner's "everyone's stress going up stresses the system reciprocally" is not written as a rule; it travels through the ties like any other anxiety. New.
- Source: as line 989.
- Constants: none.
- Effect measured: not measured.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended)

#### Childhood and the life course (`M7`)

**`M7.B.1a`** (spec line 1029, revision 28 part) — A child under 12 has their chronic anxiety recomputed each year: the parents' mean plus `witnessed_gain` times the child's own mean load that year. It is fixed at 12. A child present at the start starts from the parents' mean. New. (Revision 29 later amended the load term to average over all of childhood; that part is not under review here.)
- Source: Claude's recommendation 2026-10-13, pending owner review.
- Constants: `witnessed_gain` = 0.5 [I] (`constants.md` line 163).
- Effect measured: ad hoc diagnosis, 20 seeds, not a record: chronic anxiety reflected only the year lived at 11; its correlation with projection received was −0.26 (Leo), −0.04 (Nadia), −0.01 (Pia) (plan §4.7, lines 1253–1255).
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended)

**`M7.C.1f`** (spec line 1046) — People choose moves only from age 10 until death. Younger children still feel, react, carry anxiety and symptom load, and witness. Everyone who acts carries an age, advanced each year. New. **Owner decided the substance** ("starts acting at 10, stops acting at death").
- Source: owner decision 2026-10-13 · P14.
- Constants: `acting_age_years` = 10 [I] (the owner's value, `constants.md` line 158).
- Effect measured: not measured.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended)

**`M7.C.1g`** (spec line 1048) — At 20 a child leaves home for their own household and becomes an earner; their ties stay. The owner has not stated a launch age. New.
- Source: Claude's recommendation 2026-10-13, pending owner review.
- Constants: `launch_age_years` = 20 [I] (`constants.md` line 164).
- Effect measured: not measured.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended); launch age 20

**`M7.C.1h`** (spec line 1050) — Partners are declared in config, never invented by the engine. At pairing the partner enters with a basic level drawn within ±1 of the member's, the couple forms a new household and family with a budget set from their level, and the new tie takes investment from the member's other ties. Pairs that fail to form are not built. New. **Owner decided** "Provide settings for partners"; the row contents, the level draw and the budget formula are Claude's.
- Source: owner decision 2026-10-13 · P14; Claude's recommendation 2026-10-13, pending owner review.
- Constants: `budget_per_undifferentiation` = 1.49 [I] (`constants.md` line 175); `new_tie_investment_share` = 0.2 [I] (line 178). Partner rows drafted in `config/bowen/partners_phase_d.md` (plan §4.5: Leo pairs at 28, Nadia at 27, Pia at 29, two children each at 2 and 5 years after pairing).
- Effect measured: not measured.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended)

**`M7.C.1i`** (spec line 1052) — Each partner row also declares the couple's children and when they are born. A newborn gets a stable identifier from the parents and birth order, the parents' mean level, chronic anxiety from `M7.B.1a`, and declared starting ties; the five situations that raise parental focus are checked at the birth. New.
- Source: Claude's recommendation 2026-10-13, pending owner review.
- Constants: `birth_focus_gain` = 0.1 [I], `new_tie_investment_share` = 0.2 [I] (`constants.md` lines 178–179).
- Effect measured: not measured.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended)

**`M7.C.1j`** (spec line 1054) — Each family member's age at death is drawn once, when they enter the run, from a curve whose risk rises with age; the death falls on the yearly tick at that age. The owner has stated no rule for when people die. The coach does not die; she retires. New.
- Source: Claude's recommendation 2026-10-13, pending owner review.
- Constants: `mortality_a` = 0.0000657 [I], `mortality_b` = 0.085 [I] (`constants.md` lines 173–174).
- Effect measured: `tests/bowen/test_life_course.py::test_m7c1j_hazard_rises_with_age` (line 314) asserts the median age at death lies between 75 and 85 over 200 evenly spaced draws. No ensemble measure.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended)

**`M7.C.1k`** (spec line 1056) — `FIND_WORK` is not a relationship move. After a lost income, each household member aged 16 or over may take it each week with a chance that rises with how anxious they are; it always succeeds. Who takes it and when must come from their state, not a schedule. New. **Owner decided** "Trying equals finding" and that members find work themselves; the chance rule is Claude's.
- Source: owner decision 2026-10-13 · P13; Claude's recommendation 2026-10-13, pending owner review.
- Constants: `working_age_years` = 16 [I], `find_work_rate` = 0.1 [I], `find_work_pressure_scale` = 10.0 [I] (`constants.md` lines 165–167).
- Effect measured: not measured. The plan's test was renamed to `test_m7c1k_find_work_follows_pressure_not_a_schedule` because this form replaced learning (plan §4.7, lines 1173–1176).
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended)

**`M7.E.1`** (spec line 1078) — For a child, `M7.E.1f`'s development rule replaces the estimator from 5 to 20; the estimator governs from 20. Amendment. **Owner set the ages and the band (P14)**; replacing the estimator with `M7.E.1f`'s rule is Claude's (the spec says so at line 1078).
- Source: owner decision 2026-10-13 · P14; the replacement is Claude's recommendation 2026-10-13, pending owner review (stated in the text).
- Constants: see `M7.E.1f`.
- Effect measured: see `M11.C.6`.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended)

**`M7.E.1f`** (spec line 1090) — A child's basic level is held at the parents' mean until 5, then moves each year until 20 by how the parents and the child are functioning above or below their basic level, never more than 2% from that mean. From 20 the estimator governs. The spec records an open conflict: under ±2%, ten generations can fall at most from 100 to about 82, so `M11.C.6`'s limit (a) cannot be met; one of the two must change. New. **Owner decided** the ages, the ±2% band and the direction; the update rule and its rate are Claude's.
- Source: owner decision 2026-10-13 · P14; Claude's recommendation 2026-10-13, pending owner review.
- Constants: `development_start_age_years` = 5 [I], `development_end_age_years` = 20 [I], `development_band` = 0.02 [I] (the owner's values), `development_rate` = 0.1 [I] (`constants.md` lines 159–162).
- Effect measured: basic level at 20 was 39.31, 39.33, 39.35 for the three children (ad hoc diagnosis, 20 seeds; plan line 1256). See `M11.C.6`.
- **Owner decision still open** (plan §4.5, line 1154): which changes, the band or limit (a). Default built: neither; (a) reported as failing by construction.
- [ ] Approve  [x] Change: the band opens with the parents' acts toward the child — "No attention +-2%"; "supportive, responsible behaviour attention - up to 5% growth"; "projection up to 10% negative" (owner 2026-10-16, decided after the review, when Claude reported that ±2% also ruled out `FE07.7`'s 5–10-point step and `FE08.1`'s quantum jump; the first reading of revision 28 had kept ±2%). Form in plan §4.11, R31.B7  [ ] Reject

#### Criteria and readouts

**`M11.C.6`** (spec line 1318) — Adds a note that limit (a), the top-to-bottom fall within about ten generations, cannot be met under `M7.E.1f`'s ±2% band; the conflict is open for the owner. Amendment.
- Source: plain ⟦rev28⟧ tag, no source named.
- Constants: none.
- Effect measured: on the twelve-person family over 40 years, 50 seeds: +0.0177 ± 0.028 (p 0.894); the most-projected child's development is not lower. The plan says the test cannot tell the band from the absence of an effect (plan §4.7, lines 1201, 1222–1224).
- **Current verdict: FAIL** (`docs/phase_d2_ensemble_record.md`, per plan line 1201). The plan also notes 40 years covers one transmission, generation 2 to 3 (line 1139).
- [ ] Approve  [x] Change: option A — keep limit (a)'s first half ("MUST NOT complete in one or two generations"), drop "achievable within roughly ten", and record KS11.1 against the owner's ±2% band as contested `[X]` in `model_explainer.md`  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended)

**`M11.G.4`** (spec line 1526) — Children are now born in the run, but the evaluation's projection-line component is still not read; it is reported as not built. Amendment.
- Source: plain ⟦rev28⟧ tag, no source named.
- Constants: none.
- Effect measured: not measured.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended)

#### The state and mechanism register (`M14.A`)

**`M14.A` register rows** (spec lines 1636–1638, 1683–1684, 1723–1725) — Lists the new state and its writers: `Person.age_years`, `earner`, `inactive`; `Family.household`, `dissolved`; and three mechanisms (the life course on the yearly tick, death and its disposition, finding work in step 8) with what each reads and writes. Amendment; bookkeeping for the requirements above.
- Source: plain ⟦rev28⟧ tags, no source named.
- Constants: none.
- Effect measured: not applicable.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16 ("continue to finish the review process", read as: as recommended); the tag becomes "Claude's recommendation 2026-10-13, approved by the owner 2026-10-16" (inconsistency 7)

## Revision 29 — cutoff and focus

Revision 29 holds the parents' generation's cutoff for the whole `M11.C.10` run and reads tie states, lets a person learn from what follows a cutoff they witness, splits cutoff into reactive and principled, builds a child's chronic anxiety over all of childhood, lets a parent's anxious focus settle on a child's weakness and steer acts toward that child, and reads `M11.C.2` by age. It applies the owner's decisions of 2026-10-13; the forms are Claude's recommendation, pending owner review. Its plan (`docs/implementation_plan_phase_d.md` §4.8) was approved by the owner ("i approve proceed") and built 2026-10-14.

**`M1.B.3c`** (spec line 338) — A person who holds a principled cutoff on a tie no longer feels the reconnection pull's "too little" side on that tie. Amendment.
- Source: owner decision 2026-10-13 (a placeholder for principled cutoff); the form is Claude's recommendation 2026-10-13, pending owner review.
- Constants: none.
- Effect measured: not measured separately. The reference family declares no principled cutoff (line 344).
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M1.B.3d`** (spec lines 340–344) — Every held cutoff is either reactive or principled. A reactive cutoff (every cutoff the policy chooses) raises the cutter's anxiety each week, more for a less differentiated person; a principled cutoff is declared in the family config for one member of a tie, costs that person nothing and does not register as a loss for them, while the other member still feels the cut. New.
- Source: owner decision 2026-10-13, quoted in the spec: "a place holder for the impact of cut-off increasing anxiety (reactive cutoff / immaturity) or a principled cutoff for safety (abusive relationship for example)". The two forms are Claude's recommendation 2026-10-13, pending owner review. The owner decided the substance; review the form.
- Constants: `reactive_cutoff_gain` = 0.1 [I], owner-accepted placeholder (2026-10-13), scaled by (100 − `basic_level`) ÷ 100 (`config/bowen/constants.md` line 181).
- Effect measured: the reactive cost raises the cutter's anxiety by about +1 point of excess over 5 years (plan §4.9, point 4, citing the explainer §8.2). Removing the reactive rise in `M11.C.10` (rev 29): +65.7 ± 169.9, no detectable effect (plan §4.8, Built). In `M11.C.10[family of origin]` (rev 30): +163 ± 117, does not turn it red (spec line 1322).
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M4.C.1c`** (spec line 708) — A person holding a principled cutoff does not feel the standing "too little" load on that tie. Amendment, the companion of `M1.B.3c` and `M1.B.3d`.
- Source: owner decision 2026-10-13; form Claude's recommendation 2026-10-13, pending owner review.
- Constants: none.
- Effect measured: not measured.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M4.D.6f`** (spec lines 799–804) — When a person witnesses a cutoff, they note how everyone they are tied to is functioning; 13 weeks later their learned value for cutting off moves toward what they saw. If the family functions better after the cutoff, cutting off looks better to them; if worse, it looks worse. The kind of cutoff is not read, and the spec states a gap: a witness who learns cutoff is good gets a cutoff that is reactive when they choose it. New.
- Source: owner decision 2026-10-13, quoted in the spec: "add learning by witnessing cutoff", "It depends on what the witness sees happen to the cutoff person and others", and "Principled cutoff should teach that this is good action (it's functioning up)." Form Claude's recommendation 2026-10-13, pending owner review. The owner decided the substance; review the form (bond-weighted mean of functional change, the 13-week wait, everyone the witness is tied to).
- Constants: `witnessed_cutoff_gain` = 0.05 [I], owner-accepted; `witness_window` = 13 weeks [I] (`constants.md` lines 182–183).
- Effect measured: in the `M11.C.10` diagnosis the children's learned `CUTOFF` value was negative in both arms (−0.24 to −0.90): witnessing taught against cutoff (plan §4.9, ad hoc, 20 seeds). Removing it in `M11.C.10` at rev 29: +37.7 ± 181.2, no detectable effect (plan §4.8). Removing it in `M11.C.10[family of origin]`: −75.9 ± 132, p 0.91, turns the variant red (`docs/phase_d_mutation_record.md`, `witnessed-cutoff-learning-removed`).
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended; text fix: the stated gap sentence ("a person's own cutoff is principled only when declared") is updated for revision 30's `M1.B.3e`, under which a chosen cutoff toward a person felt as dangerous is principled in proportion

**`M7.B.1a`** (spec line 1029) — A child's chronic anxiety, before it is fixed at 12, now reads the mean of their yearly loads over every childhood year seen in the run, not only the year at 11. Amendment.
- Source: owner decision 2026-10-13, quoted in the spec: "chronic Anxiety builds over childhood." The owner confirmed the mean rather than a running sum ("builds over childhood - correct", plan §4.8 line 1291 and 1359). Form Claude's recommendation 2026-10-13, pending owner review.
- Constants: none new; uses `witnessed_gain` = 0.5 [I] from revision 28.
- Effect measured: not measured (tests only, plan §4.8 R1).
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M1.B.8a`** (spec lines 364–368) — While a parent is more anxious than usual, their attention goes more to a child who functions below the child's siblings at home, or who has a declared weakness. No child is named; the reference family declares no weakness. New.
- Source: owner decision 2026-10-13, quoted in the spec: "Parents anxious focus CAN get set on one child e.g. child has some real or percieved weakness." Owner: "no child named, it just evolves" (plan §4.8 line 1288). Form Claude's recommendation 2026-10-13, pending owner review. The owner decided the substance; review the form (the sibling comparison, `weakness` × 100 added to the shortfall).
- Constants: `perceived_weakness_gain` = 0.01 [I], owner-accepted (`constants.md` line 184); `weakness` per child, 0 to 1, 0 unless declared [I] (field register line 1639).
- Effect measured: removing `M1.B.8a` together with `M4.D.1g` in `M11.C.2`: +0.45 ± 0.99, loses the effect (plan §4.8, Built, ad hoc, 150 seeds). Not measured alone.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M7.E.1c`** (spec line 1084) — A declared `weakness` is the "reality defect" situation, and it acts from `t0` through `M1.B.8a`'s weekly shortfall rather than at birth. The other four situations do not apply at `t0`. Amendment.
- Source: owner decision 2026-10-13; form Claude's recommendation 2026-10-13, pending owner review.
- Constants: none.
- Effect measured: not measured.
- Note: the spec (line 1084) calls the other four situations at `t0` "an open owner decision"; plan §4.8 line 1355 records it as "Accepted by the owner: not applied". Unclear from the spec which is current.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended; text fix (inconsistency 4): the spec records the owner's decision (plan §4.8) that the other four situations are not applied at `t0`, instead of calling it open

**`M4.D.1g`** (spec line 753) — A person's anxious acts that move toward someone (`OVERFUNCTION`, `PURSUE`, `TRIANGLE`) are more likely to land on whoever holds more of their attention. It changes whom the act lands on, not how often that kind of act is chosen. Withdrawing acts and the self-directed channel are not steered. New.
- Source: owner decision 2026-10-13; form Claude's recommendation 2026-10-13, pending owner review.
- Constants: `focus_steer_gain` = 1.0 [I], owner-accepted (`constants.md` line 185). Plan §4.9 item 9 lists its size as an open owner decision ("has no measurable effect in Phase C").
- Effect measured: before it, the reference family's parents spread projection-type acts evenly over three children (spec line 753, plan §4.7). After the review fix, the more-projected child's share in Phase C was 0.578 against 0.580 with steering off (plan §4.8, ad hoc, 20 seeds). In the twelve-person family, act share 0.46 at gain 1 against 0.56 at 20 (plan §4.10, ad hoc).
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended; `focus_steer_gain` stays at 1, closing plan §4.9 item 9

**`M11.C.2`** (spec line 1314) — On the twelve-person family, projection-type acts are counted per year while the child is under 20, and symptom load is read as the child's mean over ages 20 to 30, instead of over the same calendar years. Amendment.
- Source: owner decision 2026-10-13, quoted in the spec: "Child load is by age". Form Claude's recommendation 2026-10-13, pending owner review.
- Constants: `projection_until_age` = 20, `load_from_age` = 20, `load_to_age` = 30 (`config/bowen/criteria.md` lines 155–157; no grade given).
- Effect measured: PASS at revision 29, +1.31 ± 0.95, p 0.005, 150 seeds; witness path off +0.27 ± 0.75 (plan §4.8, Built). FAIL at revision 30, +0.70 ± 1.5, p 0.16 (`docs/phase_d2_ensemble_record.md` line 19; plan §4.10 Results).
- Note: the spec still says the ages and `renew_every` "are to be added" to `criteria.md` at the build; they are there (lines 155–163).
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M11.C.10`, revision 29 amendment** (spec line 1322) — The treatment holds the Ravi–Dario cutoff for the whole run, renewed by a scripted reactive `CUTOFF` every 13 weeks; the baseline holds the same tie open. The readout is cut-off tie-weeks (rejection held) by generation 3 and their children, not cutoff acts. Required mutation: `M4.D.6f` and `M1.B.3d`'s reactive rise both removed. Amendment.
- Source: owner decision 2026-10-13, quoted in the spec: "hold the cutoff for the run and read tie states". The mutation rule was accepted by the owner (plan §4.8 line 1351). Form Claude's recommendation 2026-10-13, pending owner review.
- Constants: `renew_every` = 13 weeks (`criteria.md` line 163; no grade given).
- Effect measured: FAIL, −63.1 ± 200, p 0.71, 100 seeds; neither path had a detectable effect (plan §4.8, Built). Diagnosis: paired difference −31 (sd 1,179) (plan §4.9).
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended; as history: the superseded revision 11 and 29 clauses in the row are struck

**Supporting edits, revision 29** — `M6.4`'s stock-and-flow table (line 979) adds the reactive cutoff's weekly cost as a logged source. The field register adds `weakness` (line 1639) and `principled` (line 1652).
- Source: follow from `M1.B.3d` and `M1.B.8a` above.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

## Revision 30 — origin cutoff, danger, focus that locks in

Revision 30 makes a parent's cutoff an input to a child's adult intensity and closeness-seeking, adds felt danger read from received intrusion (so a cutoff toward a dangerous person is principled in proportion), makes an anxious act land with more force, makes a parent's focus lock onto one child, rewrites `M11.C.10`'s readout and seeds with a family-of-origin variant, and adds the premise `M11.C.47`. It applies the owner's decisions of 2026-10-14; the forms are Claude's recommendation, pending owner review. Its plan (§4.10) was approved by the owner ("approved, partner and own parents, b as second variant") and built 2026-10-14, with the owner's further choices on force, danger and focus made during the build.

**`M1.D.8a`** (spec line 464) — Each child keeps a running mean, from birth to 20, of the largest rejection either living parent holds on any tie. At 20 this "origin cutoff" raises their chronic anxiety once and raises how much contact they want on every tie for the rest of their life. Lower functioning and more pursuit are left to follow; neither is written. New.
- Source: owner decision 2026-10-14, quoted in the spec: "a parent cutting off represents a lower level of differentiation and a higher level of intensity in a system. Both of these should bear on the children to lower their level of functioning and increase their tendency to fuse (pursuit a lot)." This is option (a) of plan §4.9. Form Claude's recommendation 2026-10-14, pending owner review. The owner decided the substance; review the form (the window, the two inputs, set once at 20).
- Constants: `origin_cutoff_anxiety_gain` = 10.0 [I]; `origin_fusion_gain` = 0.5 [I] (`constants.md` lines 186–187); uses the existing `development_end_age_years` (20).
- Effect measured: `M11.C.10` main variant FAIL, +6.64 ± 130, p 0.43, 200 seeds (`phase_d2_ensemble_record.md` line 16). In the family-of-origin variant, removing `M1.D.8a` does not turn it red (+240 ± 116 against +162 ± 120), so the variant's pass is not evidence for `M1.D.8a` (spec line 1322; plan §4.10 Results).
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M1.B.3e`** (spec lines 346–352) — Each person feels a level of danger from each tie, from 0 to 1, read from a 26-week average of the intrusion they receive from that person above a threshold; a declared `danger` sets a floor. Accepted contact from a person felt as dangerous raises the recipient's anxiety, and a refused move costs nothing. A cutoff toward danger is principled in proportion to the danger felt (the reactive cost and "too little" term are multiplied by 1 − felt danger). The engine never names an abuser. New.
- Source: owner decision 2026-10-14, quoted in the spec: "refusing contact with a person that is abusive or is a serious threat (criminal, drug addict) where contact is very risky for one's safety", and "Agents, like people, can become demanding to the point of being abusive, either because of tremendous stress or low level of differentiation." Form Claude's recommendation 2026-10-14, pending owner review. The owner decided the substance; review the form.
- Sub-rule decided by the owner (line 350): felt danger does not fall while rejection is held, quoted: "hold felt danger while rejection is held" (owner decision 2026-10-14; found by the correctness review).
- Constants: `danger_window` = 26 weeks [I]; `danger_threshold` = 0.25 [I]; `danger_gain` = 2.0 [I] (`constants.md` lines 188–190); `danger` per person, 0 to 1, 0 unless declared (field register line 1641).
- Effect measured: an ordinary family's most intrusive pair averages about 0.2 over 26 weeks, below the threshold (spec line 352, ad hoc). Before `M4.D.1h`, danger never emerged: peak intrusion 0.14 to 0.17 whatever the level or stress (plan §4.10 Built, 10 seeds). Removing danger in `M11.C.10[family of origin]`: +102 ± 126 (spec line 1322). The reference families declare no danger.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M4.D.1h`** (spec line 755) — An automatic act's intensity rises with the actor's excess anxiety, up to three times the base intensity at the current gain. Self-directed acts keep the base intensity. The higher intensity carries into everything that reads it (intrusion, a cutoff's rejection, a refusal, a functioning move). Applies in every run. New.
- Source: owner decision 2026-10-14; the owner chose "force rises with anxiety" (quoted in the spec). The saturating form and reading excess rather than excess above usual are Claude's recommendation 2026-10-14, pending owner review. The owner chose the mechanism; review the form and the value.
- Constants: `force_anxiety_gain` = 2.0 [I] (`constants.md` line 192). Plan §4.10 records that 2 was built because at 4 a calm couple reached felt danger 0.046.
- Effect measured: unbounded, a run diverged within 23 weeks (spec line 755). At gain 2, felt danger 0.003 (calm, level 50) against 0.125 (level 15, stressed) (plan §4.10 Built, ad hoc).
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M1.B.8b`** (spec line 370) — Two terms make a parent's focus settle on one child. (1) While a parent is more anxious than usual, attention among the children shifts toward the child who already has more of it; the total is unchanged and equal children stay equal. (2) Each projection-type act a parent sends adds a little to their attention on that tie. No child is named or favoured across seeds. New.
- Source: owner decision 2026-10-14, quoted in the spec: "In a family focus can get stuck onto mainly one child, once it starts, it tends to grow." The owner chose "anxious focus compounds", then "saturate the investment", then "Shift, gain 10" (quoted in the spec). Form Claude's recommendation 2026-10-14, pending owner review.
- Constants: `focus_compound_gain` = 10.0 [I], value chosen by the owner ("Shift, gain 10"; raised from 2, `constants_changes.md` line 184); `focus_reinforce_gain` = 1.0 [I] (`constants.md` lines 191, 193).
- Effect measured: before it, the top child's share of the parents' attention fell from 0.62 to 0.47 over the years at home. Term (2) alone did not lock focus in at any gain tried. As built: share 0.87, 0.84, 0.70 and 0.69 in years 1, 3, 5 and 10; peak investment 51 median, 94 max (spec line 370; plan §4.10 Built; all ad hoc). The first, unbounded form reached 14,000 median, 4 million max (learning-qa, ad hoc).
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M11.C.10`, revision 30 amendment** (spec line 1322) — Mechanism is now `M1.D.8a`. The readout becomes the tie-weeks on which generation 3 hold rejection toward their partner or their own parents; rejection of their own children is reported, not tested. Seeds fixed at 200. A second variant, `M11.C.10[family of origin]`, holds both parents' cutoffs from their own parents for the run. Main variant mutation: `M1.D.8a` removed must turn it red; its two inputs are each reported. Amendment.
- Source: owner decision 2026-10-14, quoted in the spec: a parent's cutoff "should bear on the children to lower their level of functioning and increase their tendency to fuse"; "applies to one's partner's and parents, less to one's children", read as "partner and own parents"; "200 instead"; "b as second variant". Form Claude's recommendation 2026-10-14, pending owner review.
- Constants: `seeds` = 200, `renew_every` = 13, `years` = 40 (`criteria.md` lines 162–164).
- Effect measured: main variant FAIL, +6.64 ± 130, p 0.43; family-of-origin variant PASS, +162 ± 120, p 0.007; 200 seeds each (`phase_d2_ensemble_record.md` lines 16–17). Mutation results for the variant as listed under `M1.D.8a`, `M1.B.3d`, `M1.B.3e` and `M4.D.6f` (spec line 1322, measured 2026-10-15, ad hoc). The record prints both rows under one id (plan §4.10, TODO).
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended

**`M11.C.10[family of origin]` mutation clause restated to `M4.D.6f`** (spec line 1322) — For the family-of-origin variant, the required mutation is `M4.D.6f` removed; `M1.D.8a` removed is reported. The main variant's clause stays `M1.D.8a`. Recorded in `docs/phase_d_mutation_record.md` (`witnessed-cutoff-learning-removed`): red, FAIL, −75.9 ± 132, p 0.91, 200 seeds.
- Source: owner decision 2026-10-15, quoted in the spec: "restate the clause to M4.D.6f and add the mutant".
- **Decided 2026-10-15.** No box needed.

**`M11.C.47`** (spec lines 1359 and 1411) — A new premise check on the danger fixture: a spouse at a low level who meets a stressor every 2 weeks must be felt as more dangerous by others at home, and the cutoff toward them must be more principled, than the same spouse at a higher level with no stressors. Weeks of rejection toward them are reported. Classed as a premise (line 1411): a pass confirms the code and is never reported as a finding. New.
- Source: owner decision 2026-10-14, quoted in the spec: "this might need a seed with a low level of differentiation and higher stressors". Form Claude's recommendation 2026-10-14, pending owner review.
- Constants: `weeks` = 260, `level_low` = 15, `level_high` = 50, `stress_every` = 2, `stress_kind` = JOB_LOSS, `stress_intensity` = 120.0 (`criteria.md` lines 165–170; no grade given).
- Effect measured: PASS, 150 seeds: `felt_danger` +0.0495 ± 0.017; `principled_cutoff` +0.0202 ± 0.0083; `held_weeks` −16.6 ± 4.9, reported (`phase_d2_ensemble_record.md` line 18). Most of the danger is the child's, not the partner's (spec line 1359; 8 seeds, ad hoc). The stressed person is cut off on fewer weeks than the baseline person, "reported, not explained yet" (plan §4.10 Results). Mutation (intrusion read removed) reads 0 in every seed (100 seeds, ad hoc).
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended; the criteria settings (`M11.C.2` ages, `renew_every`, `seeds`, the `M11.C.47` fixture values) are graded `[I]` (inconsistency 6)

**Supporting edits, revision 30** — `M6.4`'s table (line 979) adds accepted contact from a person felt as dangerous as a logged source (`danger_contact`). The Phase D gate (`M12.5`, line 1561) adds `M11.C.47`. The field register adds `origin_cutoff` (line 1640), `danger` (line 1641), `received_intrusion` (line 1653) and `felt_danger` (line 1654).
- Source: follow from `M1.D.8a`, `M1.B.3e` and `M11.C.47` above.
- [x] Approve  [ ] Change: ____  [ ] Reject — owner 2026-10-16, as recommended
