# Proposal — drawing a run as a fam-diag diagram

> **Status: built 2026-10-10.** D1, D3, D4 and D5 were approved and D2 corrected by the owner (§0). The plan (§10) was
> approved the same day and built. The status report is §11. Nothing is built. Revision 1 drafted act-count
> rules for each line; the owner corrected them the same day (§1). Every threshold below is invented (`[I]`).

## 0. Owner decisions on revision 2 (2026-10-10)

- **D1 — yes.** §4's definitions become the declared readouts (`docs/readouts_phase_c.md`).
- **D2 — corrected: a pattern ends when a new pattern emerges.** No end rules of their own. §4's "Ends when" column is
  replaced by this rule.
  - **No interaction is an emerging level of distance.** A tie with no interaction reads as distance, and the level rises the
    longer it lasts.
- **D3 — yes.** Cutoff is drawn as rejection now. The engine's CUTOFF ("end contact") is opened as a separate decision item.
- **D4 — yes.** Projection is drawn now from §4's readout, and built into the model as the first Phase D item.
- **D5 — the owner's family.** Confirmed reading:
  - Grandparents: Teodor and Ana (Marta's parents); Sofia and a new husband (Ravi's parents).
  - Parents: Ravi and Marta.
  - Parents' siblings: Iris (Marta's sister) and one new sibling of Ravi, **cut off from Ravi at the start**.
  - Children: Leo, Nadia and Pia.
  - Coach: Dr Halim.
  - Bruno and Toma leave `M2.A`.

## 1. Owner corrections to revision 1 (2026-10-10)

- **Two kinds of line.** *Relationship lines* show structure: partners, parent and child. They are always drawn.
  *Emotional pattern lines* show a **repeating** pattern of activity: **3 or more times in a month** (4 weeks). A single act is
  an event, not a line.
- **A pattern line ends when behaviour shows the pattern has ended**, not just when acts stop. For example, sustained
  interaction ends a cutoff.
- **Cutoff is not the absence of interaction.** Absence is distance. Cutoff is the **intensity of rejecting
  interaction**: A refuses to interact with B.
- **Pursuit is fusion, projection, or over/underfunctioning:** going along to stay in agreement with the other.
  It is not a separate line.
- **Projection must be in the model.** It is an important emotional pattern and part of the triangling process.
- **The family:** grandparents (4), parents (2), children (3), the parents' siblings (2), the coach (1). See §6.
- **Block size of 4 weeks: approved.**

## 2. What this is for, and what it must not do

- To show one run as a family diagram you can step through, for the plan §9 review and the Run Review Log.
- It is a readout. Nothing in it passes or fails, and it changes nothing the engine computes.
- It must not draw a pattern the run did not produce. Every line must trace back to that run (§8).
- It is not a real family (`M11.F.9`). The years are synthetic (1001–1026), and a page note says so.

## 3. One definition of each pattern, not two

The model already declares pattern readouts: `docs/readouts_phase_c.md`, code `src/bowen/readouts/patterns.py`.
They were written for criteria, and are still awaiting your plan §9 review. Revision 1 invented separate rules for the
picture, which would have given "conflict" two meanings. **Revision 2 proposes changing the declared
readouts to your definitions, and the picture uses them.** Your corrections are therefore also the §9 review of
`readouts_phase_c.md`.

- No criterion reads these readouts today (checked: `src/bowen/ensemble/criteria.py` imports none of them). So
  changing them moves no criterion result.
- Cost: the readouts and their constants are inside the engine hash, so the change triggers the full record
  regeneration (about 1 hour). Recommended: batch it with the next engine change, as `CLAUDE.md` asks.

## 4. The patterns, as proposed

"Repeating" = **3 or more acts in 4 weeks** (owner). Counts are of acts between the two people in either direction unless
stated. Each pattern has a start rule and an **end rule based on behaviour**. The end rules are my proposals.

| Pattern | fam-diag symbol | Starts when, in 4 weeks | Ends when, in 4 weeks | Change from the declared readout |
|---|---|---|---|---|
| **Conflict** | conflict line | ≥ 3 CONFLICT acts on the tie, **both** members contributing | ≥ 3 calm approaches (STAY-IN-CONTACT) on the tie and no CONFLICT | was ≥ 2 from each member in 8 weeks |
| **Distance** | distance line | ≥ 3 DISTANCE acts, more than the approaches | approaches outnumber withdrawals | was 8 weeks; "both below their contact optimum" kept |
| **Over/underfunctioning** (fusion with the over side marked) | fusion line, `adequatePersonId` = over side | ≥ 3 acts in which one member takes charge (OVERFUNCTION) or the other gives way (UNDERFUNCTION) or goes along (PURSUE, §5), all in the same direction | ≥ 3 of these acts in the reverse direction, or the hardened habit leaves its pole | adds PURSUE; was "habit at pole plus one act in 8 weeks" |
| **Fusion** (no over side) | fusion line | ≥ 3 PURSUE / going-along acts **both ways** between the two | either member's DISTANCE or I-POSITION reaches ≥ 3 | new |
| **Projection** | projection line, parent → child | ≥ 3 acts from a parent focused on a child: OVERFUNCTION or PURSUE toward the child, or TRIANGLE that recruits the child | the parent's child-focused acts fall below 3 while the child's functioning holds | new; §5 |
| **Cutoff** | cutoff line, from the one refusing | ≥ 3 refusals by A: A sends CUTOFF to B within a week of an approach from B (the model's "withheld" acts are acts A held back, not refusals, so they do not count) | **sustained interaction**: contact acts on the tie in each of the 4 weeks with no refusal | was "tie non-interactive for 8 weeks". That is absence, which you say is distance; see §5 |
| **Fixed triangle** | triangle | the same outsider across ≥ 3 activations in a row (unchanged) | a different outsider, or the triangle stays inactive for 4 weeks | end rule added |

**Events** (single acts, dated, on the line or person): every act above, plus I-POSITION, DETRIANGLE, SYMPTOM_ONSET
(the person), JOB_LOSS and other exogenous events (NODAL), and Dr Halim's sessions.
"Open connection" from revision 1 is dropped as a line, because it is not a pattern. STAY-IN-CONTACT shows as an event.

## 5. Questions about the model these corrections raise

These are about the engine and the spec, not about the picture.

- **Q1 — CUTOFF in the engine is "end contact".** A delivered `CUTOFF` makes the tie non-interactive and cut off
  (`src/bowen/engine/event_effects.py:120`, spec `M1.B.3`, explainer table: "End contact"). `REDUCE_CUTOFF` restores it.
  In the seed-7 run that happens 62 and 70 times, so ties are severed and restored about every other week.
  - On your definition, the act of severing is a refusal, so it fits. But the **state** the engine leaves behind (no
    interaction) is what you call distance.
  - The readout in §4 follows your definition and does not read the state.
  - **Whether the engine should change** (for example, a cutoff that keeps the tie interactive but makes the cutter
    reject what arrives) is a spec decision. It would go to `docs/DECISIONS — …` as its own item.
- **Q2 — projection is specified but not built.** The spec has it as Phase D work:
  - the child-projection sink (`M1.D.1`), which `src/bowen/engine/sinks.py` already counts from OVERFUNCTION, PURSUE and
    TRIANGLE from a spouse to a child;
  - cross-person reinforcement (`M4.D.6e`);
  - `M11.C.2` (symptoms concentrate on the projection target) and `M2.A.2` (Nadia becomes the target as an
    outcome, not by assignment).

  The §4 readout uses the same act set as the sink, so the picture and the sink agree.
  It can show only what the current engine does; the engine does not yet *learn* projection (`M4.D.6e`).
  Building projection is the first Phase D item, not part of this exporter.
- **Q3 — pursuit.** Today PURSUE is one of the seven core acts (`M5.A.1a`). On your reading, the act stays and its
  **pattern** is fusion, over/underfunctioning, or projection depending on who pursues whom; §4 does that.
  No engine change is needed unless you want PURSUE to stop being its own act.

## 6. The family: the spec and your description differ

Spec `M2.A` declares eleven family members plus Dr Halim. The Phase C runs simulate only eight of them (`config/bowen/family_phase_c.md`).

| Generation | Spec `M2.A` | In the Phase C run | Your description |
|---|---|---|---|
| 1 | Teodor (Ana's husband), Ana, Sofia (Ravi's mother; no husband listed), Bruno (Ana's brother) | Ana, Sofia, Bruno | 4 grandparents |
| 2 | Ravi ("eldest of 3", siblings not listed), Marta, Iris (Marta's sister) | Ravi, Marta | 2 parents + 2 siblings |
| 3 | Leo (launched), Nadia, Pia, Toma (Iris's son) | Nadia, Pia | 3 children |
| — | Dr Halim | Dr Halim | coach |

- Your count reads as four grandparents. That would be Teodor, Ana, Sofia and Sofia's husband, the last of whom is not in the spec.
  It also reads as one sibling for each parent: Iris for Marta, and one of Ravi's two unlisted siblings.
- The spec has Bruno (a great-uncle, cut off) and Toma (a cousin) in place of those.
- **Which is right is a spec question**, decision D5 below. Until the four Phase D members run, the diagram would draw
  them greyed, labelled "not simulated in this run", with relationship lines only.

## 7. Decisions for you

- **D1 — make §4 the declared readouts** (one definition, batched regeneration). Recommended: yes.
- **D2 — the end rules in §4.** These are mine; you set the start rule. Approve or correct each one.
- **D3 — CUTOFF in the engine (Q1).** Draw cutoff by your definition now, and open a decision item on the engine.
  Recommended: yes.
- **D4 — projection.** Draw it now from the readout in §4, and keep building it into the model as the first Phase D item.
  Recommended: yes.
- **D5 — the family.** Correct the spec's `M2.A` to your description, or confirm the spec. Either way, the picture
  greys whoever is not simulated.

## 8. How it is checked

- **Trace-back test.** For every pattern line in the output, the start condition holds in that block. Every ended line has its
  end condition in the block where it ends.
- **Adversarial tests:**
  - CONFLICT removed from every repertoire produces no conflict lines.
  - A tie that is non-interactive with no refusals produces **no** cutoff line. This is your correction, as a test.
  - Two CONFLICT acts in a block produce an event and no line.
- **Format test.** The keys fam-diag's import needs are present, every referenced id exists, and every date parses.
- **Not automated:** fam-diag drawing it correctly. I open it in fam-diag and compare the first, middle and last block with
  `docs/review/phase_c_trace_seed7.md`, then report what I saw.

## 9. Effort

- Exporter, config and tests: about a day.
- fam-diag hiding a line after its end date: a small change to your app, under its UI regression rules. It is needed
  now, because pattern lines end.
- The readout change rides the next engine regeneration (about 1 hour, batched).
- Projection in the model and the CUTOFF engine question are Phase D and decision items, sized separately.

## 10. Build plan (awaiting approval)

IDs `V1`–`V7` are local to this proposal. Spec IDs are used verbatim.

**One regeneration, not two.** The engine hash covers `src/bowen/**` and every `config/bowen/*.md`
(`tools/ensemble_record.py:code_hash`). So both the readout change (V1) and the new family file (V3) make every record
stale, and each would cost the hour-long regeneration. They are therefore built together and regenerated once.

| ID | Deliverable | Spec | Verified by |
|---|---|---|---|
| **V1** | `docs/readouts_phase_c.md`, `src/bowen/readouts/patterns.py` and the `pattern_*` constants rewritten to §4 + §0: <ul><li>window 4 weeks, 3 acts</li><li>conflict, distance (including no interaction, with a level), over/underfunctioning, fusion, projection, cutoff as rejection, fixed triangle</li></ul> | `M5.A.1a` | `tests/bowen/test_patterns.py`: <ul><li>`test_m5a1a_conflict_needs_three_acts_in_four_weeks`: two acts read no pattern</li><li>`test_m5a1a_no_interaction_reads_distance_not_cutoff`: adversarial</li><li>`test_m5a1a_distance_level_rises_with_weeks_without_interaction`</li><li>`test_m5a1a_cutoff_needs_refusals_of_approaches`</li><li>`test_m5a1a_projection_reads_parent_to_child_acts`</li><li>`test_m5a1a_pursuit_reads_as_fusion_over_under_or_projection`</li><li>the existing document/code agreement test</li></ul> |
| **V2** | The current-pattern rule: a tie holds its pattern until a different pattern's start condition is met. If two start in the same block, the one with more acts wins, then the earlier first act `[I]`. | `M5.A.1a`, owner D2 | `test_m5a1a_pattern_holds_until_a_new_one_emerges`, `test_m5a1a_quiet_block_keeps_the_current_pattern` |
| **V3** | Spec revision 13: `M2.A` changed to the owner's family; `M2.A.0` count restated; the new members' values `[I]`, drafted for owner review. New `config/bowen/family_phase_d.md`. **Phase B and C families unchanged** (they are the closed phases' records); Bruno stays in them as history. | `M2.A`, `M2.A.0` | `tests/bowen/test_family.py::test_m2a_phase_d_family_matches_spec_table`, `test_m2a_ravi_sibling_starts_cut_off` |
| **V4** | Decision item for the engine's CUTOFF semantics (owner D3), written to `docs/DECISIONS — …` with the evidence in §5 Q1. No engine change. | `M1.B.3`, `M5.B.3` | the document |
| **V5** | `tools/famdiag_export.py` + `config/bowen/viz_famdiag.md` (symbol table, block size 4, synthetic years). Reads V1/V2's readouts; it does not define patterns itself. | `M11.F.9` (page note) | `tests/bowen/test_famdiag_export.py`: <ul><li>trace-back: every line's pattern holds in its block</li><li>a run without CONFLICT draws no conflict line</li><li>format: keys, ids and dates</li><li>the page note is present</li></ul> |
| **V6** | fam-diag: a pattern line is hidden after its end date (`isVisibleAtCutoff`). A change to your app, under its `standards/ui-regression.md`. | — | a Vitest test in fam-diag for the hide rule; the existing fam-diag suite stays green |
| **V7** | Export of the seed-7 Phase C run, opened in fam-diag. | — | **not automatable**: I compare blocks 1, 13 and 26 with `docs/review/phase_c_trace_seed7.md` and report what I saw |

**Order:** V4 (document only) → V1 + V2 → V3 → one regeneration and a sweep → V5 → V6 → V7.

**Cost:**
- V1–V3 and the regeneration: about a day, plus the hour-long regeneration.
- V5: about a day.
- V6: half a day.

**Not in this plan:**
- Building projection into the engine (`M4.D.6e`, Phase D's first item).
- Any change to the engine's CUTOFF.
- Simulating the four new Phase D members. The picture greys anyone not simulated in the run it draws.

## 11. Status report (2026-10-10)

| ID | Status | Evidence |
|---|---|---|
| V1 | done | `docs/readouts_phase_c.md`; `src/bowen/readouts/patterns.py`; `tests/bowen/test_patterns.py` (17 tests, e.g. `test_m5a1a_conflict_needs_three_acts_in_four_weeks`, `test_m5a1a_no_interaction_reads_distance_not_cutoff`, `test_m5a1a_projection_reads_parent_to_child_acts`) |
| V2 | done | `current_patterns`; `test_m5a1a_pattern_holds_until_a_new_one_emerges`, `test_m5a1a_quiet_block_keeps_the_current_pattern`, `test_m5a1a_silence_does_not_replace_a_cutoff` |
| V3 | done | spec revision 13 (`M2.A`, `M2.A.0i`, `M2.A.1`, `M2.B.2`, `M5.A.1a`); `config/bowen/family_phase_d.md`; `tests/bowen/test_family.py::test_m2a_phase_d_family_matches_spec_table`, `::test_m2a_ravi_sibling_starts_cut_off`, `::test_m2a0i_leo_declared_male` |
| V4 | done | `docs/DECISIONS — CUTOFF SEMANTICS.md` (open) |
| V5 | done | `tools/famdiag_export.py`; `config/viz/famdiag.md`; `docs/review/phase_c_famdiag_seed7.json`; `tests/bowen/test_famdiag_export.py` (17 tests, including trace-back on seeds 7, 1 and 3 and an adversarial threshold test that a mutant fails) |
| V6 | done, not merged | fam-diag branch `timeline-hides-ended-lines`, commits `4ec320f` and `7694b93`. Tests: `isSpanVisibleAtCutoff`, `isTriangleVisible`, `normalizeTriangles`, `emotionalLineSlots` and a canvas test. 1,235 tests, `tsc --noEmit` and `tsc -b` all pass. Needs fam-diag's gate file and the owner's go-ahead to merge. |
| V7 | done (by hand) | Opened in fam-diag on the branch. At 2001, 2013 and 2026 the lines drawn were exactly those the export lists: 2, 9 and 10 pattern lines; triangles only from 2017; ended lines hidden. The trace agrees at blocks 9 (Pia → Ravi going along, 3 acts) and 26 (Ravi–Sofia conflict, 3 acts, from both). |

**Changed from the plan:**
- The years are 2001–2026, not 1001–1026, because fam-diag's slider runs to the current year.
- Triangle lines carry the triangle's dates.
- fam-diag's loader now keeps a triangle's dates.
- Teodor's `basic_level` is proposed at 36 (spec inconsistency).

**Owner items arising:**
- `TODO.md`: the readout questions (lopsided pursuit; a parent's acts when projection is not read), the `[I]` values, and the missing Phase D parent ties.
- `docs/DECISIONS — CUTOFF SEMANTICS.md`.

