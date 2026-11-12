# Spec revision 39 — draft (version 2 stage 3)

> **Status: APPROVED by the owner 2026-10-25, as a whole, through `AskUserQuestion` ("Approve"), with part 1's reading
> of D19 under `M6.I.4a` (functional level returns with the pseudo-self); applied to the spec and built in full, each
> group with its code (§24), at stage 3 plan steps 6–10 (2026-10-25–26; `docs/spec_revision_log.md`'s revision 39
> lists what landed at each step and where the applied text departs from this page).** Part 1 (§1–§10): group C (P2, F1, `M6.I.4a`) and T-130's wording. Part 2 (§11–§18): group F
> (`M5.F.4`, `M5.F.2c`, `M11.C.29`'s arm, new `M11.C.49`) and group A (new `M4.C.1d`, `M4.D.3a`,
> `reciprocal_contact_share`). Part 3 (§19–§24): T-128 (`M17.E.1`, `M10.C.1a`, `config/sweep/sweep.md`) and T-014
> (`M8`, `M8.1`, `M8.5`, `M13.1`, `M11.C.7`, `M7.D.1a`). Written at plan step 4
> (`docs/implementation_plan_version_2_stage_3.md` §3.8) from §3.3–§3.5, the proposal
> `docs/PROPOSAL — DISPLACED COPING (P2).md`, the probes in `docs/review/`, and the owner's answers of 2026-10-25
> (Q1–Q6, §2 and §9; Q8–Q14, §18). Nothing under `src/bowen/` or `config/bowen/` is edited by this page, and nothing
> here moves the engine hash (`66dcbb1e8ba5`). Each group's approved text was applied in the same commit as its code
> (the owner's Q6, 2026-10-25, §9; §24 lists where each block lands); the page is kept as approved.

Every new form is Claude's recommendation, graded `[I]`, unless a ledger entry is cited. Attribution grades follow
`model_explainer.md`'s legend: `[K]` is Kerr's text (`KS`, `FE`), never Bowen's.

## 1. What this part changes

| Requirement | Change | What a run computes |
|---|---|---|
| `M1.A.5f` | **new**: displaced coping (P2) | changes (built at step 7) |
| `M1.A.5e` | "any other act leaves it" gains `M1.A.5f`'s exception; the floor paragraph gains F1's ledger, `floor_held` | changes (step 7) |
| `M1.A.4k` | the estimator's reading leaves out `floor_held` | changes (step 7) |
| `M7.E.1f` | one sentence: what `d` reads; no rule change | no |
| `M6.I.4a` | **new**: lent functioning returns with lost capacity (D19) | changes (step 7) |
| `M11.C.6` | the mutation column names the two new mutants | no |
| `M11.C.21` | the mutation column names `pseudo-self-return-removed` | no |
| `M11.5` | `M11.C.6`'s and `M11.C.21`'s notes | no |
| `M1.F.6a` | T-130: "not yet checked against the paper" replaced by the check's result | no (lands at step 6) |
| `M1.A.5c`, `M6.I.4` | **unchanged**: P2 writes no pseudo-self; `M6.I.4a` stands beside `M6.I.4` | — |

Config, applied with the code: `config/bowen/stress_response.md` gains the *displacing* table (§4);
`config/bowen/event_sizes.md`'s two item names (§6). `config/bowen/constants.md` is unchanged: no new constant.

## 2. The decisions this part rests on

| Decision | Decided | Source |
|---|---|---|
| D18, third reading | P2 as drafted plus F1; `M11.C.6` reported as read, never tuned | the owner 2026-10-25; plan §3.2 |
| D19 | `M11.C.21`: lent functioning returns when the taker's functional level falls, in proportion to the capacity lost, on `M6.I.4`'s conserved quantity | the owner 2026-10-24; plan §3.2 |
| Q4 | "draft revision 39 with P2 and F1 as recommended, reporting M11.C.6 as read" | the owner 2026-10-25; the proposal's §9 |
| Q1 | the displacing kinds: **`OVERFUNCTION` alone** (recommended) | the owner 2026-10-25, through `AskUserQuestion` |
| Q2 | F1's ledger is **shed with the swings** off the floor (recommended) | the owner 2026-10-25, through `AskUserQuestion` |
| Q3 | P2 is **conserved**: the sender gains what the receiver loses (recommended) | the owner 2026-10-25, through `AskUserQuestion` |
| T-130 wording | **quote the paper** for `JOB_LOSS` and `FIND_WORK_LOAD` (recommended) | the owner 2026-10-25, through `AskUserQuestion`; `docs/review/stage_3_t130_event_sizes.md` |

## 3. The requirement text

Each block is the text as it would stand in the spec. "Replace … with …" quotes the revision 38 text exactly, so the
edit can be applied and checked by script.

### 3.1 `M1.A.5f` — new, after `M1.A.5e`

> **M1.A.5f** — *displaced coping: a loaded person who takes charge of another meets the load through the other.* In
> a week a person is loaded (`M1.A.5e`: load above their own usual level), each act they send of a kind listed as
> *displacing* in `config/bowen/stress_response.md` **MUST** move the amount `M1.A.5e`'s coping would have given
> them, `response_up_gain × above ÷ 100 × basic_level ÷ 100`, from the act's receivers' `functional_level` to the
> sender's, split evenly over the act's living family receivers, and **MUST** carry it on both people's
> `stress_swing`, so it settles back when each is calm as `M1.A.5e` says. A receiver who holds a position toward the
> sender (`M1.B.5`) gives nothing. From each receiver, in id order, the amount that moves is the smallest of three:
> that receiver's share of the coping amount; its room above its floor, `max(0, basic_level −
> stress_floor_below_basic)`; and the sender's room below 100 left after the receivers before it. So no receiver is
> taken below the floor, the sender is not taken above 100, and the transfer is conserved exactly. The act writes no
> `pseudo_self`, so `M6.I.4` and `M1.A.5c` are untouched and the estimator (`M1.A.4k`) reads what it moves. A
> displacing act sent by a person who is not loaded does what `M1.A.5e` says of any other act: nothing. The rule reads
> no receiver's age, role, focus or readout; who sends a displacing act to whom, and under what load, is decided by
> focus (`M4.D.1g`, `M1.B.8a`), learning and the run. In the projection cycle the mother is calmer because she "sees
> the anxiety as coming from outside herself" (`KS04.7`, `[K]` `[T]`), and the cycle leaves one person in a caretaking
> position and the other in a patient or child position (`FE01.1`, `[K]` `[T]`); that the sender functions up for it
> and the receiver down is the direction those passages give (`[K]` `[D]`; `FE06.3` for the receiver's functioning
> most impaired). Reading this as what one act, `OVERFUNCTION`, does under load in any relationship is the model's
> (`[M]` `[I]`; `FE07.4`'s footnote 22, `[K]`, makes the mutual projection of parent and child the same as in all
> relationships). `FE06.3` (`[K]`) has the child give up more self than the parents; the two amounts being equal is
> the model's (`[I]`), and the exchange (`M6.I.4`) and the child's own soothing can still make the receiver's loss
> larger. The amount being the coping amount, its landing on the stress swing, the held position stopping it, the
> order of receivers, and the list of kinds are each `[I]`. No new constant.
> → `M1.A.5e`, `M1.B.5`, `M7.E.1f`, `M11.C.6`

### 3.2 `M1.A.5e` — two edits

**Edit 1.** Replace

> Any other act leaves it.

with

> Any other act leaves it, except that a loaded person's displacing act meets their load through its receiver
> (`M1.A.5f`).

**Edit 2.** After the sentence ending "with it on both, none of 20 did." and its tag `⟦rev19 · owner decision
2026-10-11⟧`, insert:

> What the floor holds **MUST** be kept on a ledger, `floor_held`: in a week the floor (or the scale's 0–100) stops
> `functional_level` from moving while the swings decay, the amount it stopped, `(change in functional_level) −
> (change in stress_swing + adjust_swing)`, is added to `floor_held`, so `functional_level − stress_swing −
> adjust_swing − floor_held` is unchanged by the bound. The swings still decay in full (the review of 2026-10-11:
> a clamped relaxation leaves no amount on the swings). In a calm week `floor_held` **MUST** be shed by
> `response_relaxation_rate`, the swings' own share, moving `functional_level` with it, and what the floor stops of
> that shed stays on the ledger. The estimator leaves `floor_held` out (`M1.A.4k`). All forms `[I]`; no new constant.
> Without it, the functioning the floor held while the swings decayed was on no ledger and the estimator credited it
> as the person's own (measured 2026-10-25, `docs/review/stage_3_displaced_coping_probe.md`).

### 3.3 `M1.A.4k` — one sentence added at the end

> The history **MUST** also leave out `floor_held` (`M1.A.5e`): functioning the stress floor held while the swings
> decayed is the floor's, not the person's own, and a basic-level rise never falls. The weekly reading is
> `functional_level − pseudo_self − max(0, adjust_swing) − floor_held`. `[I]`.

### 3.4 `M7.E.1f` — one sentence, no rule change

After "where `d` is half the parents' mean `functional_level − basic_level` over the year and half the child's own
(the circumstances: …, so siblings can differ without a rule assigning which)." insert:

> The functional levels `d` reads are each person's weekly estimator reading (`M1.A.4k`), net of pseudo-self, of a
> positive `adjust_swing` and of `floor_held`: the exchange (`M6.I.4`), a positive system adjustment (a negative one
> is read) and the stress floor do not reach development, and the stress swing, which coping, soothing (`M1.A.5e`) and displaced coping
> (`M1.A.5f`) move, does.

### 3.5 `M6.I.4a` — new, after `M6.I.4`'s row

> **M6.I.4a** — *lent functioning returns with lost capacity.* Once a week, just before `M6.I.4`'s settling back
> (update-order step 9), each living person holding borrowed (positive) pseudo-self whose capacity fell over the week
> by any cause other than `M6.I.4`'s own transfers — a collapse on the stress ledger, a symptom, `M1.A.5f`,
> `floor_held` shed — **MUST** return to the people who lent it the share of their borrowed pseudo-self equal to the
> share of capacity lost, `(s_before − s_after) ÷ s_before`, with the same of `functional_level`. `s` is `M6.I.4`'s
> room above the stress floor, `s = clip((f − m) ÷ (b − m), 0, 1)` with `m = max(0, b − stress_floor_below_basic)`
> (1 when `b = m` and `f ≥ m`, else 0), read at the week's start (`s_before`) and at this step (`s_after`), with what
> `M6.I.4`'s transfers moved that week left out of `f`; a person with `s_before` = 0 returns nothing, and so does a
> fall that stays at or above basic level, where `s` is 1. `pseudo_self` does not record lenders, so it returns to
> the partners on ties whose functioning balance names this person the over-functioner and who hold lent (negative)
> pseudo-self, in proportion to what each holds lent, none given back more than they lent nor taken above 100; a
> dead lender's share is returned against their ledger, as `M6.3e` settles a balance with the dead. Total
> pseudo-self is conserved. It is the mirror of `M6.I.4`'s capacity scaling of what an over-functioner takes (the
> owner's form for `M11.C.21`, 2026-10-16), applied to what they hold. It reads capacity, not the cause, and names no
> projection or child; whether a projection target rises past the arm without collapse (`M11.C.21`) is read, not set.
> Grade `[I]`, the owner's decision D19 (2026-10-24); no new constant. → `M6.I.4`, `M6.3e`, `M1.A.5c`, `M11.C.21`

**What this block adds to D19's wording, for the owner's review with the draft.** D19 says "lent functioning returns
when the taker's functional level falls, in proportion to the capacity lost, on `M6.I.4`'s conserved quantity". The
draft reads "functioning returns" as `functional_level` moving with the pseudo-self, as `M6.I.4`'s settling back moves
it (`moves.py::relax_pseudo_self`): without it the lender's functioning would not rise, and `M11.C.21` reads the
target's functional level; the taker, having lost capacity, then loses that much more functioning. Four forms are the
draft's, each `[I]` and copied from `M6.I.4`'s settling back or `M6.3e`: who counts as a lender (the balance names the
taker the over-functioner), the caps (what each lent, and 100), the week's timing and what `s_before` excludes, and the
dead lender's share. A reading in which only `pseudo_self` moves is the alternative; it would leave `M11.C.21`'s
target's functioning where it is.

### 3.6 `M11.C.6` — the mutation column

Replace

> `M1.A.4a`'s estimator reading the child's own history (`M7.E.1`, amended) — fixing each child's `basic_level` at its birth prior **MUST** turn this red ⟦rev11 · owner decision 2026-10-06 · A4⟧

with

> `M1.A.4a`'s estimator reading the child's own history (`M7.E.1`, amended) — fixing each child's `basic_level` at its birth prior **MUST** turn this red ⟦rev11 · owner decision 2026-10-06 · A4⟧. `M1.A.5f` and `M1.A.4k`'s `floor_held` — removing displaced coping (`displaced-coping-removed`) and crediting `floor_held` as the person's own functioning (`floor-held-credited`) **MUST** each turn it red if it passes; both are in the Phase D mutation record

### 3.7 `M11.C.21` — the mutation column

After "making it one-way **MUST** turn this red" insert: "; so **MUST** removing `M6.I.4a`'s return
(`pseudo-self-return-removed`, in the Phase D mutation record)".

### 3.8 `M11.5` — two rows

| Criterion | Class | The single rule it rests on | Note |
|---|---|---|---|
| M11.C.6 | Composite | — | partial: estimator over the child's history; generations 1–2 are initial conditions (`M2.A.0a`). The decline needs focus, learned projection, the parents' load at the time, the child's own responses and `M1.A.5f` together: `M1.A.5f` supplies the direction's sign, `floor_held` (`M1.A.4k`) none; whether it outweighs the child's own lift is read, never tuned (D18) |
| M11.C.21 | Composite *(revision 31; was Premise)* | `M6.I.4` transfer, `M6.I.4a` return | reclassified by the owner 2026-10-17; its failure is accepted as a finding; `M6.I.4a` returns lent functioning on a cause it does not choose, so a pass is a composite result |

New unit tests are premises (`M11.5`'s rule for a single rule's test): S3.1a–S3.1d (`M1.A.5f`, `floor_held`) and S3.4
(`M6.I.4a`), plan §3.7. `M11.C.2` (both variants) is unchanged, reported and not targeted.

## 4. `config/bowen/stress_response.md` — the *displacing* table (applied at step 7)

The preamble's last two sentences, "Coping — a self-directed act or `WITHHOLD` — raises functioning; any act not
listed and not self-directed leaves it. Every entry is `[I]`.", become:

> Coping — a self-directed act or `WITHHOLD` — raises functioning; a displacing act (the second table, spec revision
> 39, `M1.A.5f`) meets a loaded sender's load through its receiver; any other act leaves it. A kind may not be both
> soothing and displacing: loading fails if it is. Every entry is `[I]`; the displacing list was approved by the owner
> 2026-10-25 (`OVERFUNCTION` alone).

and a second table follows the first, read into `EventKinds.displacing`:

> ## Displacing
>
> | kind | why it meets the sender's load through the receiver |
> |---|---|
> | `OVERFUNCTION` | taking charge of another's functioning: the caretaking position (`FE01.1`) |

## 5. `docs/model_explainer.md` (applied with the code)

Under "Stress moves functioning (revisions 18 and 19, `M1.A.5e`)", two sub-bullets after "Calm is relative to the
person":

> - **Displaced coping (revision 39, `M1.A.5f`).** A loaded person who takes charge of another (`OVERFUNCTION`) meets
>   the load through them: the amount coping would have given the sender moves from the receiver's functioning to the
>   sender's, on both stress swings, and settles back when each is calm. A receiver holding a position gives nothing,
>   and the receiver's floor bounds it. `[K]` `KS04.7`, `FE01.1` for the direction ("a calmer mother"; the caretaking
>   and patient positions); the size, the list and the conservation are `[I]`.
> - **What the floor holds is on a ledger (revision 39, F1).** When the floor stops functioning from falling while the
>   swings decay, the amount it held goes to `floor_held`, which the estimator leaves out; when calm it is shed by the
>   swings' share. Without it the estimator credited floor-held functioning as the person's own. `[I]`.

Under "The estimator reads functioning without what over-functioning borrowed", a third sub-bullet: "**Nor what the
floor held (revision 39).** The weekly reading is functional level less pseudo-self, less a positive `adjust_swing`,
less `floor_held`." In §8.1, the life course, after "by half the parents' and half the child's own
functioning above basic": "each read as the estimator reads it, so the exchange and the floor do not reach it and the
stress swing does (revision 39)". In §4.4, after the paragraph on the over-functioner's capacity (`M6.I.4`), one bullet for `M6.I.4a`: "**Lent functioning returns
when the taker's capacity falls (revision 39, `M6.I.4a`).** The share of borrowed pseudo-self equal to the share of
room above the floor lost goes back to the lenders, conserved. `[I]`, the owner's D19."

## 6. T-130's wording (applied at step 6, the batch's first commit)

`M1.F.6a`: replace

> — read from the scale's published table as Wikipedia reproduces it, not yet checked against the paper (T-130) ⟦rev38 · owner 2026-10-24, stage 2 milestone review, MR-14⟧ —

with

> — checked against the paper's Table 3 (p. 216): all thirteen values agree (T-130, `docs/review/stage_3_t130_event_sizes.md`) —

`config/bowen/event_sizes.md`: the `item` column quotes the paper — `JOB_LOSS` "Fired at work"; `FIND_WORK_LOAD`
"Wife begin or stop work", its source cell adding "the gender-neutral reading and the load to the person who finds
work (`who` = self) are the model's, `[I]`, the owner's Q6 of the T-130 review (2026-10-24)"; the two missing articles corrected to the paper's
wording; "unchecked" removed from the preamble. The explainer's §11.5 sentence "read from the scale's published table as Wikipedia …" is restated the same way.
Values unchanged.

## 7. The §19 check

`M1.A.5f` states what one act does for every sender and receiver; it names no projection, no child and no outcome.
`floor_held` records functioning a bound kept and changes no act. `M6.I.4a` returns a conserved quantity on a cause it
does not choose. `M11.C.6`'s arms differ only in the projection-type acts the policy sends the child, so `M1.A.5f`
supplies the direction's sign; whether it outweighs the child's own lift is read. The proposal's §5 holds the full
check. Whether these are the set layer and not a written pattern is the owner's judgment (plan §3.12).

## 8. The stage 4 check (Q7), against `_LEDGER.md`

Read 2026-10-25: `KS04.7`, `FE01.1`, `FE06.3`, `FE07.4`, `KS00.9` against `M1.A.5e`, `M1.A.5f`, `M1.A.4k`, `M7.E.1f`,
`M6.I.4`, `M6.I.4a`, `M1.A.5c` and `M11.C.6`'s row as drafted here.

| Entry | What it says | Against the draft | Found |
|---|---|---|---|
| `FE07.4` `[K]` | the projection loop in five steps; footnote 22: "the mutual projection that occurs in all relationships" is the same | `M1.A.5f` is one rule for every sender and receiver | consistent; nothing to fix |
| `FE07.4`, `KS04.7` `[K]` | the child goes along because it is rewarded with a calmer mother (step 4) | `M1.A.5f` moves functioning only; the child's reward is learning's (`M4.D.6`), not this rule's | not in scope; no edit. The across-person reinforcement `FE07.4` names is a learning question, outside group C |
| `FE06.3` `[K]` | the parents give up self, but less than the child; the most impaired has given up most | `M1.A.5f` moves equal amounts (Q3) | recorded in `M1.A.5f`'s text: `FE06.3` has the child give up more; equal amounts are the model's `[I]`, and the exchange and the child's own soothing can still make the receiver's loss larger |
| `FE01.1` `[K]` | neither wants the position they end in; each is a prisoner of the other's functioning | the positions follow from the act's consequence, not a preference | consistent |
| `KS00.9` `[K]` | "the basic levels of the parents establish the emotional climate" | `M7.E.1f`'s parents' term reads the parents' *functioning* above basic, which `M1.A.5f` raises in a parent who displaces onto the child | **found, not fixed**: under P2 the parents' term credits the child with what a parent gained by displacing onto that child (+0.026 ± 0.019 in the joint probe). `KS00.9` names basic levels, not functioning; the term's form is the owner's (2026-10-13). The owner chose to record it (§9 Q5): T-131 |
| `KS04.7` `[K]` | the mother is calmer because she sees the anxiety as outside herself | `M1.A.5f`'s sender gains on the stress swing and settles back when calm | consistent |
| `KS04.7`, `FE01.1` `[K]` | the calmer mother; the caretaking and patient positions | `M1.A.5f` generalises the projection cycle to what `OVERFUNCTION` does under load | graded in `M1.A.5f`'s text: the passages `[T]`, the direction `[D]`, the generalisation `[M]` `[I]` (found by this session's review of the draft; the proposal's §4.1 grades the first step `[T]` `[M]`) |
| — | the floor, `floor_held`, `M6.I.4a`'s capacity share | no corpus entry states a magnitude or a floor | all `[I]`; nothing to cite |
| — | `M1.A.5c` | its table of the two channels and its consequence; it does not describe the ledgers | checked; unchanged |

## 9. Open questions for the owner

Put through `AskUserQuestion`, one at a time, recommendation first, in the session that wrote this part; both answered
2026-10-25 with the recommended option.

- **Q5, `KS00.9` against `M7.E.1f`'s parents' term.** Leave `M7.E.1f` as it is in revision 39 and record the finding
  for the stage 3 milestone review (recommended: the term is the owner's own form, D18 chose not to trace it first,
  and changing it is a new rule outside D18), or draft a rule in which the parents' term leaves out what `M1.A.5f`
  gave the parent. **Decided by the owner 2026-10-25: record it, leave `M7.E.1f`** (recommended); recorded as T-131
  (`docs/todo/2026-10-25_stage_3_revision_39.md`) for the stage 3 milestone review.
- **Q6, where the approved text lands.** Each group's spec, explainer and config text is applied in the same commit as
  its code (recommended: the documentation rule, "same commit", and the spec never describes a rule the code lacks;
  T-130's wording at step 6, group C at step 7, F at 8, A at 9), or the whole of revision 39 is applied to the spec at
  approval, before the code. **Decided by the owner 2026-10-25: with each group's code** (recommended). Until then the
  approved text stays here.

## 10. Documents that change when this part is applied

The spec (the blocks above, the front matter's status line, `version: 2.0, revision 39`); `docs/spec_revision_log.md`
(a revision 39 section); `docs/spec_index.jsonl` and `docs/spec_rules/` regenerated (`tools/spec_index.py`,
`tools/spec_rules_view.py`); `config/bowen/stress_response.md` (§4); `config/bowen/event_sizes.md` (§6);
`docs/model_explainer.md` (§5); `docs/version_2_plan.md` (stage 3's status); `docs/SPEC_REVIEW_CHECKLIST.md` and
`docs/REVIEWER_READING_LIST.md`; `CHANGELOG.md`.

---

# Part 2 — groups F and A (drafted 2026-10-25)

Written from plan §3.3.2 (group F, D20, D21), §3.3.3 (group A, D22, D23), §3.7 (S3.6–S3.15) and the I-position
probe (`docs/review/stage_3_iposition_probe.md`). Every new form is Claude's recommendation, graded `[I]`, unless a
ledger entry is cited. When applied, each new or amended block takes a `⟦rev39 · …⟧` tag naming its decision.

## 11. What this part changes

| Requirement | Change | What a run computes |
|---|---|---|
| `M5.F.4` | **rewritten**: a genuine `I-POSITION` is a mixture of self-statement and demand, weighted by the mover's self-channel weight (A, D20); the assertion form is its `w = 0` end | changes (step 8) |
| `M5.F.2c` | one sentence: no perspective threshold decides `I-POSITION`'s form | no |
| `M11.C.29` | the arm (D21): mover at level 60, F's spell paused over the act's window, completions reported | the criterion's arm (step 8) |
| `M11.C.49` | **new** criterion: a lower-level I-position raises the family's anxiety more | new (step 8) |
| `M4.C.1d` | **new**: contact that relieves its receiver returns in part to its sender (P1, D22) | changes (step 9) |
| `M4.D.3a` | one sentence: `TRIANGLE` in layer 0 (P3, D22) | changes (step 9, through config) |
| `M10.C.1a` | `reciprocal_contact_share` added to the register's list | no |
| `M11.C.42` | the mutation column names `reciprocal-contact-removed` | no |
| `M11.5` | `M11.C.49`'s row (new); notes on `M11.C.42` and `M11.C.45` | no |
| `M1.C.1` | one sentence: `bound_anxiety` is not written (plan §3.13; Q12) | no |
| `M5.D.4a`, `M11.C.32`, `M11.C.44`, `M11.C.27`, `M11.C.45`'s row | **unchanged**: the anger gate stays; `TRIANGLE` stays an inside act (D23) | — |

## 12. The decisions this part rests on

| Decision | Decided | Source |
|---|---|---|
| D20 | group F: **A**, the act part self-statement, part demand, weighted by the mover's own self-channel weight; the probe did not meet D20's condition (the act's effect on the family's excess differs by level by −0.06 ± 0.32), so D20 stands | the owner 2026-10-24; plan §3.2, §3.3.2 |
| D21 | `M11.C.29`'s arm: mover at level 60; F's spell paused for the act's PREPARE window in both arms, M's kept; a reported `completions` readout | the owner 2026-10-24 |
| D22 | group A: **P1** (reciprocal contact, general move physics) and **P3** (`TRIANGLE` to layer 0) | the owner 2026-10-24 |
| D23 | `M11.C.27[add_third]` (both) and the calm-state finding stay accepted findings unless the new rules move them; `M11.C.44` keeps `TRIANGLE` as an inside act | the owner 2026-10-24 |
| Q8 | `assertion_perspective_threshold` **retired**: every `I-POSITION` from a mover who is not angry starts a sequence (recommended) | the owner 2026-10-25, through `AskUserQuestion`; §18 |
| Q9 | `M11.C.49`'s act **starts at `DEFINE`**, unprepared, at both levels (recommended) | the owner 2026-10-25; §18 |
| Q10 | `reciprocal_contact_share` **0.5**, with §14's reasoning (recommended) | the owner 2026-10-25; §18 |
| Q11 | relief in `M4.C.1d` is read over the **whole delivery** (recommended) | the owner 2026-10-25; §18 |
| Q12 | `M1.C.1` **partial** in the coverage override, and §13.10's sentence (recommended) | the owner 2026-10-25; §18 |

## 13. The requirement text

"Replace … with …" quotes the revision 38 text exactly; each span was checked by script against the spec (§24).

### 13.1 `M5.F.4` — rewritten

Replace

> **M5.F.4** `I-POSITION` selected at low `systems_perspective` (M1.A.18) **MUST** execute as the **assertion form**: it **MUST** raise reactivity on the tie rather than lower it, and **MUST** count as negative evidence under M5.F.2a. This makes pseudo-differentiation an **output** of the model rather than a case the model must be told about separately.

with

> **M5.F.4** — *the lower the mover's level, the more an `I-POSITION` comes out as a demand.* A genuine
> `I-POSITION` — the `DEFINE` event of `M5.D`'s sequence — **MUST** be a mixture of a self-statement, share `w`, and
> a demand, share `1 − w`, where `w` is the mover's own self-channel weight that week (`M4.D.1a`: `w = f +
> perspective_channel_gain × perspective × (1 − f)`, `f = (functional_level ÷ 100)^self_channel_exponent`). The
> demand share **MUST** add `(1 − w) × assertion_gain` to the impingement the event delivers (`M4.C.1`), so it raises
> reactivity on the tie; the debit of the target's felt contact (`M5.E.7`) **MUST** be scaled by `w`; and the
> mover's outward axis **MUST** rise by `(1 − w) × assertion_evidence_gain`, negative evidence under `M5.F.2a`. The
> **assertion form** is the mixture at `w = 0` — the full `assertion_gain`, no debit, the full
> `assertion_evidence_gain` — and an angry mover's `I-POSITION` executes as it at once, with no sequence
> (`M5.D.4a`). No threshold on systems perspective decides the form: perspective enters through `w` alone, so a mover
> without it starts a sequence like any other (`M5.D.2a`) and, at `DEFINE`, says it mostly as a demand. At
> perspective 1, `w` is about 0.58 at level 40 and 0.68 at level 60; at perspective 0, 0.16 and 0.36. This keeps
> pseudo-differentiation an **output** of the model rather than a case the model must be told about separately. A
> low-level self can take an I-position, but a selfish, dogmatic and forceful one, and the manner shows the level
> (the KB12 interview, `K12.3`, `[T]`); less differentiated people tend to take reactive I-positions that polarize
> the system and intensify anxiety (`KS05.5`, `[K]` `[T]`); an effective I-position defines what the person will do,
> not what others should do (`KS06.11`, `[K]` `[T]`); many so-called I-positions are attempts to get others to change
> (`FE04.13`, `[K]` `[T]`). That the demand share falls as level rises is the direction those passages give (`[K]`
> `[D]`, with the KB12 interview); the mixture, `w` as its share, and the three terms it scales are the model's (`[M]`
> `[I]`). `L10.12` and `FE11.19` (`[B]`) support the dogmatic form, not a gradient by level. No new constant;
> `assertion_perspective_threshold` is retired from the register. → `M4.D.1a`, `M5.D.4a`, `M5.E.7`, `M5.F.2a`,
> `M5.F.2c`, `M11.C.49` ⟦rev39 · owner decision 2026-10-25⟧

**What this block adds to D20's wording, for the owner's review with the draft.** D20 says the act is "part
self-statement, part demand, weighted by the mover's own self-channel weight", and the plan says `M5.F.4`'s binary
perspective gate "becomes this continuous mixture". Three consequences are the draft's, each `[I]`: the assertion form
is kept as the mixture's `w = 0` end, so `M5.D.4a` and `M11.C.32` keep their meaning; `w` is read in the `DEFINE`
week, not the week the act was chosen; and, with the threshold gone, a mover without perspective now starts a
sequence where today the act executes at once as the assertion form (Q8). Since every person's perspective starts at
0 (`scenario/family.py`) and rises only through a landed coach contact or, slowly, through their own practice
(`M1.E.7`), that is most `I-POSITION` selections in most runs; step
8's targeted runs (`M11.C.49`, `.29`, `.32`, `.5`, `.18`, `.20`, `.33`) read the effect.

The event carries its demand share `1 − w` in place of today's assertion flag, and the flag's two readers read the
share (`[I]`, the draft's): `M11.C.32`'s count of assertion-form events (`ensemble/criteria.py`) counts share 1 only,
the angry form at `w = 0`, so its meaning holds; and `M5.D`'s test for an impinging delivery to a mover in a sequence
(`iposition.py`'s `_impinging_delivered`) counts a delivered `I-POSITION` whose demand share is above 0, since that
share delivers impingement (`M4.C.1`).

**Found, not fixed (recorded for the milestone review).** `M5.D.4a` says an `I-POSITION` emitted while the mover's
anger is above threshold executes as the assertion form; the code checks anger only when the act is selected
(`act.py`), not again at `DEFINE`. The draft's sentence ("executes as it at once, with no sequence") describes the
code; whether anger at `DEFINE` should also turn a said position into the assertion form is not drafted.

### 13.2 `M5.F.2c` — one sentence

Replace

> crossing zero at a declared point per axis, `[I]`, set at 0.5 for both. ⟦rev14 · owner decision 2026-10-10 · P3⟧

with

> crossing zero at a declared point per axis, `[I]`, set at 0.5 for both. ⟦rev14 · owner decision 2026-10-10 · P3⟧
> `I-POSITION`'s form follows the same rule: its demand share scales continuously with the mover's self-channel
> weight (`M5.F.4`), and no threshold on systems perspective decides it. The anger gate (`M5.D.4a`) is the one step
> left, and it decides whether a sequence runs, not how much of a said position is demand.

### 13.3 `M11.C.29` — the arm (D21)

Replace

> so the budget never moves. ⟦rev32 · Claude's recommendation 2026-10-17, approved by the owner 2026-10-17⟧

with

> so the budget never moves. ⟦rev32 · Claude's recommendation 2026-10-17, approved by the owner 2026-10-17⟧ *The arm
> (revision 39, the owner's D21 of 2026-10-24):* the mover F has basic and functional level 60 in both arms, as in
> `M11.C.32`; F's spell is suspended from `t0` to `t0 + prepare_ticks` in both arms, so the act's `PREPARE` window
> holds no spell event of F's, and M's spell is kept, so the third still carries load (T-020); a `completions` readout
> (`iposition:end:completed`) is reported, and the row fails if no genuine `I-POSITION` completes, since the budget
> limb depends on a completed exchange — a condition of this row, stricter than `M11.1f`, which fails a criterion
> only when its move never occurs. F's systems perspective is 1.0 in both arms, as today. *Post hoc*, like `M11.C.32`'s level: chosen after the
> diagnosis that at level 40, with the spell every 4 weeks from week 4, 58 of 100 seeds failed `PREPARE` (excess
> 20.7 ± 0.85 against `defence_threshold` 15) and 37 ran as the assertion form because the mover was angry
> (`docs/version_1_dispositions.md`; `docs/implementation_plan_phase_d.md`, the `M11.C.29` diagnosis).

### 13.4 `M11.C.49` — new row, after `M11.C.48`

> | **M11.C.49** | **The lower the mover's level, the more an I-position raises the family's anxiety.** The owner (2026-10-24): the lower the level, the more an I-position comes out as conflictual demand and the more anxiety it creates. Four runs on identical seeds, on the Phase C triad under the declared spell (`M11.3`): the mover F at basic and functional level 35 and at 60, each with a genuine `I-POSITION` toward M said at `t0` — the scripted sequence starts at `DEFINE`, unprepared (`M5.D.2a`), so the act is said in every seed at both levels — and each with a control on the same seed in which nothing is scripted. F's spell is suspended over the act's window in every run, as in `M11.C.29`, and F's systems perspective is 1.0 in every run, as in `M11.C.29`, `M11.C.32` and the probe, so the demand share is about 0.44 at level 35 and 0.32 at 60 (`M5.F.4`). The act's effect at a level is its run less its control. The act's effect on the family's summed excess anxiety over the six weeks after `t0`, and on the `CONFLICT` share of the moves on the F–M tie over those weeks, **MUST** each be larger at level 35 than at 60 (composite limbs). The target's felt impingement from the act is reported, higher at 35 (premise limb: `M5.F.4`'s demand share adds to the impingement `assault_gain` on the outward axis already gives, `M4.C.1`); completions are reported. **Confound, declared:** level also moves the contact band (`M4.C.1a`), the hold capacity (`M5.D.3`) and the outward axis (`M1.A.9`), so the two levels differ in more than the mixture; the demand-free mutant isolates `M5.F.4`'s part. → `KS05.5`, `KS06.11`, `FE04.13` (`[K]`); the KB12 interview (`K12.3`) | `test_m11c49_low_level_iposition_raises_family_anxiety` | C | `M5.F.4`'s demand share — removing it (`demand-share-removed`, `w` := 1) **MUST** turn the composite limbs red, and inverting it (`demand-share-inverted`, `w` := 1 − `w`) **MUST** reverse them; both in the Phase C mutation record |

**What this row adds to the plan.** The plan's S3.7 has two arms; the probe found that the no-act controls at the two
levels differ by as much as the acting arms (`docs/review/stage_3_iposition_probe.md`), so the draft compares the act's
effect at each level, each against its own control. The values (`t0`, the weeks, the two levels' names in
`config/bowen/criteria.md`) are set at step 8 before the row first runs, and declared `[I]`. Starting a scripted sequence at
`DEFINE` (Q9) needs a script option the criterion runner does not have today; step 8 builds it, and whether it
touches `src/bowen/` is settled there.

### 13.5 `M4.C.1d` — new, after `M4.C.1c`

> **M4.C.1d** — *contact that relieves its receiver returns in part to its sender.* When a move whose contact
> component is positive (`M4.C.1`; `config/bowen/event_kinds.md`) is delivered to its target and relieves the target
> — the target's two-sided deviation on the tie (`M4.C.1`) is lower after the delivery than before, contact and
> impingement together — a share `reciprocal_contact_share` of the contact the target took in (`M4.C.1`'s Δ contact:
> the move's contact component after the tie's conductance, route and fidelity, the target's `M4.C.2` gain and
> `M5.F.1`'s discount) **MUST** be added to the sender's felt contact on the same tie in the same step, and the sender
> **MUST** appraise it as any delivered contact is appraised: by the change in their own deviation, at their own
> steepness (`M4.C.1a`), through `appraisal_gain`, reattribution (`M4.C.6`), attention and their own `M4.C.2` gain,
> with no second scaling by conductance, which the target's share already carries. Each party's `M4.C.2` gain applies
> to what that party takes in, so it is applied once at each end. A target whose deviation does not fall returns
> nothing. Relief is judged for each delivery against the target's state before the tick's batch (`M1.F.8`), as every
> appraisal is, and the returns from several deliveries in one tick add. The rule names no kind: it applies to every
> such move, sender and target, an external agent included (`SPLIT`, `FRAME_AMBIGUITY`, `DISPLACE`); the returned
> contact is felt contact, not `investment` (`M1.B.8`), so `M1.E.7e`'s thin coach tie is unaffected. A sender whom anxiety has moved toward closeness (`M4.C.1b`) and who stands on
> the "too little" side feels the returned contact as relief; one near their optimum feels little, and one past it a
> cost. The learner (`M4.D.6`) already credits the sender's own change in acute anxiety, so the returned contact
> reaches the value of the act that drew it without a change to the learner. The insiders of a triangle are calmed by
> their alliance (`KS03.1`, `[K]` `[T]`), and closeness is wanted as well as aversive (`KS03.2`, `[K]` `[T]`); that
> the calm reaches the one who sought the contact is the direction those passages give (`[K]` `[D]`). The relief
> condition, the share, the timing and reading it through the sender's own gain are `[I]`; the share's value is in
> `config/bowen/constants.md` (`M10.C.1a`). Nothing sets a `TRIANGLE`'s relief or compares it with `DISTANCE`'s.
> → `M4.C.1`, `M4.C.1b`, `M4.C.2`, `M4.D.6`, `M1.C.1`, `M11.C.42`, `M11.C.45` ⟦rev39 · owner decision 2026-10-25⟧

**What this block adds to D22's wording.** D22 approved P1 as "reciprocal contact, general physics". Three forms are
the draft's, each `[I]`: what counts as relief (Q11); the returned contact passing through the sender's own `M4.C.2`
gain, so a sender above `defence_threshold` takes in none of it, as they take in no relief from any approach; and
"approach act" read as any move whose configured contact component is positive (today `PURSUE`, `CONFLICT`,
`OVERFUNCTION`, `UNDERFUNCTION`, `TRIANGLE`, `I-POSITION`, `STAY-IN-CONTACT`, `DETRIANGLE`, `PREVENT_ALIGNMENT`,
`REDUCE_CUTOFF`, `PROVOKE`, and the three family→external moves `SPLIT`, `FRAME_AMBIGUITY` and `DISPLACE`), so no
list is kept.

### 13.6 `M4.D.3a` — one sentence (P3)

Replace

> The ordering is a capacity limit and a gate; the policy carries no term that reads anxiety and raises the weight of older acts. ⟦rev11 · learning-qa finding 2, 2026-10-06⟧

with

> The ordering is a capacity limit and a gate; the policy carries no term that reads anxiety and raises the weight of older acts. ⟦rev11 · learning-qa finding 2, 2026-10-06⟧ `TRIANGLE` **MUST** sit in the oldest layer, with `DISTANCE` and `CUTOFF` (`config/bowen/event_kinds.md`), so falling functional level does not make a triangle less available: triangles may have roots older than language (`KS15.10`, `[K]` `[T]`), and three people together invariably gravitate to two insiders and an outsider, the well differentiated included (`FE06.20`, `[K]` `[T]`). The source orders four classes and does not place triangling among them (`KS06.1`, `KS02.3`), so the placement is `[I]` (the owner's D22, 2026-10-24); it removes a gate rather than adding a preference.

### 13.7 `M10.C.1a` — one item

After "- the trailing window and drift threshold of `M17.F.2`'s settling condition ⟦rev10 · DL §7.10(e) · SEAA⟧"
insert:

> - the share of contact a relieved receiver returns to its sender, `reciprocal_contact_share` (`M4.C.1d`), set
>   before the acceptance suite runs (`M10.B.4`)

### 13.8 `M11.C.42` — the mutation column

Replace

> `M4.D.6`'s per-triangle learned value — disabling reinforcement in both arms **MUST** turn this red |

with

> `M4.D.6`'s per-triangle learned value — disabling reinforcement in both arms **MUST** turn this red; so **MUST** removing `M4.C.1d`'s returned contact (`reciprocal-contact-removed`, in the Phase C mutation record), if it passes |

### 13.9 `M11.5` — one new row and two notes

After `M11.C.48`'s row:

| Criterion | Class | The single rule it rests on | Note |
|---|---|---|---|
| M11.C.49 | Mixed | `M5.F.4`'s demand share (the target's felt impingement limb, reported, never a finding) | the family-excess and `CONFLICT`-share limbs are composite: they need `M5.F.4`, appraisal, the target's learned response and contagion; level moves more than the mixture, and `demand-share-removed` isolates `M5.F.4`'s part |

`M11.C.42`'s note: replace "it fails at the central setting (reuse −0.17).*" with "it fails at the central setting
(reuse −0.17).* Revision 39: `M4.C.1d` returns contact to a relieved seeker and `TRIANGLE` is in layer 0
(`M4.D.3a`); reuse is read under both, and a pass needs `reciprocal-contact-removed` to turn it red." `M11.C.45`'s
note: replace "so the reclassification is withdrawn until it passes again.*" with "so the reclassification is
withdrawn until it passes again.* Revision 39: read under `M4.C.1d` and `TRIANGLE` in layer 0; quiet-when-calm needs
`M4.C.1d`'s two-sidedness, the spell and learning, so it stays composite."

### 13.10 `M1.C.1` — one sentence (plan §3.13; Q12)

Replace

> The partner still defines the triad, and their anxiety eases only through their own act (for instance a third of their own). ⟦owner decision 2026-10-09⟧

with

> The partner still defines the triad, and their anxiety eases only through their own act (for instance a third of their own). ⟦owner decision 2026-10-09⟧ `bound_anxiety` is not written: no act binds anxiety into a `Triangle`, so the release paths that read it (`event_effects.py`, `life_course.py`) release nothing, and `M1.C.1` is *partial* in the coverage record for that reason. The owner chose not to build it in version 2's stage 3 (D22).

## 14. Config (applied with each group's code)

- **`config/bowen/constants.md`** (step 8): the `assertion_perspective_threshold` row is removed (Q8). (Step 9): a row
  `| reciprocal_contact_share | 0.5 | [I] | share of the contact a relieved receiver took in | M4.C.1d |` beside
  `calm_transfer_rate` (Q10).
- **`config/bowen/constants_changes.md`**: two entries. `assertion_perspective_threshold` retired at revision 39
  (`M5.F.4`: perspective enters the form through the self-channel weight, D20). `reciprocal_contact_share` new at
  revision 39, 0.5, `[I]`, with its reasoning: no source gives a share; 0 is revision 38's model (nothing returns) and 1
  would return to the sender all the receiver took in, which no source states either; 0.5, the middle of that
  interval, is set before any criterion runs under it (`M10.B.4`), and its half-to-double sweep range, 0.25 to 1.0
  (capped at the bound), reaches the bound, so the stage-end sweep reports each direction over that whole span. Not
  tuned to a criterion. The value and this reasoning approved by the owner 2026-10-25 (Q10).
- **`config/bowen/event_kinds.md`** (step 9): `TRIANGLE`'s `layer` 2 → 0; the preamble's "`PURSUE` and `TRIANGLE`
  are placed with conflict, `[I]`: the source orders four classes and names neither (`KS06.1`, `KS02.3`)." becomes
  "`PURSUE` is placed with conflict, `[I]`, and `TRIANGLE` with distance and cutoff, layer 0, `[I]` (spec revision 39,
  `M4.D.3a`, the owner's D22): the source orders four classes and names neither (`KS06.1`, `KS02.3`); triangles may
  have roots older than language (`KS15.10`) and form among the well differentiated too (`FE06.20`)."
- **`config/bowen/criteria.md`** (step 8): `M11.C.29`'s `mover_level` 60 and the spell's pause; `M11.C.49`'s rows
  (`weeks`, `t0`, the two levels, the six-week window), each `[I]`.
- **`config/sweep/sweep.md`** (step 10): `reciprocal_contact_share` in the set of each criterion whose row or class row
  credits `M4.C.1d` (D10's rule), 0.25–1.0, `log`, bound 1; `M11.C.49` gets a set. (No row names
  `assertion_perspective_threshold`; it is in no criterion's set.)
- **Other places that name `assertion_perspective_threshold`** (step 8, with the code): `src/bowen/scenario/constants.py`,
  `src/bowen/engine/params.py` (two places) and `tests/bowen/test_config.py`'s `GOOD_ROWS` fixture row lose it;
  `tools/spec_coverage_overrides.json`'s `M1.A.18` note, which names the threshold as "the hard gate the paragraph
  rules out", is rewritten (the gate is gone; whether `M1.A.18` becomes *done* is read at step 8 against its tests).
  `config/bowen/constants_frozen.md` keeps its row: it is a snapshot, where a retired key is allowed.
- **`docs/readouts_phase_c.md`** (step 8): `M11.C.49`'s readouts and `M11.C.29`'s `completions`.

## 15. `docs/model_explainer.md` (applied with the code)

- **§19.4 (acts and patterns), under "The `I-POSITION` sequence, as built"**: the bullet "**The assertion form.** A person without systems
  perspective, or angry on the tie, does not run a sequence. …" becomes: "**The demand share (revision 39,
  `M5.F.4`).** What a person says at `DEFINE` is part statement of self and part demand, the demand share falling as
  their level and perspective rise; the demand adds impingement, takes less of the other's contact, and counts against
  the speaker. `[K]` `KS05.5`, `KS06.11`, `FE04.13` and the interview `M5.F.4` cites, for the direction; the form is `[I]`.
  **The assertion form** is the all-demand end: an angry person does not run a sequence, and their `I-POSITION` is one
  forceful claim."
- **The same bullet's interview citation** is written as `M5.F.4` writes it (the KB12 interview, `K12.3`), cited and not
  quoted (SP-1).
- **§5.6 (counterfeit moves)**: one sentence: "At a lower level an I-position comes out more as a demand (`M5.F.4`,
  revision 39); the assertion form is the all-demand end."
- **§11.3**: `M11.C.29`'s arm at revision 39 (level 60, F's spell paused over the window, completions reported) and
  `M11.C.49`, the new criterion, with its classes.
- **§19.2 (the drives and the appraisal)**: a fifth bullet, "**Contact that relieves returns in part to its sender**
  `[I]` (`M4.C.1d`, revision 39) — the insiders of a triangle are calmed by their alliance (`KS03.1`, `[K]`); a
  receiver relieved by an approach returns `reciprocal_contact_share` of the contact to the sender, who feels it by
  their own deviation, so seeking out another relieves only when closeness is wanted."
- **§19.4 (acts and patterns)**, after the `TRIANGLE` bullet: "*Revision 39:* `TRIANGLE` is in the oldest layer, with
  distance and cutoff, so falling level does not make it less available (`M4.D.3a`, `KS15.10`, `FE06.20`, `[K]`;
  the placement `[I]`); a third who is relieved by being approached returns contact to the seeker (`M4.C.1d`)."
- **§5.2, "The ordering within the automatic channel"**: after "**distance**, the oldest of all.": "`TRIANGLE` is placed
  with distance (revision 39, `[I]`)."

## 16. The §19 check (`model_explainer.md` §19)

`M5.F.4` writes the act's content — how much of it lands as pressure — which is the set layer, the same class as
`assertion_gain` and `assault_gain` today; it writes no response. Whether the target answers with `CONFLICT`,
`PURSUE` or `DISTANCE` comes from the target's learned values, and whether the family's anxiety rises comes from
appraisal: "the target feels more pressure from a lower-level mover" is a premise, and "the family's anxiety and
conflict rise" are findings (`M11.C.49`'s class row keeps them apart). `M4.C.1d` returns ordinary contact, appraised by
the sender's own two-sided deviation, so it relieves only when the sender's state makes closeness wanted; whether
`TRIANGLE` is reused (`M11.C.42`) and whether triangles are quieter when calm (`M11.C.45`) are learned from felt
relief. A rule that raised `triangle_transfer_rate` or added a bonus to a `TRIANGLE`'s learned value would write the
outcome; neither is drafted. `M4.D.3a`'s sentence removes an invented gate. Whether these are the set layer is the
owner's judgment (plan §3.12).

## 17. The stage 4 check, against `_LEDGER.md`

Read 2026-10-25: `K12.3`, `KS05.5`, `KS06.11`, `FE04.13`, `L10.12`, `FE11.19` against `M5.D`, `M5.F.2c`, `M5.F.4`;
`L10.3`, `L17.1`, `KS03.1`, `KS03.2`, `FE06.2`, `FE06.20`, `KS15.10`, `KS06.1`, `KS02.3` against `M1.C.1`, `M4.C.1`,
`M4.C.1d`, `M4.D.3a` and `TRIANGLE`'s row in `M5.A.1`'s config.

| Entry | What it says | Against the draft | Found |
|---|---|---|---|
| the KB12 interview, `K12.3` | a low-level self can hold an I-position, but a selfish, dogmatic, forceful one; the manner shows the level | `M5.F.4`'s demand share falls as level rises | consistent; cited as the interview, not as Bowen |
| `KS05.5` `[K]` | less differentiated people are vulnerable to reactive I-positions that polarize the system and intensify anxiety; the genuine one is not taken in anger or to oppose | the demand share; the anger gate (`M5.D.4a`) | consistent |
| `KS06.11` `[K]` | an effective I-position states what the person will do, not what others should do | the demand share is the part that tells the other what to do | consistent. The entry's model note reads a move addressed to another's behaviour as not an I-position at all; the mixture keeps one act with two shares, which `K12.3` (a low-level self *can* hold one) supports; recorded, no edit |
| `FE04.13` `[K]` | many so-called I-positions are attempts to get others to change **or to pry oneself loose** from intense situations | `M5.F.4` models the first; the second, distance in disguise, is `M11.C.29`'s other arm, not a share of the act | **found, not fixed**: the mixture has no distancing share. Recorded for the stage 3 milestone review; no rule drafted |
| `L10.12` `[B]` | the dogmatic, compliant or rebellious band; the non-monotone corollary was withdrawn | `w` is monotone in level | consistent: a monotone curve fits the text equally well |
| `FE11.19` `[B]` | a dogmatic person is rarely sure of self | outward axis rises with the demand share (`M5.F.2a`) | consistent |
| `KS03.1` `[K]` | the insiders are calmed by the alliance; the outsider absorbs | `M4.C.1d` returns contact to a relieved seeker; `M1.C.1`'s transfer unchanged | consistent; `M1.C.1`'s non-conservation note is outside this part |
| `KS03.2` `[K]` | closeness is wanted and aversive; both threats raise anxiety | `M4.C.1d`'s returned contact is appraised two-sidedly | consistent |
| `L10.3` `[B]` | enlarging the system lowers tension | seeking a third can relieve the seeker | consistent in direction |
| `L17.1` `[B]` | triangulation at a threshold; triangles inoperative when calm | `M4.C.1d` gives a calm seeker near the optimum little relief | consistent; whether triangles are quiet when calm stays read (`M11.C.45`) |
| `FE06.2` `[K]` | at moderate anxiety the outsider is recruited; at high stress each seeks the outside position | `M4.C.1d` with `M4.C.1b`; above `defence_threshold` the returned contact is not taken in | consistent at moderate load; the high regime is `M11.C.44`'s and is not reached by P1 (D23) |
| `FE06.20` `[K]` | three people invariably gravitate to two insiders and an outsider; the well differentiated triangle, low-key | `TRIANGLE` in layer 0: available at every level; `M1.C.3a`'s routing capacity still falls with level | consistent |
| `KS15.10` `[K]` | triangles may have roots older than language; the animal work supports the substrate only | `TRIANGLE` in the oldest layer | supports "old"; it does not place triangling with the single-celled distancing reflex, so layer 0 is `[I]` |
| `KS06.1`, `KS02.3` `[K]` | four classes ordered; distance the oldest | `TRIANGLE` beside distance | the source names neither `PURSUE` nor `TRIANGLE`; placement `[I]`, stated in `M4.D.3a` |
| — | `reciprocal_contact_share`, the relief condition, `w` as the share | no corpus entry states a magnitude | all `[I]`; nothing to cite |

## 18. Open questions for the owner (part 2)

Put through `AskUserQuestion`, one at a time, recommendation first; all answered 2026-10-25 with the recommended
option. (Q7 is the version 2 plan's stage 4 check, so this part's questions start at Q8.) Q13 and Q14 came from the
read-only correctness pass over parts 2 and 3 (below); Q14 is part 3's.

- **Q8, the perspective threshold.** (a) Retire it: every `I-POSITION` from a mover who is not angry starts a
  sequence and says its mixture at `DEFINE`; `assertion_perspective_threshold` leaves the register (recommended: the
  plan's "the binary perspective gate becomes this continuous mixture", and `M5.F.2c` forbids the step). (b) Keep it as
  the gate on starting a sequence: below it the act is said at once as the mixture at the mover's `w`. (c) Keep today's
  gate: below it the assertion form at once; the mixture only for acts that reach `DEFINE`. **Decided by the owner 2026-10-25, through `AskUserQuestion`: (a), retire it**
  (recommended).
- **Q9, how `M11.C.49`'s act is said at both levels.** At level 35, 132 of 176 sequences failed `PREPARE` in the probe.
  (a) The arm starts the scripted sequence at `DEFINE`, unprepared (`M5.D.2a`), at both levels, so the criterion reads
  what the act does when said (recommended). (b) Read only the seeds in which both levels' act reached `DEFINE`,
  counting the rest apart (a selection: the seeds calm enough). (c) Read every seed as it falls, the act's failures
  included. **Decided by the owner 2026-10-25, through `AskUserQuestion`: (a), start at `DEFINE`** (recommended).
- **Q10, `reciprocal_contact_share`'s value.** 0.5 with the reasoning in §14 (recommended); 1.0 (the sender gets back
  all the receiver took in); 0.25. **Decided by the owner 2026-10-25, through `AskUserQuestion`: 0.5** (recommended).
- **Q11, what counts as relief in `M4.C.1d`.** (a) The receiver's two-sided deviation falls over the whole delivery,
  contact and impingement together, so an intrusive approach that loads the receiver returns nothing (recommended:
  the plan's "its two-sided deviation falls"). (b) The contact component's effect alone. **Decided by the owner 2026-10-25, through `AskUserQuestion`: (a), the whole
  delivery** (recommended).
- **Q12, `M1.C.1`'s `bound_anxiety` (plan §3.13).** (a) `M1.C.1` marked *partial* in the coverage override with the
  reason, and §13.10's sentence added to the spec (recommended: the documentation rule; the plan proposed the
  override at step 4). (b) The override only. (c) Leave it as *done*. **Decided by the owner 2026-10-25, through `AskUserQuestion`: (a), partial and the
  sentence** (recommended).
- **Q13, `M11.C.49`'s mover perspective.** Under Q8 perspective enters `w`, so the row must state it. (a) 1.0 in all
  four runs, as in `M11.C.29`, `M11.C.32` and the probe: demand share 0.44 at level 35 and 0.32 at 60 (recommended:
  the measured numbers carry over; the smaller gap is the harder test). (b) 0, every person's starting value: 0.88
  and 0.64. **Decided by the owner 2026-10-25, through `AskUserQuestion`: (a), 1.0** (recommended).
- **Q14, the ratio's scope (part 3, §20).** The slow tick fires at the end of week 51, so in a 52-week run tie wear
  changes nothing after it. (a) The ratio enters a criterion only if its run is longer than `slow_tick_fast_ticks` fast
  ticks; the five 52-week criteria (`M11.C.14`, `.23`, `.25`, `.26`, `.30`) leave the set and the record lists them
  with the reason (recommended). (b) D26 as written, at least one slow tick. **Decided by the owner 2026-10-25,
  through `AskUserQuestion`: (a), longer than one slow tick** (recommended); it refines D26's wording.

---

# Part 3 — T-128 and T-014 (drafted 2026-10-25)

Written from plan §3.4 (T-128, D24–D26) and §3.5 (T-014, D27–D30). Wording only: no rule changes what a run computes,
except that D27 removes an uncalled helper (`avoidance_available`) and its test, which the batch's first commit holds.

## 19. What this part changes

| Requirement | Change | Lands |
|---|---|---|
| `M17.E.1` | names the ratio `tie_wear_to_transfer_ratio`, its range, its scope and how the record splits by it (D24–D26) | step 10, with the sweep tool |
| `M10.C.1a` | no change: its item "the sweep ranges of `M17.E.1`" covers the ratio (§20.2) | — |
| `config/sweep/sweep.md` | a preamble paragraph and a table of named ratios | step 10 |
| `M8` (preamble), `M8.1`, `M8.5`, `M13.1` | three places, not four; the avoidance gate removed (D27); fusion not built (D28) | step 6, with D27's removal |
| `M11.C.7`, `M11.5`'s row | the coach's going along with a `SPLIT` is a declared premise of the sided arm (D29) | step 6 |
| `M7.D.1a` | one sentence: the illness sizes on all three channels, and why (D30) | step 6 |
| `M11.C.21` | the quoted effect reconciled with the record (plan §3.13) | step 7, with group C |
| `docs/version_2_plan.md` | "Out of version 2" gains the avoidance gate and fusion | step 6 |

## 20. T-128 — the named ratio

### 20.1 `M17.E.1` — the ratio's sentence

Replace

> The rate at which ties harden or attenuate, relative to the rate at which anxiety moves, **MUST** be swept as a named ratio.

with

> The rate at which ties harden or attenuate, relative to the rate at which anxiety moves, **MUST** be swept as a
> named ratio: `tie_wear_to_transfer_ratio` = `tie_wear_rate` ÷ (`slow_tick_fast_ticks` × `calm_transfer_rate`), the
> conductance a tie loses per week (`M7.E.4`) against the share of the anxiety gap moved from a calmer sender's
> target to the sender per contact, times conductance (`M4.C.10`), with `slow_tick_fast_ticks` read from
> `config/bowen/constants.md`, never written as a literal. The units differ (per week against per contact), and the
> record says so beside every fraction. Its range, log-uniform, is declared in `config/sweep/sweep.md`'s table of
> named ratios (`M10.C.1a`). Each sample draws the ratio and `calm_transfer_rate`
> from their ranges and derives `tie_wear_rate` from the two, so each sample's ratio is exact; a ratio whose derived
> numerator could leave that constant's admissible bound fails at load. The ratio, with both rates, enters the set of
> every directional criterion whose run is longer than `slow_tick_fast_ticks` fast ticks, so that a slow tick fires
> before its last tick, since tie wear acts only at a slow tick, and of no other; the record lists the criteria it entered and those it did not, with the reason.
> This is an exception, made by this sentence, to the sample rule's set (a constant that acts but that no row credits
> is left out, T-126). For each such criterion the record reports *n of S* and the counts below and above the ratio's
> central value, so a direction that flips across it shows. The ratio, its range and its scope are `[I]` (the
> owner's D24–D26, 2026-10-24).

### 20.2 `M10.C.1a` — no change

`M10.C.1a` already lists "the sweep ranges of `M17.E.1`", which covers the ratio's range, and says its items are given
no value in the spec; so the ratio's central value (0.05 ÷ (52 × 0.05) = 1/52 ≈ 0.0192 today) and its range (a
quarter to four times the central value, 1/208 to 4/52) are stated in `config/sweep/sweep.md` only (§20.3).

### 20.3 `config/sweep/sweep.md` — the preamble and a table

After the paragraph ending "These are judgments; the plan's §2.10 lists them." insert:

> **Named ratios** (spec revision 39, `M17.E.1`; the owner's D24–D26 of 2026-10-24). `M17.E.1` names one ratio that is
> swept as an exception to the rule above: `tie_wear_to_transfer_ratio` = `tie_wear_rate` ÷ (`slow_tick_fast_ticks` ×
> `calm_transfer_rate`). Its central value follows from the constants, 0.05 ÷ (52 × 0.05) = 1/52 ≈ 0.0192; its range
> is a quarter to four times that, 1/208 to 4/52. Each sample draws the ratio log-uniformly over its range and `calm_transfer_rate` from its own
> (0.025–0.1), and derives `tie_wear_rate` from the two, so each sample's ratio is exact; the period constant is read
> from `config/bowen/constants.md`. The ratio and both rates enter the set of every directional criterion whose run is
> longer than `slow_tick_fast_ticks` fast ticks, so that a slow tick fires before its last tick, and of no other (a
> 52-week run, today `M11.C.14`, `.23`, `.25`, `.26` and `.30`, is not entered), and the tool lists which criteria it entered and
> which it did not, with the reason. The sample count stays at `sweep_samples`, so the ratio adds no runs. **The
> derived `tie_wear_rate` spans 0.00625 to 0.4, one eighth to eight times its central value,** wider than the
> half-to-double rule every other row follows; it stays inside its admissible bound of 1, and the parser rejects a
> ratio whose derived numerator could leave it. Every value is `[I]`.

and, after the *Ranges* table, a third table:

> ## Named ratios
>
> | ratio | numerator | denominator | period constant | low | high | scale | grade | spec |
> |---|---|---|---|---|---|---|---|---|
> | `tie_wear_to_transfer_ratio` | `tie_wear_rate` | `calm_transfer_rate` | `slow_tick_fast_ticks` | 1/208 | 4/52 | log | [I] | `M17.E.1`, D24–D26 |

(The two ends are written as the step 10 parser reads them, as fractions or as decimals 0.0048077 and 0.0769231.)

## 21. T-014 — the four questions

### 21.1 `M8` — the preamble (D27)

Replace

> This is one rule, written for four places. It **MUST** be implemented once and called from each place that uses it. Three mechanisms call it — the witness test (`M8.6`, `visibility.py`), the coach's position (`M11.C.7`, `external.py`) and triangle activity (`M1.C.3`, `recompute.py`). The fourth site, `M5.C`'s avoidance gate, is `avoidance_available`: defined and tested, but not called. Whether to build that gate is a stage 3 question ⟦rev34 · owner decision 2026-10-21 · D4⟧ → `_RESOLUTIONS.md`

with

> This is one rule, written for three places. It **MUST** be implemented once and called from each place that uses it: the witness test (`M8.6`, `visibility.py`), the coach's position (`M11.C.7`, `external.py`) and triangle activity (`M1.C.3`, `recompute.py`). A fourth, an avoidance gate for `M5.C` — a group of fewer than three live positions can avoid its issue — is not built: its uncalled helper, `avoidance_available`, was removed at revision 39, and the gate is a version 3 candidate (the owner, 2026-10-24, D27). → `_RESOLUTIONS.md`

### 21.2 `M8.1` — the code block and one sentence (D27, D28)

Replace

> **M8.1** `positions_live(group)` **MUST** count only occupants **not fused into another member of the group**.

with

> **M8.1** `positions_live(group)` **MUST** count only occupants **not fused into another member of the group**. The fused-occupant branch is not reached in a run: the engine holds no fusion state and no caller passes a fusion map (`M8.5`), so `M8.1` is *partial* in the coverage record for that reason; building it is a version 3 candidate (the owner, 2026-10-24, D28).

and delete the code block's second line:

> avoidance_available(g) = positions_live(g) < 3        # a step, not a gradient

and, in the mechanism table, replace

> | Live positions (`M8.1`) | engine | called from step 6 and from visibility | on call | group membership; fusion |

with

> | Live positions (`M8.1`) | engine | called from step 6 and from visibility | on call | group membership |

### 21.3 `M8.5` (D27, D28)

Replace

> whether a group can avoid its issue (M5.C; *revision 34:* not called yet, see the preamble), whether a third party

with

> whether a third party

and replace

> whether a fused occupant should make its triangle inactive is a theory question left for stage 3.

with

> whether a fused occupant should make its triangle inactive is not built (`M8.1`; a version 3 candidate, the owner 2026-10-24, D28).

### 21.4 `M13.1` (D27)

Replace

> M8 **MUST** land in Phase B, before the policy, because four separate Phase C mechanisms call it. Three call it; `M5.C`'s site is not built (`M8`'s preamble).

with

> M8 **MUST** land in Phase B, before the policy, because three separate Phase C mechanisms call it (`M8`'s preamble; an avoidance gate for `M5.C`, a fourth, is not built and is a version 3 candidate, D27).

### 21.5 `M11.C.7` and its `M11.5` row (D29)

Replace

> The sided arm has Marta `SPLIT` toward Dr Halim the week before each session (`M8.3`);

with

> The sided arm has Marta `SPLIT` toward Dr Halim the week before each session (`M8.3`), and the coach's taking her side then — he has no move that declines — is a declared premise of the arm, not a result: a coach who could decline would make the coach's skill a variable this criterion holds fixed (the owner, 2026-10-24, D29);

`M11.5`'s row for `M11.C.7`: replace "the deviation ordering is composite: no rule writes it" with "the deviation
ordering is composite: no rule writes it; the coach's going along with a `SPLIT` is a declared premise of the sided arm
(D29)".

### 21.6 `M7.D.1a` (D30)

Replace

> so a family's own symptoms do not count as stress from outside. → `M1.F.6a`, `M7.D.2a`

with

> so a family's own symptoms do not count as stress from outside. Every onset takes these two sizes whichever of the three channels the symptom is in (`M7.D.1`): the channels are substitutable — the same deficit presents interchangeably as physical illness, emotional illness or social dysfunction (`Ch08` · L08.3, `[B]` `[T]`, a typology, not an observed rerouting; the paper's emotional illness is the model's mental channel, `M1.A.11`) — so the channel a symptom lands in does not change the family's load by a rule. The sizes stay `[I]` (the owner, 2026-10-24, D30). → `M1.F.6a`, `M7.D.2a`

### 21.7 Outside the spec (step 6)

- **`docs/version_2_plan.md`**, "Out of version 2 (approved 2026-10-19)": two bullets — "`M5.C`'s avoidance gate
  (`M8`'s fourth place): not built; its uncalled helper removed at revision 39 (the owner, 2026-10-24, D27); a version
  3 candidate." and "Fusion in `positions_live` (`M8.1`): the engine holds no fusion state; *partial*, a version 3
  candidate (D28)."
- **`tools/spec_coverage_overrides.json`**: `M8.1` *partial*, its note the reason in §21.2 (D28). `M8.5`'s note,
  whose reason for *partial* is "the `M5.C` site is missing: `avoidance_available` has no caller", is rewritten: the
  site is not built (D27) and the helper is gone; whether `M8.5` becomes *done* is read at step 6 against its tests.
- **`docs/theory/_RESOLUTIONS.md`**, the code block that `M8`'s preamble points to: a note under it that
  `avoidance_available` was removed at revision 39 and the gate is a version 3 candidate (D27); the block stays as the
  resolution's record.
- **`docs/model_explainer.md`** §13.1: the code block loses `avoidance_available`; the closing sentence
  "`avoidance_available` is defined and tested, but no mechanism calls it." becomes "An avoidance gate is not built; its
  uncalled helper was removed at revision 39 (D27), and no family member is fused in a run (D28)." Under "An onset is
  also an illness": "on whichever channel, since the channels are substitutable (`L08.3`)".
- **T-011's** `live_positions.py:1` note is closed with D27's removal (plan §3.5).

## 22. An adjacent edit from plan §3.13 — `M11.C.21`'s quoted effect (step 7, with group C)

The spec quotes "+0.15 ± 0.19 (p 0.09)" as the regenerated result. That number is the 50-seed run in
`docs/implementation_plan_phase_d.md` (under the owner's second decisions, 2026-10-16, before the Phase D part I
record); "+0.24 ± 1.36" without the capacity rule is in the spec only, and the draft keeps it as the spec's own. The
Phase D part I record gives −0.0906 ± 0.1 at the tag (`docs/version_1_status.md`) and −0.0682 ± 0.089 at revision 38
(`docs/version_2_status.md`). Replace

> Regenerated: **fails**, +0.15 ± 0.19 (p 0.09); without the capacity rule, +0.24 ± 1.36.

with

> Measured on 50 seeds when the arm was built: **fails**, +0.15 ± 0.19 (p 0.09) (`docs/implementation_plan_phase_d.md`); without the capacity rule, +0.24 ± 1.36. The Phase D part I record gives −0.0906 ± 0.1 at the tag `version-1` (`docs/version_1_status.md`) and −0.0682 ± 0.089 at revision 38 (`docs/version_2_status.md`); revision 39's regeneration, with `M6.I.4a`, replaces them.

## 23. The stage 4 check for part 3

| Entry | Against | Found |
|---|---|---|
| `L08.3` `[B]` | `M7.D.1a`'s new sentence | the "interchangeably" passage is a typology, not an observed rerouting (the ledger's correction); the sentence says so |
| `L19.3` `[B]` | `M11.C.7`'s premise | the entry describes the move, consulting the professional without telling the others; that it takes the professional's side is `M8.3`'s reading, not the entry's; whether a professional declines is not stated, so the premise is declared, `[I]` |
| — | `M17.E.1`'s ratio, its range | Castellano et al. (DL §2.4) is a method source on coevolving networks, not the corpus; nothing in `_LEDGER.md` bears on tie wear against transfer; `[I]` |
| — | `M8`, `M8.1`, `M8.5`, `M13.1` | the "three live positions" rule (§13.1 of the explainer, `L10.4`) is unchanged; only the uncalled gate goes |

## 24. Revision 39 as a whole

**Where each block lands** (the owner's Q6): step 6, the batch's first commit — T-130's wording (§6), part 3's T-014
blocks (§21); step 7 — group C (§3–§5) and `M11.C.21`'s quoted effect (§22); step 8 — group F (§13.1–§13.4, §13.9's
`M11.C.49` row, §14, §15's group F bullets); step 9 — group A (§13.5–§13.8, §13.9's notes, §13.10, §14, §15's group A
bullets); step 10 — T-128 (§20). Each application regenerates `docs/spec_index.jsonl` and `docs/spec_rules/` and adds
to `docs/spec_revision_log.md`'s revision 39 section; the spec's front matter reads `version: 2.0, revision 39` from
step 6.

**The replace spans** were checked by script on 2026-10-25 against the revision 38 spec: each quoted span occurs in
`docs/bowen_agent_model_spec_v2.md` exactly once (27 spans, parts 1–3), except part 1's `M7.E.1f` anchor, which
quotes the sentence with an ellipsis and was checked by its two ends; the config and explainer spans of §14, §15 and
§21.7 were checked the same way against their files.

**The correctness pass** over parts 2 and 3 (`sonnet`, read-only, 2026-10-25, about 250k tokens, 9 minutes) found 8
errors and 14 lesser items. Twenty are fixed in the draft, two of them after the owner's answers (Q13, Q14), among them the threshold's other places (§14), `M8.5`'s
override and the mechanism table (§21.2, §21.7), `M8.1`'s sources, `M11.C.21`'s sources (§22), `M4.C.1d`'s factors,
batching and the coach tie, and the ratio's values moved to `config/sweep/sweep.md`. One is recorded, not fixed (`M5.D.4a`'s anger check at `DEFINE`, §13.1). One was checked and left: `M11.C.7`'s
premise sits in a row already classed Mixed, which fits `M11.5`'s classes.

**Documents that change across the revision**: the spec; `docs/spec_revision_log.md`; `docs/spec_index.jsonl`;
`docs/spec_rules/`; `config/bowen/stress_response.md`, `event_sizes.md`, `constants.md`, `constants_changes.md`,
`event_kinds.md`, `criteria.md`; `config/sweep/sweep.md`; `docs/model_explainer.md`; `docs/readouts_phase_c.md`;
`tools/spec_coverage_overrides.json` (`M1.A.18`, `M1.C.1`, `M8.1`, `M8.5`) and `docs/spec_coverage.md`;
`docs/version_2_plan.md`; `docs/theory/_RESOLUTIONS.md`; and, with group F's code, `src/bowen/scenario/constants.py`,
`src/bowen/engine/params.py` and `tests/bowen/test_config.py`;
`docs/SPEC_REVIEW_CHECKLIST.md`; `docs/REVIEWER_READING_LIST.md`; `CHANGELOG.md`.
