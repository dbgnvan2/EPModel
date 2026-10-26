# Spec revision 39 — draft (version 2 stage 3)

> **Status: DRAFT 2026-10-25, part 1 of revision 39 (group C: P2, F1 and `M6.I.4a`, with T-130's wording), for the
> owner's approval; not applied to the spec and not built.** Written at plan step 4
> (`docs/implementation_plan_version_2_stage_3.md` §3.8) from §3.3.1, the proposal
> `docs/PROPOSAL — DISPLACED COPING (P2).md`, and the owner's answers of 2026-10-25 (§2). Groups F and A, T-128
> (`M17.E.1`, `M10.C.1a`) and T-014 (`M8`, `M8.1`, `M8.5`, `M13.1`, `M11.C.7`, `M7.D.1a`) are drafted into this file
> in later sessions, as parts 2 and 3, before the owner approves revision 39 as a whole (D31: one revision). Nothing
> under `src/bowen/` or `config/bowen/` is edited by this page, and nothing here moves the engine hash
> (`66dcbb1e8ba5`). The spec stays at revision 38 until then: each group's approved text is applied in the same commit
> as its code (the owner's Q6, 2026-10-25, §9).

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
