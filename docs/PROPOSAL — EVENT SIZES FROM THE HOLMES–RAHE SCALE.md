# Proposal — event sizes from the Holmes–Rahe scale

> **Status: BUILT 2026-10-22 at spec revision 36 (`M1.F.6a`, `M6.3g`, `M7.C.1l`, `M7.D.1a`; the sizes table is the
> `event_sizes` file under the bowen config), in one batch with G1 and G2; decided by the owner 2026-10-19 (Q1–Q8 in §6); drafted 2026-10-18. Q3 (E2, the
> additions' loads) stands as decided and cites no source: the correction of 2026-10-19 below §2 is kept beside it,
> and the owner was not asked again (decided as recommended under the owner's instruction of 2026-10-22, pending
> the owner's review).** Every question is
> decided: the owner answered Q1–Q8 in §6 on 2026-10-19, each on Claude's recommendation. It is built in one batch with G1 and G2 (Q7), so it waits on G2's proposal (the
> version 2 plan was approved on 2026-10-19). The owner asked: "Can you
> use the Holmes Ray scale of life events to create a relative distribution of anxiety vs. stressor." This proposal
> compares the model's event sizes with the scale's and proposes one way to use it. Every form and number is Claude's
> recommendation, accepted by the owner question by question, and `[I]`. It changes what a run computes, so it moves the engine hash and
> regenerates every record once. Questions are Q1–Q8; Q8 was added on 2026-10-19.

## 1. The scale, and what it is not

The Social Readjustment Rating Scale (Holmes, T. H. and Rahe, R. H. 1967, *Journal of Psychosomatic Research* 11(2):
213–218) gives 43 life events a weight in **life change units** (LCU), from death of a spouse at 100 down to a minor
violation of the law at 11. The values below are from Wikipedia's adult table ("Holmes and Rahe stress scale",
revision of 25 February 2026). They have not been checked against the 1967 paper, and the paper's rating method is
not described here because it was not read.

**What it measures:** how much readjustment raters judged each event to need, in total. It does not measure anxiety,
it gives no duration, and it treats an event as weighing the same for everyone it happens to.

**Its stated limits:** there is little evidence for ranking life events this way. A 1975 critique (Mechanic) notes
that it mixes good and bad events, describes events vaguely, confounds dependent and independent variables, and is
not specific enough.

**Where it stands in this project:** it is an outside rating, **not part of the corpus and not Bowen**. In the model's
grades, a size taken from it is still an invented constant `[I]`, informed by an outside ordering. It must never be
cited as a Bowen claim. Using it is also close to calibrating constants to outside data, which `docs/version_2_plan.md`
§3 lists as out of version 2 (Q1).

## 2. What the corpus says about event size

Kerr's case material sets an event's weight by the family's ties, not by the kind of event `[D]`. These are Kerr's
cases, not Bowen's, and the deciding one is from the book's Part III case chapters, which the ledger grades `[K-ext]`
throughout (`_LEDGER.md`, per `KS15.12`): Kerr's own extension, never to be attributed to Bowen.
- **Same event, opposite outcomes, set by the tie.** Kerr's within-subject control on John Nash: a partner's
  pregnancy in 1953 produced no episode, his wife's in 1958–59 preceded his breakdown, and what differed was his
  investment in the tie (`KS22.3` `[K-ext]` `[T]`, ledger table of nodal cases).
- **The same loss lowers some members and raises others, by position.** Two grandparents' deaths lowered functioning
  across four households (`KS10.4` `[K]` `[T]`). Both parents' deaths raised it in the projection target (`KS22.8`
  `[K-ext]` `[T]`).
- **Additions count, not only losses.** The husband's cousin's week-long visit improved the wife's functioning: her
  depression and bowel symptoms eased and her diabetes blood tests improved, and the improvement reverted after the
  cousin left (`KS07.1` `[K]` `[T]`). The case shows an addition moving functioning, here upward; it is not a case of
  an addition adding load. The owner's own list of nodal events includes births and marriages (`config/bowen/nodal_events.md`).

So the scale can at most give a **base size per kind of event**. The ties then scale it, as the model's grief already
does by bond energy (`M6.3g`). The scale's equal weight for everyone is the part the theory rejects.

## 3. The model's sizes today, against the scale

**Units.** A stressor of arrival intensity `I` lasting `d` weeks delivers a total of `I × (1 + 0.5 × (d − 1))`
(`stressor_persistence` 0.5, `M1.F.6`). A week's load raises a person's anxiety by about `I ÷ functional_level`. For
example, in the seed-7 trace a job loss of 120 raised Marta's anxiety by 3.2 at a functional level near 40. "Relative"
puts the death of a spouse at 100 in both columns.

| Model event (kind, rule) | Holmes–Rahe item | LCU | Model today: total load, to whom | Model, relative | At 6.9 per LCU | At 2.55 per LCU |
|---|---|---|---|---|---|---|
| Death of a spouse (`DEATH` → `GRIEF`, `M6.3g`; bond 60: 12 a week for 114 weeks) | Death of a spouse | 100 | 690, the surviving spouse | 100 | 690 | 255 |
| Death of a parent or child (bond 50: 10 a week for 104 weeks) | Death of a close family member | 63 | 525 | 76 | 435 | 161 |
| An earner's death: lost income (`INCOME_LOSS`, `M6.3h`: 15 a week, no end) | Change in financial state | 38 | 398 in the first year, and growing | 58 in year one | 262 | 97 |
| Job loss (`JOB_LOSS`, the criteria spell, `config/bowen/criteria.md`: 120 for one week) | Dismissal from work | 47 | 120, the person | 17 | 324 | 120 |
| A family member's symptom onset (`SYMPTOM_ONSET`, `M7.D.1`: 100 for one week) | Change in health of family member | 44 | 100, each person tied to the bearer | 14 | 304 | 112 |
| One's own symptom onset | Personal injury or illness | 53 | **0**: the onset lands on the bearer's partners, not on the bearer | 0 | 366 | 135 |
| Marriage (`PARTNERING`, `M7.C.1h`) | Marriage | 50 | **0**: a nodal event, never appraised | 0 | 345 | 128 |
| Retirement (`RETIREMENT`) | Retirement | 45 | **0** | 0 | 310 | 115 |
| Birth (`BIRTH`, `M7.C.1i`) | Gain a new family member | 39 | **0** | 0 | 269 | 100 |
| A child moving out (`LAUNCH`, `M7.C.1g`) | Child leaving home | 29 | **0** | 0 | 200 | 74 |
| Finding work (`FIND_WORK`, `M7.C.1k`) | Spouse starts or stops work | 26 | **0**; it lowers the income stressor to 20% | 0 | 179 | 66 |
| A job change (`JOB_CHANGE`) | Change to different line of work | 36 | a declared kind; no run uses it | — | 248 | 92 |
| A big move (`MOVE`) | Change in residence | 20 | a declared kind; no run uses it | — | 138 | 51 |
| *not modelled* | Divorce | 73 | — | — | 504 | 186 |
| *not modelled* | Marital separation | 65 | — | — | 448 | 166 |
| *not modelled* | Marital reconciliation | 45 | — | — | 310 | 115 |
| *not modelled* | Trouble with in-laws | 29 | — | — | 200 | 74 |

The last two columns are the two anchors in Q2: 6.9 per LCU keeps a spouse's death where it is; 2.55 keeps the job
loss where it is.

**What the comparison shows:**
1. **Losses by death are close to the scale.** A close family member's death sits at 76 against the scale's 63.
2. **Shocks are light.** Job loss is 17 against 47, and a family member's illness 14 against 44. Both are one-week
   events, where grief lasts one to three years.
3. **Lost income is heavy and has no end.** 58 in its first year against 38, still growing, until someone finds work.
4. **Additions and transitions weigh nothing.** Marriage, retirement, a birth and a child leaving home are counted as
   nodal events and add no anxiety. The scale puts them at 29 to 50, and the corpus counts additions (§2).
5. **A person's own illness weighs nothing for them.** Their symptom onset loads the people tied to them.
6. **Tie-bound stocks dwarf every life event.** At a level near 40, a spouse's death delivers about 17 points of
   anxiety in total through grief, spread over two years. That excludes the death transfer of the dead person's own
   anxiety (`M6.3b`), which is separate. In week 74 of the seed-7 run, a tie's bound stock delivered 226 points to one
   person in one week (`docs/version_2_plan.md` G1). For G1's peak "around a divorce", 73 LCU at the first anchor is a
   total of about 504 intensity units, roughly 13 anxiety points at level 40.

## 4. The proposal

### E1. One base size per kind of event, from the scale, scaled by the tie

- **A config table**, `config/bowen/event_sizes.md`: each event kind, its Holmes–Rahe item and LCU, with the source
  column. Which item stands for which kind is editorial, so it lives in config (rule 9).
- **One constant**, `lcu_scale` (`[I]`), the total load per LCU.
- **An event's total load** is `lcu_scale × LCU × reach`:
  - `reach` is bond energy ÷ 100 for an event that reaches a person through a tie, as grief already does;
  - it is 1 for the person the event happens to.
- **Duration stays the model's**, declared per kind, because the scale gives none. The arrival intensity is the total
  divided by `1 + 0.5 × (d − 1)`, so a long event and a short one of the same LCU deliver the same total.
- **Lost income stays outside the scale** (Q8, decided 2026-10-19). `M6.3h`'s stressor ends with an act
  (`FIND_WORK`), not a declared duration, and its 20% remainder has no end, so there is no duration to spread the
  scale's total over. It keeps `income_loss_intensity` (15) and `income_shortfall_share` (0.2), and the table records
  that the scale does not size it.
- **The criteria spell is a test input, not a life event.** `M11.C.1`'s job loss every 4 weeks at 120 is a declared
  stress schedule, and it stays as declared unless the owner says otherwise.

### E2. Additions and transitions carry load

Marriage, birth, retirement, a child moving out and finding work become lasting stressors at their scale sizes, over a
declared duration (Q4), in addition to counting as nodal events.

### E3. A person's own illness loads them

A symptom onset loads the bearer at "personal injury or illness" (53), as well as their partners at "change in health
of family member" (44) (Q5).

### E4. Not modelled yet

Divorce, separation and reconciliation enter when the model has them. Until then, divorce's size answers G1's question
of what "around a divorce" means: 73 LCU, so 504 intensity units at the first anchor.

## 5. Checks

- **Unit test (a premise):** the configured sizes keep the scale's ratios after reach and duration. For example, a
  spouse's death totals 100 ÷ 47 times a job loss at the same reach. It confirms the code renders the table and is
  never a finding.
- **Readouts before any criterion is read:** the mean anxiety and the share of weeks above the symptom threshold over a
  40-year run, before and after. The additions in E2 add load to every family, so the level at which symptoms appear
  will move.
- **Every record is regenerated once.** Each changed result is reported beside version 1's, as a new result under a new
  rule.

## 6. Questions for the owner

1. **Q1. Use the scale at all?** It is an outside rating with weak evidence behind its ranking, and using it is close
   to the calibration `docs/version_2_plan.md` keeps out of version 2. Recommended: yes, as an ordering of base sizes
   only, graded `[I]`, with the ties scaling it. **Decided 2026-10-19: yes, as an ordering of base sizes only.** The
   values stay as read from Wikipedia's table (§1), unchecked against the 1967 paper.
2. **Q2. The anchor.** Recommended: 6.9 per LCU, which keeps a spouse's death where the grief rule puts it. The scale
   uses that event as its own anchor (100), and grief is the most worked-out stressor in the model. The alternative,
   2.55, keeps the job loss and shrinks grief to 37% of today. **Decided 2026-10-19: 6.9 per LCU.** Divorce is then
   504, which sizes G1's cut-off stock (G1's Q8).
3. **Q3. Do additions and transitions carry load (E2)?** Recommended: yes, as `KS07.1` and your list of nodal events
   suggest. **Decided 2026-10-19: yes.** *(Correction, 2026-10-19, after the decision: §2 then said the visit raised a
   symptom; the source says it improved the wife's functioning, so `KS07.1` shows an addition moving functioning, not
   adding load. The decision stands; the owner may revisit it with the corrected case.)*
4. **Q4. How long does each event weigh?** Recommended for the additions: 12 weeks `[I]` each, to be swept.
   **Decided 2026-10-19: 12 weeks each.**
5. **Q5. Does a person's own illness load them (E3)?** Recommended: yes. **Decided 2026-10-19: yes.**
6. **Q6. The mapping.** Which item stands for which kind (for example, finding work as "spouse starts or stops work",
   26, or "change to different line of work", 36) is your call, item by item. **Decided 2026-10-19: §3's table as
   drafted**, with finding work at 26.
7. **Q7. Order in version 2.** Recommended: with G1, since G1's peak is sized against divorce, and both move what a run
   computes. One regeneration for both. **Decided 2026-10-19, by the owner's answer to G1's Q9:** built in one batch
   with G1 and G2.
8. **Q8. Lost income** (raised 2026-10-19, after the draft). `M6.3h`'s stressor has no declared duration, so E1's
   rule cannot size it. Keep it outside the scale at 15 a week and its 20% remainder (recommended), or scale its weekly
   rate so a declared reference year delivers the scale's 262? **Decided 2026-10-19: keep it outside the scale** (E1).

## 7. Adjacent issues found while drafting, not fixed

- `JOB_CHANGE`, `MOVE` and `SERIOUS_ILLNESS` are declared nodal kinds that no run uses.
- The comparison shows that, in delivered anxiety, a single tie's bound stock can exceed a spouse's death about
  thirteenfold. This is a quantitative case for G1's cap (`docs/version_2_plan.md`), recorded here.
