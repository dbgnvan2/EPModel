# Proposal — drawing a run as a fam-diag diagram

> **Status: draft for owner approval, 2026-10-10.** Nothing is built. This sets how one run's
> activity becomes fam-diag symbols. Every threshold below is invented (`[I]`); none comes from the
> corpus.

## 1. What this is for, and what it must not do

- **Purpose.** To show one run as a family diagram you can step through, so that your review
  (plan §9; the Run Review Log) is done against a picture rather than only `docs/review/phase_c_trace_seed7.md`.
- **It is a readout, not a criterion.** Nothing in it passes or fails. It changes nothing the engine computes.
  The exporter goes in `tools/` (outside the engine hash), and the mapping table goes in config (§6).
- **It must not draw a pattern the run did not produce.** Every line, event and triangle in the file
  must trace back to records or state from that run. The tests in §7 check this.
- **It is not a real family** (`M11.F.9`). The people are the invented Phase C family, and the dates are
  synthetic (§3). The diagram carries a page note saying so.

## 2. Which run

- `M11.C.1`'s baseline arm, seed 7, 104 weeks. This is the same run as `docs/review/phase_c_trace_seed7.md`,
  so anything in the picture can be checked against the trace text.
- The seed and the arm are options on the exporter, so later runs can be drawn the same way.

## 3. Time

- **One fam-diag "year" is one block of 4 weeks.** 104 weeks become 26 steps on the slider, played back
  one step per second.
- **The years are synthetic.** Block 1 is year 1001, block 26 is year 1026, so that no date looks like a real one.
  Each event's date is the first day of its block's year. If fam-diag cannot parse years before 1900, the fallback is 2001–2026,
  with the page note.
- **The run is read in state at each block's end, not rebuilt from the log.** The log carries changes,
  not levels. So the exporter runs the simulation itself, as `tools/phase_c_trace.py` does, and takes a
  copy of the tie, triangle and person state after weeks 4, 8, … 104. Act counts come from the log's emitted
  events for the same block.

## 4. The mapping

"In the block" means the 4 weeks of that step. A *tie* is a pair from the family file.

| # | fam-diag symbol | Drawn when | Strength (`lineStyle` level 1–5) | Model source |
|---|---|---|---|---|
| M1 | **Conflict** line | at least 2 CONFLICT acts between the pair in the block, either direction | by count: 2, 3–4, 5–6, 7–9, 10+ | emitted `CONFLICT` events (sender → target) |
| M2 | **Distance** line | at least 2 DISTANCE acts between the pair in the block, **or** the tie's state is `distant` at block end | by count, as M1; level 3 if drawn only from tie state | emitted `DISTANCE`; `Relationship.tie_state` |
| M3 | **Cutoff** line | the tie is non-interactive (`interactive = no`) at the end of at least 3 of the block's 4 weeks | single style (fam-diag has one) | `Relationship.interactive`; `CUTOFF` / `REDUCE_CUTOFF` shown as events (M8) |
| M4 | **Over/underfunctioning**: fusion line with the over side marked (`adequatePersonId`) | the tie's functioning balance is at least 0.2 from even at block end | by size of the balance: 0.2, 0.4, 0.6, 0.8, 0.9+ | `Relationship.functioning_balance["joint"]` ∈ [−1, 1]; its sign gives the over side |
| M5 | **Triangle** | the triangle is active at the end of at least 2 of the block's weeks | low / medium / high by the number of TRIANGLE acts in it in the block: 0, 1–2, 3+ | `Triangle.active`, `inside_pair`, `outside`; emitted `TRIANGLE` |
| M6 | **Open connection** line | at least 2 STAY-IN-CONTACT acts between the pair in the block and nothing from M1–M4 on that tie | by count, as M1 | emitted `STAY-IN-CONTACT` |
| M7 | Not drawn: **projection** | — | — | §5, decision D4 |

**Events, on the line or person they belong to** (fam-diag's dated events; one per act, with week and
direction in the event's notes):

| # | Event | Where it goes | fam-diag type / category |
|---|---|---|---|
| M8 | CUTOFF, REDUCE_CUTOFF | the tie's line | EPE / Cutoff |
| M9 | OVERFUNCTION, UNDERFUNCTION | the tie's line | EPE / +/- Adequate |
| M10 | PURSUE | the tie's line | EPE / Distance, subtype "pursue" (decision D3) |
| M11 | TRIANGLE (seeker → third, about the partner) | the triangle | TRIANGLE / Primary |
| M12 | I-POSITION, DETRIANGLE | the person | EPE, subtype as the act |
| M13 | SYMPTOM_ONSET | the person | SYMPTOM, with the channel |
| M14 | JOB_LOSS and other exogenous events | the person | NODAL |
| M15 | Dr Halim's sessions | his ties | EPE, subtype "session" |

**When a tie qualifies for more than one line in a block**, all of them are drawn as separate lines.
For example, the same pair can have conflict and over/underfunctioning lines at once. That is what the run did, and
choosing one would hide the other.

**People and layout.** Laid out by generation from the family file: Ana, Sofia, Bruno on top; Ravi and
Marta's partnership in the middle; Nadia and Pia below; Dr Halim set apart to the side, joined only by his
two lines. Positions are fixed for the whole run (option 1).

## 5. Decisions for you

- **D1 — threshold of 2 acts per block for a line (M1, M2, M6).** One act draws nothing; it shows only as an event.
  Recommended: 2. The concern is that 1 makes every tie look active, and 3 hides short episodes.
- **D2 — line persistence.** fam-diag hides a line only before its start date. An ended line stays drawn
  (labelled "ended"). Under option 1, a conflict episode in block 5 stays on screen through block 26.
  - **(a)** Accept it for option 1. Each episode is a separate line with start and end dates, and the slider shows patterns building up.
  - **(b)** Change fam-diag so that a line is also hidden after its end date: one function, `isVisibleAtCutoff`.
    That is a small change to your app, and it falls under fam-diag's UI regression rules.

  Recommended: (b). Without it the picture shows only accumulation, not change.
- **D3 — PURSUE.** fam-diag has no pursuit symbol. Recommended: an event on the tie's line (M10), not a line type.
  Alternative: a distance line with an arrow ending (`lineEnding`) pointing at the one pursued.
- **D4 — projection.** The model has no projection act. The nearest readouts are a parent's
  `investment` in a child, and a person's `inside_other_problem` urge. Neither is projection as Bowen
  describes it. Recommended: leave it undrawn rather than draw a symbol the model does not produce.
- **D5 — the parents and spouses the model does not include.** Ana and Bruno are siblings with no parents in
  the file, and Ana's and Sofia's spouses are absent. fam-diag draws a child from a partnership.
  - Recommended: a greyed placeholder partner labelled "not in model", only where a child line needs one.
  - Alternative: no child lines for the grandparents' generation.
- **D6 — block size.** Recommended: 4 weeks, which gives 26 steps. Two weeks gives 52 steps but more flicker. Under option 2 this goes away
  (week by week).

## 6. Where the mapping lives

The table in §4 (thresholds, bands, which act goes to which symbol) goes in `config/bowen/viz_famdiag.md`.
It is editorial content, so it stays out of the tool's code. Changing a threshold is a config edit followed by a
re-export, which reruns one 104-week run and nothing else.

## 7. How it is checked

- **Trace-back test.** For every line in the output, the stated condition holds in that block's state or log
  (for example, ≥ 2 CONFLICT acts between that pair). Run on the seed-7 run.
- **Adversarial test.** A run with CONFLICT removed from every repertoire produces no conflict lines. A tie
  held interactive produces no cutoff line.
- **Format test.** The output has the keys fam-diag's import check needs (`people`, `partnerships`,
  `emotionalLines`), every id it references exists, and every date parses.
- **Not automated:** whether fam-diag opens and draws it correctly. I open it in fam-diag
  (`npm run dev`) and check the first, middle and last block against the trace, then report what I saw.

## 8. Effort

- Option 1 (exporter only, D2a): about a day, including the tests and config.
- D2b adds a small fam-diag change and its test.
- Week-by-week playback and moving positions (option 2) remain a separate decision after you have seen option 1.
