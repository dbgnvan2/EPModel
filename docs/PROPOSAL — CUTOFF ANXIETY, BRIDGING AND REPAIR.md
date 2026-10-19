# Proposal — cutoff anxiety, bridging and repair (version 2, G1)

> **Status: DRAFT 2026-10-19, for owner review. Version 2; nothing built, no spec revision.** It turns the owner's
> answers on G1, recorded in `docs/version_2_plan.md` (stage 3, group G, 2026-10-18), into rules, tests and a new
> criterion. The owner's decisions are quoted there and summarised in §1. Every form and number not marked as the
> owner's is Claude's recommendation, graded `[I]`. It changes what a run computes, so it moves the engine hash and
> every record regenerates once. Questions are Q1–Q9 in §8. The version 2 plan itself is not yet approved (its §5),
> so this proposal waits on that as well.

## 1. What the owner decided (2026-10-18)

| # | Decision | Owner's words, in short |
|---|---|---|
| D1 | A cut-off tie's anxiety stops growing at a peak and becomes chronic. | "it would reach a thresshold and become chronic" |
| D2 | Its size is around a divorce. | "I would rank it around that of a divorce" |
| D3 | It decays to 10% of its peak after 5 years and **stays there**. | "down to 10% of peak after 5 years"; "it stays there … a residual effect" |
| D4 | A negative contact across the cut returns it to its peak; a positive one removes 80% of what remains. | "Negative, takes it back to … the peak. positive, takes 80% … away" |
| D5 | Negative is argument, rejection, anger, fighting; positive is kinder words, a request to resume contact, an agreement to keep it. The act mapping (§3.4) is approved. | "yes to Act Mapping" |
| D6 | Removed anxiety dissipates. It goes to no one. A return to peak comes from the tie, felt as threat. | "Anxiety can just dissapate … It comes from the relationship tie not others" |
| D7 | `REDUCE_CUTOFF` no longer hands the stock to the triangle's third party. `M5.B.3`'s third-party cost stays, as the loss of the third party's distancing outlet. | "keep M5.B.3's third-party cost as the loss of … distancing outlet, and drop the transfer" |
| D8 | The peak, decay and floor apply to the cut-off tie only. A tie that is not cut off is just a tie; current processes apply. | "I was referring to the cutoff tie"; "it is just a tie, so current processes apply" |
| D9 | A tie stays cut off after rejection fades to zero, until a positive contact repairs it. Its stock keeps its value meanwhile. | "a positive contact is needed to repair the tie" |
| D10 | A distant or conflicted tie is repaired by three positive contacts, a third each. | "it moves 33% + or -"; confirmed "Yes" |
| D11 | Cutoff is extreme distance, chosen under anxiety for partial relief. It is not reached by a count of contacts. | "cutoff is in response to high levels of anxiety and the person wanting relief" |
| D12 | A new composite criterion: cut-off ties had longer runs of negative contacts before the cutoff than ties not cut off. Rebuffs are read as `DISTANCE`. | "Make it a composite criterion"; "yes" |

## 2. What the code does today

Checked 2026-10-19 against the working tree.

- **The stock.** `DISTANCE` moves `distance_binding_rate` × strength of the distancer's excess into the tie's
  `distance_bound_anxiety` (`moves.bind_distance`, `M1.D.2a`). Nothing decays it and nothing reads it while it is
  bound: it weighs on no one. It leaves the tie only whole, to people. On `REDUCE_CUTOFF` it goes to the third party
  (`moves.reduce_cutoff`). On reconciliation, the binder becoming unavailable, or a death, it goes back to the
  members.
- **Cut off.** A tie is cut off while either member holds any rejection (`Relationship.rejected()`, `M1.B.3c`).
  `lower_rejection` returns the tie to its state before the cut (`uncut_state`) as soon as neither holds any, so a
  cutoff that fades is repaired without a contact.
- **Refusal.** A move to someone whose rejection of the sender is above `rejection_refusal_threshold` (0.3) is
  refused, except `CUTOFF` and `REDUCE_CUTOFF` (`NEVER_REFUSED`, `rejection.py`). **So a `REDUCE_CUTOFF` can never be
  refused**, and the approved mapping's "a `REDUCE_CUTOFF` the other refuses" cannot happen (Q3).
- **Tie states.** `DISTANT` is entered after `hardening_run_length` (3) `DISTANCE` acts in a row
  (`consolidate.py`) and left only by reconciliation. `OPEN_CONFLICT` is declared and never entered.
- **The cost of a cutoff that exists.** `M4.C.1c`: a cut tie's "too little" side builds over time since contact, so
  `CUTOFF` relieves at once and costs later. That is the pull back toward contact, the longing. It is a different
  thing from the threat in D6, and this proposal leaves it unchanged.

The consequence that matters for the design: **with the transfer gone (D7) and the stock inert, the peak, decay and
bridging in D1–D4 would change nothing anyone feels.** For the owner's "residual effect" (D3) to be real, the stock
on a cut-off tie must weigh on the tie's members. That is the new mechanism this proposal adds (§3.1), and Q1 asks
the owner to confirm it. *Confirmed 2026-10-19 (§3.1).*

## 3. The rules

`s` is the tie's `distance_bound_anxiety`, `P` its peak. All new constants are in `config/bowen/constants.md` with
grade `[I]`. Where a number is the owner's, the row says so.

### 3.1 The stock weighs on the members while the tie is cut off (new; Q1, decided)

**Decided by the owner, 2026-10-19:** "Q1 yes: while a tie is cut off, its stock weighs on both members, scaled by
bond energy."

Each week a tie is cut off, each living member carries a load of `cutoff_threat_weight` × `s` × `bond_energy`/100,
read at appraisal like the standing load. It is a **source**, named `cutoff_threat` in `M6.4`'s table. It does not
draw down `s`: the stock is the size of the threat, not a store that empties as it is felt. Weighting by bond energy
follows the corpus, where weight comes from the tie, not the kind (`KS22.3`, `KS10.4`, as the Holmes–Rahe proposal
sets out). `cutoff_threat_weight` is sized from D2's divorce anchor (§3.7).

When the tie stops being cut off, the load stops. The stock keeps its value (D9), and is inert again, as today (D8).

### 3.2 The peak and the cap (D1, D2)

- `cutoff_stock_cap`, in anxiety points: no stock on a cut-off tie exceeds it. While the tie is cut off, `DISTANCE`
  across it still binds when it is delivered (below the refusal threshold), up to the cap. The distancer keeps the
  part that would pass the cap, so the act relieves less once the tie's stock is full.
- At the week the tie becomes cut off, a stock already above the cap is cut to it, and the excess dissipates
  (`cutoff_cap_dissipation`, a named sink). Recommended because the owner said anxiety "can just dissapate" (D6).
  The alternative, returning the excess to the members, is Q4.
- `P` is the highest `s` since the tie was cut, never above the cap.

### 3.3 Decay to a floor (D3)

Each week a tie is cut off: `s ← max(0.1 × P, s × f)`, with `f = 0.1^(1/260) ≈ 0.99118`. That is about 0.88% a week
and a half-life of about 78 weeks. From the peak it reaches 10% at week 260 and stays there. The fall dissipates,
`cutoff_decay` in `M6.4`. Constants: `cutoff_decay_weeks` 260 and `cutoff_floor_share` 0.1, both the owner's numbers.

### 3.4 Bridging a cut-off tie (D4, D5, D6)

Both directions count: a contact from either member to the other.

| Contact across a cut-off tie | Kind | Effect on `s` |
|---|---|---|
| `CONFLICT` delivered | negative | `s ← P`; the rise is a source, `cutoff_threat_renewed` |
| `CUTOFF` delivered | negative | as above |
| any move refused (`rejection.refuse`) | negative | as above |
| `REDUCE_CUTOFF` delivered | positive | `s ← 0.2 × s`; 80% dissipates, `cutoff_bridge_dissipation` |
| `STAY-IN-CONTACT` delivered | positive | as above |
| reconciliation event (`M4.A.3`) | positive | as above (but see Q5) |

At most one bridging effect applies per tie per week. If a week holds both a positive and a negative contact, the
negative applies (Q6).

`REDUCE_CUTOFF` keeps its own effect of lowering both members' rejection. It loses the transfer (D7).

`positive_bridge_dissipation` is 0.8, the owner's number.

### 3.5 The third party's cost (D7, `M5.B.3`, `L22.6`)

`M5.B.3`'s last sentence is restated: the transient cost of bridging **MUST** land on a third party whose distancing
mechanism is being stripped, **as the loss of that outlet, not as a transfer of the bound stock**. In the model, the
expected route is triangle position. When A and B reopen their tie, C, who stood with one of them against the other,
becomes the outsider of the A–B–C triangle and generates positional anxiety (`outsider_positional_gain`, `M1.C.1`,
`KS03.1`). **This is not yet shown to happen.** The test in §5 checks it. If the outlet's loss raises no third party's
anxiety, the build stops and reports to the owner. It does not write a cost to make the test pass.

### 3.6 Repair (D9, D10)

- **A cut-off tie.** `lower_rejection` and the weekly fade no longer reopen the tie. The tie leaves `CUT_OFF`, to its
  `uncut_state`, on the first positive contact (§3.4) delivered when neither member holds rejection, or that brings
  the last of it to zero. A cut is "a tie in state `CUT_OFF`", not "a tie holding rejection". The nodal-event spike's
  narrowing (version 2 plan §1.2, item 3) reads the same test, and changes with it.
- **A distant or conflicted tie.** `OPEN_CONFLICT` is entered as `DISTANT` is: after `hardening_run_length`
  `CONFLICT` acts in a row (Claude's). Each such tie holds a repair progress `r` ∈ [0, 1]. A positive contact adds
  `repair_step` (1/3, the owner's). A negative contact (`CONFLICT`, a rebuff read as `DISTANCE` toward the person, a
  refusal) takes `repair_step` away, never below 0. At `r` = 1 the tie returns to `ORDINARY` and `r` resets. A
  negative contact never makes the tie cut off. Cutoff comes only from a person choosing `CUTOFF` (D11).
- **The stock on a tie that is not cut off.** D8 says current processes apply, so nothing in §3.1–§3.4 reaches it.
  An earlier answer, "Principle of 2) applies to any tie - positive outcome - anxiety dissapates", said a positive
  contact dissipates on any tie. The two answers differ, and Q2 asks which holds.

### 3.7 Sizes (D2)

Both `cutoff_stock_cap` and `cutoff_threat_weight` are sized from the divorce anchor in `docs/PROPOSAL — EVENT SIZES
FROM THE HOLMES–RAHE SCALE.md`. Divorce is 73 life change units. At that proposal's recommended 6.9 intensity units
per unit, that is about 504, roughly 13 anxiety points delivered at functional level 40. Recommended pairing: a stock
at the cap, decaying as §3.3 with the load of §3.1 for a tie at bond energy 100, delivers about a divorce's total over
its first five years. That fixes one number given the other. The proposal fixes `cutoff_stock_cap` at the largest
stock a single cutoff in the reference families binds before the cut, as measured (not guessed) in the readout run of
§6 step 1, and derives the weight. Both wait on the Holmes–Rahe proposal's Q2 (the anchor).

## 4. The new composite criterion (D12)

Wording for the spec, its ID assigned at the revision: *on the Phase C and Phase D families, a tie's run of negative
contacts received before its first cutoff is longer than the longest run on the same seed's ties that are not cut off
over the same weeks.*

- **Negative contacts:** `CONFLICT`, `DISTANCE` toward the person (a rebuff, D12), and refusals, received across the
  tie in a row, in either direction. A positive contact (§3.4) ends a run.
- **Paired per seed over matched weeks.** For each first cutoff on a seed, its run up to the cutoff week is compared
  with the longest run on that seed's uncut ties within the same span. A tie never cut off is not counted over the
  whole run, which would give it more weeks to build a long run. Per-seed differences go to the runner's paired
  statistic (`M11.4e`) as usual.
- **Reported, not dropped:** seeds with no cutoff, and ties with no contact in the window, are counted in the record.
- **Reported beside the verdict:** the median run before a cutoff, against the owner's reference of more than 15
  rebuffs or 10 conflicts. The reference is the owner's, graded `[I]`, and does not gate.
- **Mutation:** cutoff chosen without reference to the anxiety the tie has built. The "too much" side of `M4.C.1`
  reads 0 when `CUTOFF` is scored. It must turn the criterion red. If it does not, the criterion is not counted as
  covered (W3), and a second mutant (a fixed `CUTOFF` propensity) is tried before reporting.
- **Class:** composite. Cutoff is emergent (§19).
- **Expected:** the criterion may fail at version 1's code. `TODO.md` records ties cut within weeks, and the couple's
  tie in the Phase C family held cut off in 82–89% of last-year weeks. A failure is a finding. It is measured before
  G1 is built and again after (§6).

## 5. Acceptance criteria and tests

Test names carry the spec IDs assigned at the revision; `<id>` marks where.

| # | Criterion | Test |
|---|---|---|
| A1 | A cut-off tie's members carry `cutoff_threat` proportional to `s` and bond energy; members of a tie not cut off carry none. | `tests/bowen/test_cutoff_stock.py::test_<id>_threat_weighs_only_while_cut_off` |
| A2 | `s` never exceeds the cap on a cut-off tie; a `DISTANCE` into a full stock leaves the distancer the remainder. | `::test_<id>_stock_capped_and_distancer_keeps_the_rest` |
| A3 | From its peak, `s` is within 1e-9 of 10% of the peak at week 260, and unchanged at week 520 with no contact. | `::test_<id>_decays_to_floor_and_stays` |
| A4 | Each negative contact kind in §3.4 returns `s` to `P`; each positive kind leaves 20% of `s`. | `::test_<id>_bridging_negative_and_positive` (parametrised over the six rows) |
| A5 | `REDUCE_CUTOFF` moves none of the stock to any third party. | `tests/bowen/test_moves.py::test_m5b3_no_stock_reaches_the_third_party` |
| A6 | Bridging raises a third party's own anxiety through the outlet's loss (§3.5). | `tests/bowen/test_moves.py::test_m5b3_third_party_cost_is_the_lost_outlet` (may stop the build, §3.5) |
| A7 | A tie whose rejection fades to zero stays `CUT_OFF` until a positive contact, and keeps `s`. | `tests/bowen/test_rejection.py::test_m1b3c_faded_cutoff_waits_for_a_positive_contact` |
| A8 | Three positive contacts return a `DISTANT` or `OPEN_CONFLICT` tie to `ORDINARY`; a negative contact sets progress back; no count of negatives cuts a tie off. | `tests/bowen/test_tie_repair.py::test_<id>_three_positive_contacts_repair` |
| A9 | Three `CONFLICT`s in a row put a tie in `OPEN_CONFLICT`. | `tests/bowen/test_tie_repair.py::test_<id>_conflict_run_hardens` |
| A10 | Every new source and sink balances in `M6.4`'s table every tick. | `tests/bowen/test_ledger.py::test_m6i6_transfers_balance_every_tick` and `::test_m64_every_stock_change_is_logged_by_name`, extended |
| A11 | The new composite criterion (§4), over an ensemble, with its mutant in the Phase D mutation record. | the criterion's `ensemble` test; `docs/phase_d_mutation_record.md` |
| A12 | The seed-7 week-74 release no longer happens: no one gains the stock of a bridged tie. | `tests/bowen/test_cutoff_stock.py::test_<id>_seed7_week74_no_stock_transfer` (a fixed-seed regression check, P23) |
| A13 | The nodal-event spike reads `CUT_OFF`, not rejection. | `tests/bowen/test_mechanisms.py::test_m4c1c_spike_follows_cut_off_state` |

Not code-testable: whether the sizes in §3.7 read as "around a divorce" to the owner. Proposed review: the seed-7
chart (`docs/review/phase_c_chart_seed7.html`) before and after, with the cutoff weeks marked.

## 6. Order of work, once approved

1. **The readout first, on version 1's code** (no engine change): the negative-run readout and the criterion's arm,
   run once, so the before is recorded. The readout lives in `src/bowen/readouts/`, so it does not move the engine
   hash. It also measures the stock sizes §3.7 needs.
2. **Spec revision** (version 2's first): `M1.B.3c` (the definition of cut, repair), `M1.B.3` (`OPEN_CONFLICT`'s
   entry), `M1.D.2a` (Q5), `M5.B.3` (D7), `M6.4` (five new sources and sinks), `M4.C.1c` (the spike's test), a new
   requirement for §3.1, and the new criterion. `docs/model_explainer.md`, `config/bowen/constants.md`,
   `constants_changes.md` and `docs/readouts_phase_c.md` change with it.
3. **Build**, tests first (§5). Batched with G2 and the Holmes–Rahe sizes if the owner takes them, so every record
   regenerates once: about 1 hour 50 minutes on 10 cores for the ensemble records (plan §4.8, measured), plus the
   mutation records.
4. **Report** the before and after on every criterion that reads cutoff: `M11.C.4` (a premise, which must still
   pass), `M11.C.10` and its variant, `M11.C.47`, `M11.C.17`, `M11.C.7`, and the new criterion.

## 7. Conflicts with the spec and the corpus, stated

- **`M1.D.2a` (Kerr, `KS04.1`, corroborated by `FE03.15`, `FE08.6`)** says bound anxiety (a) persists and (c) returns
  to the person if distancing is prevented. D3's decay and D4's 80% dissipation break (a) on a cut-off tie. Treating
  reconciliation as a positive bridge breaks (c), since reconciliation is the model's "distancing prevented". The
  binding itself still destroys nothing, which is `M1.D.2a`'s failing condition. Recommended: the owner's decision
  overrides (a) on cut-off ties and is recorded as the owner's, graded `[I]`, beside `M1.D.2a`'s `[K]` grade. Q5 asks
  about (c).
- **`M5.B.3` (`L22.6`, `[T]`)**: kept, in the form D7 decided (§3.5).

## 8. Questions for the owner

1. **Q1. Should the stock weigh on the members while the tie is cut off** (§3.1)? **Decided 2026-10-19: yes**, on
   both members, scaled by bond energy.
2. **Q2. On a tie that is not cut off, does a positive contact dissipate bound anxiety?** Your answers say "any tie"
   (answer 3) and "current processes apply" (later). Recommended: a third per positive contact, so the stock clears
   as the tie is repaired.
3. **Q3. `REDUCE_CUTOFF` cannot be refused today.** Drop "a refused `REDUCE_CUTOFF`" from the mapping (recommended,
   and §3.4 already counts any refused move as negative), or make it refusable?
4. **Q4. A stock above the cap at the cut:** dissipate the excess (recommended), or return it to the members?
5. **Q5. Reconciliation:** a positive bridge whose 80% dissipates (as the mapping you approved says), or keep
   `M1.D.2a`(c), with the stock returned to the members (recommended, since it keeps a Kerr-sourced rule)?
6. **Q6. A positive and a negative contact in the same week:** the negative applies (recommended), or they cancel?
7. **Q7. `OPEN_CONFLICT` entered by three `CONFLICT`s in a row**, as `DISTANT` is by three `DISTANCE`s? Recommended:
   yes.
8. **Q8. Sizes:** fix the cap from the measured stocks and derive the weight from the divorce anchor (§3.7)?
   Recommended: yes, after the Holmes–Rahe proposal's Q2.
9. **Q9. Batch** this with G2 and the Holmes–Rahe sizes, so every record regenerates once? Recommended: yes.
