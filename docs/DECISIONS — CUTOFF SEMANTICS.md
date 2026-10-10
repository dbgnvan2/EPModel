---
tags: [model-bt, decisions]
status: OPEN — for owner decision; raised 2026-10-10 (visualization proposal, D3)
date: 2026-10-10
---

# Decision — what the engine's CUTOFF does

**Raised by:** the owner's correction to the visualization mapping, 2026-10-10
(`docs/PROPOSAL — RUN VISUALIZATION MAPPING.md` §1 and §5 Q1):

> cutoff is not about no interactions, which is distance, it is about the intensity of rejecting
> interaction. Person A refuses to interact with B.

**Already decided (D3, 2026-10-10):**
- The **pattern** readout follows the owner's definition: repeated refusals of approaches
  (`docs/readouts_phase_c.md`).
- A tie that is merely silent reads as **distance**, at a level that rises the longer the silence lasts.

**Still open:** whether the **engine's act** should change. This document records the question. It changes nothing.

## What the engine does now

- A delivered `CUTOFF` makes the tie non-interactive and sets its state to `cut_off`, and removes impingement at once
  (`src/bowen/engine/event_effects.py:120`, `apply_delivered_cutoffs`; spec `M1.B.3`; explainer act table: "End contact").
- `REDUCE_CUTOFF` restores the tie (`src/bowen/engine/moves.py:312`, spec `M5.B.3`).
- In the seed-7 Phase C run (`M11.C.1` baseline, 104 weeks), 62 `CUTOFF` and 70 `REDUCE_CUTOFF` acts were emitted. Ties are
  severed and restored about every other week.

## Where it departs from the owner's definition

- **The act** fits the definition: A severs the tie with B, which is a refusal.
- **The state the act leaves** does not fit. The tie carries no interaction at all, which the owner calls distance.
- **Rejection has no intensity in the engine.** A cutoff is all or nothing, and nothing records how hard A rejects B's
  approaches.

## Options

| | Change | What it costs |
|---|---|---|
| **A** | **No engine change.** Cutoff stays "end contact"; only the readout follows the owner's definition. | Nothing now. The engine's word "cutoff" and the readout's word mean different things. Every document that reads "cut off" as a state (`M1.B.3`'s four tie states, `M7.A.2b`, `M4.D.3b`) keeps the engine meaning. |
| **B** | **Cutoff as rejection.** The tie stays interactive. While a cutoff holds, A's appraisal of what B sends is replaced by rejection: B's approaches raise B's anxiety and do not reach A. The intensity of rejection is a per-tie quantity that rises with each refusal and falls with sustained interaction. | A spec revision (`M1.B.3`, `M5.B.3`, `M6.I.7`). An engine change, so every Phase C record changes. The Phase B and C gates would have been run on the old meaning; their records stay as history. |
| **C** | **A then B.** Keep A for now; put B on the Phase D plan, with the projection work. | As A until Phase D, then as B. |

**Recommendation: C.** Phase C is closed. B changes what every Phase C record means, and is better designed
alongside Phase D's projection and triangle work, which also reads refusals. Until then the readout carries the
owner's meaning, and the explainer and spec gain a note that the engine's `CUTOFF` is the act of severing, not the
pattern.

**Owner decision:** _open_
