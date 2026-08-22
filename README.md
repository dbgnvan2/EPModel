# EPModel

A simulation of emotional process at unit, family and societal level, built on Bowen family systems theory.

> **The project is mid-pivot.** The original engine is a stress field over 10,000 units on a 100×100 grid. It is being replaced by a model of **behaving agents in one three-generation family**, because the grid substrate cannot express the part of the theory the project now wants — there is no dyad in it, and therefore no triangle. The grid engine still runs and its tests stay green until the replacement passes its acceptance criteria.

## Where to start

| If you want to | Read |
|---|---|
| understand **what the model is and why** | [`docs/agent_model_proposal.html`](docs/agent_model_proposal.html) |
| understand **every part of the model**, without reading Bowen | [`docs/model_explainer.md`](docs/model_explainer.md) |
| **build it** | [`docs/bowen_agent_model_spec_v2.md`](docs/bowen_agent_model_spec_v2.md) |
| check **what the source actually says** | [`docs/theory/_LEDGER.md`](docs/theory/_LEDGER.md) |
| know **where the work is up to** | [`docs/theory/_STATUS.md`](docs/theory/_STATUS.md) |

## The theory work

All 22 chapters of Bowen's *Family Therapy in Clinical Practice* have been extracted twice — a cold pass chapter by chapter, then a comparative pass re-reading each chapter against the whole book. The output is in `docs/theory/`: 149 per-chapter findings, ten cross-chapter convergences with an independence audit, and a resolution document for the contradictions.

**Two results from that work govern everything else in this repo:**

1. **The corpus contains no validation of any kind.** No instrument, no rater procedure, no comparison group, and no number ever assigned to a person, in any of the 22 chapters. It supports *directions, orderings and mechanisms*, and almost no *magnitudes*. Every constant in the model is invented, and the source cannot narrow the range even in principle.

2. **The first pass over the corpus over-read it, consistently in one direction** — making the source look more quantitative and more decided than it is. Nineteen findings were withdrawn on the second pass, including two numeric "calibration targets" that turned out to be manufactured from illustrations Bowen explicitly bounded. Every claim in the explainer is therefore graded, so an invented constant can never be mistaken for a sourced one.

## Layout

```
src/engine.py        the grid engine — frozen in behaviour, still under test
src/main.py          orchestration, UI, I/O
src/bowen/           the agent model (not yet started)
tests/               36 tests, all passing
docs/theory/         the corpus extraction, ledger and convergences
docs/                proposal, explainer, v2 spec, frozen v1.2 spec
```

## Running

```bash
python3 -m pytest tests/
```

```bash
python3 src/main.py
```

Requires Python 3, NumPy and Pygame. `requirements.txt` is present but not yet tracked — see `TODO.md`.

## Status

The v2 specification is **drafted and awaiting approval**. No code moves until it is approved: the project's convention is spec → plan → build, each approved before the next.
