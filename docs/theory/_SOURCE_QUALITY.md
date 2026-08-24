---
tags: [model-bt, source, method]
status: current
date: 2026-08-22
---

# Recorded sources — quality, and what each can support

Three recorded corpora have been added alongside the book. They are not one evidence class, and the differences decide what may be extracted from each.

## Measured, not assumed

| Source | Speaker attribution | Measured error | What it supports |
|---|---|---|---|
| **The book**, 22 chapters | n/a — published prose | none | Everything. A single word is legitimately load-bearing (Ch16's "Other than the emotional distance"). |
| **1979 Basic Video Series**, 6 tapes, ~43k words | single speaker, so no ambiguity | **97.2%** word agreement between two independent transcriptions of Tape 1 | Substance and mechanism. **Not** verbatim quotation. |
| **Kerr–Bowen interviews**, Otter folder, #1 only, ~5.6k words | **clean** — `Speaker 1` = Kerr, `Speaker 2` = Bowen, consistent | same ASR class | Substance, mechanism, and attributed claims. The best of the recorded material. |
| **Kerr–Bowen interviews**, Hijack folder, all 15, ~93k words | **none** — one label, `Google Chrome`, covers both men | **90.3%** best-case Bowen precision against an **86.6%** baseline | Topic coverage and the shape of an exchange. **Not** attributed claims. |

## Why the Hijack set cannot support attributed extraction

Interview #1 exists in both folders, so the Otter version provides ground truth for the Hijack version. Measured against it:

- The whole file is **one undifferentiated block** with a single timestamp and a single speaker label. Timestamp alignment is impossible; there is no turn structure to recover.
- A question-mark heuristic — treat sentences ending in `?` as Kerr, the rest as Bowen — scores **88.2%**, against **86.6%** for simply labelling every word Bowen. It adds 1.6 points and is worthless.
- Restricting to long non-question spans reaches **90.3%**. **About one word in ten attributed to Bowen would be Kerr's.**

The reason the heuristic fails is the reason this matters: **Kerr's dangerous turns are not questions.** He makes long declarative paraphrases — "So you're saying number one is people just to be aware that they're stuck…" — which read exactly like theory claims, and which Bowen sometimes *corrects* in the next breath. Recording an interviewer's paraphrase as the author's claim is the precise failure this project has spent two passes undoing.

**Asymmetry of consequence.** Missing a Bowen claim costs completeness. Attributing Kerr's paraphrase to Bowen costs correctness, and does so invisibly. Only the second is unacceptable, so extraction from this set must be biased hard toward precision.

## One confabulation, and the scan that bounded it

Interview #1's Hijack opening reads: "I'm Dr. Michael Kerr, **Director of the Center for Disease Control and Prevention at the Michael Kerr.** and Director of Training at the Georgetown Family Center."

The Otter version has no such clause. It is fluent, plausible and invented — the signature of an LLM-assisted cleanup, and a categorically worse failure than word-level ASR error.

**It appears to be a one-off.** A scan of all 15 openings and of institutional proper nouns across the set found no other instance; the CIA references in #8 are genuine content (Bowen on whether one could analyse someone whose work cannot be discussed). The rest is ordinary ASR noise — `INAUDIBLEINAUDIBLE`, `(static)`, "Being on tape. Being on tape.", "Dr. Murray Boyne".

So the blocker is attribution, not fabrication. That is a narrower problem with a cheaper fix.

## The fix

**Re-export interviews #2–#15 from Otter with speaker labels.** The Otter folder proves the same audio yields clean `Speaker 1` / `Speaker 2` separation. That single step converts ~93,000 words of late-period Bowen from topic-level material into a fully extractable corpus — larger, later, and more directly interrogated than anything else the project has.

## What the Hijack set supports meanwhile

Not a ledger-quality extraction. It does support:

- **Topic coverage** — which interviews bear on which parts of the model (below), so extraction is targeted the moment attribution exists.
- **The shape of an exchange** — recording that an interview *raises* a question and *lands* somewhere, without asserting who said what. Where Kerr paraphrases and Bowen corrects, the correction is the valuable part and survives without attribution.
- **Negative findings** — a topic being absent across 15 interviews is attribution-independent and reliable.

Any finding drawn from it **MUST** be marked attribution-uncertain and **MUST NOT** be quoted.

## Coverage map — Kerr–Bowen series against model areas

Term frequencies, speaker-agnostic. Highest counts per area in bold.

| # | Interview | Bears most on |
|---|---|---|
| 1 | Family Systems Theory & Therapy | fusion/togetherness (**20**); the misconceptions framing — *analysed, from the Otter version* |
| 2 | The Theoretical Base | measurement/research (17), anxiety (5) |
| 3 | Systems Therapy | measurement (17), reactivity (6) |
| 4 | **Anxiety and Emotional Reactivity** | **anxiety (74)**, sinks/symptom (**23**), reactivity (18), triangles (7) — the richest for A3, C2, A1 |
| 5 | Defining a Self in Family of Origin, Pt 1 | **triangles (22)**, multigenerational (17), cutoff (7) |
| 6 | Defining a Self in Family of Origin, Pt 2 | reactivity (**19**), triangles (16), fusion (16), cutoff (9) |
| 7 | Obstacles to Systems | multigenerational (6), measurement (7) |
| 8 | Family Therapy with Schizophrenia | measurement (12), anxiety (8) |
| 9 | Background to Systems Thinking | measurement (9), societal (4) |
| 10 | **Emotional Process in Society** | **societal (90)**, sinks/symptom (14) — the dials, C3 and C4 |
| 11 | Supernatural Phenomena | measurement (22), societal (14), fusion (9) |
| 12 | **Background Aspects of Differentiation** | **fusion/togetherness (38)**, differentiating move (**7**) — A4, A5 |
| 13 | The Changing World | societal (28) |
| 14 | The Best of Family Therapy | triangles (6), societal (4) |
| 15 | **A New Concept of the Midbrain** | **measurement/biology (127)** — bears directly on the biochemical-markers thread in `_KERR_INTERVIEWS.md` |

**The corpus is richest exactly where the model is weakest.** #4 on anxiety and reactivity speaks to A3, which the 1979 lectures left silent. #10 and #13 on society speak to the three dials, which have almost no corpus grounding. #12 on differentiation speaks to A4 and A5. #15 on the midbrain is the biological grounding Bowen said he expected and never had.
