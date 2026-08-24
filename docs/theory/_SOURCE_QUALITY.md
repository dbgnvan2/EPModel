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
| **Kerr–Bowen interviews**, Hijack folder, all 15, ~93k words | **none** — one label covers both men, and per-person attribution is unrecoverable | **90.3%** best-case Bowen precision against an **86.6%** baseline | **Concepts, mechanisms and claims, cited to the interview rather than the man.** Kerr is Bowen's successor and equally authoritative, so this is a dialogue between two valid sources. Not verbatim quotation. |

## ⚠ Superseded — the attribution problem was mis-framed

**Corrected on direction from the project owner, 2026-08-22.** The analysis below treated Kerr's turns as contamination of Bowen's. That was wrong. **Dr Michael Kerr was Bowen's successor and is as authoritative on Bowen theory as Bowen.** The interviews are two authorities in dialogue, not one author plus an interviewer, so a claim's value does not depend on which of the two made it.

The measurements below stand as measurements. Their **consequence** does not. What they establish is narrower than what the section originally concluded:

- Speaker attribution is not recoverable from these files. **This no longer blocks extraction**, because findings can be attributed to the *source* — "Kerr–Bowen #4" — rather than to a person.
- **Questions are ~99% Kerr**, per the project owner, and question-shaped text is the one thing the heuristic detects reliably. That is enough for the one distinction that still matters.
- The residual case worth care is not "who said it" but **where an exchange contains a correction** — one speaker restating or narrowing the other. There the landing point of the exchange is the finding, and it is recoverable from the text without knowing who spoke, because a correction announces itself.

**Extraction rule, as adopted:** cite the interview, not the man. Record question-shaped claims as Kerr's framing. Where an exchange corrects itself, record the landing point. Do not quote verbatim — the ASR constraint is unaffected by any of this.

### The measurements, retained

Interview #1 exists in both folders, so the Otter version provides ground truth for the Hijack version:

- The whole file is **one undifferentiated block** with a single timestamp and a single speaker label. There is no turn structure to recover.
- A question-mark heuristic scores **88.2%**, against **86.6%** for labelling every word Bowen — it adds 1.6 points.
- Restricting to long non-question spans reaches **90.3%**.

The heuristic fails because Kerr's longer turns are declarative paraphrases rather than questions. Under the corrected framing that is a feature of a dialogue between two authorities, not a defect in the source.

## One confabulation, and the scan that bounded it

Interview #1's Hijack opening reads: "I'm Dr. Michael Kerr, **Director of the Center for Disease Control and Prevention at the Michael Kerr.** and Director of Training at the Georgetown Family Center."

The Otter version has no such clause. It is fluent, plausible and invented — the signature of an LLM-assisted cleanup, and a categorically worse failure than word-level ASR error.

**It appears to be a one-off.** A scan of all 15 openings and of institutional proper nouns across the set found no other instance; the CIA references in #8 are genuine content (Bowen on whether one could analyse someone whose work cannot be discussed). The rest is ordinary ASR noise — `INAUDIBLEINAUDIBLE`, `(static)`, "Being on tape. Being on tape.", "Dr. Murray Boyne".

So the blocker is attribution, not fabrication. That is a narrower problem with a cheaper fix.

## The fix

**Still worth doing: re-export interviews #2–#15 from Otter with speaker labels.** The Otter folder proves the same audio yields clean `Speaker 1` / `Speaker 2` separation. It is no longer a blocker, but it would let the ledger record which of the two men said what, and would settle the correction cases directly rather than by inference — larger, later, and more directly interrogated than anything else the project has.

## What the Hijack set supports

A full extraction, cited to the interview. It also supports:

- **Topic coverage** — which interviews bear on which parts of the model (below), so extraction is targeted the moment attribution exists.
- **The shape of an exchange** — recording that an interview *raises* a question and *lands* somewhere, without asserting who said what. Where Kerr paraphrases and Bowen corrects, the correction is the valuable part and survives without attribution.
- **Negative findings** — a topic being absent across 15 interviews is attribution-independent and reliable.

Any finding drawn from it **MUST** cite the interview rather than a person, and **MUST NOT** be quoted verbatim.

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
