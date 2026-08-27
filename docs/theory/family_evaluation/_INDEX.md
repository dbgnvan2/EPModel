---
tags: [model-bt, source, family-evaluation]
status: complete — pass 1, pass 2, ledger fold, and spec revision 4 applied 2026-08-27
date: 2026-08-26
---

# Kerr & Bowen, *Family Evaluation: An Approach Based on Bowen Theory* (1988)

The project's **sixth corpus source**, requested by the project owner on 2026-08-24 and queued
behind *Bowen Theory's Secrets* in `kerr_book/_INDEX.md`.

## ⚠ The source numbering, stated once so it stops being confusing

The ledger numbers its sections **in order of addition**, and one of those sections is not a
corpus source at all:

| Ledger section | What it is | Corpus source? |
|---|---|---|
| — | The book, 22 chapters (`L…`) | yes — 1st |
| — | 1979 Basic Video Series (`Tape N`) | yes — 2nd |
| — | Kerr–Bowen interviews, 15 (`KB…`) | yes — 3rd |
| — | Kerr, *Bowen Theory's Secrets* 2019 (`KS…`) | yes — 4th |
| — | External measures — the DSI | yes — 5th, but not a Bowen text |
| **Source 6** | **the domain expert (`U1`–`U14`)** | **no** — expert-supplied and cross-source resolutions |
| **Source 7** | **this book (`FE…`)** | **yes — 6th corpus source** |

`kerr_book/_INDEX.md:52` calls this book "source #6". That was written before the domain-expert
block took the number. **It is Source 7 in the ledger and the sixth corpus source.** The two counts
differ and both are correct; anywhere it matters, say which is meant.

**Citation tag: `FE`.** Findings are numbered `FENN.M`, where `NN` is the chapter (`00` = the
Introduction, `11` = the Epilogue). Quote verbatim; give the chapter.

## Evidence class — joint top, with the papers and the 2019 book

| Property | This source |
|---|---|
| Medium | **published book, author-edited** |
| Verbatim quotation | **permitted** |
| Date | **1988** — later than every paper (1957–76), later than the 1979 lectures, contemporaneous with the interviews, earlier than *Bowen Theory's Secrets* (2019) |
| Attribution | **mixed, and it matters** — see below |

## ⚠ Attribution — this book has two authors writing separately, not together

This is the **only** source in the project with a split byline, and the split is not cosmetic.
The attribution rule for every finding:

| Segment | Author | Cite as | Weight |
|---|---|---|---|
| Introduction, Chapters 1–10 | **Kerr** | `[K]` | Bowen theory **as Kerr states it**, 1988. Same class as the 2019 book, thirty-one years earlier. |
| Chapter 11, *Epilogue: An Odyssey Toward Science* | **Bowen** | `[B]` | **Bowen's own late written prose — the latest primary Bowen text in the project**, and the only one written after the 22 papers. |

**Every `FE` finding MUST carry `[K]` or `[B]`.** A finding from Chapters 1–10 **MUST NOT** be
attributed to Bowen without a quoted attribution inside the text itself, and a finding from the
Epilogue **MUST NOT** be attributed to Kerr. Authorship is verified from the text at the head of
each extraction file, not assumed from this table.

Where a claim looks like **Kerr's extension** rather than Bowen's, it carries `[K-ext]`, the same
grade the 2019 book uses, and **MUST NOT** be attributed to Bowen.

## Precedence

Standing rule is *latest wins, published papers break ties on ambiguity.* This book sits **between**
the papers and the 2019 book, and is written rather than spoken. Consequences:

- Against the **1979 lectures** and the **interviews** (both ASR): this book wins on any formulation
  where the difference could be transcription rather than revision.
- Against **the 22 papers**: later, so it wins on revision — but the papers are Bowen's own and
  Chapters 1–10 are not, so a Kerr formulation **MUST NOT** silently overwrite a Bowen one. Record
  both and mark the divergence.
- Against **Kerr 2019**: earlier. Where the two Kerr texts differ, **2019 wins** and the change is
  itself a finding — thirty-one years of the same author is the only longitudinal control the
  project has on a single voice.
- **The Epilogue is a special case.** It is Bowen, written, and later than every paper. On any
  point it addresses it carries the highest weight in the corpus.

## Structure — 12 segments, 173,212 words

| Segment | Words |
|---|---|
| Introduction | 2,248 |
| 1 · Toward a Natural Systems Theory | 10,705 |
| 2 · The Emotional System | 13,622 |
| 3 · Individuality and Togetherness | 12,884 |
| 4 · Differentiation of Self | 10,321 |
| 5 · Chronic Anxiety | 9,495 |
| 6 · Triangles | 12,304 |
| 7 · Nuclear Family Emotional System | 27,065 |
| 8 · Multigenerational Emotional Process | 15,378 |
| 9 · Symptom Development | 11,839 |
| 10 · Family Evaluation | 25,908 |
| 11 · Epilogue — An Odyssey Toward Science | 21,443 |

Source texts: `~/Downloads/bowen_rag/source_files/Family Evaluation_*.txt`

## Why this source matters more than its position in the queue suggests

Three things only this book can settle:

1. **`M1.A.9a`'s two-axis identity definition is cited to this book and this book has not been read.**
   The formulation the model is built on — *for self without being selfish, for other without being
   selfless* — reaches the project **secondhand**, through Kerr in 2019 quoting himself citing
   Kerr & Bowen 1988 (`KS24.4`). Reading the primary is the point of this run.
2. **The general-systems objection** (`M11.F.3`, `M11.F.3a`) is argued at length in this book's own
   Introduction, in the authors' words rather than the project's.
3. **The Epilogue is late Bowen in writing.** Every other late-Bowen source in the project is a
   transcript under a no-verbatim-quotation constraint.

## Process — identical to the Bowen book and the 2019 book

**Pass 1** per segment → **pass 2** comparative re-read against all five existing sources →
**pass 3** fold into `_LEDGER.md` as Source 7 → **spec revision 4** → documentation.

## Progress

| Segment | Pass 1 | Findings |
|---|---|---|
| Introduction | ✅ | `fe00.md` — 7 |
| 1 · Toward a Natural Systems Theory | ✅ | `fe01.md` — 17 |
| 2 · The Emotional System | ✅ | `fe02.md` — 20 ⚠ 4 new-requirement candidates |
| 3 · Individuality and Togetherness | ✅ | `fe03.md` — 20 ⭐ **`M1.A.9a` located in its primary source** |
| 4 · Differentiation of Self | ✅ | `fe04.md` — 21 ⚠ 2 contradictions with the spec (`M1.A.3`, `M5.D.4`); §13.3 settles |
| 5 · Chronic Anxiety | ✅ | `fe05.md` — 20 ⚠ new `M11.C` criterion candidate |
| 6 · Triangles | ✅ | `fe06.md` — 20 ⚠ the four-proposition 2×2 |
| 7 · Nuclear Family Emotional System | ✅ | `fe07.md` — 22 ⚠ a mechanism for **one** of `M12.2`'s two stated unknowns *(pass 1 said both; corrected at `_FE_PASS2.md` §3.1)* |
| 8 · Multigenerational Emotional Process | ✅ | `fe08.md` — 18 ⚠ **contradicts `M1.A.11b`**; supplies the quantum-jump rate |
| 9 · Symptom Development | ✅ | `fe09.md` — 16 |
| 10 · Family Evaluation | ✅ | `fe10.md` — 18 ⭐ the ten-component **readout schema** |
| 11 · Epilogue — An Odyssey Toward Science | ✅ | `fe11.md` — 20 ⭐⭐ **Bowen's own**; the master theory as a prediction engine |
| **📗 PASS 1 COMPLETE** | **✅** | **12 segments · 219 findings** |
| **PASS 2** — comparative re-read | ✅ | `_FE_PASS2.md` — 15 quotations verified verbatim, **6 pass-1 readings corrected** |
| **PASS 3** — fold into `_LEDGER.md` | ✅ | **Source 7**, sections `FE-A`–`FE-F` |
| **SPEC REVISION 4** | ✅ | **applied 2026-08-27** — all nine decisions taken; see the spec's revision-4 log |

## What this source changed in the spec

| | |
|---|---|
| **Corrected** | `M5.D.4` was **inverted**. The re-read of Ch13 authorised by decision A3 found the chapter mentions anger exactly once, and it is the mover's: freedom *from* anger is the gate. `FE04.4` and `FE11.19` agree, about the same person's anger |
| **Amended** | `M1.A.11c` (channel prior movable and overridable) · `M1.A.3b` (awareness as readout, licence as implementation) · `M7.A.1a` (dependence gates the slow clock) |
| **Unchanged on review** | `M12.2` stands in full — `FE07.3` names the locus of the first unknown, not the rule |
| **New** | ~20 requirements, 9 acceptance criteria (`M11.C.25`–`M11.C.33`), the `M11.G` readout schema, `M10.C.4`'s twelve sourced bounds, and three prohibitions (`M12.5`, `M11.F.7`, `M11.F.8`) |

**The single largest change is that the belief layer becomes a channel** (`M9.6`, `M9.7`) rather than a
parallel store — chronic anxiety runs on *what might be*, and Bowen's own correction of his term has the
transfer running **through descriptions**, which makes `M9` what the projection process operates *on*.
