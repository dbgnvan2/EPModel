# The spec's current rules, by module

*Generated from the spec at revision 39 by `tools/spec_rules_view.py`. Not normative: where this and the spec differ, the spec wins. Do not edit.*

One file per `## M` module of `docs/bowen_agent_model_spec_v2.md`, with §0 as `M00.md`. The view removes,
mechanically and nothing else: the front matter's `status:` line; every `⟦…⟧` revision tag (583 of them); every block-quote line, the source quotations (17 lines; the `→ file · ID` pointers stay); and the revision-history sections, of which the spec has none since revision 35 moved them to `docs/spec_revision_log.md`. No sentence of a requirement body is removed: `history_in_body.md` lists the 30 IDs whose bodies hold a sentence matching a history phrase (after revision 35 moved the narratives to `docs/spec_revision_log.md`, Part E of `docs/plan_model_ergonomics.md`, these are the sentences that pass left in place, each a "stays" row of `docs/spec_revision_35_dispositions.md`).

To find an ID's module, grep `docs/spec_index.jsonl` for it: the record's `module` field names the `##` heading, and `lines` the spec's lines.

| File | Module | Bytes | Words |
|---|---|---|---|
| `M00.md` | 0. Status, scope, and how to read | 8,932 | 1,406 |
| `M01.md` | M1 — Core objects | 104,646 | 15,906 |
| `M02.md` | M2 — The reference family | 12,571 | 2,108 |
| `M03.md` | M3 — Clocks, ordering and determinism | 9,007 | 1,460 |
| `M04.md` | M4 — The fast tick, step by step | 43,158 | 6,594 |
| `M05.md` | M5 — The move repertoire | 23,682 | 3,649 |
| `M06.md` | M6 — Invariants | 15,348 | 2,426 |
| `M07.md` | M7 — The slow tick | 29,855 | 4,608 |
| `M08.md` | M8 — The live-position predicate | 5,462 | 850 |
| `M09.md` | M9 — The belief layer | 4,529 | 705 |
| `M10.md` | M10 — Parameters | 13,103 | 1,990 |
| `M11.md` | M11 — Acceptance criteria | 105,309 | 16,144 |
| `M12.md` | M12 — Out of scope, and must never be built | 3,092 | 472 |
| `M13.md` | M13 — Implementation order | 7,328 | 1,089 |
| `M14.md` | M14 — Spec coverage | 28,352 | 4,442 |
| `M15.md` | M15 — Family diagram import  `Phase E` | 10,329 | 1,616 |
| `M16.md` | M16 — The run log and the readable trace | 12,091 | 1,889 |
| `M17.md` | M17 — Phase E: ensemble runner, arms and readouts | 24,121 | 3,607 |

**Size.** The view's 18 module files hold 460,915 bytes (70,961 words) against the spec's 494,406 bytes (75,702 words): 93% of the spec by bytes, 94% by words.
