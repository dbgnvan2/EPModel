# The spec's current rules, by module

*Generated from the spec at revision 34 by `tools/spec_rules_view.py`. Not normative: where this and the spec differ, the spec wins. Do not edit.*

One file per `## M` module of `docs/bowen_agent_model_spec_v2.md`, with §0 as `M00.md`. The view removes,
mechanically and nothing else: the front matter's `status:` line; every `⟦…⟧` revision tag (551 of them); every block-quote line, the source quotations (17 lines; the `→ file · ID` pointers stay); and the revision-history sections, from the spec's line 2117 to its end. No sentence of a requirement body is removed, the `*Revision N:*` sentences included: `history_in_body.md` lists the 87 IDs whose bodies hold one, for the owner's decision (Part E of `docs/plan_model_ergonomics.md`).

To find an ID's module, grep `docs/spec_index.jsonl` for it: the record's `module` field names the `##` heading, and `lines` the spec's lines.

| File | Module | Bytes | Words |
|---|---|---|---|
| `M00.md` | 0. Status, scope, and how to read | 8,577 | 1,341 |
| `M01.md` | M1 — Core objects | 94,359 | 14,299 |
| `M02.md` | M2 — The reference family | 12,953 | 2,173 |
| `M03.md` | M3 — Clocks, ordering and determinism | 9,059 | 1,468 |
| `M04.md` | M4 — The fast tick, step by step | 41,540 | 6,336 |
| `M05.md` | M5 — The move repertoire | 21,039 | 3,227 |
| `M06.md` | M6 — Invariants | 12,802 | 2,016 |
| `M07.md` | M7 — The slow tick | 27,057 | 4,166 |
| `M08.md` | M8 — The live-position predicate | 5,332 | 822 |
| `M09.md` | M9 — The belief layer | 4,587 | 713 |
| `M10.md` | M10 — Parameters | 13,061 | 1,986 |
| `M11.md` | M11 — Acceptance criteria | 102,670 | 15,725 |
| `M12.md` | M12 — Out of scope, and must never be built | 3,116 | 476 |
| `M13.md` | M13 — Implementation order | 7,081 | 1,046 |
| `M14.md` | M14 — Spec coverage | 25,584 | 4,002 |
| `M15.md` | M15 — Family diagram import  `Phase E` | 10,645 | 1,668 |
| `M16.md` | M16 — The run log and the readable trace | 12,522 | 1,961 |
| `M17.md` | M17 — Phase E: ensemble runner, arms and readouts | 19,994 | 2,933 |

**Size.** The view's 18 module files hold 431,978 bytes (66,358 words) against the spec's 513,409 bytes (78,938 words): 84% of the spec by bytes, 84% by words.
