# EPModel v2 — where each act places the family's difficulty, and which channel it focuses

> Read by `src/bowen/io/load.py` into `EventKinds.attribution` and `EventKinds.focus` (spec revision 25, Phase D B8;
> Claude's recommendation, approved by the owner 2026-10-16). These are classification rules, so they live here and not in the
> code (rule 9). **Every row is `[I]`.**
>
> - `locus` is whom a person who sends, receives or witnesses the act takes the difficulty to lie with (`M9.6`):
>   `other` — for the sender, the target; for the target, the sender; for a witness, half each; `target`; `sender`;
>   `—` — the act places it nowhere. The three attribution patterns of `M9.6` are not written: "each says the
>   other", "both say the same one" and "both say the children" follow from which acts a family makes.
> - `focus` is the symptom channel a family's anxious attention to the target moves (`M1.A.11c`'s family-focus term):
>   caretaking reads as the body, worry and talk-about as the mind, blame as conduct. `—` focuses none.

| kind | locus | focus | why |
|---|---|---|---|
| `CONFLICT` | other | social | each blames the other; blame is about conduct |
| `DISTANCE` | other | — | the one moved away from is where it lies, for each |
| `CUTOFF` | other | — | as distance |
| `OVERFUNCTION` | target | physical | the one taken care of is the one with the problem; caretaking reads as the body |
| `PURSUE` | target | mental | the one chased and worried about |
| `UNDERFUNCTION` | sender | — | the one giving way is the one with the problem, to both |
| `TRIANGLE` | — | — | its target is the recruited third, not the one worried about, so it places and focuses nothing (review 2026-10-12) |
