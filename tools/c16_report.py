"""Generate report §11 (`M11.C.16`'s class) from the mutation and occupancy records.

Purpose: every number in docs/phase_c_completion_report.md §11 comes from docs/phase_c_mutation_record.md and
         docs/phase_c_level_occupancy.md, so a stale number is a failing test, not a review finding (2026-10-09).
Spec:    docs/bowen_agent_model_spec_v2.md#M11.5, #M11.C.16
Tests:   tests/bowen/test_ensemble_record.py::test_m115_report_section_11_is_generated

    python3 tools/c16_report.py

This rewrites §11 from its heading to the next top-level heading (``bounds``). The module computes ``SECTION`` when
imported. Its qualitative claims (whether an inversion reverses C.16, whether removing its grounds turns it red) are
chosen from the mutant results too, so a regenerated record cannot leave a false sentence behind.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
rec = (REPO / "docs" / "phase_c_mutation_record.md").read_text(encoding="utf-8")
d = json.loads(re.search(r"```json\n(.*?)\n```", rec, re.S).group(1))
c16 = {r["mutant"]: r for r in d if r["criterion"] == "M11.C.16"}
def ro(r, name):
    return next(x for x in r["readouts"] if x["readout"] == name)
def v(r, name):
    x = ro(r, name); return f"{x['mean_difference']:+.3f} ± {x['half_width']:.3f}"
central = c16["appraisal-sum-order-reversed"]
rows = ["| Mutant | Kind | Entropy difference | Top-move share difference | Result |", "|---|---|---|---|---|",
        f"| *none (representation mutants: unchanged)* | — | {v(central,'repertoire_entropy')} | {v(central,'top_move_share')} | pass |"]
for m, r in c16.items():
    if r["kind"] == "representation":
        continue
    res = r["result"] if r["result"] == "survived" else f"**{r['result']}**"
    rows.append(f"| `{m}` | {r['kind']} | {v(r,'repertoire_entropy')} | {v(r,'top_move_share')} | {res} |")
N = lambda m, k: v(c16[m], k)
occ = json.loads(re.search(r"```json\n(.*?)\n```", (REPO / "docs" / "phase_c_level_occupancy.md").read_text(encoding="utf-8"), re.S).group(1))
O = lambda variant, arm, key, span="whole_run": f"{next(r for r in occ if r['variant'] == variant and r['arm'] == arm)[span][key]:.1%}"
cited = c16["c16-cited-grounds-removed"]
cited_red = cited["result"] != "survived"
if cited_red:
    CITED = (f"Removing the two cited rules alone, with availability kept (`c16-cited-grounds-removed`), also turns it red: "
             f"entropy {N('c16-cited-grounds-removed','repertoire_entropy')}.")
    CITED_PROOF = ("`c16-cited-grounds-removed` turning it red, with availability kept, is that clause's proof for the "
                   "cited rules alone.")
else:
    CITED = (f"Removing the two cited rules alone, with availability kept (`c16-cited-grounds-removed`), leaves it passing: "
             f"entropy {N('c16-cited-grounds-removed','repertoire_entropy')}. Removing availability alone also leaves it "
             f"passing ({N('availability-level-independent','repertoire_entropy')}). "
             + ("Either set alone keeps a passing result; only removing both removes it, which is the pattern "
                "`M11.5`'s redundancy clause describes." if c16["c16-grounds-removed"]["result"] != "survived" else
                "Removing both together also leaves it passing, though much reduced (above), so no removal of these "
                "rules turns it red, and the redundancy pattern is not shown."))
    CITED_PROOF = ("`c16-grounds-removed`, which removes availability with them, would be that clause's proof; either set "
                   "alone leaves it passing.")
inverted = c16["availability-level-inverted"]["result"]
grounds = c16["c16-grounds-removed"]["result"]
reversed_ = [m for m, r in c16.items() if r["result"] == "reversed"]
RED = lambda result: "turns it red" if result != "survived" else "leaves it passing"
REVERSED_NOTE = ("none here is" if not reversed_ else
                 "here " + ", ".join(f"`{m}`" for m in reversed_) + (" is" if len(reversed_) == 1 else " are"))
if inverted == "reversed":
    INVERSION = (f"- **Inverting `M4.D.3a`'s availability reverses C.16**: entropy "
                 f"{N('availability-level-inverted','repertoire_entropy')} and top-move share "
                 f"{N('availability-level-inverted','top_move_share')}, both intervals wholly on the opposite side. "
                 "When the owner kept C.16 composite (2026-10-08) this inversion only cancelled it; since the triangle "
                 "change of 2026-10-09 it reverses it, so the class goes back to the owner (`TODO.md`). **This inversion "
                 "is still weak where C.16 runs.**")
    FLIP = ("`M11.1d`'s flip is sufficient for a premise, not necessary. The availability inversion now flips C.16, "
            "which by that test would make it one; the owner's decision to keep it composite was made when it only "
            "cancelled, and is put back to the owner.")
else:
    INVERSION = (f"- **No inversion run reverses C.16.** Inverting `M4.D.3a`'s availability "
                 f"{'cancels it' if inverted == 'red' else 'leaves it passing'}: entropy "
                 f"{N('availability-level-inverted','repertoire_entropy')} and top-move share "
                 f"{N('availability-level-inverted','top_move_share')}. "
                 + ("That is a FAIL, so the mutant is red, but the direction is not flipped. " if inverted == "red" else "")
                 + "**This inversion is weak where C.16 runs.**")
    FLIP = ("`M11.1d`'s flip is sufficient for a premise, not necessary. No inversion run here flips C.16, so that "
            "test does not make it one.")
if grounds == "survived":
    CITED_PROOF = ("but no mutant here proves it: `c16-grounds-removed`, which removes the cited rules with "
                   "availability, now leaves C.16 passing.")
sec = f"""## 11. Whether `M11.C.16` is reclassified — decided 2026-10-08

The owner asked whether C.16 should be reclassified, now that it passes but survives its named mutant. **Decided: it
stays composite.** Its row's rationale, and the mutation clause it carries, are not met.

*Two corrections were made on the way, both from csdp sweeps the same day. The first check ran an availability
inversion that saturated at the layer clamp over C.16's levels, so it removed availability instead of inverting it.
It now reflects availability about level 30, and `tests/bowen/test_ensemble_record.py` checks, from the levels in
config, that it falls between the arms and that the saturated form and the deletion would be caught. Every
level-reading rule `M11.C.1` lists was then run as a deletion and an inversion mutant on C.16; before, only two of the
nine had been. Second, the corrected inversion turned C.16 red, and this section briefly called it a premise. The
re-sweep found that the inversion cancels the result rather than reversing it, which is not `M11.1d`'s flip, and the
owner kept C.16 composite. Third, later sweeps asked for the mutants that remove the rules C.16's own criterion row
cites as its grounds, which `M11.5`'s redundancy clause needs as its proof, with and without availability (below).*

The table is generated from `docs/phase_c_mutation_record.md`. Differences are the lowered arm minus the baseline arm,
mean ± 95% half-width, at the central setting. The criterion tests a negative entropy difference; the top-move share
is reported beside it and never tested. A red mutant is marked **reversed** only when every gating readout's interval
lies wholly on the opposite side; {REVERSED_NOTE}.

""" + "\n".join(rows) + f"""

What this shows:

{INVERSION}
  It equals the deletion (every layer fully available) at level 20 and below on both layers, and at 40 and below on
  layer 1. Under the mutant, over the whole run, the lowered arm's members spend {O('availability-level-inverted','treatment','deletion_on_every_layer')} of member-weeks at level 20 or
  below and {O('availability-level-inverted','treatment','deletion_on_layer_1')} at 40 or below; the baseline arm's, {O('availability-level-inverted','baseline','deletion_on_every_layer')} and {O('availability-level-inverted','baseline','deletion_on_layer_1')}. Over the last
  52 weeks, which the entropy reads, the lowered arm's are {O('availability-level-inverted','treatment','deletion_on_every_layer','window')} and {O('availability-level-inverted','treatment','deletion_on_layer_1','window')}
  (`docs/phase_c_level_occupancy.md`, which also gives the unmodified shares; these are shares of member-weeks, not
  of selections). So it mostly reverses layer 2 only. A stronger
  inversion was not tried.
- **Availability carries part of the narrowing, not all of it.** Removing it leaves entropy at
  {N('availability-level-independent','repertoire_entropy')}, against {v(central,'repertoire_entropy')} unmutated. No other single rule removes
  the result; the level-blind mutant, which removes all nine at once, leaves no difference at all.
- **C.16's cited grounds.** Its criterion row cites `M4.C.1a` (KS03.2: steepness and band) and `M1.C.3a` (routing).
  Removing those together with `M4.D.3a`'s availability (`c16-grounds-removed`) {RED(grounds)}: entropy
  {N('c16-grounds-removed','repertoire_entropy')}. {CITED}
- `M5.D.3`'s hold capacity never acts in C.16's runs: both hold mutants reproduce the unmutated numbers exactly. Those
  two mutants test nothing here.
- **Learning changes the result's size, not its sign.** `M4.D.6` disabled leaves entropy at
  {N('learner-disabled','repertoire_entropy')}, within the half-width of the unmutated difference, and lowers the top-move share
  difference to {N('learner-disabled','top_move_share')}. Inverted, it deepens the narrowing
  ({N('learner-inverted','repertoire_entropy')}). With availability already removed, removing the learner as well takes the
  top-move share difference from {N('availability-level-independent','top_move_share')} to {N('availability-and-learner-removed','top_move_share')}.

Why composite, and not premise:

- {FLIP}
- `M11.5`'s redundancy clause makes a result a premise when several rules each state it, as at `M11.C.1`, where every
  level-reading rule states earlier onset at a lower level. C.16's pattern of mutants looks the same (each single
  deletion survives and only joint deletions remove it), but of the nine rules only availability states a narrower
  repertoire at a lower level. The other eight state other things: `M4.C.1a` a steeper appraisal and a narrower
  band, `M1.A.6` a lower symptom threshold, `M4.A.5` a larger self term, `M4.D.1a` a larger automatic share of each
  selection, `M1.A.9` more initial outside-ness, `M1.C.3a` more triangle routing, `M5.D.3` less I-POSITION hold. A
  narrower repertoire follows from them jointly; their formulas do not name it. On that reading the clause does not apply, and the owner kept C.16 composite. **The reading
  is contestable:** C.16's own criterion row gives as its grounds `M4.C.1a`'s source text (poorly differentiated
  people "are very prone to shut down and distance or to react aggressively", KS03.2) and `M1.C.3a`'s "stays fixed".
  If those rules are read as stating the narrowing, the redundancy clause would make C.16 a premise; {CITED_PROOF} Which reading holds is a judgement about what the rules
  state, not a mutant result; the mutants are consistent with both. The class stays composite by owner decision, and
  `TODO.md` puts the reading back to the owner with this result.

What does not hold:

- *The row's rationale.* Revision 11 restated C.16 so that learning produces the narrowing. Learning moves its size,
  but the direction survives with the learner disabled, and with availability and the learner removed together.
- *The row's mutation clause*, "disabling `M4.D.6` … MUST turn this red". It does not.

So C.16 stays partial in `docs/spec_coverage.md` (an override records why), the spec's `M11.5` row and criterion row
say the clause is not met, and `TODO.md` asks whether to restate the criterion so that it tests a learned contribution.
"""
SECTION = sec
REPORT = REPO / "docs" / "phase_c_completion_report.md"
HEADING = "## 11. Whether `M11.C.16` is reclassified"


def bounds(text: str) -> tuple[int, int]:
    """Where §11 starts and ends in the report: from its heading to the next top-level heading, or the end."""
    start = text.index(HEADING)
    end = text.find("\n## ", start + len(HEADING))
    return start, (len(text) if end < 0 else end + 1)


def main() -> int:
    text = REPORT.read_text(encoding="utf-8")
    start, end = bounds(text)
    REPORT.write_text(text[:start] + SECTION + ("\n" + text[end:] if end < len(text) else ""), encoding="utf-8")
    print(f"wrote {REPORT.relative_to(REPO)} §11")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
