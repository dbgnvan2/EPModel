"""Generate report §11 (`M11.C.16`'s class) from the mutation and occupancy records, and the three lines elsewhere that
restate it.

Purpose: every number in docs/phase_c_completion_report.md §11 comes from docs/phase_c_mutation_record.md and
         docs/phase_c_level_occupancy.md, so a stale number is a failing test, not a review finding (2026-10-09).
Spec:    docs/bowen_agent_model_spec_v2.md#M11.5, #M11.C.16
Tests:   tests/bowen/test_ensemble_record.py::test_m115_report_section_11_is_generated,
         tests/bowen/test_ensemble_record.py::test_t148_c16_sentences_follow_the_mutant_results

    python3 tools/c16_report.py

This rewrites §11 from its heading to the next top-level heading (``bounds``), the report's two `M11.C.16` table rows
(``ROW_CLASS``, ``ROW_STATUS``) and the spec's `M11.5` row (``SPEC_ROW``). The module computes them when imported.
Its qualitative claims (whether an inversion reverses C.16, whether removing a rule turns it red, whether learning
carries the result alone) are chosen from the mutant results; the sentences that were literal text until the stage 3
milestone review (MR-1, T-148: the learning bullet, "no other single rule", "what does not hold") are computed too. A
result it has no wording for (a ``broken`` mutant, a representation mutant that is not ``unchanged``, or the learner
disabled turning C.16 red, which would change the section's conclusion) stops the run instead of writing a sentence,
as do a mutant listed twice and a sweep record with no `M11.C.16` row; every row is found before any file is written,
so a stop leaves the files as they were. That holds for the claims listed here; a sentence
added later as literal text is not checked by anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.record_json import record_json  # noqa: E402

E, T = "repertoire_entropy", "top_move_share"
CENTRAL = "appraisal-sum-order-reversed"  # a representation mutant: the unmutated numbers
JOINT = ("level-blind", "availability-and-learner-removed", "c16-cited-grounds-removed", "c16-grounds-removed")
WORDED = ("reversed", "red", "survived")


def _join(names: list[str]) -> str:
    quoted = [f"`{n}`" for n in names]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " and " + quoted[-1]


def render(mutants: list[dict], occupancy: list[dict], ensemble: list[dict], sweep: list[dict]) -> dict[str, str]:
    """Purpose: §11, the two report rows and the `M11.5` row, from the records' rows (``record_json``'s form).
    Spec:    docs/bowen_agent_model_spec_v2.md#M11.5, #M11.C.16
    Tests:   tests/bowen/test_ensemble_record.py::test_t148_c16_sentences_follow_the_mutant_results"""
    names = [r["mutant"] for r in mutants if r["criterion"] == "M11.C.16"]
    if len(names) != len(set(names)):  # a dict would keep the last of two rows silently
        raise SystemExit(f"M11.C.16's mutant rows name a mutant twice: {sorted(n for n in set(names) if names.count(n) > 1)}")
    c16 = {r["mutant"]: r for r in mutants if r["criterion"] == "M11.C.16"}

    def ro(r, name):
        return next(x for x in r["readouts"] if x["readout"] == name)

    def v(r, name):
        x = ro(r, name)
        return f"{x['mean_difference']:+.3f} ± {x['half_width']:.3f}"

    def mean(m, k):
        return ro(c16[m], k)["mean_difference"]

    def hw(m, k):
        return ro(c16[m], k)["half_width"]

    def spans_zero(m):
        return all(abs(x["mean_difference"]) <= x["half_width"] for x in c16[m]["readouts"])

    def result(m):
        res = c16[m]["result"]
        if res not in WORDED:  # "broken" proves nothing; write no sentence about it
            raise SystemExit(f"{m}: result {res!r} has no wording in §11; fix the run or the generator")
        return res

    # The table's first row says the representation mutants left C.16 unchanged; every other mutant's result is one
    # §11 has words for. Either check failing stops the tool rather than print a sentence the record does not support.
    representation = {m: r["result"] for m, r in c16.items() if r["kind"] == "representation"}
    if CENTRAL not in representation or any(res != "unchanged" for res in representation.values()):
        raise SystemExit(f"M11.C.16's representation mutants are not all unchanged: {representation}")
    for m, r in c16.items():
        if r["kind"] != "representation":
            result(m)
    central = c16[CENTRAL]
    rows = ["| Mutant | Kind | Entropy difference | Top-move share difference | Result |", "|---|---|---|---|---|",
            f"| *none (representation mutants: unchanged)* | — | {v(central, E)} | {v(central, T)} | pass |"]
    for m, r in c16.items():
        if r["kind"] == "representation":
            continue
        res = r["result"] if r["result"] == "survived" else f"**{r['result']}**"
        rows.append(f"| `{m}` | {r['kind']} | {v(r, E)} | {v(r, T)} | {res} |")
    N = lambda m, k: v(c16[m], k)  # noqa: E731
    O = lambda variant, arm, key, span="whole_run": f"{next(r for r in occupancy if r['variant'] == variant and r['arm'] == arm)[span][key]:.1%}"  # noqa: E731,E501
    RED = lambda res: "turns it red" if res != "survived" else "leaves it passing"  # noqa: E731

    cited, grounds, inverted = (result(m) for m in ("c16-cited-grounds-removed", "c16-grounds-removed",
                                                     "availability-level-inverted"))
    avail_alone = result("availability-level-independent")
    cited_red = cited != "survived"
    if cited_red:
        CITED = (f"Removing the two cited rules alone, with availability kept (`c16-cited-grounds-removed`), "
                 f"{'also ' if grounds != 'survived' else ''}turns it red: entropy {N('c16-cited-grounds-removed', E)}.")
        CITED_PROOF = ("`c16-cited-grounds-removed` turning it red, with availability kept, is that clause's proof for the "
                       "cited rules alone.")
    else:
        CITED = (f"Removing the two cited rules alone, with availability kept (`c16-cited-grounds-removed`), leaves it passing: "
                 f"entropy {N('c16-cited-grounds-removed', E)}. Removing availability alone "
                 f"{RED(avail_alone)} ({N('availability-level-independent', E)}). "
                 + ("Availability alone carries it, so the pattern `M11.5`'s redundancy clause describes, in which "
                    "either set alone keeps a passing result, is not shown." if avail_alone != "survived" else
                    "Either set alone keeps a passing result; only removing both removes it, which is the pattern "
                    "`M11.5`'s redundancy clause describes." if grounds != "survived" else
                    f"Removing both together also leaves it passing ({N('c16-grounds-removed', E)}), so "
                    "no removal of these rules turns it red, and the redundancy pattern is not shown."))
        CITED_PROOF = ("`c16-grounds-removed`, which removes availability with them, would be that clause's proof; either set "
                       "alone leaves it passing." if avail_alone == "survived" else
                       "`c16-grounds-removed`, which removes availability with them, cannot be that clause's proof, since "
                       "removing availability alone turns it red.")
    # T-148: the single-rule results. A single-rule mutant is any non-representation mutant that is not one of the joint
    # removals; a named or deletion mutant removes its rule, a sign-inverted one inverts it.
    single = [m for m, r in c16.items() if r["kind"] != "representation" and m not in JOINT]
    red_del = [m for m in single if c16[m]["kind"] != "sign-inverted" and m != "availability-level-independent"
               and c16[m]["result"] != "survived"]
    red_inv = [m for m in single if c16[m]["kind"] == "sign-inverted" and m != "availability-level-inverted"
               and c16[m]["result"] != "survived"]
    rev_inv = [m for m in red_inv if c16[m]["result"] == "reversed"]
    reversed_ = [m for m, r in c16.items() if r["result"] == "reversed"]
    REVERSED_NOTE = ("none here is" if not reversed_ else
                     "here " + ", ".join(f"`{m}`" for m in reversed_) + (" is" if len(reversed_) == 1 else " are"))
    if inverted == "reversed":
        INVERSION = (f"- **Inverting `M4.D.3a`'s availability reverses C.16**: entropy "
                     f"{N('availability-level-inverted', E)} and top-move share "
                     f"{N('availability-level-inverted', T)}; the gating readout's interval lies wholly on the "
                     "opposite side. "
                     "When the owner kept C.16 composite (2026-10-08) this inversion only cancelled it; since the triangle "
                     "change of 2026-10-09 it reverses it. The owner was asked again and kept it composite (2026-10-09). **This inversion "
                     "is still weak where C.16 runs.**")
        FLIP = ("`M11.1d`'s flip is sufficient for a premise, not necessary. The availability inversion now flips C.16, "
                "which by that test would make it one. The owner kept it composite when the inversion only cancelled it "
                "(2026-10-08), and again when asked after it began to reverse it (2026-10-09).")
    else:
        INVERSION = (("- **No inversion run reverses C.16.** Inverting `M4.D.3a`'s availability " if not rev_inv else
                      f"- **Inverting `M4.D.3a`'s availability does not reverse C.16; {_join(rev_inv)} "
                      f"do{'es' if len(rev_inv) == 1 else ''} (below).** It ")
                     + f"{'cancels it' if inverted == 'red' else 'leaves it passing'}: entropy "
                     f"{N('availability-level-inverted', E)} and top-move share "
                     f"{N('availability-level-inverted', T)}. "
                     + ("That is a FAIL, so the mutant is red, but the direction is not flipped. " if inverted == "red" else "")
                     + "**This inversion is weak where C.16 runs.**")
        FLIP = ("`M11.1d`'s flip is sufficient for a premise, not necessary. "
                + ("No inversion run here flips C.16, so that test does not make it one." if not rev_inv else
                   f"{_join(rev_inv)} flip{'s' if len(rev_inv) == 1 else ''} C.16, which by that test would make it "
                   "one; the class is the owner's decision (`M11.5`), and this result goes back to the owner."))
    if grounds == "survived" and not cited_red:
        CITED_PROOF = ("but no mutant here proves it: `c16-grounds-removed`, which removes the cited rules with "
                       "availability, now leaves C.16 passing.")

    # T-148: the single-rule sentences, from the results.
    share = 1.0 - mean("availability-level-independent", E) / mean(CENTRAL, E)
    AVAIL_HEAD = ("**Availability carries part of the narrowing, not all of it.**" if avail_alone == "survived" else
                  "**Removing availability removes the narrowing.**")
    REMOVAL = ("No other single rule's removal turns it red" if not red_del else
               f"Removing {_join(red_del)} alone also turns it red ("
               + "; ".join(f"`{m}` {N(m, E)}" for m in red_del) + ")")
    INVERTS = ("" if not red_inv else
               f" Of the other inversions, {_join(red_inv)} turn{'s' if len(red_inv) == 1 else ''} it red ("
               + "; ".join((f"`{m}` " if len(red_inv) > 1 else "entropy ") + N(m, E) for m in red_inv) + ")"
               + (f", {_join(rev_inv)} reversing it." if rev_inv else ", without reversing it."))
    blind = result("level-blind")
    BLIND = ("the level-blind mutant, which removes all nine at once, "
             + ("leaves no difference at all" if blind != "survived" and spans_zero("level-blind") else
                f"{RED(blind)} ({N('level-blind', E)})"))
    SINGLE = f"{REMOVAL}; {BLIND}.{INVERTS}"
    red_single = red_del + (["availability-level-independent"] if avail_alone != "survived" else [])
    PATTERN = ("each single deletion survives and only joint deletions remove it" if not red_single else
               f"only joint deletions remove it, but for {_join(red_single)}")
    holds = ("hold-level-independent", "hold-level-inverted")
    exact = all(c16[m]["readouts"] == central["readouts"] for m in holds)
    if exact:
        HOLD = ("`M5.D.3`'s hold capacity never acts in C.16's runs: both hold mutants reproduce the unmutated numbers "
                "exactly. Those two mutants test nothing here.")
        HOLD_ROW = "`M5.D.3`'s hold capacity never acts in this criterion's runs."
    else:
        near = all(abs(mean(m, E) - mean(CENTRAL, E)) <= hw(CENTRAL, E) for m in holds)
        both = all(result(m) == "survived" for m in holds)
        HOLD = (f"`M5.D.3`'s hold capacity acts in C.16's runs: the two hold mutants "
                f"{'both leave it passing' if both else 'do not both leave it passing'} (entropy "
                + " and ".join(N(m, E) for m in holds) + f", against {v(central, E)} unmutated), "
                + f"{'each within' if near else 'not all within'} the unmutated difference's half-width.")
        HOLD_ROW = (f"`M5.D.3`'s hold mutants {'both survive' if both else 'do not both survive'}"
                    f", {'each within' if near else 'not all within'} the unmutated difference's half-width.")
    ld, li, al = (result(m) for m in ("learner-disabled", "learner-inverted", "availability-and-learner-removed"))
    if ld != "survived":
        raise SystemExit("learner-disabled turns C.16 red: §11's conclusion (composite, the mutation clause not met) no "
                         "longer follows; the owner decides its wording before the generator writes one")
    WORD = {"survived": "still passing", "red": "red", "reversed": "reversed"}
    within = abs(mean("learner-disabled", E) - mean(CENTRAL, E)) <= hw(CENTRAL, E)
    lower = mean("learner-disabled", T) < mean(CENTRAL, T)
    inv_e = mean("learner-inverted", E)
    INV_WORD = ("deepens the narrowing" if inv_e < mean(CENTRAL, E) - hw(CENTRAL, E) else
                "weakens the narrowing" if inv_e > mean(CENTRAL, E) + hw(CENTRAL, E) else
                "leaves the narrowing within the half-width of the unmutated difference")
    LEARN_HEAD = ("**Learning changes the result's size, not its sign.**" if al == "survived" else
                  "**Learning alone does not carry the result; with availability removed as well, removing it "
                  + ("reverses" if al == "reversed" else "removes") + " it.**")
    LEARN = (f"- {LEARN_HEAD} `M4.D.6` disabled leaves entropy at {N('learner-disabled', E)}, "
             f"{'within' if within else 'outside'} the half-width of the unmutated difference, and "
             f"{'lowers' if lower else 'raises'} the top-move share difference to {N('learner-disabled', T)}. Inverted, it "
             f"{INV_WORD} ({N('learner-inverted', E)}). With availability already removed, removing the learner as well "
             f"takes the entropy difference from {N('availability-level-independent', E)} to "
             f"{N('availability-and-learner-removed', E)} ({WORD[al]}) and the top-move share difference from "
             f"{N('availability-level-independent', T)} to {N('availability-and-learner-removed', T)}.")
    RATIONALE = ("Learning moves its size, but the direction survives with the learner disabled, and with availability "
                 "and the learner removed together." if al == "survived" else
                 "The direction survives with the learner disabled; with availability and the learner removed together "
                 f"it is {'reversed' if al == 'reversed' else 'removed'} ({N('availability-and-learner-removed', E)}), "
                 "so learning carries the result only jointly with availability, not alone.")

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
- {AVAIL_HEAD} Removing it leaves entropy at
  {N('availability-level-independent', E)}, against {v(central, E)} unmutated. {SINGLE}
- **C.16's cited grounds.** Its criterion row cites `M4.C.1a` (KS03.2: steepness and band) and `M1.C.3a` (routing).
  Removing those together with `M4.D.3a`'s availability (`c16-grounds-removed`) {RED(grounds)}: entropy
  {N('c16-grounds-removed', E)}. {CITED}
- {HOLD}
{LEARN}

Why composite, and not premise:

- {FLIP}
- `M11.5`'s redundancy clause makes a result a premise when several rules each state it, as at `M11.C.1`, where every
  level-reading rule states earlier onset at a lower level. C.16's pattern of mutants looks the same ({PATTERN}), but of the nine rules only availability states a narrower
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

- *The row's rationale.* Revision 11 restated C.16 so that learning produces the narrowing. {RATIONALE}
- *The row's mutation clause*, "disabling `M4.D.6` … MUST turn this red". It does not.

So C.16 stays partial in `docs/spec_coverage.md` (the tool reads the unmet clause from the criterion row), the spec's
`M11.5` row and criterion row say the clause is not met, and `TODO.md` asks whether to restate the criterion so that it tests a learned contribution.
"""

    # The report's two table rows and the spec's `M11.5` row, from the same results (T-148).
    ens = next(r for r in ensemble if r["criterion"] == "M11.C.16")
    settings = [r for r in sweep if r["criterion"] == "M11.C.16"]
    if not settings:
        raise SystemExit("the sweep record has no M11.C.16 row, so the rows could not say how many settings it passes at")
    passing = sum(r["outcome"] == "PASS" for r in settings)
    SWEEP = (f"passes at all {len(settings)} settings" if passing == len(settings) else
             f"passes at {passing} of {len(settings)} settings")
    red_all = [m for m, r in c16.items() if r["kind"] != "representation" and r["result"] == "red"]
    REVERSES = ("no mutant reverses it" if not reversed_ else
                f"{_join(reversed_)} revers{'es' if len(reversed_) == 1 else 'e'} it")
    status = "partial — passes; its mutation clause is not met, §11" if ens["outcome"] == "PASS" else \
        f"not done — {ens['outcome']} at central, §11"
    ROW_CLASS = (f"| `M11.C.16` repertoire concentration depends on level | composite (§11) | {ens['outcome']} "
                 f"({ens['seeds']}) — after §10's precision rule; undetermined at 500 seeds before | red under "
                 f"{_join(red_all) if red_all else 'no mutant that does not reverse it'}; {REVERSES}; every other mutant survives, its required "
                 f"mutant, the learner disabled, among them (§11) | {SWEEP} | {status} |")
    ROW_STATUS = (f"| `M11.C.16` | {'partial' if ens['outcome'] == 'PASS' else 'not done'} | {ens['outcome']}; its "
                  f"required mutant (learner disabled) survives; {REVERSES} | The repertoire narrows at a lower level "
                  f"with the learner disabled{'' if al == 'survived' else ', but not with availability removed as well'}; "
                  f"removing availability alone takes {share:.0%} of the entropy difference away. Composite by owner "
                  "decision (§11). |")
    ROW_INV = ("No inversion run reverses it." if not [m for m in reversed_ if c16[m]["kind"] == "sign-inverted"] else
               f"{_join([m for m in reversed_ if c16[m]['kind'] == 'sign-inverted'])} reverses it.")
    AVAIL_INV = {"reversed": "reverses it", "red": "cancels it (its gating interval spans zero)",
                 "survived": "leaves it passing"}[inverted]
    SPEC_ROW = (
        "| M11.C.16 | Composite | — | *Checked 2026-10-08 from `M11.1d`'s mutants (`docs/phase_c_mutation_record.md`); "
        "class kept by owner decision. All nine level-reading rules listed at `M11.C.1` were run as deletion and "
        "inversion mutants, with the learner; this row's mutant results are generated by `tools/c16_report.py` from "
        f"that record. {ROW_INV} Inverting `M4.D.3a`'s layer availability {AVAIL_INV} ({N('availability-level-inverted', E)}), "
        "but that inversion equals the deletion at level 20 and below on both layers "
        f"({O('availability-level-inverted', 'treatment', 'deletion_on_every_layer')} of the lowered arm's member-weeks "
        f"under it) and at 40 and below on layer 1 ({O('availability-level-inverted', 'treatment', 'deletion_on_layer_1')}; "
        f"over the entropy's window, {O('availability-level-inverted', 'treatment', 'deletion_on_every_layer', 'window')} "
        f"and {O('availability-level-inverted', 'treatment', 'deletion_on_layer_1', 'window')}; "
        "`docs/phase_c_level_occupancy.md`), so a stronger inversion might reverse it; none was tried. Removing "
        f"availability alone {RED(avail_alone)} and takes {share:.0%} of the entropy difference away. {SINGLE} "
        f"{HOLD_ROW} With the learner disabled it still passes"
        + (", and with availability removed as well." if al == "survived" else
           f"; with availability removed as well it is {WORD[al]} ({N('availability-and-learner-removed', E)}).")
        + " Composite rests on a reading: of the nine rules only availability names a narrower repertoire, so the "
        "premise definition's redundancy clause does not apply. The criterion row's own grounds (`M4.C.1a`'s KS03.2 "
        "text, `M1.C.3a`'s \"stays fixed\") could be read as stating it, which would make it a premise. Removing those "
        f"two rules together (`c16-cited-grounds-removed`) {RED(cited)}; removing them with availability "
        f"(`c16-grounds-removed`) {RED(grounds)}"
        + (", so the redundancy clause's proof holds only with availability in the set." if grounds != "survived"
           and not cited_red else ".")
        + " The mutants fit both readings; the reading is the owner's. The row's rationale (learning produces the "
        "narrowing) and the mutation clause above are not met (see the criterion row). A same-day reclassification to "
        "premise, which read the availability inversion's cancellation as a flip, was withdrawn. Report §11.* |")
    return {"section": sec, "row_class": ROW_CLASS, "row_status": ROW_STATUS, "spec_row": SPEC_ROW}


def _records() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    docs = REPO / "docs"
    return (record_json((docs / "phase_c_mutation_record.md").read_text(encoding="utf-8")),
            record_json(docs / "phase_c_level_occupancy.md"),
            record_json((docs / "phase_c_ensemble_record.md").read_text(encoding="utf-8")),
            record_json((docs / "phase_c_sweep_record.md").read_text(encoding="utf-8")))


RENDERED = render(*_records())
SECTION = RENDERED["section"]
REPORT = REPO / "docs" / "phase_c_completion_report.md"
SPEC = REPO / "docs" / "bowen_agent_model_spec_v2.md"
HEADING = "## 11. Whether `M11.C.16` is reclassified"
# Each row: its file, the text it starts with, and the heading after which it is the only such line (§10's table has a
# row that starts like §13's status row).
ROW_PREFIXES = {"row_class": (REPORT, "| `M11.C.16` repertoire concentration depends on level |", None),
                "row_status": (REPORT, "| `M11.C.16` | ", "## 13. Phase C close-out"),
                "spec_row": (SPEC, "| M11.C.16 | Composite |", None)}


def bounds(text: str) -> tuple[int, int]:
    """Where §11 starts and ends in the report: from its heading to the next top-level heading, or the end."""
    start = text.index(HEADING)
    end = text.find("\n## ", start + len(HEADING))
    return start, (len(text) if end < 0 else end + 1)


def row_line(text: str, prefix: str, after: str | None = None) -> str:
    """The one line of ``text`` (from the heading ``after`` on, when given) that starts with ``prefix``; more or fewer
    is an error, never a guess."""
    found = [line for line in text[text.index(after) if after else 0:].splitlines() if line.startswith(prefix)]
    if len(found) != 1:
        raise SystemExit(f"{len(found)} lines start with {prefix!r}, not one")
    return found[0]


def main() -> int:
    rows = []  # every row found before any file is written, so a missing row leaves nothing half-regenerated
    for key, (path, prefix, after) in ROW_PREFIXES.items():
        text = path.read_text(encoding="utf-8")
        old = row_line(text, prefix, after)
        if text.count(old) != 1:
            raise SystemExit(f"{path.name}: the row to replace is not there exactly once")
        rows.append((path, old, RENDERED[key]))
    text = REPORT.read_text(encoding="utf-8")
    start, end = bounds(text)
    REPORT.write_text(text[:start] + SECTION + ("\n" + text[end:] if end < len(text) else ""), encoding="utf-8")
    for path, old, new in rows:
        path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    print(f"wrote {REPORT.relative_to(REPO)} §11 and its two `M11.C.16` rows, and {SPEC.relative_to(REPO)}'s `M11.5` row")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
