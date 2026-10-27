"""Run the Phase C acceptance criteria over adaptive ensembles and write the committed record.

Purpose: produce docs/phase_c_ensemble_record.md — each criterion's verdict, seeds used,
         per-readout difference, interval, margin, p, the fallback rate and any flag — stamped
         with a hash of the code and config it ran against (plan D7).
Spec:    docs/bowen_agent_model_spec_v2.md#M17.A.1, #M11.D.18, #M11.5, #M13.4
Tests:   tests/bowen/test_ensemble_record.py::test_m134_ensemble_record_is_current,
         tests/bowen/test_ensemble_record.py::test_mr23_record_renders_from_the_cache,
         tests/bowen/test_ensemble_record.py::test_mr23_a_stale_entry_is_rerun

    python3 tools/ensemble_record.py [--workers N] [--only ID ...]

Each verdict is cached in docs/records_cache/phase_c_ensemble.json (stage 1 milestone review MR-23, built at stage 3
step 10), keyed as the Phase D records' results are: by ``result_hash`` (``code_hash`` with no tool: ``src/bowen``
outside ``render/``, ``config/bowen`` and the rules) and the criterion's key. This tool and its renderer are not in
the key, so a change to either re-renders the record from the cache; a change to the engine reruns every criterion.
``--only`` runs only the named criteria that have no result; the record is written when every criterion has one.

The record is generated, never hand-edited. A default-suite test re-renders it from its cache and fails when the
two differ, so a stale record cannot stand in for a run.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RECORD = REPO / "docs" / "phase_c_ensemble_record.md"
CACHE_NAME = "phase_c_ensemble"  # docs/records_cache/<name>.json (MR-23)
# RULE_KEYS is an input to every verdict; ``code_hash`` hashes its value, so the other records' cache keys do not
# change when this file's hashing code does (Phase D plan B2). This file is not in its own record's hash (MR-23): the
# record is re-rendered from its cache and compared whole, so a renderer change shows as a difference, not a rerun.
RULE_KEYS = ("ensemble_block", "ensemble_cap", "ensemble_precision", "ensemble_margin", "ensemble_alpha",
             "fallback_flag_rate", "equivalence_margin")


def code_hash(*tools: Path) -> str:
    """Hash of everything a verdict depends on: engine, policy, scenario, ensemble, readouts and config.

    ``tools`` adds the generating tools whose own content is an input to a record, so changing one
    makes that record stale too: the mutant list of ``tools/mutation_record.py`` and the settings of
    ``tools/sweep_record.py``. Each such record names its tools in its own ``HASHED_TOOLS``. This file's
    ``RULE_KEYS`` (which constants become the ensemble rules, an input to every record) is hashed by value.

    ``src/bowen/render/`` is left out (version 2 stage 1, S1.8a): it turns records into text and no verdict reads it
    (``tests/bowen/test_engine_purity.py::test_s1_8b_only_run_imports_render``), so a renderer change reruns nothing.
    Tests: tests/bowen/test_ensemble_record.py::test_s1_8a_render_is_not_hashed
    """
    digest = hashlib.sha256()
    digest.update(f"RULE_KEYS={RULE_KEYS!r}".encode())
    bowen = REPO / "src" / "bowen"
    paths = sorted(p for p in bowen.rglob("*.py") if p.relative_to(bowen).parts[0] != "render")
    paths += sorted((REPO / "config" / "bowen").rglob("*.md"))
    paths += sorted(Path(t).resolve() for t in tools)
    for path in paths:
        if path.name == "constants_changes.md":
            continue  # a log of changes, not an input
        digest.update(path.relative_to(REPO).as_posix().encode())
        digest.update(code_text(path).encode() if path.suffix == ".py" else path.read_bytes())
    return digest.hexdigest()


def code_text(path: Path) -> str:
    """A Python file as the hash sees it: its tokens without comments, docstrings or blank lines, so editing prose in
    the engine does not rerun every record (learnings P38). Only a docstring's own token is dropped, so code sharing
    its line stays hashed, and every other string is kept exactly, blank lines and spaces inside it included (Phase D
    plan B2; the two gaps found 2026-10-09).

    The text is the tokens' strings joined by a separator, not ``untokenize``'s output, and an f-string is taken as its
    exact source text: Python 3.12 splits an f-string into several tokens where 3.11 gives one, so anything built from
    the split tokens would hash differently by version (review 2026-10-10; the suite runs on 3.11 in CI). Docstrings
    are found with ``ast``, whose column offsets count UTF-8 bytes, converted to the characters ``tokenize`` counts.
    """
    return source_text(path.read_text(encoding="utf-8"))


def source_text(source: str) -> str:
    """``code_text`` for source already read, such as a function's from ``inspect.getsource`` (stage 1 rider (iv)).

    Lines are split as ``tokenize`` reads them, at ``\\n`` only (S1.5b): ``str.splitlines`` also breaks at U+2028 and
    other separators that can sit inside a string, which shifted every later line's number.
    Tests: tests/bowen/test_ensemble_record.py::test_s1_5b_code_text_keeps_a_line_separator_inside_a_string
    """
    import ast
    import io
    import tokenize

    lines = io.StringIO(source).readlines()

    def chars(row: int, byte_col: int) -> int:
        return len(lines[row - 1].encode("utf-8")[:byte_col].decode("utf-8", errors="ignore"))

    def text_between(start: tuple[int, int], end: tuple[int, int]) -> str:
        (r0, c0), (r1, c1) = start, end
        if r0 == r1:
            return lines[r0 - 1][c0:c1]
        return lines[r0 - 1][c0:] + "".join(lines[r0:r1 - 1]) + lines[r1 - 1][:c1]

    docstrings = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docstrings.add((first.value.lineno, chars(first.value.lineno, first.value.col_offset)))
    fstring_start = getattr(tokenize, "FSTRING_START", None)
    fstring_end = getattr(tokenize, "FSTRING_END", None)
    kept, depth, opened = [], 0, None
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if fstring_start is not None and token.type == fstring_start:
            depth += 1
            opened = opened or token.start
            continue
        if depth:
            if token.type == fstring_end:
                depth -= 1
                if not depth:
                    kept.append(text_between(opened, token.end))
                    opened = None
            continue
        if token.type in (tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER):
            continue
        if token.type == tokenize.STRING and token.start in docstrings:
            continue
        kept.append(token.string)
    return "\x00".join(kept)


def result_hash() -> str:
    """The hash of the code that computes a verdict (``code_hash`` with no tool). The record prints it; each cached
    result is keyed by it (MR-23, as ``tools/phase_d_record.py``'s S1.8c)."""
    return code_hash()


def build(cache, workers: int | None = None, only: list[str] | None = None) -> str:
    """Purpose: the record, from the cache. With ``workers``, a criterion with no result for the current code is run
             first and cached (with ``only``, only those it names); a criterion still without a result raises
             KeyError, so a test can check the committed record without running anything.
    Spec:    docs/bowen_agent_model_spec_v2.md#M13.4; stage 1 milestone review MR-23 (T-012)
    Tests:   tests/bowen/test_ensemble_record.py::test_mr23_record_renders_from_the_cache,
             tests/bowen/test_ensemble_record.py::test_mr23_a_stale_entry_is_rerun
    """
    from src.bowen.ensemble.criteria import CRITERIA, NOT_BUILT
    from src.bowen.ensemble.runner import Verdict, run_criterion
    from src.bowen.io.load import load_constants
    from tools.record_cache import result_key

    constants = load_constants()
    rules = {k: constants[k] for k in RULE_KEYS}
    engine = result_hash()
    verdicts = []
    for cid, criterion in CRITERIA.items():
        key = result_key(engine, None, cid)
        entry = cache.get(key)
        if entry is None and workers is not None and (not only or any(cid.startswith(o) for o in only)):
            started = time.time()
            verdict = dataclasses.replace(run_criterion(criterion, rules, workers=workers), criterion=cid)
            cache.put(key, dataclasses.asdict(verdict))
            cache.flush()  # keep every finished result, whatever happens to the next criterion
            entry = cache.get(key)
            print(f"{cid}: {verdict.outcome} ({verdict.seeds} seeds, {time.time() - started:.0f} s)", flush=True)
        if entry is None:
            raise KeyError(cid)
        verdicts.append((Verdict(**entry), None))
    return render(verdicts, NOT_BUILT, rules, engine)


def scripted_cell(scripted: dict) -> str:
    """Each arm's scripted acts not made, of all it scripted, and why: "—" for a criterion that scripts none."""
    if not scripted:
        return "—"
    cells = []
    for arm, c in scripted.items():
        why = ", ".join(f"{n} {s}" for s, n in c.items() if s != "made" and n)
        cells.append(f"{arm} {sum(c.values()) - c['made']} of {sum(c.values())}" + (f" ({why})" if why else ""))
    return "; ".join(cells)


def render(verdicts, not_built, rules, hash_: str) -> str:
    lines = [
        "# Phase C ensemble record",
        "",
        "Generated by `tools/ensemble_record.py`; do not edit. Each criterion's two arms ran on the same keyed",
        "seeds (`M3.D.4a`), stopped adaptively (`M17.A.1`, `M13.4`): `UNDETERMINED` at the seed cap does **not**",
        "pass. Every verdict carries its class (`M11.5`). **A premise passing confirms that the code renders the",
        "spec; it is not a finding about families.** The constants are the Phase C freeze of 2026-10-07.",
        "",
        f"code_hash: {hash_}",
        "",
        "Rules: " + ", ".join(f"`{k}` = {v}" for k, v in rules.items()) + ".",
        "",
        "| Criterion | Class | Verdict | Seeds | Readouts (difference treatment − baseline; ± half-width; p after Holm where tested) | Scripted acts not made, of all scripted (why) | Fallback rate | Flags |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for v, _ in verdicts:
        parts = []
        for r in v.readouts:
            text = f"`{r['readout']}` {r['mean_difference']:+.3g} ± {r['half_width']:.2g}"
            if r.get("report_only"):
                text += " (reported only)"
            elif r["direction"] == 0:
                side = {1: " (one-sided: the treatment no lower than the baseline less the bound)",
                        -1: " (one-sided: the treatment no higher than the baseline plus the bound)"}.get(r.get("bound_side"), "")
                text += (f", equivalence bound {r['equivalence_bound']:.3g}{side}, "
                         f"{'within' if r['within_bound'] else 'outside'}")
            else:
                text += f", p {r['p']:.3g}, {'holds' if r.get('holds') else 'does not hold'}"
            parts.append(text)
        flags = list(v.flagged) + [f"move never occurred: {m}" for m in v.missing_moves]
        lines.append(f"| `{v.criterion}` | {v.cls} | **{v.outcome}** | {v.seeds} | {'; '.join(parts)} | "
                     f"{scripted_cell(v.scripted)} | {v.fallback_rate:.2f} | {'; '.join(flags) or '—'} |")
    lines += ["", "## Not built in Phase C", ""]
    lines += [f"- `{cid}` — {why}." for cid, why in not_built.items()]
    lines += ["", "## Fallback rate by person (`M11.D.18`)", ""]
    for v, _ in verdicts:
        worst = sorted(v.fallback_by_person.items(), key=lambda kv: -kv[1])[:3]
        lines.append(f"- `{v.criterion}`: " + ", ".join(f"{p} {r:.2f}" for p, r in worst))
    lines += ["", "## Machine-readable", "", "```json",
              json.dumps([{"criterion": v.criterion, "class": v.cls, "outcome": v.outcome, "seeds": v.seeds,
                           "readouts": v.readouts, "fallback_rate": v.fallback_rate, "flags": v.flagged,
                           "missing_moves": v.missing_moves, "scripted": v.scripted} for v, _ in verdicts], indent=1, default=float),
              "```", ""]
    return "\n".join(lines)


def main() -> int:
    from tools.record_cache import Cache

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    cache = Cache(CACHE_NAME)
    try:
        text = build(cache, workers=args.workers, only=args.only)
    except KeyError as missing:
        raise SystemExit(f"no result for {missing} on the current code: run without --only, or name it") from None
    cache.save()  # every criterion's entry was used; entries for older code go
    RECORD.write_text(text, encoding="utf-8")
    print(f"wrote {RECORD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
