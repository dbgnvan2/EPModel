"""Guards on the arXiv sweep script's deduplication.

Purpose: keep `papers/sweep.py` from re-proposing papers already in the
catalogue.
Spec:    none — this is project tooling, not the model.
Tests:   this file

The bug this file exists for, found 2026-09-21 while reading the script before
committing it: the arXiv API returns *versioned* ids ("2609.17331v1") while
`papers/INDEX.md` carries *unversioned* ones ("2609.17331"), and the membership
test compared the two directly. It could therefore never be true, so every
already-catalogued paper was re-proposed on every weekly run — a silent failure,
because the output still looked like a plausible candidate list.

Same class as P6: a check that claims to filter against an artefact, never
verified against the artefact's actual form.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_PATH = REPO_ROOT / "papers" / "sweep.py"


def _load_sweep():
    spec = importlib.util.spec_from_file_location("sweep_under_test", SWEEP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = _load_sweep()


def test_base_id_strips_the_version_suffix():
    assert sweep.base_id("2609.17331v1") == "2609.17331"
    assert sweep.base_id("2609.17331v12") == "2609.17331"
    assert sweep.base_id("2609.17331") == "2609.17331"
    # Four-digit sequence numbers are still used by older ids.
    assert sweep.base_id("2509.1805v2") == "2509.1805"


def test_catalogue_ids_are_unversioned_whichever_form_the_index_uses(tmp_path):
    index = tmp_path / "INDEX.md"
    index.write_text(
        "| a.pdf | T | A | 2026 | 2609.17331 | ARCH | digested |\n"
        "| b.pdf | T | A | 2026 | 2603.11084v2 | METHOD | new |\n",
        encoding="utf-8",
    )
    assert sweep.already_catalogued(index) == {"2609.17331", "2603.11084"}


def test_missing_index_yields_an_empty_set(tmp_path):
    assert sweep.already_catalogued(tmp_path / "nope.md") == set()


def test_main_drops_a_catalogued_paper_returned_with_a_version_suffix(tmp_path, capsys):
    """The regression, exercised through main() so it covers the call site.

    Reverting either base_id() or its use in main() turns this red.
    """
    known = sweep.already_catalogued(REPO_ROOT / "papers" / "INDEX.md")
    assert known, "the real catalogue should not be empty"
    catalogued = sorted(known)[0]

    hits = [
        {  # already in INDEX.md, returned versioned as the API returns it
            "id": f"{catalogued}v1",
            "title": "Already catalogued",
            "published": "2099-01-01",
            "authors": "A",
            "abstract": "",
            "cats": "cs.AI",
        },
        {  # genuinely new
            "id": "2699.99999v3",
            "title": "Brand new",
            "published": "2099-01-01",
            "authors": "B",
            "abstract": "",
            "cats": "cs.AI",
        },
    ]
    out = tmp_path / "candidates.json"
    argv = ["sweep.py", "--days", "100000", "--out", str(out)]
    with mock.patch.object(sweep, "fetch", lambda q: b""), \
            mock.patch.object(sweep, "parse", lambda b: [dict(h) for h in hits]), \
            mock.patch.object(sweep.time, "sleep", lambda s: None), \
            mock.patch.object(sweep.sys, "argv", argv):
        assert sweep.main() == 0
    capsys.readouterr()

    written = json.loads(out.read_text(encoding="utf-8"))
    ids = {c["id"] for c in written}
    assert ids == {"2699.99999v3"}, (
        "the catalogued paper was re-proposed; dedupe is comparing a versioned "
        "id against an unversioned catalogue"
    )


def test_seen_is_keyed_unversioned_so_two_versions_are_one_candidate(tmp_path, capsys):
    """Two facet queries returning different versions of the same new paper
    must yield one candidate, not two."""
    hits_v1 = [{"id": "2698.11111v1", "title": "P", "published": "2099-01-01",
                "authors": "A", "abstract": "", "cats": "cs.AI"}]
    hits_v2 = [{"id": "2698.11111v2", "title": "P", "published": "2099-01-01",
                "authors": "A", "abstract": "", "cats": "cs.AI"}]
    batches = iter([hits_v1, hits_v2] + [[]] * 10)
    out = tmp_path / "candidates.json"
    argv = ["sweep.py", "--days", "100000", "--out", str(out)]
    with mock.patch.object(sweep, "fetch", lambda q: b""), \
            mock.patch.object(sweep, "parse", lambda b: [dict(h) for h in next(batches)]), \
            mock.patch.object(sweep.time, "sleep", lambda s: None), \
            mock.patch.object(sweep.sys, "argv", argv):
        assert sweep.main() == 0
    capsys.readouterr()

    written = json.loads(out.read_text(encoding="utf-8"))
    assert len(written) == 1, "the same paper was counted twice under two versions"


def test_the_real_catalogue_contains_no_versioned_ids():
    """If a versioned id is ever written into INDEX.md, the regex still reduces
    it, so the two forms cannot drift apart again."""
    known = sweep.already_catalogued(REPO_ROOT / "papers" / "INDEX.md")
    assert all(sweep.base_id(k) == k for k in known)
