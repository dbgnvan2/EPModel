"""M10.C.4's twelve corpus magnitudes stay checks, never config keys.

Purpose: fail when a config key the engine can read names one of the twelve, or when an engine module imports the
         checks module that holds them.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.D.12, #M10.C.4b
Tests:   this file
"""

from __future__ import annotations

import re
from pathlib import Path

from src.bowen.scenario.constants import RETIRED, SCHEMA
from tools.corpus_checks import MAGNITUDES, offending_keys

REPO = Path(__file__).resolve().parents[1]
KEY_ROW = re.compile(r"^\|\s*`([a-z0-9_]+)`\s*\|", re.M)


def _config_keys() -> set[str]:
    keys = set(SCHEMA) | set(RETIRED)
    for path in (REPO / "config" / "bowen").glob("constants*.md"):
        keys |= set(KEY_ROW.findall(path.read_text(encoding="utf-8")))
    return keys


def test_m11d12_corpus_magnitudes_are_not_config_keys():
    """Every key the parser accepts, and every key row in config/bowen/constants*.md, is matched against the names
    of the twelve. Mutation: a key `species_median_basic_level` planted among them is flagged."""
    keys = _config_keys()
    assert len(MAGNITUDES) == 12 and len(keys) > 100
    assert offending_keys(keys | {"species_median_basic_level"}) != []
    assert offending_keys(keys) == []


def test_m10c4b_engine_does_not_import_the_checks_module():
    """M10.C.4b: no engine code path imports the checks module. Mutation: an import of it in a src/bowen file."""
    importing = sorted(p.relative_to(REPO).as_posix() for p in (REPO / "src" / "bowen").rglob("*.py")
                       if re.search(r"\bcorpus_checks\b", p.read_text(encoding="utf-8")))
    assert importing == []
