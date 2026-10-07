"""Suite-wide check that no test changed the repository's real artifacts.

Purpose: fingerprint the committed config, spec and model source at session
         start and fail the session if any changed by its end (P28: detect an
         escape rather than assume the guard held).
Spec:    docs/bowen_agent_model_spec_v2.md#M11.D.7
Tests:   tests/bowen/test_test_hygiene.py::test_m11d7_fingerprint_detects_a_changed_artifact
"""

import hashlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WATCHED = ("config/bowen", "src/bowen", "docs/bowen_agent_model_spec_v2.md", "tests")


def fingerprint(repo: Path = REPO) -> dict[str, str]:
    files = []
    for entry in WATCHED:
        path = repo / entry
        files += sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts) if path.is_dir() else [path]
    return {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


@pytest.fixture(scope="session", autouse=True)
def m11d7_repository_artifacts_unchanged():
    before = fingerprint()
    yield
    after = fingerprint()
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    assert not changed, f"tests changed repository files: {changed}"
