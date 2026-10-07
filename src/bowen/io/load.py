"""Reading configuration files from disk.

Purpose: the one place config files are read; everything else takes text.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.D.1, #M16.B.1
Tests:   tests/bowen/test_config.py::test_m101_repository_constants_load_and_are_graded
"""

from __future__ import annotations

from pathlib import Path

from src.bowen.engine.events import EventKinds
from src.bowen.scenario.constants import Constants, parse_constants
from src.bowen.scenario.event_kinds import parse_event_kinds
from src.bowen.scenario.family import FamilyInstance, build_family

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config" / "bowen"


def read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_constants(path: Path = CONFIG_DIR / "constants.md") -> Constants:
    """Purpose: read and parse the constants register.
    Spec:    docs/bowen_agent_model_spec_v2.md#M10.B.1
    Tests:   tests/bowen/test_config.py::test_m101_repository_constants_load_and_are_graded
    """
    return parse_constants(read_text(path), source=str(path))


def load_event_kinds(path: Path = CONFIG_DIR / "event_kinds.md") -> EventKinds:
    """Purpose: read and parse the event-kind vocabulary.
    Spec:    docs/bowen_agent_model_spec_v2.md#M10.B.1
    Tests:   tests/bowen/test_events.py::test_m10b1_event_kinds_come_from_config
    """
    return parse_event_kinds(read_text(path), source=str(path))


def load_family(path: Path = CONFIG_DIR / "family_reduced.md", constants: Constants | None = None) -> FamilyInstance:
    """Purpose: read and build a family declaration.
    Spec:    docs/bowen_agent_model_spec_v2.md#M2.1
    Tests:   tests/bowen/test_family.py::test_m21_family_declared_in_markdown
    """
    return build_family(read_text(path), constants or load_constants(), source=str(path))
