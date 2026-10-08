"""The counterfeit detector — which axis of a person's position failed.

Purpose: report, for a person, whether their position fails on the outward axis
         (the forceful declarer), the inward axis (the compliant accommodator),
         both, or neither — never one scalar.
Spec:    docs/bowen_agent_model_spec_v2.md#M5.F.2b, #M1.A.9a, #M11.C.19
Tests:   tests/bowen/test_outside_ness.py::test_m5f2b_detector_reports_which_axis_failed

"Rugged individualism and compliance, therefore, are two sides of the same coin"
(`FE03.1`, `[K]`): the two counterfeits are equal magnitudes of the same force in
opposite directions, and they need opposite corrections. A scalar score lands them
at the same value. This reads the two axes against the gate's two thresholds
(`M5.C.1`, revision 12 P9) and keeps them apart.

A readout. The engine never imports it (`M11.D.22`).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bowen.engine.objects import Person
from src.bowen.engine.outside_ness import axes
from src.bowen.engine.params import EngineParams

OUTWARD = "outward"
INWARD = "inward"


@dataclass(frozen=True)
class CounterfeitReading:
    outward: float
    inward: float
    failed: frozenset[str]

    @property
    def counterfeit(self) -> bool:
        return bool(self.failed)


def read_counterfeit(person: Person, params: EngineParams) -> CounterfeitReading:
    """Purpose: which axis of a person's position failed, against the gate's two thresholds.
    Spec:    docs/bowen_agent_model_spec_v2.md#M5.F.2b
    Tests:   tests/bowen/test_outside_ness.py::test_m5f2b_detector_reports_which_axis_failed
    """
    outward, inward = axes(person)
    failed = set()
    if outward > params.outside_ness_threshold_outward:
        failed.add(OUTWARD)
    if inward > params.outside_ness_threshold_inward:
        failed.add(INWARD)
    return CounterfeitReading(outward, inward, frozenset(failed))
