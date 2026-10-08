"""The policy's declared tie-break and fallback rules (`M4.D.1f`).

Purpose: hold the two rules ``config/bowen/policy.md`` declares, each checked against
         the forms the policy implements, so a rule cannot be declared and ignored.
Spec:    docs/bowen_agent_model_spec_v2.md#M4.D.1f
Tests:   tests/bowen/test_policy.py::test_m4d1f_rules_come_from_config
"""

from __future__ import annotations

from dataclasses import dataclass

# The forms the policy implements. A rule naming anything else is refused at load.
TIE_BREAKS = frozenset({"equal_probability"})
FALLBACKS = frozenset({"hold"})


@dataclass(frozen=True)
class PolicyRules:
    tie_break: str
    fallback: str

    def __post_init__(self) -> None:
        if self.tie_break not in TIE_BREAKS:
            raise ValueError(f"tie_break {self.tie_break!r} is not implemented; one of {sorted(TIE_BREAKS)}")
        if self.fallback not in FALLBACKS:
            raise ValueError(f"fallback {self.fallback!r} is not implemented; one of {sorted(FALLBACKS)}")
