"""Toy arms for the ensemble runner's own tests — importable, so worker processes can run them."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from src.bowen.ensemble.runner import ArmResult


def noisy(arm: str, seed: int, settings: dict) -> ArmResult:
    """A readout of seed-keyed noise plus ``effect`` in the treatment arm."""
    rng = np.random.default_rng([seed, 99])
    value = float(rng.normal(0.0, settings.get("noise", 1.0)))
    extra = float(rng.normal(0.0, 1.0))  # the same draw in both arms, so swapping which arm is noisy is exact
    if arm == "treatment":
        value += settings.get("effect", 0.0) + settings.get("extra_noise", 0.0) * extra
    else:
        value += settings.get("baseline_extra_noise", 0.0) * extra
    moves = Counter({"CUTOFF": settings.get("cutoffs", 1)})
    selections = Counter({"ravi": 10})
    fallbacks = Counter({"ravi": settings.get("fallbacks", 0)})
    return ArmResult({"x": value, "bounded": settings.get("bounded", 0.5)}, moves, fallbacks, selections)
