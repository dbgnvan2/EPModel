"""Counter-based, event-keyed random draws.

Purpose: make every stochastic draw a pure function of the run seed and a
         structural event key, so two arms draw the same number for the same event.
Spec:    docs/bowen_agent_model_spec_v2.md#M3.D.4, #M3.D.4a, #M3.D.4b, #M3.D.4c
Tests:   tests/bowen/test_draws.py

How a draw is made, fixed and documented as M3.D.4a requires:

1. The event key is encoded canonically: the draw class, then each component
   as ``name=type:value``, joined by ``|``, as UTF-8.
2. BLAKE2b with a 32-byte digest turns that encoding into four little-endian
   unsigned 64-bit words. This is the Philox **counter**. Python's built-in
   ``hash()`` is salted per process and is never used.
3. The run seed is the Philox **key** (``[seed, 0]``).
4. A fresh Philox generator at that key and counter yields ``n`` raw 64-bit
   words; each becomes a uniform in [0, 1) as ``(word >> 11) * 2**-53``.
   Raw words are used rather than NumPy's ``Generator.random`` so the mapping
   is ours and does not depend on a NumPy convenience method.

No generator outlives a draw, so the engine holds no mutable generator state.
The per-run cache (M3.D.4c) holds drawn values, not generator state, and lives
on a ``DrawService`` instance, never at module level (M3.D.4).
"""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from src.bowen.engine.identifiers import PersonId

_MAX_SEED = 2**64
_UNIT = 2.0**-53


class Keying(enum.Enum):
    """M3.D.4b's keying kinds — what counts as *the same event* in two arms."""

    SLOT = "slot"            # tick, actor, purpose, index — a partner enters only through state
    DYAD = "dyad"            # a different partner is a different chance event
    COARSE = "coarse"        # deliberately coarse, marked as such
    PER_PERSON = "per person"


@dataclass(frozen=True)
class DrawClass:
    name: str
    where: str
    components: tuple[str, ...]
    keying: Keying


def _cls(name: str, where: str, components: str, keying: Keying) -> DrawClass:
    return DrawClass(name, where, tuple(c.strip() for c in components.split(",")), keying)


# M3.D.4b's table, one entry per row and in its order. The test
# `test_m3d4b_table_matches_the_spec` compares this with the spec's table, so
# the two cannot drift apart silently.
DRAW_CLASSES: Mapping[str, DrawClass] = MappingProxyType(
    {
        c.name: c
        for c in (
            _cls("move_selection", "M4.D.1", "tick, actor, purpose, index", Keying.SLOT),
            _cls("channel_mixing_noise", "M4.D.1a", "tick, actor, purpose, index", Keying.SLOT),
            _cls("per_hop_fidelity", "M1.F.4", "tick, actor, partner, purpose, index", Keying.DYAD),
            _cls("edge_latency", "M1.B.11", "tick, actor, partner, purpose, index", Keying.DYAD),
            _cls("receiver_appraisal_noise", "M4.C", "tick, actor, partner, purpose, index", Keying.DYAD),
            _cls("witness_overhearing", "M1.F.1b", "tick, actor, partner, purpose, index", Keying.DYAD),
            _cls("exogenous_spell", "M1.F.6", "family, spell class, occurrence index", Keying.COARSE),
            _cls("symptom_onset", "M4.C.3", "tick, person, channel", Keying.PER_PERSON),
            _cls("mortality", "M7.C.1", "slow tick, person, purpose", Keying.PER_PERSON),
            _cls("tie_break_fallback", "M4.D.1f", "tick, actor, purpose, index", Keying.SLOT),
        )
    }
)

# What each component may hold. Only structural identity is admitted: counts of
# ticks and indices, stable identifiers, and labels. A float is never a key
# component, which is how M3.D.4a's ban on state quantities is enforced by type.
_INTEGER_COMPONENTS = {"tick", "slow tick", "index", "occurrence index"}
_PERSON_COMPONENTS = {"actor", "partner", "person"}
_LABEL_COMPONENTS = {"purpose", "family", "spell class", "channel"}


class DrawKeyError(ValueError):
    """A draw key is undeclared, incomplete, or carries something other than structural identity."""


@dataclass(frozen=True)
class DrawKey:
    """Purpose: the canonical key of one chance event.
    Spec:    docs/bowen_agent_model_spec_v2.md#M3.D.4a, #M3.D.4b
    Tests:   tests/bowen/test_draws.py::test_m3d4a_key_rejects_state_quantities
    """

    draw_class: str
    components: tuple[tuple[str, object], ...]

    @classmethod
    def make(cls, draw_class: str, **components: object) -> "DrawKey":
        if draw_class not in DRAW_CLASSES:
            raise DrawKeyError(f"undeclared draw class {draw_class!r} (M3.D.4b)")
        declared = DRAW_CLASSES[draw_class].components
        given = {name.replace("_", " "): value for name, value in components.items()}
        if set(given) != set(declared):
            raise DrawKeyError(
                f"{draw_class} needs exactly {list(declared)}, got {sorted(given)}"
            )
        for name, value in given.items():
            _check_component(draw_class, name, value)
        return cls(draw_class, tuple((name, given[name]) for name in declared))

    def encode(self) -> bytes:
        parts = [self.draw_class]
        for name, value in self.components:
            if isinstance(value, PersonId):
                parts.append(f"{name}=person:{value.value}")
            elif isinstance(value, enum.Enum):
                parts.append(f"{name}=label:{value.value}")
            elif isinstance(value, int):
                parts.append(f"{name}=int:{value}")
            else:
                parts.append(f"{name}=label:{value}")
        return "|".join(parts).encode("utf-8")


def _check_component(draw_class: str, name: str, value: object) -> None:
    where = f"{draw_class}.{name}"
    if name in _INTEGER_COMPONENTS:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DrawKeyError(f"{where} must be a non-negative int, got {value!r}")
    elif name in _PERSON_COMPONENTS:
        if not isinstance(value, PersonId):
            raise DrawKeyError(f"{where} must be a PersonId, got {value!r}")
    elif name in _LABEL_COMPONENTS:
        if isinstance(value, enum.Enum):
            value = value.value
        if not isinstance(value, str) or not value or "|" in value or "=" in value:
            raise DrawKeyError(f"{where} must be a non-empty label, got {value!r}")
    else:  # pragma: no cover — guarded by the declared table
        raise DrawKeyError(f"{where}: no rule for this component")


def counter_for(key: DrawKey) -> np.ndarray:
    """Purpose: the fixed, documented key-to-counter function (M3.D.4a).
    Spec:    docs/bowen_agent_model_spec_v2.md#M3.D.4a
    Tests:   tests/bowen/test_draws.py::test_m3d4a_draws_agree_across_processes
    """
    digest = hashlib.blake2b(key.encode(), digest_size=32).digest()
    return np.frombuffer(digest, dtype="<u8").astype(np.uint64)


class RepeatedDrawKey(RuntimeError):
    """M3.D.4c — a key was queried twice in a debug run."""


class DrawService:
    """Purpose: serve keyed uniforms for one run, caching each key's values.
    Spec:    docs/bowen_agent_model_spec_v2.md#M3.D.4a, #M3.D.4c
    Tests:   tests/bowen/test_draws.py::test_m3d4c_repeated_key_raises_in_debug
    """

    def __init__(self, seed: int, *, debug: bool = False) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < _MAX_SEED:
            raise ValueError(f"seed must be an int in [0, 2**64), got {seed!r}")
        self._key = np.array([seed, 0], dtype=np.uint64)
        self._debug = debug
        self._cache: dict[DrawKey, tuple[float, ...]] = {}

    def uniforms(self, key: DrawKey, n: int) -> tuple[float, ...]:
        """A fixed number of uniforms in [0, 1) for one key (inverse transform only)."""
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"n must be a positive int, got {n!r}")
        cached = self._cache.get(key)
        if cached is not None:
            if self._debug:
                raise RepeatedDrawKey(f"draw key queried twice: {key.encode().decode()}")
            if len(cached) != n:
                raise ValueError(f"key drawn earlier with n={len(cached)}, now n={n}")
            return cached
        generator = np.random.Philox(key=self._key, counter=counter_for(key))
        values = tuple(float((int(word) >> 11) * _UNIT) for word in generator.random_raw(n))
        self._cache[key] = values
        return values

    def uniform(self, key: DrawKey) -> float:
        return self.uniforms(key, 1)[0]

    def categorical(self, key: DrawKey, weights: Sequence[float]) -> int:
        """One uniform, inverted through the cumulative weights. Weights need not sum to one."""
        total = float(sum(weights))
        if not weights or total <= 0 or any(w < 0 for w in weights):
            raise ValueError("weights must be non-negative with a positive sum")
        target = self.uniform(key) * total
        running = 0.0
        for index, weight in enumerate(weights):
            running += weight
            if target < running:
                return index
        return len(weights) - 1  # only reachable through rounding at the top edge
