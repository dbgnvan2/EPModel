"""The four signal channels a move carries and the four cues each one carries (T-202).

Purpose: the data a move's ``signals`` field holds when the social cues are on: four signal channels — words, tone,
         face, posture — each a cue vector over attention, approval, expectations and distress, each cue in [−1, 1],
         the threat pole negative. The engine sets it at act (``cues.sender_signals``); no policy or script sets it.
Spec:    docs/bowen_agent_model_spec_v2.md#M1.F.1c, #M4.C.1e
Tests:   tests/bowen/test_social_cues.py::test_m1f1c_sc3_move_carries_four_channels_when_on,
         tests/bowen/test_social_cues.py::test_m1f1c_sc3_signals_reject_values_outside_the_unit_interval

The words *signal channel* are kept apart from `M4.D.1a`'s channels of selection (``events.Channel``), and the cue
*attention* from `M4.C.8`'s ``events.Attention`` (what an event directs the receiver's attention at). Kept in its own
module so that ``events`` can name the type without importing the cue mechanics.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class Cue(enum.Enum):
    """The four cues, in the order every cue vector holds them (`M4.C.1e`). The list is open (`KS10.1`)."""

    ATTENTION = "attention"
    APPROVAL = "approval"
    EXPECTATIONS = "expectations"
    DISTRESS = "distress"


CUES: tuple[Cue, ...] = tuple(Cue)
NEUTRAL: tuple[float, ...] = (0.0,) * len(CUES)


class SignalChannel(enum.Enum):
    """The four signal channels (`M1.F.1c`): words are chosen; tone, face and posture leak (`M4.C.13`)."""

    WORDS = "words"
    TONE = "tone"
    FACE = "face"
    POSTURE = "posture"


NONVERBAL: tuple[SignalChannel, ...] = (SignalChannel.TONE, SignalChannel.FACE, SignalChannel.POSTURE)


@dataclass(frozen=True)
class Signals:
    """Purpose: one move's four signal channels, each a cue vector in [−1, 1] per cue.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.F.1c
    Tests:   tests/bowen/test_social_cues.py::test_m1f1c_sc3_signals_reject_values_outside_the_unit_interval
    """

    words: tuple[float, ...] = NEUTRAL
    tone: tuple[float, ...] = NEUTRAL
    face: tuple[float, ...] = NEUTRAL
    posture: tuple[float, ...] = NEUTRAL

    def __post_init__(self) -> None:
        for channel in SignalChannel:
            vector = getattr(self, channel.value)
            if len(vector) != len(CUES):
                raise ValueError(f"{channel.value}: a cue vector holds {len(CUES)} cues, got {len(vector)}")
            if any(not -1.0 <= v <= 1.0 for v in vector):
                raise ValueError(f"{channel.value}: every cue is in [-1, 1], got {vector}")

    def channel(self, channel: SignalChannel) -> tuple[float, ...]:
        return getattr(self, channel.value)
