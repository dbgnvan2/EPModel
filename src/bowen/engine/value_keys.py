"""The key of a learned value, ``band|position|act|where``: one format for every reader and writer.

Purpose: the policy reads learned values (``policy.value_key``), and the learner and the I-POSITION abort read and write
         them; a key built by hand in one place silently stops matching when the format changes (learning-qa review
         2026-10-14).
Spec:    docs/bowen_agent_model_spec_v2.md#M4.D.2, #M4.D.6f
Tests:   tests/bowen/test_cutoff_and_focus.py::test_m4d6f_the_key_written_is_the_key_the_policy_reads
"""

from __future__ import annotations

INSIDE, OUTSIDE = "inside", "outside"  # policy.triangle_position's two answers (M4.D.2)
POSITIONS = (INSIDE, OUTSIDE)


def learned_value_key(band: str, position: str, kind: str, where: str) -> str:
    return f"{band}|{position}|{kind}|{where}"


def key_parts(key: str) -> tuple[str, str, str, str]:
    """A key's (band, position, act, where), the inverse of ``learned_value_key``."""
    band, position, act, where = key.split("|")
    return band, position, act, where


def act_of(key: str) -> str:
    return act_and_where(key)[0]


def act_and_where(key: str) -> tuple[str, str]:
    _band, _position, act, where = key_parts(key)
    return act, where
