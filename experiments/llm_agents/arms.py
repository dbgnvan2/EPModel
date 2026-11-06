"""The five arms and each run's fixed permutations.

Purpose: say what each arm shows the agent (§2), and fix per seed the person tokens, A1's opaque act tokens and A3's
         derangement of the plain labels inside each legality group (§3.3).
Spec:    docs/plan_v3_llm_agent_experiment.md#2, #3.3
Tests:   tests/experiments/llm_agents/test_prompts.py::test_lx5_labels_differ_between_seeds,
         tests/experiments/llm_agents/test_prompts.py::test_lx6_shuffle_is_a_derangement_within_groups

The permutations come from a generator seeded by ``sha256(purpose, seed)``, never from the engine's draw service, so
building them changes no engine draw (LX.19). The same seed gives the same permutations in every process.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Mapping

from experiments.llm_agents.config import GROUPS, Label

ARMS = ("A0", "A1", "A2", "A3", "A4")


@dataclass(frozen=True)
class ArmSpec:
    name: str
    llm: bool          # an LLM decides for the family members
    opaque: bool       # act labels are opaque tokens
    shuffled: bool     # plain labels sit on other acts' mechanics
    feedback: bool     # the agent sees its tension and its change


SPECS: Mapping[str, ArmSpec] = {
    "A0": ArmSpec("A0", llm=False, opaque=False, shuffled=False, feedback=False),
    "A1": ArmSpec("A1", llm=True, opaque=True, shuffled=False, feedback=True),
    "A2": ArmSpec("A2", llm=True, opaque=False, shuffled=False, feedback=True),
    "A3": ArmSpec("A3", llm=True, opaque=False, shuffled=True, feedback=True),
    "A4": ArmSpec("A4", llm=True, opaque=False, shuffled=False, feedback=False),
}


def rng(purpose: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"experiments.llm_agents|{purpose}|{seed}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def derangement(items: list[str], r: random.Random) -> dict[str, str]:
    """A permutation with no fixed point; a group of one keeps its own (§3.3)."""
    if len(items) < 2:
        return {i: i for i in items}
    while True:
        shuffled = items[:]
        r.shuffle(shuffled)
        if all(a != b for a, b in zip(items, shuffled)):
            return dict(zip(items, shuffled))


@dataclass(frozen=True)
class Permutations:
    """One run's fixed mappings. ``label_of[kind]`` is the kind whose plain label the agent sees on ``kind``."""

    person_order: tuple[str, ...]       # person ids in the order their tokens P1, P2, ... are given
    act_token: Mapping[str, str]        # kind -> K-token (A1)
    label_of: Mapping[str, str]         # kind -> kind whose label is shown (identity outside A3)

    def as_record(self) -> dict:
        return {"person_order": list(self.person_order), "act_token": dict(sorted(self.act_token.items())),
                "label_of": dict(sorted(self.label_of.items()))}


def permutations(arm: str, seed: int, people: list[str], labels: Mapping[str, Label]) -> Permutations:
    """Purpose: the run's tokens and A3's shuffle, fixed by (arm, seed)."""
    spec = SPECS[arm]
    order = sorted(people)
    rng("person_tokens", seed).shuffle(order)
    kinds = sorted(labels)
    tokens = [f"K{i}" for i in range(1, len(kinds) + 1)]
    rng("act_tokens", seed).shuffle(tokens)
    label_of = {k: k for k in kinds}
    if spec.shuffled:
        r = rng("a3_shuffle", seed)
        for group in GROUPS:
            label_of.update(derangement(sorted(k for k in kinds if labels[k].group == group), r))
    return Permutations(tuple(order), dict(zip(kinds, tokens)), label_of)
