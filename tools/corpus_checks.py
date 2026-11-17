"""The twelve corpus magnitudes of M10.C.4, kept as checks and never as parameters.

Purpose: the checks module M10.C.4b names. Each entry is a bound or a shape stated in the corpus, read only by
         acceptance criteria and readout comparisons. It lives in tools/, outside the engine and the engine hash;
         no module under src/bowen may import it, and no config key may carry one of these quantities.
         Each entry names the quantity, so the M11.D.12 guard matches a config key by what it names, not by its
         value: a guard on value alone would flag every ordinary 5 or 40 in the register.
Spec:    docs/bowen_agent_model_spec_v2.md#M10.C.4, #M10.C.4b, #M11.D.12
Tests:   tests/test_corpus_checks.py::test_m11d12_corpus_magnitudes_are_not_config_keys,
         tests/test_corpus_checks.py::test_m10c4b_engine_does_not_import_the_checks_module
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Magnitude:
    name: str
    value: str  # as M10.C.4 states it; these are bounds and shapes, so most are not one number
    source: str
    key_pattern: str  # a config key whose name matches this carries the quantity (M11.D.12)


MAGNITUDES: tuple[Magnitude, ...] = (
    Magnitude("species median basic_level", "about 40, below M1.A.3's transition", "FE03.5",
              r"species|median_(basic_)?level"),
    Magnitude("band edges", "0-10 very poor; >60 well; >70 very well; 100 unreachable", "FE03.5",
              r"band_edge|level_band|very_poor|very_well|unreachable|max_basic_level"),
    Magnitude("pseudo-self transfer, worked", "35 + 35 -> 55 + 15, conserved exactly", "FE04.1",
              r"pseudo_self_transfer_(worked|example|points|amount)"),
    Magnitude("per-generation step, typical", "< 5 points", "FE08.7",
              r"per_generation|generation_step|step_per_generation|generational_(step|change)"),
    Magnitude("per-generation step, consequential", "5-10 points", "FE07.7",
              r"consequential_step|per_generation|generation_step"),
    Magnitude("parent-to-child bound", "never 30 points", "FE07.7",
              r"parent_child|parent_to_child|child_level_gap|max_level_gap"),
    Magnitude("quantum jump", "about 10 points per generation over two generations, three conjunctive conditions",
              "FE08.1", r"quantum|jump"),
    Magnitude("full traversal", "3 generations (quantum jump) to 5-10 (typical), symmetric in both directions",
              "FE08.2", r"traversal|generations_to"),
    Magnitude("generation length", "about 25 years", "FE08.2",
              r"generation_(length|years|span)|years_per_generation"),
    Magnitude("sibling-position decay", "a gap of 5 years or more reduces predictability", "FE10.7",
              r"sibling_(gap|spacing|position)|birth_order"),
    Magnitude("treatability floor", "upper two-thirds of the schizophrenic range", "FE11.10",
              r"treatab|schizophren"),
    Magnitude("borrowed-gain gap, worked", "apparent 40-45, true 25-30, revealed by a matriarch's death", "FE04.8",
              r"borrowed|apparent_level|true_level|matriarch"),
)


def offending_keys(keys) -> list[tuple[str, str]]:
    """Each (key, magnitude name) where a config key's name names one of the twelve."""
    return sorted((key, m.name) for key in keys for m in MAGNITUDES if re.search(m.key_pattern, key))
