"""Reading configuration files from disk.

Purpose: the one place config files are read; everything else takes text.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.D.1, #M16.B.1
Tests:   tests/bowen/test_config.py::test_m101_repository_constants_load_and_are_graded
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from src.bowen.engine.events import ACT_CLASSES, EventKinds, EventSize
from src.bowen.policy.rules import PolicyRules
from src.bowen.scenario.constants import RETIRED, SCHEMA, Constants, parse_constants
from src.bowen.scenario.event_kinds import parse_event_kinds
from src.bowen.scenario.family import FamilyInstance, build_family, partners_file
from src.bowen.scenario.policy_rules import parse_policy_rules
from src.bowen.scenario.scripted_source import ScriptedSource, build_script

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
    kinds = parse_event_kinds(read_text(path), source=str(path))
    reach = load_nodal_reach(kinds)
    attribution, focus = load_attribution(kinds)
    classes = load_act_classes(kinds)
    contacts = load_contact_classes(kinds)
    return dataclasses.replace(kinds, nodal=frozenset(reach), nodal_reach=reach, soothing=load_soothing(kinds),
                               displacing=load_displacing(kinds),
                               steering=load_steering(kinds), attribution=attribution, focus=focus, **classes,
                               negative_contacts=contacts["negative"], positive_contacts=contacts["positive"],
                               rebuffs=contacts["rebuff"], sizes=load_event_sizes(kinds))


CONTACT_CLASSES = ("negative", "positive", "rebuff")
SIZE_COLUMNS = ("kind", "who", "item", "lcu", "weeks", "source")


def load_contact_classes(kinds: EventKinds, path: Path = CONFIG_DIR / "contact_classes.md") -> dict[str, frozenset[str]]:
    """Purpose: read which acts are negative or positive contact across a tie, and the rebuffs the negative-run
    readout also counts; every set must be declared, once, with declared moves only, and the sets must not overlap.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.D.2b, #M1.D.2c, #M11.C.48
    Tests:   tests/bowen/test_cutoff_stock.py::test_m1d2b_contact_classes_come_from_config
    """
    from src.bowen.scenario.config_parse import ConfigError, parse_table_document

    doc = parse_table_document(read_text(path), columns=("set", "acts", "used by"), metadata_keys=frozenset(),
                               source=str(path))
    sets = {row["set"]: frozenset(a.strip() for a in row["acts"].split(",") if a.strip()) for row in doc.rows}
    if set(sets) != set(CONTACT_CLASSES) or len(sets) != len(doc.rows):
        raise ConfigError(f"{path}: the sets must be exactly {sorted(CONTACT_CLASSES)}, each once; found {sorted(sets)}")
    unknown = sorted(a for acts in sets.values() for a in acts if a not in kinds.moves())
    if unknown:  # a misspelt act would match nothing: no bridge, no repair, no run
        raise ConfigError(f"{path}: not a declared move: {unknown}")
    empty = sorted(c for c, acts in sets.items() if not acts)
    if empty:
        raise ConfigError(f"{path}: a set with no acts: {empty}")
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if sets[a] & sets[b]:  # an act that is both positive and negative would renew and bridge at once
                raise ConfigError(f"{path}: {sorted(sets[a] & sets[b])} in both {a} and {b}")
    return sets


def load_event_sizes(kinds: EventKinds, path: Path = CONFIG_DIR / "event_sizes.md") -> dict[tuple[str, str], EventSize]:
    """Purpose: read each sized event's Holmes–Rahe life change units and the weeks it weighs, keyed (kind, who); a row
    whose `lcu` column says the scale does not size it is recorded with no size.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.F.6a
    Tests:   tests/bowen/test_event_sizes.py::test_m1f6a_sizes_come_from_config_and_keep_the_scales_ratios
    """
    from src.bowen.scenario.config_parse import ConfigError, parse_table_document

    doc = parse_table_document(read_text(path), columns=SIZE_COLUMNS, metadata_keys=frozenset(), source=str(path))
    sizes: dict[tuple[str, str], EventSize] = {}
    for row in doc.rows:
        kind, who = row["kind"].strip("`"), row["who"].strip()
        if kind not in kinds.mechanisms:  # a misspelt kind would size nothing, and the event would arrive at 0
            raise ConfigError(f"{path}: {kind} is not a declared event kind")
        if (kind, who) in sizes:
            raise ConfigError(f"{path}: {kind}/{who} is declared twice")
        if row["lcu"].strip() == "—":  # outside the scale, by decision (lost income, `M6.3h`)
            continue
        try:
            weeks = None if row["weeks"].strip() == "—" else int(row["weeks"])
            sizes[(kind, who)] = EventSize(kind, who, row["item"].strip(), float(row["lcu"]), weeks)
        except ValueError as e:
            raise ConfigError(f"{path}: {kind}/{who}: {e}") from None
    return sizes


def load_act_classes(kinds: EventKinds, path: Path = CONFIG_DIR / "act_classes.md") -> dict[str, frozenset[str]]:
    """Purpose: read the act classes the engine and the policy read; every class must be declared, once, and every act
             a declared move (version 2 stage 1, S1.3; rule 9).
    Spec:    docs/bowen_agent_model_spec_v2.md#M5.B.4, #M5.B.5, #M8.3, #M10.B.2
    Tests:   tests/bowen/test_config.py::test_s1_3a_act_classes_come_from_config,
             tests/bowen/test_config.py::test_s1_3b_an_undeclared_act_fails_to_load
    """
    from src.bowen.scenario.config_parse import ConfigError, parse_table_document

    doc = parse_table_document(read_text(path), columns=("class", "acts", "used by"), metadata_keys=frozenset(),
                               source=str(path))
    classes = {row["class"]: frozenset(a.strip() for a in row["acts"].split(",") if a.strip()) for row in doc.rows}
    if set(classes) != set(ACT_CLASSES) or len(classes) != len(doc.rows):
        raise ConfigError(f"{path}: the classes must be exactly {sorted(ACT_CLASSES)}, each once; found {sorted(classes)}")
    unknown = sorted(a for acts in classes.values() for a in acts if a not in kinds.moves())
    if unknown:  # a misspelt act would match nothing, and the class would silently never apply
        raise ConfigError(f"{path}: not a declared move: {unknown}")
    empty = sorted(c for c, acts in classes.items() if not acts)
    if empty:  # an empty class silently never applies, as a misspelt act does
        raise ConfigError(f"{path}: a class with no acts: {empty}")
    if not classes["side_taking"] <= classes["family_to_external"]:  # M8.3: made toward an external agent (M5.B.4)
        raise ConfigError(f"{path}: a side-taking act must be a family-to-external move: "
                          f"{sorted(classes['side_taking'] - classes['family_to_external'])}")
    return classes


def load_attribution(kinds: EventKinds, path: Path = CONFIG_DIR / "attribution.md") -> tuple[dict, dict]:
    """Purpose: read where each act places the family's difficulty, and which channel it focuses.
    Spec:    docs/bowen_agent_model_spec_v2.md#M9.6, #M1.A.11c
    Tests:   tests/bowen/test_attribution.py::test_m96_attribution_rules_come_from_config
    """
    from src.bowen.scenario.config_parse import ConfigError, parse_table_document

    doc = parse_table_document(read_text(path), columns=("kind", "locus", "focus", "why"), metadata_keys=frozenset(),
                               source=str(path))
    rows = [(row["kind"].strip("`"), row["locus"].strip(), row["focus"].strip()) for row in doc.rows]
    unknown = sorted(k for k, _, _ in rows if k not in kinds.moves())
    if unknown or len({k for k, _, _ in rows}) != len(rows):  # a misspelt act would never place or focus anything
        raise ConfigError(f"{path}: each act a declared move, once; not declared: {unknown}")
    if any(l not in ("other", "target", "sender", "—") or f not in ("physical", "mental", "social", "—")
           for _, l, f in rows):
        raise ConfigError(f"{path}: locus is other, target, sender or —; focus a channel or —")
    return ({k: l for k, l, _ in rows if l != "—"}, {k: f for k, _, f in rows if f != "—"})


def load_steering(kinds: EventKinds, path: Path = CONFIG_DIR / "perspective_steering.md") -> dict[str, int]:
    """Purpose: read which acts systems perspective steers toward or away from.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.D.1a
    Tests:   tests/bowen/test_policy.py::test_m4d1a_perspective_steers_away_from_triangling_and_blaming
    """
    from src.bowen.scenario.config_parse import ConfigError, parse_table_document

    doc = parse_table_document(read_text(path), columns=("kind", "direction", "why"), metadata_keys=frozenset(),
                               source=str(path))
    rows = [(row["kind"].strip("`"), row["direction"].strip()) for row in doc.rows]
    unknown = sorted(k for k, _ in rows if k not in kinds.moves())
    if unknown or len({k for k, _ in rows}) != len(rows) or any(d not in ("toward", "away") for _, d in rows):
        raise ConfigError(f"{path}: each steered act a declared move, once, 'toward' or 'away'; not declared: {unknown}")
    return {k: 1 if d == "toward" else -1 for k, d in rows}


DISPLACING = "## Displacing"  # config/bowen/stress_response.md's second table (revision 39, M1.A.5f)


def _stress_response_parts(path: Path) -> tuple[str, str | None, int]:
    """The soothing table's text, the displacing section's text after its heading (None if the file has none) and the
    line that heading is on, so a parse error names the file's own line."""
    lines = read_text(path).splitlines()
    at = [i for i, line in enumerate(lines) if line.strip() == DISPLACING]
    if len(at) > 1:
        from src.bowen.scenario.config_parse import ConfigError

        raise ConfigError(f"{path}: {DISPLACING!r} appears {len(at)} times")
    if not at:
        return "\n".join(lines), None, 0
    return "\n".join(lines[:at[0]]), "\n".join(lines[at[0] + 1:]), at[0] + 1


def load_soothing(kinds: EventKinds, path: Path = CONFIG_DIR / "stress_response.md") -> frozenset[str]:
    """Purpose: read which acts avoid or soothe under load.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.5e
    Tests:   tests/bowen/test_stress_response.py::test_m1a5e_soothing_acts_come_from_config
    """
    from src.bowen.scenario.config_parse import ConfigError, parse_table_document

    soothing, _, _ = _stress_response_parts(path)
    doc = parse_table_document(soothing, columns=("kind", "why it avoids or soothes"),
                               metadata_keys=frozenset(), source=str(path))
    acts = [row["kind"].strip("`") for row in doc.rows]
    unknown = sorted(a for a in acts if a not in kinds.moves())
    if unknown or len(set(acts)) != len(acts):  # a misspelt act would never lower anyone's functioning
        raise ConfigError(f"{path}: every soothing act must be a declared move, once; not declared: {unknown}")
    return frozenset(acts)


def load_displacing(kinds: EventKinds, path: Path = CONFIG_DIR / "stress_response.md") -> frozenset[str]:
    """Purpose: read the acts through whose receiver a loaded sender meets their load (displaced coping).
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.5f
    Tests:   tests/bowen/test_config.py::test_m1a5f_s3_1c_displacing_kinds_come_from_config

    The ``## Displacing`` section of ``stress_response.md``. Each kind a declared move, listed once, and never also
    soothing: an act cannot both lower its sender's functioning and meet their load through another.
    """
    from src.bowen.scenario.config_parse import ConfigError, parse_table_document

    _, section, offset = _stress_response_parts(path)
    if section is None:
        raise ConfigError(f"{path}: no {DISPLACING!r} section")
    doc = parse_table_document(section, columns=("kind", "why it meets the sender's load through the receiver"),
                               metadata_keys=frozenset(), source=str(path), line_offset=offset)
    acts = [row["kind"].strip("`") for row in doc.rows]
    unknown = sorted(a for a in acts if a not in kinds.moves())
    if unknown or len(set(acts)) != len(acts):  # a misspelt act would never displace anything
        raise ConfigError(f"{path}: every displacing act must be a declared move, once; not declared: {unknown}")
    both = sorted(set(acts) & load_soothing(kinds, path))
    if both:
        raise ConfigError(f"{path}: an act may not be both soothing and displacing: {both}")
    return frozenset(acts)


def load_nodal_reach(kinds: EventKinds, path: Path = CONFIG_DIR / "nodal_events.md") -> dict[str, str]:
    """Purpose: read which event kinds are nodal events — any major event on a person — and how far each reaches.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.4k
    Tests:   tests/bowen/test_slow_tick.py::test_m1a4k_nodal_kinds_come_from_config
    """
    from src.bowen.scenario.config_parse import ConfigError, parse_table_document

    doc = parse_table_document(read_text(path), columns=("kind", "reach", "what it is"), metadata_keys=frozenset(),
                               source=str(path))
    rows = [(row["kind"].strip("`"), row["reach"].strip()) for row in doc.rows]
    unknown = sorted(k for k, _ in rows if k not in kinds.mechanisms)
    if unknown or len({k for k, _ in rows}) != len(rows):  # a misspelt kind would never count; breadth would fall
        raise ConfigError(f"{path}: every nodal kind must be a declared event kind, once; not declared: {unknown}")
    if any(r not in ("reached", "ties") for _, r in rows):
        raise ConfigError(f"{path}: reach is 'reached' or 'ties'")
    return dict(rows)


def load_family(path: Path = CONFIG_DIR / "family_reduced.md", constants: Constants | None = None) -> FamilyInstance:
    """Purpose: read and build a family declaration.
    Spec:    docs/bowen_agent_model_spec_v2.md#M2.1
    Tests:   tests/bowen/test_family.py::test_m21_family_declared_in_markdown
    """
    text = read_text(path)
    partners = partners_file(text)  # spec revision 28: a family with a life course names its partner settings
    partners_path = Path(path).parent / partners if partners else None
    return build_family(text, constants or load_constants(), source=str(path),
                        partners_text=read_text(partners_path) if partners_path else None,
                        partners_source=str(partners_path))


def load_script(
    path: Path = CONFIG_DIR / "script_phase_b.md",
    kinds: EventKinds | None = None,
    family: FamilyInstance | None = None,
) -> ScriptedSource:
    """Purpose: read and build a script.
    Spec:    docs/bowen_agent_model_spec_v2.md#M10.B.1
    Tests:   tests/bowen/test_script.py::test_m13_script_parses_and_validates

    The script is spec §M13's Phase B driver, and lives in markdown config like every other value.
    """
    return build_script(read_text(path), kinds or load_event_kinds(), family or load_family(), source=str(path))


def load_frozen_constants(path: Path = CONFIG_DIR / "constants_frozen.md") -> Constants:
    """Purpose: the snapshot of the constants as frozen before the acceptance suite first ran.
    Spec:    docs/bowen_agent_model_spec_v2.md#M10.B.4, #M16.A.7
    Tests:   tests/bowen/test_phase_b_gate.py::test_m10b4_constants_frozen_before_suite
    """
    # A snapshot predates later keys and may hold retired ones; the change log
    # accounts for both (test_m10b4_constants_frozen_before_suite).
    return parse_constants(
        read_text(path), source=str(path), schema={**SCHEMA, **RETIRED}, require_all=False,
    )


SPEC_PATH = Path(__file__).resolve().parents[3] / "docs" / "bowen_agent_model_spec_v2.md"


def load_spec_revision(path: Path = SPEC_PATH) -> str:
    """Purpose: the spec revision a run is produced against, read from the spec's front matter.
    Spec:    docs/bowen_agent_model_spec_v2.md#M16.A.1
    Tests:   tests/bowen/test_log.py::test_m16a1_header_hashes_the_resolved_config
    """
    for line in read_text(path).splitlines()[:10]:
        if line.startswith("version:"):
            return line.split(":", 1)[1].strip()
    raise ValueError(f"{path}: no version line in the front matter")


PATTERN_SETS = frozenset({"approach", "conflict", "withdrawal", "over", "under", "fusion", "projection", "refusal"})


def load_pattern_acts(path: Path = CONFIG_DIR / "pattern_acts.md") -> dict[str, frozenset[str]]:
    """Purpose: read which acts each pattern readout counts; every set must be declared, and no other.
    Spec:    docs/bowen_agent_model_spec_v2.md#M5.A.1a
    Tests:   tests/bowen/test_patterns.py::test_b2_pattern_act_sets_come_from_config
    """
    from src.bowen.scenario.config_parse import ConfigError, parse_table_document

    doc = parse_table_document(read_text(path), columns=("set", "acts", "used by"), metadata_keys=frozenset(),
                               source=str(path))
    sets = {row["set"]: frozenset(a.strip() for a in row["acts"].split(",") if a.strip()) for row in doc.rows}
    if set(sets) != PATTERN_SETS or len(sets) != len(doc.rows):
        raise ConfigError(f"{path}: the sets must be exactly {sorted(PATTERN_SETS)}, each once; found {sorted(sets)}")
    unknown = sorted(a for acts in sets.values() for a in acts if a not in load_event_kinds().moves())
    if unknown:  # a misspelt act would match nothing and the pattern would silently never read (review 2026-10-10)
        raise ConfigError(f"{path}: not a declared move: {unknown}")
    return sets


def load_policy_rules(path: Path = CONFIG_DIR / "policy.md") -> PolicyRules:
    """Purpose: read the policy's declared tie-break and fallback rules.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.D.1f
    Tests:   tests/bowen/test_policy.py::test_m4d1f_rules_come_from_config
    """
    return parse_policy_rules(read_text(path), source=str(path))
