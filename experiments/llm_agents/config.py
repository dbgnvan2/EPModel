"""The experiment's four config files, read strictly.

Purpose: load `llm.md`, `act_labels.md`, `banned_terms.md` and `prompt_text.md` into typed objects, refusing an
         unknown, missing or malformed entry, so no setting, label or prompt word lives in code (rules 8, 9; L1).
Spec:    docs/plan_v3_llm_agent_experiment.md#3.1, #3.2, #4, #5, #8
Tests:   tests/experiments/llm_agents/test_prompts.py::test_lx4_no_banned_term_in_any_arm,
         tests/experiments/llm_agents/test_budget.py::test_lx11_estimate_over_cap_refuses

Pure: reads files, writes nothing. The engine's own parser (`parse_table_document`) reads each file's one table.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from src.bowen.scenario.config_parse import ConfigError, parse_table_document

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config"
GROUPS = ("tie", "triad", "outside", "reopen")


def _plain(cell: str) -> str:
    return cell.strip().strip("`").strip()


def _table(path: Path, columns: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    doc = parse_table_document(path.read_text(encoding="utf-8"), columns=columns, metadata_keys=frozenset(),
                               source=str(path))
    return tuple({k: _plain(v) for k, v in row.items()} for row in doc.rows)


@dataclass(frozen=True)
class Label:
    kind: str
    group: str
    option: str
    seen: str


@dataclass(frozen=True)
class Phase:
    name: str
    arms: tuple[str, ...]
    seeds: tuple[int, ...]
    weeks: int
    model: str | None
    pool: str | None
    calls: int | None = None


@dataclass(frozen=True)
class LLMConfig:
    settings: Mapping[str, str]
    phases: Mapping[str, Phase]

    def get(self, key: str) -> str:
        if key not in self.settings:
            raise ConfigError(f"llm.md has no setting {key!r}")
        return self.settings[key]

    def number(self, key: str) -> float:
        return float(self.get(key))

    def integer(self, key: str) -> int:
        value = self.get(key)
        if not value.isdigit():
            raise ConfigError(f"llm.md: {key} must be a whole number, got {value!r}")
        return int(value)

    def listed(self, key: str) -> tuple[str, ...]:
        return tuple(v.strip() for v in self.get(key).split(",") if v.strip())

    def models(self) -> tuple[str, ...]:
        return (self.get("model_main"), self.get("model_cross_check"))

    def thinking(self, model: str) -> str:
        return self.get(f"thinking.{model}")

    def effort(self, model: str) -> str:
        return self.get(f"effort.{model}")

    def prices(self, model: str) -> tuple[float, float]:
        return self.number(f"price_input.{model}"), self.number(f"price_output.{model}")

    def cap(self, pool: str) -> float:
        return self.number(f"cap_{pool}_usd")


REQUIRED = frozenset({
    "model_main", "model_cross_check", "prices_read", "max_tokens", "timeout_seconds", "max_attempts",
    "backoff_base_seconds", "backoff_cap_seconds", "retry_after_max_seconds", "max_concurrency", "max_fallback_share", "history_rounds",
    "estimate_input_tokens", "estimate_output_tokens", "worst_case_chars_per_token", "cap_main_usd",
    "cap_cross_check_usd", "cap_total_usd", "family", "horizon_weeks", "spells", "audit_prompt_weeks",
    "evaluation_window_weeks", "evaluation_readouts",
})
PER_MODEL = ("thinking", "effort", "price_input", "price_output")
PHASES = ("baseline", "smoke", "pilot", "main", "cross_check")
THINKING = frozenset({"disabled", "between_tools"})


def _phase(name: str, value: str) -> Phase:
    fields = {}
    for part in value.split(";"):
        key, sep, val = part.partition("=")
        if not sep:
            raise ConfigError(f"llm.md phase.{name}: {part!r} is not key=value")
        fields[key.strip()] = val.strip()
    if set(fields) - {"arms", "seeds", "weeks", "model", "pool", "calls"} or not {"arms", "seeds", "weeks", "model",
                                                                                  "pool"} <= set(fields):
        raise ConfigError(f"llm.md phase.{name}: fields {sorted(fields)}")
    model = None if fields["model"] == "none" else fields["model"]
    pool = None if fields["pool"] == "none" else fields["pool"]
    return Phase(name, tuple(a.strip() for a in fields["arms"].split(",")),
                 tuple(int(s) for s in fields["seeds"].split(",")), int(fields["weeks"]), model, pool,
                 int(fields["calls"]) if "calls" in fields else None)


@functools.cache
def load_llm(path: Path = CONFIG / "llm.md") -> LLMConfig:
    """Purpose: the model, scale, budget and phase settings, with every model's per-model rows present (L1)."""
    rows = _table(path, ("setting", "value"))
    settings: dict[str, str] = {}
    for row in rows:
        if row["setting"] in settings:
            raise ConfigError(f"{path}: {row['setting']} given twice")
        settings[row["setting"]] = row["value"]
    phases = {k.split(".", 1)[1]: _phase(k.split(".", 1)[1], v) for k, v in settings.items() if k.startswith("phase.")}
    plain = {k: v for k, v in settings.items() if not k.startswith("phase.")}
    models = (plain.get("model_main"), plain.get("model_cross_check"))
    expected = REQUIRED | {f"{p}.{m}" for p in PER_MODEL for m in models}
    if set(plain) != expected:
        raise ConfigError(f"{path}: missing {sorted(expected - set(plain))}, unknown {sorted(set(plain) - expected)}")
    if set(phases) != set(PHASES):
        raise ConfigError(f"{path}: phases {sorted(phases)}, expected {sorted(PHASES)}")
    for m in models:
        if plain[f"thinking.{m}"] not in THINKING:
            raise ConfigError(f"{path}: thinking.{m} must be one of {sorted(THINKING)}")
    for p in phases.values():
        if p.model is not None and p.model not in models:
            raise ConfigError(f"{path}: phase {p.name} names model {p.model!r}, not one of {models}")
        if p.pool is not None and f"cap_{p.pool}_usd" not in plain:
            raise ConfigError(f"{path}: phase {p.name} names pool {p.pool!r}, which has no cap")
    return LLMConfig(MappingProxyType(plain), MappingProxyType(phases))


@functools.cache
def load_labels(path: Path = CONFIG / "act_labels.md") -> Mapping[str, Label]:
    """Purpose: the plain label of every act kind, with its legality group (§3.2, §3.3)."""
    labels = {}
    for row in _table(path, ("kind", "group", "option", "seen")):
        if row["group"] not in GROUPS:
            raise ConfigError(f"{path}: {row['kind']} has group {row['group']!r}, not one of {GROUPS}")
        if row["kind"] in labels:
            raise ConfigError(f"{path}: {row['kind']} given twice")
        for col in ("option", "seen"):
            if "{target}" not in row[col]:
                raise ConfigError(f"{path}: {row['kind']}'s {col} has no {{target}}")
        if (row["group"] == "triad") != ("{third}" in row["option"]):
            raise ConfigError(f"{path}: {row['kind']}: a triad act's option names {{third}}, no other does")
        labels[row["kind"]] = Label(row["kind"], row["group"], row["option"], row["seen"])
    return MappingProxyType(labels)


@functools.cache
def load_banned(path: Path = CONFIG / "banned_terms.md") -> tuple[tuple[str, str], ...]:
    """Purpose: the lint's terms with their class (§3.4, LX.4); the family file's names are added by the lint."""
    rows = _table(path, ("term", "class"))
    return tuple((r["term"], r["class"]) for r in rows)


TEXT_KEYS = frozenset({
    "system", "round_header", "tension_first", "tension_line", "tie_header", "tie_line", "contact_open",
    "contact_broken", "contact_stopped", "belief_header", "belief_line", "belief_none", "seen_header", "seen_target",
    "seen_witness", "seen_opaque_target", "seen_opaque_witness", "seen_outside", "seen_none", "history_header",
    "history_line", "history_line_no_feedback", "history_none", "option_header", "option_line", "option_opaque",
    "option_opaque_triad", "option_hold", "answer", "reask",
})


@functools.cache
def load_text(path: Path = CONFIG / "prompt_text.md") -> Mapping[str, str]:
    """Purpose: every fixed word of the prompt (§3.1); the renderer only fills fields."""
    rows = _table(path, ("key", "text"))
    text = {r["key"]: r["text"] for r in rows}
    if set(text) != TEXT_KEYS or len(text) != len(rows):
        raise ConfigError(f"{path}: missing {sorted(TEXT_KEYS - set(text))}, unknown {sorted(set(text) - TEXT_KEYS)}")
    return MappingProxyType(text)
