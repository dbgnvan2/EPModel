"""Parse the policy's tie-break and fallback rules from markdown.

Purpose: load ``config/bowen/policy.md`` into ``PolicyRules``, strictly.
Spec:    docs/bowen_agent_model_spec_v2.md#M4.D.1f, #M10.B.2
Tests:   tests/bowen/test_policy.py::test_m4d1f_rules_come_from_config
"""

from __future__ import annotations

from src.bowen.policy.rules import PolicyRules
from src.bowen.scenario.config_parse import ConfigError, parse_table_document

COLUMNS = ("rule", "value", "spec")
RULES = ("tie_break", "fallback")


def parse_policy_rules(text: str, *, source: str = "<policy rules>") -> PolicyRules:
    document = parse_table_document(text, columns=COLUMNS, metadata_keys=frozenset(), source=source)
    values: dict[str, str] = {}
    for row, line in zip(document.rows, document.row_lines):
        rule = row["rule"].strip("`")
        if rule not in RULES:
            raise ConfigError(f"{source}:{line}: unknown rule {rule!r}; expected {list(RULES)}")
        if rule in values:
            raise ConfigError(f"{source}:{line}: duplicate rule {rule!r}")
        values[rule] = row["value"]
    missing = [r for r in RULES if r not in values]
    if missing:
        raise ConfigError(f"{source}: missing rules {missing}")
    try:
        return PolicyRules(**values)
    except ValueError as error:
        raise ConfigError(f"{source}: {error}") from None
