"""Shared builders for the client-layer tests: a config with changed settings, and a short phase."""

import dataclasses
from types import MappingProxyType

from experiments.llm_agents.config import Phase, _phase, load_llm


def config(**changes):
    cfg = load_llm()
    settings = dict(cfg.settings)
    for k, v in changes.items():
        settings[k.replace("__", ".")] = str(v)
    phases = dict(cfg.phases)
    for k, v in changes.items():
        if k.startswith("phase__"):
            phases[k.split("__", 1)[1]] = _phase(k.split("__", 1)[1], str(v))
    plain = {k: v for k, v in settings.items() if not k.startswith("phase.")}
    return dataclasses.replace(cfg, settings=MappingProxyType(plain), phases=MappingProxyType(phases))


def phase(weeks=3, arms=("A1",), seeds=(1,), name="main"):
    cfg = load_llm()
    return Phase(name, tuple(arms), tuple(seeds), weeks, cfg.get("model_main"), "main")
