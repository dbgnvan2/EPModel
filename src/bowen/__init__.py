"""EPModel v2 — the agent model over one three-generation family.

The contract is docs/bowen_agent_model_spec_v2.md; the build order is
docs/implementation_plan_phase_b.md. `engine/` is pure (no I/O, no UI);
`scenario/` parses configuration text; `io/` is the only place files are read
or written.
"""
