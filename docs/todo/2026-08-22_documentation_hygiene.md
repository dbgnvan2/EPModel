# Documentation hygiene, deferred

- [x] T-112 ⟨later⟩ **`§n` cross-references are ambiguous across documents.** All 50 in the v2 spec resolve against `model_explainer.md`, but several sentences address other documents ("proposal §9", "the frozen spec's §5.4") and `§5.4` exists in both. A traceability scan over `§` cannot tell the targets apart. Prefix them with the document.
  ⟦batch: the next spec revision (wording only; ride with engine batch E1's revision)⟧ *(triage 2026-11-17, Claude's recommendation, pending owner review)*
  *Built 2026-11-17 in spec revision 64 (engine batch E1), Claude's recommendation, pending owner review; see CHANGELOG.md's revision 64 entry. References to the proposal, the frozen spec and the phase C and D plans now name the document. One stays unresolved: `M11.C`'s "Report §11" names no document, and the report it means was not found.*
