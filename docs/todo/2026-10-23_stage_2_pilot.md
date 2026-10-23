# Stage 2's pilot — open after the reviews, 2026-10-23

Raised by the two reviews of the pilot's build commit (`docs/implementation_plan_version_2.md` §2.10, step 6). Items
the reviews found and the fix commit did not take.

- [ ] T-124 ⟨later⟩ **The Phase D mutation record's `engine_hash` label is that tool's own digest:**
  `tools/phase_d_mutation_record.py::engine_hash` hashes `code_hash(CHILD)` plus the source of its child functions, so
  the hash printed in `docs/phase_d_mutation_record.md` is not `tools/record_cache.py::engine_hash()` (`66dcbb1e8ba5`),
  which the other records and the sweep print. Version 1 did the same, nothing compares the two, and each is a
  correct key for its own cache. Either rename the Phase D label (*record hash*) or print both, so a reader does not
  take the two for one number; a one-line change plus the record's re-render from its cache, no rerun.
