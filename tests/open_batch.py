"""Stage 3's open batch (D31): the records keyed by the engine hash stay stale from plan step 6 until step 10.

Purpose: under D31 (`docs/implementation_plan_version_2_stage_3.md` §3.6, §3.8) stage 3's rule changes land in four
         commits, steps 6–9, each of which moves the engine hash, and every record is regenerated once, at step 10.
         Between, a record-current test fails by that decision, not by a fault. ``stale_in_open_batch`` marks such a
         test. While ``OPEN`` is set and the record the test reads is byte-identical to the file as the batch found
         it (``AT_OPEN``, its sha256 at ``OPENED_AT``), a failing test is run again with the hash computed on the
         tree the batch found (``src/bowen``, ``config/bowen`` and ``tools`` at ``OPENED_AT``), and each mutation
         record tool's ``MUTANTS`` cut to the mutants that tool had then (``mutants_as_found``): if it then passes,
         the hash is the only thing that makes it fail, and it is an expected failure that names the decision; if it
         fails again, it fails as usual. A passing test passes; a record changed since the batch opened (regenerated
         in part, or edited) fails as usual. Step 10 regenerates every record and sets ``OPEN`` to None.
Spec:    docs/implementation_plan_version_2_stage_3.md#3.6 (D31); CLAUDE.md — "A red suite is never normal"
Tests:   tests/test_open_batch.py

Not a way to hide a stale record: ``tests/test_open_batch.py`` fails while ``OPEN`` is set and some but not all listed
records are as the batch found them (a regeneration in part, which D31 does not allow), or none is (the batch is over:
close it), and while ``OPEN`` is None and a test still carries the mark.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import io
import re
import subprocess
import sys
import tarfile
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# The batch, while it is open: why the records are stale and when they stop being. None once step 10 has run.
# Closed at plan step 10 (2026-10-26): every record regenerated once on the engine hash 2a3b84e2d89a and the marks
# taken off. Kept, with the machinery below, so a later batch can reopen it with its own OPENED_AT and AT_OPEN.
OPEN: str | None = None

OPENED_AT = "9ebf3274554a32f77a401304ad1dec587fc99022"  # 9ebf327, the commit whose records the batch keeps (spec revision 38, engine hash 66dcbb1e8ba5)
HASHED_TREES = ("src/bowen", "config/bowen", "tools")  # everything ``tools.ensemble_record.code_hash`` reads
# ``tools`` is pinned with the engine: steps 7–9 add mutants to the mutation records' tools, which D31 also leaves
# stale until step 10, so an edit to a hashed tool is excused as a rule edit is; the records still must be as found.
# A mutant added since changes a record's render as well as its hash (a criterion that does not pass lists it under
# "Not run"), so the rerun also leaves out each mutant the tool did not have at OPENED_AT (step 7, 2026-10-26).
# A criterion added since is likewise not in the records the batch keeps, so the rerun also leaves out each criterion
# the criteria modules did not have at OPENED_AT (step 8, 2026-10-26: M11.C.49).

# Each record a marked test reads, with its sha256 as the batch found it (commit 9ebf327, spec revision 38).
AT_OPEN = {
    "docs/phase_c_ensemble_record.md": "4ac9cbdccbda9aac7d45bfcd09ace70c369b8151f9ef937b63a094daeb3e41af",
    "docs/phase_c_mutation_record.md": "c5d7a427d0168fa1df8a6cd80bfcaad7db24dfe1127cd26a0555a16e0af696e1",
    "docs/phase_c_sweep_record.md": "2c696f324d36d0678a912f033997a172b80966f408faa94657a1593cf77a52df",
    "docs/phase_c_level_occupancy.md": "11018ea02eb477a78259cf37a10d5a973c89d0e34d4161ad957e0bdc8f835a1e",
    "docs/phase_c_diagnostic_record.md": "ffb9e7918caebaecdc45f69850e2825f88f532f3091df73a895f5a1c94db9d24",
    "docs/phase_d_ensemble_record.md": "ed6998a564209df9ae55e19875d96b6434849e7cb8fcf2ca7f370f62abd2925f",
    "docs/phase_d2_ensemble_record.md": "6df6784d9bf3483c3b9e3a369bc4ed23a859f50843d5ada486cf7237027fe40b",
    "docs/phase_d_mutation_record.md": "e9e83a80fb054c339735d578d882b628b8aa50282ea81e93277a89dbffb9d7bc",
    "docs/version_2_sweep_record.md": "d170e3d11bd7478495c3775da5f867645f65f4073389aa36b95da6fb4d3c6341",
}

_DEFINES_MUTANTS = re.compile(r"^MUTANTS\b[^=\n]*=", re.M)  # a module-level assignment, not an import
_found_tree: Path | None = None  # the hashed trees at OPENED_AT, extracted once per session
TEMP = None  # pytest's TempPathFactory, set by tests/conftest.py: a test may write only under it (M11.D.7)


def as_found(record: str, repo: Path = REPO) -> bool:
    """Whether ``record`` is byte-identical to the file as the batch found it."""
    return hashlib.sha256((repo / record).read_bytes()).hexdigest() == AT_OPEN[record]


def tree_as_found() -> Path:
    """A directory under pytest's temporary directory holding ``HASHED_TREES`` as they stood at ``OPENED_AT``
    (``git archive``; the checkout needs history, as CI's ``fetch-depth: 0`` gives it)."""
    global _found_tree
    if _found_tree is None:
        archive = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", OPENED_AT, *HASHED_TREES],
                                 capture_output=True, check=True, timeout=60).stdout
        if TEMP is None:
            raise RuntimeError("the open batch's tree needs pytest's temporary directory (tests/conftest.py)")
        root = TEMP.mktemp("open_batch").resolve()  # resolved: code_hash compares resolved paths
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(root, filter="data")
            else:  # an interpreter older than the extraction filters; the archive is this repo's own history
                tar.extractall(root)
        _found_tree = root
    return _found_tree


@contextlib.contextmanager
def hash_as_found(namespace: dict | None = None):
    """Every ``tools.ensemble_record.code_hash`` reads the tree the batch found, so the hash is the batch's opening one.

    Every record hash goes through that function (``record_cache.engine_hash``, ``level_occupancy.record_hash`` and
    ``phase_d_mutation_record.engine_hash`` import it when called, and a module that imported it at load has its own
    name for it); a test file that loaded a tool from its file (``namespace``, the test's globals) has its own copy of
    the function, found by its source file and patched as well."""
    from tools import ensemble_record

    found = tree_as_found()
    original = ensemble_record.code_hash
    source = original.__code__.co_filename
    modules = [m for m in _reachable_modules(namespace or {})
               if getattr(getattr(m, "code_hash", None), "__code__", None) is not None
               and m.code_hash.__code__.co_filename == source]

    def code_hash(*tools: Path) -> str:
        moved = [found / Path(t).resolve().relative_to(REPO) for t in tools]
        with patched(ensemble_record, "REPO", found):
            return original(*moved)

    with contextlib.ExitStack() as stack:
        for module in modules:
            stack.enter_context(patched(module, "code_hash", code_hash))
        for module, kept in mutants_as_found(namespace or {}, found):
            stack.enter_context(patched(module, "MUTANTS", kept))
        for holder, name, cut in criteria_as_found(namespace or {}, found):
            stack.enter_context(patched(holder, name, cut))
        yield


def _own(text: str, mutant_id: str) -> str:
    """The text of ``Mutant("mutant_id"``'s definition in ``text``: from it to the next ``Mutant(``, or the end."""
    start = text.find(f'Mutant("{mutant_id}"')
    end = text.find("Mutant(", start + 1)
    return text[start:end if end >= 0 else len(text)]


def mutants_as_found(namespace: dict, found: Path) -> list[tuple[types.ModuleType, tuple]]:
    """Each reachable tool module under ``tools/`` that holds ``MUTANTS``, with that tuple cut to the mutants whose
    ``Mutant("id"`` the file that defines it held at ``OPENED_AT`` (none, if the file did not exist), and each kept
    mutant's criteria cut to those its own definition there named, quoted (from its ``Mutant("id"`` to the next
    ``Mutant(``; step 9 added ``M11.C.45`` to ``one-sided-deviation``, a key other mutants already quoted). A module
    that imported the tuple (the diagnostic record's tool imports the mutation record's) gets the defining file's cut,
    as the same object. A mutant or criterion added since is a tool edit D31 leaves for step 10; one whose edits
    changed since keeps its id here, and its result key moves, so a test that reads it still fails as usual. A mutant
    whose definition then named no criterion quoted (its criteria are built, not written out) keeps them all."""
    tools = (REPO / "tools").resolve()
    holders: dict[int, list[types.ModuleType]] = {}
    for module in _reachable_modules(namespace):
        path = getattr(module, "__file__", None)
        if path and isinstance(getattr(module, "MUTANTS", None), tuple) and tools in Path(path).resolve().parents:
            holders.setdefault(id(module.MUTANTS), []).append(module)
    out = []
    for modules in holders.values():
        defining = [m for m in modules if _DEFINES_MUTANTS.search(Path(m.__file__).read_text(encoding="utf-8"))]
        if len(defining) != 1:
            raise RuntimeError(f"MUTANTS held by {[m.__name__ for m in modules]} is defined by {len(defining)} of them")
        then = found / Path(defining[0].__file__).resolve().relative_to(REPO)
        text = then.read_text(encoding="utf-8") if then.exists() else ""
        kept = tuple(dataclasses.replace(m, criteria=tuple(c for c in m.criteria if f'"{c}"' in _own(text, m.id))
                                         or m.criteria)
                     for m in modules[0].MUTANTS if f'Mutant("{m.id}"' in text)
        out += [(module, kept) for module in modules]
    return out


_CRITERIA_TABLE = re.compile(r"^CRITERIA(_D2?)?$")  # the criteria tables of src/bowen/ensemble


def criteria_as_found(namespace: dict, found: Path) -> list[tuple[object, str, dict]]:
    """Each criteria table of ``src/bowen/ensemble`` reachable from ``namespace``, cut to the criteria whose id (or,
    for a keyed variant such as ``M11.C.2[twelve-person family]``, whose base id) the file that defines it held,
    quoted, at ``OPENED_AT``: a criterion added since (step 8's ``M11.C.49``) is not in a record the batch keeps, so
    a test that walks the table and the record fails by D31, as a mutant added since does. Returned as (holder,
    name, cut table), the holder a module or ``namespace`` itself where it holds the same table by that name. Each
    table is cut once, by the file that assigns it: a module that imported a table by name gets that cut, as the same
    object, never one made from its own file (the review of 2026-10-26), as ``mutants_as_found`` does."""
    ensemble = (REPO / "src" / "bowen" / "ensemble").resolve()
    modules = _reachable_modules(namespace)
    out = []
    for module in modules:
        path = getattr(module, "__file__", None)
        if not path or Path(path).resolve().parent != ensemble:
            continue
        now = Path(path).read_text(encoding="utf-8")
        then = found / Path(path).resolve().relative_to(REPO)
        text = then.read_text(encoding="utf-8") if then.exists() else ""
        for name, table in vars(module).items():
            if not (_CRITERIA_TABLE.match(name) and isinstance(table, dict)
                    and re.search(rf"^{name}\b[^=\n]*=", now, re.M)):
                continue
            cut = {k: v for k, v in table.items() if re.search(rf'"{re.escape(k.split("[")[0])}["\[]', text)}
            out += [(holder, n, cut) for holder in modules for n, v in vars(holder).items() if v is table]
            out += [(namespace, n, cut) for n, v in namespace.items() if v is table]
    return out


def _reachable_modules(namespace: dict) -> list[types.ModuleType]:
    """``sys.modules``, the modules ``namespace`` holds, and every module this repo's own modules hold, followed through:
    a tool loaded from its file (``module_from_spec``) is not registered, and may hold another such copy."""
    seen: dict[int, types.ModuleType] = {}
    stack = list(sys.modules.values()) + [v for v in namespace.values() if isinstance(v, types.ModuleType)]
    while stack:
        module = stack.pop()
        if not isinstance(module, types.ModuleType) or id(module) in seen:
            continue
        seen[id(module)] = module
        path = getattr(module, "__file__", None)
        if path and REPO in Path(path).resolve().parents:
            stack += [v for v in vars(module).values() if isinstance(v, types.ModuleType)]
    return list(seen.values())


@contextlib.contextmanager
def registry_as_found(namespace: dict | None = None):
    """The criteria the batch found, for a tool that reads the registry beside the committed records: each criteria
    table cut as the rerun cuts it (``criteria_as_found``), and ``tools.version_1_status.spec_criteria`` without a
    criterion added since ``OPENED_AT``, which no record the batch keeps can list (step 8's ``M11.C.49``;
    ``tools/version_2_status.py`` stops on a registered criterion with no record row). Restored on any exit."""
    import src.bowen.ensemble.criteria  # noqa: F401 — loaded first, so the tables are there to cut
    import src.bowen.ensemble.criteria_d  # noqa: F401
    import src.bowen.ensemble.criteria_d2  # noqa: F401
    import tools.version_1_status as v1

    cuts = criteria_as_found(namespace or {}, tree_as_found())
    kept = {k.split("[")[0] for _, _, cut in cuts for k in cut}
    added = {k.split("[")[0] for k in v1.registered()} - kept
    original = v1.spec_criteria
    with contextlib.ExitStack() as stack:
        for holder, name, cut in cuts:
            stack.enter_context(patched(holder, name, cut))
        stack.enter_context(patched(v1, "spec_criteria",
                                    lambda: {k: v for k, v in original().items() if k not in added}))
        yield added


@contextlib.contextmanager
def patched(module, name: str, value):
    """``module.name`` (or, for a dict, ``module[name]``) set to ``value`` for the block, and restored after it, on
    any exit."""
    if isinstance(module, dict):
        before = module[name]
        module[name] = value
        try:
            yield
        finally:
            module[name] = before
        return
    before = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, before)


def stale_in_open_batch(record: str):
    """Mark a record-current test: its failure is expected only while the batch is open, ``record`` is as found, and
    the test passes when the hash is the batch's opening one."""
    if record not in AT_OPEN:
        raise KeyError(f"{record} is not a record the batch lists (AT_OPEN)")

    def wrap(test):
        @functools.wraps(test)
        def run(*args, **kwargs):
            try:
                return test(*args, **kwargs)
            except (AssertionError, KeyError) as stale:
                if not (OPEN and as_found(record)):
                    raise
                try:
                    with hash_as_found(test.__globals__):
                        test(*args, **kwargs)
                except Exception as other:
                    raise AssertionError(f"{test.__name__} fails with the hash the batch found too, so its failure "
                                         f"is not the batch's: {type(other).__name__}: {other}") from stale
                pytest.xfail(f"{record} is stale by decision: {OPEN} ({type(stale).__name__}: {str(stale)[:160]}); "
                             f"the test passes with the hash the batch found")

        run.open_batch_record = record
        return run

    return wrap
