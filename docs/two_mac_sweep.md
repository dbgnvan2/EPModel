# Running the stage 2 sweep on two Macs

**Status: BUILT 2026-10-23** (`docs/implementation_plan_version_2.md` §2.10); first used for part II's sweep, 2026-10-23–24 (four criteria on Mac Mini1, three here).

How to split the constant sweep (`tools/version_2_sweep.py`, `M17.E.1`) between this Mac (davemini2) and Mac
Mini1, 10 cores each, and combine the results into one record. The tools are `tools/version_2_sweep.py --criteria`
(with `--recompute` for the cross-check) and `--render-only`, `tools/merge_cache.py` and `tools/two_mac_sweep.py`.
Their tests are `tests/test_sweep_two_macs.py`.

## What the tools guarantee, and what they do not

- A result is cached by `result_key(engine hash, the sample's edits, criterion key)`. The same key on the two Macs
  names the same run. **If the two Macs compute bit-identical numbers**, the union of their caches is the cache one
  run would have built. Nothing in a split by criterion tests that, because the two shares share no key. The
  cross-check below does, and it is run before every split.
- `tools/merge_cache.py` writes the union of two cache files. It writes nothing, and never picks one, when:
  - one key holds two different results;
  - the two files record different NumPy versions, or either records none;
  - a sweep is running on this Mac. A running sweep rewrites the whole file after each run, so a merge made
    meanwhile would be lost.
- The two Macs must run the same NumPy:
  - `requirements-sweep.txt` pins it (`numpy==1.26.4`, the version every cached result to 2026-10-23 was made
    under).
  - The cache file records it under `_meta`, outside the result keys.
  - A sweep run refuses a NumPy other than the pin, other than the cache's, or a cache that holds results and
    records no version.
  - `requirements.txt` stays unpinned, because CI tests Python 3.11–3.13 and NumPy 1.26.4 has no wheel for 3.13.
    `--render-only` does not check NumPy.
- A run with `--criteria` does the following:
  - It checks every declared key first: an ensemble record holds it, and its class is directional.
  - It refuses a key the config does not declare.
  - It runs only its share and writes only the cache. It writes every entry the cache held, so no result from the
    other Mac or from an earlier run is lost.
  - It never writes `docs/version_2_sweep_record.md`. A partial record would replace the whole one and fail
    `tests/test_version_2_sweep.py::test_s2_11_sweep_record_is_current` until both shares were merged, so the
    record is rendered once, after the merge.
- `--render-only` runs nothing. If a result is missing, it stops and writes nothing.
- A key listed under *Declared, not run* in `config/sweep/sweep.md` is shown in the record as that and is run only
  when `--criteria` names it. Once it has a result in the cache, the record is refused until its row there is
  removed.

## Before you start

1. Both Macs are on the same commit of `main`, pushed from this Mac. On Mac Mini1, the owner runs
   `git pull --ff-only` in `~/ProjectsMini1/EPModel`. Nothing is committed or pushed from Mac Mini1.
2. On Mac Mini1, use only `~/ProjectsMini1/EPModel/.venv/bin/python` (Python 3.12.3, NumPy 1.26.4). Plain
   `python3` there is Homebrew's 3.14 with no NumPy.
3. On this Mac, `python3 -c "import numpy; print(numpy.__version__)"` prints `1.26.4`. A new sweep machine installs
   `pip install -r requirements-sweep.txt`.
4. Check that the two Macs match:

   ```
   python3 tools/two_mac_sweep.py check
   ```

   It compares the commit, clean tracked files (the sweep cache aside), both engine hashes, the Python version, the
   CPU type, and NumPy against the pin, and stops on any difference. A different macOS version is printed, not
   refused.

## The cross-check (before every split)

Mac Mini1 runs one already-cached criterion again. Its new results are then merged against this Mac's committed
ones, and the merge stops if any number differs. `M11.C.16` is the cheapest: 8 samples and no rerun in the pilot
(8 of the pilot's 24 child runs, which took 476 s together here).

```
python3 tools/two_mac_sweep.py start --criteria M11.C.16 --recompute --workers 10
python3 tools/two_mac_sweep.py status
python3 tools/two_mac_sweep.py fetch /path/to/scratch/mini1_check.json
python3 tools/merge_cache.py /path/to/scratch/mini1_check.json
```

If all goes well, the merge prints `0 result(s) added`: every recomputed result is already here and equal. `--recompute` drops that
criterion's cached results on Mac Mini1 and runs them again. A sample whose rerun fails keeps its earlier result and
is counted, so that sample is not compared; a count above 0 there means the check is incomplete. If the merge
refuses, the two Macs do not compute the same numbers: do not split, and report it.

## Split the criteria

List the keys still to sweep from the Ranges table of `config/sweep/sweep.md`. Results already cached are read, not
rerun. Split the keys into two shares of about equal projected time. No per-criterion projection for part II's
seven exists yet (the whole is projected at 23–40 h on one Mac), so split them by count, four and three, and put the
criterion expected to take longest in the smaller share. A key with spaces or brackets, such as
`M11.C.7[joint over sided]`, is quoted on the command line.

## Run both shares

Start Mac Mini1's share first:

```
python3 tools/two_mac_sweep.py start --criteria KEY_A KEY_B KEY_C --workers 10
```

`start` goes through these steps in order:

1. It checks the keys against the config here.
2. It runs `check`.
3. It refuses if a sweep is already running on Mac Mini1, or if `pgrep` fails there.
4. It runs this command on Mac Mini1, detached, with idle sleep held off:

   ```
   cd ~/ProjectsMini1/EPModel && nohup sh -c 'echo "START $(date) …"; caffeinate -i .venv/bin/python tools/version_2_sweep.py --criteria KEY_A KEY_B KEY_C --workers 10; echo "EXIT $? $(date)"' >> ~/epmodel_sweep.log 2>&1 < /dev/null &
   ```

5. After 15 s it checks that the run is still alive. If the run has stopped, it prints the log.

Then start this Mac's share:

```
caffeinate -i python3 tools/version_2_sweep.py --criteria KEY_D KEY_E KEY_F KEY_G --workers 10
```

Check Mac Mini1 at any time:

```
python3 tools/two_mac_sweep.py status
```

`status` prints `running`, `stopped` or `error`, then the log's last `START` and `EXIT` lines (which run, and its
exit status), then its last lines. A FAILED sample is not cached and the run exits 1. Run the same share again;
only the missing samples run.

## Combine and render

1. When both runs have stopped (`status` shows the `EXIT` line of this share), copy Mac Mini1's cache to a new file.
   `fetch` refuses an existing path:

   ```
   python3 tools/two_mac_sweep.py fetch /path/to/scratch/mini1_sweep_v2.json
   ```

   This runs `scp davemini@mini1.local:ProjectsMini1/EPModel/docs/records_cache/sweep_v2.json <path>`.
2. Once this Mac's own share has stopped, merge Mac Mini1's copy into this Mac's cache:

   ```
   python3 tools/merge_cache.py /path/to/scratch/mini1_sweep_v2.json
   ```

   If the merge refuses, stop and report the refusal. Do not edit either file.
3. If the shares ran keys listed under *Declared, not run* in `config/sweep/sweep.md`, remove their rows there
   first. Then render the record from the merged cache, with no run, and check it:

   ```
   python3 tools/version_2_sweep.py --render-only
   python3 -m pytest tests/test_version_2_sweep.py -q
   ```

4. Commit the cache and the record from this Mac, and push. Mac Mini1's checkout then holds its own modified copy of
   the cache. The owner discards it there (`git checkout -- docs/records_cache/sweep_v2.json`) and pulls.

## Not verified

- That the two Macs give bit-identical results for every criterion. The cross-check was run twice, 2026-10-23, and
  the merge found all 361 results equal, 0 added, each time: on `M11.C.16` (Phase C; Mac Mini1 recomputed its 8
  results in 3 min 7 s) and on `M11.C.23` (Phase D part I; Mac Mini1 dropped its 9 cached results and ran them
  again, 8 first runs and sample 1's rerun at 200 seeds, in 1 min 16 s; its cache file came out byte-identical to the
  committed one). Two of the 27 swept criteria were compared, one of each child and one 200-seed rerun; no part II
  criterion is declared yet. Both Macs are arm64, with Python 3.12.3 and NumPy 1.26.4; macOS is 15.7.9 here and
  15.7.8 on Mac Mini1.
- `tools/two_mac_sweep.py` against a live Mac Mini1 was run once, 2026-10-23: `check`, `start`, `status` and `fetch`
  all worked over SSH. A dropped connection and its retries have not been seen live; the tests stub them.
