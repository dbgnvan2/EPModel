#!/usr/bin/env python3
"""arXiv sweep for the EPModel method-literature watch.

Deterministic discovery half of the weekly cron job: runs the six facet
queries, dedupes, drops withdrawn and already-catalogued papers, and writes
candidates to `_candidates.json` for the agent to judge.

Design notes (learned the hard way, 2026-09-13):
- arXiv throttles a generic `Mozilla/5.0` UA far harder than a descriptive one.
- 6 queries back-to-back trip 429/503; space them >= 6s and honour Retry-After.
- Semantic Scholar 429s without a key, so discovery is arXiv-only here.

Usage:
    python3 sweep.py [--days N] [--out PATH]
Stdout is a human summary; candidates land in the JSON file (default
`_candidates.json` beside this script).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
UA = "epmodel-method-watch/1.0 (local research monitor; weekly)"
BASE = "https://export.arxiv.org/api/query"
HERE = Path(__file__).resolve().parent

QUERIES = {
    "ARCH": 'abs:"generative agents" OR ti:"simulacra"',
    "FIDELITY": 'all:"algorithmic fidelity" OR all:"silicon samples"',
    "TRAIT": 'abs:"large language model" AND (abs:persona OR abs:personality OR abs:emotion)',
    "ADAPT": 'abs:("self-adapting" OR "self-improving") AND abs:"language model"',
    "DYNAMICS": 'abs:("emergent misalignment" OR alignment) AND (abs:agent OR abs:simulation)',
    "MECH": 'abs:("temporal difference" OR "reinforcement learning") AND abs:(addiction OR habit OR anxiety)',
}

# Group 1 is the unversioned id; the optional suffix lets the catalogue carry
# either form ("2609.17331" or "2609.17331v1") and still be recognised.
_ARXIV_ID_RE = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")
_VERSION_SUFFIX_RE = re.compile(r"v\d+$")


def base_id(arxiv_id: str) -> str:
    """Strip the version suffix: '2609.17331v1' -> '2609.17331'.

    The API returns versioned ids while INDEX.md carries unversioned ones, so
    every comparison between the two must go through this.
    """
    return _VERSION_SUFFIX_RE.sub("", arxiv_id.strip())


def already_catalogued(index_path: Path) -> set[str]:
    """Unversioned ids already in the catalogue."""
    if not index_path.exists():
        return set()
    return set(_ARXIV_ID_RE.findall(index_path.read_text(encoding="utf-8")))


def fetch(query: str) -> bytes:
    url = BASE + "?" + urllib.parse.urlencode(
        {"search_query": query, "start": 0, "max_results": 15,
         "sortBy": "submittedDate", "sortOrder": "descending"}
    )
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    last_exc = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as ex:
            last_exc = ex
            if ex.code == 429:
                wait = int(ex.headers.get("Retry-After", 30))
                print(f"    429 -> honouring Retry-After, sleeping {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            wait = 10 * (attempt + 1)
            print(f"    HTTP {ex.code} -> retry {attempt + 1} in {wait}s", file=sys.stderr)
            time.sleep(wait)
        except Exception as ex:  # URLError / timeout
            last_exc = ex
            wait = 10 * (attempt + 1)
            print(f"    {ex!r} -> retry {attempt + 1} in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise last_exc


def parse(xml_bytes: bytes) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    out = []
    for e in root.findall("a:entry", NS):
        def txt(tag):
            el = e.find(f"a:{tag}", NS)
            return (el.text or "").strip().replace("\n", " ") if el is not None else ""

        def sub(elem, tag):
            sub = elem.find(f"a:{tag}", NS)
            return (sub.text or "").strip() if sub is not None else ""

        aid = txt("id").split("/abs/")[-1]
        out.append({
            "id": aid,
            "title": txt("title"),
            "published": txt("published")[:10],
            "authors": ", ".join(sub(a, "name") for a in e.findall("a:author", NS)),
            "abstract": txt("summary"),
            "cats": ", ".join(c.get("term") or "" for c in e.findall("a:category", NS)),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14, help="recency window in days (default 14)")
    ap.add_argument("--out", type=Path, default=HERE / "_candidates.json")
    args = ap.parse_args()

    cutoff = (datetime.now(timezone.utc) - timedelta(days=args.days)).strftime("%Y-%m-%d")
    known = already_catalogued(HERE / "INDEX.md")

    seen: dict[str, dict] = {}
    for facet, q in QUERIES.items():
        try:
            hits = parse(fetch(q))
        except Exception as ex:
            print(f"=== {facet} FAILED after retries: {ex} ===", file=sys.stderr)
            time.sleep(6)
            continue
        fresh = 0
        for h in hits:
            # The API returns versioned ids; `known` and `seen` are keyed on the
            # unversioned form, or a catalogued paper is re-proposed every run.
            key = base_id(h["id"])
            if key in seen:
                continue
            withdrawn = "withdrawn" in h["abstract"].lower()
            if withdrawn or h["published"] < cutoff or key in known:
                continue
            h["facet"] = facet
            seen[key] = h
            fresh += 1
        print(f"=== {facet}: {len(hits)} raw -> {fresh} new candidates ===")
        time.sleep(6)

    candidates = sorted(seen.values(), key=lambda h: h["published"], reverse=True)
    args.out.write_text(json.dumps(candidates, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nTOTAL new candidates: {len(candidates)} -> {args.out.name}")
    for h in candidates:
        print(f"  [{h['published']}] {h['id']}  {h['facet']:9s}  {h['title'][:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
