#!/usr/bin/env python3
"""
Checks every bibliography entry of paper/refs.bib against Crossref (DOI) or arXiv (eprint id).

    python ml/reporting/verify_refs.py [--bib paper/refs.bib]

An entry passes when the registry returns a record whose title matches the .bib title (case, punctuation and
LaTeX braces ignored; token overlap >= 0.8) and whose year differs by at most one (arXiv preprints often predate
the venue). Entries without a DOI or eprint (reports, software, web pages) are listed as "not machine-verified"
and must be checked by hand (their URLs were opened when the citation was added).
Output: results/paper/references_check.json; exit 1 if any DOI/eprint entry fails.
"""

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
BIB = REPO_ROOT / "paper" / "refs.bib"
OUT = REPO_ROOT / "results" / "paper" / "references_check.json"
UA = {"User-Agent": "mango-orchard-ai reference check"}


def parse_bib(text: str) -> List[Dict[str, str]]:
    """Minimal BibTeX reader for this file's style: one field per line, values in braces."""
    entries = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        fields = {"type": m.group(1).lower(), "key": m.group(2).strip()}
        for f in re.finditer(r"^\s*(\w+)\s*=\s*\{(.*)\},?\s*$", m.group(3), re.M):
            fields[f.group(1).lower()] = f.group(2)
        entries.append(fields)
    return entries


def _tokens(title: str) -> List[str]:
    title = re.sub(r"\\[a-zA-Z]+|[{}]", "", title).lower()
    return re.findall(r"[a-z0-9]+", title)


def title_match(a: str, b: str) -> float:
    ta, tb = set(_tokens(a)), set(_tokens(b))
    return len(ta & tb) / max(len(ta | tb), 1)


def _get(url: str, timeout: float = 60) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def crossref(doi: str) -> Optional[dict]:
    m = json.loads(_get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
    year = (m.get("published-print") or m.get("published-online") or m.get("issued") or {}).get("date-parts", [[None]])[0][0]
    title = " ".join([(m.get("title") or [""])[0], *(m.get("subtitle") or [])])   # Crossref splits off subtitles
    return {"title": title, "year": year, "venue": (m.get("container-title") or [""])[0]}


def arxiv(eprint: str) -> Optional[dict]:
    xml = _get("http://export.arxiv.org/api/query?id_list=" + urllib.parse.quote(eprint))
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = ET.fromstring(xml).find("a:entry", ns)
    if entry is None or entry.find("a:title", ns) is None:
        return None
    return {"title": " ".join(entry.find("a:title", ns).text.split()),
            "year": int(entry.find("a:published", ns).text[:4]), "venue": "arXiv"}


def check(entry: Dict[str, str], lookup=None) -> dict:
    lookup = lookup or {"doi": crossref, "eprint": arxiv}
    kind = "doi" if "doi" in entry else "eprint" if "eprint" in entry else None
    if kind is None:
        return {"key": entry["key"], "status": "not machine-verified", "url": entry.get("url") or entry.get("howpublished")}
    try:
        rec = lookup[kind](entry[kind])
    except Exception as e:  # network or registry error is a failure, never a pass
        return {"key": entry["key"], "status": "error", kind: entry[kind], "error": str(e)[:200]}
    if rec is None:
        return {"key": entry["key"], "status": "not found", kind: entry[kind]}
    score = title_match(entry.get("title", ""), rec["title"])
    year_ok = rec["year"] is not None and entry.get("year", "").isdigit() and abs(int(entry["year"]) - rec["year"]) <= 1
    return {"key": entry["key"], kind: entry[kind], "status": "ok" if score >= 0.8 and year_ok else "mismatch",
            "title_overlap": round(score, 3), "registry_title": rec["title"], "registry_year": rec["year"],
            "bib_year": entry.get("year")}


def main(bib: Path = BIB, out: Path = OUT) -> int:
    results = []
    for e in parse_bib(Path(bib).read_text(encoding="utf-8")):
        results.append(check(e))
        print(f"{results[-1]['status']:>22}  {e['key']}", flush=True)
        time.sleep(1)                                    # be polite to the public APIs
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "bib": str(Path(bib).relative_to(REPO_ROOT)), "entries": results}, indent=2),
                   encoding="utf-8")
    return 0 if all(r["status"] in ("ok", "not machine-verified") for r in results) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bib", type=Path, default=BIB)
    sys.exit(main(ap.parse_args().bib))
