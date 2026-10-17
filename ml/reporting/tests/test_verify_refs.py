"""Reference checker: BibTeX parsing, title matching and the pass/fail rules (registries stubbed, no network)."""

from ml.reporting import verify_refs as V

BIB = """@article{a2020,
  title   = {{MangoLeafBD}: A comprehensive image dataset},
  year    = {2023},
  doi     = {10.1/x}
}

@misc{web,
  title        = {Some page},
  howpublished = {\\url{https://example.org}}
}
"""


def test_parse_bib():
    entries = V.parse_bib(BIB)
    assert [e["key"] for e in entries] == ["a2020", "web"]
    assert entries[0]["doi"] == "10.1/x" and entries[0]["type"] == "article"


def test_title_match_ignores_latex_and_case():
    assert V.title_match("{MangoLeafBD}: A comprehensive image dataset", "MangoLeafBD - a comprehensive IMAGE dataset") == 1.0
    assert V.title_match("Deep residual learning", "Attention is all you need") == 0.0


def test_check_rules():
    a, web = V.parse_bib(BIB)
    ok = {"doi": lambda d: {"title": "MangoLeafBD: a comprehensive image dataset", "year": 2023, "venue": ""}}
    assert V.check(a, ok)["status"] == "ok"
    wrong_year = {"doi": lambda d: {"title": "MangoLeafBD: a comprehensive image dataset", "year": 2019, "venue": ""}}
    assert V.check(a, wrong_year)["status"] == "mismatch"
    wrong_title = {"doi": lambda d: {"title": "Something else entirely", "year": 2023, "venue": ""}}
    assert V.check(a, wrong_title)["status"] == "mismatch"

    def boom(_):
        raise OSError("offline")
    assert V.check(a, {"doi": boom})["status"] == "error"          # a network failure never passes
    assert V.check(web)["status"] == "not machine-verified"
