"""Results tables: values come only from results/; missing results are marked pending, never invented."""

import json

import pytest

from ml.reporting import results_tables as RT


def test_missing_results_are_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(RT, "RESULTS", tmp_path)
    rows = RT.headline_rows()
    assert rows and all(v.startswith("pending (`results/") for _, _, v, _ in rows)
    md = RT.results_md()
    assert "_pending (`results/disease/tables/cv_comparison.md`)_" in md


def test_values_are_read_from_results(tmp_path, monkeypatch):
    monkeypatch.setattr(RT, "RESULTS", tmp_path)
    (tmp_path / "pest").mkdir()
    (tmp_path / "pest" / "deployed.json").write_text(json.dumps({"model": "yolo11n", "test_map50_95": 0.4567}))
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "latency.json").write_text(json.dumps({"scan_types": {"dual": {"median_ms": 512.4}}}))
    values = {m: v for _, m, v, _ in RT.headline_rows()}
    assert values["Test mAP50-95 (selected by val; non-mango proxy data)"] == "0.457 (yolo11n)"
    assert values["Median POST /api/analyze latency, dual scan, CPU"] == "512 ms"
    malformed = tmp_path / "ohi"
    malformed.mkdir()
    (malformed / "sensitivity.json").write_text(json.dumps({"results": {}}))   # wrong shape -> pending
    assert next(v for c, m, v, _ in RT.headline_rows() if m.startswith("Max tier")).startswith("pending")


def test_latex_macros_pending_and_values(tmp_path, monkeypatch):
    monkeypatch.setattr(RT, "RESULTS", tmp_path)
    tex = RT.latex_numbers()
    assert r"\newcommand{\pestTestMap}{\pending{pest/deployed.json}}" in tex
    (tmp_path / "pest").mkdir()
    (tmp_path / "pest" / "deployed.json").write_text(json.dumps({"model": "yolo11n", "test_map50_95": 0.4567}))
    tex = RT.latex_numbers()
    assert r"\newcommand{\pestTestMap}{0.457}" in tex
    assert tex.count(r"\newcommand") == len(set(line.split("}")[0] for line in tex.splitlines()
                                                if line.startswith(r"\newcommand")))   # no duplicate macro names


def test_latex_tables_pending_without_results(tmp_path, monkeypatch):
    monkeypatch.setattr(RT, "RESULTS", tmp_path)
    tex = RT.latex_tables()
    for label in ("tab:cv", "tab:pest", "tab:xai", "tab:ohi"):
        assert r"\label{%s}" % label in tex
    assert r"\pending{" in tex and "0." not in tex                     # no number without a results file


def test_latex_tables_from_repository_results():
    if not (RT.RESULTS / "disease" / "summary.json").exists():
        pytest.skip("no disease results")
    tex = RT.latex_tables()
    assert tex.count(r"\begin{table}") == tex.count(r"\end{table}") >= 4


def test_readme_block_replacement():
    text = f"intro\n{RT.START}\nold\n{RT.END}\nrest"
    new = RT.update_readme(text, "| a |")
    assert "old" not in new and "| a |" in new and new.endswith("rest")
    with pytest.raises(ValueError):
        RT.update_readme("no markers", "x")


def test_repository_readme_has_the_block():
    text = RT.README.read_text(encoding="utf-8")
    assert RT.START in text and RT.END in text
