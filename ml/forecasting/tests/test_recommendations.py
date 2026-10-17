"""The treatment database: every entry verified, deep-linked and free of pesticides banned or restricted for mango."""

import json
import re
from pathlib import Path

import pytest

from ml.forecasting.recommendations import DB_PATH, RecommendationEngine

DB = json.loads(Path(DB_PATH).read_text(encoding="utf-8"))
ENTRIES = [(cond, r) for cond, recs in DB.items() if not cond.startswith("_") for r in recs]

# From the CIB&RC "List of pesticides which are banned, refused registration and restricted in use"
# (updated 31.07.2026): banned outright, or banned for mango / for fruits eaten raw.
BANNED_FOR_MANGO = ["malathion", "dimethoate", "dichlorvos", "dichlorovos", "phosphamidon", "tridemorph", "benomyl",
                    "dinocap", "carbaryl", "monocrotophos", "methomyl", "dicofol", "fenthion", "endosulfan"]
NOT_SUGGESTED = ["streptocycline", "streptomycin", "tetracycline"]   # agricultural phase-out ordered in draft


def _suggested_text(r):
    # the product, ingredient and dose fields are what the user is told to apply
    return " ".join(str(r[k]) for k in ("product_name", "active_ingredient", "dosage_rate", "title")).lower()


@pytest.mark.parametrize("cond, rec", ENTRIES)
def test_entry_is_verified_and_deep_linked(cond, rec):
    assert rec["citation_url"].startswith("https://agritech.tnau.ac.in/")
    assert re.search(r"\.(html|pdf)$", rec["citation_url"]), "must link to the page, not the portal home"
    assert len(rec["citation_quote"]) > 20
    assert rec["verified_on"] == DB["_meta"]["verified_on"]
    assert rec["waiting_period_days"] is None or isinstance(rec["waiting_period_days"], int)


@pytest.mark.parametrize("cond, rec", ENTRIES)
def test_no_banned_or_phased_out_chemical_is_suggested(cond, rec):
    text = _suggested_text(rec)
    for name in BANNED_FOR_MANGO + NOT_SUGGESTED:
        assert name not in text, (cond, name)


def test_meta_is_not_a_condition():
    engine = RecommendationEngine()
    assert "_meta" not in engine.conditions
    assert engine.meta["regulatory_list_url"].startswith("https://ppqs.gov.in/")


def test_unverified_pests_are_never_issued_by_a_scan():
    engine = RecommendationEngine()
    # the pipeline passes confirmed pests only; the engine itself just looks conditions up
    assert engine.get_recommendations_for_scan(None, []) == []
    assert engine.get_recommendations_for_scan("Healthy", []) == []


def test_sql_seed_is_generated_from_the_verified_database():
    import importlib.util
    root = Path(DB_PATH).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("generate_seed", root / "database" / "seeds" / "generate_seed.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    assert (root / "database" / "seeds" / "001_seed_recommendations.sql").read_text(encoding="utf-8") == gen.render(DB)
