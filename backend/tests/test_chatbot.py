"""Chatbot matching, Tamil coverage and safe fallback (no HTTP needed)."""

import pytest

from app.api.chatbot import KEYWORDS, TA_NAMES, build_reply, match_conditions
from ml.forecasting.recommendations import recommendation_engine


def test_every_condition_has_english_and_tamil_keywords_and_db_advice():
    assert set(KEYWORDS) == set(TA_NAMES) == set(recommendation_engine.conditions)
    for cond, kws in KEYWORDS.items():
        assert kws["en"] and kws["ta"], cond


@pytest.mark.parametrize("msg", ["large-scale orchard planning", "how do I escalate this?", "upscale market prices",
                                 "my borerline question"])
def test_english_matches_whole_words_only(msg):
    assert match_conditions(msg, "en") == []


@pytest.mark.parametrize("msg, expected", [
    ("Scales on my leaves", ["Scale Insect"]),
    ("lots of fruit flies this year", ["Fruit Fly"]),
    ("Mealybugs climbing the trunk", ["Mealy Bug"]),
    ("sooty mould after hoppers", ["Sooty Mold", "Mango Hopper"]),
    ("POWDERY MILDEW!", ["Powdery Mildew"]),
])
def test_english_matching(msg, expected):
    assert match_conditions(msg, "en") == expected


@pytest.mark.parametrize("msg, expected", [
    ("சாம்பல் நோய்க்கு என்ன மருந்து?", ["Powdery Mildew"]),          # case suffix on the keyword
    ("இலைகளில் கரும்பூஞ்சை உள்ளது", ["Sooty Mold"]),
    ("பழ ஈ தாக்குதல்", ["Fruit Fly"]),
    ("மாவுப்பூச்சிகள் அதிகம்", ["Mealy Bug"]),
    ("செதில் பூச்சி மற்றும் தத்துப்பூச்சி", ["Scale Insect", "Mango Hopper"]),
    ("தண்டு துளைப்பான் பிரச்சினை", ["Stem Borer"]),
    ("anthracnose பற்றி சொல்லுங்கள்", ["Anthracnose"]),               # English name inside a Tamil question
])
def test_tamil_matching(msg, expected):
    assert match_conditions(msg, "ta") == expected


def test_tamil_keyword_inside_another_word_does_not_match():
    # "மாவுப்பூச்சி" must start a word; here it is the end of a longer word
    assert match_conditions("கோதுமைமாவுப்பூச்சி", "ta") == []


def test_unknown_question_gets_safe_fallback_not_a_guess():
    r = build_reply("My coconut tree has yellow leaves", "en")
    assert r.matched_conditions == [] and r.sources == []
    assert "will not guess" in r.reply and "agriculture officer" in r.reply
    t = build_reply("தென்னை மரத்தில் மஞ்சள் இலைகள்", "ta")
    assert t.matched_conditions == [] and "KVK" in t.reply


def test_greeting_gets_welcome():
    assert "Welcome" in build_reply("hello", "en").reply
    assert "வணக்கம்" in build_reply("வணக்கம்", "ta").reply


def test_unverified_pests_carry_a_notice_in_both_languages():
    assert "unverified" in build_reply("stem borer", "en").reply
    assert "சரிபார்க்கப்படவில்லை" in build_reply("தண்டு துளைப்பான்", "ta").reply
    assert "unverified" not in build_reply("fruit fly", "en").reply


def test_tamil_reply_keeps_doses_from_the_database_verbatim():
    r = build_reply("சாம்பல் நோய்", "ta")
    dose = recommendation_engine.get_recommendations_for_condition("Powdery Mildew")[0]["dosage_rate"]
    assert dose in r.reply and "லேபிளில்" in r.reply


def test_reply_is_capped_at_three_conditions():
    r = build_reply("anthracnose canker mildew sooty scale", "en")
    assert len(r.matched_conditions) == 3


def test_cultural_advice_omits_missing_product_and_dose():
    from app.api.chatbot import _advice_lines
    lines, _, _ = _advice_lines("Fruit Fly")
    assert lines and not any("N/A" in line for line in lines)
