"""
Bilingual (English + Tamil) keyword-based advisory chatbot.

Answers for every disease and pest come from the same verified, source-cited recommendations database used by
scan results (ml/forecasting/recommendations_db.json), so advice stays consistent everywhere and nothing is
invented here.

Matching:
- English: whole words (optional plural "s"/"es"; hyphens join words), so "scale" does not match "large-scale".
- Tamil: a keyword must start a word (Tamil adds case suffixes, e.g. "சாம்பல் நோய்க்கு"), so keywords are matched
  as word prefixes rather than with \\b, which does not work across Tamil vowel signs.
- No match -> a safe fallback that lists what the advisor covers and points to the local agriculture officer.
Tamil replies keep product names and doses in English, exactly as in the cited source, to avoid translation
errors in quantities.
"""

import re
from typing import Dict, List, Literal, Optional, Tuple

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.config import settings  # noqa: F401  (makes the ml package importable)
from ml.forecasting.recommendations import recommendation_engine
from ml.pest_detection.inference import UNRELIABLE_CLASSES

router = APIRouter()
MAX_CONDITIONS_PER_REPLY = 3


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    language: Literal["en", "ta"] = "en"
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    language: str
    sources: List[str]
    source_urls: List[str] = []
    matched_conditions: List[str] = []


# condition -> English keywords (whole words) and Tamil keywords (word prefixes)
KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "Anthracnose": {"en": ["anthracnose", "black spot", "blossom blight"],
                    "ta": ["ஆந்த்ராக்னோஸ்", "இலைக்கருகல்", "இலை கருகல்", "கருப்பு புள்ளி", "கரும்புள்ளி"]},
    "Bacterial Canker": {"en": ["bacterial canker", "canker", "bacterial spot"],
                         "ta": ["பாக்டீரியா", "பாக்டீரியல்", "கேங்கர்", "கேன்கர்", "அழுகல் புள்ளி"]},
    "Powdery Mildew": {"en": ["powdery mildew", "mildew", "white powder"],
                       "ta": ["சாம்பல் நோய்", "சாம்பல்நோய்", "வெண்பூஞ்சை", "வெள்ளை பூஞ்சை", "பவுடரி"]},
    "Sooty Mold": {"en": ["sooty mold", "sooty mould", "sooty", "black mold", "black mould"],
                   "ta": ["கரும்பூஞ்சை", "கரும் பூஞ்சை", "கரும்பூசணம்", "சூட்டி"]},
    "Fruit Fly": {"en": ["fruit fly", "fruit flies", "fruitfly"],
                  "ta": ["பழ ஈ", "பழஈ", "பழ ஈக்கள்"]},
    "Mealy Bug": {"en": ["mealy bug", "mealybug"],
                  "ta": ["மாவுப்பூச்சி", "மாவு பூச்சி"]},
    "Scale Insect": {"en": ["scale insect", "scale", "scales"],
                     "ta": ["செதில் பூச்சி", "செதில்பூச்சி"]},
    "Mango Hopper": {"en": ["hopper", "leafhopper", "mango hopper"],
                     "ta": ["தத்துப்பூச்சி", "தத்து பூச்சி"]},
    "Stem Borer": {"en": ["stem borer", "trunk borer", "borer"],
                   "ta": ["தண்டு துளைப்பான்", "தண்டுத்துளைப்பான்", "துளைப்பான்"]},
}

TA_NAMES = {
    "Anthracnose": "ஆந்த்ராக்னோஸ் (இலைக்கருகல்)", "Bacterial Canker": "பாக்டீரியா கேங்கர்",
    "Powdery Mildew": "சாம்பல் நோய்", "Sooty Mold": "கரும்பூஞ்சை", "Fruit Fly": "பழ ஈ",
    "Mealy Bug": "மாவுப்பூச்சி", "Scale Insect": "செதில் பூச்சி", "Mango Hopper": "தத்துப்பூச்சி",
    "Stem Borer": "தண்டு துளைப்பான்",
}

COVERED_EN = ", ".join(KEYWORDS)
COVERED_TA = ", ".join(TA_NAMES.values())

EN_WELCOME = (
    "Welcome to the Mango Health AI Advisor. Ask about a mango disease (Anthracnose, Powdery Mildew, "
    "Bacterial Canker, Sooty Mold) or pest (Fruit Fly, Mealy Bug, Scale Insect, Mango Hopper, Stem Borer). "
    "Advice is quoted from TNAU and National Horticulture Mission publications."
)
EN_FALLBACK = (
    "I could not find a mango disease or pest I know in your question, so I will not guess. I can advise on: "
    f"{COVERED_EN}. For any other problem, please contact your local agriculture officer or Krishi Vigyan Kendra (KVK)."
)
TA_WELCOME = (
    "வணக்கம்! மா மரத்தின் நோய் அல்லது பூச்சி பற்றிக் கேளுங்கள்: " + COVERED_TA + ". "
    "பதில்கள் TNAU மற்றும் தேசிய தோட்டக்கலை இயக்க (NHM) வெளியீடுகளில் இருந்து எடுக்கப்பட்டவை."
)
TA_FALLBACK = (
    "மன்னிக்கவும், உங்கள் கேள்வியில் உள்ள நோய் அல்லது பூச்சியை என்னால் அடையாளம் காண முடியவில்லை; "
    "ஊகித்து பதில் சொல்ல மாட்டேன். நான் பதில் அளிக்கக்கூடியவை: " + COVERED_TA + ". "
    "பிற பிரச்சினைகளுக்கு உங்கள் அருகிலுள்ள வேளாண் அலுவலர் அல்லது வேளாண் அறிவியல் நிலையத்தை (KVK) அணுகவும்."
)
GREETINGS = {"en": ("hi", "hello", "hey", "help", "start"), "ta": ("வணக்கம்", "உதவி")}

EN_UNVERIFIED = ("Note: automatic photo detection of {name} is unverified (fewer than 100 training images), so a "
                 "scan will never recommend this treatment; the advice below is for a pest you have confirmed "
                 "yourself.")
TA_UNVERIFIED = ("குறிப்பு: {name} படங்கள் மூலம் கண்டறிதல் சரிபார்க்கப்படவில்லை (100-க்கும் குறைவான பயிற்சி "
                 "படங்கள்). நீங்களே உறுதி செய்த பூச்சிக்கு மட்டும் கீழ்க்கண்ட ஆலோசனையைப் பயன்படுத்தவும்.")
EN_LABEL = "Before using any product, check that its label allows use on mango and follow its pre-harvest interval."
TA_LABEL = ("எந்த மருந்தையும் பயன்படுத்தும் முன், அதன் லேபிளில் மா பயிருக்கு அனுமதி உள்ளதா என்பதையும் "
            "அறுவடைக்கு முந்தைய இடைவெளியையும் சரிபார்க்கவும்.")
TA_DOSES_NOTE = "(அளவுகள் தவறாக மொழிபெயர்க்கப்படாமல் இருக்க, மருந்து பெயர்களும் அளவுகளும் ஆங்கிலத்தில் உள்ளன.)"


def _en_pattern(keyword: str) -> re.Pattern:
    # hyphens count as part of a word, so "large-scale" is not "scale"
    return re.compile(r"(?<![\w-])" + re.escape(keyword) + r"(?:s|es)?(?![\w-])", re.IGNORECASE)


def _ta_pattern(keyword: str) -> re.Pattern:
    return re.compile(r"(?:^|[\s\.,;:!?\"'()\-])" + re.escape(keyword))


PATTERNS = {lang: [(cond, (_en_pattern if lang == "en" else _ta_pattern)(kw))
                   for cond, kws in KEYWORDS.items() for kw in kws[lang]]
            for lang in ("en", "ta")}


def match_conditions(message: str, language: str) -> List[str]:
    """Conditions mentioned in the message, in order of first mention. Tamil questions may also use English names."""
    hits: Dict[str, int] = {}
    langs = ("ta", "en") if language == "ta" else ("en",)
    for lang in langs:
        for cond, pat in PATTERNS[lang]:
            m = pat.search(message)
            if m and (cond not in hits or m.start() < hits[cond]):
                hits[cond] = m.start()
    ordered = sorted(hits, key=hits.get)
    # a specific phrase ("stem borer") and its generic word ("borer") map to the same condition, so no duplicates
    return ordered[:MAX_CONDITIONS_PER_REPLY]


def _advice_lines(condition: str) -> Tuple[List[str], List[str], List[str]]:
    recs = recommendation_engine.get_recommendations_for_condition(condition)
    lines = []
    for i, r in enumerate(recs, 1):
        phi = (f" Last spray at least {r['waiting_period_days']} days before harvest."
               if r.get("waiting_period_days") is not None else "")
        # cultural measures have no product or dose ("N/A"); leave those fields out instead of printing "N/A, N/A"
        product = ", ".join(v for v in (r["product_name"], r["dosage_rate"]) if v and v != "N/A")
        lines.append(f"{i}. {r['title']}: {product + '. ' if product else ''}{r['application_guidance']}{phi}"
                     f" [Source: {r['citation_source']}]")
    return lines, list(dict.fromkeys(r["citation_source"] for r in recs)), \
        list(dict.fromkeys(r["citation_url"] for r in recs))


def build_reply(message: str, language: str) -> ChatResponse:
    text = message.strip()
    conditions = match_conditions(text, language)
    if not conditions:
        is_greeting = text.lower().strip("!. ") in GREETINGS[language] or not text
        reply = (EN_WELCOME if is_greeting else EN_FALLBACK) if language == "en" else \
            (TA_WELCOME if is_greeting else TA_FALLBACK)
        return ChatResponse(reply=reply, language=language, sources=[], source_urls=[], matched_conditions=[])

    blocks, sources, urls = [], [], []
    for cond in conditions:
        lines, srcs, links = _advice_lines(cond)
        sources += srcs
        urls += links
        if language == "en":
            head = [EN_UNVERIFIED.format(name=cond)] if cond in UNRELIABLE_CLASSES else []
            blocks.append("\n".join(head + [f"Recommended management for {cond}:"] + lines))
        else:
            head = [TA_UNVERIFIED.format(name=TA_NAMES[cond])] if cond in UNRELIABLE_CLASSES else []
            blocks.append("\n".join(head + [f"{TA_NAMES[cond]} ({cond}) - பரிந்துரைக்கப்பட்ட மேலாண்மை "
                                            f"{TA_DOSES_NOTE}"] + lines))
    blocks.append(EN_LABEL if language == "en" else TA_LABEL)
    return ChatResponse(reply="\n\n".join(blocks), language=language, sources=list(dict.fromkeys(sources)),
                        source_urls=list(dict.fromkeys(urls)), matched_conditions=conditions)


@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Bilingual agricultural advisor endpoint (English + Tamil)
    delivering verified, source-cited recommendations.
    """
    return build_reply(request.message, request.language)
