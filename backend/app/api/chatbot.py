"""
Bilingual (English + Tamil) keyword-based advisory chatbot.

English answers for diseases and confirmed pests are generated from the same
source-cited recommendations database used by scan results
(ml/forecasting/recommendations_db.json), so advice stays consistent everywhere.
"""

from typing import Dict, List, Literal, Optional, Tuple

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.config import settings  # noqa: F401  (makes the ml package importable)
from ml.forecasting.recommendations import recommendation_engine

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    language: Literal["en", "ta"] = "en"
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    language: str
    sources: List[str]


# keyword -> recommendations_db condition. More specific phrases first.
EN_KEYWORDS: List[Tuple[str, str]] = [
    ("anthracnose", "Anthracnose"),
    ("black spot", "Anthracnose"),
    ("bacterial canker", "Bacterial Canker"),
    ("canker", "Bacterial Canker"),
    ("powdery mildew", "Powdery Mildew"),
    ("mildew", "Powdery Mildew"),
    ("sooty mo", "Sooty Mold"),  # sooty mold / sooty mould
    ("fruit fly", "Fruit Fly"),
    ("mealy bug", "Mealy Bug"),
    ("mealybug", "Mealy Bug"),
    ("scale", "Scale Insect"),
]

UNRELIABLE_REPLIES: Dict[str, str] = {
    "hopper": (
        "Notice: Automated vision detection for Mango Hopper (Idioscopus spp.) is flagged as unverified "
        "under Inviolable Rule 3 (sample size < 100 images). For field management, ICAR-CISH recommends spraying "
        "Thiamethoxam 25% WG @ 0.25 g/L or Imidacloprid 17.8% SL @ 0.3 mL/L at panicle emergence. "
        "(Citations: ICAR-CISH & DPPQS IPM Guidelines)."
    ),
    "stem borer": (
        "Notice: Automated vision detection for Stem Borer (Batocera rufomaculata) is flagged as unverified "
        "under Inviolable Rule 3. Agronomic management: clean boreholes with a wire spoke and inject "
        "Dichlorvos 76% EC @ 5 mL/hole or plug with mud soaked in kerosene. (Citation: TNAU Agritech Portal)."
    ),
}

EN_WELCOME = (
    "Welcome to the Mango Health AI Advisor. You can ask about diseases (Anthracnose, Powdery Mildew, "
    "Bacterial Canker, Sooty Mold) or pests (Fruit Fly, Mealy Bug, Scale Insect, Mango Hopper, Stem Borer). "
    "Advice is drawn from ICAR and TNAU publications."
)

TA_ANTHRACNOSE_KEYWORDS = ["anthracnose", "ஆந்த்ராக்னோஸ்", "கருப்பு புள்ளி", "தீ கருகல்", "இலை நோய்"]
TA_ANTHRACNOSE_REPLY = (
    "மா மரத்தில் ஆந்த்ராக்னோஸ் (இலைக்கருகல் நோய்) தென்பட்டால், "
    "காப்பர் ஆக்சிகுளோரைடு 50% WP மருந்தை ஒரு லிட்டர் தண்ணீருக்கு 3 கிராம் "
    "என்ற அளவில் கலந்து தெளிக்கவும். புதிய தளிர்கள் வரும்போது அசோக்சிஸ்ட்ரோபின் "
    "(1 மி.லி / லிட்டர்) தெளிப்பது பூஞ்சானைக் கட்டுப்படுத்தும். "
    "(ஆதாரம்: தமிழ்நாடு வேளாண்மைப் பல்கலைக்கழகம் - TNAU)."
)
TA_WELCOME = (
    "வணக்கம்! உங்கள் மாந்தோட்டத்தின் இலை நோய் அல்லது பூச்சி தாக்குதல் பற்றிய "
    "கேள்விகளை கேட்கலாம். துல்லியமான மேலாண்மை முறைகளை ICAR மற்றும் TNAU வழிகாட்டுதலின்படி வழங்குகிறோம்."
)


def _reply_from_db(condition: str) -> Tuple[str, List[str]]:
    recs = recommendation_engine.get_recommendations_for_condition(condition)
    if not recs:
        return EN_WELCOME, ["ICAR-CISH Lucknow", "TNAU Agritech Portal"]
    lines = [f"Recommended management for {condition}:"]
    for i, r in enumerate(recs, 1):
        waiting = f" Waiting period: {r['waiting_period_days']} days." if r.get("waiting_period_days") else ""
        lines.append(
            f"{i}. {r['title']} - {r['product_name']} at {r['dosage_rate']}. {r['application_guidance']}{waiting}"
        )
    sources = list(dict.fromkeys(r["citation_source"] for r in recs))
    return "\n".join(lines), sources


def _english_reply(msg: str) -> Tuple[str, List[str]]:
    for keyword, reply in UNRELIABLE_REPLIES.items():
        if keyword in msg:
            return reply, ["ICAR-CISH Lucknow", "TNAU Agritech Portal"]
    for keyword, condition in EN_KEYWORDS:
        if keyword in msg:
            return _reply_from_db(condition)
    return EN_WELCOME, ["ICAR-CISH Lucknow", "TNAU Agritech Portal"]


@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Bilingual agricultural advisor endpoint (English + Tamil)
    delivering source-cited recommendations from ICAR and TNAU.
    """
    msg = request.message.lower()

    if request.language == "ta":
        reply = TA_ANTHRACNOSE_REPLY if any(k in msg for k in TA_ANTHRACNOSE_KEYWORDS) else TA_WELCOME
        return ChatResponse(reply=reply, language="ta", sources=["TNAU Agritech Portal", "ICAR-CISH"])

    reply, sources = _english_reply(msg)
    return ChatResponse(reply=reply, language="en", sources=sources)
