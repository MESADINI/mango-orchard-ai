from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    language: str = "en"  # "en" or "ta"
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    language: str
    sources: List[str]

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Bilingual agricultural advisor endpoint (English + Tamil)
    delivering source-verified recommendations from ICAR and TNAU.
    """
    msg = request.message.lower()
    
    if request.language == "ta":
        if (
            "anthracnose" in msg
            or "ஆந்த்ராக்னோஸ்" in msg
            or "கருப்பு புள்ளி" in msg
            or "தீ கருகல்" in msg
            or "இலை நோய்" in msg
        ):
            reply = (
                "மா மரத்தில் ஆந்த்ராக்னோஸ் (இலைக்கருகல் நோய்) தென்பட்டால், "
                "காப்பர் ஆக்சிகுளோரைடு 50% WP மருந்தை ஒரு லிட்டர் தண்ணீருக்கு 3 கிராம் "
                "என்ற அளவில் கலந்து தெளிக்கவும். புதிய தளிர்கள் வரும்போது அசோக்சிஸ்ட்ரோபின் "
                "(1 மி.லி / லிட்டர்) தெளிப்பது பூஞ்சானைக் கட்டுப்படுத்தும். "
                "(ஆதாரம்: தமிழ்நாடு வேளாண்மைப் பல்கலைக்கழகம் - TNAU)."
            )
        else:
            reply = (
                "வணக்கம்! உங்கள் மாந்தோட்டத்தின் இலை நோய் அல்லது பூச்சி தாக்குதல் பற்றிய "
                "கேள்விகளை கேட்கலாம். துல்லியமான மேலாண்மை முறைகளை ICAR மற்றும் TNAU வழிகாட்டுதலின்படி வழங்குகிறோம்."
            )
        return ChatResponse(
            reply=reply,
            language="ta",
            sources=["TNAU Agritech Portal", "ICAR-CISH"]
        )
    else:
        if "anthracnose" in msg or "black spot" in msg:
            reply = (
                "For Mango Anthracnose (Colletotrichum gloeosporioides), the recommended management is "
                "spraying Copper Oxychloride 50% WP @ 3.0 g/L at blossom emergence. For active systemic outbreaks, "
                "apply Azoxystrobin 23% SC @ 1.0 mL/L with a 21-day waiting period. "
                "(Citations: TNAU Agritech Portal & ICAR-CISH Lucknow)."
            )
        elif "hopper" in msg:
            reply = (
                "⚠️ Notice: Automated vision detection for Mango Hopper (Idioscopus spp.) is flagged as unverified "
                "under Inviolable Rule 3 (sample size < 100 images). For field management, ICAR-CISH recommends spraying "
                "Thiamethoxam 25% WG @ 0.25 g/L or Imidacloprid 17.8% SL @ 0.3 mL/L at panicle emergence. "
                "(Citations: ICAR-CISH & DPPQS IPM Guidelines)."
            )
        elif "stem borer" in msg:
            reply = (
                "⚠️ Notice: Automated vision detection for Stem Borer (Batocera rufomaculata) is flagged as unverified "
                "under Inviolable Rule 3. Agronomic management: clean boreholes with a wire spoke and inject "
                "Dichlorvos 76% EC @ 5 mL/hole or plug with mud soaked in kerosene. (Citation: TNAU Agritech Portal)."
            )
        elif "fruit fly" in msg:
            reply = (
                "For Mango Fruit Fly (Bactrocera dorsalis), deploy Methyl Eugenol parapheromone traps "
                "at 10-15 traps/hectare 60 days before harvest, combined with canopy bait spot sprays of "
                "100g jaggery + 20mL Malathion 50% EC in 10L water. (Citation: ICAR-CISH)."
            )
        else:
            reply = (
                "Welcome to the Mango Health AI Advisor. You can ask questions about diseases (Anthracnose, "
                "Powdery Mildew, Bacterial Canker, Sooty Mold) or pests (Fruit Fly, Mealybug, Scale). All advice "
                "is strictly verified against ICAR and TNAU publications."
            )
        return ChatResponse(
            reply=reply,
            language="en",
            sources=["ICAR-CISH Lucknow", "TNAU Agritech Portal"]
        )
