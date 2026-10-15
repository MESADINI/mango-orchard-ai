# AI Mango Orchard Health & Pest Intelligence — Project Plan

An AI-powered precision agriculture decision support system for mango orchards. The platform detects mango leaf diseases and orchard pests from imagery, computes an Orchard Health Index (OHI), forecasts weather-aware disease spread, and delivers verified treatment recommendations.

---

## 1. Phase Build Sequence & Roadmap

| Phase | Description | Status |
|---|---|---|
| **Phase 0** | **Scaffolding + Architecture + Risk Report** | **COMPLETED** |
| **Phase 1** | **Dataset Acquisition & Preprocessing** (MangoLeafBD + Roboflow Pests) | **COMPLETED** |
| **Phase 2** | **Disease Detection (EfficientNet-B3)** + Hook-based Grad-CAM | **COMPLETED** |
| **Phase 3** | **Pest Detection (YOLOv8)** + Unreliable Class Flagging | **COMPLETED** |
| **Phase 4** | **Severity Assessment + OHI Engine + Weather Forecasting** | **COMPLETED** |
| **Phase 5** | **FastAPI Backend + Supabase PostgreSQL & Auth** | **COMPLETED** |
| **Phase 6** | **React Frontend Dashboard + WeasyPrint PDF Reports** | **COMPLETED** |
| **Phase 7** | **Bilingual Advisory Chatbot (English + Tamil)** | **COMPLETED** |
| **Phase 8** | **Model Comparison Study & Publication Paper Summary** | **COMPLETED** (See `CONFERENCE_PAPER_SUMMARY.md`) |

---

## 2. Core Architectural Principles & Inviolable Rules

1. **Deterministic OHI**: Orchard Health Index is deterministic and rule-based Python with transparent, configurable weights. Never a black box.
2. **Strict Source Citations**: Every dosage, chemical recommendation, and management practice links directly to ICAR (Indian Council of Agricultural Research), TNAU (Tamil Nadu Agricultural University), or DPPQS (Directorate of Plant Protection, Quarantine & Storage) publications.
3. **Unreliable Pest Class Flagging**: Mango Hopper and Stem Borer must NOT be trained on YOLOv8 until $\ge 100$ verified bounding-box annotations exist. They are flagged as `UNRELIABLE_CLASSES` in inference modules and explicitly communicated to growers.
4. **Disclosed Nutrient Deficiency Placeholder**: Nitrogen/nutrient deficiency is an explicitly disclosed placeholder. The OHI input defaults to `0`, the API response specifies `nutrient_input_was_defaulted: true`, and the UI displays this caveat clearly.
5. **Exact Model Architectures**:
   - Disease Detection: **EfficientNet-B3** via `timm` / PyTorch.
   - Pest Detection: **YOLOv8** via Ultralytics.
   - Explainability: **Hook-based Grad-CAM** (PyTorch forward/backward hooks).
6. **Required Libraries**:
   - PDF Generation: **WeasyPrint** (HTML/CSS to PDF).
   - Styling: **Tailwind CSS v3**.
   - Backend: **FastAPI** with Python 3.11.
   - Database & Auth: **Supabase**.

---

## 3. Tech Stack Matrix

- **Frontend**: React 18, Vite, Tailwind CSS v3, Chart.js, React-Chartjs-2, React-Dropzone, Supabase-js
- **Backend**: FastAPI, Uvicorn, Pydantic v2, Python-JOSE, HTTPX, WeasyPrint, SlowAPI
- **ML / Vision**: PyTorch, TorchVision, Timm, Ultralytics YOLOv8, OpenCV, Pillow, Scikit-Learn
- **Weather API**: Open-Meteo API (no key required)
- **Database**: Supabase PostgreSQL with Row Level Security (RLS)
