# 🥭 AI Mango Orchard Health & Pest Intelligence

> Precision agriculture decision support system detecting mango leaf diseases and pests, computing Orchard Health Index (OHI), forecasting microclimate pathogen spread, and providing verified ICAR/TNAU-sourced treatment protocols.

---

## System Architecture

```
mango-orchard-ai/
├── backend/            # Python 3.11 + FastAPI backend
├── frontend/           # React 18 + Vite + Tailwind CSS v3
├── ml/                 # ML pipelines (EfficientNet-B3, YOLOv8, Grad-CAM, OHI)
├── database/           # Supabase PostgreSQL schema migrations and seed scripts
└── docs/               # OHI methodology and verified treatment source citations
```

## Quick Start Guide

### 1. Environment Setup

#### Backend
```bash
cd backend
python -m venv venv
# On macOS / Linux:
source venv/bin/activate
# On Windows:
# .\venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env
# Configure SUPABASE_URL, SUPABASE_ANON_KEY in .env

uvicorn app.main:app --reload
# Health verification: http://localhost:8000/health -> {"status":"ok","version":"0.1.0"}
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
# Dashboard available at http://localhost:3000
```

#### Full Stack (Docker)
```bash
docker-compose up --build
```

---

## Phase 1: Dataset Acquisition

Run the dataset ingestion scripts:
```bash
# 1. Ensure Kaggle API token is at ~/.kaggle/kaggle.json (chmod 600)
python ml/datasets/phase1_disease.py

# 2. Ensure ROBOFLOW_API_KEY is exported
export ROBOFLOW_API_KEY="your_key"
python ml/datasets/phase1_pest.py

# Or run the unified orchestrator:
python ml/datasets/download_datasets.py
```

See [ml/datasets/README.md](ml/datasets/README.md) and [docs/OHI_METHODOLOGY.md](docs/OHI_METHODOLOGY.md) for details.
