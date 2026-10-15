# 🥭 AI Mango Orchard Health & Pest Intelligence

A decision-support web app for mango growers. Upload a photo of a mango leaf (and optionally a canopy or
pest-trap photo) and the system:

1. **Classifies leaf disease** with an EfficientNet-B3 model (Anthracnose, Bacterial Canker, Powdery Mildew,
   Sooty Mold, Leaf Spot, Healthy) and draws a **Grad-CAM heatmap** showing which parts of the leaf drove the prediction.
2. **Estimates lesion area** (% of the leaf that looks diseased) with HSV colour segmentation.
3. **Detects pests** with a YOLOv8 model (Fruit Fly, Mealy Bug, Scale Insect). Mango Hopper and Stem Borer
   are always flagged **UNVERIFIED** because fewer than 100 labelled examples exist (Rule 3).
4. **Fetches live weather** for the orchard from Open-Meteo and forecasts 7-day Anthracnose / Powdery Mildew spread risk.
5. Combines everything into a transparent, rule-based **Orchard Health Index (OHI, 0–100)**
   — see [docs/OHI_METHODOLOGY.md](docs/OHI_METHODOLOGY.md).
6. Shows **treatment protocols with ICAR / TNAU / DPPQS citations**, a bilingual (English/Tamil) advisor chatbot,
   scan history and a downloadable report.

> **Important — model weights are not included.** The repository does not contain trained model files
> (`*.pth`, `*.pt` are git-ignored). Until you train them (see [Training the models](#training-the-models)),
> the app still runs, but disease classification and pest detection are reported as **unavailable**, the OHI is
> marked **PARTIAL**, and no fabricated diagnosis is ever shown. Lesion-area estimation, weather risk, the
> chatbot, farm management and history all work without weights.

---

## Tech stack

| Layer | Technologies |
|---|---|
| Frontend | React 18, Vite 5, Tailwind CSS 3, React Router, Axios, react-dropzone |
| Backend | Python 3.11, FastAPI, Pydantic v2, SQLite (built-in), PyJWT + bcrypt, WeasyPrint (PDF) |
| ML | PyTorch, timm (EfficientNet-B3), Ultralytics YOLOv8, OpenCV, scikit-learn |
| External | Open-Meteo weather API (no key), optional Supabase Auth |

## Folder structure

```
mango-orchard-ai/
├── backend/
│   ├── app/
│   │   ├── api/          # Routes: analyze, auth, farm, report, weather, chatbot
│   │   ├── core/         # config, SQLite store, security (JWT/bcrypt), auth dependency, demo seed
│   │   ├── services/     # pipeline_service: upload validation + ML pipeline bridge
│   │   └── main.py       # FastAPI app
│   ├── tests/            # API tests
│   ├── data/             # SQLite database (created at runtime, git-ignored)
│   ├── media/            # Uploaded images and result overlays (runtime)
│   └── requirements*.txt
├── frontend/             # React app (pages: Dashboard, Scan, History, Chatbot, Farm Admin)
├── ml/
│   ├── pipeline.py       # End-to-end orchestration used by the backend
│   ├── disease_detection/  # EfficientNet-B3 inference, training, evaluation
│   ├── pest_detection/     # YOLOv8 inference + training
│   ├── explainability/     # Hook-based Grad-CAM
│   ├── severity/           # Lesion segmentation + OHI engine
│   ├── forecasting/        # Open-Meteo client, spread forecaster, cited recommendations DB
│   └── datasets/           # Dataset download/preparation scripts and sample images
├── database/             # Supabase/PostgreSQL schema (optional, for a hosted deployment)
├── docs/                 # OHI methodology and source citations
├── conftest.py, pytest.ini, testing_support.py   # Test configuration
└── docker-compose.yml
```

## Prerequisites

- **Python 3.11** (other 3.10–3.12 versions likely work but are untested)
- **Node.js 18+** and npm (tested with Node 24)
- Internet access for the live weather forecast (everything else works offline)
- Optional: Docker Desktop, for the containerised setup
- Optional: GTK/Pango libraries for real PDF output (see Troubleshooting) — otherwise reports are served as printable HTML

## Installation & running (local development)

### 1. Backend

```bash
cd backend
python -m venv venv
# Windows:  venv\Scripts\activate        macOS/Linux:  source venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env          # then set JWT_SECRET to a long random value
uvicorn app.main:app --reload --port 8000
```

Check it: <http://localhost:8000/health> (also shows whether trained models are installed) and the
interactive API docs at <http://localhost:8000/docs>.

On first start an empty database is seeded with a **demo account** — `grower@mango.ai` / `orchard123` —
and a sample orchard. Change or disable it via `DEMO_USER_EMAIL` / `DEMO_USER_PASSWORD` in `.env`.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:3000> and sign in. The Vite dev server proxies `/api`, `/media` and `/health` to the
backend on port 8000 (override with `VITE_PROXY_TARGET`).

### 3. Command-line pipeline (no web app)

```bash
python ml/run_full_pipeline.py                          # uses the bundled sample images
python ml/run_full_pipeline.py --leaf my_leaf.jpg --pest my_trap.jpg --lat 12.52 --lon 78.21
```

### Docker (alternative)

```bash
docker compose up --build
```

Frontend: <http://localhost:3000> (nginx, proxies the API) · Backend: <http://localhost:8000>.
`./ml` is mounted into the backend container, so weights you train on the host are picked up.
Put secrets such as `JWT_SECRET` in `backend/.env`.

## Training the models

1. **Get data**
   - Disease: put a Kaggle token at `~/.kaggle/kaggle.json`, then `python ml/datasets/phase1_disease.py`
     (downloads MangoLeafBD and creates `ml/datasets/processed/disease/{train,val,test}/<Class>/`).
     MangoLeafBD has no *Leaf Spot* class — add a supplemental source or that class cannot be learned.
   - Pests: `export ROBOFLOW_API_KEY=...` then `python ml/datasets/phase1_pest.py`, and build a YOLO `data.yaml`.
   - Without credentials these scripts generate *synthetic cartoon images*; they are only useful for testing
     the code paths, **not** for training a model you would trust.
2. **Train**
   ```bash
   pip install kaggle roboflow           # only needed for dataset download
   python ml/disease_detection/train.py --epochs 25      # -> ml/disease_detection/weights/efficientnet_b3_best.pth
   python ml/disease_detection/evaluate.py               # precision / recall / F1 / confusion matrix
   python ml/pest_detection/train.py --data path/to/data.yaml   # -> ml/pest_detection/weights/yolov8_best.pt
   ```
3. Restart the backend. `/health` should now report `disease_model_trained: true` / `pest_model_trained: true`.

Checkpoints written by `train.py` embed their class list, so labels always map correctly at inference time.

## Testing

From the repository root, with the backend virtual environment active:

```bash
python -m pytest            # 41 tests: ML pipeline, pest/disease detectors, Grad-CAM, OHI, all API endpoints
cd frontend && npm run lint && npm run build
```

Tests use a temporary database and media folder, stub the weather service (no network needed), and exercise the
trained-model code path with a randomly initialised EfficientNet-B3 checkpoint.

## Main API endpoints

| Method & path | Auth | Purpose |
|---|---|---|
| `GET /health` | – | Service status, which models are trained |
| `POST /api/auth/register`, `POST /api/auth/login`, `GET /api/auth/me` | –/–/✓ | Accounts and JWT tokens |
| `POST /api/analyze/` | ✓ | Multipart upload: `file` (leaf), `pest_file`, `scan_type` (`dual`/`disease`/`pest`), `farm_id`, `latitude`, `longitude` |
| `GET /api/analyze/`, `GET /api/analyze/{scan_id}` | ✓ | Scan history / single scan |
| `GET /api/report/generate/{scan_id}` | ✓ | PDF report (HTML fallback) |
| `GET /api/weather/forecast?latitude=&longitude=` | – | 7-day disease-spread forecast (503 if Open-Meteo is unreachable) |
| `GET/POST /api/farm/`, `GET/PUT /api/farm/{id}`, `GET/POST /api/farm/{id}/trees` | ✓ | Orchards and tagged trees |
| `POST /api/chatbot/` | – | English/Tamil advisory chatbot |

## Troubleshooting

| Problem | Fix |
|---|---|
| Scans say "Disease classification unavailable" | Expected until you train models — see above. |
| Report opens as a web page, not a PDF | WeasyPrint needs Pango/Cairo. On Windows install the GTK3 runtime (see WeasyPrint docs); the Docker image already includes them. Use the browser's *Print → Save as PDF* meanwhile. |
| Dashboard shows "Weather service (Open-Meteo) is currently unreachable" | No internet access, or Open-Meteo is down. Scans still work and disclose that default weather was used. |
| Frontend shows "Cannot reach the backend server" | Start the backend on port 8000, or set `VITE_PROXY_TARGET`. |
| Log warns `JWT_SECRET is set to the insecure default` | Set a random `JWT_SECRET` in `backend/.env`. Changing it signs everyone out. |
| Forgot the demo password / want a fresh start | Stop the backend and delete `backend/data/mango_orchard.db`; it is recreated and re-seeded on start. |
| First scan is slow | Model loading and CPU inference; later scans are faster. |

## Known limitations

- No trained weights are shipped, and the bundled sample images are synthetic drawings, not field photos.
- The chatbot is keyword-based, not a language model. English disease/pest answers come from the same cited
  database as scan recommendations; Tamil coverage is limited to Anthracnose plus a greeting.
- Treatment dosages and citations are reproduced from the project's database (`ml/forecasting/recommendations_db.json`)
  and have not been independently re-verified; confirm against current label and extension guidance before use.
- Uploaded images are served from `/media` to anyone who has the (random, unguessable) URL.
- The Supabase SQL schema in `database/` is not used by the running backend, which stores data in SQLite.
