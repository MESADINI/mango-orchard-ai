# Live demo guide

A 5-minute walkthrough of the Mango Orchard Health app on this Windows laptop (RTX 3050; CPU works too).

## One-time setup
| Step | Command (PowerShell, repo root) |
|---|---|
| Python 3.11 backend venv | `python -m venv backend\venv` then `backend\venv\Scripts\pip install -r backend\requirements-dev.txt`. For the GPU build of torch, see the header of `requirements-ml.txt`. |
| Frontend (Node 18+) | `cd frontend; npm ci; cd ..` |
| Model weights | Not in git. Expected at `ml\disease_detection\weights\disease_classifier.pth` and `ml\pest_detection\weights\pest_detector.pt`. Check with `backend\venv\Scripts\python.exe ml\model_registry.py verify` (exit 0 = both match `ml\model_registry.json`). Rebuild with `python tasks.py train-disease` / `train-pest` (see README). |
| Demo account and farm | `backend\venv\Scripts\python.exe demo_data\seed_demo.py` (idempotent) |

## Start and stop
```powershell
powershell -ExecutionPolicy Bypass -File .\run_demo.ps1     # opens http://localhost:3000 when both servers respond
powershell -ExecutionPolicy Bypass -File .\stop_demo.ps1    # stops backend and frontend
```
The launcher checks the weights and GPU, starts the backend (:8000) and frontend (:3000) in two windows, waits until
both respond, then opens the browser. Backend log: `logs\backend.log`.

**Login:** `grower@mango.ai` / `orchard123` (demo-only account).
**Before an audience:** `backend\venv\Scripts\python.exe demo_data\seed_demo.py --clear-history` empties the scan
history.

## The 5-minute script
Sample images are in `demo_data\images\`. Expected outputs come from real runs: `demo_data\MANIFEST.md` and the
rehearsal screenshots in `demo_data\screenshots\`. The OHI includes live weather, so it can differ by a few points
from the values below.

| Time | Click | Image | What appears on screen |
|---|---|---|---|
| 0:00 | Log in | – | Dashboard: orchard *Krishnagiri Mango Valley*, 4 trees, nutrient disclosure banner, model status "Trained" in the sidebar |
| 0:30 | Scroll the dashboard | – | **7-day disease-spread forecast** from Open-Meteo: per day anthracnose / powdery mildew / bacterial canker levels from NHM bulletin thresholds (`02_dashboard_weather.png`) |
| 1:00 | *Scan & Diagnose* → **Disease Only** → drop image → *Run Diagnostic* | `03_bacterial_canker_1.jpg` | **Bacterial Canker, 96%**, lesion area **16.1%**, Grad-CAM heatmap; OHI ≈ **55–56 "High Infestation Risk"**, marked PARTIAL (pest not assessed); TNAU/NHM-cited treatments: sanitation and windbreaks, copper oxychloride 0.3% (`03_…png`). About 1–2 s. |
| 2:00 | **Pests Only** | `10_pest_trap_fruit_fly.jpg` | **2 × Fruit Fly (84%, 83%)**, each badged **"UNVERIFIED: model not trained on mango"**; boxes drawn on the image; no OHI ("No disease or pest assessment is available"); no treatment (`04_…png`) |
| 2:45 | **Dual Scan**: leaf + trap | `05_powdery_mildew_1.jpg` + `10_pest_trap_fruit_fly.jpg` | **Powdery Mildew, 97%**, lesion **20.1%**; the same unverified fruit flies, **excluded from the OHI**; OHI ≈ **56**; wettable sulphur 0.2% (TNAU / NHM) (`05_…png`) |
| 3:15 | **Disease Only** | `11_unsure_external.jpg` (independent image) | **"Unsure — please retake the photo"** banner: top guess Healthy at **70%**, below the 0.815 threshold. No OHI, no treatment (`06_…png`). Its true label is Bacterial Canker: the model would have been wrong, and it abstains. |
| 3:45 | *Scan History* → **Report** on any row | – | Table of the scans just made. The report opens in a new tab with OHI, caveats, penalty breakdown and citations (`07_…`, `08_…png`). It opens as an HTML page (see Troubleshooting); use *Print → Save as PDF*. |
| 4:15 | *Advisor Chatbot* | – | Ask: "How do I treat anthracnose?", "What is the dose for powdery mildew?", "How to control fruit fly?". Each answer lists cited TNAU/NHM measures. Click **தமிழ் (Tamil)** and ask "ஆந்த்ராக்னோஸ் மருந்தளவு என்ன?" (`09_…`, `10_…png`) |
| 4:45 | Back to the dashboard | – | Close with the limitations line below |

Other images if asked: `01/02_anthracnose`, `04_bacterial_canker_2`, `06_powdery_mildew_2`, `07_sooty_mold_1`
(lesion 65%, OHI 45.9 "Severe Outbreak / Critical" without weather), `09_healthy` (Healthy 95%, OHI 100 without weather). `08_sooty_mold_2` is correctly
classified but below the threshold, so it also asks for a retake.

## What to say
- **Problem.** Mango growers lose yield to anthracnose, powdery mildew, bacterial canker and pests. Expert advice
  is scarce at the moment a grower notices symptoms.
- **Pipeline.** One photo → an EfficientNet-B3 leaf-disease classifier with a Grad-CAM heatmap (where the model
  looked) → a lesion-area estimate → optionally a YOLO11n pest detector on a trap or canopy photo → a 7-day
  weather-risk check from NHM bulletin thresholds → treatment advice.
- **Orchard Health Index.** A transparent 0–100 score.
  - It combines the disease, pest and weather penalties with fixed, documented expert-assigned weights (their sensitivity is analysed in the paper).
  - Anything not assessed is left out and disclosed, never assumed healthy; the score is marked PARTIAL.
  - Each penalty is shown, so the number can be checked by hand.
- **Honesty features.**
  - *Retake photo:* when the model is unsure, it says so instead of guessing.
  - *Unverified pests:* the pest model was trained on trap, grape and coffee images, not mango. Its detections are
    labelled as such and kept out of the score.
  - *Cited advice:* every treatment quotes TNAU or NHM, links to the source, and is checked against India's
    banned-pesticide list.
- **Limitations (say this plainly).** Accuracy on the benchmark dataset is very high, but on independent photos it
  is weak: macro-F1 about 0.12. The system has no field validation yet. It is decision support, not a replacement
  for an agronomist.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `ERROR: port 8000 (or 3000) is already in use` | Run `.\stop_demo.ps1`. If another program owns the port, the script names it; close it. |
| "GPU not visible: running on CPU" | The demo still works; scans take a little longer. Check `nvidia-smi`; close other GPU programs; reconnect the charger (some laptops disable the dGPU on battery). |
| "WARNING: model weights missing" / sidebar shows the model as not trained | Copy the two weight files to the paths in *One-time setup*, then run `ml\model_registry.py verify`. Without weights the app still runs and reports "unavailable"; it never invents a result. |
| No internet | The dashboard shows "Weather forecast unavailable…". Scans still work; each scan states that weather was excluded from the OHI. A forecast fetched in the last 10 minutes is reused if the connection drops briefly. |
| Blank page at :3000 | Wait for the `frontend :3000` window to show "ready", then refresh (Ctrl+F5). Check that window and `logs\backend.log` for errors. If it persists: `cd frontend; npm ci`. |
| Login fails | `backend\venv\Scripts\python.exe demo_data\seed_demo.py` restores the demo account and password. |
| Report opens as a web page, not a PDF | WeasyPrint needs the GTK3 runtime on Windows. Use the browser's *Print → Save as PDF*. |
| A heatmap looks rotated compared with the preview | The browser applies the photo's EXIF rotation; the model, as in training, reads the raw pixels. The heatmap shows what the model saw. (`01_anthracnose_1.jpg` has such a tag; the script above avoids it.) |
| A scan says "Analysis failed" | See `logs\backend.log`; restart with `.\stop_demo.ps1` then `.\run_demo.ps1`. |

## Backup plan
- **Screenshots:** `demo_data\screenshots\01_login.png` … `10_chatbot_tamil.png` show every step of the script.
  They are real captures of this app at 1366×768.
- **Recording:** with the app running:
  ```powershell
  cd demo_data; npm install --no-save playwright-core@1.47.2; npx playwright-core install ffmpeg; cd ..
  node demo_data\capture_demo.cjs --video
  ```
  This drives Microsoft Edge through the whole script, refreshes the screenshots and saves
  `demo_data\screenshots\walkthrough.webm`. It also prints the scan times and any browser errors.
- **If the browser path fails but the backend runs:** open `http://localhost:8000/docs`, run `POST /api/auth/login` with the demo
  login, click *Authorize* with the returned token, then run `POST /api/analyze` with a sample image.
