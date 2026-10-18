# Starts the live demo: FastAPI backend on :8000 and the React (Vite) frontend on :3000, each in its own window,
# waits until both respond, then opens the browser. Stop everything with .\stop_demo.ps1.
#   powershell -ExecutionPolicy Bypass -File .\run_demo.ps1 [-NoBrowser]
param([switch]$NoBrowser)
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Backend = Join-Path $Root "backend"
$Frontend = Join-Path $Root "frontend"
$Python = Join-Path $Backend "venv\Scripts\python.exe"
$PidFile = Join-Path $Root ".demo_pids.json"
$LogDir = Join-Path $Root "logs"
$AppUrl = "http://localhost:3000"
$HealthUrl = "http://127.0.0.1:8000/health"

function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

# 1. Prerequisites
if (-not (Test-Path $Python)) { Fail "backend venv not found. One-time setup: see DEMO.md (python -m venv backend\venv ...)." }
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { Fail "npm not found. Install Node.js 18+ (see DEMO.md)." }
if (-not (Test-Path (Join-Path $Frontend "node_modules"))) { Fail "frontend\node_modules missing. Run once: cd frontend; npm ci" }
foreach ($port in 8000, 3000) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
        Fail "port $port is already in use. Run .\stop_demo.ps1 (or close the program using it) and try again."
    }
}

# 2. Model weights must match the registry (the app still runs without them, but shows 'unavailable')
& $Python (Join-Path $Root "ml\model_registry.py") verify *> $null
if ($LASTEXITCODE -eq 0) { Write-Host "Model weights: OK (match ml\model_registry.json)" -ForegroundColor Green }
else { Write-Host "WARNING: model weights missing or not matching the registry; scans will report models as unavailable. See DEMO.md > Troubleshooting." -ForegroundColor Yellow }

# 3. GPU check (CPU fallback is automatic)
$gpu = & $Python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else '')" 2>$null
if ($gpu) { Write-Host "GPU: $gpu" -ForegroundColor Green }
else { Write-Host "GPU not visible: running on CPU (scans take a little longer)." -ForegroundColor Yellow }

# 4. Start backend and frontend in separate windows (backend log also written to logs\backend.log)
New-Item -ItemType Directory -Force $LogDir | Out-Null
$backendCmd = "`$Host.UI.RawUI.WindowTitle = 'Mango demo - backend :8000'; Set-Location '$Backend'; " +
              "& '.\venv\Scripts\Activate.ps1'; " +
              "cmd /c 'python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 2>&1' | Tee-Object -FilePath '$LogDir\backend.log'"
$frontendCmd = "`$Host.UI.RawUI.WindowTitle = 'Mango demo - frontend :3000'; Set-Location '$Frontend'; " +
               "npm run dev -- --port 3000 --strictPort"
$be = Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $backendCmd -PassThru
$fe = Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $frontendCmd -PassThru
@{ backend = $be.Id; frontend = $fe.Id } | ConvertTo-Json | Set-Content -Encoding utf8 $PidFile
Write-Host "Started backend (window PID $($be.Id)) and frontend (window PID $($fe.Id))."

# 5. Wait until both respond (the backend loads and warms up the models first)
function Wait-Url($url, $seconds) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        try { $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5; if ($r.StatusCode -eq 200) { return $true } } catch {}
        Start-Sleep -Seconds 2
    }
    return $false
}
Write-Host -NoNewline "Waiting for backend $HealthUrl ... "
if (-not (Wait-Url $HealthUrl 180)) { Fail "backend did not respond within 180 s. Check the 'backend :8000' window for the error." }
Write-Host "ready" -ForegroundColor Green
Write-Host -NoNewline "Waiting for frontend $AppUrl ... "
if (-not (Wait-Url $AppUrl 90)) { Fail "frontend did not respond within 90 s. Check the 'frontend :3000' window for the error." }
Write-Host "ready" -ForegroundColor Green

$models = (Invoke-RestMethod "http://127.0.0.1:8000/api/models")
Write-Host ("Disease model {0}: {1} | Pest model {2}: {3} (trained on non-mango data)" -f `
    $models.disease_classifier.version, $models.disease_classifier.status, $models.pest_detector.version, $models.pest_detector.status)
Write-Host "Demo login: grower@mango.ai / orchard123   (stop with .\stop_demo.ps1)" -ForegroundColor Cyan
if (-not $NoBrowser) { Start-Process $AppUrl }
