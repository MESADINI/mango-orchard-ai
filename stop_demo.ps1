# Stops the demo started by run_demo.ps1: the two windows and their child processes (uvicorn, node), then anything
# still listening on :8000 / :3000 that belongs to this demo (python or node).
#   powershell -ExecutionPolicy Bypass -File .\stop_demo.ps1
$Root = $PSScriptRoot
$PidFile = Join-Path $Root ".demo_pids.json"

if (Test-Path $PidFile) {
    $ids = Get-Content $PidFile -Raw | ConvertFrom-Json
    foreach ($name in "backend", "frontend") {
        $id = $ids.$name
        if ($id -and (Get-Process -Id $id -ErrorAction SilentlyContinue)) {
            taskkill /PID $id /T /F *> $null
            Write-Host "Stopped $name (PID $id and children)."
        }
    }
    Remove-Item $PidFile -Force
}

foreach ($port in 8000, 3000) {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($procId in ($conns | Select-Object -ExpandProperty OwningProcess -Unique)) {
        $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if ($p -and $p.ProcessName -in @("python", "node", "uvicorn")) {
            taskkill /PID $procId /T /F *> $null
            Write-Host "Stopped $($p.ProcessName) on port $port (PID $procId)."
        } elseif ($p) {
            Write-Host "Port $port is used by $($p.ProcessName) (PID $procId), not by the demo; left running." -ForegroundColor Yellow
        }
    }
}
Write-Host "Demo stopped."
