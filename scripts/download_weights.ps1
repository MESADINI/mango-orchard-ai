# Downloads the deployed model weights from the GitHub release and verifies each SHA-256 against
# ml/model_registry.json. Files that are already present and correct are skipped.
#   powershell -ExecutionPolicy Bypass -File .\scripts\download_weights.ps1 [-Repo owner/name] [-Tag v1.0]
# Until the release is published, its assets are visible only to collaborators: the script then falls back to the
# GitHub CLI (`gh release download`, after `gh auth login`).
param([string]$Repo = "MESADINI/mango-orchard-ai", [string]$Tag = "v1.0")
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$Root = Split-Path $PSScriptRoot -Parent
$registry = (Get-Content (Join-Path $Root "ml\model_registry.json") -Raw | ConvertFrom-Json).models

function Get-Sha([string]$p) { (Get-FileHash -Algorithm SHA256 $p).Hash.ToLower() }

function Get-Asset([string]$name, [string]$dest) {
    New-Item -ItemType Directory -Force (Split-Path $dest -Parent) | Out-Null
    $tmp = "$dest.download"
    try {
        Invoke-WebRequest -Uri "https://github.com/$Repo/releases/download/$Tag/$name" -OutFile $tmp -UseBasicParsing
    } catch {
        if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
            throw "Could not download $name from release $Tag of $Repo (not published yet?). Install the GitHub CLI and run 'gh auth login', or ask the repository owner to publish the release."
        }
        Write-Host "  public download failed; trying the GitHub CLI (release may still be a draft)"
        $dir = Split-Path $dest -Parent
        gh release download $Tag -R $Repo -p $name -D $dir --clobber
        if ($LASTEXITCODE -ne 0) { throw "gh release download failed for $name" }
        Move-Item -Force (Join-Path $dir $name) $tmp
    }
    Move-Item -Force $tmp $dest
}

$failed = $false
foreach ($m in $registry) {
    $dest = Join-Path $Root $m.path
    $name = Split-Path $m.path -Leaf
    if ((Test-Path $dest) -and ((Get-Sha $dest) -eq $m.sha256)) { Write-Host "OK (present)  $name"; continue }
    Write-Host "Downloading   $name ..."
    Get-Asset $name $dest
    if ((Get-Sha $dest) -eq $m.sha256) { Write-Host "OK (verified) $name  sha256 $($m.sha256.Substring(0,12))" }
    else { Remove-Item -Force $dest; Write-Host "SHA-256 MISMATCH for $name; file removed." -ForegroundColor Red; $failed = $true }
    if ($m.name -eq "pest_detector") {
        # deployment metadata (classes, host_verified flags) must describe exactly this checkpoint
        $meta = [IO.Path]::ChangeExtension($dest, ".json")
        Get-Asset (Split-Path $meta -Leaf) $meta
        if ((Get-Content $meta -Raw | ConvertFrom-Json).sha256 -ne $m.sha256) {
            Remove-Item -Force $meta; Write-Host "pest_detector.json does not match the registry; removed." -ForegroundColor Red; $failed = $true
        }
    }
}
if ($failed) { exit 1 }
Write-Host "All model weights match ml\model_registry.json."
