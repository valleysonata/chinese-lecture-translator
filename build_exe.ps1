# Reproducible Windows .exe build for the Chinese Lecture Interpreter.
# Run from anywhere: .\build_exe.ps1   (repo root is resolved from this file)
# Result: dist\ChineseLectureInterpreter\ChineseLectureInterpreter.exe
param(
    [switch]$SkipInstall   # skip the pyinstaller install/upgrade step
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not $SkipInstall) {
    Write-Host "==> Ensuring PyInstaller is installed..."
    py -m pip install --upgrade pyinstaller | Out-Null
}

Write-Host "==> Building (windowed exe, no console)..."
py -m PyInstaller --noconfirm --clean ChineseLectureInterpreter.spec

if ($LASTEXITCODE -ne 0) {
    Write-Error "PyInstaller failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

$exe = Join-Path $PSScriptRoot "dist\ChineseLectureInterpreter\ChineseLectureInterpreter.exe"
if (-not (Test-Path $exe)) {
    Write-Error "Build finished but exe not found at $exe"
    exit 1
}

$sizeMb = [math]::Round((Get-ChildItem (Split-Path $exe) -Recurse | Measure-Object Length -Sum).Sum / 1MB, 0)
Write-Host "==> Built: $exe  (~$sizeMb MB folder)"
Write-Host "    API key stays external: place a .env with GROQ_API_KEY next to the exe."
