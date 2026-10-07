# Public release packaging for the Chinese Lecture Interpreter.
# Run AFTER .\build_exe.ps1:   .\package_release.ps1
#
# Result: releases\ChineseLectureInterpreter-v<APP_VERSION>-Windows-x64.zip
#   Contents: ChineseLectureInterpreter\{exe, _internal\, .env.example,
#             GETTING_STARTED.txt, glossary.example.txt}
#   NEVER included: .env (personal API key), storage\ (session logs/wavs)
param(
    [string]$Version = ""   # defaults to APP_VERSION in config.py
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$exeDir = Join-Path $PSScriptRoot "dist\ChineseLectureInterpreter"
$exe = Join-Path $exeDir "ChineseLectureInterpreter.exe"
if (-not (Test-Path $exe)) {
    Write-Error "Exe not found at $exe - run .\build_exe.ps1 first."
    exit 1
}

if (-not $Version) {
    if ((Get-Content config.py -Raw) -match 'APP_VERSION\s*=\s*"([^"]+)"') {
        $Version = $Matches[1]
    } else {
        Write-Error "Could not read APP_VERSION from config.py; pass -Version."
        exit 1
    }
}

$name = "ChineseLectureInterpreter"
$stageRoot = Join-Path $PSScriptRoot "releases\stage"
$stage = Join-Path $stageRoot $name
$zip = Join-Path $PSScriptRoot "releases\$name-v$Version-Windows-x64.zip"

Write-Host "==> Staging $name v$Version ..."
if (Test-Path $stageRoot) { Remove-Item $stageRoot -Recurse -Force }
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Copy-Item (Join-Path $exeDir "*") $stage -Recurse -Force

# Strip local state and secrets - the personal .env must never ship.
Remove-Item (Join-Path $stage ".env") -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $stage "storage") -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $stage "__pycache__") -Recurse -Force -ErrorAction SilentlyContinue

# User setup files next to the exe
Copy-Item (Join-Path $PSScriptRoot ".env.example") $stage
Copy-Item (Join-Path $PSScriptRoot "glossary.example.txt") $stage -ErrorAction SilentlyContinue

$gettingStarted = @"
Chinese Lecture Interpreter v$Version
=====================================

1. GET AN API KEY (one-time setup)
   - Create a free key at https://console.groq.com/keys
   - Rename ".env.example" to ".env" and paste your key after GROQ_API_KEY=

2. OPEN THE APP
   - Double-click ChineseLectureInterpreter.exe

3. FIRST-RUN CHECK
   - Pick your microphone in the "Mic" dropdown
   - Press "Test mic" and speak: the level meter must bounce
   - (Optional) Drop today's lecture PDF onto the window

4. START LISTENING
   - Press Start. Chinese appears first, English fills in below.
   - Tick "Subtitle bar" for floating captions over your slides.
   - Close the window to stop; a session log is saved to storage\session_logs\

Docs: https://github.com/valleysonata/chinese-lecture-translator
"@
Set-Content -Path (Join-Path $stage "GETTING_STARTED.txt") -Value $gettingStarted -Encoding utf8

Write-Host "==> Compressing ..."
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "releases") | Out-Null
Compress-Archive -Path $stage -DestinationPath $zip -Force

# Fail loudly if a secret or local state slipped in.
$entries = [IO.Compression.ZipFile]::OpenRead($zip).Entries | ForEach-Object { $_.FullName }
[IO.Compression.ZipFile]::OpenRead($zip).Dispose()
$bad = $entries | Where-Object { $_ -match '(^|[/\\])\.env$' -or $_ -match '(^|[/\\])storage[/\\]' }
if ($bad) {
    Write-Error ("Release zip contains forbidden entries:`n" + ($bad -join "`n"))
    exit 1
}
if (-not ($entries | Where-Object { $_ -like "*$name.exe" })) {
    Write-Error "Release zip is missing $name.exe"
    exit 1
}
if (-not ($entries | Where-Object { $_ -like "*.env.example" })) {
    Write-Error "Release zip is missing .env.example"
    exit 1
}

$sizeMb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
$hash = (Get-FileHash $zip -Algorithm SHA256).Hash
Remove-Item $stageRoot -Recurse -Force

Write-Host "==> Release package: $zip  ($sizeMb MB)"
Write-Host "    SHA256: $hash"
Write-Host "    Entries: $($entries.Count) | no .env | no storage\"
