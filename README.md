# Chinese Lecture Interpreter

A lightweight Windows desktop application designed to help STEM students follow Mandarin-taught lectures in natural English using course slides as context.

---

## Architecture & Progress

- [x] **Phase 1: Microphone + Local Silero VAD**
- [x] **Phase 2: Mandarin ASR Engine (Groq Whisper + Code-Switching)**
- [x] **Phase 3: Fast English Live Translation Layer (Groq Qwen/LLM)**
- [x] **Phase 3.1: Anti-Hallucination & Conservative Meaning Translation**
- [x] **Phase 3.1.1: Pipeline Hardening (VAD speech filtering, session logs, drop metrics)**
- [x] **Phase 3.1.2: Bounded Audio Queue (drop-oldest backpressure, drop counters)**
- [x] **Phase 4: Course Context (lecture PDF slides + optional manual glossary)**
- [x] **Phase 5: PyQt6 Main Window UI (normal window + optional transparent subtitle bar)**
- [ ] **Phase 6: Full 60-90 Minute Endurance Test**

Contributor guidance, repo layout, and commit conventions live in [AGENTS.md](AGENTS.md).

---

## Setup & Configuration

1. **Install Dependencies**:
   ```powershell
   py -m pip install -r requirements.txt
   ```

2. **Configure API Key**:
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

---

## Build the Windows .exe (optional)

```powershell
.\build_exe.ps1
```

Produces `dist\ChineseLectureInterpreter\ChineseLectureInterpreter.exe` — a
windowed app that launches without Python or a terminal (PyQt6 UI, microphone
capture, PDF loading, and the ASR/translation dependencies are all bundled).

The API key is **never bundled**: put a `.env` with `GROQ_API_KEY=...` next
to the exe. Session logs and audio chunks are created in a `storage\` folder
next to the exe as well.

---

## How to Test

### Phase 1: Microphone + Silero VAD Capture
```powershell
py scripts\test_phase1.py --list-devices
py scripts\test_phase1.py
```

### Phase 2: Mandarin ASR Pipeline
```powershell
py scripts\test_phase2.py --file storage\audio_chunks\segment_001_20261007_215400_1.6s.wav
py scripts\test_phase2.py
```

### Phase 3 & 3.1: Fast English Live Translation Pipeline
```powershell
# Run benchmark across Pure English, Pure Mandarin, Mixed, and Edge Cases
py scripts\test_phase3.py --benchmark-cases

# Test translation directly on custom text
py scripts\test_phase3.py --text "這個 node 已經不平衡，所以我們需要做 right rotation。"

# Run continuous live stream: Mic -> VAD -> ASR -> Fast English Translation
# (add --duration 120 for a timed run)
py scripts\test_phase3.py
```
*Live runs write a JSONL session log to `storage\session_logs\` containing every
transcript, translation, latency, and dropped item for after-action review.*

### Phase 4: Course Context (PDF slides + optional glossary)

**Primary: the lecture PDF.** Drop it onto the app window (or pass
`--slides lecture.pdf`). The extracted text is injected into the translation
system prompt as context, so the LLM uses the slide's topic and terminology:

```powershell
py scripts\app.py --slides lecture.pdf        # preload at launch
py scripts\app.py                             # or drag-and-drop the PDF in
```

Scanned/image-only PDFs with no extractable text are reported in the drop
zone and the app continues with glossary-only context.

**Optional/advanced: a manual glossary.** A plain-text file (one term per
line, `#` comments allowed) — see `glossary.example.txt`. Terms are injected
into both the Whisper prompt and the translation system prompt so English
renderings stay consistent:

```powershell
py scripts\app.py --slides lecture.pdf --glossary glossary.txt
py scripts\test_phase3.py --benchmark-cases --glossary glossary.txt
```

### Main Window App (recommended for lectures)
A normal, draggable, closable desktop window: status row (state, mic level,
queue depths, drops, average delay), a PDF drop zone, and a scrolling feed of
translation cards — Chinese line first, English fills in below with a
per-line delay badge; errors show as badges instead of vanishing.

```powershell
# Live lecture mode: window opens idle, press Start (or pass --autostart)
py scripts\app.py --slides lecture.pdf --autostart

# UI smoke test (no mic, no API)
py scripts\app.py --demo --duration 15

# Helpers
py scripts\app.py --list-devices   # pick a microphone index
py scripts\app.py --device 1       # use a specific microphone
```

The Phase 5 transparent subtitle bar is now an **optional mode**: check
"Subtitle bar" in the window (or pass `--overlay`) to show the always-on-top,
click-through bar over your slides; uncheck to hide it. Close the window to
stop — a summary and session log are written to `storage\session_logs\`.
If the window misbehaves, the console pipeline is the fallback launcher:
`py scripts\test_phase3.py --glossary glossary.txt`.

### Offline Regression Tests (no microphone, no API calls)
```powershell
py tests\test_vad.py
py tests\test_hallucination_filter.py
py tests\test_backpressure.py
py tests\test_audio_backpressure.py
py tests\test_error_classification.py
py tests\test_glossary.py
py tests\test_slides.py               # PDF extraction + prompt composition
py tests\test_overlay_ui.py           # opens a window briefly; no mic/API
py tests\test_app_ui.py               # main window self-check; no mic/API
```
