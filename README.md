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
- [x] **Phase 4: Course Glossary Integration (`--glossary` manual glossary file)**
- [x] **Phase 5: PyQt6 Transparent Overlay UI (bottom subtitle bar, click-through)**
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

### Phase 4: Course Glossary
Create a plain-text glossary (one term per line, `#` comments allowed) — see
`glossary.example.txt`. Terms are injected into both the Whisper prompt and the
translation system prompt so English renderings stay consistent:

```powershell
py scripts\test_phase3.py --glossary glossary.txt
py scripts\test_phase3.py --benchmark-cases --glossary glossary.txt
```

### Phase 5: Transparent Subtitle Overlay (recommended for lectures)
A frameless, always-on-top, click-through bar at the bottom of the screen shows
the latest English line large, with the previous 3 lines faded above it:

```powershell
# Live lecture mode (mic + glossary)
py scripts\overlay.py --glossary glossary.txt

# UI smoke test (no mic, no API) and one-shot translation check
py scripts\overlay.py --demo --duration 15
py scripts\overlay.py --text "這個節點已經不平衡" --duration 10

# Helpers
py scripts\overlay.py --list-devices   # pick a microphone index
py scripts\overlay.py --device 1       # use a specific microphone
```
Press Ctrl+C in the console to stop; a summary and session log are written to
`storage\session_logs\`. If the overlay misbehaves, the console pipeline is the
fallback launcher: `py scripts\test_phase3.py --glossary glossary.txt`.

### Offline Regression Tests (no microphone, no API calls)
```powershell
py tests\test_vad.py
py tests\test_hallucination_filter.py
py tests\test_backpressure.py
py tests\test_audio_backpressure.py
py tests\test_error_classification.py
py tests\test_glossary.py
py tests\test_overlay_ui.py           # opens a window briefly; no mic/API
```
