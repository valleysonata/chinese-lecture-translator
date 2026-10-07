# Chinese Lecture Interpreter

A lightweight Windows desktop application designed to help STEM students follow Mandarin-taught lectures in natural English using course slides as context.

---

## Architecture & Progress

- [x] **Phase 1: Microphone + Local Silero VAD**
- [x] **Phase 2: Mandarin ASR Engine (Groq Whisper + Code-Switching)**
- [x] **Phase 3: Fast English Live Translation Layer (Groq Qwen/LLM)**
- [x] **Feat 3.1: Anti-Hallucination & Conservative Meaning Translation**
- [x] **Feat 3.1.1: Pipeline Hardening (VAD speech filtering, session logs, drop metrics)**
- [ ] **Phase 4: Slide Context & Course Glossary Integration**
- [ ] **Phase 5: PyQt6 Transparent Overlay UI**
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

### Offline Regression Tests (no microphone, no API calls)
```powershell
py tests\test_vad.py
py tests\test_hallucination_filter.py
py tests\test_backpressure.py
```
