# Chinese Lecture Interpreter 

A lightweight Windows desktop application designed to help STEM student follow Mandarin-taught lectures in natural English using course slides as context.

---

## Architecture & Progress

- [x] **Phase 1: Microphone + Local Silero VAD**
- [x] **Phase 2: Mandarin ASR Engine (Groq Whisper Cloud + Worker)**
- [ ] **Phase 3: Fast English Translation Layer**
- [ ] **Phase 4: Slide Context & Course Glossary Integration**
- [ ] **Phase 5: PyQt6 Transparent Overlay UI**
- [ ] **Phase 6: Full 60-90 Minute Endurance Test**

---

## Setup & Configuration

1. **Install Dependencies**:
   ```powershell
   py -m pip install -r requirements.txt
   ```

2. **Configure API Key (Phase 2+)**:
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

---

## How to Test

### Phase 1: Microphone + Silero VAD Capture
```powershell
# List available input devices
py scripts\test_phase1.py --list-devices

# Test live audio capture and segmentation
py scripts\test_phase1.py
```
*Audio chunks are saved to `storage/audio_chunks/`.*

### Phase 2: Mandarin ASR Pipeline
```powershell
# Test ASR directly against a saved audio segment
py scripts\test_phase2.py --file storage\audio_chunks\segment_001_20261007_215400_1.6s.wav

# Run live Mic -> VAD -> ASR stream
py scripts\test_phase2.py
```
