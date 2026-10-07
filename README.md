# CS Lecture Interpreter for NYCU

A lightweight Windows desktop application designed to help an NYCU CS student follow Mandarin-taught lectures in natural English using course slides as context.

## Current Status: Phase 1 (Microphone + Local Silero VAD)

Phase 1 implements continuous 16 kHz mono microphone capture with local Silero VAD utterance segmentation.

### Features in Phase 1
- **16 kHz Mono Audio Capture**: Low-latency streaming via `sounddevice`.
- **Local Silero VAD**: Evaluates voice activity in 32ms frames (<1ms CPU overhead).
- **Utterance Segmentation**:
  - Pre-speech padding (retains consonant onsets without clipping).
  - Minimum speech duration check (rejects short clicks/pops).
  - Silence pause tolerance (800ms natural pause before finalizing an utterance).
  - Maximum chunk duration cap (12s limit to prevent accumulating latency).
- **Audio Verification Exporter**: Automatically saves finalized speech segments as 16-bit PCM WAV files in `storage/audio_chunks/`.

### Testing Phase 1

1. **List Audio Input Devices**:
   ```powershell
   py scripts\test_phase1.py --list-devices
   ```

2. **Run Live Microphone + VAD Test**:
   ```powershell
   py scripts\test_phase1.py
   ```
   *(Or specify a device index: `py scripts\test_phase1.py --device 1`)*

3. **Verify Audio Chunks**:
   Speak a phrase into your microphone. When you pause for ~0.8s, the completed utterance is displayed and saved to `storage/audio_chunks/`.
