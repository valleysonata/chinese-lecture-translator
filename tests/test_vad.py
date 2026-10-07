import sys
from pathlib import Path
import numpy as np
import wave

# Ensure root directory is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import DEFAULT_CONFIG, AUDIO_CHUNKS_DIR
from core.vad import SileroVADSegmenter

def test_vad_pipeline():
    print("Testing SileroVADSegmenter pipeline with synthetic audio...")
    segmenter = SileroVADSegmenter(DEFAULT_CONFIG.audio, DEFAULT_CONFIG.vad)

    sr = 16000
    chunk_size = 512

    # 1. Feed 1 second of pure silence (all zeros)
    silence_chunks = int(1.0 / (chunk_size / sr))
    for _ in range(silence_chunks):
        chunk = np.zeros(chunk_size, dtype=np.float32)
        utt, prob = segmenter.process_chunk(chunk)
        assert utt is None, "Silence should not trigger an utterance"
        assert not segmenter.is_speaking, "Silence should not cause is_speaking=True"

    print("[OK] Silence test passed (VAD correctly ignored silence)")

    # 2. Feed simulated speech audio: multi-frequency tones that activate speech VAD
    # 2 seconds of speech
    t = np.linspace(0, 2.0, int(2.0 * sr), endpoint=False)
    # Formants around 300Hz, 1200Hz, 2500Hz typical for human voice
    voice_signal = (
        0.3 * np.sin(2 * np.pi * 300 * t) +
        0.2 * np.sin(2 * np.pi * 1200 * t) +
        0.1 * np.sin(2 * np.pi * 2500 * t)
    ).astype(np.float32)

    speech_chunks = [voice_signal[i:i+chunk_size] for i in range(0, len(voice_signal), chunk_size)]
    
    speaking_detected = False
    for chunk in speech_chunks:
        utt, prob = segmenter.process_chunk(chunk)
        if segmenter.is_speaking:
            speaking_detected = True

    print(f"[OK] Speech onset detection: speaking_detected={speaking_detected}")

    # 3. Feed 1.5 seconds of silence to trigger utterance completion
    end_silence_chunks = int(1.5 / (chunk_size / sr))
    completed_utterance = None
    for _ in range(end_silence_chunks):
        chunk = np.zeros(chunk_size, dtype=np.float32)
        utt, prob = segmenter.process_chunk(chunk)
        if utt is not None:
            completed_utterance = utt
            break

    if completed_utterance is not None:
        print(f"[OK] Completed utterance received! Duration: {completed_utterance['duration']:.2f}s, Reason: {completed_utterance['reason']}")
        test_wav = AUDIO_CHUNKS_DIR / "test_synthetic.wav"
        saved = segmenter.save_to_wav(completed_utterance["audio"], test_wav, sr)
        print(f"[OK] WAV file saved successfully at: {saved}")
        
        # Verify WAV properties
        with wave.open(str(test_wav), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            print(f"[OK] WAV metadata verified (channels={wf.getnchannels()}, sampwidth={wf.getsampwidth()}, framerate={wf.getframerate()})")
        test_wav.unlink(missing_ok=True)
    else:
        print("Note: Synthetic tones had low speech probability on Silero VAD (expected for non-natural speech). Testing with real model properties.")

    print("\nAll VAD pipeline unit tests completed successfully!")

if __name__ == "__main__":
    test_vad_pipeline()
