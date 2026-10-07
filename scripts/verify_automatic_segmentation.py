import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_CONFIG
from core.vad import SileroVADSegmenter

def run_test():
    print("Testing SileroVADSegmenter automatic finalization with ZERO user input...")
    segmenter = SileroVADSegmenter(DEFAULT_CONFIG.audio, DEFAULT_CONFIG.vad)
    
    sr = 16000
    chunk_size = 512
    chunk_dur = chunk_size / sr # 0.032s

    # Generate speech-like signal (2 seconds)
    t = np.linspace(0, 2.0, int(2.0 * sr), endpoint=False)
    voice = (0.4 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 1200 * t)).astype(np.float32)
    # Silero requires realistic speech frequencies / patterns, or we can mock prob
    
    # Let's inspect Silero threshold behavior
    silence_chunk = np.zeros(chunk_size, dtype=np.float32)

    # Force speech state directly to test finalization
    segmenter.is_speaking = True
    segmenter.speech_start_time = time.time()
    segmenter.current_utterance = [silence_chunk] * 50 # 1.6s of audio
    segmenter.silence_chunks_count = 0

    print(f"Initial state: is_speaking={segmenter.is_speaking}, current_chunks={len(segmenter.current_utterance)}")
    print(f"Required silence chunks: {segmenter.required_silence_chunks} ({DEFAULT_CONFIG.vad.silence_duration_ms} ms)")

    finalized_utterance = None
    chunks_fed = 0
    start_sim = time.time()

    # Feed silence chunks until automatic finalization triggers
    while finalized_utterance is None and chunks_fed < 100:
        chunks_fed += 1
        utt, prob = segmenter.process_chunk(silence_chunk)
        if utt is not None:
            finalized_utterance = utt
            break

    elapsed_ms = chunks_fed * (chunk_dur * 1000)
    print(f"Chunks fed: {chunks_fed} (~{elapsed_ms:.0f} ms of simulated silence)")
    
    if finalized_utterance is not None:
        print("[SUCCESS] Utterance was automatically finalized by silence!")
        print(f"  Duration: {finalized_utterance['duration']:.2f}s")
        print(f"  Reason  : {finalized_utterance['reason']}")
        print(f"  Zero keyboard or Enter intervention required.")
    else:
        print("[FAILURE] Utterance was not finalized.")

if __name__ == "__main__":
    run_test()
