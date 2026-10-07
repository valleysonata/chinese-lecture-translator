import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np

# Ensure root directory is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import DEFAULT_CONFIG, AUDIO_CHUNKS_DIR
from core.audio_capture import AudioCapture
from core.vad import SileroVADSegmenter

def format_vu(level: float, length: int = 15) -> str:
    """Creates a visual ASCII VU bar."""
    filled = int(min(1.0, level * 5) * length)
    return "#" * filled + "-" * (length - filled)

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Test Microphone + Silero VAD Capture")
    parser.add_argument("--device", type=int, default=None, help="Audio input device index")
    parser.add_argument("--list-devices", action="store_true", help="List available audio input devices and exit")
    parser.add_argument("--duration", type=int, default=None, help="Auto stop after N seconds (optional)")
    args = parser.parse_args()

    if args.list_devices:
        print("\n--- Available Audio Input Devices ---")
        devices = AudioCapture.list_input_devices()
        for idx, name, hostapi in devices:
            print(f" [{idx}] {name} (Host API: {hostapi})")
        print("-------------------------------------\n")
        return

    config = DEFAULT_CONFIG
    if args.device is not None:
        config.audio.device_index = args.device

    print("=" * 60)
    print("  PHASE 1: Microphone + Silero VAD Verification")
    print("=" * 60)
    print(f" Sample Rate : {config.audio.sample_rate} Hz (Mono)")
    print(f" Chunk Size  : {config.audio.chunk_size} samples (32 ms)")
    print(f" VAD Model   : Silero VAD (local PyTorch CPU)")
    print(f" Speech Thresh: {config.vad.threshold}")
    print(f" Silence Wait : {config.vad.silence_duration_ms} ms")
    print(f" Max Utterance: {config.vad.max_speech_duration_s} s")
    print(f" Output Dir  : {AUDIO_CHUNKS_DIR}")
    print("=" * 60)
    print("Starting audio capture... (Speak into your microphone, Ctrl+C to exit)\n")

    capture = AudioCapture(config.audio)
    segmenter = SileroVADSegmenter(config.audio, config.vad)

    try:
        capture.start()
    except Exception as e:
        print(f"\n[ERROR] Failed to start microphone: {e}")
        print("Tip: Run with --list-devices to view valid input device indices.")
        return

    start_time = time.time()
    segment_count = 0
    total_speech_time = 0.0

    try:
        while True:
            chunk = capture.get_chunk(timeout=0.1)
            if chunk is None:
                continue

            # Calculate RMS audio level
            rms = float(np.sqrt(np.mean(chunk**2))) if len(chunk) > 0 else 0.0
            
            utterance, prob = segmenter.process_chunk(chunk)

            # Terminal status line
            status = "SPEECH" if segmenter.is_speaking else "IDLE  "
            bar = format_vu(rms)
            prob_bar = format_vu(prob, length=10)
            
            print(
                f"\r[{status}] Level: {bar} | VAD: {prob:0.2f} [{prob_bar}]",
                end="",
                flush=True
            )

            # When speech finishes, save audio segment
            if utterance is not None:
                segment_count += 1
                dur = utterance["duration"]
                total_speech_time += dur
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"segment_{segment_count:03d}_{timestamp_str}_{dur:.1f}s.wav"
                out_path = AUDIO_CHUNKS_DIR / filename
                
                saved_path = segmenter.save_to_wav(utterance["audio"], out_path, config.audio.sample_rate)
                
                print(f"\n >>> [SAVED UTTERANCE #{segment_count:02d}] Duration: {dur:.2f}s | Saved: {saved_path}")

            if args.duration and (time.time() - start_time) >= args.duration:
                print(f"\nAuto-stopping after {args.duration}s limit.")
                break

    except KeyboardInterrupt:
        print("\n\nStopping audio capture...")
    finally:
        capture.stop()
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print(" PHASE 1 SUMMARY")
        print("=" * 60)
        print(f" Total Runtime      : {elapsed:.1f} s")
        print(f" Utterances Detected: {segment_count}")
        print(f" Total Speech Time  : {total_speech_time:.1f} s")
        print(f" Audio Chunks Saved : {AUDIO_CHUNKS_DIR}")
        print("=" * 60)

if __name__ == "__main__":
    main()
