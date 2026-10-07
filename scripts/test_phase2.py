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

def disable_quickedit():
    """Disables Windows console QuickEdit mode so mouse clicks never pause terminal execution."""
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            h_stdin = kernel32.GetStdHandle(-10) # STD_INPUT_HANDLE
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(h_stdin, ctypes.byref(mode)):
                # ENABLE_QUICK_EDIT_MODE = 0x0040, ENABLE_EXTENDED_FLAGS = 0x0080
                new_mode = (mode.value & ~0x0040) | 0x0080
                kernel32.SetConsoleMode(h_stdin, new_mode)
        except Exception:
            pass

from config import DEFAULT_CONFIG, AUDIO_CHUNKS_DIR
from core.audio_capture import AudioCapture
from core.vad import SileroVADSegmenter
from core.asr import ASRWorker
from providers.groq_asr import GroqASREngine

def format_vu(level: float, length: int = 15) -> str:
    filled = int(min(1.0, level * 5) * length)
    return "#" * filled + "-" * (length - filled)

def test_file_mode(file_path: Path):
    """Test ASR directly against an audio file."""
    print("=" * 60)
    print(f"  PHASE 2: Testing ASR with File: {file_path.name}")
    print("=" * 60)
    
    try:
        engine = GroqASREngine()
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        print("Please configure GROQ_API_KEY in .env before running Phase 2 test.")
        return

    print("Submitting to Groq Whisper...")
    res = engine.transcribe(file_path)
    
    if res["success"]:
        print("\n" + "=" * 60)
        print(" [ASR RESULT]")
        print("=" * 60)
        print(f" Mandarin Transcript: {res['transcript']}")
        print(f" Audio Duration     : {res['duration']} s")
        print(f" API Latency        : {res['latency']} s")
        print("=" * 60)
    else:
        print(f"\n[ASR FAILED]: {res['error']}")

def live_mic_mode(device_index: int | None = None, duration: int | None = None):
    """Continuous automatic pipeline: Mic -> VAD -> Automatic Finalization -> ASR -> Terminal"""
    disable_quickedit()

    config = DEFAULT_CONFIG
    if device_index is not None:
        config.audio.device_index = device_index

    try:
        engine = GroqASREngine()
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        print("Please configure GROQ_API_KEY in .env before running Phase 2 live test.")
        return

    print("=" * 60)
    print("  PHASE 2: Automatic Live Mandarin ASR Pipeline")
    print("=" * 60)
    print(f" Sample Rate : {config.audio.sample_rate} Hz (Mono)")
    print(f" VAD Model   : Silero VAD (CPU)")
    print(f" Silence Wait: {config.vad.silence_duration_ms} ms (automatic trigger)")
    print(f" ASR Engine  : Groq Cloud ({config.asr.model})")
    print(f" Language    : Mandarin ({config.asr.language})")
    print("=" * 60)
    print("Starting pipeline... Speak, pause naturally (~0.8s), and watch output.")
    print("NO ENTER KEY NEEDED. Ctrl+C to exit.\n")

    stats = {
        "asr_count": 0,
        "total_asr_latency": 0.0
    }

    def on_asr_result(res: dict):
        """Asynchronous callback fired immediately when Groq responds."""
        stats["asr_count"] += 1
        stats["total_asr_latency"] += res["asr_latency"]
        count = stats["asr_count"]

        if res["success"]:
            text = res["transcript"]
            lat = res["asr_latency"]
            delay = res["total_delay"]
            dur = res["audio_duration"]
            # Clear line and print result
            print(f"\r" + " " * 75 + "\r", end="", flush=True)
            print(f">>> [ASR #{count}] Latency: {lat}s | Delay: {delay}s | Dur: {dur:.1f}s")
            print(f"    TRANSCRIPT: {text}\n")
        else:
            print(f"\r" + " " * 75 + "\r", end="", flush=True)
            print(f">>> [ASR ERROR #{count}]: {res['error']}\n")

    capture = AudioCapture(config.audio)
    segmenter = SileroVADSegmenter(config.audio, config.vad)
    worker = ASRWorker(engine=engine, on_result=on_asr_result)

    try:
        capture.start()
        worker.start()
    except Exception as e:
        print(f"[ERROR] Could not start audio stream: {e}")
        return

    start_time = time.time()
    segment_count = 0

    try:
        while True:
            # 1. Non-blocking retrieval of raw 32ms microphone audio frame
            chunk = capture.get_chunk(timeout=0.05)
            if chunk is not None:
                rms = float(np.sqrt(np.mean(chunk**2))) if len(chunk) > 0 else 0.0
                utterance, prob = segmenter.process_chunk(chunk)

                status = "SPEAKING" if segmenter.is_speaking else "LISTENING"
                bar = format_vu(rms)
                prob_bar = format_vu(prob, length=10)
                q_depth = worker.queue.qsize()

                print(
                    f"\r[{status:9s}] Vol: {bar} | VAD: {prob:0.2f} [{prob_bar}] | Backlog: {q_depth} ",
                    end="",
                    flush=True
                )

                # Automatic utterance trigger: VAD detected ~800ms silence or max duration
                if utterance is not None:
                    segment_count += 1
                    dur = utterance["duration"]
                    print(f"\r" + " " * 75 + "\r", end="", flush=True)
                    print(f"--- [VAD AUTO-FINALIZED #{segment_count}] Speech: {dur:.1f}s -> Submitting to Groq...")

                    # Save local copy for verification
                    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path = AUDIO_CHUNKS_DIR / f"phase2_seg_{segment_count:03d}_{ts_str}.wav"
                    segmenter.save_to_wav(utterance["audio"], out_path, config.audio.sample_rate)

                    # Non-blocking dispatch to ASR worker
                    submitted = worker.submit_utterance(utterance)
                    if not submitted:
                        print(f"[WARNING] Dropped segment #{segment_count} due to queue backlog!")

            if duration and (time.time() - start_time) >= duration:
                print(f"\nAuto-stopping after {duration}s limit.")
                break

    except KeyboardInterrupt:
        print("\n\nStopping pipeline...")
    finally:
        capture.stop()
        worker.stop()
        elapsed = time.time() - start_time
        count = stats["asr_count"]
        avg_lat = (stats["total_asr_latency"] / count) if count > 0 else 0.0
        print("\n" + "=" * 60)
        print(" PHASE 2 SUMMARY")
        print("=" * 60)
        print(f" Runtime          : {elapsed:.1f} s")
        print(f" Utterances Sent  : {segment_count}")
        print(f" Transcripts Done : {count}")
        print(f" Avg ASR Latency  : {avg_lat:.2f} s")
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Mandarin ASR Pipeline")
    parser.add_argument("--file", type=str, default=None, help="Path to an audio file to test ASR directly")
    parser.add_argument("--device", type=int, default=None, help="Audio input device index")
    parser.add_argument("--duration", type=int, default=None, help="Auto stop after N seconds")
    args = parser.parse_args()

    if args.file:
        test_file_mode(Path(args.file))
    else:
        live_mic_mode(device_index=args.device, duration=args.duration)

if __name__ == "__main__":
    main()
