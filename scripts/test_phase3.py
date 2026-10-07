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
from core.live_translator import TranslationWorker
from providers.groq_asr import GroqASREngine
from providers.groq_translation import GroqTranslationEngine

def format_vu(level: float, length: int = 15) -> str:
    filled = int(min(1.0, level * 5) * length)
    return "#" * filled + "-" * (length - filled)

def test_text_mode(text: str):
    """Test translation layer directly on a text string."""
    print("=" * 60)
    print("  PHASE 3: Direct Text Translation Test")
    print("=" * 60)
    print(f" Input Text: {text}")
    print(" Submitting to Translation Engine...")

    try:
        engine = GroqTranslationEngine()
        res = engine.translate(text)
    except Exception as e:
        print(f"[ERROR]: {e}")
        return

    if res["success"]:
        print("\n" + "=" * 60)
        print(f" English Translation : {res['translation']}")
        print(f" Translation Latency: {res['latency']} s")
        print("=" * 60)
    else:
        print(f"\n[FAILED]: {res['error']}")

def test_file_mode(file_path: Path):
    """Test ASR + Translation on an audio file."""
    print("=" * 60)
    print(f"  PHASE 3: Audio File Pipeline Test: {file_path.name}")
    print("=" * 60)

    try:
        asr_engine = GroqASREngine()
        trans_engine = GroqTranslationEngine()
    except Exception as e:
        print(f"[ERROR]: {e}")
        return

    print("Transcribing with Groq Whisper...")
    asr_res = asr_engine.transcribe(file_path)
    if not asr_res["success"]:
        print(f"[ASR FAILED]: {asr_res['error']}")
        return

    transcript = asr_res["transcript"]
    print(f" >>> [ASR] ({asr_res['latency']}s lat) ZH: {transcript}")

    print("Translating to natural English...")
    trans_res = trans_engine.translate(transcript)
    if trans_res["success"]:
        print(f" >>> [LIVE EN] ({trans_res['latency']}s lat) EN: {trans_res['translation']}")
    else:
        print(f" >>> [TRANSLATION FAILED]: {trans_res['error']}")
    print("=" * 60)

def live_pipeline_mode(device_index: int | None = None, duration: int | None = None):
    """Continuous Live Pipeline: Mic -> VAD -> ASR -> Fast English Translation"""
    disable_quickedit()

    config = DEFAULT_CONFIG
    if device_index is not None:
        config.audio.device_index = device_index

    try:
        asr_engine = GroqASREngine()
        trans_engine = GroqTranslationEngine()
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        print("Please configure GROQ_API_KEY in .env before running Phase 3.")
        return

    print("=" * 60)
    print("  PHASE 3: Live End-to-End English Translation Pipeline")
    print("=" * 60)
    print(f" Audio Input : 16 kHz Mono | VAD: Silero (CPU)")
    print(f" ASR Model   : {config.asr.model} (Mandarin + English Code-switching)")
    print(f" Trans Model : {config.translation.model} (Natural CS English)")
    print(f" Silence Wait: {config.vad.silence_duration_ms} ms (automatic trigger)")
    print("=" * 60)
    print("Speak in Mandarin or mixed Chinese/English, then pause naturally (~0.8s).")
    print("Live English will follow immediately. Press Ctrl+C to stop.\n")

    stats = {
        "utt_count": 0,
        "asr_count": 0,
        "trans_count": 0,
        "total_asr_latency": 0.0,
        "total_trans_latency": 0.0,
        "total_delay": 0.0
    }

    # Asynchronous translation callback: prints English as soon as LLM responds
    def on_translation_done(trans_payload: dict):
        stats["trans_count"] += 1
        stats["total_trans_latency"] += trans_payload["translation_latency"]
        stats["total_delay"] += trans_payload["total_delay"]
        c = stats["trans_count"]

        if trans_payload["success"]:
            en = trans_payload["english_translation"]
            lat = trans_payload["translation_latency"]
            tot = trans_payload["total_delay"]
            print(f"\r" + " " * 80 + "\r", end="", flush=True)
            print(f"     >>> [LIVE EN #{c:02d}] ({lat}s lat | {tot}s total delay)")
            print(f"         EN: \"{en}\"\n")
        else:
            print(f"\r" + " " * 80 + "\r", end="", flush=True)
            print(f"     >>> [TRANSLATION ERROR #{c:02d}]: {trans_payload['error']}\n")

    # Translation worker running in background
    trans_worker = TranslationWorker(engine=trans_engine, on_result=on_translation_done)

    # Asynchronous ASR callback: prints transcript & forwards to translation worker
    def on_asr_done(asr_payload: dict):
        stats["asr_count"] += 1
        stats["total_asr_latency"] += asr_payload["asr_latency"]
        c = stats["asr_count"]

        if asr_payload["success"]:
            text = asr_payload["transcript"]
            lat = asr_payload["asr_latency"]
            dur = asr_payload["audio_duration"]
            print(f"\r" + " " * 80 + "\r", end="", flush=True)
            print(f" >>> [ASR #{c:02d}] ({lat}s lat | {dur:.1f}s speech)")
            print(f"     ZH: {text}")

            # Non-blocking dispatch to translation worker (independent pipeline)
            trans_worker.submit_transcript(asr_payload)
        else:
            print(f"\r" + " " * 80 + "\r", end="", flush=True)
            print(f" >>> [ASR ERROR #{c:02d}]: {asr_payload['error']}\n")

    asr_worker = ASRWorker(engine=asr_engine, on_result=on_asr_done)
    capture = AudioCapture(config.audio)
    segmenter = SileroVADSegmenter(config.audio, config.vad)

    try:
        capture.start()
        asr_worker.start()
        trans_worker.start()
    except Exception as e:
        print(f"[ERROR] Could not start pipeline: {e}")
        return

    start_time = time.time()

    try:
        while True:
            chunk = capture.get_chunk(timeout=0.05)
            if chunk is not None:
                rms = float(np.sqrt(np.mean(chunk**2))) if len(chunk) > 0 else 0.0
                utterance, prob = segmenter.process_chunk(chunk)

                status = "SPEAKING" if segmenter.is_speaking else "LISTENING"
                bar = format_vu(rms)
                prob_bar = format_vu(prob, length=10)
                asr_q = asr_worker.queue.qsize()
                trans_q = trans_worker.queue.qsize()

                print(
                    f"\r[{status:9s}] Vol: {bar} | VAD: {prob:0.2f} [{prob_bar}] | ASR Q:{asr_q} Trans Q:{trans_q} ",
                    end="",
                    flush=True
                )

                if utterance is not None:
                    stats["utt_count"] += 1
                    u_count = stats["utt_count"]
                    dur = utterance["duration"]
                    print(f"\r" + " " * 80 + "\r", end="", flush=True)
                    print(f"--- [VAD AUTO-FINALIZED #{u_count:02d}] Speech: {dur:.1f}s -> Submitting to ASR...")

                    # Save local copy for debugging
                    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path = AUDIO_CHUNKS_DIR / f"phase3_seg_{u_count:03d}_{ts_str}.wav"
                    segmenter.save_to_wav(utterance["audio"], out_path, config.audio.sample_rate)

                    # Non-blocking dispatch to ASR
                    asr_worker.submit_utterance(utterance)

            if duration and (time.time() - start_time) >= duration:
                print(f"\nAuto-stopping after {duration}s limit.")
                break

    except KeyboardInterrupt:
        print("\n\nStopping pipeline...")
    finally:
        capture.stop()
        asr_worker.stop()
        trans_worker.stop()
        elapsed = time.time() - start_time
        tc = stats["trans_count"]
        ac = stats["asr_count"]
        avg_asr = (stats["total_asr_latency"] / ac) if ac > 0 else 0.0
        avg_trans = (stats["total_trans_latency"] / tc) if tc > 0 else 0.0
        avg_delay = (stats["total_delay"] / tc) if tc > 0 else 0.0

        print("\n" + "=" * 60)
        print(" PHASE 3 SUMMARY")
        print("=" * 60)
        print(f" Total Runtime        : {elapsed:.1f} s")
        print(f" Utterances Detected  : {stats['utt_count']}")
        print(f" Transcripts Produced : {ac}")
        print(f" Translations Done    : {tc}")
        print(f" Avg ASR Latency      : {avg_asr:.2f} s")
        print(f" Avg Trans Latency    : {avg_trans:.2f} s")
        print(f" Avg Total Lag Behind : {avg_delay:.2f} s")
        print("=" * 60)

def test_benchmark_cases():
    """Run benchmark testing across Pure English, Pure Mandarin, and Mixed Code-Switching."""
    print("=" * 65)
    print("  PHASE 3.1: Benchmark Translation Quality & Conservatism")
    print("=" * 65)
    cases = [
        ("Pure English", "This node is unbalanced, so we perform a right rotation."),
        ("Pure Mandarin", "這個節點已經不平衡，所以我們需要做右旋。"),
        ("Mixed Code-Switching", "這個 node 已經不平衡，所以我們需要做 right rotation。"),
        ("Short Phrase (No outro)", "謝謝大家"),
        ("Awkward Spoken Grammar", "我們就需要出with rotation"),
        ("Incomplete Sentence", "如果這個 node 的 balance factor 大於 1"),
        ("Trail-off / Pause", "好，再看看")
    ]

    try:
        engine = GroqTranslationEngine()
    except Exception as e:
        print(f"[ERROR]: {e}")
        return

    for category, text in cases:
        res = engine.translate(text)
        print(f"[{category}]")
        print(f"  INPUT : {text}")
        if res["success"]:
            print(f"  OUTPUT: {res['translation']} ({res['latency']}s)")
        else:
            print(f"  ERROR : {res['error']}")
        print("-" * 65)

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Fast English Live Translation Pipeline")
    parser.add_argument("--text", type=str, default=None, help="Directly test translation on a text string")
    parser.add_argument("--file", type=str, default=None, help="Test ASR + translation on an audio file")
    parser.add_argument("--benchmark-cases", action="store_true", help="Run benchmark across Pure English, Mandarin, and Mixed cases")
    parser.add_argument("--device", type=int, default=None, help="Audio input device index")
    parser.add_argument("--duration", type=int, default=None, help="Auto stop after N seconds")
    args = parser.parse_args()

    if args.benchmark_cases:
        test_benchmark_cases()
    elif args.text:
        test_text_mode(args.text)
    elif args.file:
        test_file_mode(Path(args.file))
    else:
        live_pipeline_mode(device_index=args.device, duration=args.duration)

if __name__ == "__main__":
    main()
