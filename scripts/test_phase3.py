import os
import sys
import time
import argparse
import threading
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

from config import DEFAULT_CONFIG, AUDIO_CHUNKS_DIR, SESSION_LOGS_DIR
from core.audio_capture import AudioCapture
from core.vad import SileroVADSegmenter
from core.asr import ASRWorker
from core.live_translator import TranslationWorker
from core.session_log import SessionLogger
from core.glossary import (
    load_glossary,
    build_asr_prompt,
    build_translation_prompt,
    describe as describe_glossary,
)
from providers.groq_asr import GroqASREngine
from providers.groq_translation import GroqTranslationEngine, SYSTEM_PROMPT


def apply_glossary(terms: list) -> str | None:
    """
    Phase 4: inject glossary terms into both prompts.
    Mutates the shared ASR config prompt (once per process) and returns the
    translation system prompt override (None when no glossary is loaded).
    """
    if not terms:
        return None
    DEFAULT_CONFIG.asr.initial_prompt = build_asr_prompt(
        DEFAULT_CONFIG.asr.initial_prompt, terms
    )
    return build_translation_prompt(SYSTEM_PROMPT, terms)

def format_vu(level: float, length: int = 15) -> str:
    filled = int(min(1.0, level * 5) * length)
    return "#" * filled + "-" * (length - filled)

def test_text_mode(text: str, system_prompt: str | None = None):
    """Test translation layer directly on a text string."""
    print("=" * 60)
    print("  PHASE 3: Direct Text Translation Test")
    print("=" * 60)
    print(f" Input Text: {text}")
    print(" Submitting to Translation Engine...")

    try:
        engine = GroqTranslationEngine(system_prompt=system_prompt)
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

def test_file_mode(file_path: Path, system_prompt: str | None = None):
    """Test ASR + Translation on an audio file."""
    print("=" * 60)
    print(f"  PHASE 3: Audio File Pipeline Test: {file_path.name}")
    print("=" * 60)

    try:
        asr_engine = GroqASREngine()
        trans_engine = GroqTranslationEngine(system_prompt=system_prompt)
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

def live_pipeline_mode(
    device_index: int | None = None,
    duration: int | None = None,
    system_prompt: str | None = None,
    glossary_terms: list | None = None
):
    """Continuous Live Pipeline: Mic -> VAD -> ASR -> Fast English Translation"""
    disable_quickedit()

    config = DEFAULT_CONFIG
    if device_index is not None:
        config.audio.device_index = device_index

    try:
        asr_engine = GroqASREngine()
        trans_engine = GroqTranslationEngine(system_prompt=system_prompt)
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
    print(describe_glossary(glossary_terms or []))
    print("=" * 60)
    print("Speak in Mandarin or mixed Chinese/English, then pause naturally (~0.8s).")
    print("Live English will follow immediately. Press Ctrl+C to stop.\n")

    stats = {
        "utt_count": 0,
        "asr_count": 0,
        "trans_count": 0,
        "dropped_asr": 0,
        "dropped_trans": 0,
        "empty_transcript": 0,
        "truncated_trans": 0,
        "rate_limited": 0,
        "total_asr_latency": 0.0,
        "total_trans_latency": 0.0,
        "total_delay": 0.0
    }

    # One lock for every console write: the main VU line and the worker callbacks
    # all emit multi-line/overwrite sequences, so writes must stay atomic.
    print_lock = threading.Lock()

    session_path = SESSION_LOGS_DIR / f"phase3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    session = SessionLogger(session_path)
    session.log("session_start", {
        "asr_model": config.asr.model,
        "trans_model": config.translation.model,
        "silence_ms": config.vad.silence_duration_ms,
        "glossary_terms": len(glossary_terms or [])
    })

    # Asynchronous translation callback: prints English as soon as LLM responds
    def on_translation_done(trans_payload: dict):
        stats["trans_count"] += 1
        stats["total_trans_latency"] += trans_payload["translation_latency"]
        stats["total_delay"] += trans_payload["total_delay"]
        c = stats["trans_count"]

        # Track truncation (finish_reason="length" means max_tokens cut the output)
        if trans_payload["success"] and trans_payload.get("finish_reason") == "length":
            stats["truncated_trans"] += 1
        if not trans_payload["success"] and trans_payload.get("error_type") == "rate_limit":
            stats["rate_limited"] += 1

        session.log("translation", {
            "success": trans_payload["success"],
            "mandarin_transcript": trans_payload["mandarin_transcript"],
            "english_translation": trans_payload["english_translation"],
            "translation_latency": trans_payload["translation_latency"],
            "asr_latency": trans_payload["asr_latency"],
            "total_delay": trans_payload["total_delay"],
            "queue_backlog": trans_payload["queue_backlog"],
            "finish_reason": trans_payload.get("finish_reason"),
            "error": trans_payload["error"],
            "error_type": trans_payload.get("error_type")
        })

        with print_lock:
            if trans_payload["success"]:
                en = trans_payload["english_translation"]
                lat = trans_payload["translation_latency"]
                tot = trans_payload["total_delay"]
                trunc = " [TRUNCATED]" if trans_payload.get("finish_reason") == "length" else ""
                print("\r" + " " * 80 + "\r", end="", flush=True)
                print(f"     >>> [LIVE EN #{c:02d}] ({lat}s lat | {tot}s total delay){trunc}")
                print(f"         EN: \"{en}\"\n")
            else:
                etype = trans_payload.get("error_type") or "error"
                print("\r" + " " * 80 + "\r", end="", flush=True)
                print(f"     >>> [TRANSLATION {etype.upper()} #{c:02d}]: {trans_payload['error']}\n")

    # Translation worker running in background
    trans_worker = TranslationWorker(engine=trans_engine, on_result=on_translation_done)

    # Asynchronous ASR callback: prints transcript & forwards to translation worker
    def on_asr_done(asr_payload: dict):
        stats["asr_count"] += 1
        stats["total_asr_latency"] += asr_payload["asr_latency"]
        c = stats["asr_count"]

        session.log("asr", {
            "success": asr_payload["success"],
            "transcript": asr_payload["transcript"],
            "asr_latency": asr_payload["asr_latency"],
            "audio_duration": asr_payload["audio_duration"],
            "total_delay": asr_payload["total_delay"],
            "queue_backlog": asr_payload["queue_backlog"],
            "error": asr_payload["error"],
            "error_type": asr_payload.get("error_type")
        })

        if not asr_payload["success"] and asr_payload.get("error_type") == "rate_limit":
            stats["rate_limited"] += 1

        if asr_payload["success"]:
            text = asr_payload["transcript"]
            lat = asr_payload["asr_latency"]
            dur = asr_payload["audio_duration"]
            with print_lock:
                print("\r" + " " * 80 + "\r", end="", flush=True)
                print(f" >>> [ASR #{c:02d}] ({lat}s lat | {dur:.1f}s speech)")
                print(f"     ZH: {text}")

            if not text.strip():
                # Hallucination filter emptied the transcript: skip translation, but record it.
                stats["empty_transcript"] += 1
                session.log("drop", {"stage": "translation", "reason": "empty_transcript"})
                return

            # Non-blocking dispatch to translation worker (independent pipeline)
            if not trans_worker.submit_transcript(asr_payload):
                stats["dropped_trans"] += 1
                session.log("drop", {
                    "stage": "translation",
                    "reason": "queue_full",
                    "transcript": text
                })
                with print_lock:
                    print(f"     [WARN] Translation queue full - dropped ASR #{c:02d}\n")
        else:
            with print_lock:
                print("\r" + " " * 80 + "\r", end="", flush=True)
                etype = asr_payload.get("error_type") or "error"
                print(f" >>> [ASR {etype.upper()} #{c:02d}]: {asr_payload['error']}\n")

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

                with print_lock:
                    print(
                        f"\r[{status:9s}] Vol: {bar} | VAD: {prob:0.2f} [{prob_bar}] | ASR Q:{asr_q} Trans Q:{trans_q} ",
                        end="",
                        flush=True
                    )

                if utterance is not None:
                    stats["utt_count"] += 1
                    u_count = stats["utt_count"]
                    dur = utterance["duration"]
                    with print_lock:
                        print("\r" + " " * 80 + "\r", end="", flush=True)
                        print(f"--- [VAD AUTO-FINALIZED #{u_count:02d}] Speech: {dur:.1f}s -> Submitting to ASR...")

                    # Save local copy for debugging
                    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path = AUDIO_CHUNKS_DIR / f"phase3_seg_{u_count:03d}_{ts_str}.wav"
                    segmenter.save_to_wav(utterance["audio"], out_path, config.audio.sample_rate)

                    # Non-blocking dispatch to ASR (drops are counted, never silent)
                    if not asr_worker.submit_utterance(utterance):
                        stats["dropped_asr"] += 1
                        session.log("drop", {
                            "stage": "asr",
                            "reason": "queue_full",
                            "audio_duration": dur
                        })
                        with print_lock:
                            print(f"[WARN] ASR queue full - dropped utterance #{u_count:02d}")

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

        summary = {
            "runtime_s": round(elapsed, 1),
            "utterances": stats["utt_count"],
            "transcripts": ac,
            "translations": tc,
            "dropped_asr": stats["dropped_asr"],
            "dropped_translation": stats["dropped_trans"],
            "empty_transcripts": stats["empty_transcript"],
            "truncated_translations": stats["truncated_trans"],
            "rate_limit_errors": stats["rate_limited"],
            "dropped_audio_frames": capture.dropped_frames,
            "asr_worker_drops": asr_worker.dropped_count,
            "translation_worker_drops": trans_worker.dropped_count,
            "avg_asr_latency": round(avg_asr, 3),
            "avg_translation_latency": round(avg_trans, 3),
            "avg_total_delay": round(avg_delay, 3)
        }
        session.log("session_end", summary)
        session.close()

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
        print(f" Dropped (ASR Q full) : {stats['dropped_asr']}")
        print(f" Dropped (Trans Q full): {stats['dropped_trans']}")
        print(f" Empty Transcripts    : {stats['empty_transcript']}")
        print(f" Truncated Translates : {stats['truncated_trans']}")
        print(f" Rate-Limit Errors    : {stats['rate_limited']}")
        print(f" Dropped (Audio Q full): {capture.dropped_frames}")
        print(f" Session Log          : {session_path}")
        print("=" * 60)

def test_benchmark_cases(system_prompt: str | None = None):
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
        engine = GroqTranslationEngine(system_prompt=system_prompt)
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
    parser.add_argument("--glossary", type=str, default=None, help="Path to glossary .txt (one term per line) injected into ASR + translation prompts")
    args = parser.parse_args()

    # Phase 4: load glossary and inject into prompts before engines are built
    terms: list = []
    if args.glossary:
        try:
            terms = load_glossary(args.glossary)
        except (FileNotFoundError, UnicodeDecodeError) as e:
            print(f"[ERROR] {e}")
            return
        print(describe_glossary(terms))
    system_prompt = apply_glossary(terms)

    if args.benchmark_cases:
        test_benchmark_cases(system_prompt)
    elif args.text:
        test_text_mode(args.text, system_prompt)
    elif args.file:
        test_file_mode(Path(args.file), system_prompt)
    else:
        live_pipeline_mode(
            device_index=args.device,
            duration=args.duration,
            system_prompt=system_prompt,
            glossary_terms=terms
        )

if __name__ == "__main__":
    main()
