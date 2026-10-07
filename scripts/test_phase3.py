import sys
import time
import argparse
from pathlib import Path

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

from config import DEFAULT_CONFIG
from core.glossary import (
    load_glossary,
    apply_glossary,
    describe as describe_glossary,
)
from core.pipeline import LivePipeline
from providers.groq_asr import GroqASREngine
from providers.groq_translation import GroqTranslationEngine, SYSTEM_PROMPT

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

    try:
        pipeline = LivePipeline(
            system_prompt=system_prompt,
            glossary_terms=glossary_terms,
            device_index=device_index,
            wav_prefix="phase3_seg",
            log_prefix="phase3",
        )
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        print("Please configure GROQ_API_KEY in .env before running Phase 3.")
        return

    def on_event(evt: dict):
        etype = evt["type"]
        with pipeline.print_lock:
            if etype == "utterance":
                print("\r" + " " * 80 + "\r", end="", flush=True)
                print(f"--- [VAD AUTO-FINALIZED #{evt['utt_index']:02d}] "
                      f"Speech: {evt['duration']:.1f}s -> Submitting to ASR...")

            elif etype == "asr":
                c = evt["asr_index"]
                print("\r" + " " * 80 + "\r", end="", flush=True)
                if evt["success"]:
                    print(f" >>> [ASR #{c:02d}] ({evt['asr_latency']}s lat | "
                          f"{evt['audio_duration']:.1f}s speech)")
                    print(f"     ZH: {evt['transcript']}")
                else:
                    et = evt.get("error_type") or "error"
                    print(f" >>> [ASR {et.upper()} #{c:02d}]: {evt['error']}\n")

            elif etype == "drop":
                if evt["reason"] == "empty_transcript":
                    pass  # already counted + logged; nothing to show
                elif evt["stage"] == "asr":
                    print(f"[WARN] ASR queue full - dropped utterance "
                          f"#{evt.get('utt_index', 0):02d}")
                else:
                    print(f"     [WARN] Translation queue full - dropped ASR "
                          f"#{evt.get('asr_index', 0):02d}\n")

            elif etype == "translation":
                c = evt["trans_index"]
                print("\r" + " " * 80 + "\r", end="", flush=True)
                if evt["success"]:
                    trunc = " [TRUNCATED]" if evt.get("finish_reason") == "length" else ""
                    print(f"     >>> [LIVE EN #{c:02d}] ({evt['translation_latency']}s lat | "
                          f"{evt['total_delay']}s total delay){trunc}")
                    print(f"         EN: \"{evt['english_translation']}\"\n")
                else:
                    et = evt.get("error_type") or "error"
                    print(f"     >>> [TRANSLATION {et.upper()} #{c:02d}]: {evt['error']}\n")

    pipeline.on_event = on_event

    print("=" * 60)
    print("  PHASE 3: Live End-to-End English Translation Pipeline")
    print("=" * 60)
    print(" Audio Input : 16 kHz Mono | VAD: Silero (CPU)")
    print(f" ASR Model   : {config.asr.model} (Mandarin + English Code-switching)")
    print(f" Trans Model : {config.translation.model} (Natural CS English)")
    print(f" Silence Wait: {config.vad.silence_duration_ms} ms (automatic trigger)")
    print("=" * 60)
    print("Speak in Mandarin or mixed Chinese/English, then pause naturally (~0.8s).")
    print("Live English will follow immediately. Press Ctrl+C to stop.\n")

    try:
        pipeline.start()
    except Exception as e:
        print(f"[ERROR] Could not start pipeline: {e}")
        pipeline.stop()
        return

    start_time = time.time()
    try:
        while True:
            status = pipeline.process_once(timeout=0.05)
            if status is not None:
                vu = format_vu(status["rms"])
                prob_bar = format_vu(status["prob"], length=10)
                with pipeline.print_lock:
                    print(
                        f"\r[{status['status']:9s}] Vol: {vu} | VAD: {status['prob']:.2f} [{prob_bar}] | "
                        f"ASR Q:{status['asr_q']} Trans Q:{status['trans_q']} ",
                        end="", flush=True,
                    )
            if duration and (time.time() - start_time) >= duration:
                print(f"\nAuto-stopping after {duration}s limit.")
                break
    except KeyboardInterrupt:
        print("\n\nStopping pipeline...")
    finally:
        summary = pipeline.stop()
        _print_summary(summary)


def _print_summary(s: dict, title: str = "PHASE 3 SUMMARY"):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)
    print(f" Total Runtime        : {s['runtime_s']} s")
    print(f" Utterances Detected  : {s['utterances']}")
    print(f" Transcripts Produced : {s['transcripts']}")
    print(f" Translations Done    : {s['translations']}")
    print(f" Avg ASR Latency      : {s['avg_asr_latency']} s")
    print(f" Avg Trans Latency    : {s['avg_translation_latency']} s")
    print(f" Avg Total Lag Behind : {s['avg_total_delay']} s")
    print(f" Dropped (ASR Q full) : {s['dropped_asr']}")
    print(f" Dropped (Trans Q full): {s['dropped_translation']}")
    print(f" Empty Transcripts    : {s['empty_transcripts']}")
    print(f" Truncated Translates : {s['truncated_translations']}")
    print(f" Rate-Limit Errors    : {s['rate_limit_errors']}")
    print(f" Dropped (Audio Q full): {s['dropped_audio_frames']}")
    print(f" Worker Q Drops (A/T) : {s['asr_worker_drops']} / {s['translation_worker_drops']}")
    print(f" Session Log          : {s['session_log']}")
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
    system_prompt = apply_glossary(DEFAULT_CONFIG.asr, terms, SYSTEM_PROMPT)

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
