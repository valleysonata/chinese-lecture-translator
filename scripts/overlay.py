"""
Phase 5: Transparent subtitle overlay.

Frameless, always-on-top, translucent bar pinned to the bottom of the screen
that renders live English subtitles over slides/IDE windows. Click-through is
enabled by default so the bar never blocks clicks (use --clickable to disable).

Modes:
  (default)     live pipeline: mic -> VAD -> ASR -> translation (needs GROQ_API_KEY)
  --demo        synthetic subtitle stream; no mic, no API (UI smoke test)
  --text STR    translate one string and show it; no mic
  --list-devices  enumerate microphones and exit

Examples:
  py scripts\\overlay.py --glossary glossary.txt
  py scripts\\overlay.py --demo --duration 15
  py scripts\\overlay.py --text "這個節點已經不平衡" --duration 10
"""
import sys
import time
import queue
import signal
import argparse
from pathlib import Path

# Ensure root directory is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QPainter, QColor
except ImportError:
    print("[ERROR] PyQt6 is required for the overlay UI.")
    print("        Install it with: py -m pip install PyQt6")
    print("        Console fallback: py scripts\\test_phase3.py --glossary glossary.txt")
    sys.exit(1)

from config import DEFAULT_CONFIG
from core.audio_capture import AudioCapture
from core.glossary import (
    load_glossary,
    apply_glossary,
    describe as describe_glossary,
)
from core.pipeline import LivePipeline
from providers.groq_translation import GroqTranslationEngine, SYSTEM_PROMPT
# Shared console helpers (importing these has no side effects beyond stdout setup)
from scripts.test_phase3 import disable_quickedit, _print_summary


DEMO_LINES = [
    ("我們先看這個 binary tree 的結構",
     "Let's start with the structure of this binary tree."),
    ("左子樹的高度比右子樹高兩層",
     "The left subtree is two levels taller than the right subtree."),
    ("所以這個節點是不平衡的",
     "So this node is unbalanced."),
    ("我們需要對它做一次右旋",
     "We need to perform a right rotation on it."),
    ("旋轉之後 balance factor 就回到一了",
     "After the rotation, the balance factor is back to one."),
    ("接下來我們看時間複雜度",
     "Next, let's look at the time complexity."),
]

BAR_HEIGHT = 176


class SubtitleOverlay(QWidget):
    """Bottom subtitle bar: latest line large, previous lines faded above."""

    def __init__(self, screen_index: int = 0, click_through: bool = True):
        super().__init__()
        self.max_prev = 3
        self.prev_lines: list[str] = []
        self._has_content = False
        self._click_through = False

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        if click_through:
            try:
                self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, True)
                self._click_through = True
            except AttributeError:
                pass  # flag unavailable: bar stays clickable but still visible

        screens = QApplication.screens()
        screen = screens[min(screen_index, len(screens) - 1)]
        geo = screen.availableGeometry()
        self.setGeometry(
            geo.x() + 8,
            geo.y() + geo.height() - BAR_HEIGHT - 8,
            geo.width() - 16,
            BAR_HEIGHT,
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 6, 16, 8)
        root.setSpacing(1)

        self.status_label = QLabel("", self)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.status_label.setStyleSheet("color: rgba(170, 210, 255, 150); font-size: 10px;")
        root.addWidget(self.status_label)

        self.prev_label = QLabel("", self)
        self.prev_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prev_label.setStyleSheet("color: rgba(255, 255, 255, 100); font-size: 12px;")
        self.prev_label.setWordWrap(True)
        root.addWidget(self.prev_label)

        self.zh_label = QLabel("", self)
        self.zh_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zh_label.setStyleSheet("color: rgba(210, 210, 210, 150); font-size: 13px;")
        self.zh_label.setWordWrap(True)
        root.addWidget(self.zh_label)

        self.en_label = QLabel("", self)
        self.en_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.en_label.setStyleSheet("color: rgba(255, 255, 255, 255); font-size: 24px; font-weight: 600;")
        self.en_label.setWordWrap(True)
        root.addWidget(self.en_label)

    # ------------------------------------------------------------------ painting

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 155))
        painter.drawRoundedRect(self.rect().adjusted(4, 4, -4, -4), 14, 14)
        painter.end()

    # ------------------------------------------------------------------ content

    @property
    def click_through(self) -> bool:
        return self._click_through

    def set_hint(self, text: str) -> None:
        if not self._has_content:
            self.en_label.setText(text)

    def add_translation(self, zh: str, en: str) -> None:
        if self._has_content and self.en_label.text():
            self.prev_lines.insert(0, self.en_label.text())
            del self.prev_lines[self.max_prev:]
        self._has_content = True
        self.en_label.setText(en)
        self.zh_label.setText(zh or "")
        self.prev_label.setText("\n".join(self.prev_lines))

    def set_status(self, text: str, alert: bool = False) -> None:
        self.status_label.setText(text)
        if alert:
            self.status_label.setStyleSheet("color: rgba(255, 130, 130, 235); font-size: 10px; font-weight: 600;")
        else:
            self.status_label.setStyleSheet("color: rgba(170, 210, 255, 150); font-size: 10px;")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5: Transparent subtitle overlay")
    parser.add_argument("--text", type=str, default=None,
                        help="Translate one string into the overlay (no mic)")
    parser.add_argument("--demo", action="store_true",
                        help="Synthetic subtitle stream for UI testing (no mic, no API)")
    parser.add_argument("--device", type=int, default=None, help="Audio input device index")
    parser.add_argument("--duration", type=int, default=None,
                        help="Auto stop after N seconds (defaults: demo=20, text=15, live=forever)")
    parser.add_argument("--glossary", type=str, default=None,
                        help="Path to glossary .txt (one term per line) injected into ASR + translation prompts")
    parser.add_argument("--screen", type=int, default=0,
                        help="Screen index for the overlay (default 0 = primary)")
    parser.add_argument("--clickable", action="store_true",
                        help="Disable click-through: let the overlay accept mouse input")
    parser.add_argument("--list-devices", action="store_true", help="List audio input devices and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    disable_quickedit()

    if args.list_devices:
        for idx, name, hostapi in AudioCapture.list_input_devices():
            print(f"[{idx}] {name} (hostapi {hostapi})")
        return 0

    # Phase 4: glossary injection before any engine is constructed
    terms: list = []
    if args.glossary:
        try:
            terms = load_glossary(args.glossary)
        except (FileNotFoundError, UnicodeDecodeError) as e:
            print(f"[ERROR] {e}")
            return 1
    system_prompt = apply_glossary(DEFAULT_CONFIG.asr, terms, SYSTEM_PROMPT)

    mode = "demo" if args.demo else ("text" if args.text else "live")
    duration = args.duration
    if duration is None:
        duration = {"demo": 20, "text": 15}.get(mode)  # live runs until Ctrl+C

    app = QApplication(sys.argv)
    overlay = SubtitleOverlay(screen_index=args.screen, click_through=not args.clickable)
    overlay.show()
    overlay.raise_()

    # Graceful Ctrl+C: ask Qt to leave its event loop; cleanup happens after exec_()
    signal.signal(signal.SIGINT, lambda *_: app.quit())

    start_time = time.time()
    ui_state = {"main": mode.upper()}
    alert = {"msg": "", "until": 0.0}
    event_q: queue.Queue = queue.Queue()
    pipeline: LivePipeline | None = None
    demo_state = {"i": 0}

    # ------------------------------------------------------------------ mode setup

    if args.demo:
        overlay.set_hint("Demo mode — synthetic subtitles")
        ui_state["main"] = "DEMO"

        def show_next_demo():
            zh, en = DEMO_LINES[demo_state["i"] % len(DEMO_LINES)]
            demo_state["i"] += 1
            overlay.add_translation(zh, en)

        show_next_demo()
        demo_timer = QTimer(app)
        demo_timer.timeout.connect(show_next_demo)
        demo_timer.start(1600)

    elif args.text:
        overlay.set_hint("Translating…")
        ui_state["main"] = "TEXT"
        try:
            engine = GroqTranslationEngine(system_prompt=system_prompt)
            res = engine.translate(args.text)
        except Exception as e:
            print(f"[ERROR]: {e}")
            overlay.close()
            return 1
        if res["success"]:
            overlay.add_translation(args.text, res["translation"])
            print(f" >>> ({res['latency']}s lat) EN: {res['translation']}")
        else:
            print(f"[FAILED]: {res['error']}")
            overlay.close()
            return 1

    else:
        overlay.set_hint("Listening…")
        try:
            pipeline = LivePipeline(
                system_prompt=system_prompt,
                glossary_terms=terms,
                device_index=args.device,
                wav_prefix="overlay_seg",
                log_prefix="overlay",
                on_event=event_q.put,
            )
        except ValueError as e:
            print(f"[ERROR] {e}")
            print("Please configure GROQ_API_KEY in .env before running the overlay.")
            overlay.close()
            return 1

        print("=" * 60)
        print("  PHASE 5: Live Subtitle Overlay")
        print("=" * 60)
        print(" Audio Input : 16 kHz Mono | VAD: Silero (CPU)")
        print(f" ASR Model   : {DEFAULT_CONFIG.asr.model} (Mandarin + English Code-switching)")
        print(f" Trans Model : {DEFAULT_CONFIG.translation.model} (Natural CS English)")
        print(f" Silence Wait: {DEFAULT_CONFIG.vad.silence_duration_ms} ms (automatic trigger)")
        print(describe_glossary(terms))
        print(f" Overlay     : bottom bar | click-through: {overlay.click_through}")
        print(" Stop        : Ctrl+C in this console" + (f" (auto-stop in {duration}s)" if duration else ""))
        print("=" * 60 + "\n")

        try:
            pipeline.start()
        except Exception as e:
            print(f"[ERROR] Could not start pipeline: {e}")
            pipeline.stop()
            overlay.close()
            return 1

        def live_tick():
            status = None
            for _ in range(4):  # catch up to real time if the UI was busy
                st = pipeline.process_once(timeout=0.0)
                if st is not None:
                    status = st

            while True:
                try:
                    evt = event_q.get_nowait()
                except queue.Empty:
                    break
                etype = evt.get("type")
                if etype == "translation":
                    if evt["success"]:
                        overlay.add_translation(evt["mandarin_transcript"], evt["english_translation"])
                    else:
                        alert["msg"] = (evt.get("error_type") or "error").upper().replace("_", " ")
                        alert["until"] = time.time() + 5
                elif etype == "asr" and not evt["success"]:
                    alert["msg"] = (evt.get("error_type") or "error").upper().replace("_", " ")
                    alert["until"] = time.time() + 5

            drops = (
                pipeline.stats["dropped_asr"]
                + pipeline.stats["dropped_trans"]
                + pipeline.capture.dropped_frames
            )
            parts = []
            if status is not None:
                parts.append(status["status"])
                parts.append(f"ASR Q:{status['asr_q']} Trans Q:{status['trans_q']}")
            parts.append(f"drops:{drops}")
            ui_state["main"] = " | ".join(parts)

        live_timer = QTimer(app)
        live_timer.timeout.connect(live_tick)
        live_timer.start(30)

    # ------------------------------------------------------------------ shared clock/status render

    def render_status():
        elapsed = int(time.time() - start_time)
        text = f"{ui_state['main']} | {elapsed // 60:02d}:{elapsed % 60:02d}"
        if alert["msg"] and time.time() < alert["until"]:
            text += f"  !! {alert['msg']}"
            overlay.set_status(text, alert=True)
        else:
            overlay.set_status(text, alert=False)

    clock_timer = QTimer(app)
    clock_timer.timeout.connect(render_status)
    clock_timer.start(500)
    render_status()

    if duration:
        QTimer.singleShot(duration * 1000, app.quit)

    if mode != "live":  # live mode already printed its detailed header above
        print("=" * 60)
        print(f"  PHASE 5 OVERLAY ({mode} mode)")
        print("=" * 60)
        print(describe_glossary(terms))
        print(f" Overlay : bottom bar | click-through: {overlay.click_through}")
        print(" Stop    : Ctrl+C in this console" + (f" (auto-stop in {duration}s)" if duration else ""))
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------ run + shutdown

    app.exec()

    if pipeline is not None:
        summary = pipeline.stop()
        _print_summary(summary, title="PHASE 5 SUMMARY")
    overlay.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
