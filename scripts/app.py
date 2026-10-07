"""
Chinese Lecture Interpreter — main window application (base UI).

A normal, draggable, closable desktop window that drives the shared live
pipeline (mic -> VAD -> ASR -> translation):

  - Mic row: input-device picker + "Test mic" live meter (level + VAD score)
    so a dead/muted microphone is caught before the lecture starts
  - Status row: state, mic level, queue depths, drops, average delay
  - Slides drop zone: drag-and-drop (or browse) a lecture PDF; its text is
    injected into the translation system prompt as primary context
  - Feed: scrolling cards, ZH line first, EN translation fills in below,
    per-line delay badge, errors rendered in place (never vanish)
  - Start/Stop button: the app opens idle; the mic is only opened on request
  - Overlay toggle: the Phase 5 subtitle bar exists only as an optional mode

Modes:
  (default)      idle window; press Start (or pass --autostart) for live use
  --demo         synthetic subtitle stream; no mic, no API (UI smoke test)
  --list-devices enumerate microphones and exit

Examples:
  py scripts\\app.py --glossary glossary.txt --slides lecture.pdf
  py scripts\\app.py --autostart --duration 120
  py scripts\\app.py --demo --duration 15
"""
import sys
import time
import queue
import signal
import argparse
from pathlib import Path
from dataclasses import replace

import numpy as np

# Ensure root directory is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Frozen (PyInstaller, --windowed) builds have no console: give print() a
# silent target instead of crashing on a None stream.
if sys.stdout is None or sys.stderr is None:
    class _NullStream:
        def write(self, _text):
            return 0

        def flush(self):
            pass

    if sys.stdout is None:
        sys.stdout = _NullStream()
    if sys.stderr is None:
        sys.stderr = _NullStream()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton, QCheckBox,
        QComboBox, QProgressBar, QScrollArea, QFrame, QFileDialog,
        QHBoxLayout, QVBoxLayout,
    )
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QDragEnterEvent, QDropEvent
except ImportError:
    print("[ERROR] PyQt6 is required for the main window UI.")
    print("        Install it with: py -m pip install PyQt6")
    print("        Console fallback: py scripts\\test_phase3.py --glossary glossary.txt")
    sys.exit(1)

from config import DEFAULT_CONFIG, APP_NAME, APP_VERSION
from core.audio_capture import AudioCapture
from core.glossary import (
    load_glossary,
    apply_glossary,
    describe as describe_glossary,
)
from core.slides import extract_pdf_text, compose_system_prompt, SlidesError
from core.vad import SileroVADSegmenter
from core.pipeline import LivePipeline
from core.overlay_window import SubtitleOverlay
from providers.groq_translation import SYSTEM_PROMPT
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

# Bound the feed so a 90-minute lecture cannot bloat the UI
MAX_FEED_CARDS = 200

APP_STYLES = """
QWidget#root { background: #17171b; }
QLabel { color: #d9d9de; font-size: 13px; background: transparent; }
QLabel#titleLabel { color: #ffffff; font-size: 17px; font-weight: 700; }
QLabel#hintLabel { color: #8a8a93; font-size: 13px; }
QLabel#emptyLabel { color: #6a6a73; font-size: 14px; }
QLabel#zhLabel { color: #9aa2ab; font-size: 13px; }
QLabel#enLabel { color: #f2f2f5; font-size: 16px; font-weight: 600; }
QLabel#delayLabel { color: #7ec97e; font-size: 12px; font-weight: 600; }
QLabel#statusLabel { color: #c9c9d1; font-size: 12px; }
QPushButton { background: #26262d; color: #e6e6ea; border: 1px solid #3c3c45;
              border-radius: 6px; padding: 6px 16px; font-size: 13px; }
QPushButton:hover { background: #30303a; }
QPushButton:checked { background: #a83a3a; border-color: #c05050; }
QPushButton:disabled { color: #6a6a73; }
QCheckBox { color: #c9c9d1; font-size: 13px; }
QComboBox { background: #26262d; color: #e6e6ea; border: 1px solid #3c3c45;
            border-radius: 6px; padding: 5px 10px; font-size: 13px; }
QComboBox:hover { background: #30303a; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView { background: #26262d; color: #e6e6ea;
                              selection-background-color: #3c3c45; }
QFrame#feedCard { background: #202026; border-radius: 8px; }
QScrollArea { border: none; }
QProgressBar { background: #26262d; border: 1px solid #3c3c45; border-radius: 4px;
               text-align: center; max-width: 110px; max-height: 10px; }
QProgressBar::chunk { background: #4f8f4f; }
"""

DROP_STYLE_BASE = """
QFrame#dropZone { border: 2px dashed #45454f; border-radius: 10px; background: #1c1c21; }
"""
DROP_STYLE_DRAG = """
QFrame#dropZone { border: 2px dashed #6ea8ff; border-radius: 10px; background: #20262f; }
"""


class SlidesDropZone(QFrame):
    """Drag-and-drop target for the lecture PDF, with a browse fallback."""

    def __init__(self, on_pdf):
        super().__init__()
        self._on_pdf = on_pdf
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        self.setStyleSheet(DROP_STYLE_BASE)

        row = QHBoxLayout(self)
        row.setContentsMargins(14, 10, 14, 10)
        row.setSpacing(12)

        hint = QLabel("Drop the lecture PDF here")
        hint.setObjectName("hintLabel")
        self.status = QLabel("No slides loaded (PDF optional)")
        self.status.setObjectName("hintLabel")
        self.browse_btn = QPushButton("Load slides…")

        row.addWidget(hint)
        row.addWidget(self.status, 1)
        row.addWidget(self.browse_btn)
        self.browse_btn.clicked.connect(self._browse)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load lecture PDF", "", "PDF files (*.pdf)"
        )
        if path:
            self._on_pdf(path)

    @staticmethod
    def _pdf_paths(mime) -> list:
        return [
            url.toLocalFile() for url in mime.urls()
            if url.toLocalFile().lower().endswith(".pdf")
        ]

    def dragEnterEvent(self, event: QDragEnterEvent):
        if self._pdf_paths(event.mimeData()):
            event.acceptProposedAction()
            self.setStyleSheet(DROP_STYLE_DRAG)
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet(DROP_STYLE_BASE)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent):
        paths = self._pdf_paths(event.mimeData())
        if paths:
            event.acceptProposedAction()
            self._on_pdf(paths[0])
        else:
            event.ignore()
        self.setStyleSheet(DROP_STYLE_BASE)


class MainWindow(QMainWindow):
    """
    Base application window: normal window chrome (draggable/closable) hosting
    the live feed, slides drop zone, status row, Start/Stop and overlay toggle.
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        glossary_terms: list | None = None,
        device_index: int | None = None,
        screen_index: int = 0,
        clickable: bool = False,
        demo: bool = False,
    ):
        super().__init__()
        self.base_prompt = system_prompt or SYSTEM_PROMPT
        self.glossary_terms = glossary_terms or []
        self.device_index = device_index
        self._screen_index = screen_index
        self._clickable = clickable
        self.demo = demo

        self.pipeline: LivePipeline | None = None
        self.event_q: queue.Queue = queue.Queue()
        self._overlay: SubtitleOverlay | None = None
        self._last_st: dict | None = None
        self._alert_msg = ""
        self._alert_until = 0.0
        self._delay_sum = 0.0
        self._delay_n = 0
        self._start_ts = time.time()
        self._shutdown_done = False
        self._anon_keys = 0

        # Mic test mode (preview only: no ASR, no translation, no session log)
        self._mic_capture: AudioCapture | None = None
        self._mic_segmenter: SileroVADSegmenter | None = None
        self._mic_test_prob = 0.0
        self._mic_test_rms = 0.0
        self._mic_test_timer = QTimer(self)
        self._mic_test_timer.timeout.connect(self._mic_test_tick)

        self.slides_text: str | None = None
        self.slides_path: str | None = None
        self.slides_info: dict | None = None  # pending session-log entry

        self.cards: dict = {}          # asr_index -> widget refs
        self.card_order: list = []     # insertion order for the feed cap

        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(560, 380)
        self.resize(900, 660)

        root = QWidget(self)
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(10)
        root.setStyleSheet(APP_STYLES)

        # ---------------------------------------------------------------- header
        header = QHBoxLayout()
        title = QLabel("Chinese Lecture Interpreter")
        title.setObjectName("titleLabel")
        header.addWidget(title)
        header.addStretch(1)

        self.start_btn = QPushButton("Start")
        self.start_btn.setCheckable(True)
        self.start_btn.setMinimumWidth(90)
        self.start_btn.toggled.connect(self._toggle_pipeline)
        header.addWidget(self.start_btn)

        self.overlay_check = QCheckBox("Subtitle bar")
        self.overlay_check.setToolTip(
            "Show the always-on-top transparent subtitle bar over other windows"
        )
        self.overlay_check.toggled.connect(self._toggle_overlay)
        header.addWidget(self.overlay_check)
        layout.addLayout(header)

        # ---------------------------------------------------------------- mic row
        mic_row = QHBoxLayout()
        mic_row.setSpacing(8)
        mic_label = QLabel("Mic")
        mic_label.setObjectName("statusLabel")
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(240)
        self.device_combo.setToolTip(
            "Audio input device. Use 'Test mic' to verify it carries your voice."
        )
        self._populate_devices(prefer=device_index)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)

        self.test_btn = QPushButton("Test mic")
        self.test_btn.setCheckable(True)
        self.test_btn.setToolTip(
            "Open the selected mic without starting the lecture session — "
            "speak and watch the level meter and VAD score."
        )
        self.test_btn.toggled.connect(self._toggle_mic_test)

        mic_row.addWidget(mic_label)
        mic_row.addWidget(self.device_combo, 1)
        mic_row.addWidget(self.test_btn)
        layout.addLayout(mic_row)

        # ---------------------------------------------------------------- status
        status_row = QHBoxLayout()
        status_row.setSpacing(14)
        self.status_label = QLabel("STOPPED")
        self.status_label.setObjectName("statusLabel")
        status_row.addWidget(self.status_label, 1)
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setValue(0)
        self.level_bar.setTextVisible(False)
        self.level_bar.setFixedWidth(110)
        status_row.addWidget(self.level_bar)
        layout.addLayout(status_row)

        # ---------------------------------------------------------------- slides
        self.drop_zone = SlidesDropZone(self.load_slides)
        layout.addWidget(self.drop_zone)

        # ---------------------------------------------------------------- feed
        self.feed_scroll = QScrollArea()
        self.feed_scroll.setWidgetResizable(True)
        self.feed_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        feed_root = QWidget()
        feed_root.setObjectName("root")
        self._feed_layout = QVBoxLayout(feed_root)
        self._feed_layout.setContentsMargins(2, 4, 2, 4)
        self._feed_layout.setSpacing(8)
        self._empty_label = QLabel("Press Start to listen — translations appear here")
        self._empty_label.setObjectName("emptyLabel")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._feed_layout.addWidget(self._empty_label)
        self._feed_layout.addStretch(1)
        self._empty_removed = False
        self.feed_scroll.setWidget(feed_root)
        layout.addWidget(self.feed_scroll, 1)

        # ---------------------------------------------------------------- timers
        self._live_timer = QTimer(self)
        self._live_timer.timeout.connect(self._live_tick)

        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._refresh_status)
        self._clock_timer.start(500)

        self._demo_timer: QTimer | None = None
        if demo:
            self.start_btn.setEnabled(False)
            self.start_btn.setText("Demo")
            self.test_btn.setEnabled(False)
            self.device_combo.setEnabled(False)
            self._demo_state = 0
            self._demo_timer = QTimer(self)
            self._demo_timer.timeout.connect(self._demo_step)
            self._demo_timer.start(1600)
            self._demo_step()

        self._refresh_status()

    # ------------------------------------------------------------------ slides

    def load_slides(self, path) -> bool:
        """Extract the PDF and hot-inject its text into the translation prompt."""
        try:
            text, pages = extract_pdf_text(path)
        except SlidesError as e:
            self.slides_text = None
            self.slides_path = None
            self.slides_info = None
            self.drop_zone.status.setText(f"⚠ {e}")
            self.drop_zone.status.setStyleSheet("color: #ff8d8d;")
            return False

        name = Path(path).name
        self.slides_text = text
        self.slides_path = str(path)
        self.slides_info = {"file": name, "pages": pages, "chars": len(text)}
        self.drop_zone.status.setText(f"{name} — {pages} page(s), ~{len(text) / 1000:.1f}k chars")
        self.drop_zone.status.setStyleSheet("color: #7ec97e;")

        if self.pipeline is not None:
            # Engine reads system_prompt per request: effective on the next translation
            self.pipeline.trans_engine.system_prompt = compose_system_prompt(
                self.base_prompt, text
            )
            self.pipeline.session.log("slides", self.slides_info)
            self.slides_info = None  # already recorded for this session
        return True

    # ------------------------------------------------------------------ pipeline

    def _toggle_pipeline(self, checked: bool) -> None:
        if checked:
            self._start_pipeline()
        else:
            self._stop_pipeline()

    def _start_pipeline(self) -> None:
        if self._mic_capture is not None:
            self._stop_mic_test()  # free the device before the pipeline opens it
            self._set_test_checked(False)
        prompt = compose_system_prompt(self.base_prompt, self.slides_text)
        try:
            self.pipeline = LivePipeline(
                system_prompt=prompt,
                glossary_terms=self.glossary_terms,
                device_index=self.device_index,
                wav_prefix="app_seg",
                log_prefix="app",
                on_event=self.event_q.put,
            )
            self.pipeline.start()
        except Exception as e:
            detail = str(e) or e.__class__.__name__
            if isinstance(e, ValueError) and "GROQ_API_KEY" in str(e):
                detail = f"{e} — set it in .env next to this app"
            self._alert(f"start failed: {detail}", seconds=8)
            print(f"[ERROR] Could not start pipeline: {e}")
            if self.pipeline is not None:
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
            self.pipeline = None
            self._set_start_checked(False)
            self.start_btn.setText("Start")
            self._refresh_status()
            return

        if self.slides_info is not None:
            self.pipeline.session.log("slides", self.slides_info)
            self.slides_info = None

        self._last_st = None
        self.start_btn.setText("Stop")
        self.test_btn.setEnabled(False)
        self.device_combo.setEnabled(False)
        self._live_timer.start(30)
        self._refresh_status()

    def _stop_pipeline(self) -> None:
        self._live_timer.stop()
        pipeline, self.pipeline = self.pipeline, None
        self._last_st = None
        if pipeline is not None:
            summary = pipeline.stop()
            _print_summary(summary, title="APP SUMMARY")
        self._set_start_checked(False)
        self.start_btn.setText("Start" if not self.demo else "Demo")
        self.test_btn.setEnabled(not self.demo)
        self.device_combo.setEnabled(not self.demo)
        self._refresh_status()

    def _set_start_checked(self, value: bool) -> None:
        if self.start_btn.isChecked() != value:
            self.start_btn.blockSignals(True)
            self.start_btn.setChecked(value)
            self.start_btn.blockSignals(False)

    # ------------------------------------------------------------------ mic test

    def _populate_devices(self, prefer: int | None = None) -> None:
        """Fill the mic picker: 'System default' plus every input device."""
        self.device_combo.clear()
        self.device_combo.addItem("System default", None)
        select = 0
        for idx, name, _hostapi in AudioCapture.list_input_devices():
            self.device_combo.addItem(f"[{idx}] {name}", idx)
            if prefer is not None and idx == prefer:
                select = self.device_combo.count() - 1
        self.device_combo.setCurrentIndex(select)
        self.device_index = self.device_combo.currentData()

    def _on_device_changed(self, _index: int) -> None:
        self.device_index = self.device_combo.currentData()
        if self._mic_capture is not None:
            # Live switch: reopen the preview on the newly selected device
            if not self._open_mic_test():
                self._set_test_checked(False)
                self.test_btn.setText("Test mic")
                self.level_bar.setValue(0)

    def _set_test_checked(self, value: bool) -> None:
        if self.test_btn.isChecked() != value:
            self.test_btn.blockSignals(True)
            self.test_btn.setChecked(value)
            self.test_btn.blockSignals(False)

    def _toggle_mic_test(self, checked: bool) -> None:
        if checked:
            self._start_mic_test()
        else:
            self._stop_mic_test()

    def _start_mic_test(self) -> None:
        if self.pipeline is not None or self.demo:
            self._set_test_checked(False)
            return
        if not self._open_mic_test():
            self._set_test_checked(False)

    def _open_mic_test(self) -> bool:
        """Open the selected mic and stream a live level/VAD preview."""
        self._close_mic_test()
        audio_cfg = replace(DEFAULT_CONFIG.audio, device_index=self.device_index)
        try:
            self._mic_capture = AudioCapture(audio_cfg)
            self._mic_capture.start()
            if self._mic_segmenter is None:
                # Model load is one-time; device does not matter to the segmenter
                self._mic_segmenter = SileroVADSegmenter(audio_cfg, DEFAULT_CONFIG.vad)
        except Exception as e:  # bad device, device busy, permissions...
            self._close_mic_test()
            detail = str(e) or e.__class__.__name__
            self._alert(f"mic test failed: {detail}", seconds=8)
            print(f"[ERROR] Mic test failed: {e}")
            return False
        self._mic_test_prob = 0.0
        self._mic_test_rms = 0.0
        self._mic_test_timer.start(30)
        self.test_btn.setText("Stop test")
        self._refresh_status()
        return True

    def _close_mic_test(self) -> None:
        self._mic_test_timer.stop()
        if self._mic_capture is not None:
            try:
                self._mic_capture.stop()
            except Exception:
                pass
            self._mic_capture = None

    def _stop_mic_test(self) -> None:
        self._close_mic_test()
        self.test_btn.setText("Test mic")
        self.level_bar.setValue(0)
        self._refresh_status()

    def _mic_test_tick(self) -> None:
        """Drain mic chunks: update the level bar and Silero speech probability."""
        if self._mic_capture is None:
            return
        for _ in range(4):  # keep up with 32 ms chunks
            chunk = self._mic_capture.get_chunk(timeout=0.0)
            if chunk is None:
                break
            if len(chunk) == 0:
                continue
            self._mic_test_rms = float(np.sqrt(np.mean(np.square(chunk))))
            if self._mic_segmenter is not None:
                _, prob = self._mic_segmenter.process_chunk(chunk)
                self._mic_test_prob = prob
        self.level_bar.setValue(min(100, int(self._mic_test_rms * 400)))

    # ------------------------------------------------------------------ event pump

    def _live_tick(self) -> None:
        if self.pipeline is None:
            return
        status = None
        for _ in range(4):  # catch up to real time if the UI was busy
            st = self.pipeline.process_once(timeout=0.0)
            if st is not None:
                status = st
        if status is not None:
            self._last_st = status
            self.level_bar.setValue(min(100, int(status.get("rms", 0.0) * 400)))

        while True:
            try:
                evt = self.event_q.get_nowait()
            except queue.Empty:
                break
            self._handle_event(evt)

    def _handle_event(self, evt: dict) -> None:
        etype = evt.get("type")
        if etype == "asr":
            if not evt.get("success"):
                self._alert(evt.get("error_type") or "error")
                return
            text = (evt.get("transcript") or "").strip()
            if not text:
                return  # hallucination-filtered: nothing to show
            self._add_card(evt.get("asr_index"), text)

        elif etype == "translation":
            self._fill_card(evt)

        elif etype == "drop":
            reason = evt.get("reason") or "queue_full"
            idx = evt.get("asr_index")
            if idx is not None and idx in self.cards:
                self._mark_card(idx, f"⚠ translation dropped ({reason})", "warn")
            elif evt.get("stage") == "asr":
                self._alert(f"asr {reason}")

    # ------------------------------------------------------------------ feed cards

    def _next_key(self):
        self._anon_keys += 1
        return f"anon{self._anon_keys}"

    def _add_card(self, asr_index, zh: str) -> None:
        key = asr_index if asr_index is not None else self._next_key()
        if key in self.cards:
            return

        if not self._empty_removed:
            self._empty_removed = True
            self._empty_label.hide()
            self._feed_layout.removeWidget(self._empty_label)

        frame = QFrame()
        frame.setObjectName("feedCard")
        body = QVBoxLayout(frame)
        body.setContentsMargins(12, 8, 12, 8)
        body.setSpacing(4)

        top = QHBoxLayout()
        top.setSpacing(10)
        zh_label = QLabel(zh)
        zh_label.setObjectName("zhLabel")
        zh_label.setWordWrap(True)
        delay_label = QLabel("")
        delay_label.setObjectName("delayLabel")
        delay_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(zh_label, 1)
        top.addWidget(delay_label, 0, Qt.AlignmentFlag.AlignTop)
        body.addLayout(top)

        en_label = QLabel("…")
        en_label.setObjectName("enLabel")
        en_label.setWordWrap(True)
        body.addWidget(en_label)

        self.cards[key] = {
            "frame": frame, "zh": zh_label, "en": en_label, "delay": delay_label,
        }
        self.card_order.append(key)
        # Insert above the bottom stretch so cards stay top-aligned
        self._feed_layout.insertWidget(max(0, self._feed_layout.count() - 1), frame)

        while len(self.card_order) > MAX_FEED_CARDS:
            old_key = self.card_order.pop(0)
            old = self.cards.pop(old_key, None)
            if old is not None:
                self._feed_layout.removeWidget(old["frame"])
                old["frame"].setParent(None)
                old["frame"].deleteLater()

        QTimer.singleShot(0, self._scroll_to_bottom)

    def _fill_card(self, evt: dict) -> None:
        idx = evt.get("asr_index")
        card = self.cards.get(idx) if idx is not None else None

        delay = evt.get("total_delay") or 0.0
        if evt.get("success"):
            self._delay_sum += delay
            self._delay_n += 1

        if card is None:
            # Card was capped out or never created: keep the bar + alert in sync
            if not evt.get("success"):
                self._alert(evt.get("error_type") or "error")
            elif self._overlay is not None and self._overlay.isVisible():
                self._overlay.add_translation(
                    evt.get("mandarin_transcript") or "", evt.get("english_translation") or ""
                )
            return

        if evt.get("success"):
            en = evt.get("english_translation") or ""
            truncated = evt.get("finish_reason") == "length"
            if truncated:
                en = (en + " …").strip()
            card["en"].setText(en)
            card["en"].setStyleSheet("color: #e6c07b;" if truncated else "")
            card["delay"].setText(f"{delay:.1f}s")
            card["delay"].setStyleSheet("color: #e6c07b;" if truncated else "")
            if truncated:
                self._alert("translation truncated")
            if self._overlay is not None and self._overlay.isVisible():
                self._overlay.add_translation(
                    evt.get("mandarin_transcript") or card["zh"].text(), en
                )
        else:
            err = (evt.get("error_type") or "error").upper().replace("_", " ")
            card["en"].setText(f"⚠ {err}")
            card["en"].setStyleSheet("color: #ff8d8d;")
            card["en"].setToolTip(evt.get("error") or "")
            card["delay"].setText("—")
            self._alert(evt.get("error_type") or "error")

        QTimer.singleShot(0, self._scroll_to_bottom)

    def _mark_card(self, key, text: str, kind: str) -> None:
        card = self.cards.get(key)
        if card is None:
            return
        color = {"warn": "#e6c07b", "error": "#ff8d8d"}.get(kind, "")
        card["en"].setText(text)
        card["en"].setStyleSheet(f"color: {color};" if color else "")
        QTimer.singleShot(0, self._scroll_to_bottom)

    def _scroll_to_bottom(self) -> None:
        bar = self.feed_scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    # ------------------------------------------------------------------ demo mode

    def _demo_step(self) -> None:
        zh, en = DEMO_LINES[self._demo_state % len(DEMO_LINES)]
        self._demo_state += 1
        idx = self._demo_state
        self._handle_event({
            "type": "asr", "asr_index": idx, "success": True, "transcript": zh,
        })
        delay = round(1.0 + 0.5 * ((idx * 7) % 5) / 4, 1)
        self._handle_event({
            "type": "translation", "trans_index": idx, "asr_index": idx,
            "success": True, "mandarin_transcript": zh,
            "english_translation": en, "total_delay": delay,
            "finish_reason": "stop",
        })

    # ------------------------------------------------------------------ overlay

    def _toggle_overlay(self, checked: bool) -> None:
        if checked:
            if self._overlay is None:
                self._overlay = SubtitleOverlay(
                    screen_index=self._screen_index,
                    click_through=not self._clickable,
                )
            self._overlay.show()
            self._overlay.raise_()
        elif self._overlay is not None:
            self._overlay.hide()

    # ------------------------------------------------------------------ status

    def _alert(self, msg: str, seconds: float = 5.0) -> None:
        self._alert_msg = (msg or "error").upper().replace("_", " ")
        self._alert_until = time.time() + seconds

    def _refresh_status(self) -> None:
        parts = []
        if self.demo:
            parts.append("DEMO")
        elif self.pipeline is not None:
            st = self._last_st
            parts.append(st["status"] if st else "LISTENING")
            if st is not None:
                parts.append(f"ASR Q:{st['asr_q']} Trans Q:{st['trans_q']}")
            drops = (
                self.pipeline.stats["dropped_asr"]
                + self.pipeline.stats["dropped_trans"]
                + self.pipeline.capture.dropped_frames
            )
            parts.append(f"drops:{drops}")
        elif self._mic_capture is not None:
            parts.append("MIC TEST — speak now")
            parts.append(f"level {self._mic_test_rms:.3f}")
            parts.append(f"vad {self._mic_test_prob:.2f}")
        else:
            parts.append("STOPPED")

        elapsed = int(time.time() - self._start_ts)
        parts.append(f"{elapsed // 60:02d}:{elapsed % 60:02d}")
        if self._delay_n:
            parts.append(f"avg delay {self._delay_sum / self._delay_n:.1f}s")

        alerting = bool(self._alert_msg) and time.time() < self._alert_until
        if alerting:
            parts.append(f"!! {self._alert_msg}")
        self.status_label.setText(" | ".join(parts))
        if alerting:
            self.status_label.setStyleSheet("color: #ff8d8d; font-weight: 600;")
        elif self.pipeline is not None or self._mic_capture is not None:
            self.status_label.setStyleSheet("color: #c9c9d1;")
        else:
            self.status_label.setStyleSheet("color: #8a8a93;")

    # ------------------------------------------------------------------ lifecycle

    def shutdown(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self._clock_timer.stop()
        self._live_timer.stop()
        self._close_mic_test()
        if self._demo_timer is not None:
            self._demo_timer.stop()
        if self.pipeline is not None:
            self._stop_pipeline()
        if self._overlay is not None:
            self._overlay.close()

    def closeEvent(self, event) -> None:
        self.shutdown()
        event.accept()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chinese Lecture Interpreter: main window app"
    )
    parser.add_argument("--demo", action="store_true",
                        help="Synthetic subtitle stream for UI testing (no mic, no API)")
    parser.add_argument("--device", type=int, default=None, help="Audio input device index")
    parser.add_argument("--duration", type=int, default=None,
                        help="Auto close after N seconds (default: demo=20)")
    parser.add_argument("--glossary", type=str, default=None,
                        help="Optional advanced: glossary .txt (one term per line) "
                             "injected into ASR + translation prompts")
    parser.add_argument("--slides", type=str, default=None,
                        help="Lecture PDF loaded at launch (primary context source; "
                             "can also be dropped onto the window)")
    parser.add_argument("--autostart", action="store_true",
                        help="Open the microphone immediately instead of idle")
    parser.add_argument("--overlay", action="store_true",
                        help="Show the transparent subtitle bar at launch")
    parser.add_argument("--screen", type=int, default=0,
                        help="Screen index for the subtitle bar (default 0 = primary)")
    parser.add_argument("--clickable", action="store_true",
                        help="Subtitle bar accepts mouse input (default: click-through)")
    parser.add_argument("--list-devices", action="store_true",
                        help="List audio input devices and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    disable_quickedit()

    if args.list_devices:
        for idx, name, hostapi in AudioCapture.list_input_devices():
            print(f"[{idx}] {name} (hostapi {hostapi})")
        return 0

    # Glossary (optional, advanced): injected before any engine is constructed
    terms: list = []
    if args.glossary:
        try:
            terms = load_glossary(args.glossary)
        except (FileNotFoundError, UnicodeDecodeError) as e:
            print(f"[ERROR] {e}")
            return 1
    system_prompt = apply_glossary(DEFAULT_CONFIG.asr, terms, SYSTEM_PROMPT) or SYSTEM_PROMPT

    app = QApplication(sys.argv)
    window = MainWindow(
        system_prompt=system_prompt,
        glossary_terms=terms,
        device_index=args.device,
        screen_index=args.screen,
        clickable=args.clickable,
        demo=args.demo,
    )

    slides_ok = True
    if args.slides:
        slides_ok = window.load_slides(args.slides)
        if not slides_ok:
            print(f"[WARN] Slides not loaded: {window.drop_zone.status.text()}")

    if args.overlay:
        window.overlay_check.setChecked(True)  # signal drives the toggle
    window.show()

    # Graceful Ctrl+C: ask Qt to leave its event loop; shutdown runs after exec()
    signal.signal(signal.SIGINT, lambda *_: app.quit())

    duration = args.duration
    if duration is None and args.demo:
        duration = 20
    if duration:
        QTimer.singleShot(duration * 1000, window.close)

    print("=" * 60)
    print(f"  {APP_NAME.upper()} v{APP_VERSION}")
    print("=" * 60)
    print(f" Mode          : {'DEMO (no mic/API)' if args.demo else 'window (press Start or --autostart)'}")
    print(f" Input device  : {window.device_combo.currentText()}")
    print(f" ASR Model     : {DEFAULT_CONFIG.asr.model} (Mandarin + English Code-switching)")
    print(f" Trans Model   : {DEFAULT_CONFIG.translation.model} (Natural CS English)")
    print(f" Silence Wait  : {DEFAULT_CONFIG.vad.silence_duration_ms} ms (automatic trigger)")
    print(describe_glossary(terms))
    if args.slides:
        state = window.drop_zone.status.text() if slides_ok else "FAILED (see window)"
        print(f" Slides        : {state}")
    print(f" Subtitle bar  : {'on' if window.overlay_check.isChecked() else 'off (toggle in window)'}")
    print(" Stop          : close the window" + (f" (auto-close in {duration}s)" if duration else ""))
    print("=" * 60 + "\n")

    if args.autostart and not args.demo:
        window.start_btn.setChecked(True)  # signal drives startup

    app.exec()
    window.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
