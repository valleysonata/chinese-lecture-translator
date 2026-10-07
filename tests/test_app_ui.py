"""
Main window app self-check (offline: no mic, no API calls).

Verifies the base UI is a normal window (draggable/closable, NOT frameless),
feed card lifecycle (ZH first -> EN fill -> delay badge -> errors in place),
feed cap, slides drop-zone loading + prompt injection, overlay-as-a-mode
toggle, and Start/Stop with a stubbed pipeline.

Run: py tests\\test_app_ui.py
"""
import io
import sys
import tempfile
import contextlib
from pathlib import Path

# Ensure root directory is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
except ImportError:
    print("[SKIP] PyQt6 not installed; main window UI tests skipped.")
    sys.exit(0)


class _StubSession:
    def __init__(self):
        self.events = []

    def log(self, kind, payload=None):
        self.events.append((kind, payload))


class _StubEngine:
    def __init__(self, system_prompt=None):
        self.system_prompt = system_prompt


class _StubCapture:
    dropped_frames = 0


class _StubPipeline:
    """Stand-in for LivePipeline: no mic, no threads, no API."""

    def __init__(self, system_prompt=None, glossary_terms=None, device_index=None,
                 wav_prefix="seg", log_prefix="session", on_event=None):
        self.trans_engine = _StubEngine(system_prompt)
        self.session = _StubSession()
        self.capture = _StubCapture()
        self.stats = {"dropped_asr": 0, "dropped_trans": 0}
        self.on_event = on_event
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def process_once(self, timeout=0.0):
        return {"status": "LISTENING", "rms": 0.05, "prob": 0.0, "asr_q": 0, "trans_q": 0}

    def stop(self):
        self.stopped = True
        return {
            "runtime_s": 0.1, "utterances": 0, "transcripts": 0, "translations": 0,
            "dropped_asr": 0, "dropped_translation": 0, "empty_transcripts": 0,
            "truncated_translations": 0, "rate_limit_errors": 0,
            "dropped_audio_frames": 0, "asr_worker_drops": 0,
            "translation_worker_drops": 0, "avg_asr_latency": 0.0,
            "avg_translation_latency": 0.0, "avg_total_delay": 0.0,
            "session_log": "stub.jsonl",
        }


class _FailingPipeline(_StubPipeline):
    """Simulates a missing API key."""

    def __init__(self, **kwargs):
        raise ValueError("GROQ_API_KEY is not set.")


def test_main_window():
    print("=" * 60)
    print("  Main Window App Self-Check")
    print("=" * 60)

    app = QApplication.instance() or QApplication([])
    import scripts.app as app_mod
    from core.overlay_window import SubtitleOverlay

    win = app_mod.MainWindow()
    win.show()
    app.processEvents()

    # --- normal window chrome -----------------------------------------
    from config import APP_NAME, APP_VERSION
    assert win.windowTitle() == f"{APP_NAME} v{APP_VERSION}"
    flags = win.windowFlags()
    assert not (flags & Qt.WindowType.FramelessWindowHint), "base UI must be a normal window"
    assert not (flags & Qt.WindowType.WindowStaysOnTopHint), "base UI must not be always-on-top"
    assert win.minimumWidth() > 0 and win.minimumHeight() > 0
    print("[OK] normal window: title set, frameless/topmost absent (draggable, closable)")

    # --- card lifecycle -------------------------------------------------
    assert not win._empty_removed, "empty hint should show before the first card"
    win._handle_event({"type": "asr", "asr_index": 1, "success": True,
                       "transcript": "這個節點已經不平衡"})
    assert 1 in win.cards, "ASR event did not create a card"
    assert win._empty_removed, "empty hint should be removed after the first card"
    assert win.cards[1]["en"].text() == "…", "EN placeholder missing before translation"

    win._handle_event({"type": "translation", "asr_index": 1, "success": True,
                       "mandarin_transcript": "這個節點已經不平衡",
                       "english_translation": "This node is unbalanced.",
                       "total_delay": 1.4, "finish_reason": "stop"})
    assert "unbalanced" in win.cards[1]["en"].text()
    assert win.cards[1]["delay"].text() == "1.4s"
    assert win._delay_n == 1 and win._delay_sum > 1.0
    print("[OK] card lifecycle: ZH first, EN fills in, delay badge = 1.4s")

    # --- truncated translation renders a warning ------------------------
    win._handle_event({"type": "asr", "asr_index": 2, "success": True,
                       "transcript": "接下來我們看時間複雜度"})
    win._handle_event({"type": "translation", "asr_index": 2, "success": True,
                       "mandarin_transcript": "接下來我們看時間複雜度",
                       "english_translation": "Next we look at the time",
                       "total_delay": 2.0, "finish_reason": "length"})
    assert "…" in win.cards[2]["en"].text(), "truncation marker missing"
    assert "e6c07b" in win.cards[2]["en"].styleSheet(), "truncated card should be amber"
    print("[OK] truncated translation marked in place (amber + ellipsis)")

    # --- failed translation renders red error ---------------------------
    win._handle_event({"type": "asr", "asr_index": 3, "success": True,
                       "transcript": "左子樹比較高"})
    win._handle_event({"type": "translation", "asr_index": 3, "success": False,
                       "mandarin_transcript": "左子樹比較高", "english_translation": "",
                       "total_delay": 0.4, "finish_reason": None,
                       "error_type": "rate_limit", "error": "429"})
    assert "RATE LIMIT" in win.cards[3]["en"].text()
    assert "ff8d8d" in win.cards[3]["en"].styleSheet(), "failed card should be red"
    print("[OK] failed translation kept visible as a red RATE LIMIT badge")

    # --- dropped translation warns on its card --------------------------
    win._handle_event({"type": "asr", "asr_index": 4, "success": True,
                       "transcript": "時間複雜度是 O(n)"})
    win._handle_event({"type": "drop", "stage": "translation",
                       "reason": "queue_full", "asr_index": 4})
    assert "dropped" in win.cards[4]["en"].text()
    print("[OK] dropped translation marked on the card")

    # --- ASR failure raises a status alert, not a card -------------------
    win._handle_event({"type": "asr", "asr_index": 5, "success": False,
                       "error_type": "rate_limit", "error": "429"})
    assert 5 not in win.cards
    win._refresh_status()
    assert "RATE LIMIT" in win.status_label.text()
    print("[OK] ASR failure surfaces as a status alert (no phantom card)")

    # --- slides: load, prompt injection, failure path --------------------
    from tests.test_slides import _make_pdf_bytes
    with tempfile.TemporaryDirectory() as td:
        pdf = Path(td) / "lecture.pdf"
        pdf.write_bytes(_make_pdf_bytes(["Binary Tree Lecture", "AVL Rotations"]))

        assert win.load_slides(str(pdf)) is True
        assert "lecture.pdf" in win.drop_zone.status.text()
        assert "2 page" in win.drop_zone.status.text()
        assert win.slides_text is not None and "Binary Tree Lecture" in win.slides_text
        assert win.slides_info is not None, "pending slides log entry missing (idle)"
        print(f"[OK] slides loaded: {win.drop_zone.status.text()}")

        bad = Path(td) / "missing.pdf"
        assert win.load_slides(str(bad)) is False
        assert "⚠" in win.drop_zone.status.text()
        assert win.slides_text is None, "failed load must clear previous slides text"
        print(f"[OK] failed load reported in the drop zone: {win.drop_zone.status.text()}")

        # Reload for the pipeline test below
        assert win.load_slides(str(pdf)) is True

    # --- overlay is a mode: toggle from the main window -------------------
    assert win._overlay is None, "overlay must not exist until toggled on"
    win.overlay_check.setChecked(True)
    app.processEvents()
    assert win._overlay is not None and win._overlay.isVisible()
    assert win._overlay.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
    assert win._overlay.click_through, "subtitle bar should stay click-through"
    win.overlay_check.setChecked(False)
    app.processEvents()
    assert not win._overlay.isVisible(), "unchecking should hide the bar"
    print("[OK] overlay is an optional mode: created on check, hidden on uncheck")

    # --- mic picker + Test mic preview (stubbed: no hardware) -------------
    class _StubMicCapture:
        """Stand-in for AudioCapture: canned chunks, no device."""
        dropped_frames = 0

        def __init__(self, config=None):
            self.config = config
            self.started = False
            self.chunks = []

        def start(self):
            self.started = True

        def stop(self):
            self.started = False
            self.chunks = []

        def get_chunk(self, timeout=0.1):
            return self.chunks.pop(0) if self.chunks else None

        @staticmethod
        def list_input_devices():
            return [(1, "Stub Mic", 0)]

    class _StubSegmenter:
        """Stand-in for SileroVADSegmenter: fixed speech probability."""
        def __init__(self, audio_config=None, vad_config=None):
            self.audio_config = audio_config

        def process_chunk(self, chunk):
            return None, 0.42

    import numpy as np

    real_capture = app_mod.AudioCapture
    real_segmenter = app_mod.SileroVADSegmenter
    app_mod.AudioCapture = _StubMicCapture
    app_mod.SileroVADSegmenter = _StubSegmenter
    try:
        assert win.device_combo.count() >= 1
        assert win.device_combo.itemText(0) == "System default"
        assert win.device_combo.itemData(0) is None
        assert win.device_index is None, "System default must map to device_index None"
        print(f"[OK] mic picker populated: {win.device_combo.count()} entries "
              "(system default + inputs)")

        # Re-populate against the stub list and select the stub device
        win._populate_devices(prefer=1)
        assert win.device_combo.count() == 2
        assert win.device_combo.currentIndex() == 1
        assert win.device_index == 1, "selecting a device must update device_index"
        print("[OK] selecting an input device updates device_index (feeds Start)")

        # Test mic: opens the selected device, streams level + VAD preview
        win.test_btn.setChecked(True)
        app.processEvents()
        cap = win._mic_capture
        assert cap is not None and cap.started, "Test mic did not open the device"
        assert cap.config.device_index == 1, "preview must use the selected device"
        assert win._mic_test_timer.isActive(), "preview timer not running"
        assert win.test_btn.text() == "Stop test"
        win._refresh_status()
        assert "MIC TEST" in win.status_label.text()
        print("[OK] Test mic opens the selected device and shows MIC TEST status")

        # Feed a known-amplitude chunk: rms 0.1 -> level bar 40, VAD readout
        cap.chunks.append(np.full(512, 0.1, dtype=np.float32))
        win._mic_test_tick()
        assert win.level_bar.value() == 40  # 0.1 * 400
        assert abs(win._mic_test_prob - 0.42) < 1e-6
        win._refresh_status()
        assert "vad 0.42" in win.status_label.text()
        print("[OK] preview tick: level meter tracks rms, VAD prob readout shown")

        # Switching devices reopens the preview on the new device
        old_cap = win._mic_capture
        win.device_combo.setCurrentIndex(0)  # back to System default
        assert win.device_index is None
        assert win._mic_capture is not old_cap, "device switch must reopen the preview"
        assert win._mic_capture.config.device_index is None
        print("[OK] switching devices reopens the preview live")

        # Stop test: device closed, timer off, meter reset
        win.test_btn.setChecked(False)
        app.processEvents()
        assert win._mic_capture is None, "Test mic did not release the device"
        assert not win._mic_test_timer.isActive()
        assert win.level_bar.value() == 0
        assert win.test_btn.text() == "Test mic"
        print("[OK] stopping the test releases the device and resets the meter")
    finally:
        app_mod.AudioCapture = real_capture
        app_mod.SileroVADSegmenter = real_segmenter

    # --- Start/Stop with a stub pipeline ---------------------------------
    original = app_mod.LivePipeline
    app_mod.LivePipeline = _StubPipeline
    try:
        # A running mic test must be released before the pipeline opens the device
        app_mod.AudioCapture = _StubMicCapture
        app_mod.SileroVADSegmenter = _StubSegmenter
        win._populate_devices(prefer=1)
        win.test_btn.setChecked(True)
        assert win._mic_capture is not None
        app_mod.AudioCapture = real_capture
        app_mod.SileroVADSegmenter = real_segmenter

        win.start_btn.setChecked(True)  # toggled signal -> _start_pipeline
        stub = win.pipeline
        assert stub is not None and stub.started, "Start did not construct/start the pipeline"
        assert win._mic_capture is None, "Start must stop the mic test first"
        assert not win.test_btn.isEnabled() and not win.device_combo.isEnabled(), \
            "mic controls must lock while the pipeline runs"
        assert win.start_btn.text() == "Stop"
        assert win._live_timer.isActive(), "live pump timer not running"
        assert stub.trans_engine.system_prompt is not None
        assert "LECTURE SLIDES" in stub.trans_engine.system_prompt, \
            "slides must be composed into the prompt at start"
        assert "Binary Tree" in stub.trans_engine.system_prompt
        assert win.slides_info is None, "slides log entry should be recorded on start"
        assert any(kind == "slides" for kind, _ in stub.session.events)
        print("[OK] Start: pipeline constructed with slides prompt + slides session log")

        # Pump one tick: stub status flows into the status row + level meter
        win._live_tick()
        app.processEvents()
        assert "LISTENING" in win.status_label.text()
        assert win.level_bar.value() == 20  # rms 0.05 * 400
        print("[OK] live tick updates status row + mic level meter")

        # Hot-swap: loading slides while running rewrites the engine prompt
        win.pipeline.trans_engine.system_prompt = "BASE"
        with tempfile.TemporaryDirectory() as td:
            pdf = Path(td) / "live.pdf"
            pdf.write_bytes(_make_pdf_bytes(["Stack and Queue"]))
            assert win.load_slides(str(pdf)) is True
            assert "STACK".lower() in stub.trans_engine.system_prompt.lower()
        assert any(kind == "slides" for kind, _ in stub.session.events), \
            "hot-swapped slides must be session-logged"
        print("[OK] slides hot-swap while running: prompt rewritten + logged")

        # Stop: summary printed (silenced here), timer off, button reset
        with contextlib.redirect_stdout(io.StringIO()):
            win.start_btn.setChecked(False)
        assert stub.stopped, "Stop did not stop the pipeline"
        assert win.pipeline is None
        assert win.start_btn.text() == "Start"
        assert not win._live_timer.isActive()
        print("[OK] Stop: pipeline stopped, summary printed, pump halted")

        # Start failure (missing API key): alert + graceful reset
        app_mod.LivePipeline = _FailingPipeline
        win.start_btn.setChecked(True)
        win._refresh_status()
        assert win.pipeline is None
        assert not win.start_btn.isChecked(), "failed start must uncheck the button"
        assert "START FAILED" in win.status_label.text()
        # _alert() renders snake_case as spaced caps (rate_limit -> RATE LIMIT)
        assert "GROQ API KEY" in win.status_label.text()
        print("[OK] failed start (missing key) alerts and resets the button")

        # Backward-compat: a run without slides must not require them
        win.slides_text = None
        app_mod.LivePipeline = _StubPipeline
        win.start_btn.setChecked(True)
        assert win.pipeline.trans_engine.system_prompt == win.base_prompt
        with contextlib.redirect_stdout(io.StringIO()):
            win.start_btn.setChecked(False)
        print("[OK] no slides -> base prompt passed through untouched")
    finally:
        app_mod.LivePipeline = original

    # --- feed cap ---------------------------------------------------------
    for i in range(10, 10 + app_mod.MAX_FEED_CARDS + 50):
        win._handle_event({"type": "asr", "asr_index": i, "success": True,
                           "transcript": f"第 {i} 句"})
    assert len(win.cards) == app_mod.MAX_FEED_CARDS, \
        f"feed cap violated: {len(win.cards)} cards"
    assert len(win.card_order) == app_mod.MAX_FEED_CARDS
    # layout = cards + the bottom stretch (empty hint was removed earlier)
    assert win._feed_layout.count() == app_mod.MAX_FEED_CARDS + 1
    assert 1 not in win.cards, "oldest card should have been evicted"
    print(f"[OK] feed capped at {app_mod.MAX_FEED_CARDS} cards, oldest evicted")

    # --- shutdown is idempotent and stops a running pipeline --------------
    app_mod.LivePipeline = _StubPipeline
    try:
        win.start_btn.setChecked(True)
        stub = win.pipeline
        with contextlib.redirect_stdout(io.StringIO()):
            win.close()
        app.processEvents()
        assert stub is not None and stub.stopped, "close must stop the pipeline"
        win.shutdown()  # second call: no-op, must not raise
    finally:
        app_mod.LivePipeline = original
    print("[OK] closeEvent stops the pipeline; shutdown() is idempotent")

    # --- demo mode ---------------------------------------------------------
    demo_win = app_mod.MainWindow(demo=True)
    demo_win.show()
    app.processEvents()
    assert not demo_win.start_btn.isEnabled(), "demo must not open the mic"
    assert not demo_win.test_btn.isEnabled(), "demo must not offer a mic test"
    assert not demo_win.device_combo.isEnabled(), "demo must not offer device selection"
    assert demo_win.start_btn.text() == "Demo"
    assert demo_win._demo_timer.isActive()
    assert len(demo_win.cards) >= 1, "demo should seed its first line immediately"
    demo_win._demo_step()
    assert len(demo_win.cards) >= 2, "demo step did not add a card"
    demo_win.close()
    app.processEvents()
    print("[OK] demo mode: Start disabled, synthetic cards render")

    win.deleteLater()
    app.processEvents()
    print("[OK] Main window self-check passed\n")


if __name__ == "__main__":
    test_main_window()
    print("All main window UI tests completed successfully!")
