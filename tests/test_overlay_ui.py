"""
Phase 5 overlay UI self-check (offline: no mic, no API calls).

Verifies the SubtitleOverlay widget renders: frameless/topmost flags,
bottom-of-screen geometry, click-through extended style at the Win32 level,
subtitle line rotation, and actual painted pixels (dark bar + white text).

Run: py tests\\test_overlay_ui.py
"""
import sys
from pathlib import Path

# Ensure root directory is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
except ImportError:
    print("[SKIP] PyQt6 not installed; overlay UI tests skipped.")
    sys.exit(0)


def _pixel_stats(qimage, x0, y0, w, h, step=2):
    """Count bright (text) and dark (bar) pixels in a region of a QImage."""
    bright = dark = total = 0
    for y in range(y0, min(y0 + h, qimage.height()), step):
        for x in range(x0, min(x0 + w, qimage.width()), step):
            c = qimage.pixelColor(x, y)
            lum = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
            total += 1
            if c.alpha() > 200 and lum > 230:
                bright += 1
            if lum < 35 and c.alpha() > 100:
                dark += 1
    return bright, dark, total


def test_overlay_widget():
    print("=" * 60)
    print("  Phase 5: Overlay Widget Self-Check")
    print("=" * 60)

    app = QApplication.instance() or QApplication([])
    from core.overlay_window import SubtitleOverlay, BAR_HEIGHT

    overlay = SubtitleOverlay(screen_index=0, click_through=True)
    overlay.show()
    app.processEvents()

    # --- window flags -------------------------------------------------
    flags = overlay.windowFlags()
    assert flags & Qt.WindowType.FramelessWindowHint, "missing FramelessWindowHint"
    assert flags & Qt.WindowType.WindowStaysOnTopHint, "missing WindowStaysOnTopHint"
    assert flags & Qt.WindowType.Tool, "missing Tool flag"
    print("[OK] frameless + always-on-top + tool-window flags set")

    # --- visibility + geometry (bottom of primary screen) -------------
    assert overlay.isVisible(), "overlay widget not visible after show()"
    app.processEvents()
    from PyQt6.QtWidgets import QApplication as _QA
    geo = _QA.primaryScreen().availableGeometry()
    rect = overlay.geometry()
    assert rect.width() == geo.width() - 16, f"width {rect.width()} != {geo.width() - 16}"
    bottom_gap = (geo.y() + geo.height()) - (rect.y() + rect.height())
    assert 0 <= bottom_gap <= 16, f"bar bottom gap {bottom_gap}px off the screen bottom"
    assert rect.height() == BAR_HEIGHT, f"height {rect.height()} != {BAR_HEIGHT}"
    print(f"[OK] visible at screen bottom (gap {bottom_gap}px, {rect.width()}x{rect.height()})")

    # --- click-through at the Win32 level -----------------------------
    assert overlay.click_through, "click-through flag was not applied"
    if sys.platform == "win32":
        import ctypes
        WS_EX_TRANSPARENT = 0x20
        ex_style = ctypes.windll.user32.GetWindowLongW(int(overlay.winId()), -20)
        assert ex_style & WS_EX_TRANSPARENT, f"WS_EX_TRANSPARENT not set (exstyle=0x{ex_style:x})"
        print("[OK] WS_EX_TRANSPARENT set: clicks pass through the bar")
    else:
        print("[OK] click-through flag reported by Qt (Win32 check skipped)")

    # --- subtitle content rotation ------------------------------------
    overlay.set_hint("placeholder hint")
    assert overlay.en_label.text() == "placeholder hint", "set_hint did not apply to empty state"

    overlay.add_translation("第一句", "First translated line.")
    assert overlay.en_label.text() == "First translated line."
    assert overlay.zh_label.text() == "第一句"
    assert overlay.prev_label.text() == ""
    print("[OK] first line renders as latest subtitle")

    overlay.add_translation("第二句", "Second translated line.")
    overlay.add_translation("第三句", "Third translated line.")
    overlay.add_translation("第四句", "Fourth translated line.")
    assert overlay.en_label.text() == "Fourth translated line."
    assert overlay.prev_label.text().splitlines() == [
        "Third translated line.",
        "Second translated line.",
        "First translated line.",
    ], f"unexpected history: {overlay.prev_label.text()!r}"
    overlay.set_hint("hint must not override content")
    assert overlay.en_label.text() == "Fourth translated line.", "set_hint overwrote live content"
    print("[OK] history rotates: latest large, 3 faded above")

    # --- status strip --------------------------------------------------
    overlay.set_status("LISTENING | ASR Q:0 | 00:05")
    assert overlay.status_label.text() == "LISTENING | ASR Q:0 | 00:05"
    overlay.set_status("LISTENING  !! RATE LIMIT", alert=True)
    assert "RATE LIMIT" in overlay.status_label.text()
    print("[OK] status strip updates (normal + alert)")

    # --- actual painted pixels -----------------------------------------
    # Force the pending layout so label geometry matches the new content...
    overlay.layout().activate()
    # ...then render the widget offline and check that the dark translucent
    # bar and the white subtitle text really got painted.
    image = overlay.grab().toImage()
    assert not image.isNull(), "widget grab produced a null image"

    # Region with no label text: left margin inside the rounded rectangle.
    # grab() returns device pixels; label geometry is logical -> scale by DPR.
    dpr = overlay.devicePixelRatioF()
    bright_l, dark_l, total_l = _pixel_stats(
        image, round(8 * dpr), round((BAR_HEIGHT // 2 - 20) * dpr), round(8 * dpr), round(40 * dpr)
    )
    assert dark_l > total_l * 0.8, f"bar background not painted: dark {dark_l}/{total_l}"

    # Region of the large EN label: white glyphs must be present
    en_geo = overlay.en_label.geometry()
    bright_t, dark_t, total_t = _pixel_stats(
        image, round(en_geo.x() * dpr), round(en_geo.y() * dpr),
        round(en_geo.width() * dpr), round(en_geo.height() * dpr)
    )
    assert bright_t > 150, f"subtitle text not painted: bright {bright_t}/{total_t}"
    print(f"[OK] pixels: dark bar background ({dark_l}/{total_l}), "
          f"white text glyphs ({bright_t}/{total_t})")

    overlay.close()
    app.processEvents()
    print("[OK] Overlay widget self-check passed\n")


if __name__ == "__main__":
    test_overlay_widget()
    print("All overlay UI tests completed successfully!")
