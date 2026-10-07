"""
Phase 5: Transparent subtitle overlay widget (shared by all entry points).

Frameless, always-on-top, translucent bar pinned to the bottom of the screen
that renders live English subtitles over slides/IDE windows. Click-through is
enabled by default so the bar never blocks clicks.

This is a pure widget: the main window app (scripts/app.py) toggles it as an
optional mode; no pipeline logic lives here.
"""
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor

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
