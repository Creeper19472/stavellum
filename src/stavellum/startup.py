"""A short, skippable visual cover before the project welcome window."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QGuiApplication,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPixmap,
)
from PySide6.QtWidgets import QToolButton, QWidget

from . import __version__
from .branding import bind_application_icon

STARTUP_ART_PATH = Path(__file__).with_name("assets") / "startup-cover.png"


class StartupSplash(QWidget):
    """Keep the artwork separate from controls and never delay a user's skip."""

    finished = Signal()
    close_requested = Signal()
    HOLD_MS = 1400

    def __init__(self) -> None:
        super().__init__(None, Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle("Stavellum")
        bind_application_icon(self)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAccessibleName("Stavellum 启动封面")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._art = QPixmap(str(STARTUP_ART_PATH))
        self._ready = False
        self._skip_requested = False
        self._finished = False
        self.cancelled = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._finish)
        width, height = 960, 600
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            scale = min(1.0, (available.width() - 48) / width,
                        (available.height() - 48) / height)
            width, height = max(1, round(width * scale)), max(1, round(height * scale))
            self.setFixedSize(width, height)
            self.move(available.center() - self.rect().center())
        else:
            self.setFixedSize(width, height)
        self.close_button = QToolButton(self)
        self.close_button.setText("×")
        self.close_button.setToolTip("关闭 Stavellum")
        self.close_button.setAccessibleName("关闭 Stavellum")
        self.close_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.close_button.setCursor(Qt.CursorShape.ArrowCursor)
        self.close_button.setFixedSize(28, 28)
        self.close_button.move(width - 38, 10)
        self.close_button.setStyleSheet("""
            QToolButton { color: #d1e3df; background: transparent; border: 0;
                          border-radius: 4px; font-size: 18px; }
            QToolButton:hover, QToolButton:focus { background: rgba(0, 0, 0, 80); }
        """)
        self.close_button.clicked.connect(self.close)

    def mark_ready(self) -> None:
        if self.cancelled or self._finished or self._ready:
            return
        self._ready = True
        self.update()
        if self._skip_requested:
            self._finish()
        else:
            self._timer.start(self.HOLD_MS)

    def _finish(self) -> None:
        if not self._ready or self._finished or self.cancelled:
            return
        self._finished = True
        self._timer.stop()
        # Let the owner show the next window before hiding this one.
        self.finished.emit()
        self.hide()

    def _skip(self) -> None:
        self._skip_requested = True
        self._finish()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._skip()
        super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space}:
            self._skip()
            event.accept()
        elif event.key() == Qt.Key.Key_Escape:
            self.close()
            event.accept()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.cancelled = True
        self._timer.stop()
        self.close_requested.emit()
        event.accept()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.scale(self.width() / 960, self.height() / 600)
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(0, 0, 960, 600), 6, 6)
        painter.setClipPath(clip)
        painter.fillRect(QRectF(0, 0, 960, 600), QColor("#042226"))
        if not self._art.isNull():
            source = QRectF(self._art.rect())
            target_ratio = 960 / 600
            if source.width() / source.height() > target_ratio:
                crop = source.height() * target_ratio
                source.setLeft((source.width() - crop) / 2)
                source.setWidth(crop)
            else:
                crop = source.width() / target_ratio
                source.setTop((source.height() - crop) / 2)
                source.setHeight(crop)
            painter.drawPixmap(QRectF(0, 0, 960, 600), self._art, source)
        shade = QLinearGradient(0, 0, 730, 0)
        shade.setColorAt(0, QColor(0, 20, 24, 190))
        shade.setColorAt(0.52, QColor(0, 20, 24, 100))
        shade.setColorAt(1, QColor(0, 20, 24, 0))
        painter.fillRect(QRectF(0, 0, 960, 600), shade)
        # A restrained grid echoes the reference splash's compositional rhythm.
        blocks = ((0, 0, 115), (1, 0, 80), (2, 0, 130), (3, 0, 45),
                  (0, 1, 65), (2, 1, 85), (3, 1, 40), (0, 2, 105),
                  (1, 2, 50), (2, 3, 95), (3, 3, 45), (0, 4, 65),
                  (1, 4, 95), (3, 4, 80), (4, 4, 45), (0, 5, 100), (2, 5, 65))
        for column, row, opacity in blocks:
            painter.fillRect(QRectF(column * 100, row * 100, 100, 100),
                             QColor(0, 39, 35, opacity))
        font = QFont("Segoe UI")
        font.setPixelSize(60)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(QRectF(76, 67, 430, 85), "Stavellum")
        font.setPixelSize(27)
        font.setWeight(QFont.Weight.Normal)
        painter.setFont(font)
        painter.drawText(QRectF(80, 165, 280, 45), __version__)
        font.setPixelSize(16)
        painter.setFont(font)
        painter.drawText(QRectF(80, 490, 360, 32), "FLP / MIDI  →  SCORE")
        font.setPixelSize(12)
        painter.setFont(font)
        painter.setPen(QColor("#d1e3df"))
        hint = "点击或按 Enter 继续" if self._ready else "正在启动…"
        painter.drawText(QRectF(680, 545, 245, 25), Qt.AlignmentFlag.AlignRight, hint)
