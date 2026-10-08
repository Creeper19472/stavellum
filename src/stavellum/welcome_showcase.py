"""A standalone, purely presentational landing-page showcase.

Run it directly to view the design as a small frameless window:

    uv run python -m stavellum.welcome_showcase

Nothing here is wired to application state: the entries are static samples
and the buttons only animate on hover.
"""

from __future__ import annotations

import sys

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .branding import APPLICATION_DESCRIPTION, bind_application_icon, logo_pixmap
from .qt import ensure_app

_ACCENT = "#edc398"
_ACCENT_HOVER = "#f7d6b3"
_ACCENT_TEXT = "#20242a"
_PANEL = "#1a212b"
_PANEL_HOVER = "#212b37"
_PANEL_BORDER = "#2a3441"
_TEXT = "#e8ecf1"
_MUTED = "#8b98a8"

_SAMPLE_PROJECTS = (
    ("三乐器示例.stproj", r"artifacts\demo\三乐器示例.stproj"),
    ("协奏曲总谱.stproj", r"D:\谱面\协奏曲总谱.stproj"),
    ("夜曲·四手联弹.stproj", r"D:\谱面\夜曲·四手联弹.stproj"),
)


class ShowcasePage(QWidget):
    """Static reproduction of the landing-page design study."""

    def __init__(self) -> None:
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setWindowTitle("Stavellum · 设计展示")
        bind_application_icon(self)
        self.setFixedSize(780, 540)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setStyleSheet(f"""
            QWidget {{ color: {_TEXT}; font-size: 13px; }}
            QLabel#Tagline {{ color: {_MUTED}; }}
            QLabel#VersionChip {{
                color: {_ACCENT}; border: 1px solid rgba(237, 195, 152, 90);
                border-radius: 9px; padding: 1px 9px; font-size: 11px;
            }}
            QLabel#SectionHeading {{ color: {_MUTED}; font-size: 12px; letter-spacing: 2px; }}
            QLabel#SampleName {{ font-size: 14px; font-weight: 600; }}
            QLabel#SamplePath {{ color: {_MUTED}; font-size: 11px; }}
            QPushButton {{ border: 0; }}
            QPushButton#Primary {{
                color: {_ACCENT_TEXT}; background: {_ACCENT};
                border-radius: 9px; font-size: 14px; font-weight: 600;
                padding: 11px 26px;
            }}
            QPushButton#Primary:hover {{ background: {_ACCENT_HOVER}; }}
            QPushButton#Secondary {{
                color: {_TEXT}; background: {_PANEL}; border: 1px solid {_PANEL_BORDER};
                border-radius: 9px; font-size: 14px; padding: 11px 24px;
            }}
            QPushButton#Secondary:hover {{ background: {_PANEL_HOVER}; }}
            QPushButton#TextLink {{ color: {_ACCENT}; padding: 4px 2px; }}
            QPushButton#TextLink:hover {{ color: {_ACCENT_HOVER}; }}
        """)
        logo = logo_pixmap().scaled(
            56, 56, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(34, 30, 30, 24)
        layout.setSpacing(0)

        header = QHBoxLayout()
        header.setSpacing(14)
        logo_label = QLabel()
        logo_label.setPixmap(logo)
        titles = QVBoxLayout()
        titles.setSpacing(3)
        name_row = QHBoxLayout()
        name_row.setSpacing(10)
        title_label = QLabel("Stavellum")
        title_font = title_label.font()
        title_font.setPointSize(21)
        title_font.setBold(True)
        title_font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 105)
        title_label.setFont(title_font)
        chip = QLabel(f"v{__version__}")
        chip.setObjectName("VersionChip")
        name_row.addWidget(title_label)
        name_row.addWidget(chip)
        name_row.addStretch(1)
        titles.addLayout(name_row)
        tagline = QLabel(APPLICATION_DESCRIPTION)
        tagline.setObjectName("Tagline")
        titles.addWidget(tagline)
        header.addWidget(logo_label)
        header.addLayout(titles, 1)
        layout.addLayout(header)
        layout.addSpacing(28)

        actions = QHBoxLayout()
        actions.setSpacing(12)
        primary = QPushButton("新建工程")
        primary.setObjectName("Primary")
        primary.setCursor(Qt.CursorShape.PointingHandCursor)
        secondary = QPushButton("打开工程…")
        secondary.setObjectName("Secondary")
        secondary.setCursor(Qt.CursorShape.PointingHandCursor)
        actions.addWidget(primary)
        actions.addWidget(secondary)
        actions.addStretch(1)
        layout.addLayout(actions)
        layout.addSpacing(8)
        link = QPushButton("生成示例工程")
        link.setObjectName("TextLink")
        link.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(link)
        layout.addSpacing(26)

        heading = QLabel("最近打开")
        heading.setObjectName("SectionHeading")
        layout.addWidget(heading)
        layout.addSpacing(8)
        for name, path in _SAMPLE_PROJECTS:
            card = QWidget()
            card.setStyleSheet(f"""
                QWidget#Card {{ background: {_PANEL}; border: 1px solid {_PANEL_BORDER};
                                border-radius: 10px; }}
            """)
            card.setObjectName("Card")
            card.setCursor(Qt.CursorShape.PointingHandCursor)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(14, 9, 14, 9)
            card_layout.setSpacing(2)
            card_title = QLabel(name)
            card_title.setObjectName("SampleName")
            card_path = QLabel(path)
            card_path.setObjectName("SamplePath")
            card_layout.addWidget(card_title)
            card_layout.addWidget(card_path)
            layout.addWidget(card)
        layout.addStretch(1)

        footer = QHBoxLayout()
        hint = QLabel("设计稿 · 仅展示，未接入任何功能")
        hint.setObjectName("Tagline")
        footer.addWidget(hint, 1)
        version = QLabel(f"Stavellum v{__version__}")
        version.setObjectName("Tagline")
        footer.addWidget(version, 0, Qt.AlignmentFlag.AlignRight)
        layout.addLayout(footer)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        background = QLinearGradient(0, 0, 0, self.height())
        background.setColorAt(0.0, QColor("#1b222c"))
        background.setColorAt(1.0, QColor("#0e131a"))
        painter.fillRect(self.rect(), background)
        self._paint_art(painter)

    @staticmethod
    def _paint_art(painter: QPainter) -> None:
        width, height = 780, 540
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        glow = QRadialGradient(QPointF(width * 0.86, height * 0.30), width * 0.5)
        glow.setColorAt(0.0, QColor(88, 140, 255, 40))
        glow.setColorAt(0.55, QColor(88, 140, 255, 14))
        glow.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.fillRect(QRectF(0, 0, width, height), glow)
        stroke = QLinearGradient(width * 0.58, 0, width, 0)
        stroke.setColorAt(0.0, QColor(232, 236, 241, 0))
        stroke.setColorAt(0.45, QColor(232, 236, 241, 38))
        stroke.setColorAt(1.0, QColor(232, 236, 241, 12))
        painter.setPen(QPen(stroke, 1.4))
        for index in range(5):
            offset = index * 16
            line = QPainterPath()
            start_y = height * (0.20 + 0.014 * index)
            line.moveTo(width * 0.56, start_y + offset * 0.4)
            line.cubicTo(width * 0.74, start_y - 70 + offset,
                         width * 0.86, height * 0.66 + offset,
                         width + 26, height * (0.56 + 0.055 * index) + offset)
            painter.drawPath(line)
        notes = ((0.68, 0.26, 5.4), (0.76, 0.36, 4.4), (0.83, 0.50, 6.0),
                 (0.90, 0.32, 3.8), (0.94, 0.60, 4.8))
        for progress_x, progress_y, radius in notes:
            x, y = width * progress_x, height * progress_y
            halo = QRadialGradient(QPointF(x, y), radius * 3.2)
            halo.setColorAt(0.0, QColor(237, 195, 152, 95))
            halo.setColorAt(1.0, QColor(237, 195, 152, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(halo)
            painter.drawEllipse(QPointF(x, y), radius * 3.2, radius * 3.2)
            painter.setBrush(QColor(240, 208, 170, 220))
            painter.drawEllipse(QPointF(x, y), radius, radius)
        painter.restore()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.windowHandle():
            self.windowHandle().startSystemMove()


def main() -> None:
    ensure_app(offscreen=False)
    application = QApplication.instance()
    page = ShowcasePage()
    page.show()
    sys.exit(application.exec())


if __name__ == "__main__":
    main()
