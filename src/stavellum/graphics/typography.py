"""Bundled score typography shared by raster frames and printable metadata."""

from __future__ import annotations

import unicodedata
from functools import lru_cache

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QFont,
    QPainter,
    QTextCharFormat,
    QTextLayout,
    QTextOption,
)

from .font_registry import CJK_FAMILY, LATIN_FAMILY


@lru_cache(maxsize=128)
def _font(pixel_size: int, cjk: bool, italic: bool = False) -> QFont:
    font = QFont(CJK_FAMILY if cjk else LATIN_FAMILY)
    font.setStyleName("Regular" if cjk else "Italic" if italic else "Roman")
    font.setPixelSize(pixel_size)
    font.setWeight(QFont.Weight.Normal)
    font.setItalic(italic and not cjk)
    font.setKerning(True)
    # Explicit ranges must not silently acquire an unrelated installed typeface.
    font.setStyleStrategy(QFont.StyleStrategy.NoFontMerging)
    return font


def script_runs(text: str) -> list[tuple[str, bool]]:
    """Split script ranges consistently for Qt text shaping and SVG text."""
    runs: list[tuple[str, bool]] = []
    previous_cjk = False
    start = 0
    for position, character in enumerate(text):
        cjk = (
            previous_cjk if unicodedata.category(character).startswith("M")
            else unicodedata.east_asian_width(character) in ("W", "F")
            or 0xFF00 <= ord(character) <= 0xFFEF
        )
        if position and cjk != previous_cjk:
            runs.append((text[start:position], previous_cjk))
            start = position
        previous_cjk = cjk
    if text:
        runs.append((text[start:], previous_cjk))
    return runs


def _formats(text: str, pixel_size: int, italic: bool = False) -> list[QTextLayout.FormatRange]:
    ranges = []
    position = 0
    for text_run, cjk in script_runs(text):
        length = sum(2 if ord(character) > 0xFFFF else 1 for character in text_run)
        value = QTextCharFormat()
        value.setFont(_font(pixel_size, cjk, italic))
        entry = QTextLayout.FormatRange()
        entry.start, entry.length, entry.format = position, length, value
        ranges.append(entry)
        position += length
    return ranges


def _layout(
    text: str,
    pixel_size: int | float,
    *,
    device=None,
    width: float | None = None,
    alignment=Qt.AlignmentFlag.AlignLeft,
    wrap: bool = False,
    italic: bool = False,
) -> QTextLayout:
    from .qt import ensure_app

    ensure_app()
    size = max(1, round(pixel_size))
    # QTextLayout uses the Unicode line separator for explicit line breaks.
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\u2028")
    layout = QTextLayout(text, _font(size, False, italic), device)
    layout.setFormats(_formats(text, size, italic))
    option = QTextOption()
    option.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignAbsolute)
    option.setWrapMode(
        QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere if wrap
        else QTextOption.WrapMode.NoWrap
    )
    layout.setTextOption(option)
    layout.beginLayout()
    y = 0.0
    while True:
        line = layout.createLine()
        if not line.isValid():
            break
        line.setLineWidth(max(1.0, width) if width is not None else 1e9)
        text_width = line.naturalTextWidth()
        if width is None:
            line.setLineWidth(text_width)
        x = 0.0
        if width is not None:
            if alignment & Qt.AlignmentFlag.AlignHCenter:
                x = (width - text_width) / 2
            elif alignment & Qt.AlignmentFlag.AlignRight:
                x = width - text_width
        line.setPosition(QPointF(x, y))
        y += line.height()
    layout.endLayout()
    return layout


def draw_text(
    painter: QPainter, text: str, pixel_size: int | float, baseline: QPointF,
) -> None:
    """Draw a shaped mixed-script string with its first line on this baseline."""
    if not text:
        return
    layout = _layout(text, pixel_size, device=painter.device())
    layout.draw(painter, baseline - QPointF(0, layout.lineAt(0).ascent()))


def text_height(text: str, pixel_size: int | float, width: float, *, device=None,
                wrap: bool = False) -> float:
    """Measure the same shaped lines that the rectangle drawing helper uses."""
    if not text:
        return 0.0
    layout = _layout(text, pixel_size, device=device, width=width, wrap=wrap)
    return sum(layout.lineAt(i).height() for i in range(layout.lineCount()))


def draw_text_rect(
    painter: QPainter,
    text: str,
    pixel_size: int | float,
    rect: QRectF,
    alignment=Qt.AlignmentFlag.AlignCenter,
    wrap: bool = False,
) -> None:
    """Shape, optionally wrap, and align metadata within a bounded rectangle."""
    if not text or rect.isEmpty():
        return
    layout = _layout(text, pixel_size, device=painter.device(), width=rect.width(),
                     alignment=alignment, wrap=wrap)
    height = sum(layout.lineAt(i).height() for i in range(layout.lineCount()))
    top = rect.top()
    if alignment & Qt.AlignmentFlag.AlignVCenter:
        top += (rect.height() - height) / 2
    elif alignment & Qt.AlignmentFlag.AlignBottom:
        top += rect.height() - height
    painter.save()
    painter.setClipRect(rect, Qt.ClipOperation.IntersectClip)
    layout.draw(painter, QPointF(rect.left(), top))
    painter.restore()
