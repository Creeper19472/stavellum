"""Reuse Verovio's music font outlines for score-adjacent decorations."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import verovio
from PySide6.QtSvg import QSvgRenderer

from .qt import ensure_app

MUSIC_FAMILY = "Leland"
MUSIC_FALLBACK = "Bravura"
_METRONOME_GLYPH = "ECA5"  # SMuFL metNoteQuarterUp, with a text-sized stem.


@lru_cache(maxsize=1)
def _metronome_svg() -> bytes:
    """Cache the outline and its matching font metrics, rather than a QObject."""
    resource_path = Path(verovio.toolkit().getResourcePath())
    try:
        metrics = ET.parse(resource_path / f"{MUSIC_FAMILY}.xml").getroot()
        bounds = metrics.find(f"g[@c='{_METRONOME_GLYPH}']")
        if bounds is None:
            raise ValueError("缺少速度记号边界")
        x, y, width, height = (float(bounds.get(key, "nan")) for key in ("x", "y", "w", "h"))
        if not all(math.isfinite(value) for value in (x, y, width, height)) or min(width, height) <= 0:
            raise ValueError("速度记号边界无效")
        glyph = ET.parse(resource_path / MUSIC_FAMILY / f"{_METRONOME_GLYPH}.xml").getroot()
        # Font metrics use an upward y axis; the outline already carries a
        # scale(1,-1) transform for SVG. Flip only its enclosing bounding box.
        root = ET.Element("svg", {
            "xmlns": "http://www.w3.org/2000/svg",
            "viewBox": f"{x:g} {-y - height:g} {width:g} {height:g}",
            "fill": "white",
        })
        root.append(glyph)
        return ET.tostring(root, encoding="utf-8")
    except (OSError, ET.ParseError, ValueError) as error:
        raise RuntimeError(f"无法加载 Verovio 的 {MUSIC_FAMILY} 速度记号资源。") from error


def metronome_renderer() -> QSvgRenderer:
    """Create a renderer in the caller's Qt thread from the cached vector data."""
    ensure_app()
    renderer = QSvgRenderer(_metronome_svg())
    if not renderer.isValid():
        raise RuntimeError(f"无法渲染 {MUSIC_FAMILY} 速度记号。")
    return renderer
