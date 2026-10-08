"""Exercise the actual Leland outline and font-independent SVG output."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from importlib.resources import files

import pytest
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QImage, QPainter

from stavellum.graphics.musicfont import _metronome_svg, metronome_renderer


def test_metronome_outline_is_visible_white_and_scales_as_a_vector():
    renderer = metronome_renderer()
    box = renderer.viewBoxF()
    assert box.width() > 0 and box.height() > box.width()
    svg = ET.fromstring(_metronome_svg())
    assert svg.find(".//{http://www.w3.org/2000/svg}path") is not None
    assert not svg.findall(".//{http://www.w3.org/2000/svg}text")
    areas = []
    for height in (60, 120):
        image = QImage(160, 160, QImage.Format.Format_RGBA8888)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(painter, QRectF(10, 10, height * box.width() / box.height(), height))
        painter.end()
        ink = [image.pixelColor(x, y) for y in range(160) for x in range(160)
               if image.pixelColor(x, y).alpha() > 0]
        assert ink and all(color.red() == color.green() == color.blue() == 255 for color in ink)
        areas.append(len(ink))
    assert areas[1] / areas[0] == pytest.approx(4, rel=0.12)


def test_independent_renderers_do_not_share_a_qobject():
    first = metronome_renderer()
    second = metronome_renderer()
    assert first is not second
    assert first.viewBoxF() == second.viewBoxF()


def test_leland_source_and_license_are_bundled():
    directory = files("stavellum").joinpath("fonts")
    assert "SIL OPEN FONT LICENSE" in directory.joinpath("Leland-LICENSE.txt").read_text()
    assert "Leland Version 0.80" in directory.joinpath("Leland-FONTLOG.txt").read_text()
