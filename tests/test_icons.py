"""Bundled instrument SVG matching, reuse, and proportional raster output."""

import pytest
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QImage, QPainter

from stavellum.graphics import icons
from stavellum.graphics.icons import icon_renderer
from stavellum.graphics.qt import ensure_app


@pytest.fixture(autouse=True)
def fresh_cache():
    ensure_app()
    icons._load_icon.cache_clear()
    yield
    icons._load_icon.cache_clear()


@pytest.mark.parametrize(("kind", "filename"), [
    ("Violin I", "violin-thin-100.svg"), ("cello", "violin-thin-100.svg"),
    ("double_bass", "violin-thin-100.svg"),
    ("第一小提琴", "violin-thin-100.svg"), ("bell", "bell-thin-100.svg"),
    ("Celesta", "bell-thin-100.svg"), ("percussion", "drum-thin-100.svg"),
    ("打击乐组", "drum-thin-100.svg"), ("piano", "piano-thin-100.svg"),
    ("Electric Piano", "piano-keyboard-thin-100.svg"),
    ("piano-keyboard", "piano-keyboard-thin-100.svg"),
    ("violin-thin-100.svg", "violin-thin-100.svg"),
])
def test_instrument_aliases_select_valid_packaged_resources(kind, filename):
    assert icons._resource_name(kind) == filename
    renderer = icon_renderer(kind)
    assert renderer is not None and renderer.isValid()
    assert renderer.aspectRatioMode() == Qt.AspectRatioMode.KeepAspectRatio


def test_aliases_reuse_the_same_renderer_and_unknowns_request_fallback():
    assert icon_renderer("violin") is icon_renderer("cello")
    assert icons._load_icon.cache_info().misses == 1
    assert icon_renderer("bassoon") is None
    assert icon_renderer("bass") is None  # The project uses this ID for electric bass.
    assert icon_renderer("electric bass") is None
    assert icon_renderer("flute") is None
    assert icon_renderer("unknown") is None
    assert icon_renderer("") is None


@pytest.mark.parametrize("kind", ["bell", "drum", "piano", "keyboard", "violin"])
def test_packaged_icons_render_without_stretching_or_clipping(kind):
    renderer = icon_renderer(kind)
    image = QImage(220, 220, QImage.Format.Format_RGBA8888)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    renderer.render(painter, QRectF(10, 10, 200, 200))
    painter.end()
    pixels = [(x, y) for y in range(image.height()) for x in range(image.width())
              if image.pixelColor(x, y).alpha() > 0]
    assert pixels
    xs, ys = zip(*pixels, strict=True)
    width = max(xs) - min(xs) + 1
    height = max(ys) - min(ys) + 1
    box = renderer.viewBoxF()
    assert width / height == pytest.approx(box.width() / box.height(), abs=0.04)
    assert min(xs) >= 10 and max(xs) < 210
    assert min(ys) >= 10 and max(ys) < 210


@pytest.mark.parametrize("payload", [None, b"<svg>invalid"])
def test_missing_or_invalid_resource_returns_none_for_legacy_fallback(monkeypatch, payload):
    class Resource:
        def joinpath(self, *parts):
            return self

        def read_bytes(self):
            if payload is None:
                raise FileNotFoundError("missing packaged icon")
            return payload

    monkeypatch.setattr(icons, "files", lambda package: Resource())
    assert icon_renderer("piano") is None
