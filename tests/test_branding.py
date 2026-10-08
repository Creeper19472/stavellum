"""Logo variants follow their background and window icons follow the Qt theme."""

from types import SimpleNamespace

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QColor, QImage, QPalette
from PySide6.QtWidgets import QWidget

from stavellum.graphics import branding
from stavellum.graphics.qt import ensure_app


def test_logo_variants_and_unreadable_dark_fallback(tmp_path, monkeypatch):
    regular = QImage(str(branding.LOGO_PATH))
    dark = QImage(str(branding.LOGO_DARK_PATH))
    assert not regular.isNull() and not dark.isNull()
    assert regular.hasAlphaChannel() and dark.hasAlphaChannel()
    assert branding.logo_image() == regular
    assert branding.logo_image(dark=True) == dark
    for path in (tmp_path / "missing.png", tmp_path / "broken.png"):
        if path.name == "broken.png":
            path.write_bytes(b"invalid image")
        monkeypatch.setattr(branding, "LOGO_DARK_PATH", path)
        assert branding.logo_image(dark=True) == regular


@pytest.mark.parametrize("scheme,background,expected", [
    (Qt.ColorScheme.Dark, "white", True),
    (Qt.ColorScheme.Light, "black", False),
    (Qt.ColorScheme.Unknown, "black", True),
    (Qt.ColorScheme.Unknown, "white", False),
])
def test_theme_detection_prefers_known_scheme_and_falls_back_to_palette(
        monkeypatch, scheme, background, expected):
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(background))
    app = SimpleNamespace(styleHints=lambda: SimpleNamespace(colorScheme=lambda: scheme),
                          palette=lambda: palette)
    monkeypatch.setattr(branding, "QGuiApplication", SimpleNamespace(instance=lambda: app))
    assert branding._dark_environment() == expected


def test_window_and_application_icons_update_on_theme_and_palette_signals(monkeypatch):
    app = ensure_app()
    original = app.windowIcon()
    window = QWidget()
    dark = False
    monkeypatch.setattr(branding, "_dark_environment", lambda: dark)

    def current_pixels(icon):
        return icon.pixmap(64, 64).toImage()

    try:
        branding.bind_application_icon(window)
        assert current_pixels(window.windowIcon()) == current_pixels(branding.application_icon())
        light_image = current_pixels(window.windowIcon())
        dark = True
        app.styleHints().colorSchemeChanged.emit(Qt.ColorScheme.Dark)
        assert current_pixels(window.windowIcon()) == current_pixels(branding.application_icon())
        assert current_pixels(app.windowIcon()) != light_image
        dark = False
        app.paletteChanged.emit(app.palette())
        assert current_pixels(window.windowIcon()) == light_image
        assert current_pixels(app.windowIcon()) == light_image
    finally:
        window.deleteLater()
        app.sendPostedEvents(window, QEvent.Type.DeferredDelete)
        app.setWindowIcon(original)


def test_welcome_banner_uses_light_logo_even_in_dark_environment(monkeypatch):
    from stavellum.ui.welcome import _MusicBanner

    ensure_app()
    monkeypatch.setattr(branding, "_dark_environment", lambda: True)
    banner = _MusicBanner()
    assert banner._logo.toImage() == branding.logo_image().convertToFormat(
        QImage.Format.Format_ARGB32_Premultiplied)
    banner.deleteLater()
