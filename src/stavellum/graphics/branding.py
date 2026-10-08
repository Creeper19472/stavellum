"""Shared, packaged visual identity for the desktop application."""

from importlib.resources import files
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Slot
from PySide6.QtGui import QGuiApplication, QIcon, QImage, QPalette, QPixmap

LOGO_PATH = Path(files("stavellum")) / "assets/logo.png"
LOGO_DARK_PATH = LOGO_PATH.with_name("logo-dark.png")
APPLICATION_DESCRIPTION = "基于受支持格式的自动乐谱化与演示实用程序"


def logo_image(*, dark: bool = False) -> QImage:
    """Choose by the actual background, falling back if the dark asset cannot load."""
    if dark:
        image = QImage(str(LOGO_DARK_PATH))
        if not image.isNull():
            return image
    return QImage(str(LOGO_PATH))


def _dark_environment() -> bool:
    app = QGuiApplication.instance()
    if app is None:
        return False
    scheme = app.styleHints().colorScheme()
    if scheme != Qt.ColorScheme.Unknown:
        return scheme == Qt.ColorScheme.Dark
    return app.palette().color(QPalette.ColorRole.Window).lightnessF() < 0.5


def application_icon() -> QIcon:
    return QIcon(QPixmap.fromImage(logo_image(dark=_dark_environment())))


def logo_pixmap() -> QPixmap:
    return QPixmap.fromImage(logo_image())


class _ThemeIcon(QObject):
    """Window-owned signal receiver; Qt disconnects it when the window is destroyed."""

    def __init__(self, window):
        super().__init__(window)
        app = QGuiApplication.instance()
        app.styleHints().colorSchemeChanged.connect(self.refresh)
        app.paletteChanged.connect(self.refresh)
        self.refresh()

    @Slot()
    def refresh(self):
        icon = application_icon()
        QGuiApplication.instance().setWindowIcon(icon)
        self.parent().setWindowIcon(icon)


def bind_application_icon(window) -> None:
    _ThemeIcon(window)
