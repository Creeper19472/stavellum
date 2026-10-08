"""Qt lifetime helper shared by desktop and headless worker processes."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_application = None
_fonts_loaded = False


def prepare_render_app(settings):
    """Select a GPU-capable platform before the process creates its Qt application.

    Use Windows' native QPA integration for the Vulkan instance and device.
    QRhi renders offscreen without displaying a window. Existing applications
    keep their platform; auto rendering can fall back to CPU.
    """
    from PySide6.QtWidgets import QApplication

    gpu_requested = getattr(settings, "render_backend", "auto") != "cpu"
    if QApplication.instance() is None and gpu_requested and sys.platform == "win32":
        os.environ["QT_QPA_PLATFORM"] = "windows"
    return ensure_app(offscreen=not gpu_requested)


def ensure_app(offscreen: bool = True):
    global _application, _fonts_loaded
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QApplication

    existing = QApplication.instance()
    if existing is None:
        if offscreen:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        _application = QApplication([])
    else:
        _application = existing
    # The Windows offscreen plugin can expose an empty system font database.
    # Explicit registration keeps CJK titles identical in preview, PDF and video.
    if not _fonts_loaded:
        fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        for name in ("msyh.ttc", "msyhbd.ttc", "simsun.ttc", "segoeui.ttf", "times.ttf"):
            path = fonts / name
            if path.is_file():
                QFontDatabase.addApplicationFont(str(path))
        if "Microsoft YaHei" in QFontDatabase.families():
            _application.setFont(QFont("Microsoft YaHei", 10))
        from .font_registry import register_fonts

        register_fonts()
        _fonts_loaded = True
    return _application
