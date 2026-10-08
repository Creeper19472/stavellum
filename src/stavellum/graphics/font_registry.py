"""Register bundled font faces once after Qt application creation."""

from __future__ import annotations

from importlib.resources import files

from PySide6.QtGui import QFontDatabase, QGuiApplication

LATIN_FAMILY = "Edwin"
CJK_FAMILY = "Source Han Serif SC"
_font_ids: list[int] = []


def register_fonts() -> None:
    """Register package resources after Qt starts; never consult installed fonts."""
    if _font_ids:
        return
    if QGuiApplication.instance() is None:
        raise RuntimeError("注册乐谱字体前必须先创建 QApplication。")
    registered = []
    for filename, family in (
        ("Edwin-Roman.otf", LATIN_FAMILY),
        ("Edwin-Italic.otf", LATIN_FAMILY),
        ("SourceHanSerifSC-Regular.otf", CJK_FAMILY),
    ):
        resource = files("stavellum").joinpath("fonts", filename)
        try:
            data = resource.read_bytes()
        except (FileNotFoundError, OSError) as error:
            for previous in registered:
                QFontDatabase.removeApplicationFont(previous)
            raise RuntimeError(f"缺少随应用打包的乐谱字体：{filename}") from error
        font_id = QFontDatabase.addApplicationFontFromData(data)
        if font_id < 0 or family not in QFontDatabase.applicationFontFamilies(font_id):
            if font_id >= 0:
                QFontDatabase.removeApplicationFont(font_id)
            for previous in registered:
                QFontDatabase.removeApplicationFont(previous)
            raise RuntimeError(f"无法加载随应用打包的乐谱字体：{filename}")
        registered.append(font_id)
    _font_ids.extend(registered)
