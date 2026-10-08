"""Capture real desktop widgets offscreen with isolated preferences and demo media."""

from __future__ import annotations

import argparse
from pathlib import Path

from PySide6.QtCore import QSettings, QSignalBlocker, Qt
from PySide6.QtWidgets import QLabel, QLineEdit, QListWidget

from stavellum.compilation_cache import preview_key
from stavellum.demo import create_demo_document
from stavellum.gui import MainWindow
from stavellum.qt import ensure_app
from stavellum.scene import compile_scene
from stavellum.startup import StartupSplash


def capture(output: Path, *, public_paths: bool = False) -> None:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    app = ensure_app(offscreen=True)
    settings = QSettings(str(output / "preview.ini"), QSettings.Format.IniFormat)
    settings.setFallbacksEnabled(False)
    window = MainWindow(settings=settings)
    splash = StartupSplash()

    def snapshot(widget, name: str, width: int, height: int) -> None:
        widget.resize(width, height)
        widget.show()
        app.processEvents()
        if public_paths:
            # Keep real widget layout and demo media, but replace local paths
            # in screenshots intended for a public PR. Signals stay blocked
            # while display-only fixture text is changed.
            def public_text(text: str) -> str:
                return text.replace(str(output), str(Path("C:/Music/Stavellum"))).replace(
                    output.as_posix(), "C:/Music/Stavellum")

            for label in widget.findChildren(QLabel):
                label.setText(public_text(label.text()))
            for edit in widget.findChildren(QLineEdit):
                blocker = QSignalBlocker(edit)
                edit.setText(public_text(edit.text()))
                del blocker
            app.processEvents()
            for listing in widget.findChildren(QListWidget):
                blocker = QSignalBlocker(listing)
                for index in range(listing.count()):
                    item = listing.item(index)
                    item.setText(public_text(item.text()))
                    path = item.data(Qt.ItemDataRole.UserRole)
                    if isinstance(path, str):
                        item.setData(Qt.ItemDataRole.UserRole, public_text(path))
                del blocker
        if not widget.grab().save(str(output / f"{name}.png")):
            raise OSError(f"Unable to save {name}.png")
        widget.hide()

    try:
        splash.mark_ready()
        splash._timer.stop()
        snapshot(splash, "startup-cover", splash.width(), splash.height())
        snapshot(window.welcome, "welcome-empty", 1060, 720)
        document = create_demo_document(output / "demo", bars=16)
        document.metadata.title = "光与弦 · 三乐器示例"
        document.settings.render_backend = "cpu"
        path = str(output / "demo" / "demo.stproj")
        window.welcome.set_recent_projects([
            path, str(output / "夜曲 · 四手联弹.stproj"),
            str(output / "协奏曲总谱.stproj"),
        ])
        snapshot(window.welcome, "welcome-recent", 1060, 720)
        snapshot(window.welcome, "welcome-small", 900, 620)
        # The capture compiles synchronously, avoiding spawned UI workers.
        document.project.timing_confirmed = False
        window.set_document(document, path)
        document.project.timing_confirmed = True
        window.timing_confirmed.setChecked(True)
        window._pending_preview_key = preview_key(document)
        window._compiled(compile_scene(document))
        window._dirty = False
        window._update_title()
        window._update_actions()
        window.seek_to_milliseconds(4500)
        snapshot(window, "editor", 1480, 920)
        for index, name in ((1, "editor-parts"), (2, "editor-settings")):
            window.tabs.setCurrentIndex(index)
            snapshot(window, name, 1480, 920)
        window.tabs.setCurrentIndex(0)
        snapshot(window, "editor-small", 1000, 700)
        window.wizard.reset(document.project.source_path)
        snapshot(window.wizard, "wizard", 900, 760)
    finally:
        splash.close()
        splash.deleteLater()
        window._dirty = False
        window.close()
        window.deleteLater()
        app.processEvents()
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ui-preview"))
    parser.add_argument("--public-paths", action="store_true",
                        help="Use placeholder paths in screenshots for public sharing")
    args = parser.parse_args()
    capture(args.output, public_paths=args.public_paths)


if __name__ == "__main__":
    main()
