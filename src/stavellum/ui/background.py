"""Spawned workers keep notation and encoding away from the desktop event loop."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
import traceback
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal


def _worker(operation: str, payload: Any, messages: Any, cancelled: Any) -> None:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = None
    try:
        if operation == "video":
            from stavellum.graphics.qt import prepare_render_app

            app = prepare_render_app(payload[0].settings)
        elif operation in {"compile", "parts"}:
            from stavellum.graphics.qt import ensure_app

            app = ensure_app(offscreen=True)

        def progress(fraction: float, message: str = "") -> None:
            messages.put(("progress", max(0.0, min(1.0, float(fraction))), str(message)))

        def progress_detail(snapshot) -> None:
            messages.put(("progress_detail", snapshot))

        if cancelled.is_set():
            messages.put(("cancelled",))
            return
        if operation == "compile":
            from stavellum.presentation.scene import compile_scene

            document, options = payload if isinstance(payload, tuple) else (payload, {})
            result = compile_scene(document, progress=progress, cancel=cancelled.is_set, **options)
        elif operation == "import":
            from stavellum.importers import import_project

            source, arrangement_index = payload
            progress(0.05, "正在读取音符与编曲…")
            result = import_project(source, arrangement_index=arrangement_index)
        elif operation == "load":
            from stavellum.domain.models import load_document

            result = load_document(payload)
        elif operation == "demo":
            from stavellum.demo import create_demo_document

            result = create_demo_document(output_dir=payload)
        elif operation == "video":
            from stavellum.exporting.export import export_video

            document, destination = payload
            result = export_video(document, destination, progress=progress, cancel=cancelled.is_set,
                                  progress_detail=progress_detail)
        elif operation == "parts":
            from stavellum.engraving.notation import export_parts

            document, destination = payload
            result = export_parts(document, destination, progress=progress, cancel=cancelled.is_set,
                                  progress_detail=progress_detail)
        else:
            raise ValueError(f"未知后台任务：{operation}")
        if cancelled.is_set() and operation not in {"video", "parts"}:
            messages.put(("cancelled",))
        else:
            messages.put(("result", result))
    except BaseException as exc:
        if cancelled.is_set():
            messages.put(("cancelled",))
        else:
            diagnostic_text = "\n\n".join(
                f"[{item.severity}] {item.code}: {item.message}"
                for item in getattr(exc, "diagnostics", [])
            )
            details = (diagnostic_text + "\n\n" if diagnostic_text else "") + traceback.format_exc()
            messages.put(("error", str(exc), details))
    finally:
        # Keep the Qt application alive through all SVG/PDF/font operations.
        del app


class BackgroundJob(QObject):
    """A single process with bounded UI polling and cooperative cancellation.

    Successful results are normal Python data. GUI resources stay in the GUI process.
    A cancelled job gets five seconds to close its encoder before forced termination.
    """

    progress = Signal(float, str)
    progress_detail = Signal(object)
    succeeded = Signal(object)
    failed = Signal(str, str)
    cancelled = Signal()
    finished = Signal()

    def __init__(self, operation: str, payload: Any, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.operation = operation
        self.payload = payload
        self._context = mp.get_context("spawn")
        self._messages = self._context.Queue()
        self._cancel_event = self._context.Event()
        self._process: Any = None
        self._cancel_requested_at: float | None = None
        self._done = False
        self._dead_since: float | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._poll)

    @property
    def running(self) -> bool:
        return self._process is not None and not self._done

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("后台任务只能启动一次。")
        self._process = self._context.Process(
            target=_worker,
            args=(self.operation, self.payload, self._messages, self._cancel_event),
            name=f"stavellum-{self.operation}",
        )
        self._process.start()
        self._timer.start()

    def cancel(self) -> None:
        if self.running and self._cancel_requested_at is None:
            self._cancel_event.set()
            self._cancel_requested_at = self._cancel_requested_at or time.monotonic()
            self.progress.emit(0.0, "正在取消，等待后台清理…")

    def shutdown(self) -> None:
        """Used only when the window is closing; never leave an orphaned worker."""
        self._timer.stop()
        if self._process is not None and self._process.pid is not None:
            self._cancel_event.set()
            self._process.join(timeout=0.3)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1)
        self._close_queue()
        self._done = True

    def _poll(self) -> None:
        if self._done:
            return
        while True:
            try:
                message = self._messages.get_nowait()
            except queue.Empty:
                break
            kind = message[0]
            if kind == "progress":
                self.progress.emit(message[1], message[2])
            elif kind == "progress_detail":
                self.progress_detail.emit(message[1])
            elif kind == "result":
                self._done = True
                self.succeeded.emit(message[1])
                self._finish()
                return
            elif kind == "error":
                self._done = True
                self.failed.emit(message[1], message[2])
                self._finish()
                return
            elif kind == "cancelled":
                self._done = True
                self.cancelled.emit()
                self._finish()
                return
        if self._cancel_requested_at and time.monotonic() - self._cancel_requested_at > 5:
            self._process.terminate()
            self._process.join(timeout=0.5)
            self._done = True
            self.cancelled.emit()
            self._finish()
        elif self._process is not None and not self._process.is_alive():
            # The queue feeder can deliver a final message just after process exit.
            self._dead_since = self._dead_since or time.monotonic()
            if time.monotonic() - self._dead_since > 0.3:
                self._done = True
                self.failed.emit(
                    f"后台进程意外退出（退出码 {self._process.exitcode}）。", "未收到任务结果。"
                )
                self._finish()

    def _finish(self) -> None:
        self._done = True
        self._timer.stop()
        if self._process is not None:
            self._process.join(timeout=0.1)
        self._close_queue()
        self.finished.emit()

    def _close_queue(self) -> None:
        try:
            self._messages.close()
            self._messages.cancel_join_thread()
        except (ValueError, OSError):
            pass


def default_demo_directory() -> str:
    """Store generated media in a writable user directory, not the installed package."""
    from PySide6.QtCore import QStandardPaths

    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
    return str(Path(base) / "demo")
