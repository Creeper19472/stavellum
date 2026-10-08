"""A persistent, modeless view of one export and its measured processing progress."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from stavellum.domain.progress import ExportEstimator, ExportProgress

_VIDEO_PHASES = (
    ("check", "检查原曲音频与设置"),
    ("compile", "刻谱和计算动画"),
    ("prepare", "初始化渲染与编码器"),
    ("frames", "逐帧处理"),
    ("finalize", "等待编码完成及 MP4 封装"),
    ("save", "保存结果与报告"),
)
_PART_PHASES = (
    ("compile", "准备谱面"),
    ("xml", "生成 MusicXML"),
    ("pdf", "绘制 PDF"),
    ("save", "保存文件"),
)


def _duration_label(seconds: float) -> str:
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@dataclass
class _Step:
    item: QListWidgetItem
    name: str
    status: str = "pending"
    started: float | None = None
    stopped: float | None = None

    def refresh(self, elapsed: float) -> None:
        symbol = {"pending": "○", "active": "●", "complete": "✓", "failed": "✕",
                  "cancelled": "—", "retry": "↻", "skipped": "—"}[self.status]
        suffix = ""
        if self.started is not None:
            stop = elapsed if self.stopped is None else self.stopped
            suffix = f"    {_duration_label(round(stop - self.started))}"
        if self.status == "retry":
            suffix += "（回退重试）"
        elif self.status == "skipped":
            suffix += "（未执行）"
        self.item.setText(f"{symbol} {self.name}{suffix}")


class ExportProgressDialog(QDialog):
    """Closing or pressing Escape hides the view; cancellation is always explicit."""

    cancel_requested = Signal()

    def __init__(self, operation: str, destination: str,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if operation not in {"video", "parts"}:
            raise ValueError(f"未知导出任务：{operation}")
        self.operation = operation
        self.destination = str(Path(destination).resolve())
        self._output_directory = (Path(self.destination).parent if operation == "video"
                                  else Path(self.destination))
        self.running = True
        self._cancel_requested = False
        self._started = time.monotonic()
        self._elapsed_anchor = self._started
        self._reported_elapsed = 0.0
        self._finished_elapsed: float | None = None
        self._finished_step_elapsed: float | None = None
        self._detail: ExportProgress | None = None
        self._estimator = ExportEstimator(operation)
        self._attempt = 1
        self._part_index: int | None = None
        self._metadata: dict[str, str] = {}
        self._steps: list[_Step] = []
        self._phase_steps: dict[str, _Step] = {}
        self._active_step: _Step | None = None
        self.setWindowTitle("MP4 导出进度" if operation == "video" else "分谱导出进度")
        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.resize(660, 640)
        self._build_ui()
        self._add_steps(_VIDEO_PHASES if operation == "video" else _PART_PHASES)
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._refresh_metrics)
        self._timer.start()
        self._refresh_metrics()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.status_label = QLabel("正在启动后台导出…")
        self.status_label.setTextFormat(Qt.TextFormat.PlainText)
        self.status_label.setWordWrap(True)
        self.counter_label = QLabel("等待处理数据")
        self.counter_label.setTextFormat(Qt.TextFormat.PlainText)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        layout.addWidget(self.status_label)
        layout.addWidget(self.counter_label)
        layout.addWidget(self.progress_bar)
        metrics = QFormLayout()
        self.elapsed_label = QLabel()
        self.speed_label = QLabel("估算中")
        self.remaining_label = QLabel("估算中")
        self.completion_label = QLabel("估算中")
        for name, widget in (("已用时间", self.elapsed_label), ("处理速度", self.speed_label),
                             ("预计剩余时间", self.remaining_label),
                             ("预计完成时刻", self.completion_label)):
            metrics.addRow(name, widget)
        layout.addLayout(metrics)
        self.estimate_note = QLabel("按当前处理速度估算，文件收尾耗时另计")
        self.estimate_note.setWordWrap(True)
        layout.addWidget(self.estimate_note)
        layout.addWidget(QLabel("处理步骤与耗时"))
        self.steps_list = QListWidget()
        layout.addWidget(self.steps_list, 1)
        self.path_label = QLabel(f"输出位置：{self.destination}")
        self.path_label.setTextFormat(Qt.TextFormat.PlainText)
        self.path_label.setWordWrap(True)
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.metadata_label = QLabel("渲染后端与编码器将在初始化后显示。" if self.operation == "video"
                                     else "MusicXML 与 PDF 分谱")
        self.metadata_label.setTextFormat(Qt.TextFormat.PlainText)
        self.metadata_label.setWordWrap(True)
        self.metadata_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.path_label)
        layout.addWidget(self.metadata_label)
        self.error_toggle = QToolButton()
        self.error_toggle.setText("错误详情")
        self.error_toggle.setCheckable(True)
        self.error_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.error_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.error_toggle.toggled.connect(self._toggle_error)
        self.error_toggle.hide()
        self.error_details = QPlainTextEdit()
        self.error_details.setReadOnly(True)
        self.error_details.setMaximumHeight(150)
        self.error_details.hide()
        layout.addWidget(self.error_toggle)
        layout.addWidget(self.error_details)
        buttons = QHBoxLayout()
        self.open_folder_button = QPushButton("打开输出文件夹")
        self.open_folder_button.setEnabled(False)
        self.open_folder_button.clicked.connect(self._open_output_folder)
        self.cancel_button = QPushButton("取消导出")
        self.cancel_button.clicked.connect(self.request_cancel)
        self.hide_button = QPushButton("隐藏")
        self.hide_button.clicked.connect(self.hide)
        buttons.addWidget(self.open_folder_button)
        buttons.addStretch()
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.hide_button)
        layout.addLayout(buttons)

    def _add_steps(self, phases: tuple[tuple[str, str], ...], prefix: str = "") -> None:
        for phase, name in phases:
            item = QListWidgetItem(self.steps_list)
            step = _Step(item, prefix + name)
            self._steps.append(step)
            self._phase_steps[phase] = step
            step.refresh(self._elapsed())

    def _elapsed(self) -> float:
        if self._finished_elapsed is not None:
            return self._finished_elapsed
        now = time.monotonic()
        return max(now - self._started, self._reported_elapsed + now - self._elapsed_anchor)

    def _step_elapsed(self) -> float:
        if self._finished_step_elapsed is not None:
            return self._finished_step_elapsed
        return self._reported_elapsed + max(0.0, time.monotonic() - self._elapsed_anchor)

    def update_progress(self, detail: ExportProgress) -> None:
        if not self.running or self._cancel_requested:
            return
        self._detail = detail
        now = time.monotonic()
        self._reported_elapsed = max(self._reported_elapsed, detail.elapsed_seconds)
        self._elapsed_anchor = now
        self._estimator.update(detail, now)
        elapsed = detail.elapsed_seconds
        if detail.attempt != self._attempt:
            if self._active_step:
                self._active_step.status = "retry"
                self._active_step.stopped = elapsed
                self._active_step = None
            for old_step in self._phase_steps.values():
                if old_step.status == "pending":
                    old_step.status = "skipped"
            self._attempt = detail.attempt
            self._add_steps(_VIDEO_PHASES[3:], f"第 {detail.attempt} 次尝试 · ")
        if (self.operation == "parts" and detail.part_name
                and (self._part_index is None or (detail.phase == "xml"
                                                  and detail.completed != self._part_index))):
            if self._part_index is not None:
                self._add_steps(_PART_PHASES[1:], f"{detail.part_name} · ")
            else:
                for phase, name in _PART_PHASES[1:]:
                    self._phase_steps[phase].name = f"{detail.part_name} · {name}"
            self._part_index = detail.completed
        step = self._phase_steps.get(detail.phase)
        if step is not self._active_step:
            if self._active_step:
                self._active_step.status = "complete"
                self._active_step.stopped = elapsed
            self._active_step = step
            if step:
                step.status = "active"
                step.started = elapsed
                step.stopped = None
                self.steps_list.scrollToItem(step.item)
        self.status_label.setText(detail.message or (step.name if step else "正在确认导出结果…"))
        self._update_counter(detail)
        self._update_bar(detail)
        for field, label in (("render_backend", "渲染后端"), ("video_encoder", "编码器"),
                             ("fallback_reason", "回退原因")):
            if value := getattr(detail, field):
                self._metadata[label] = value
        if self.operation == "video":
            self._metadata["尝试"] = str(detail.attempt)
        if self._metadata:
            self.metadata_label.setText("\n".join(f"{name}：{value}" for name, value in self._metadata.items()))
        self._refresh_metrics()

    def _update_counter(self, detail: ExportProgress) -> None:
        unit = "帧" if self.operation == "video" else "份分谱"
        text = (f"已处理 {detail.completed:,} / {detail.total:,} {unit}" if detail.total
                else "等待处理数据")
        if detail.part_name:
            text += f"    当前乐器：{detail.part_name}"
        if detail.pages:
            text += f"    PDF：{detail.page} / {detail.pages} 页"
        self.counter_label.setText(text)

    def _update_bar(self, detail: ExportProgress) -> None:
        if self._cancel_requested:
            return
        if not detail.total:
            self.progress_bar.setRange(0, 0)
            return
        fraction = max(0.0, min(1.0, detail.completed / detail.total))
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(min(999, int(fraction * 1000)))

    def _refresh_metrics(self) -> None:
        elapsed = self._elapsed()
        self.elapsed_label.setText(_duration_label(elapsed))
        for step in self._steps:
            step.refresh(self._step_elapsed())
        if not self.running or self._cancel_requested:
            self.speed_label.setText("—")
            self.remaining_label.setText("正在清理" if self.running else "—")
            self.completion_label.setText("—")
            return
        detail = self._detail
        finishing = (detail is not None and (detail.phase in {"finalize", "done"}
                     or self.operation == "video" and detail.phase == "save"))
        estimate = self._estimator.estimate(time.monotonic())
        speed = estimate.speed
        remaining = estimate.remaining_seconds
        if self.operation == "video" and (detail is None or detail.phase != "frames"):
            speed = remaining = None
        if speed is None:
            self.speed_label.setText("正在收尾" if finishing else "估算中")
        elif self.operation == "video":
            text = f"{speed:.1f} 帧/秒"
            if detail and detail.fps > 0:
                text += f"（{speed / detail.fps:.2f}×）"
            self.speed_label.setText(text)
        else:
            self.speed_label.setText(f"{speed * 60:.2f} 份/分钟（平均每份 {1 / speed:.1f} 秒）")
        if remaining is None:
            text = "正在收尾" if finishing else "估算中"
            self.remaining_label.setText(text)
            self.completion_label.setText(text)
        else:
            self.remaining_label.setText(_duration_label(remaining))
            predicted = datetime.now().astimezone() + timedelta(seconds=remaining)
            self.completion_label.setText(predicted.strftime("%Y-%m-%d %H:%M:%S"))

    def request_cancel(self) -> None:
        if not self.running or self._cancel_requested:
            return
        self._cancel_requested = True
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("正在取消…")
        self.status_label.setText("正在取消，等待后台清理…")
        if self.progress_bar.maximum() == 0:
            self.progress_bar.setRange(0, 1000)
            self.progress_bar.setValue(0)
        self._refresh_metrics()
        self.cancel_requested.emit()

    def _finish(self, status: str, message: str) -> None:
        if not self.running:
            return
        self._finished_elapsed = self._elapsed()
        self._finished_step_elapsed = self._step_elapsed()
        self.running = False
        self._timer.stop()
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("取消导出")
        if self._active_step:
            self._active_step.status = status
            self._active_step.stopped = self._finished_step_elapsed
        self.status_label.setText(message)
        if self.progress_bar.maximum() == 0:
            self.progress_bar.setRange(0, 1000)
            self.progress_bar.setValue(0)
        self._refresh_metrics()

    def finish_success(self, result: Any) -> None:
        if not self.running:
            return
        self._finish("complete", "导出成功")
        self.progress_bar.setValue(1000)
        self.open_folder_button.setEnabled(True)
        if isinstance(result, (str, Path)):
            self._output_directory = Path(result).resolve().parent
        elif isinstance(result, list) and result:
            self._output_directory = Path(result[0]).resolve().parent

    def finish_failure(self, message: str, details: str = "") -> None:
        if not self.running:
            return
        self._finish("failed", f"导出失败：{message}")
        self.error_details.setPlainText(details or message)
        self.error_toggle.show()

    def finish_cancelled(self) -> None:
        self._finish("cancelled", "导出已取消，后台清理完成")

    def _toggle_error(self, checked: bool) -> None:
        self.error_details.setVisible(checked)
        self.error_toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

    def _open_output_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._output_directory)))

    def reject(self) -> None:
        self.hide()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.hide()
        event.ignore()
