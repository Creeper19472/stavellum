"""Export view lifecycle and measured progress are independent from task cancellation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from stavellum.domain.progress import ExportProgress
from stavellum.ui import progress_dialog as dialog_module
from stavellum.ui.progress_dialog import ExportProgressDialog

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def clock(monkeypatch):
    value = [100.0]
    monkeypatch.setattr(dialog_module.time, "monotonic", lambda: value[0])
    return value


@pytest.fixture
def dialog(app, tmp_path, clock):
    widget = ExportProgressDialog("video", str(tmp_path / "output.mp4"))
    yield widget
    widget.finish_cancelled()
    widget.hide()
    widget.deleteLater()
    app.processEvents()


def test_close_escape_and_hide_allow_reopen_without_cancelling(app, dialog):
    requested = []
    dialog.cancel_requested.connect(lambda: requested.append(True))
    assert not dialog.isModal()
    dialog.show()
    app.processEvents()
    dialog.close()
    assert not dialog.isVisible()
    assert dialog.running
    dialog.show()
    event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(dialog, event)
    assert not dialog.isVisible()
    assert dialog.running
    dialog.show()
    dialog.hide_button.click()
    assert not dialog.isVisible()
    assert requested == []
    dialog.show()
    assert dialog.isVisible()


def test_cancel_requested_once_and_waits_for_backend_terminal_state(dialog):
    requested = []
    dialog.cancel_requested.connect(lambda: requested.append(True))
    dialog.update_progress(ExportProgress("frames", "正在逐帧处理", 2,
                                          completed=25, total=100, fps=25))
    before = dialog.progress_bar.value()
    dialog.cancel_button.click()
    dialog.request_cancel()
    assert requested == [True]
    assert dialog.running
    assert not dialog.cancel_button.isEnabled()
    assert "清理" in dialog.status_label.text()
    assert dialog.speed_label.text() == "—"
    assert dialog.remaining_label.text() == "正在清理"
    dialog.update_progress(ExportProgress("frames", "后台仍在处理", 3,
                                          completed=30, total=100, fps=25))
    assert "清理" in dialog.status_label.text()
    assert dialog.progress_bar.value() == before
    dialog.finish_cancelled()
    assert not dialog.running
    assert "清理完成" in dialog.status_label.text()
    assert dialog.progress_bar.value() < 1000
    assert not dialog._timer.isActive()


def test_video_speed_eta_finalize_and_success_hidden(dialog, clock):
    dialog.update_progress(ExportProgress("frames", "逐帧处理", 0,
                                          completed=0, total=100, fps=25))
    assert dialog.remaining_label.text() == "估算中"
    clock[0] += 2
    dialog.update_progress(ExportProgress("frames", "逐帧处理", 2,
                                          completed=50, total=100, fps=25))
    assert dialog.speed_label.text() == "25.0 帧/秒（1.00×）"
    assert dialog.remaining_label.text() == "00:00:02"
    assert len(dialog.completion_label.text()) == 19
    clock[0] += 2
    dialog.update_progress(ExportProgress("finalize", "等待编码与封装", 4,
                                          completed=100, total=100, fps=25))
    assert dialog.progress_bar.value() < 1000
    assert dialog.remaining_label.text() == "正在收尾"
    assert dialog.speed_label.text() == "正在收尾"
    dialog.update_progress(ExportProgress("done", "完成保存", 4,
                                          completed=100, total=100, fps=25))
    assert dialog.running
    assert dialog.progress_bar.value() == 999
    dialog.hide()
    dialog.finish_success(dialog.destination)
    assert not dialog.isVisible()
    assert not dialog.running
    assert dialog.progress_bar.value() == 1000
    assert dialog.open_folder_button.isEnabled()
    assert dialog.status_label.text() == "导出成功"
    elapsed = dialog.elapsed_label.text()
    clock[0] += 30
    dialog._refresh_metrics()
    assert dialog.elapsed_label.text() == elapsed
    dialog.show()
    assert dialog.status_label.text() == "导出成功"


def test_batched_stage_messages_keep_worker_durations(dialog, clock):
    clock[0] += 20  # The GUI only gets to consume all queued snapshots afterwards.
    for phase, elapsed in [("check", 0), ("compile", 1), ("prepare", 3), ("frames", 4),
                           ("finalize", 14), ("save", 16), ("done", 17)]:
        dialog.update_progress(ExportProgress(phase, "", elapsed, total=100,
                                              completed=100 if elapsed >= 14 else 0))
    assert "00:00:02" in dialog.steps_list.item(1).text()
    assert "00:00:10" in dialog.steps_list.item(3).text()
    assert "00:00:02" in dialog.steps_list.item(4).text()
    assert dialog.elapsed_label.text() == "00:00:20"


def test_retry_skips_unexecuted_steps_and_save_remains_finalizing(dialog, clock):
    dialog.update_progress(ExportProgress("frames", "", 0, total=100))
    clock[0] += 2
    dialog.update_progress(ExportProgress("frames", "CPU 重导", 2, attempt=2, total=100))
    rows = [dialog.steps_list.item(index).text() for index in range(dialog.steps_list.count())]
    assert any("未执行" in row for row in rows)
    assert not any("第 2 次尝试 · 初始化" in row for row in rows)
    dialog.update_progress(ExportProgress("save", "保存", 3, attempt=2, total=100,
                                          completed=100))
    assert dialog.remaining_label.text() == "正在收尾"


def test_stall_and_retry_reset_estimate_preserve_total_elapsed(dialog, clock):
    dialog.update_progress(ExportProgress("frames", "GPU 处理中", 0,
                                          completed=0, total=100, fps=25,
                                          render_backend="gpu", video_encoder="h264_nvenc"))
    clock[0] += 2
    dialog.update_progress(ExportProgress("frames", "GPU 处理中", 2,
                                          completed=50, total=100, fps=25))
    assert "25.0" in dialog.speed_label.text()
    clock[0] += 5
    dialog._refresh_metrics()
    assert dialog.remaining_label.text() == "估算中"
    dialog.update_progress(ExportProgress("prepare", "CPU 重导", 7, attempt=2,
                                          completed=0, total=100, fps=25,
                                          video_encoder="libx264", fallback_reason="NVENC unavailable"))
    assert dialog.elapsed_label.text() == "00:00:07"
    assert dialog.remaining_label.text() == "估算中"
    assert "libx264" in dialog.metadata_label.text()
    assert "NVENC unavailable" in dialog.metadata_label.text()
    assert "尝试：2" in dialog.metadata_label.text()
    assert any("回退重试" in dialog.steps_list.item(index).text()
               for index in range(dialog.steps_list.count()))
    assert any("第 2 次尝试" in dialog.steps_list.item(index).text()
               for index in range(dialog.steps_list.count()))


def test_parts_show_pages_average_and_repeated_stages(app, tmp_path, clock):
    widget = ExportProgressDialog("parts", str(tmp_path))
    try:
        widget.update_progress(ExportProgress("compile", "准备", 0, total=2, unit="parts"))
        clock[0] += 1
        widget.update_progress(ExportProgress("xml", "生成", 1, total=2, unit="parts",
                                              part_name="小提琴"))
        clock[0] += 2
        widget.update_progress(ExportProgress("pdf", "绘制", 3, total=2, unit="parts",
                                              part_name="小提琴", page=2, pages=5))
        assert "小提琴" in widget.counter_label.text()
        assert "2 / 5 页" in widget.counter_label.text()
        assert widget.remaining_label.text() == "估算中"
        clock[0] += 2
        widget.update_progress(ExportProgress("save", "保存", 5, completed=1, total=2,
                                              unit="parts", part_name="小提琴"))
        assert widget.speed_label.text() == "15.00 份/分钟（平均每份 4.0 秒）"
        assert widget.remaining_label.text() == "00:00:04"
        clock[0] += 1
        widget.update_progress(ExportProgress("xml", "生成", 6, completed=1, total=2,
                                              unit="parts", part_name="大提琴"))
        rows = [widget.steps_list.item(index).text() for index in range(widget.steps_list.count())]
        assert any("✓ 小提琴 · 绘制 PDF" in row for row in rows)
        assert any("大提琴 · 生成 MusicXML" in row for row in rows)
        assert len(rows) == 7
        widget.update_progress(ExportProgress("done", "保存完成", 6, completed=2, total=2,
                                              unit="parts"))
        assert widget.progress_bar.value() == 999
        widget.finish_success([str(tmp_path / "one.pdf"), str(tmp_path / "one.musicxml")])
        assert widget.progress_bar.value() == 1000
    finally:
        widget.finish_cancelled()
        widget.deleteLater()
        app.processEvents()


def test_failure_retains_details_without_showing_hidden_window(dialog):
    dialog.hide()
    dialog.finish_failure("FFmpeg 不可用", "traceback and encoder stderr")
    assert not dialog.isVisible()
    assert not dialog.running
    assert dialog.progress_bar.value() < 1000
    assert not dialog.open_folder_button.isEnabled()
    dialog.show()
    assert dialog.error_toggle.isVisible()
    assert not dialog.error_details.isVisible()
    dialog.error_toggle.click()
    assert dialog.error_details.isVisible()
    assert dialog.error_details.toPlainText() == "traceback and encoder stderr"
    dialog.error_toggle.click()
    assert not dialog.error_details.isVisible()
    dialog.finish_success(dialog.destination)
    assert "导出失败" in dialog.status_label.text()


def test_open_folder_uses_result_directory(dialog, tmp_path, monkeypatch):
    urls = []
    monkeypatch.setattr(dialog_module.QDesktopServices, "openUrl", urls.append)
    path = tmp_path / "actual" / "final.mp4"
    dialog.finish_success(str(path))
    dialog.open_folder_button.click()
    assert Path(urls[0].toLocalFile()) == path.parent


def test_duplicate_part_names_have_separate_stage_history(app, tmp_path, clock):
    widget = ExportProgressDialog("parts", str(tmp_path))
    try:
        widget.update_progress(ExportProgress("xml", "生成", 0, total=2, unit="parts",
                                              part_name="小提琴"))
        clock[0] += 2
        widget.update_progress(ExportProgress("pdf", "绘制", 2, total=2, unit="parts",
                                              part_name="小提琴"))
        clock[0] += 1
        widget.update_progress(ExportProgress("save", "保存", 3, completed=1, total=2,
                                              unit="parts", part_name="小提琴"))
        clock[0] += 1
        widget.update_progress(ExportProgress("xml", "生成", 4, completed=1, total=2,
                                              unit="parts", part_name="小提琴"))
        assert widget.steps_list.count() == 7
        assert widget._phase_steps["xml"].status == "active"
        assert widget._steps[1].status == "complete"
    finally:
        widget.finish_cancelled()
        widget.deleteLater()
        app.processEvents()
