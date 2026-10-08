"""Three-step, non-destructive configuration of a new imported project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent, QKeyEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from stavellum.domain.models import PartMapping
from stavellum.exporting.audio import AUDIO_FILE_FILTER
from stavellum.graphics.branding import bind_application_icon

SOURCE_FILE_FILTER = "音乐来源 (*.flp *.mid *.midi);;FL Studio 工程 (*.flp);;MIDI 文件 (*.mid *.midi)"
SOURCE_SUFFIXES = frozenset({".flp", ".mid", ".midi"})

# Keep the persisted mapping field names at this UI boundary; no project format change is needed.
PROCESSING_CHOICES = (
    (
        "auto_simplify_accidentals",
        "简化音符记法",
        "选择音高相同的等音拼写，减少不必要的临时升降号；保留调号与必要的变音记号，不删除音符。",
    ),
    (
        "auto_ottava",
        "自动八度移位",
        "在连续高低音区使用 8va、8vb、15ma 或 15mb，减少加线；仅在不影响同谱表其他声部时应用，实际音高不变。",
    ),
    (
        "auto_staccato",
        "跳音识别（保守）",
        "仅在演奏时间与上下文足够明确时推断跳音标记；保留原始音符与播放时间。",
    ),
    (
        "auto_grace",
        "前倚音识别（保守）",
        "仅在演奏时间与上下文足够明确时推断前倚音记法；保留原始音符与播放时间。",
    ),
    (
        "auto_dynamics",
        "力度渐变识别",
        "将明确的 FLP 音量自动化转换为渐强／渐弱发夹线；仅适用于 FLP，来源冲突或路由不明确时保留诊断。",
    ),
)


@dataclass(frozen=True)
class NewProjectOptions:
    """Import input captured when the user completes the wizard."""

    source_path: str
    audio_path: str
    processing: dict[str, bool]


def _label(text: str, *, name: str = "") -> QLabel:
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    if name:
        label.setObjectName(name)
    return label


def _existing_file(text: str) -> Path | None:
    try:
        path = Path(text.strip()).expanduser()
        return path.resolve() if text.strip() and path.is_file() else None
    except (OSError, RuntimeError, ValueError):
        return None


class ProjectWizard(QWidget):
    """Collect choices without modifying the current editor document."""

    create_requested = Signal(object)
    cancel_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Tool)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setWindowTitle("新工程向导 · Stavellum")
        bind_application_icon(self)
        self.resize(900, 760)
        self.setObjectName("ProjectWizard")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._busy = False
        self._cancelling = False
        self.setStyleSheet("""
            QWidget#ProjectWizard, QWidget#ProjectWizard QWidget {
                background: #17212b; color: #edf1f4;
            }
            QWidget#ProjectWizard QLabel, QWidget#ProjectWizard QCheckBox {
                color: #edf1f4; background: transparent;
            }
            QWidget#ProjectWizard QLabel#WizardSubtitle,
            QWidget#ProjectWizard QLabel#WizardDescription { color: #aebcc7; }
            QWidget#ProjectWizard QFrame#WizardHeader { background: #20303d; border-radius: 8px; }
            QWidget#ProjectWizard QLabel#WizardTitle { color: #f3dcc0; font-size: 22pt; font-weight: 600; }
            QWidget#ProjectWizard QLabel#StepTitle { color: #f3dcc0; font-size: 16pt; font-weight: 600; }
            QWidget#ProjectWizard QWidget#WizardStep { background: #17212b; }
            QWidget#ProjectWizard QScrollArea { border: none; background: #17212b; }
            QWidget#ProjectWizard QLineEdit,
            QWidget#ProjectWizard QPlainTextEdit {
                background: #20303d; color: #edf1f4; border: 1px solid #435360;
                border-radius: 5px; padding: 8px; selection-background-color: #906f49;
            }
            QWidget#ProjectWizard QLineEdit:focus { border-color: #d4ab77; }
            QWidget#ProjectWizard QPushButton,
            QWidget#ProjectWizard QToolButton {
                background: #283947; color: #edf1f4; border: 1px solid #435360;
                border-radius: 5px; padding: 8px 16px;
            }
            QWidget#ProjectWizard QPushButton:hover,
            QWidget#ProjectWizard QToolButton:hover { background: #354b5d; border-color: #d4ab77; }
            QWidget#ProjectWizard QPushButton:disabled { color: #798791; border-color: #344451; }
            QWidget#ProjectWizard QPushButton#WizardPrimary {
                background: #f3dcc0; color: #17212b; border-color: #f3dcc0; font-weight: 600;
            }
            QWidget#ProjectWizard QPushButton#WizardPrimary:hover { background: #fbe9d4; }
            QWidget#ProjectWizard QPushButton#WizardPrimary:disabled { background: #52606a; color: #a6afb5; }
            QWidget#ProjectWizard QLabel#WizardError { color: #ffc2b2; }
            QWidget#ProjectWizard QLabel#WizardSummary {
                background: #20303d; border: 1px solid #435360; border-radius: 7px; padding: 14px;
            }
            QWidget#ProjectWizard QCheckBox { spacing: 9px; font-weight: 600; }
            QWidget#ProjectWizard QCheckBox::indicator { width: 18px; height: 18px; }
        """)
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)

        header = QFrame()
        header.setObjectName("WizardHeader")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(24, 20, 24, 20)
        header_layout.addWidget(_label("新工程向导", name="WizardTitle"))
        header_layout.addWidget(_label("从音乐来源到五线谱，先为这份工程选好起点。", name="WizardSubtitle"))
        root.addWidget(header)

        progress = QHBoxLayout()
        self.step_labels: list[QLabel] = []
        for title in ("1  选择来源", "2  原曲音频", "3  自动处理"):
            label = _label(title)
            self.step_labels.append(label)
            progress.addWidget(label, 1)
        root.addLayout(progress)

        self.steps = QStackedWidget()
        root.addWidget(self.steps, 1)
        self._step_contents: list[QWidget] = []
        self._build_source_step()
        self._build_audio_step()
        self._build_processing_step()

        self.error_label = _label("", name="WizardError")
        self.error_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.error_label)
        self.error_details_button = QToolButton()
        self.error_details_button.setText("显示错误详情")
        self.error_details_button.setCheckable(True)
        self.error_details_button.toggled.connect(self._toggle_error_details)
        root.addWidget(self.error_details_button, alignment=Qt.AlignmentFlag.AlignLeft)
        self.error_details = QPlainTextEdit()
        self.error_details.setReadOnly(True)
        self.error_details.setMaximumHeight(150)
        root.addWidget(self.error_details)

        self.task_status = QWidget()
        task_layout = QHBoxLayout(self.task_status)
        self.job_label = QLabel("就绪")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        task_layout.addWidget(self.job_label, 2)
        task_layout.addWidget(self.progress, 1)
        root.addWidget(self.task_status)
        self.task_status.hide()

        navigation = QHBoxLayout()
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        navigation.addWidget(self.cancel_button)
        navigation.addStretch(1)
        self.back_button = QPushButton("上一步")
        self.back_button.clicked.connect(self._back)
        navigation.addWidget(self.back_button)
        self.skip_button = QPushButton("跳过音频")
        self.skip_button.clicked.connect(self._skip_audio)
        navigation.addWidget(self.skip_button)
        self.next_button = QPushButton("下一步")
        self.next_button.setObjectName("WizardPrimary")
        self.next_button.clicked.connect(self._next)
        navigation.addWidget(self.next_button)
        self.create_button = QPushButton("创建工程")
        self.create_button.setObjectName("WizardPrimary")
        self.create_button.clicked.connect(self._create)
        navigation.addWidget(self.create_button)
        root.addLayout(navigation)

        self.source_edit.textChanged.connect(self._choices_changed)
        self.audio_edit.textChanged.connect(self._choices_changed)
        self.steps.currentChanged.connect(self._update_navigation)
        for control in self.processing_checks.values():
            control.toggled.connect(self._update_summary)
        self.reset()

    def _new_step(self, title: str, description: str) -> QVBoxLayout:
        content = QWidget()
        content.setObjectName("WizardStep")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 20)
        layout.setSpacing(12)
        layout.addWidget(_label(title, name="StepTitle"))
        layout.addWidget(_label(description, name="WizardDescription"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        self.steps.addWidget(scroll)
        self._step_contents.append(content)
        return layout

    def _build_source_step(self) -> None:
        layout = self._new_step(
            "选择 FLP 或 MIDI 文件",
            "支持 FL Studio 工程（.flp）及 MIDI（.mid、.midi）。默认导入第一个 Arrangement；编曲、速度与拍号可在编辑器中校正。",
        )
        layout.addSpacing(12)
        layout.addWidget(_label("音乐来源"))
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("选择文件，或输入 FLP / MIDI 的完整路径")
        self.source_edit.setAccessibleName("音乐来源路径")
        layout.addWidget(self.source_edit)
        self.source_browse_button = QPushButton("选择 FLP / MIDI…")
        self.source_browse_button.clicked.connect(self._browse_source)
        layout.addWidget(self.source_browse_button, alignment=Qt.AlignmentFlag.AlignLeft)
        self.source_validation_label = _label("", name="WizardDescription")
        layout.addWidget(self.source_validation_label)
        layout.addStretch(1)

    def _build_audio_step(self) -> None:
        layout = self._new_step(
            "添加原曲音频（可选）",
            "绑定原曲音频后，可在编辑器中跟随音频试听与预览。也可以跳过，稍后在编辑器中补选。",
        )
        layout.addSpacing(12)
        layout.addWidget(_label("原曲音频"))
        self.audio_edit = QLineEdit()
        self.audio_edit.setPlaceholderText("可留空；选择音频，或输入音频文件路径")
        self.audio_edit.setAccessibleName("原曲音频路径")
        layout.addWidget(self.audio_edit)
        audio_buttons = QHBoxLayout()
        self.audio_browse_button = QPushButton("选择音频…")
        self.audio_browse_button.clicked.connect(self._browse_audio)
        audio_buttons.addWidget(self.audio_browse_button)
        self.audio_clear_button = QPushButton("清除音频")
        self.audio_clear_button.clicked.connect(self.audio_edit.clear)
        audio_buttons.addWidget(self.audio_clear_button)
        audio_buttons.addStretch(1)
        layout.addLayout(audio_buttons)
        layout.addStretch(1)

    def _build_processing_step(self) -> None:
        layout = self._new_step(
            "选择自动处理方案",
            "这些选项将应用到本次导入的所有分谱。处理仅在适用条件满足时生效；进入编辑器后仍可逐分谱调整。",
        )
        self.processing_checks: dict[str, QCheckBox] = {}
        for field, title, description in PROCESSING_CHOICES:
            control = QCheckBox(title)
            control.setToolTip(description)
            self.processing_checks[field] = control
            layout.addWidget(control)
            detail = _label(description, name="WizardDescription")
            detail.setContentsMargins(28, 0, 0, 8)
            layout.addWidget(detail)
        self.summary_label = _label("", name="WizardSummary")
        self.summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.summary_label)
        layout.addStretch(1)

    def reset(self, source_path: str = "") -> None:
        """Start a fresh session; backward navigation never calls this method."""
        self._busy = False
        self._cancelling = False
        self.source_edit.setText(str(source_path))
        self.audio_edit.clear()
        defaults = PartMapping("", "", [])
        for field, control in self.processing_checks.items():
            control.setChecked(getattr(defaults, field))
        self.steps.setCurrentIndex(0)
        self.set_error("")
        self.set_busy(False)
        self._choices_changed()

    def closeEvent(self, event: QCloseEvent) -> None:
        event.ignore()
        if not self._cancelling:
            self.cancel_requested.emit()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            if not self._cancelling:
                self.cancel_requested.emit()
            event.accept()
        else:
            super().keyPressEvent(event)

    def set_busy(self, busy: bool, cancelling: bool = False) -> None:
        self._busy = busy
        self._cancelling = busy and cancelling
        for content in self._step_contents:
            content.setEnabled(not busy)
        self.cancel_button.setText("正在取消…" if self._cancelling else "取消")
        self._update_navigation()

    def set_error(self, message: str, details: str = "") -> None:
        self.error_label.setText(message)
        self.error_label.setVisible(bool(message))
        self.error_details.setPlainText(details)
        self.error_details_button.setChecked(False)
        self.error_details_button.setVisible(bool(details))
        self.error_details.hide()

    def _toggle_error_details(self, visible: bool) -> None:
        self.error_details.setVisible(visible)
        self.error_details_button.setText("收起错误详情" if visible else "显示错误详情")

    def _source_validation(self) -> str:
        value = self.source_edit.text().strip()
        if not value:
            return "请选择一个 FLP 或 MIDI 文件。"
        try:
            suffix = Path(value).suffix.lower()
        except ValueError:
            suffix = ""
        if suffix not in SOURCE_SUFFIXES:
            return "来源格式不受支持，请选择 .flp、.mid 或 .midi 文件。"
        if _existing_file(value) is None:
            return "音乐来源不存在或不是文件，请重新选择。"
        return ""

    def _choices_changed(self) -> None:
        self.source_validation_label.setText(self._source_validation() or "来源已就绪，可以继续。")
        self.audio_clear_button.setEnabled(not self._busy and bool(self.audio_edit.text()))
        self._update_summary()
        self._update_navigation()

    def _update_summary(self) -> None:
        enabled = [title for field, title, _ in PROCESSING_CHOICES
                   if self.processing_checks[field].isChecked()]
        self.summary_label.setText(
            f"音乐来源：{self.source_edit.text().strip() or '尚未选择'}\n"
            f"原曲音频：{self.audio_edit.text().strip() or '跳过（不绑定音频）'}\n"
            f"自动处理：{'、'.join(enabled) or '全部关闭'}"
        )

    def _update_navigation(self) -> None:
        index = self.steps.currentIndex()
        for number, label in enumerate(self.step_labels):
            label.setStyleSheet(
                "color: #f3dcc0; font-weight: 600; padding: 8px; border-bottom: 2px solid #d4ab77;"
                if number == index else "color: #aebcc7; padding: 8px; border-bottom: 2px solid #435360;"
            )
        self.back_button.setEnabled(not self._busy and index > 0)
        self.skip_button.setVisible(index == 1)
        self.skip_button.setEnabled(not self._busy)
        self.next_button.setVisible(index < 2)
        self.next_button.setEnabled(not self._busy and (index != 0 or not self._source_validation()))
        self.create_button.setVisible(index == 2)
        self.create_button.setEnabled(not self._busy and not self._source_validation())
        self.cancel_button.setEnabled(not self._cancelling)

    def _back(self) -> None:
        if not self._busy and self.steps.currentIndex() > 0:
            self.set_error("")
            self.steps.setCurrentIndex(self.steps.currentIndex() - 1)

    def _next(self) -> None:
        if self._busy:
            return
        if self.steps.currentIndex() == 0 and self._source_validation():
            self.set_error(self._source_validation())
            return
        if self.steps.currentIndex() < 2:
            self.set_error("")
            self.steps.setCurrentIndex(self.steps.currentIndex() + 1)

    def _skip_audio(self) -> None:
        if not self._busy and self.steps.currentIndex() == 1:
            self.audio_edit.clear()
            self._next()

    def _create(self) -> None:
        if self._busy or self.steps.currentIndex() != 2:
            return
        error = self._source_validation()
        if error:
            self.set_error(error)
            return
        source = _existing_file(self.source_edit.text())
        audio_value = self.audio_edit.text().strip()
        audio = _existing_file(audio_value)
        if audio_value and audio is None:
            self.set_error("原曲音频不存在或不是文件，请返回上一步重新选择或跳过音频。")
            return
        # Recheck at submission: the source may disappear while earlier steps are open.
        if source is None:
            self.set_error("音乐来源不存在，请重新选择。")
            return
        self.set_error("")
        self.create_requested.emit(NewProjectOptions(
            source_path=str(source),
            audio_path=str(audio) if audio is not None else "",
            processing={field: control.isChecked() for field, control in self.processing_checks.items()},
        ))

    def _browse_source(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "选择音乐来源", self.source_edit.text(), SOURCE_FILE_FILTER,
        )
        if filename:
            self.source_edit.setText(filename)

    def _browse_audio(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "选择原曲音频", self.audio_edit.text(), AUDIO_FILE_FILTER,
        )
        if filename:
            self.audio_edit.setText(filename)
