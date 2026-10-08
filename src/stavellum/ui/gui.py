"""Chinese desktop workflow for importing, correcting, previewing and exporting scores."""

from __future__ import annotations

import copy
import json
import math
import time
import uuid
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings, QSignalBlocker, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QCloseEvent, QImage, QPainter, QPalette, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from stavellum.domain.models import (
    ANIMATION_DURATIONS,
    ANIMATION_PRESETS,
    Metadata,
    PartMapping,
    ProjectDocument,
    RenderSettings,
    save_document,
)
from stavellum.domain.progress import ExportProgress
from stavellum.exporting.audio import AUDIO_FILE_FILTER
from stavellum.graphics.branding import application_icon, bind_application_icon
from stavellum.graphics.icons import icon_thumbnail, resolve_icon

from .background import BackgroundJob, default_demo_directory
from .new_project import NewProjectOptions, ProjectWizard
from .progress_dialog import ExportProgressDialog
from .welcome import RecentProjects, WelcomePage


def _identity() -> str:
    return f"part-{uuid.uuid4().hex[:12]}"


def _set_mapping_tracks(mapping: PartMapping, tracks: list[str]) -> None:
    mapping.track_ids = tracks
    mapping.articulations = {
        track_id: technique for track_id, technique in mapping.articulations.items()
        if track_id in tracks
    }


def _seconds_label(seconds: float) -> str:
    seconds = max(0, round(seconds * 100))
    return f"{seconds // 6000:02d}:{seconds // 100 % 60:02d}.{seconds % 100:02d}"


def _section(layout: QVBoxLayout, title: str) -> QFormLayout:
    group = QGroupBox(title)
    form = QFormLayout(group)
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    layout.addWidget(group)
    return form


class PreviewCanvas(QWidget):
    """Scale a rendered frame for display without changing the export dimensions."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.frame = QImage()
        self.setMinimumSize(400, 240)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.black)
        self.setPalette(palette)

    def set_frame(self, frame: QImage) -> None:
        self.frame = frame
        self.update()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if self.frame.isNull():
            painter.setPen(Qt.GlobalColor.gray)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "打开工程或示例，生成五线谱预览")
            return
        size = self.frame.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        x, y = (self.width() - size.width()) // 2, (self.height() - size.height()) // 2
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(x, y, self.frame.scaled(size, Qt.AspectRatioMode.KeepAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation))


class MainWindow(QMainWindow):
    def __init__(self, project_path: str | None = None, *, settings: QSettings | None = None) -> None:
        super().__init__()
        self.document: ProjectDocument | None = None
        self.project_path = ""
        self.renderer: Any = None
        self._scene: Any = None
        self._applied_preview_key = ""
        self._pending_preview_key = ""
        self._job: BackgroundJob | None = None
        self._export_dialog: ExportProgressDialog | None = None
        self._dirty = False
        self._loading = False
        self._part_index = -1
        self._position = 0.0
        self._playing = False
        self._play_started = 0.0
        self._play_origin = 0.0
        self._play_rate = 1.0
        self._audio_clock_running = False
        self._audio_tail_running = False
        self._seeking_audio = False
        self._awaiting_audio_start = False
        self._last_error = ""
        self.recent_projects = RecentProjects(settings)
        self._primary_window: QWidget = self
        self._closing = False
        self._wizard_check_unsaved = True
        self._wizard_importing = False
        self._wizard_cancelling = False
        self._compile_after_job = False
        self.setWindowTitle("Stavellum · 五线谱演示")
        bind_application_icon(self)
        self.resize(1480, 920)
        self.audio_output = QAudioOutput(self)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.positionChanged.connect(self._audio_position_changed)
        self.player.durationChanged.connect(self._update_transport)
        self.player.mediaStatusChanged.connect(self._audio_status_changed)
        self.player.playbackStateChanged.connect(self._audio_playback_state_changed)
        self.player.errorOccurred.connect(self._audio_error)
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(round(1000 / RenderSettings().fps))
        self.timer.timeout.connect(self._tick)
        self._build_menu()
        self._build_ui()
        self._update_actions()
        if project_path:
            QTimer.singleShot(0, lambda: self.open_path(project_path))

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("文件")
        self.new_action = QAction("创建新工程…", self)
        self.new_action.setShortcut("Ctrl+N")
        self.new_action.triggered.connect(lambda: self._new_project())
        self.open_action = QAction("打开 FLP / MIDI / 项目…", self)
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.triggered.connect(self._choose_project)
        self.demo_action = QAction("打开三乐器示例", self)
        self.demo_action.triggered.connect(self._open_demo)
        self.audio_action = QAction("选择原曲音频…", self)
        self.audio_action.triggered.connect(self._choose_audio)
        self.save_action = QAction("保存项目", self)
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.triggered.connect(self.save_project)
        self.save_as_action = QAction("项目另存为…", self)
        self.save_as_action.triggered.connect(lambda: self.save_project(save_as=True))
        self.welcome_action = QAction("欢迎页", self)
        self.welcome_action.triggered.connect(self._show_welcome)
        for action in (self.new_action, self.open_action, self.demo_action):
            menu.addAction(action)
        menu.addSeparator()
        for action in (self.audio_action, self.save_action, self.save_as_action):
            menu.addAction(action)
        menu.addSeparator()
        menu.addAction(self.welcome_action)
        menu.addSeparator()
        menu.addAction("退出", self.close)
        export_menu = self.menuBar().addMenu("导出")
        self.export_video_action = QAction("导出 MP4…", self)
        self.export_video_action.triggered.connect(self._export_video)
        self.export_parts_action = QAction("导出 MusicXML / PDF 分谱…", self)
        self.export_parts_action.triggered.connect(self._export_parts)
        export_menu.addAction(self.export_video_action)
        export_menu.addAction(self.export_parts_action)
        preview_menu = self.menuBar().addMenu("预览")
        self.rebuild_action = QAction("强制重新制谱", self)
        self.rebuild_action.triggered.connect(lambda: self.compile_preview(force_rebuild=True))
        self.clear_cache_action = QAction("清除制谱缓存", self)
        self.clear_cache_action.triggered.connect(self._clear_compilation_cache)
        preview_menu.addAction(self.rebuild_action)
        preview_menu.addAction(self.clear_cache_action)

    def _build_ui(self) -> None:
        central = QWidget()
        outer = QVBoxLayout(central)
        self.welcome = WelcomePage()
        self.welcome.close_requested.connect(self.close)
        self.welcome.cancel_task_requested.connect(self._cancel_job)
        self.destroyed.connect(self.welcome.deleteLater)
        self.welcome.new_requested.connect(lambda: self._new_project())
        self.welcome.open_requested.connect(self._choose_project)
        self.welcome.resume_requested.connect(self._resume_editor)
        self.welcome.project_requested.connect(self.open_path)
        self.welcome.demo_requested.connect(self._open_demo)
        self.welcome.set_recent_projects(self.recent_projects.paths())
        self.wizard = ProjectWizard(self)
        self.wizard.create_requested.connect(self._create_project)
        self.wizard.cancel_requested.connect(self._cancel_wizard)
        self.editor = QWidget()
        editor_layout = QVBoxLayout(self.editor)
        self.compile_button = QPushButton("更新预览")
        self.compile_button.clicked.connect(self.compile_preview)
        splitter = QSplitter()
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(420)
        self.tabs.addTab(self._source_tab(), "来源与时间")
        self.tabs.addTab(self._parts_tab(), "分谱与乐器")
        self.tabs.addTab(self._settings_tab(), "画面与文字")
        right = QWidget()
        right_layout = QVBoxLayout(right)
        preview_actions = QHBoxLayout()
        preview_actions.addWidget(QLabel("谱面预览"))
        preview_actions.addStretch(1)
        preview_actions.addWidget(self.compile_button)
        right_layout.addLayout(preview_actions)
        self.preview = PreviewCanvas()
        right_layout.addWidget(self.preview, 1)
        self.preview_status = QLabel("尚未生成谱面")
        right_layout.addWidget(self.preview_status)
        transport = QHBoxLayout()
        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(self.toggle_playback)
        self.seek = QSlider(Qt.Orientation.Horizontal)
        self.seek.setRange(0, 0)
        self.seek.sliderMoved.connect(self.seek_to_milliseconds)
        self.seek.sliderPressed.connect(self._pause_for_seek)
        self.seek.sliderReleased.connect(lambda: self.seek_to_milliseconds(self.seek.value()))
        self.time_label = QLabel("00:00.00 / 00:00.00")
        transport.addWidget(self.play_button)
        transport.addWidget(self.seek, 1)
        transport.addWidget(self.time_label)
        right_layout.addLayout(transport)
        splitter.addWidget(self.tabs)
        splitter.addWidget(right)
        splitter.setSizes([470, 1000])
        editor_layout.addWidget(splitter, 1)
        outer.addWidget(self.editor, 1)
        progress_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.job_label = QLabel("就绪")
        self.cancel_button = QPushButton("取消任务")
        self.cancel_button.clicked.connect(self._cancel_job)
        self.export_detail_button = QPushButton("导出详情")
        self.export_detail_button.clicked.connect(self._show_export_details)
        progress_row.addWidget(self.job_label, 2)
        progress_row.addWidget(self.progress, 1)
        progress_row.addWidget(self.export_detail_button)
        progress_row.addWidget(self.cancel_button)
        outer.addLayout(progress_row)
        self.setCentralWidget(central)

    def _source_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        widget = QWidget()
        layout = QVBoxLayout(widget)
        source_form = _section(layout, "来源文件与 Arrangement")
        self.source_label = QLabel("尚未打开来源")
        self.source_label.setWordWrap(True)
        self.audio_label = QLabel("未选择音频；可以无声预览")
        self.audio_label.setWordWrap(True)
        source_form.addRow(self.source_label)
        source_form.addRow(self.audio_label)
        form = _section(layout, "速度、拍号与音频对齐")
        self.arrangement = QComboBox()
        self.reimport_button = QPushButton("按所选 Arrangement 重新导入")
        self.reimport_button.clicked.connect(self._reimport)
        self.bpm = QDoubleSpinBox()
        self.bpm.setRange(1, 999)
        self.bpm.setDecimals(3)
        self.numerator = QSpinBox()
        self.numerator.setRange(1, 64)
        self.denominator = QComboBox()
        self.denominator.addItems(["1", "2", "4", "8", "16", "32"])
        self.timing_confirmed = QCheckBox("确认整曲使用此固定速度与拍号")
        self.offset = QDoubleSpinBox()
        self.offset.setRange(-3600, 3600)
        self.offset.setDecimals(3)
        self.offset.setSingleStep(0.01)
        self.offset.setSuffix(" 秒")
        source_form.addRow("Arrangement", self.arrangement)
        source_form.addRow(self.reimport_button)
        form.addRow("速度 BPM", self.bpm)
        time_signature = QHBoxLayout()
        time_signature.addWidget(self.numerator)
        time_signature.addWidget(QLabel("/"))
        time_signature.addWidget(self.denominator)
        form.addRow("拍号", time_signature)
        form.addRow(self.timing_confirmed)
        form.addRow("音频内乐谱零点", self.offset)
        help_label = QLabel("偏移为音频中乐谱第一个拍点的位置。正值保留前奏留白，负值裁去谱面开头。")
        help_label.setWordWrap(True)
        form.addRow(help_label)
        diagnostics_form = _section(layout, "导入诊断")
        self.diagnostics = QTextBrowser()
        self.diagnostics.setMinimumHeight(180)
        self.diagnostics.setOpenExternalLinks(False)
        diagnostics_form.addRow(self.diagnostics)
        layout.setStretch(2, 1)
        for field in (self.bpm, self.numerator, self.offset):
            field.valueChanged.connect(self._mark_dirty)
        self.denominator.currentIndexChanged.connect(self._mark_dirty)
        self.timing_confirmed.toggled.connect(self._mark_dirty)
        scroll.setWidget(widget)
        return scroll

    def _parts_tab(self) -> QWidget:
        outer = QWidget()
        layout = QVBoxLayout(outer)
        self.parts = QListWidget()
        self.parts.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.parts.setMaximumHeight(180)
        self.parts.currentRowChanged.connect(self._select_part)
        layout.addWidget(self.parts)
        row = QHBoxLayout()
        for caption, slot in (("新建", self._add_part), ("合并所选", self._merge_parts),
                              ("按来源拆分", self._split_part), ("↑", lambda: self._move_part(-1)),
                              ("↓", lambda: self._move_part(1)), ("删除", self._delete_part)):
            button = QPushButton(caption)
            button.clicked.connect(slot)
            row.addWidget(button)
        layout.addLayout(row)
        midi_split_button = QPushButton("将选中的来源轨道按 MIDI 通道拆分")
        midi_split_button.clicked.connect(self._split_midi_channels)
        layout.addWidget(midi_split_button)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        editor = QWidget()
        editor_layout = QVBoxLayout(editor)
        part_form = _section(editor_layout, "分谱名称与来源轨道")
        icon_form = _section(editor_layout, "乐器与图标")
        notation_form = _section(editor_layout, "谱表与音高记法")
        recognition_form = _section(editor_layout, "节奏与自动识别")
        mapping_form = _section(editor_layout, "来源映射与奏法")
        self.part_name = QLineEdit()
        self.part_enabled = QCheckBox("参与总谱、预览和导出")
        self.part_tracks = QListWidget()
        self.part_tracks.setMinimumHeight(110)
        self.part_tracks.setMaximumHeight(170)
        self.instrument = QComboBox()
        self.instrument.setEditable(True)
        for key, label in (("unknown", "未识别"), ("violin", "小提琴"), ("viola", "中提琴"),
                           ("cello", "大提琴"), ("double_bass", "低音提琴"), ("bass", "电贝斯"),
                           ("piano", "钢琴"), ("flute", "长笛"), ("clarinet", "单簧管"),
                           ("oboe", "双簧管"), ("bassoon", "大管"), ("trumpet", "小号"),
                           ("horn", "圆号"), ("trombone", "长号"), ("tuba", "大号"),
                           ("guitar", "吉他"), ("harp", "竖琴"), ("bell", "钟 / 铃"),
                           ("percussion", "打击乐")):
            self.instrument.addItem(f"{label} ({key})", key)
        self.icon = QLineEdit()
        self.icon.setPlaceholderText("留空自动匹配；例如 violin 或 fa:solid:guitar")
        icon_controls = QWidget()
        self._icon_controls = icon_controls
        icon_layout = QVBoxLayout(icon_controls)
        icon_layout.setContentsMargins(0, 0, 0, 0)
        icon_row = QHBoxLayout()
        self.icon_preview = QLabel()
        self.icon_preview.setFixedSize(48, 48)
        self.icon_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_preview.setStyleSheet("background: #171717; color: white;")
        icon_row.addWidget(self.icon_preview)
        icon_row.addWidget(self.icon, 1)
        self.icon_name = QLabel()
        self.icon_name.setWordWrap(True)
        self.icon_name.hide()
        icon_row.addWidget(self.icon_name, 1)
        icon_layout.addLayout(icon_row)
        icon_actions = QHBoxLayout()
        self.choose_icon_button = QPushButton("选择图标…")
        self.auto_icon_button = QPushButton("恢复自动")
        self.choose_icon_button.clicked.connect(self._choose_icon)
        self.auto_icon_button.clicked.connect(lambda: self._set_icon(""))
        for button in (self.choose_icon_button, self.auto_icon_button):
            icon_actions.addWidget(button)
        icon_layout.addLayout(icon_actions)
        self.icon.textChanged.connect(self._update_icon_preview)
        self.use_icon = QCheckBox("使用乐器图标")
        self.use_icon.setToolTip("在预览与视频中显示此分谱的图标；关闭后保留所选图标")
        self.use_icon.toggled.connect(icon_controls.setEnabled)
        icon_controls.setEnabled(self.use_icon.isChecked())
        self.clef = QComboBox()
        for label, key in (("自动", "auto"), ("高音", "treble"), ("低音", "bass"),
                           ("中音", "alto"), ("次中音", "tenor"), ("打击乐", "percussion")):
            self.clef.addItem(label, key)
        self.key_signature = QComboBox()
        self.key_signature.addItem("自动判断", None)
        for value in range(-7, 8):
            label = "C / a（无升降号）" if value == 0 else f"{abs(value)} 个{'降' if value < 0 else '升'}号"
            self.key_signature.addItem(label, value)
        self.auto_simplify_accidentals = QCheckBox("自动简化音符记法（保留调号）")
        self.auto_simplify_accidentals.setToolTip("选择音高相同的等音拼写，减少不必要的临时升降号；保留必要的变音记号，不改变调号、源音高或播放时间")
        self.auto_ottava = QCheckBox("自动八度移位")
        self.auto_ottava.setToolTip("用 8va、8vb、15ma、15mb 简化连续高低音区，仅在明显减少加线且不影响同谱表其他声部时应用；不改变源音高、实际演奏音高或播放时间")
        self.transpose = QSpinBox()
        self.transpose.setRange(-48, 48)
        self.transpose.setSuffix(" 半音")
        self.quantization = QComboBox()
        for value in (8, 16, 32, 64):
            self.quantization.addItem(f"1/{value}", value)
        self.triplets = QCheckBox("识别三连音")
        self.auto_staccato = QCheckBox("尽力识别跳音（保守）")
        self.auto_grace = QCheckBox("尽力识别倚音（保守）")
        self.auto_dynamics = QCheckBox("识别音量自动化的渐强／渐弱")
        self.auto_dynamics.setToolTip("将明确的 FL 音量自动化转换为发夹线；来源冲突或路由不明确时保留诊断")
        for control in (self.auto_staccato, self.auto_grace):
            control.setToolTip("仅在演奏时间和上下文足够明确时推断；原始音符与播放时间保持不变")
        self.grand_staff = QCheckBox("钢琴双谱表（整体显隐）")
        self.percussion = QCheckBox("使用打击乐记谱")
        self.keyswitches = QLineEdit()
        self.keyswitches.setPlaceholderText("明确指定 MIDI 音高，例如 0, 1, 24；默认不过滤")
        self.percussion_map = QPlainTextEdit()
        self.percussion_map.setMaximumHeight(100)
        self.percussion_map.setPlaceholderText('JSON 对象，例如 {"36": "kick", "38": "snare"}；也支持 C5 等音高')
        self.articulations = QPlainTextEdit()
        self.articulations.setMaximumHeight(100)
        self.articulations.setPlaceholderText('JSON 对象，来源轨道 ID → 奏法，例如 {"track-1": "pizz."}')
        for form, rows in (
            (part_form, (("分谱名称", self.part_name), ("", self.part_enabled),
                         ("来源轨道", self.part_tracks))),
            (icon_form, (("乐器", self.instrument), ("", self.use_icon),
                         ("图标", icon_controls))),
            (notation_form, (("谱号", self.clef), ("调号", self.key_signature),
                             ("记谱移调", self.transpose), ("", self.grand_staff),
                             ("", self.percussion), ("", self.auto_simplify_accidentals),
                             ("", self.auto_ottava))),
            (recognition_form, (("量化格点", self.quantization), ("", self.triplets),
                                ("", self.auto_staccato), ("", self.auto_grace),
                                ("", self.auto_dynamics))),
            (mapping_form, (("Keyswitch 音高", self.keyswitches),
                            ("打击乐音高映射", self.percussion_map),
                            ("来源奏法", self.articulations))),
        ):
            for label, control in rows:
                if label:
                    form.addRow(label, control)
                else:
                    form.addRow(control)
        self.apply_part_button = QPushButton("应用分谱设置")
        self.apply_part_button.clicked.connect(self.apply_part)
        editor_layout.addWidget(self.apply_part_button)
        info = QLabel("勾选轨道会将其移入此分谱；原分谱若变空会自动停用。用合并保留声部身份，用拆分分离来源轨道。")
        info.setWordWrap(True)
        editor_layout.addWidget(info)
        editor_layout.addStretch(1)
        scroll.setWidget(editor)
        layout.addWidget(scroll, 1)
        self._part_editor = editor
        for control in (self.part_name, self.icon, self.keyswitches):
            control.textEdited.connect(self._mark_dirty)
        for control in (self.part_enabled, self.use_icon, self.triplets, self.auto_staccato,
                        self.auto_grace, self.auto_dynamics, self.auto_simplify_accidentals,
                        self.auto_ottava,
                        self.grand_staff, self.percussion):
            control.toggled.connect(self._mark_dirty)
        for control in (self.instrument, self.clef, self.key_signature, self.quantization):
            control.currentIndexChanged.connect(self._mark_dirty)
        self.instrument.editTextChanged.connect(self._mark_dirty)
        self.instrument.currentIndexChanged.connect(self._instrument_changed)
        self.transpose.valueChanged.connect(self._mark_dirty)
        self.part_tracks.itemChanged.connect(self._mark_dirty)
        self.percussion_map.textChanged.connect(self._mark_dirty)
        self.articulations.textChanged.connect(self._mark_dirty)
        return outer

    def _settings_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        widget = QWidget()
        layout = QVBoxLayout(widget)
        metadata_form = _section(layout, "曲目信息")
        frame_form = _section(layout, "画面尺寸与谱区布局")
        text_form = _section(layout, "文字位置与字号")
        animation_form = _section(layout, "声部动画")
        overlay_form = _section(layout, "开场与报幕")
        logo_form = _section(layout, "Logo")
        encoding_form = _section(layout, "渲染与视频编码")
        self.metadata_controls: dict[str, QLineEdit] = {}
        for key, label in (("title", "曲名"), ("subtitle", "副标题"),
                           ("composer", "作曲"), ("arranger", "编曲")):
            control = QLineEdit()
            control.textEdited.connect(self._mark_dirty)
            self.metadata_controls[key] = control
            metadata_form.addRow(label, control)
        self.settings_controls: dict[str, Any] = {}
        self.render_backend = QComboBox()
        self.render_backend.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.render_backend.setMinimumContentsLength(12)
        for label, key in (("自动（优先 RHI Vulkan，失败回退 CPU）", "auto"), ("CPU", "cpu"),
                           ("RHI Vulkan（不可用时报错）", "gpu")):
            self.render_backend.addItem(label, key)
        self.render_backend.currentIndexChanged.connect(self._mark_dirty)
        self.render_backend.currentTextChanged.connect(self.render_backend.setToolTip)
        self.render_backend.setToolTip(self.render_backend.currentText())
        encoding_form.addRow("帧渲染", self.render_backend)
        self.video_encoder = QComboBox()
        self.video_encoder.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.video_encoder.setMinimumContentsLength(12)
        for label, key in (("自动（优先 NVIDIA NVENC，失败回退 CPU）", "auto"),
                           ("CPU / libx264", "libx264"),
                           ("NVIDIA H.264 NVENC（不可用时报错）", "h264_nvenc")):
            self.video_encoder.addItem(label, key)
        self.video_encoder.currentIndexChanged.connect(self._encoder_options_changed)
        self.video_encoder.currentTextChanged.connect(self.video_encoder.setToolTip)
        self.video_encoder.setToolTip(self.video_encoder.currentText())
        encoding_form.addRow("视频编码", self.video_encoder)
        self.animation_preset = QComboBox()
        for name, label in (("fast", "快速"), ("medium", "中速"), ("slow", "慢速"),
                            ("very_slow", "极慢"), ("custom", "自定义")):
            self.animation_preset.addItem(label, name)
        self.animation_preset.currentIndexChanged.connect(self._animation_preset_changed)
        animation_form.addRow("动画速度预设", self.animation_preset)
        self.fps = QComboBox()
        self.fps.addItems(["24", "25", "30", "50", "60"])
        self.fps.currentIndexChanged.connect(self._mark_dirty)
        frame_form.addRow("帧率", self.fps)
        specs = [
            ("width", "宽度", 320, 7680, 2), ("height", "高度", 240, 4320, 2),
            ("staff_scale", "基础谱面缩放", 0.3, 3.0, 0.05),
            ("score_left", "谱区左边界（比例）", 0.0, 1.0, 0.01),
            ("score_right", "谱区右边界（比例）", 0.0, 1.0, 0.01),
            ("score_top", "谱区顶部（比例）", 0.0, 1.0, 0.01),
            ("score_bottom", "谱区底部（比例）", 0.0, 1.0, 0.01),
            ("header_width", "固定头部宽度（比例）", 0.01, 0.5, 0.01),
            ("title_x", "文字左位置（比例）", 0.0, 1.0, 0.01),
            ("title_y", "曲名顶部（比例）", 0.0, 1.0, 0.01),
            ("title_font_size", "曲名字号", 10, 200, 1),
            ("subtitle_font_size", "副标题字号", 8, 100, 1),
            ("credits_font_size", "署名字号", 8, 100, 1),
            ("enter_seconds", "入场时长（秒）", 0.01, 60.0, 0.05),
            ("exit_seconds", "退场时长（秒）", 0.01, 60.0, 0.05),
            ("reflow_seconds", "声部居中与缩放时长（秒）", 0.01, 60.0, 0.05),
            ("animation_stable_seconds", "动画后最短稳定时间（秒）", 0.0, 60.0, 0.1),
            ("intro_delay_seconds", "开场停顿（秒）", 0.0, 3600.0, 0.1),
            ("overlay_enter_seconds", "报幕淡入（秒）", 0.01, 60.0, 0.05),
            ("overlay_exit_seconds", "报幕淡出（秒）", 0.01, 60.0, 0.05),
            ("announcement_hold_seconds", "报幕停留（秒）", 0.0, 3600.0, 0.1),
            ("cache_megabytes", "图块缓存（MiB）", 8, 1024, 8),
            ("crf", "CPU H.264 CRF（较小更清晰）", 0, 51, 1),
            ("nvenc_cq", "NVENC CQ（较小更清晰）", 1, 51, 1),
        ]
        for key, label, minimum, maximum, step in specs:
            if isinstance(minimum, int) and isinstance(maximum, int):
                control = QSpinBox()
            else:
                control = QDoubleSpinBox()
                control.setDecimals(3)
            control.setRange(minimum, maximum)
            control.setSingleStep(step)
            if key == "staff_scale":
                control.setToolTip("少声部时自动放大，最多为基础大小的两倍；仍受谱区空间限制")
            elif key == "animation_stable_seconds":
                control.setToolTip("动画完成后至少保持指定秒数；不足时直接采用后续稳定布局，0 关闭此规则")
            control.valueChanged.connect(self._animation_duration_changed if key in ANIMATION_DURATIONS
                                         else self._mark_dirty)
            self.settings_controls[key] = control
            if key.startswith("title_") or key in {"subtitle_font_size", "credits_font_size"}:
                form = text_form
            elif key in {"enter_seconds", "exit_seconds", "reflow_seconds", "animation_stable_seconds"}:
                form = animation_form
            elif key in {"intro_delay_seconds", "overlay_enter_seconds", "overlay_exit_seconds", "announcement_hold_seconds"}:
                form = overlay_form
            elif key in {"cache_megabytes", "crf", "nvenc_cq"}:
                form = encoding_form
            else:
                form = frame_form
            form.addRow(label, control)
        self.announcement_auto_hide = QCheckBox("报幕停留后淡出")
        self.announcement_auto_hide.toggled.connect(self._overlay_options_changed)
        overlay_form.insertRow(1, self.announcement_auto_hide)
        self._overlay_options_changed()
        self.logo_enabled = QCheckBox("在画面右下角显示软件 Logo")
        self.logo_enabled.toggled.connect(self._logo_options_changed)
        logo_form.addRow(self.logo_enabled)
        self.logo_display_mode = QComboBox()
        for label, key in (("常驻", "persistent"), ("淡入后常驻", "fade_in"),
                           ("开场淡入，停留后淡出", "intro")):
            self.logo_display_mode.addItem(label, key)
        self.logo_display_mode.currentIndexChanged.connect(self._logo_options_changed)
        logo_form.addRow("Logo 显示方式", self.logo_display_mode)
        for key, label, minimum, maximum, step in (
            ("logo_size_ratio", "Logo 大小（画面短边比例）", 0.02, 0.25, 0.01),
            ("logo_opacity", "Logo 最大不透明度", 0.0, 1.0, 0.05),
            ("logo_enter_seconds", "Logo 淡入（秒）", 0.01, 60.0, 0.05),
            ("logo_hold_seconds", "Logo 停留（秒）", 0.0, 3600.0, 0.1),
            ("logo_exit_seconds", "Logo 淡出（秒）", 0.01, 60.0, 0.05),
        ):
            control = QDoubleSpinBox()
            control.setDecimals(3)
            control.setRange(minimum, maximum)
            control.setSingleStep(step)
            control.setToolTip("预览与视频共用；从画面零秒计时，独立于报幕和动画速度预设")
            control.valueChanged.connect(self._mark_dirty)
            self.settings_controls[key] = control
            logo_form.addRow(label, control)
        self._logo_options_changed()
        self.preset = QComboBox()
        self.preset.addItems(["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"])
        self.preset.currentIndexChanged.connect(self._mark_dirty)
        encoding_form.insertRow(4, "CPU 编码预设", self.preset)
        self.nvenc_preset = QComboBox()
        self.nvenc_preset.addItems([f"p{value}" for value in range(1, 8)])
        self.nvenc_preset.setToolTip("p1 最快，p7 最慢；独立于 CPU 编码预设")
        self.nvenc_preset.currentIndexChanged.connect(self._mark_dirty)
        encoding_form.addRow("NVENC 编码预设", self.nvenc_preset)
        self._encoder_options_changed()
        layout.addStretch(1)
        scroll.setWidget(widget)
        return scroll

    def _encoder_options_changed(self, *args: Any) -> None:
        encoder = self.video_encoder.currentData()
        self.settings_controls["crf"].setEnabled(encoder != "h264_nvenc")
        self.preset.setEnabled(encoder != "h264_nvenc")
        self.settings_controls["nvenc_cq"].setEnabled(encoder != "libx264")
        self.nvenc_preset.setEnabled(encoder != "libx264")
        self._mark_dirty()

    def _animation_preset_changed(self, *args: Any) -> None:
        if self._loading:
            return
        durations = ANIMATION_PRESETS.get(self.animation_preset.currentData())
        if durations is not None:
            for name, duration in zip(ANIMATION_DURATIONS, durations, strict=True):
                control = self.settings_controls[name]
                blocker = QSignalBlocker(control)
                control.setValue(duration)
                del blocker
        self._mark_dirty()

    def _animation_duration_changed(self, *args: Any) -> None:
        if self._loading:
            return
        blocker = QSignalBlocker(self.animation_preset)
        self.animation_preset.setCurrentIndex(self.animation_preset.findData("custom"))
        del blocker
        self._mark_dirty()

    def _overlay_options_changed(self, *args: Any) -> None:
        self.settings_controls["announcement_hold_seconds"].setEnabled(
            self.announcement_auto_hide.isChecked())
        self._mark_dirty()

    def _mark_dirty(self, *args: Any) -> None:
        if self._loading or self.document is None:
            return
        self._dirty = True
        self.preview_status.setText("设置已修改；点击“更新预览”应用")
        self._update_title()

    def _logo_options_changed(self, *args: Any) -> None:
        enabled = self.logo_enabled.isChecked()
        mode = self.logo_display_mode.currentData()
        self.logo_display_mode.setEnabled(enabled)
        for name in ("logo_size_ratio", "logo_opacity"):
            self.settings_controls[name].setEnabled(enabled)
        self.settings_controls["logo_enter_seconds"].setEnabled(enabled and mode != "persistent")
        for name in ("logo_hold_seconds", "logo_exit_seconds"):
            self.settings_controls[name].setEnabled(enabled and mode == "intro")
        self._mark_dirty()

    def _update_title(self) -> None:
        name = Path(self.project_path).name if self.project_path else (
            self.document.metadata.title if self.document else "未命名"
        )
        self.setWindowTitle(f"{'* ' if self._dirty else ''}{name} · Stavellum")

    def _update_actions(self) -> None:
        busy = self._job is not None
        has_document = self.document is not None
        for control in (self.audio_action, self.compile_button, self.export_video_action,
                        self.export_parts_action, self.save_action, self.save_as_action,
                        self.rebuild_action):
            control.setEnabled(has_document and not busy)
        self.clear_cache_action.setEnabled(not busy)
        for control in (self.open_action, self.demo_action, self.new_action,
                        self.welcome_action):
            control.setEnabled(not busy)
        self.welcome.set_has_document(has_document)
        self.welcome.set_busy(busy)
        self.wizard.set_busy(busy, cancelling=self._wizard_cancelling)
        self.welcome.task_status.setVisible(busy)
        self.wizard.task_status.setVisible(busy and self._wizard_importing)
        self.welcome.cancel_task_button.setEnabled(
            busy and getattr(self._job, "_cancel_requested_at", None) is None)
        self.reimport_button.setEnabled(has_document and not busy and bool(
            self.document and self.document.project.source_type in {"flp", "midi"}
        ))
        self.cancel_button.setEnabled(busy and getattr(self._job, "_cancel_requested_at", None) is None)
        self.export_detail_button.setEnabled(self._export_dialog is not None)
        self.tabs.setEnabled(has_document and not busy)
        self.play_button.setEnabled(self.renderer is not None and not busy)
        self.seek.setEnabled(self.renderer is not None and not busy)

    def _switch_window(self, target: QWidget) -> None:
        if self._closing:
            return
        wizard_visible = self.wizard.isVisible()
        wizard_geometry = self.wizard.geometry()
        if self.wizard.parentWidget() is not target:
            self.wizard.setParent(target, Qt.WindowType.Tool)
            self.wizard.setWindowModality(Qt.WindowModality.NonModal)
        self._primary_window.hide()
        self._primary_window = target
        if target.isMinimized():
            target.showNormal()
        else:
            target.show()
        target.raise_()
        target.activateWindow()
        if wizard_visible:
            self.wizard.setGeometry(wizard_geometry)
            self.wizard.show()
            self.wizard.raise_()

    def _dialog_parent(self) -> QWidget:
        return self.wizard if self.wizard.isVisible() else self._primary_window

    def _set_job_message(self, message: str) -> None:
        for window in (self, self.welcome, self.wizard):
            window.job_label.setText(message)

    def _set_job_progress(self, value: int) -> None:
        for window in (self, self.welcome, self.wizard):
            window.progress.setValue(value)

    def _show_welcome(self) -> None:
        if self._job is not None:
            return
        self.pause_playback()
        self.welcome.set_recent_projects(self.recent_projects.paths())
        self._switch_window(self.welcome)
        self._update_actions()

    def _resume_editor(self) -> None:
        if self.document is not None and self._job is None:
            self._switch_window(self)
            self._update_actions()

    def _new_project(self, source_path: str = "", *, check_unsaved: bool = True) -> None:
        if self._job is not None:
            return
        if self.wizard.isVisible():
            self.wizard.raise_()
            self.wizard.activateWindow()
            return
        self._wizard_check_unsaved = check_unsaved
        self._wizard_cancelling = False
        self.wizard.reset(source_path)
        self.wizard.setParent(self._primary_window, Qt.WindowType.Tool)
        self.wizard.show()
        self.wizard.raise_()
        self.wizard.activateWindow()
        self._update_actions()

    def _hide_wizard(self) -> None:
        self.wizard.hide()
        self._wizard_cancelling = False
        self._update_actions()

    def _cancel_wizard(self) -> None:
        if not self.wizard.isVisible():
            return
        if self._wizard_importing and self._job is not None:
            self._cancel_job()
        else:
            self._hide_wizard()

    def _create_project(self, options: NewProjectOptions) -> None:
        if self._job is not None or not self.wizard.isVisible():
            return
        source = Path(options.source_path)
        if not source.is_file() or source.suffix.lower() not in {".flp", ".mid", ".midi"}:
            self.wizard.set_error("请选择存在的 FLP 或 MIDI 来源文件。")
            return
        if options.audio_path and not Path(options.audio_path).is_file():
            self.wizard.set_error("原曲音频不存在，请重新选择或跳过音频步骤。")
            return
        if self._wizard_check_unsaved and not self._confirm_replace():
            return
        self._wizard_importing = True
        self._wizard_cancelling = False
        self.wizard.set_error("")
        self.pause_playback()
        self._start_job("import", (str(source.resolve()), 0),
                        lambda project: self._wizard_imported(project, options))

    def _wizard_imported(self, project: Any, options: NewProjectOptions) -> None:
        from stavellum.domain.mapping import suggest_mappings

        if self._wizard_cancelling:
            return
        mappings = suggest_mappings(project)
        for mapping in mappings:
            for name, enabled in options.processing.items():
                setattr(mapping, name, enabled)
        document = ProjectDocument(project, mappings, options.audio_path,
                                   metadata=Metadata(title=project.name))
        self.wizard.hide()
        self.set_document(document)
        self._dirty = True
        self._update_title()

    def _record_recent_project(self, path: str) -> None:
        self.recent_projects.record(path)
        self.welcome.set_recent_projects(self.recent_projects.paths())

    def _project_loaded(self, document: ProjectDocument, path: str) -> None:
        self.set_document(document, path)
        self._record_recent_project(path)

    def _choose_project(self) -> None:
        if self._job is not None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self._dialog_parent(), "打开来源或保存的项目", "",
            "可用工程 (*.flp *.mid *.midi *.stproj);;FL Studio (*.flp);;MIDI (*.mid *.midi);;项目 (*.stproj)",
        )
        if path:
            self.open_path(path)

    def open_path(self, path: str, *, check_unsaved: bool = True) -> None:
        if self._closing or self._job is not None:
            return
        suffix = Path(path).suffix.lower()
        if suffix in {".flp", ".mid", ".midi"}:
            self._new_project(path, check_unsaved=check_unsaved)
            return
        if suffix != ".stproj":
            self._show_error("请选择 FLP、MIDI 或 .stproj 工程文件。")
            return
        if not Path(path).is_file():
            self._show_error(f"工程文件不存在：{path}")
            return
        if check_unsaved and not self._confirm_replace():
            return
        self.pause_playback()
        self._start_job("load", path, lambda doc: self._project_loaded(doc, path))

    def _open_demo(self) -> None:
        if self._job is None and self._confirm_replace():
            self._start_job("demo", default_demo_directory(), self.set_document)

    def _source_imported(self, project: Any) -> None:
        from stavellum.domain.mapping import suggest_mappings

        old = self.document if self._job and getattr(self._job, "reimporting", False) else None
        suggested = suggest_mappings(project)
        if old:
            track_ids = {t.track_id for t in project.tracks}
            retained = []
            used = set()
            for mapping in copy.deepcopy(old.mappings):
                _set_mapping_tracks(mapping, [t for t in mapping.track_ids if t in track_ids])
                if mapping.track_ids:
                    retained.append(mapping)
                    if mapping.enabled:
                        used.update(mapping.track_ids)
            part_ids = {mapping.part_id for mapping in retained}
            for mapping in suggested:
                _set_mapping_tracks(mapping, [t for t in mapping.track_ids if t not in used])
                if mapping.track_ids:
                    while mapping.part_id in part_ids:
                        mapping.part_id = _identity()
                    part_ids.add(mapping.part_id)
                    retained.append(mapping)
            document = ProjectDocument(project, retained, old.audio_path,
                                       copy.deepcopy(old.settings), copy.deepcopy(old.metadata),
                                       icon_assets=copy.deepcopy(old.icon_assets))
        else:
            document = ProjectDocument(project, suggested, metadata=Metadata(title=project.name))
        self.set_document(document)
        self._dirty = True
        self._update_title()

    def set_document(self, document: ProjectDocument, path: str = "") -> None:
        self.pause_playback()
        self._close_renderer()
        self.document = document
        self.project_path = path
        self._loading = True
        self._part_index = -1
        self._scene = None
        self._applied_preview_key = ""
        self._pending_preview_key = ""
        self.preview.set_frame(QImage())
        project = document.project
        self.source_label.setText(
            f"{project.name}  ·  {project.source_type.upper()} {project.source_version}\n"
            f"{project.source_path}\n{len(project.tracks)} 个来源轨道，{len(project.notes)} 个音符，PPQ {project.ppq}"
        )
        self.arrangement.clear()
        self.arrangement.addItems(project.arrangement_names or ["默认编曲"])
        self.arrangement.setCurrentIndex(project.arrangement_index)
        self.bpm.setValue(project.bpm)
        self.numerator.setValue(project.numerator)
        self.denominator.setCurrentText(str(project.denominator))
        self.timing_confirmed.setChecked(project.timing_confirmed)
        self.offset.setValue(document.settings.score_start_in_audio_sec)
        for key, control in self.metadata_controls.items():
            control.setText(getattr(document.metadata, key))
        for key, control in self.settings_controls.items():
            control.setValue(getattr(document.settings, key))
        self.animation_preset.setCurrentIndex(
            self.animation_preset.findData(document.settings.animation_preset()))
        self.announcement_auto_hide.setChecked(document.settings.announcement_auto_hide)
        self._overlay_options_changed()
        self.logo_enabled.setChecked(document.settings.logo_enabled)
        self.logo_display_mode.setCurrentIndex(
            self.logo_display_mode.findData(document.settings.logo_display_mode))
        self._logo_options_changed()
        self.fps.setCurrentText(str(document.settings.fps))
        self.preset.setCurrentText(document.settings.preset)
        self.render_backend.setCurrentIndex(self.render_backend.findData(document.settings.render_backend))
        self.video_encoder.setCurrentIndex(self.video_encoder.findData(document.settings.video_encoder))
        self.nvenc_preset.setCurrentText(document.settings.nvenc_preset)
        self._encoder_options_changed()
        self._refresh_parts(0)
        self._set_audio(document.audio_path)
        self._show_diagnostics()
        self._loading = False
        self._update_icon_preview()
        self._dirty = False
        self._position = 0.0
        self._switch_window(self)
        self._update_title()
        self._update_actions()
        self._update_transport()
        self._compile_after_job = False
        if project.timing_confirmed:
            # A result receiver runs before the import/load worker finishes.
            if self._job is not None:
                self._compile_after_job = True
            else:
                QTimer.singleShot(0, self.compile_preview)
        else:
            self.tabs.setCurrentIndex(0)
            self.preview_status.setText("请在“来源与时间”确认固定速度与拍号，再更新预览")

    def _show_diagnostics(self, compiled: Any = ()) -> None:
        if not self.document:
            return
        messages = list(self.document.project.diagnostics)
        codes = {(item.code, item.track_id, item.message) for item in messages}
        messages.extend(item for item in compiled if (item.code, item.track_id, item.message) not in codes)
        labels = {"error": "错误", "warning": "提示", "info": "信息"}
        text = "\n\n".join(
            f"[{labels.get(d.severity, d.severity)}] {d.code}\n{d.message}"
            + (f"\n轨道：{d.track_id}" if d.track_id else "") for d in messages
        )
        self.diagnostics.setPlainText(text or "未发现导入问题。请核对声部归属、奏法、谱号和音频起点。")

    def _reimport(self) -> None:
        if not self.document or self._job:
            return
        if not self._gather_document():
            return
        source = self.document.project.source_path
        if not Path(source).is_file():
            self._show_error("来源文件不存在，请重新选择 FLP 或 MIDI。")
            return
        self._start_job("import", (source, self.arrangement.currentIndex()), self._source_imported)
        if self._job:
            self._job.reimporting = True

    def _choose_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self._dialog_parent(), "选择原曲音频", "", AUDIO_FILE_FILTER)
        if path and self.document:
            self.document.audio_path = path
            self._set_audio(path)
            self._mark_dirty()

    def _set_audio(self, path: str) -> None:
        self.pause_playback()
        self.player.stop()
        if path and Path(path).is_file():
            self.player.setSource(QUrl.fromLocalFile(str(Path(path).resolve())))
            self.audio_label.setText(f"原曲音频：{path}")
        else:
            self.player.setSource(QUrl())
            self.audio_label.setText(f"音频文件不存在：{path}" if path else "未选择音频；可以无声预览")

    def _refresh_parts(self, selected: int = 0) -> None:
        if not self.document:
            return
        blocker = QSignalBlocker(self.parts)
        self.parts.clear()
        for mapping in self.document.mappings:
            self.parts.addItem(f"{'●' if mapping.enabled else '○'} {mapping.name}  ·  {len(mapping.track_ids)} 轨")
        selected = min(max(0, selected), len(self.document.mappings) - 1)
        self.parts.setCurrentRow(selected)
        del blocker
        self._part_index = -1
        self._load_part(selected)

    def _select_part(self, index: int) -> None:
        if self._loading or not self.document or index == self._part_index:
            return
        previous = self._part_index
        if previous >= 0 and not self.apply_part(refresh=False):
            blocker = QSignalBlocker(self.parts)
            self.parts.setCurrentRow(previous)
            del blocker
            return
        self._load_part(index)

    def _load_part(self, index: int) -> None:
        self._part_index = index
        valid = bool(self.document and 0 <= index < len(self.document.mappings))
        self._part_editor.setEnabled(valid)
        if not valid or not self.document:
            return
        was_loading = self._loading
        self._loading = True
        mapping = self.document.mappings[index]
        self.part_name.setText(mapping.name)
        self.part_enabled.setChecked(mapping.enabled)
        self.part_tracks.clear()
        for track in self.document.project.tracks:
            item = QListWidgetItem(f"{track.name} [{track.track_id}]")
            item.setData(Qt.ItemDataRole.UserRole, track.track_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if track.track_id in mapping.track_ids
                               else Qt.CheckState.Unchecked)
            self.part_tracks.addItem(item)
        instrument_index = self.instrument.findData(mapping.instrument)
        if instrument_index >= 0:
            self.instrument.setCurrentIndex(instrument_index)
        else:
            self.instrument.setCurrentText(mapping.instrument)
        self.icon.setText(mapping.icon)
        self.use_icon.setChecked(mapping.use_icon)
        self.clef.setCurrentIndex(self.clef.findData(mapping.clef))
        self.key_signature.setCurrentIndex(self.key_signature.findData(mapping.key_signature))
        self.auto_simplify_accidentals.setChecked(mapping.auto_simplify_accidentals)
        self.auto_ottava.setChecked(mapping.auto_ottava)
        self.transpose.setValue(mapping.transpose)
        self.quantization.setCurrentIndex(self.quantization.findData(mapping.quantization))
        self.triplets.setChecked(mapping.triplets)
        self.auto_staccato.setChecked(mapping.auto_staccato)
        self.auto_grace.setChecked(mapping.auto_grace)
        self.auto_dynamics.setChecked(mapping.auto_dynamics)
        self.grand_staff.setChecked(mapping.grand_staff)
        self.percussion.setChecked(mapping.percussion)
        self.keyswitches.setText(", ".join(map(str, mapping.keyswitches)))
        self.percussion_map.setPlainText(json.dumps(mapping.percussion_map, ensure_ascii=False, indent=2))
        self.articulations.setPlainText(json.dumps(mapping.articulations, ensure_ascii=False, indent=2))
        self._loading = was_loading
        self._update_icon_preview()

    def apply_part(self, checked: bool = False, *, refresh: bool = True) -> bool:
        if not self.document or self._part_index < 0:
            return True
        try:
            previous = self.document.mappings[self._part_index]
            mapping = copy.deepcopy(previous)
            mapping.name = self.part_name.text().strip() or "未命名分谱"
            mapping.track_ids = [
                self.part_tracks.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.part_tracks.count())
                if self.part_tracks.item(i).checkState() == Qt.CheckState.Checked
            ]
            mapping.enabled = self.part_enabled.isChecked()
            if mapping.enabled and not mapping.track_ids:
                raise ValueError("已启用分谱至少需要一个来源轨道。")
            mapping.instrument = self._instrument_value()
            mapping.icon = self.icon.text().strip()
            resolve_icon(mapping.icon, self.document.icon_assets)
            mapping.use_icon = self.use_icon.isChecked()
            if mapping.icon == "none":
                mapping.icon = ""
                mapping.use_icon = False
            mapping.clef = self.clef.currentData()
            mapping.key_signature = self.key_signature.currentData()
            mapping.auto_simplify_accidentals = self.auto_simplify_accidentals.isChecked()
            mapping.auto_ottava = self.auto_ottava.isChecked()
            mapping.transpose = self.transpose.value()
            mapping.quantization = self.quantization.currentData()
            mapping.triplets = self.triplets.isChecked()
            mapping.auto_staccato = self.auto_staccato.isChecked()
            mapping.auto_grace = self.auto_grace.isChecked()
            mapping.auto_dynamics = self.auto_dynamics.isChecked()
            mapping.grand_staff = self.grand_staff.isChecked()
            mapping.percussion = self.percussion.isChecked()
            mapping.keyswitches = [int(v.strip()) for v in self.keyswitches.text().replace("，", ",").split(",") if v.strip()]
            if any(not 0 <= key <= 127 for key in mapping.keyswitches):
                raise ValueError("Keyswitch 的 MIDI 音高必须在 0 到 127 之间。")
            for field, editor in (("percussion_map", self.percussion_map),
                                  ("articulations", self.articulations)):
                value = json.loads(editor.toPlainText().strip() or "{}")
                if not isinstance(value, dict) or any(not isinstance(v, str) for v in value.values()):
                    raise ValueError("映射必须为 JSON 对象，且每个值为文字。")
                setattr(mapping, field, {str(k): v for k, v in value.items()})
            removed = set(previous.track_ids) - set(mapping.track_ids)
            if removed.intersection(mapping.articulations):
                mapping.articulations = {
                    track_id: technique for track_id, technique in mapping.articulations.items()
                    if track_id not in removed
                }
                loading = self._loading
                self._loading = True
                self.articulations.setPlainText(json.dumps(mapping.articulations, ensure_ascii=False, indent=2))
                self._loading = loading
            changed = mapping != self.document.mappings[self._part_index]
            self.document.mappings[self._part_index] = mapping
            if self.icon.text().strip() == "none":
                self.icon.clear()
                self.use_icon.setChecked(False)
            if mapping.enabled:
                moved = set(mapping.track_ids)
                for index, other in enumerate(self.document.mappings):
                    if index != self._part_index and other.enabled:
                        remaining = [t for t in other.track_ids if t not in moved]
                        if remaining != other.track_ids:
                            changed = True
                            _set_mapping_tracks(other, remaining)
                            if not remaining:
                                other.enabled = False
            if changed:
                self._mark_dirty()
            if refresh:
                self._refresh_parts(self._part_index)
            return True
        except (ValueError, TypeError) as exc:
            self._show_error(f"分谱设置无效：{exc}")
            return False

    def _instrument_value(self) -> str:
        index = self.instrument.currentIndex()
        if index >= 0 and self.instrument.currentText() == self.instrument.itemText(index):
            return self.instrument.currentData() or "unknown"
        return self.instrument.currentText().strip() or "unknown"

    def _instrument_changed(self, *args: Any) -> None:
        if not self._loading:
            self._update_icon_preview()

    def _set_icon(self, reference: str) -> None:
        self.icon.setText(reference)
        self._mark_dirty()

    def _update_icon_preview(self, *args: Any) -> None:
        if self._loading or not self.document:
            return
        reference = self.icon.text().strip() or self._instrument_value()
        asset = self.document.icon_assets.get(reference.removeprefix("asset:")) if reference.startswith("asset:") else None
        self.icon.setVisible(asset is None)
        self.icon_name.setVisible(asset is not None)
        if asset is not None:
            if asset.source.startswith("fontawesome:"):
                self.icon_name.setText("Font Awesome · " + asset.source.removeprefix("fontawesome:").replace(":", "/"))
            else:
                self.icon_name.setText(asset.name)
        try:
            image = icon_thumbnail(reference, self.document.icon_assets)
            self.icon_preview.setPixmap(QPixmap.fromImage(image))
            self.icon_preview.setToolTip(asset.name if asset else reference)
        except (ValueError, OSError, RuntimeError) as exc:
            self.icon_preview.setText("!")
            self.icon_preview.setToolTip(str(exc))

    def _choose_icon(self) -> None:
        if not self.document:
            return
        from .icon_picker import IconPicker

        dialog = IconPicker(self.recent_projects.settings, self)
        if dialog.exec() == dialog.DialogCode.Accepted and dialog.selection is not None:
            reference, asset = dialog.selection
            if asset is not None:
                self.document.icon_assets.setdefault(reference[6:], asset)
            self._set_icon(reference)
        dialog.deleteLater()

    def _add_part(self) -> None:
        if not self.document or not self.apply_part(refresh=False):
            return
        self.document.mappings.append(PartMapping(_identity(), "新分谱", [], enabled=False))
        self._refresh_parts(len(self.document.mappings) - 1)
        self._mark_dirty()

    def _merge_parts(self) -> None:
        if not self.document or not self.apply_part(refresh=False):
            return
        selected = sorted(self.parts.row(item) for item in self.parts.selectedItems())
        if len(selected) < 2:
            self.statusBar().showMessage("按 Ctrl 选择至少两个分谱后合并。", 5000)
            return
        first = copy.deepcopy(self.document.mappings[selected[0]])
        first.track_ids = list(dict.fromkeys(
            t for i in selected for t in self.document.mappings[i].track_ids
        ))
        first.articulations = {
            t: articulation for i in selected
            for t, articulation in self.document.mappings[i].articulations.items()
        }
        first.enabled = bool(first.track_ids)
        first.name = " + ".join(self.document.mappings[i].name for i in selected)
        self.document.mappings[selected[0]] = first
        for index in reversed(selected[1:]):
            del self.document.mappings[index]
        self._refresh_parts(selected[0])
        self._mark_dirty()

    def _split_part(self) -> None:
        if not self.document or not self.apply_part(refresh=False) or self._part_index < 0:
            return
        mapping = self.document.mappings[self._part_index]
        if len(mapping.track_ids) < 2:
            self.statusBar().showMessage("此分谱只有一个来源轨道。", 5000)
            return
        names = {t.track_id: t.name for t in self.document.project.tracks}
        split = []
        for track_id in mapping.track_ids:
            part = copy.deepcopy(mapping)
            part.part_id = _identity()
            part.name = names.get(track_id, track_id)
            part.track_ids = [track_id]
            part.articulations = {track_id: mapping.articulations[track_id]} if track_id in mapping.articulations else {}
            split.append(part)
        index = self._part_index
        self.document.mappings[index:index + 1] = split
        self._refresh_parts(index)
        self._mark_dirty()

    def _split_midi_channels(self) -> None:
        if not self.document or not self.apply_part(refresh=False) or self._part_index < 0:
            return
        from stavellum.domain.mapping import split_track_by_midi_channel

        mapping = copy.deepcopy(self.document.mappings[self._part_index])
        item = self.part_tracks.currentItem()
        if item is None:
            self.statusBar().showMessage("先在来源轨道列表中点击需要拆分的轨道。", 5000)
            return
        track_id = item.data(Qt.ItemDataRole.UserRole)
        if track_id not in mapping.track_ids:
            self.statusBar().showMessage("请选择属于当前分谱的来源轨道。", 5000)
            return
        try:
            new_ids = split_track_by_midi_channel(self.document.project, track_id)
        except ValueError as exc:
            self._show_error(str(exc))
            return
        if new_ids == [track_id]:
            self.statusBar().showMessage("此来源轨道只包含一个 MIDI 通道。", 5000)
            return
        names = {track.track_id: track.name for track in self.document.project.tracks}
        replacements = []
        remaining = [t for t in mapping.track_ids if t != track_id]
        if remaining:
            original = copy.deepcopy(mapping)
            original.track_ids = remaining
            original.articulations.pop(track_id, None)
            replacements.append(original)
        for new_id in new_ids:
            part = copy.deepcopy(mapping)
            part.part_id = _identity()
            part.name = names[new_id]
            part.track_ids = [new_id]
            part.articulations = ({new_id: mapping.articulations[track_id]}
                                  if track_id in mapping.articulations else {})
            replacements.append(part)
        index = self._part_index
        self.document.mappings[index:index + 1] = replacements
        for part in self.document.mappings:
            if track_id in part.track_ids:
                part.track_ids = [t for t in part.track_ids if t != track_id] + new_ids
                articulation = part.articulations.pop(track_id, None)
                if articulation:
                    part.articulations.update({new_id: articulation for new_id in new_ids})
        self._refresh_parts(index)
        self._show_diagnostics()
        self._mark_dirty()

    def _move_part(self, direction: int) -> None:
        if not self.document or not self.apply_part(refresh=False):
            return
        index, other = self._part_index, self._part_index + direction
        if 0 <= index < len(self.document.mappings) and 0 <= other < len(self.document.mappings):
            self.document.mappings[index], self.document.mappings[other] = (
                self.document.mappings[other], self.document.mappings[index]
            )
            self._refresh_parts(other)
            self._mark_dirty()

    def _delete_part(self) -> None:
        if self.document and 0 <= self._part_index < len(self.document.mappings):
            index = self._part_index
            del self.document.mappings[index]
            self._refresh_parts(index)
            self._mark_dirty()

    def _gather_document(self, *, validate: bool = True) -> bool:
        if not self.document or not self.apply_part(refresh=False):
            return False
        document = self.document
        document.project.bpm = self.bpm.value()
        document.project.numerator = self.numerator.value()
        document.project.denominator = int(self.denominator.currentText())
        document.project.timing_confirmed = self.timing_confirmed.isChecked()
        document.settings.score_start_in_audio_sec = self.offset.value()
        for key, control in self.metadata_controls.items():
            setattr(document.metadata, key, control.text())
        for key, control in self.settings_controls.items():
            setattr(document.settings, key, control.value())
        document.settings.fps = int(self.fps.currentText())
        document.settings.preset = self.preset.currentText()
        document.settings.render_backend = self.render_backend.currentData()
        document.settings.video_encoder = self.video_encoder.currentData()
        document.settings.nvenc_preset = self.nvenc_preset.currentText()
        document.settings.announcement_auto_hide = self.announcement_auto_hide.isChecked()
        document.settings.logo_enabled = self.logo_enabled.isChecked()
        document.settings.logo_display_mode = self.logo_display_mode.currentData()
        if validate:
            try:
                document.validate()
            except ValueError as exc:
                self._show_error(str(exc))
                return False
        return True

    def save_project(self, checked: bool = False, *, save_as: bool = False) -> bool:
        if not self._gather_document(validate=False):
            return False
        path = self.project_path
        if save_as or not path:
            path, _ = QFileDialog.getSaveFileName(self._dialog_parent(), "保存项目", path or "五线谱演示.stproj",
                                                 "Stavellum 项目 (*.stproj)")
            if not path:
                return False
            if not path.lower().endswith(".stproj"):
                path += ".stproj"
        try:
            save_document(self.document, path)
        except (OSError, ValueError) as exc:
            self._show_error(f"保存失败：{exc}")
            return False
        self.project_path = path
        self._dirty = False
        self._record_recent_project(path)
        self._update_title()
        self.statusBar().showMessage(f"项目已保存：{path}", 6000)
        return True

    def compile_preview(self, checked: bool = False, *, force_rebuild: bool = False) -> None:
        from stavellum.presentation.compilation_cache import preview_key

        if self._closing or self._job is not None or not self._gather_document():
            return
        key = preview_key(self.document)
        if not force_rebuild and self.renderer is not None and key == self._applied_preview_key:
            self._show_renderer_status(force=True)
            return
        self.pause_playback()
        self._pending_preview_key = key
        self.preview_status.setText("正在准备预览…")
        document = copy.deepcopy(self.document)
        payload = (document, {"force_rebuild": True}) if force_rebuild else document
        self._start_job("compile", payload, self._compiled)

    def _clear_compilation_cache(self) -> None:
        from stavellum.presentation.compilation_cache import CompilationCache

        if self._job is not None:
            return
        cleared = CompilationCache().clear()
        self.statusBar().showMessage("制谱缓存已清除" if cleared else "部分缓存无法清除", 6000)

    def _compiled(self, scene: Any) -> None:
        from stavellum.rendering.render import FrameRenderer

        renderer = FrameRenderer(scene)
        position = min(self._position, scene.settings.presentation_duration(
            self.player.duration() / 1000, scene.score_duration))
        try:
            frame = renderer.render_frame(position)
        except Exception:
            renderer.close()
            raise
        self._close_renderer()
        self._scene = scene
        self.renderer = renderer
        self._applied_preview_key = self._pending_preview_key
        self.timer.setInterval(max(1, round(1000 / scene.settings.fps)))
        self._position = position
        self._show_diagnostics(getattr(scene, "diagnostics", []))
        self._show_renderer_status(force=True)
        self._update_transport()
        self.preview.set_frame(frame)

    def _export_video(self) -> None:
        if not self._gather_document():
            return
        if not self.document.audio_path or not Path(self.document.audio_path).is_file():
            self._show_error("请选择存在的原曲音频，再导出有声视频。")
            return
        path, _ = QFileDialog.getSaveFileName(self._dialog_parent(), "导出视频", "五线谱演示.mp4", "MP4 视频 (*.mp4)")
        if path:
            if not path.lower().endswith(".mp4"):
                path += ".mp4"
            self.pause_playback()
            self._start_job("video", (copy.deepcopy(self.document), path), self._export_finished)

    def _export_parts(self) -> None:
        if not self._gather_document():
            return
        path = QFileDialog.getExistingDirectory(self._dialog_parent(), "选择分谱导出文件夹")
        if path:
            self.pause_playback()
            self._start_job("parts", (copy.deepcopy(self.document), path), self._export_finished)

    def _export_finished(self, result: Any) -> None:
        self._set_job_message("导出完成；可查看导出详情")
        if self._export_dialog:
            self._export_dialog.finish_success(result)

    def _show_export_details(self) -> None:
        if self._export_dialog:
            self._export_dialog.show()
            self._export_dialog.raise_()
            self._export_dialog.activateWindow()

    def _start_job(self, operation: str, payload: Any, receiver: Any) -> None:
        if self._closing or self._job is not None:
            return
        job = BackgroundJob(operation, payload, self)
        self._job = job
        job.progress.connect(self._job_progress)
        job.progress_detail.connect(self._job_detail)
        job.succeeded.connect(lambda result: self._job_result(receiver, result))
        job.failed.connect(self._job_error)
        job.cancelled.connect(self._job_cancelled)
        job.finished.connect(self._job_finished)
        self._set_job_progress(0)
        self._set_job_message({"load": "正在打开项目…", "import": "正在读取来源…",
                                "demo": "正在生成示例…", "compile": "正在准备预览…",
                                "parts": "正在导出分谱…", "video": "正在逐帧导出视频…"}[operation])
        if operation in {"video", "parts"}:
            if self._export_dialog:
                self._export_dialog.deleteLater()
            self._export_dialog = ExportProgressDialog(operation, str(payload[1]), self)
            self._export_dialog.cancel_requested.connect(self._cancel_job)
            self._export_dialog.show()
        self._update_actions()
        try:
            job.start()
        except (OSError, RuntimeError) as exc:
            self._job_error(str(exc), "后台进程无法启动。")
            job.shutdown()
            self._job_finished()

    def _job_result(self, receiver: Any, result: Any) -> None:
        if (self._closing or (self._wizard_importing and self._wizard_cancelling)
                or (self._job is not None and self._job._cancel_requested_at is not None)):
            return
        try:
            receiver(result)
            self._set_job_progress(1000)
            if self._job and self._job.operation not in {"video", "parts"}:
                self._set_job_message("任务完成")
        except Exception as exc:
            if self._wizard_importing:
                self._job_error(f"读取后台结果失败：{exc}", str(exc))
            else:
                self._show_error(f"读取后台结果失败：{exc}")

    def _job_progress(self, fraction: float, message: str) -> None:
        if self._job and self._job.operation in {"video", "parts"}:
            return
        if self._wizard_importing and self._wizard_cancelling:
            return
        self._set_job_progress(round(fraction * 1000))
        if message:
            self._set_job_message(message)
            if self._job and self._job.operation == "compile":
                self.preview_status.setText(message)

    def _job_detail(self, detail: ExportProgress) -> None:
        if (not self._job or self._job.operation not in {"video", "parts"}
                or not self._export_dialog or not self._export_dialog.running
                or self._job._cancel_requested_at is not None):
            return
        self._export_dialog.update_progress(detail)
        if detail.total > 0:
            self._set_job_progress(min(999, round(detail.completed / detail.total * 1000)))
        if detail.message:
            self._set_job_message(detail.message)

    def _job_cancelled(self) -> None:
        self._set_job_message("任务已取消")
        if self._job and self._job.operation == "compile":
            self.preview_status.setText("预览更新已取消；原预览已保留" if self.renderer else "预览更新已取消")
        if self._wizard_importing:
            self._wizard_cancelling = True
        if self._job and self._job.operation in {"video", "parts"} and self._export_dialog:
            self._export_dialog.finish_cancelled()

    def _job_error(self, message: str, details: str) -> None:
        self._last_error = details
        self._set_job_message("任务失败")
        if self._wizard_importing:
            if not self._wizard_cancelling:
                self.wizard.set_error(message, details)
            return
        if self._job and self._job.operation in {"video", "parts"} and self._export_dialog:
            self._export_dialog.finish_failure(message, details)
        if self._job and self._job.operation == "compile":
            self.preview_status.setText("谱面生成失败，请查看错误并修改设置")
        box = QMessageBox(QMessageBox.Icon.Critical, "任务失败", message,
                          parent=self._dialog_parent())
        box.setDetailedText(details)
        box.exec()

    def _job_finished(self) -> None:
        job = self._job
        self._job = None
        if job:
            job.deleteLater()
        if self._wizard_importing:
            self._wizard_importing = False
            if self._wizard_cancelling:
                self._hide_wizard()
        self._update_actions()
        if self._compile_after_job:
            self._compile_after_job = False
            QTimer.singleShot(0, self.compile_preview)

    def _cancel_job(self) -> None:
        if self._job and self._job.running and self._job._cancel_requested_at is None:
            if self._wizard_importing:
                self._wizard_cancelling = True
            self._job.cancel()
            if self._job.operation in {"video", "parts"} and self._export_dialog:
                self._export_dialog.request_cancel()
            self._set_job_message("正在取消，等待后台清理…")
            self._update_actions()
            self.cancel_button.setEnabled(False)

    def _duration(self) -> float:
        if not self.document:
            return 0.0
        if self.renderer is not None:
            scene = self.renderer.scene
            score_duration = scene.score_duration
        else:
            score_duration = self.document.project.duration_seconds
        return self._preview_settings().presentation_duration(self.player.duration() / 1000,
                                                              score_duration)

    def _preview_settings(self) -> RenderSettings:
        if self.renderer is not None:
            return self.renderer.scene.settings
        return self.document.settings if self.document is not None else RenderSettings()

    def _in_intro_delay(self) -> bool:
        return self._position < self._preview_settings().intro_delay_seconds

    def _synchronize_audio(self) -> None:
        self._audio_clock_running = False
        self._audio_tail_running = self._in_audio_tail()
        audio_requested = (self._playing and not self._in_intro_delay()
                           and not self._audio_tail_running and not self.player.source().isEmpty())
        self._awaiting_audio_start = audio_requested
        self._seeking_audio = True
        try:
            seconds = self._preview_settings().audio_time(self._position)
            if self._in_intro_delay() or self._audio_tail_running or self.player.source().isEmpty():
                self.player.pause()
                self.player.setPosition(self.player.duration() if self._audio_tail_running
                                        else round(seconds * 1000))
            else:
                self.player.setPosition(round(seconds * 1000))
                if self._playing:
                    self.player.play()
        finally:
            self._seeking_audio = False
        self._audio_clock_running = (self._playing and not self._in_intro_delay()
                                     and not self._audio_tail_running
                                     and not self.player.source().isEmpty()
                                     and self._media_clock_active())
        self._awaiting_audio_start = audio_requested and not self._audio_clock_running

    def _update_transport(self, *args: Any) -> None:
        duration = self._duration()
        self.seek.setRange(0, math.ceil(duration * 1000))
        if not self.seek.isSliderDown():
            self.seek.setValue(round(self._position * 1000))
        self.time_label.setText(f"{_seconds_label(self._position)} / {_seconds_label(duration)}")

    def toggle_playback(self) -> None:
        if self._playing:
            self.pause_playback()
            return
        if self.renderer is None:
            return
        if self._position >= self._duration():
            self.seek_to_milliseconds(0)
        self._playing = True
        self._anchor_playback_clock(self._position)
        self._synchronize_audio()
        self.play_button.setText("暂停")
        self.timer.setInterval(max(1, round(1000 / self._preview_settings().fps)))
        self.timer.start()

    def pause_playback(self) -> None:
        if self._playing:
            self._advance_playback_clock(start_audio=False)
        self._playing = False
        self._audio_clock_running = False
        self._audio_tail_running = False
        self._awaiting_audio_start = False
        self._anchor_playback_clock(self._position)
        self.player.pause()
        self.timer.stop()
        if hasattr(self, "play_button"):
            self.play_button.setText("播放")

    def _pause_for_seek(self) -> None:
        self.pause_playback()

    def seek_to_milliseconds(self, milliseconds: int) -> None:
        self._position = min(self._duration(), max(0.0, milliseconds / 1000))
        self._anchor_playback_clock(self._position)
        self._synchronize_audio()
        self._update_transport()
        self._render_position()

    def _audio_position_changed(self, milliseconds: int) -> None:
        if (self._playing and not self._seeking_audio and not self._in_intro_delay()
                and not self._audio_tail_running and not self.player.source().isEmpty()
                and self.player.mediaStatus() != QMediaPlayer.MediaStatus.EndOfMedia):
            seconds = self._preview_settings().presentation_time(milliseconds / 1000)
            if self._media_clock_active():
                self._resume_audio_clock()
                self._slew_media_clock(seconds)

    def _audio_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        if (not self._playing or self._seeking_audio or self._in_intro_delay()
                or self.player.source().isEmpty()):
            return
        if self._audio_tail_running and status != QMediaPlayer.MediaStatus.EndOfMedia:
            return
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            if (self._awaiting_audio_start
                    or self.player.mediaStatus() != QMediaPlayer.MediaStatus.EndOfMedia):
                return
            if not self._audio_tail_running:
                self._position = max(self._position, self._preview_settings().presentation_time(
                    self.player.duration() / 1000))
                self._anchor_playback_clock(self._position)
            self._audio_tail_running = True
            self._audio_clock_running = False
        elif self._media_clock_active():
            self._resume_audio_clock()
        else:
            self._freeze_audio_clock()

    def _audio_playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        if (not self._playing or self._seeking_audio or self._in_intro_delay()
                or self._audio_tail_running or self.player.source().isEmpty()
                or self.player.mediaStatus() == QMediaPlayer.MediaStatus.EndOfMedia):
            return
        if self._media_clock_active():
            self._resume_audio_clock()
        else:
            self._freeze_audio_clock()

    def _freeze_audio_clock(self) -> None:
        # The last displayed frame is the stable anchor, even when the media
        # getter still reports an older, coarse timestamp during buffering.
        self._audio_clock_running = False
        self._anchor_playback_clock(self._position)

    def _resume_audio_clock(self) -> None:
        if self._audio_clock_running:
            return
        self._anchor_playback_clock(self._position)
        self._audio_clock_running = True
        self._audio_tail_running = False
        self._awaiting_audio_start = False
        self._slew_media_clock(self._preview_settings().presentation_time(
            self.player.position() / 1000))

    def _slew_media_clock(self, seconds: float) -> None:
        now = time.monotonic()
        current = self._play_origin + max(0.0, now - self._play_started) * self._play_rate
        current = min(current, self._duration())
        self._position = current
        self._play_origin = current
        self._play_started = now
        # Qt media positions can arrive late and in coarse timestamp steps.
        # Correct phase at at most 5% speed over a half-second horizon without
        # replacing the continuous clock or holding frames until audio catches up.
        correction = max(-0.05, min(0.05, (seconds - current) / 0.5))
        self._play_rate = 1.0 + correction

    def _anchor_playback_clock(self, seconds: float) -> None:
        self._play_origin = seconds
        self._play_started = time.monotonic()
        self._play_rate = 1.0

    def _media_clock_active(self) -> bool:
        return (self.player.mediaStatus() == QMediaPlayer.MediaStatus.BufferedMedia
                and self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState)

    def _in_audio_tail(self) -> bool:
        return (not self.player.source().isEmpty() and self.player.duration() > 0
                and self._position >= self._preview_settings().presentation_time(
                    self.player.duration() / 1000))

    def _advance_playback_clock(self, *, start_audio: bool = True) -> None:
        if self._awaiting_audio_start and self._media_clock_active():
            self._resume_audio_clock()
        predicted = self._play_origin + max(0.0, time.monotonic() - self._play_started) * self._play_rate
        if self.player.source().isEmpty():
            self._position = predicted
        elif self._in_intro_delay():
            delay = self._preview_settings().intro_delay_seconds
            self._position = min(predicted, delay)
            if predicted >= delay:
                # Media starts at audio zero. Discard time spent waiting between
                # timer ticks and let buffering begin from this exact boundary.
                self._anchor_playback_clock(delay)
                if start_audio:
                    self._synchronize_audio()
        elif self._audio_tail_running:
            self._position = predicted
        elif self._audio_clock_running and self._media_clock_active():
            self._position = min(predicted, self._preview_settings().presentation_time(
                self.player.duration() / 1000))
        self._position = min(self._position, self._duration())

    def _audio_error(self, error: Any, message: str) -> None:
        if error != QMediaPlayer.Error.NoError:
            self.pause_playback()
            self.player.stop()
            self.player.setSource(QUrl())
            self._update_transport()
            self.audio_label.setText(f"此音频无法预览：{message}。可无声预览谱面；视频导出将尝试读取原曲音频。")

    def _tick(self) -> None:
        if not self._playing:
            return
        self._advance_playback_clock()
        if self._position >= self._duration():
            self._position = self._duration()
            self.pause_playback()
        self._update_transport()
        self._render_position()

    def _render_position(self) -> None:
        if self.renderer is not None:
            try:
                self.preview.set_frame(self.renderer.render_frame(self._position))
                self._show_renderer_status()
            except Exception as exc:
                self.pause_playback()
                self._close_renderer()
                self.preview_status.setText(f"预览失败：{exc}")
                self._update_actions()

    def _show_renderer_status(self, *, force: bool = False) -> None:
        if self.renderer is None:
            return
        if not force and self.preview_status.text().startswith("设置已修改"):
            return
        backend = ("RHI Vulkan" if getattr(self.renderer, "render_backend", "cpu") == "gpu"
                   else "CPU")
        text = f"预览已更新（{backend}）；导出将逐帧渲染"
        reasons = getattr(self.renderer, "fallback_reasons", ())
        if reasons:
            text += "；GPU 回退：" + "；".join(reasons)
        if self.preview_status.text() != text:
            self.preview_status.setText(text)

    def _close_renderer(self) -> None:
        renderer, self.renderer = self.renderer, None
        close = getattr(renderer, "close", None)
        if close is not None:
            close()

    def _show_error(self, message: str) -> None:
        QMessageBox.warning(self._dialog_parent(), "请检查设置", message)

    def _confirm_replace(self) -> bool:
        if not self._dirty:
            return True
        answer = QMessageBox.question(
            self._dialog_parent(), "保存修改", "当前项目有未保存修改，是否先保存？",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        return answer == QMessageBox.StandardButton.Discard or (
            answer == QMessageBox.StandardButton.Save and self.save_project()
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._closing:
            event.accept()
            return
        if not self._confirm_replace():
            event.ignore()
            return
        self._closing = True
        self.pause_playback()
        if self._job:
            self._job.shutdown()
            self._job = None
        if self._export_dialog:
            self._export_dialog.hide()
        self._close_renderer()
        self.wizard.hide()
        self.welcome._allow_close = True
        self.welcome.close()
        event.accept()


def run_gui(project_path: str | None = None) -> int:
    from stavellum.graphics.qt import prepare_render_app

    app = prepare_render_app(RenderSettings())
    app.setApplicationName("Stavellum")
    app.setWindowIcon(application_icon())
    app.setOrganizationName("Stavellum")
    window = MainWindow(project_path)
    window._show_welcome()
    return app.exec()
