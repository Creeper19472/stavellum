"""The desktop landing page and its persisted recent-project list."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QRectF, QSettings, QSize, Qt, Signal
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QFont,
    QPainter,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizeGrip,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .branding import APPLICATION_DESCRIPTION, bind_application_icon
from .theme import DESKTOP_STYLESHEET, apply_desktop_theme, desktop_icon

# The palette echoes the editor's dark chrome with the product's warm accent.
_PANEL_HOVER = "#212b37"
_PANEL_BORDER = "#2a3441"
_TEXT = "#e8ecf1"
_MUTED = "#a4b2c3"
_ACCENT = "#edc398"


class RecentProjects:
    """Remember successful opens and saves without dropping missing files."""

    KEY = "welcome/recentProjects"
    LIMIT = 10

    def __init__(self, settings: QSettings | None = None) -> None:
        self.settings = (settings if settings is not None
                         else QSettings("Stavellum", "Stavellum"))

    @staticmethod
    def _absolute(path: str) -> str:
        return os.path.abspath(os.path.expanduser(path))

    def paths(self) -> list[str]:
        raw = self.settings.value(self.KEY, [])
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, (list, tuple)):
            return []
        result: list[str] = []
        seen: set[str] = set()
        for value in raw:
            if not isinstance(value, str) or Path(value).suffix.lower() != ".stproj":
                continue
            path = self._absolute(value)
            key = os.path.normcase(path)
            if key not in seen:
                result.append(path)
                seen.add(key)
            if len(result) == self.LIMIT:
                break
        return result

    def record(self, path: str) -> None:
        if Path(path).suffix.lower() != ".stproj":
            return
        absolute = self._absolute(path)
        key = os.path.normcase(absolute)
        paths = [absolute, *(p for p in self.paths() if os.path.normcase(p) != key)]
        self.settings.setValue(self.KEY, paths[:self.LIMIT])
        self.settings.sync()


class _RecentProjectDelegate(QStyledItemDelegate):
    """Keep project names readable and elide long paths within their own row."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._file_icon = desktop_icon("file")

    def sizeHint(self, option, index) -> QSize:
        return QSize(0, 64)

    def paint(self, painter, option, index) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        rect = QRectF(option.rect).adjusted(1, 3, -1, -3)
        painter.setPen(Qt.PenStyle.NoPen)
        if selected or hovered:
            painter.setBrush(QColor(_PANEL_HOVER))
            painter.drawRoundedRect(rect, 6, 6)
        title = str(index.data(Qt.ItemDataRole.DisplayRole)).split("\n", 1)[0]
        path = str(index.data(Qt.ItemDataRole.UserRole))
        text_rect = option.rect.adjusted(46, 10, -16, -10)
        self._file_icon.paint(painter, option.rect.left() + 14, option.rect.top() + 22, 18, 18)
        font = QFont(option.font)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(_TEXT if option.state & QStyle.StateFlag.State_Enabled else _MUTED))
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                         painter.fontMetrics().elidedText(title, Qt.TextElideMode.ElideRight, text_rect.width()))
        font.setBold(False)
        font.setPixelSize(11)
        painter.setFont(font)
        painter.setPen(QColor(_MUTED))
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
                         painter.fontMetrics().elidedText(path, Qt.TextElideMode.ElideMiddle, text_rect.width()))
        painter.restore()


class WelcomePage(QWidget):
    """Frameless landing window; MainWindow owns document and application lifetime."""

    new_requested = Signal()
    open_requested = Signal()
    resume_requested = Signal()
    project_requested = Signal(str)
    demo_requested = Signal()
    close_requested = Signal()
    cancel_task_requested = Signal()

    _GUIDE_SECTIONS = (
        ("从来源到谱面", "创建新工程，选择 FLP 或 MIDI；可选原曲音频，再选择自动处理方案。"),
        ("检查来源与时间", "导入默认使用第一个 Arrangement。进入编辑器后可调整编曲、速度、拍号和音频同步；FLP 未确认固定时钟时，需人工核对速度与拍号。"),
        ("整理分谱", "检查建议的乐器分组，在分谱设置中调整音符简化、八度移位和演奏法识别。"),
        ("预览与导出", "生成谱面预览，保存 .stproj 工程，或导出分谱和滚动谱视频。"),
    )
    _ABOUT_SECTIONS = (
        (f"Stavellum  {__version__}", APPLICATION_DESCRIPTION),
        ("一个工程，完整保留", "工程保存来源信息、分谱、原曲音频路径和渲染设置，便于继续编辑和再次导出。"),
        ("开源许可", "Stavellum 使用 GNU GPL v3 或更新版本。第三方库与字体保留各自的许可证；完整文本及来源随发行包提供。"),
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle("欢迎 · Stavellum")
        bind_application_icon(self)
        self.resize(1060, 720)
        self.setMinimumSize(900, 620)
        self._allow_close = False
        self._busy = False
        self._has_document = False
        self.setObjectName("WelcomePage")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        apply_desktop_theme(self)
        self.setStyleSheet(DESKTOP_STYLESHEET + f"""
            QWidget#WelcomePage {{ background: #10161e; }}
            QLabel#StartBrand {{ color: {_MUTED}; font-size: 14px; }}
            QLabel#StartTitle {{ color: {_TEXT}; font-size: 22px; font-weight: 600; }}
            QLabel#SectionHeading {{ color: {_MUTED}; font-size: 12px; }}
            QToolButton#StartAction {{
                background: transparent; border: 1px solid {_PANEL_BORDER};
                border-radius: 6px; padding: 5px 12px;
            }}
            QToolButton#StartAction:hover {{ background: {_PANEL_HOVER}; border-color: #718197; }}
            QToolButton#StartAction:focus {{ border-color: {_ACCENT}; }}
            QToolButton#StartAction:disabled {{ color: #768496; border-color: {_PANEL_BORDER}; }}
            QPushButton#LinkButton {{
                background: transparent; border: 1px solid transparent;
                color: {_MUTED}; padding: 4px 2px;
            }}
            QPushButton#LinkButton:hover {{ color: {_TEXT}; }}
            QPushButton#LinkButton:focus {{ border-color: {_ACCENT}; }}
            QToolButton#WindowButton {{
                background: transparent; border: 0; border-radius: 5px;
                color: {_MUTED}; font-size: 15px; padding: 0;
            }}
            QToolButton#WindowButton:hover {{ background: {_PANEL_HOVER}; color: {_TEXT}; }}
            QListWidget#RecentList {{ background: transparent; border: 0; padding: 0; }}
        """)
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 10, 18, 16)
        root.setSpacing(0)
        chrome = QHBoxLayout()
        chrome.setSpacing(4)
        self.header = QLabel("Stavellum")
        self.header.setObjectName("StartBrand")
        self.header.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        chrome.addWidget(self.header)
        chrome.addStretch(1)
        self.minimize_button = self._window_button("—", "最小化", self.showMinimized)
        self.close_button = self._window_button("×", "关闭", self._close)
        chrome.addWidget(self.minimize_button)
        chrome.addWidget(self.close_button)
        root.addLayout(chrome)
        self.pages = QStackedWidget()
        root.addWidget(self.pages, 1)
        self.pages.addWidget(self._build_main_page())
        self.pages.addWidget(self._text_page(self._GUIDE_SECTIONS, "使用指南"))
        self.pages.addWidget(self._text_page(self._ABOUT_SECTIONS, "关于"))
        self._grip = QSizeGrip(self)
        self._grip.setFixedSize(18, 18)
        self._grip.setStyleSheet("background: transparent;")

    # --- painting and window chrome -------------------------------------

    def mousePressEvent(self, event) -> None:
        if (event.button() == Qt.MouseButton.LeftButton
                and event.position().y() < 52 and self.windowHandle() is not None):
            self.windowHandle().startSystemMove()
        super().mousePressEvent(event)

    def _window_button(self, glyph: str, tip: str, handler) -> QToolButton:
        button = QToolButton()
        button.setObjectName("WindowButton")
        button.setText(glyph)
        button.setToolTip(tip)
        button.setAccessibleName(tip)
        button.setFixedSize(30, 28)
        button.clicked.connect(handler)
        return button

    def _close(self) -> None:
        self.close()

    # --- page construction -----------------------------------------------

    def _start_action(self, icon: str, title: str, handler) -> QToolButton:
        button = QToolButton()
        button.setObjectName("StartAction")
        button.setText(title)
        button.setIcon(desktop_icon(icon))
        button.setIconSize(QSize(18, 18))
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        button.setFixedHeight(34)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setToolTip(title)
        button.setAccessibleName(title)
        button.clicked.connect(handler)
        return button

    def _build_main_page(self) -> QWidget:
        page = QWidget()
        page_layout = QHBoxLayout(page)
        page_layout.setContentsMargins(36, 32, 36, 6)
        workspace = QWidget()
        workspace.setMinimumWidth(640)
        workspace.setMaximumWidth(780)
        page_layout.addStretch(1)
        page_layout.addWidget(workspace, 3)
        page_layout.addStretch(1)
        layout = QVBoxLayout(workspace)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        title = QLabel("开始创作")
        title.setObjectName("StartTitle")
        layout.addWidget(title)
        layout.addSpacing(6)
        subtitle = QLabel("新建工程，或继续最近的作品。")
        subtitle.setObjectName("Tagline")
        layout.addWidget(subtitle)
        layout.addSpacing(22)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.new_button = self._start_action("new", "新建工程", lambda: self._emit(self.new_requested))
        self.new_button.setShortcut("Ctrl+N")
        self.new_button.setToolTip("新建工程 (Ctrl+N)")
        self.open_button = self._start_action("open", "打开工程", lambda: self._emit(self.open_requested))
        self.open_button.setShortcut("Ctrl+O")
        self.open_button.setToolTip("打开工程 (Ctrl+O)")
        actions.addWidget(self.new_button)
        actions.addWidget(self.open_button)
        actions.addStretch(1)
        self.demo_button = QPushButton("试用示例")
        self.demo_button.setObjectName("LinkButton")
        self.demo_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.demo_button.clicked.connect(lambda: self._emit(self.demo_requested))
        actions.addWidget(self.demo_button)
        layout.addLayout(actions)
        layout.addSpacing(28)

        heading_row = QHBoxLayout()
        self.list_heading = QLabel("最近工程")
        self.list_heading.setObjectName("SectionHeading")
        heading_row.addWidget(self.list_heading)
        heading_row.addStretch(1)
        self.resume_button = QPushButton("返回编辑器")
        self.resume_button.setObjectName("LinkButton")
        self.resume_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.resume_button.clicked.connect(lambda: self._emit(self.resume_requested))
        self.resume_button.hide()
        heading_row.addWidget(self.resume_button)
        layout.addLayout(heading_row)
        layout.addSpacing(8)

        self.recent_list = QListWidget()
        self.recent_list.setObjectName("RecentList")
        self.recent_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.recent_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.recent_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.recent_list.setItemDelegate(_RecentProjectDelegate(self.recent_list))
        self.recent_list.setAccessibleName("最近工程")
        self.recent_list.setSpacing(0)
        self.recent_list.setCursor(Qt.CursorShape.PointingHandCursor)
        self.recent_list.itemActivated.connect(self._activate_recent)
        self.recent_list.currentItemChanged.connect(self._selection_changed)
        layout.addWidget(self.recent_list, 1)

        self.empty_label = QLabel("还没有最近工程\n新建工程，或打开已有的 .stproj 文件。")
        self.empty_label.setObjectName("Tagline")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setWordWrap(True)
        layout.addWidget(self.empty_label, 1)
        layout.addSpacing(10)

        footer = QHBoxLayout()
        self.path_label = QLabel("成功打开或保存的工程会出现在这里。")
        self.path_label.setObjectName("Tagline")
        self.path_label.setTextFormat(Qt.TextFormat.PlainText)
        self.path_label.setWordWrap(True)
        footer.addWidget(self.path_label, 1)
        self.selected_button = self._start_action("open", "打开所选", self._activate_selection)
        self.selected_button.setEnabled(False)
        footer.addWidget(self.selected_button)
        layout.addLayout(footer)
        layout.addSpacing(16)

        self.task_status = QWidget()
        task_inner = QHBoxLayout(self.task_status)
        task_inner.setContentsMargins(0, 0, 0, 0)
        task_inner.setSpacing(12)
        self.job_label = QLabel("就绪")
        self.job_label.setObjectName("Tagline")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.cancel_task_button = QPushButton("取消任务")
        self.cancel_task_button.clicked.connect(self.cancel_task_requested.emit)
        task_inner.addWidget(self.job_label, 0)
        task_inner.addWidget(self.progress, 1)
        task_inner.addWidget(self.cancel_task_button, 0)
        layout.addWidget(self.task_status)
        self.task_status.hide()

        bottom = QHBoxLayout()
        bottom.setSpacing(18)
        guide_button = QPushButton("使用指南")
        guide_button.setObjectName("LinkButton")
        guide_button.clicked.connect(lambda: self.pages.setCurrentIndex(1))
        about_button = QPushButton("关于")
        about_button.setObjectName("LinkButton")
        about_button.clicked.connect(lambda: self.pages.setCurrentIndex(2))
        bottom.addWidget(guide_button)
        bottom.addWidget(about_button)
        bottom.addStretch(1)
        self.version_label = QLabel(f"v{__version__}")
        self.version_label.setObjectName("Tagline")
        bottom.addWidget(self.version_label)
        layout.addLayout(bottom)
        return page

    def _text_page(self, sections: tuple[tuple[str, str], ...], title: str) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        content = QWidget()
        content.setObjectName("TextPage")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(30, 26, 30, 20)
        layout.setSpacing(14)
        back = QPushButton("← 返回")
        back.setObjectName("LinkButton")
        back.setCursor(Qt.CursorShape.PointingHandCursor)
        back.clicked.connect(lambda: self.pages.setCurrentIndex(0))
        layout.addWidget(back)
        heading = QLabel(title)
        heading.setObjectName("PanelTitle")
        heading_font = heading.font()
        heading_font.setPointSize(18)
        heading_font.setBold(True)
        heading.setFont(heading_font)
        layout.addSpacing(6)
        layout.addWidget(heading)
        layout.addSpacing(6)
        for section_title, text in sections:
            section = QLabel(section_title)
            font = section.font()
            font.setBold(True)
            section.setFont(font)
            section.setWordWrap(True)
            body = QLabel(text)
            body.setObjectName("Tagline")
            body.setWordWrap(True)
            layout.addWidget(section)
            layout.addWidget(body)
        layout.addStretch()
        scroll.setWidget(content)
        return scroll

    # --- behavior ---------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._allow_close:
            event.accept()
        else:
            event.ignore()
            self.close_requested.emit()
            # The shared exit handler may approve this very close request.
            event.setAccepted(self._allow_close)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._grip.move(self.width() - 18, self.height() - 18)

    def set_recent_projects(self, paths: list[str]) -> None:
        self.recent_list.clear()
        for path in paths:
            source = Path(path)
            name = source.name
            if not source.is_file():
                name += "  （文件已移动或删除）"
            item = QListWidgetItem()
            item.setText(f"{name}\n{path}")
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(str(path))
            self.recent_list.addItem(item)
        self.empty_label.setVisible(not paths)
        self.recent_list.setVisible(bool(paths))
        if paths:
            self.recent_list.setCurrentRow(0)
        self._selection_changed()

    def set_has_document(self, has_document: bool) -> None:
        self._has_document = has_document
        self.resume_button.setVisible(has_document)
        self.resume_button.setEnabled(has_document and not self._busy)

    def set_busy(self, busy: bool) -> None:
        # The task bar (cancel button included) lives inside the main page,
        # so only the individual action controls are disabled — never the
        # page stack that hosts the running task's progress.
        self._busy = busy
        for control in (self.new_button, self.open_button, self.demo_button,
                        self.selected_button, self.recent_list):
            control.setEnabled(not busy)
        self.resume_button.setEnabled(self._has_document and not busy)
        self._selection_changed()

    def _emit(self, signal) -> None:
        if not self._busy:
            signal.emit()

    def _selection_changed(self, *_args) -> None:
        # Signals may arrive while the page itself is still being built.
        if not hasattr(self, "selected_button"):
            return
        item = self.recent_list.currentItem()
        self.selected_button.setEnabled(item is not None and not self._busy)
        if item is not None:
            self.path_label.setText("双击工程，或按 Enter 打开。")
            self.path_label.setToolTip(item.data(Qt.ItemDataRole.UserRole))
        else:
            self.path_label.setText("成功打开或保存的工程会出现在这里。")
            self.path_label.setToolTip("")

    def _activate_recent(self, item: QListWidgetItem) -> None:
        if not self._busy:
            self.project_requested.emit(item.data(Qt.ItemDataRole.UserRole))

    def _activate_selection(self) -> None:
        if (item := self.recent_list.currentItem()) is not None:
            self._activate_recent(item)
