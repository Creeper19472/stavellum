"""The desktop landing page and its persisted recent-project list."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSettings, QSize, Qt, Signal
from PySide6.QtGui import QCloseEvent, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from stavellum import __version__
from stavellum.graphics.branding import APPLICATION_DESCRIPTION, bind_application_icon, logo_pixmap


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


class _MusicBanner(QWidget):
    """Branded banner with a packaged logo and resolution-independent music art."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._logo = logo_pixmap()
        self.setMinimumHeight(150)
        self.setMaximumHeight(210)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setAccessibleName(f"Stavellum {__version__}，{APPLICATION_DESCRIPTION}")

    def sizeHint(self) -> QSize:
        return QSize(960, 190)

    @staticmethod
    def _note(painter: QPainter, x: float, y: float, up: bool = True) -> None:
        painter.save()
        painter.translate(x, y)
        painter.rotate(-18)
        painter.setBrush(QColor("#183844"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(-7, -4, 14, 8))
        painter.restore()
        painter.setPen(QPen(QColor("#183844"), 2.4))
        direction = -1 if up else 1
        stem_x = x + 6 if up else x - 6
        painter.drawLine(QPointF(stem_x, y), QPointF(stem_x, y + direction * 29))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#fff4e9"))
        # Preserve the shapes' proportions while adapting their horizontal anchors.
        scale = self.height() / 190
        painter.scale(scale, scale)
        view_width = self.width() / scale
        split = view_width * 0.4
        painter.fillRect(QRectF(0, 0, split, 190), QColor("#f1cfaa"))
        ink = QColor("#183844")
        painter.setPen(ink)
        title = QFont("Georgia")
        title.setPixelSize(35)
        painter.setFont(title)
        title_width = painter.fontMetrics().horizontalAdvance("Stavellum")
        title.setPixelSize(max(20, min(35, int(35 * (split - 60) / title_width))))
        painter.setFont(title)
        painter.drawPixmap(QRectF(30, 12, 82, 82), self._logo, QRectF(self._logo.rect()))
        painter.drawText(QRectF(30, 100, split - 60, 42), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "Stavellum")
        subtitle = QFont(self.font())
        subtitle.setPixelSize(17)
        painter.setFont(subtitle)
        caption_width = painter.fontMetrics().horizontalAdvance(APPLICATION_DESCRIPTION)
        subtitle.setPixelSize(max(12, min(17, int(17 * 2 * (split - 64) / caption_width))))
        painter.setFont(subtitle)
        caption = APPLICATION_DESCRIPTION
        if painter.fontMetrics().horizontalAdvance(caption) > split - 64:
            caption = caption.replace("的", "的\n", 1)
        painter.drawText(
            QRectF(32, 143, split - 64, 42),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap,
            caption,
        )
        painter.setPen(QPen(QColor("#183844"), 1.0))
        for index in range(5):
            y = 121 + index * 7
            painter.drawLine(QPointF(split + 32, y), QPointF(view_width - 46, y))
        note_spacing = min(48, (view_width - split - 78) * 0.09)
        for index, y in enumerate((142, 128, 135, 121, 114)):
            x = split + 63 + index * note_spacing
            self._note(painter, x, y)
        # A violin silhouette with scroll, strings, bridge, and a diagonal bow.
        painter.save()
        painter.translate(view_width - 128, 98)
        painter.rotate(22)
        body = QPainterPath()
        body.moveTo(-9, -37)
        body.cubicTo(-34, -37, -39, -12, -20, -7)
        body.cubicTo(-10, -4, -13, 8, -24, 11)
        body.cubicTo(-49, 28, -30, 55, 0, 55)
        body.cubicTo(30, 55, 49, 28, 24, 11)
        body.cubicTo(13, 8, 10, -4, 20, -7)
        body.cubicTo(39, -12, 34, -37, 9, -37)
        body.closeSubpath()
        painter.setPen(QPen(ink, 2))
        painter.setBrush(QColor("#e2ac70"))
        painter.drawPath(body)
        painter.setBrush(ink)
        painter.drawRoundedRect(QRectF(-4, -76, 8, 113), 2, 2)
        painter.drawEllipse(QRectF(-8, -90, 16, 16))
        painter.setPen(QPen(QColor("#fff4e9"), 0.8))
        for x in (-2, 0, 2):
            painter.drawLine(QPointF(x, -78), QPointF(x, 39))
        painter.setPen(QPen(ink, 3))
        painter.drawLine(QPointF(-9, 13), QPointF(9, 13))
        for side in (-1, 1):
            sound_hole = QPainterPath()
            sound_hole.moveTo(side * 15, -3)
            sound_hole.cubicTo(side * 25, -7, side * 9, 27, side * 19, 23)
            painter.drawPath(sound_hole)
        painter.setPen(QPen(ink, 3))
        painter.drawLine(QPointF(40, -81), QPointF(54, 62))
        painter.setPen(QPen(QColor("#ba8956"), 1.2))
        painter.drawLine(QPointF(45, -81), QPointF(59, 62))
        painter.restore()
        painter.setPen(ink)
        subtitle.setPixelSize(13)
        painter.setFont(subtitle)
        painter.drawText(QRectF(view_width - 135, 164, 105, 20), Qt.AlignmentFlag.AlignRight,
                         f"v{__version__}")


class WelcomePage(QWidget):
    """Independent welcome window; MainWindow owns document and application lifetime."""

    new_requested = Signal()
    open_requested = Signal()
    resume_requested = Signal()
    project_requested = Signal(str)
    demo_requested = Signal()
    close_requested = Signal()
    cancel_task_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("欢迎 · Stavellum")
        bind_application_icon(self)
        self.resize(1000, 700)
        self._allow_close = False
        self._busy = False
        self._has_document = False
        self.setObjectName("WelcomePage")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("""
            QWidget#WelcomePage, QWidget#WelcomePage QWidget {
                background: #242c35; color: #e6e9ed;
            }
            QWidget#WelcomePage QLabel, QWidget#WelcomePage QTabWidget,
            QWidget#WelcomePage QListWidget, QWidget#WelcomePage QScrollArea {
                color: #e6e9ed; background: #242c35;
            }
            QWidget#WelcomePage QTabWidget::pane { border: 1px solid #42505b; }
            QWidget#WelcomePage QTabBar::tab {
                background: #242c35; color: #bbc5cf; padding: 10px 20px;
                border-bottom: 3px solid transparent;
            }
            QWidget#WelcomePage QTabBar::tab:selected {
                color: #ffffff; border-bottom: 3px solid #e6b57e;
            }
            QWidget#WelcomePage QListWidget { border: 0; outline: 0; }
            QWidget#WelcomePage QListWidget::item { padding: 12px 10px; }
            QWidget#WelcomePage QListWidget#ProjectCategories::item { padding: 6px 10px; }
            QWidget#WelcomePage QListWidget::item:selected {
                background: #385263; color: #ffffff;
            }
            QWidget#WelcomePage QPushButton {
                color: #e6e9ed; background: #303e49; border: 1px solid #52616b;
                border-radius: 4px; padding: 8px 14px;
            }
            QWidget#WelcomePage QPushButton:hover { background: #425665; }
            QWidget#WelcomePage QPushButton:disabled { color: #7a8893; }
            QWidget#WelcomePage QPushButton#CreateProject {
                color: #202d35; background: #edc398; border-color: #edc398;
            }
            QWidget#WelcomePage QPushButton#CreateProject:hover { background: #f7d6b3; }
            QWidget#WelcomePage QPushButton#CreateProject:disabled {
                color: #697078; background: #938475; border-color: #938475;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 12)
        layout.setSpacing(0)
        self.banner = _MusicBanner(self)
        layout.addWidget(self.banner)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_start_page(), "开始使用")
        self.tabs.addTab(self._text_page((
            ("从来源到谱面", "创建新工程，选择 FLP 或 MIDI；可选原曲音频，再选择自动处理方案。"),
            ("检查来源与时间", "导入默认使用第一个 Arrangement。进入编辑器后可调整编曲、速度、拍号和音频同步；FLP 未确认固定时钟时，需人工核对速度与拍号。"),
            ("整理分谱", "检查建议的乐器分组，在分谱设置中调整音符简化、八度移位和演奏法识别。"),
            ("预览与导出", "生成谱面预览，保存 .stproj 工程，或导出分谱和滚动谱视频。"),
        )), "使用指南")
        self.tabs.addTab(self._text_page((
            (f"Stavellum  {__version__}", APPLICATION_DESCRIPTION),
            ("一个工程，完整保留", "工程保存来源信息、分谱、原曲音频路径和渲染设置，便于继续编辑和再次导出。"),
            ("开源许可", "Stavellum 使用 GNU GPL v3 或更新版本。第三方库与字体保留各自的许可证；完整文本及来源随发行包提供。"),
        )), "关于")
        layout.addWidget(self.tabs, 1)
        buttons = QHBoxLayout()
        buttons.setContentsMargins(12, 12, 12, 0)
        buttons.setSpacing(24)
        self.resume_button = QPushButton("返回编辑器")
        self.resume_button.clicked.connect(lambda: self._emit(self.resume_requested))
        self.resume_button.hide()
        self.open_button = QPushButton("打开工程…")
        self.open_button.clicked.connect(lambda: self._emit(self.open_requested))
        self.new_button = QPushButton("创建新工程")
        self.new_button.setObjectName("CreateProject")
        self.new_button.clicked.connect(lambda: self._emit(self.new_requested))
        buttons.addWidget(self.resume_button)
        buttons.addStretch(1)
        buttons.addWidget(self.open_button)
        buttons.addWidget(self.new_button)
        layout.addLayout(buttons)
        self.task_status = QWidget()
        task_layout = QHBoxLayout(self.task_status)
        self.job_label = QLabel("就绪")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.cancel_task_button = QPushButton("取消任务")
        self.cancel_task_button.clicked.connect(self.cancel_task_requested.emit)
        task_layout.addWidget(self.job_label, 2)
        task_layout.addWidget(self.progress, 1)
        task_layout.addWidget(self.cancel_task_button)
        layout.addWidget(self.task_status)
        self.task_status.hide()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._allow_close:
            event.accept()
        else:
            event.ignore()
            self.close_requested.emit()
            # The shared exit handler may approve this very close request.
            event.setAccepted(self._allow_close)

    def _text_page(self, sections: tuple[tuple[str, str], ...]) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)
        for title, text in sections:
            heading = QLabel(title)
            font = heading.font()
            font.setBold(True)
            heading.setFont(font)
            heading.setWordWrap(True)
            body = QLabel(text)
            body.setWordWrap(True)
            layout.addWidget(heading)
            layout.addWidget(body)
        layout.addStretch()
        scroll.setWidget(content)
        return scroll

    def _build_start_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(8, 12, 14, 12)
        layout.setSpacing(16)
        self.categories = QListWidget()
        self.categories.setObjectName("ProjectCategories")
        self.categories.addItems(["最近工程", "示例工程"])
        self.categories.setFixedWidth(116)
        self.categories.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.categories.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.categories.setAccessibleName("工程分类")
        layout.addWidget(self.categories)
        column = QVBoxLayout()
        self.list_heading = QLabel("最近打开的工程")
        heading_font = self.list_heading.font()
        heading_font.setBold(True)
        self.list_heading.setFont(heading_font)
        column.addWidget(self.list_heading)
        self.lists = QStackedWidget()
        self.lists.setMinimumHeight(80)
        self.lists.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.recent_list = QListWidget()
        self.recent_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.recent_list.setAccessibleName("最近工程")
        self.recent_list.itemActivated.connect(self._activate_recent)
        self.recent_list.currentItemChanged.connect(self._selection_changed)
        recent_page = QWidget()
        recent_layout = QVBoxLayout(recent_page)
        recent_layout.setContentsMargins(0, 0, 0, 0)
        self.empty_label = QLabel("还没有最近工程。\n创建新工程，或打开已有的 .stproj 工程开始。")
        self.empty_label.setWordWrap(True)
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        recent_layout.addWidget(self.empty_label)
        recent_layout.addWidget(self.recent_list, 1)
        self.lists.addWidget(recent_page)
        self.demo_list = QListWidget()
        self.demo_list.addItem("三乐器示例\n体验弦乐、Pizz 奏法与钟琴的完整工程")
        self.demo_list.setWordWrap(True)
        self.demo_list.setAccessibleName("示例工程")
        self.demo_list.itemActivated.connect(lambda _item: self._emit(self.demo_requested))
        self.demo_list.currentItemChanged.connect(self._selection_changed)
        self.demo_list.setCurrentRow(0)
        self.lists.addWidget(self.demo_list)
        column.addWidget(self.lists, 1)
        self.path_label = QLabel("成功打开或保存的工程会出现在这里。")
        self.path_label.setTextFormat(Qt.TextFormat.PlainText)
        self.path_label.setWordWrap(True)
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        column.addWidget(self.path_label)
        self.selected_button = QPushButton("打开所选工程")
        self.selected_button.clicked.connect(self._activate_selection)
        self.selected_button.setEnabled(False)
        column.addWidget(self.selected_button, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addLayout(column, 1)
        self.categories.currentRowChanged.connect(self._category_changed)
        self.categories.setCurrentRow(0)
        return page

    def set_recent_projects(self, paths: list[str]) -> None:
        self.recent_list.clear()
        for path in paths:
            source = Path(path)
            text = source.name
            if not source.is_file():
                text += "  （文件已移动或删除）"
            item = QListWidgetItem(text)
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
        self._busy = busy
        self.tabs.setEnabled(not busy)
        self.new_button.setEnabled(not busy)
        self.open_button.setEnabled(not busy)
        self.resume_button.setEnabled(self._has_document and not busy)
        self._selection_changed()

    def _emit(self, signal) -> None:
        if not self._busy:
            signal.emit()

    def _category_changed(self, row: int) -> None:
        self.lists.setCurrentIndex(max(0, row))
        self.list_heading.setText("示例工程" if row == 1 else "最近打开的工程")
        self._selection_changed()

    def _selection_changed(self, *_args) -> None:
        # Signals may arrive while the page itself is still being built.
        if not hasattr(self, "selected_button"):
            return
        demo = self.lists.currentIndex() == 1
        item = (self.demo_list if demo else self.recent_list).currentItem()
        self.selected_button.setEnabled(item is not None and not self._busy)
        self.selected_button.setText("打开示例工程" if demo else "打开所选工程")
        if demo:
            self.path_label.setText("自动生成示例工程、MIDI 和合成音频。")
        elif item is not None:
            self.path_label.setText(item.data(Qt.ItemDataRole.UserRole))
        else:
            self.path_label.setText("成功打开或保存的工程会出现在这里。")

    def _activate_recent(self, item: QListWidgetItem) -> None:
        if not self._busy:
            self.project_requested.emit(item.data(Qt.ItemDataRole.UserRole))

    def _activate_selection(self) -> None:
        if self.lists.currentIndex() == 1:
            self._emit(self.demo_requested)
        elif (item := self.recent_list.currentItem()) is not None:
            self._activate_recent(item)
