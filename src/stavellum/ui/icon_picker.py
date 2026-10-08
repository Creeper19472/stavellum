"""Offline icon selection with cancellable indexing and visible-item thumbnails."""

from __future__ import annotations

from collections import OrderedDict

from PySide6.QtCore import QAbstractListModel, QModelIndex, QSettings, QSize, Qt, QThread, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from stavellum.domain.models import IconAsset
from stavellum.graphics.icons import (
    ICON_RESOURCES,
    IMAGE_FILE_FILTER,
    IconEntry,
    free_catalog,
    icon_thumbnail,
    import_icon,
    scan_fontawesome,
)

LIBRARIES_KEY = "icons/fontawesome_directories"
# Keep interrupted workers alive until cooperative cancellation finishes, even
# when the modal dialog has already closed. They never touch widgets or Qt images.
_SCANS: set[QThread] = set()


class LibraryScan(QThread):
    indexed = Signal(object, object)

    def __init__(self, directories: list[str]):
        super().__init__()
        self.directories = directories

    def run(self) -> None:
        entries, errors = [], []
        for directory in self.directories:
            if self.isInterruptionRequested():
                return
            try:
                entries.extend(scan_fontawesome(directory, self.isInterruptionRequested))
            except (ValueError, OSError) as exc:
                errors.append(str(exc))
        if not self.isInterruptionRequested():
            self.indexed.emit(entries, errors)


class IconListModel(QAbstractListModel):
    """QListView requests decoration only for visible rows; cache is bounded."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.entries: list[IconEntry] = []
        self.thumbnails: OrderedDict[IconEntry, QIcon] = OrderedDict()

    def set_entries(self, entries):
        self.beginResetModel()
        self.entries = list(entries)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.entries)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.entries):
            return None
        entry = self.entries[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return f"{entry.name}\n{entry.style}"
        if role == Qt.ItemDataRole.ToolTipRole:
            return f"{entry.name} · {entry.style}\n{entry.source}"
        if role == Qt.ItemDataRole.SizeHintRole:
            return QSize(160, 128)
        if role == Qt.ItemDataRole.DecorationRole:
            if entry not in self.thumbnails:
                try:
                    reference, asset = entry.asset()
                    image = icon_thumbnail(reference, {reference[6:]: asset})
                    thumbnail = QIcon(QPixmap.fromImage(image))
                except (ValueError, OSError, RuntimeError):
                    thumbnail = QIcon()
                self.thumbnails[entry] = thumbnail
                if len(self.thumbnails) > 128:
                    self.thumbnails.popitem(last=False)
            self.thumbnails.move_to_end(entry)
            return self.thumbnails[entry]
        return None


class IconPicker(QDialog):
    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择乐器图标")
        self.resize(760, 570)
        self.settings = settings
        self.selection: tuple[str, IconAsset | None] | None = None
        self._closing = False
        self._thread: LibraryScan | None = None
        self._local_entries: list[IconEntry] = []
        self._free_entries = list(free_catalog())
        raw = settings.value(LIBRARIES_KEY, [])
        self.directories = list(dict.fromkeys(x for x in raw if isinstance(x, str))) if isinstance(raw, list) else []
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        builtin = QWidget()
        builtin_layout = QVBoxLayout(builtin)
        self.builtin = QListWidget()
        self.builtin.setViewMode(QListView.ViewMode.IconMode)
        self.builtin.setResizeMode(QListView.ResizeMode.Adjust)
        self.builtin.setMovement(QListView.Movement.Static)
        self.builtin.setIconSize(QSize(48, 48))
        self.builtin.setGridSize(QSize(138, 104))
        self.builtin.setStyleSheet("QListWidget { background: #171717; color: white; }")
        for kind, label in (("violin", "小提琴 / 弦乐"), ("piano", "钢琴"),
                            ("keyboard", "键盘"), ("drum", "鼓 / 打击乐"), ("bell", "铃")):
            item = QListWidgetItem(QIcon(QPixmap.fromImage(icon_thumbnail(kind))), label)
            item.setData(Qt.ItemDataRole.UserRole, kind)
            item.setSizeHint(QSize(138, 104))
            self.builtin.addItem(item)
        self.builtin.setCurrentRow(0)
        self.builtin.itemDoubleClicked.connect(self._choose)
        builtin_layout.addWidget(self.builtin)
        self.tabs.addTab(builtin, "项目内置")

        custom = QWidget()
        custom_layout = QVBoxLayout(custom)
        custom_layout.addWidget(QLabel("导入 SVG、PNG、JPEG 或 WebP。图片保留原颜色和透明度，并随项目保存。"))
        self.import_button = QPushButton("选择图片…")
        self.import_button.clicked.connect(self._import_file)
        custom_layout.addWidget(self.import_button)
        self.custom_preview = QLabel()
        self.custom_preview.setMinimumHeight(100)
        self.custom_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.custom_preview.setStyleSheet("background: #171717;")
        custom_layout.addWidget(self.custom_preview)
        self.custom_name = QLabel("尚未选择图片")
        custom_layout.addWidget(self.custom_name)
        custom_layout.addStretch()
        self._custom_selection = None
        self.tabs.addTab(custom, "导入图片")

        fa = QWidget()
        fa_layout = QVBoxLayout(fa)
        filters = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("按图标名称搜索，例如 guitar、music")
        self.source = QComboBox()
        self.style = QComboBox()
        self.source.setMaximumWidth(220)
        self.style.setMaximumWidth(160)
        filters.addWidget(self.search, 1)
        filters.addWidget(self.source)
        filters.addWidget(self.style)
        fa_layout.addLayout(filters)
        self.model = IconListModel(self)
        self.view = QListView()
        self.view.setModel(self.model)
        self.view.setViewMode(QListView.ViewMode.IconMode)
        self.view.setResizeMode(QListView.ResizeMode.Adjust)
        self.view.setMovement(QListView.Movement.Static)
        self.view.setIconSize(QSize(48, 48))
        self.view.setGridSize(QSize(160, 128))
        self.view.setWordWrap(True)
        self.view.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.view.setUniformItemSizes(True)
        self.view.setStyleSheet("QListView { background: #171717; color: white; }")
        self.view.doubleClicked.connect(self._choose)
        fa_layout.addWidget(self.view, 1)
        controls = QHBoxLayout()
        self.add_library = QPushButton("连接 Free / Pro 图标库…")
        self.remove_library = QPushButton("移除此图标库")
        self.rescan = QPushButton("重新扫描")
        self.cancel_scan = QPushButton("取消扫描")
        for button in (self.add_library, self.remove_library, self.rescan, self.cancel_scan):
            controls.addWidget(button)
        self.add_library.clicked.connect(self._add_library)
        self.remove_library.clicked.connect(self._remove_library)
        self.rescan.clicked.connect(self._scan)
        self.cancel_scan.clicked.connect(self._cancel_scan)
        self.cancel_scan.setEnabled(False)
        fa_layout.addLayout(controls)
        self.tabs.addTab(fa, "Font Awesome")

        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.buttons.accepted.connect(self._choose)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.search.textChanged.connect(self._filter)
        self.source.currentIndexChanged.connect(self._filter)
        self.style.currentIndexChanged.connect(self._filter)
        self._refresh_filters()
        if self.directories:
            self._scan()

    def _refresh_filters(self):
        old_source, old_style = self.source.currentData(), self.style.currentData()
        self.source.blockSignals(True)
        self.style.blockSignals(True)
        self.source.clear()
        self.source.addItem("所有来源", None)
        self.source.addItem("内置 Free 7.3.1", "Free 7.3.1")
        for directory in self.directories:
            self.source.addItem(directory, directory)
        self.style.clear()
        self.style.addItem("所有样式", None)
        for style in sorted({entry.style for entry in self._free_entries + self._local_entries}):
            self.style.addItem(style, style)
        self.source.setCurrentIndex(max(0, self.source.findData(old_source)))
        self.style.setCurrentIndex(max(0, self.style.findData(old_style)))
        self.source.blockSignals(False)
        self.style.blockSignals(False)
        self._filter()

    def _filter(self, *args):
        query = self.search.text().strip().casefold().removeprefix("fa-")
        source, style = self.source.currentData(), self.style.currentData()
        self.model.set_entries(entry for entry in self._free_entries + self._local_entries
                               if (not source or entry.source == source) and (not style or entry.style == style)
                               and all(word in entry.name.casefold() for word in query.split()))
        self.remove_library.setEnabled(source in self.directories)

    def _import_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图标图片", "", IMAGE_FILE_FILTER)
        if not path:
            return
        try:
            selection = import_icon(path)
            reference, asset = selection
            image = icon_thumbnail(reference, {reference[6:]: asset}, 96)
        except (ValueError, OSError) as exc:
            self.status.setText(str(exc))
            return
        self._custom_selection = selection
        self.custom_preview.setPixmap(QPixmap.fromImage(image))
        self.custom_name.setText(asset.name)
        self.status.clear()

    def _remember_directories(self):
        self.settings.setValue(LIBRARIES_KEY, self.directories)
        self.settings.sync()

    def _add_library(self):
        directory = QFileDialog.getExistingDirectory(self, "选择已解压的 Font Awesome 图标库")
        if directory and directory not in self.directories:
            self.directories.append(directory)
            self._remember_directories()
            self._refresh_filters()
            self._scan()

    def _remove_library(self):
        directory = self.source.currentData()
        if directory in self.directories:
            self.directories.remove(directory)
            self._local_entries = [entry for entry in self._local_entries if entry.source != directory]
            self._remember_directories()
            self._refresh_filters()
            self._scan()

    def _scan(self):
        self._cancel_scan()
        if not self.directories:
            self._local_entries = []
            self._refresh_filters()
            self.status.clear()
            return
        worker = LibraryScan(self.directories.copy())
        self._thread = worker
        _SCANS.add(worker)
        worker.indexed.connect(self._indexed)
        worker.finished.connect(lambda: _SCANS.discard(worker))
        worker.finished.connect(worker.deleteLater)
        self.cancel_scan.setEnabled(True)
        self.status.setText("正在扫描本地图标库…")
        worker.start()

    def _cancel_scan(self):
        if self._thread is not None:
            self._thread.requestInterruption()
            self._thread = None
            self.status.setText("图标库扫描已取消。")
        self.cancel_scan.setEnabled(False)

    def _indexed(self, entries, errors):
        if self._closing or self.sender() is not self._thread:
            return
        self._thread = None
        self.cancel_scan.setEnabled(False)
        self._local_entries = entries
        self.model.thumbnails.clear()
        self._refresh_filters()
        self.status.setText("\n".join(errors) if errors else f"已加载 {len(entries)} 个本地图标。")

    def _choose(self, *args):
        try:
            if self.tabs.currentIndex() == 0:
                kind = self.builtin.currentItem().data(Qt.ItemDataRole.UserRole)
                # Keep legacy builtin names; original pictograms are application resources.
                if kind not in ICON_RESOURCES:
                    return
                self.selection = (kind, None)
            elif self.tabs.currentIndex() == 1:
                if self._custom_selection is None:
                    self.status.setText("请先选择图片。")
                    return
                self.selection = self._custom_selection
            else:
                index = self.view.currentIndex()
                if not index.isValid():
                    self.status.setText("请选择一个 Font Awesome 图标。")
                    return
                self.selection = self.model.entries[index.row()].asset()
        except (ValueError, OSError) as exc:
            self.status.setText(str(exc))
            return
        self.accept()

    def done(self, result):
        self._closing = True
        self._cancel_scan()
        super().done(result)
