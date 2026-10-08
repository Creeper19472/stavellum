"""Shared desktop colors and controls, independent of score rendering typography."""

from importlib.resources import files

from PySide6.QtGui import QColor, QIcon, QPalette
from PySide6.QtWidgets import QWidget

BACKGROUND = "#10161e"
SURFACE = "#19222d"
BORDER = "#303d4d"
TEXT = "#edf2f7"
MUTED = "#a4b2c3"
ACCENT = "#edc398"

DESKTOP_STYLESHEET = """
    QWidget { color: #edf2f7; font-size: 13px; }
    QMainWindow, QDialog, QWidget#ProjectWizard { background: #10161e; }
    QWidget#SettingsContent, QWidget#WizardStep, QTabWidget > QWidget,
    QWidget#qt_scrollarea_viewport { background: #10161e; }
    QWidget#TransportBar, QWidget#WorkflowCard {
        background: #19222d; border: 1px solid #303d4d; border-radius: 12px;
    }
    QLabel { background: transparent; }
    QLabel#Muted, QLabel#Tagline, QLabel#WizardSubtitle,
    QLabel#WizardDescription { color: #a4b2c3; }
    QLabel#Eyebrow { color: #edc398; font-size: 11px; font-weight: 600; }
    QLabel#HeroTitle { font-size: 28px; font-weight: 600; }
    QLabel#BrandTitle { font-size: 27px; font-weight: 600; }
    QLabel#PanelTitle { font-size: 16px; font-weight: 600; }
    QLabel#VersionChip, QLabel#FormatChip {
        color: #edc398; background: #283039; border: 1px solid #48505a;
        border-radius: 8px; padding: 3px 9px; font-size: 11px;
    }
    QPushButton, QToolButton {
        background: #222e3c; border: 1px solid #3a485a; border-radius: 7px;
        padding: 7px 12px; color: #edf2f7;
    }
    QPushButton:hover, QToolButton:hover { background: #2b3a4c; border-color: #718197; }
    QPushButton:pressed, QToolButton:pressed { background: #34465a; }
    QPushButton:focus, QToolButton:focus { border-color: #edc398; }
    QPushButton:disabled, QToolButton:disabled { color: #768496; background: #19222d; border-color: #303d4d; }
    QToolButton#IconButton { background: transparent; border: 1px solid transparent; padding: 5px; border-radius: 6px; }
    QToolButton#IconButton:hover { background: #222e3c; border-color: #3a485a; }
    QToolButton#IconButton:pressed { background: #34465a; }
    QToolButton#IconButton:focus { border-color: #edc398; }
    QToolButton#IconButton:disabled { background: transparent; border-color: transparent; }
    QPushButton#PrimaryButton, QPushButton#CreateProject, QPushButton#WizardPrimary,
    QToolButton#PrimaryButton {
        background: #edc398; color: #18212b; border-color: #edc398; font-weight: 600;
    }
    QPushButton#PrimaryButton:hover, QPushButton#CreateProject:hover,
    QPushButton#WizardPrimary:hover, QToolButton#PrimaryButton:hover { background: #f7d6b3; }
    QPushButton#PrimaryButton:disabled, QPushButton#CreateProject:disabled,
    QPushButton#WizardPrimary:disabled, QToolButton#PrimaryButton:disabled {
        background: #4a4237; color: #b1a18e; border-color: #4a4237;
    }
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit,
    QTextBrowser, QListWidget {
        background: #121a24; color: #edf2f7; border: 1px solid #3a485a;
        border-radius: 6px; padding: 5px; selection-background-color: #3d526b;
        selection-color: #ffffff;
    }
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
    QPlainTextEdit:focus, QListWidget:focus { border-color: #edc398; }
    QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
    QComboBox:disabled { color: #768496; border-color: #303d4d; }
    QComboBox { padding-right: 22px; }
    QComboBox::drop-down { border: none; width: 22px; }
    QComboBox::down-arrow, QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
        image: url("CHEVRON_DOWN"); width: 12px; height: 12px;
    }
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
        image: url("CHEVRON_UP"); width: 12px; height: 12px;
    }
    QSpinBox::up-button, QDoubleSpinBox::up-button {
        subcontrol-origin: border; subcontrol-position: top right; width: 20px;
        border: none; background: #222e3c; border-top-right-radius: 5px;
    }
    QSpinBox::down-button, QDoubleSpinBox::down-button {
        subcontrol-origin: border; subcontrol-position: bottom right; width: 20px;
        border: none; background: #222e3c; border-bottom-right-radius: 5px;
    }
    QSpinBox, QDoubleSpinBox { padding-right: 23px; }
    QComboBox QAbstractItemView { background: #19222d; selection-background-color: #3d526b; }
    QGroupBox {
        background: #19222d; border: 1px solid #303d4d; border-radius: 9px;
        margin-top: 15px; padding: 17px 10px 10px; font-weight: 600;
    }
    QGroupBox::title { subcontrol-origin: margin; left: 13px; padding: 0 5px; color: #edc398; }
    QTabWidget::pane { border: 1px solid #303d4d; border-radius: 9px; top: -1px; }
    QTabBar::tab {
        background: #19222d; color: #a4b2c3; padding: 11px 13px;
        border-bottom: 2px solid transparent;
    }
    QTabBar::tab:selected { color: #edc398; background: #222e3c; border-bottom-color: #edc398; }
    QTabBar::tab:hover { color: #edf2f7; background: #222e3c; }
    QScrollArea { background: transparent; border: none; }
    QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
    QScrollBar::handle:vertical { background: #43536a; border-radius: 3px; min-height: 28px; }
    QScrollBar::handle:vertical:hover { background: #718197; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
    QSplitter::handle { background: transparent; width: 12px; }
    QSplitter::handle:hover { background: #303d4d; }
    QCheckBox { spacing: 8px; background: transparent; }
    QCheckBox:disabled { color: #768496; }
    QCheckBox::indicator { width: 16px; height: 16px; }
    QListWidget::item { padding: 5px; border-radius: 4px; }
    QListWidget::item:selected { background: #3d526b; color: #ffffff; }
    QSlider::groove:horizontal { height: 5px; background: #354358; border-radius: 2px; }
    QSlider::sub-page:horizontal { background: #edc398; border-radius: 2px; }
    QSlider::handle:horizontal { width: 14px; margin: -5px 0; border-radius: 7px; background: #edc398; }
    QSlider:disabled::handle:horizontal, QSlider:disabled::sub-page:horizontal { background: #768496; }
    QProgressBar { border: none; background: #303d4d; border-radius: 3px; max-height: 6px; }
    QProgressBar::chunk { background: #edc398; border-radius: 3px; }
    QMenuBar { background: #10161e; padding: 3px 10px; }
    QMenuBar::item { padding: 5px 10px; background: transparent; }
    QMenuBar::item:selected { background: #2b3a4c; border-radius: 4px; }
    QMenu { background: #19222d; border: 1px solid #3a485a; padding: 5px; }
    QMenu::item { padding: 7px 26px; }
    QMenu::item:selected { background: #3d526b; }
    QMenu::item:disabled { color: #768496; }
    QMenu::separator { height: 1px; background: #303d4d; margin: 4px 8px; }
    QStatusBar { color: #a4b2c3; background: #10161e; }
    QToolTip { background: #222e3c; color: #edf2f7; border: 1px solid #718197; padding: 6px; }
    QFrame#WizardHeader { background: #19222d; border: 1px solid #303d4d; border-radius: 12px; }
    QLabel#WizardTitle { color: #edf2f7; font-size: 26px; font-weight: 600; }
    QLabel#StepTitle { color: #edc398; font-size: 20px; font-weight: 600; }
    QLabel#WizardError { color: #ffc2b2; }
    QLabel#WizardSummary { background: #19222d; border: 1px solid #303d4d; border-radius: 9px; padding: 14px; }
""".replace(
    "CHEVRON_UP", str(files("stavellum").joinpath("assets", "chevron-up.svg")).replace("\\", "/")
).replace(
    "CHEVRON_DOWN", str(files("stavellum").joinpath("assets", "chevron-down.svg")).replace("\\", "/")
)


def apply_desktop_theme(widget: QWidget) -> None:
    """Theme one desktop window and its children without changing rendered frames."""
    palette = widget.palette()
    for role, color in (
        (QPalette.ColorRole.Window, BACKGROUND),
        (QPalette.ColorRole.WindowText, TEXT),
        (QPalette.ColorRole.Base, "#121a24"),
        (QPalette.ColorRole.AlternateBase, SURFACE),
        (QPalette.ColorRole.Text, TEXT),
        (QPalette.ColorRole.Button, SURFACE),
        (QPalette.ColorRole.ButtonText, TEXT),
        (QPalette.ColorRole.Highlight, "#3d526b"),
        (QPalette.ColorRole.HighlightedText, "#ffffff"),
        (QPalette.ColorRole.PlaceholderText, MUTED),
    ):
        palette.setColor(role, QColor(color))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#768496"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#768496"))
    widget.setPalette(palette)
    widget.setStyleSheet(DESKTOP_STYLESHEET)


def desktop_icon(name: str) -> QIcon:
    """Load a small vector action icon that remains sharp at desktop scale factors."""
    return QIcon(str(files("stavellum").joinpath("assets", f"ui-{name}.svg")))
