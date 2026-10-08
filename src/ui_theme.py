#!/usr/bin/env python
"""
Shared design system for the Subtitle Maker desktop apps.

Both :mod:`src.gui` and :mod:`src.translate_gui` pull their colours, spacing
and control painting from here so the two windows can never drift apart.

Design direction — "editing desk":
    Paper-white surfaces, one amber accent, hairline separators instead of
    nested cards, and a single flat action bar. The queue is the only raised
    surface, which makes the work area read as primary and every setting as
    secondary.

    Amber is split in two on purpose: :data:`ACCENT` is a bright fill meant to
    carry dark ink, :data:`ACCENT_TEXT` is the darker variant meant to be read
    as text or used as a border on white. One value cannot do both jobs.

Qt stylesheets cannot draw a tick mark or a chevron without shipping image
assets, so :class:`CheckBox`, :class:`ChevronComboBox` and :class:`ArrowSpinBox`
paint those glyphs themselves.
"""
from __future__ import annotations

import re

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPolygon,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSizePolicy,
    QStyleOptionButton,
    QVBoxLayout,
    QWidget,
)


# --------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------

BG = "#eef0f3"           # window: a touch cooler than SURFACE so panels lift
SURFACE = "#ffffff"      # primary raised panel (queue)
SURFACE_ALT = "#fbfbfc"  # inputs
SURFACE_HOVER = "#f2f4f7"
BORDER = "#dfe2e8"       # hairline
BORDER_STRONG = "#c2c7d0"

TEXT = "#1e2128"
TEXT_DIM = "#5b6270"
TEXT_FAINT = "#8b929e"

# Amber needs two values on a light ground: FILL is bright enough to sit under
# dark ink, INK is dark enough to stay readable as text on white. Reusing one
# value for both is what makes light themes look washed out.
ACCENT = "#f0a83c"       # fills: primary button, progress, checkbox tick
ACCENT_HOVER = "#f7b755"
ACCENT_INK = "#1e2128"   # text drawn on ACCENT
ACCENT_TEXT = "#a86f10"  # amber as text / borders on light backgrounds
ACCENT_SOFT = "#fdf4e3"  # amber-tinted fill (badge)
ACCENT_SOFT_BORDER = "#f0dfba"

DANGER = "#cf4238"
DANGER_TEXT = "#b03a30"
SUCCESS = "#1f9d5f"
INFO = "#2b7fc7"
WARN = "#b07d18"
NEUTRAL_SOFT = "#eef0f3"
NEUTRAL_TEXT = "#5b6270"
# Disabled controls keep >=3:1 so an off button still reads as a button.
# Both values verified with the WCAG relative-luminance formula.
DISABLED_FILL = "#eceef1"
DISABLED_FILL_ACCENT = "#f0dfba"
DISABLED_INK = "#6f7683"
DISABLED_INK_STRONG = "#7a818e"   # hand-painted glyphs when a control is off

UI_FONT = '"Segoe UI Variable Text", "Segoe UI", "Microsoft YaHei UI"'
MONO_FONT = '"Cascadia Mono", Consolas, "Courier New"'


APP_STYLE = f"""
QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: {UI_FONT};
    font-size: 10pt;
}}
QMainWindow, QDialog {{ background: {BG}; }}
QToolTip {{
    background: {SURFACE_ALT};
    color: {TEXT};
    border: 1px solid {BORDER_STRONG};
    padding: 5px 8px;
}}

/* ---- text roles ---- */
QLabel {{ background: transparent; }}
QLabel#appTitle {{ font-size: 15pt; font-weight: 700; color: {TEXT}; }}
QLabel#appSubtitle {{ font-size: 9pt; color: {TEXT_DIM}; }}
QLabel#sectionLabel {{
    font-size: 9pt;
    font-weight: 700;
    color: {TEXT_DIM};
    letter-spacing: 1px;
}}
QLabel#fieldLabel {{ font-size: 9pt; color: {TEXT_DIM}; }}
QLabel#dim {{ font-size: 9pt; color: {TEXT_DIM}; }}
QLabel#faint {{ font-size: 9pt; color: {TEXT_FAINT}; }}
QLabel#status {{ font-size: 10pt; font-weight: 600; color: {TEXT}; }}
QLabel#meterValue {{ font-family: {MONO_FONT}; font-size: 9pt; color: {TEXT}; }}
QLabel#badge {{
    font-family: {MONO_FONT};
    font-size: 8pt;
    font-weight: 700;
    color: {ACCENT_TEXT};
    background: {ACCENT_SOFT};
    border: 1px solid {ACCENT_SOFT_BORDER};
    border-radius: 9px;
    padding: 3px 9px;
}}
QLabel#dropTitle {{ font-size: 12pt; font-weight: 700; color: {TEXT}; }}
QLabel#dropHint {{ font-size: 9pt; color: {TEXT_FAINT}; }}
QLabel#emptyState {{ font-size: 10pt; color: {TEXT_FAINT}; }}
QLabel#link {{ font-size: 9pt; color: {TEXT_DIM}; padding: 4px 6px; }}
QLabel#link:hover {{ color: {ACCENT_TEXT}; }}
QLabel#logTitle {{ font-size: 9pt; font-weight: 700; color: {TEXT_FAINT}; }}

/* ---- structure ---- */
QFrame#panel {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
QFrame#hairline {{
    background: {BORDER};
    border: 0;
    max-height: 1px;
    min-height: 1px;
}}
QFrame#vline {{
    background: {BORDER};
    border: 0;
    max-width: 1px;
    min-width: 1px;
}}
QFrame#dropZone {{
    background: #f9fafc;
    border: 1px dashed {BORDER_STRONG};
    border-radius: 10px;
}}
QFrame#dropZone:hover {{ border-color: {ACCENT_TEXT}; background: {ACCENT_SOFT}; }}

/* ---- inputs ---- */
QLineEdit, QComboBox, QSpinBox {{
    background: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 6px 10px;
    min-height: 18px;
    color: {TEXT};
    selection-background-color: {ACCENT};
    selection-color: {ACCENT_INK};
}}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover {{ border-color: {BORDER_STRONG}; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{ border: 1px solid {ACCENT_TEXT}; }}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {{
    background: #f3f4f7;
    color: {TEXT_FAINT};
    border-color: {BORDER};
}}
QLineEdit::placeholder {{ color: {TEXT_FAINT}; }}

QComboBox {{ padding-right: 26px; }}
QComboBox::drop-down {{ border: 0; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {SURFACE_ALT};
    border: 1px solid {BORDER_STRONG};
    border-radius: 8px;
    padding: 4px;
    outline: 0;
    selection-background-color: {ACCENT_SOFT};
    selection-color: {TEXT};
}}

QSpinBox {{ padding-right: 26px; }}
QSpinBox::up-button, QSpinBox::down-button {{
    subcontrol-origin: border;
    width: 20px;
    background: transparent;
    border: 0;
}}
QSpinBox::up-button {{ subcontrol-position: top right; margin: 1px 2px 0 0; }}
QSpinBox::down-button {{ subcontrol-position: bottom right; margin: 0 2px 1px 0; }}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
    background: {SURFACE_HOVER};
    border-radius: 4px;
}}

/* ---- buttons ---- */
QPushButton {{
    background: {SURFACE};
    border: 1px solid {BORDER_STRONG};
    border-radius: 7px;
    padding: 7px 14px;
    color: {TEXT};
}}
QPushButton:hover {{ background: {SURFACE_HOVER}; border-color: #aeb4be; }}
QPushButton:pressed {{ background: #e8eaee; }}
QPushButton:disabled {{
    background: {DISABLED_FILL};
    color: {DISABLED_INK};
    border-color: {BORDER};
}}

QPushButton#primary {{
    background: {ACCENT};
    border: 1px solid #dd9a2c;
    color: {ACCENT_INK};
    font-weight: 700;
    padding: 9px 28px;
}}
QPushButton#primary:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QPushButton#primary:pressed {{ background: #dd9a2c; }}
QPushButton#primary:disabled {{
    background: {DISABLED_FILL_ACCENT};
    border-color: {DISABLED_FILL_ACCENT};
    color: {DISABLED_INK};
}}
QPushButton#danger {{
    background: transparent;
    border: 1px solid #e3b4b0;
    color: {DANGER_TEXT};
    padding: 9px 20px;
}}
QPushButton#danger:hover {{ background: #fdf0ef; border-color: {DANGER}; }}
QPushButton#ghost {{
    background: transparent;
    border: 1px solid {BORDER};
    color: {TEXT_DIM};
    padding: 7px 12px;
}}
QPushButton#ghost:hover {{
    background: {SURFACE_ALT};
    color: {TEXT};
    border-color: {BORDER_STRONG};
}}

/* ---- checkboxes ---- */
QCheckBox {{ spacing: 8px; color: {TEXT_DIM}; background: transparent; }}
QCheckBox:hover {{ color: {TEXT}; }}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 5px;
    border: 1px solid {BORDER_STRONG};
    background: {SURFACE};
}}
QCheckBox::indicator:hover {{ border-color: {ACCENT_TEXT}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: #dd9a2c; }}
QCheckBox:disabled {{ color: {TEXT_FAINT}; }}
QCheckBox:disabled::indicator {{ background: {DISABLED_FILL}; border-color: {BORDER}; }}

/* ---- table ---- */
QTableWidget {{
    background: {SURFACE};
    alternate-background-color: #fafbfc;
    border: 0;
    gridline-color: {BORDER};
    selection-background-color: {ACCENT_SOFT};
    selection-color: {TEXT};
    outline: 0;
}}
QTableWidget::item {{ padding: 0px 8px; }}
QTableWidget::item:selected {{ background: {ACCENT_SOFT}; }}
QHeaderView {{ background: transparent; }}
QHeaderView::section {{
    background: transparent;
    color: {TEXT_FAINT};
    border: 0;
    border-bottom: 1px solid {BORDER};
    padding: 5px 8px;
    font-size: 9pt;
    font-weight: 700;
}}
QTableCornerButton::section {{ background: transparent; border: 0; }}

/* ---- meters ---- */
QProgressBar {{
    background: #e7e9ee;
    border: 0;
    border-radius: 3px;
    text-align: center;
    color: transparent;
    min-height: 6px;
    max-height: 6px;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QProgressBar#edge {{
    background: #e7e9ee;
    border-radius: 0;
    min-height: 3px;
    max-height: 3px;
}}
QProgressBar#edge::chunk {{ background: {ACCENT}; border-radius: 0; }}

/* ---- terminal ---- */
QTextEdit {{
    background: #fafbfc;
    border: 1px solid {BORDER};
    border-radius: 8px;
    color: {TEXT_DIM};
    selection-background-color: {ACCENT};
    selection-color: {ACCENT_INK};
    padding: 8px 10px;
}}

/* ---- scrollbars ---- */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{
    background: #c8ccd4;
    border-radius: 4px;
    min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{ background: #aeb4be; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{
    background: #c8ccd4;
    border-radius: 4px;
    min-width: 28px;
}}
QScrollBar::handle:horizontal:hover {{ background: #aeb4be; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; border: 0; background: none; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
QScrollArea {{ border: 0; background: transparent; }}
"""


# --------------------------------------------------------------------------
# Structure helpers
# --------------------------------------------------------------------------

def hairline(vertical: bool = False) -> QFrame:
    """A 1px rule — used instead of nesting one bordered box inside another."""
    rule = QFrame()
    rule.setObjectName("vline" if vertical else "hairline")
    if vertical:
        rule.setFixedWidth(1)
    else:
        rule.setFixedHeight(1)
    return rule


def section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("sectionLabel")
    return label


def field_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("fieldLabel")
    return label


def panel() -> tuple[QFrame, QVBoxLayout]:
    """The one raised surface in the window; rhythm lives here, nowhere else."""
    frame = QFrame()
    frame.setObjectName("panel")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(10)
    return frame, layout


def section(title: str) -> tuple[QWidget, QVBoxLayout]:
    """A borderless group: small-caps label, hairline, then content.

    Replaces the old "every group is a bordered card" pattern so a long
    settings column reads as one continuous list.
    """
    holder = QWidget()
    layout = QVBoxLayout(holder)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    layout.addWidget(section_label(title))
    layout.addWidget(hairline())
    return holder, layout


def inline(widget: QWidget) -> QWidget:
    """Keep a checkbox at its natural width inside a stretched column.

    Added straight to a QVBoxLayout, a checkbox expands to the full row and
    reads as a selectable banner instead of a checkbox.
    """
    widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return widget


def status_colors(text: str) -> tuple[str, str]:
    """(background, foreground) for a queue status cell.

    On a light ground the fill is only a shade off the row colour — a column
    of tinted chips reads as a heatmap and drowns the file names. The text
    colour carries the status; the fill only groups it.
    """
    if text.startswith("完成") or text == "已有字幕":
        return "#e9f4ed", SUCCESS
    if text in {"处理中", "翻译中"}:
        return "#eaf1f8", INFO
    if "失败" in text:
        return "#fbeceb", DANGER
    if text in {"已停止", "已取消"}:
        return "#f9f2e4", WARN
    return NEUTRAL_SOFT, NEUTRAL_TEXT


LOG_SUCCESS = "#158a4f"
LOG_FAILURE = "#b5372c"
LOG_WARNING = "#9a6c12"
LOG_NEUTRAL = "#5b6270"
SUCCESS_LOG_HEX = LOG_SUCCESS
FAILURE_LOG_HEX = LOG_FAILURE


def log_level_color(message: str) -> str:
    """Terminal colour for a log line."""
    lowered = message.lower()
    if "失败" in message or "error" in lowered:
        return LOG_FAILURE
    failed_match = re.search(r"\bfailed\s*[:=]\s*(\d+)\b|\b(\d+)\s+failed\b", lowered)
    if "failed" in lowered and (
        failed_match is None
        or int(next(group for group in failed_match.groups() if group)) > 0
    ):
        return LOG_FAILURE
    if "完成" in message or "success" in lowered or "已翻译" in message:
        return LOG_SUCCESS
    if "警告" in message or "warn" in lowered:
        return LOG_WARNING
    return LOG_NEUTRAL


# --------------------------------------------------------------------------
# Controls that need hand-painted glyphs
# --------------------------------------------------------------------------

class CheckBox(QCheckBox):
    """Checkbox with a real tick mark.

    A stylesheet can only fill the indicator box; the tick is drawn on top so
    the checked state is unambiguous on the dark surface.
    """

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API naming
        super().paintEvent(event)
        if not self.isChecked():
            return
        option = QStyleOptionButton()
        self.initStyleOption(option)
        box = self.style().subElementRect(
            self.style().SubElement.SE_CheckBoxIndicator, option, self
        )
        if box.width() <= 0:
            return
        pen = QPen(QColor(ACCENT_INK), 2.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        path = QPainterPath(QPoint(box.left() + 4, box.center().y()))
        path.lineTo(box.left() + box.width() * 0.44, box.bottom() - 5)
        path.lineTo(box.right() - 4, box.top() + 5)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(pen)
        painter.drawPath(path)
        painter.end()


class ChevronComboBox(QComboBox):
    """Combo box with a painted chevron, so no image asset is needed."""

    _arrow_width = 22

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API naming
        super().paintEvent(event)
        color = QColor(TEXT_DIM if (self.isEnabled() and self.underMouse()) else TEXT_FAINT)
        if not self.isEnabled():
            color = QColor(DISABLED_INK)
        center_x = self.width() - self._arrow_width // 2 - 2
        center_y = self.height() // 2
        path = QPainterPath(QPoint(center_x - 4, center_y - 2))
        path.lineTo(center_x, center_y + 2)
        path.lineTo(center_x + 4, center_y - 2)
        pen = QPen(color, 1.4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(pen)
        painter.drawPath(path)
        painter.end()


class ArrowSpinBox(QSpinBox):
    """Spin box whose arrows stay visible under the application stylesheet."""

    _button_width = 20

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API naming
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(TEXT_DIM) if self.isEnabled() else QColor(DISABLED_INK))
        center_x = self.width() - self._button_width // 2 - 3
        self._draw_arrow(painter, center_x, self.height() // 4, pointing_up=True)
        self._draw_arrow(painter, center_x, self.height() * 3 // 4, pointing_up=False)
        painter.end()

    @staticmethod
    def _draw_arrow(painter: QPainter, x: int, y: int, *, pointing_up: bool) -> None:
        if pointing_up:
            points = [QPoint(x - 3, y + 2), QPoint(x + 3, y + 2), QPoint(x, y - 2)]
        else:
            points = [QPoint(x - 3, y - 2), QPoint(x + 3, y - 2), QPoint(x, y + 2)]
        painter.drawPolygon(QPolygon(points))


class DropZone(QFrame):
    """Empty-state intake target; doubles as the queue's placeholder page."""

    files_dropped = Signal(list)
    choose_requested = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        title: str = "拖放音视频或 SRT 文件",
        hint: str = "MP4 / M4A / MP3 / WAV / FLAC / OGG / SRT，也可以拖入整个文件夹",
        show_button: bool = True,
        button_text: str = "选择文件",
    ):
        super().__init__(parent)
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(168)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 18, 16, 18)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        mark = QLabel("＋")
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setStyleSheet(
            f"color: {ACCENT_TEXT}; font-size: 20pt; background: transparent;"
        )
        mark.setFixedHeight(30)
        heading = QLabel(title)
        heading.setObjectName("dropTitle")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note = QLabel(hint)
        note.setObjectName("dropHint")
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note.setWordWrap(True)
        # A fixed measure keeps the extension list breaking at a sensible point
        # instead of wherever the panel happens to end.
        note.setFixedWidth(430)

        layout.addWidget(mark)
        layout.addWidget(heading)
        if show_button:
            choose = QPushButton(button_text)
            choose.setObjectName("ghost")
            choose.clicked.connect(self.choose_requested.emit)
            row = QHBoxLayout()
            row.addStretch()
            row.addWidget(choose)
            row.addStretch()
            layout.addSpacing(2)
            layout.addLayout(row)
        layout.addWidget(note, 0, Qt.AlignmentFlag.AlignHCenter)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt API naming
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802 - Qt API naming
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802 - Qt API naming
        event.accept()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt API naming
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()