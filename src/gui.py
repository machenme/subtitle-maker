#!/usr/bin/env python
"""
ASR Pipeline desktop UI built with PySide6.

The transcription engine remains in the existing Python modules. This module
only owns the desktop presentation, user input, and worker-thread signals.

Usage:
    uv run python -m src.gui
"""
from __future__ import annotations

import html
import logging
import os
import copy
from queue import Queue
import sys
import subprocess
import threading
import time
from pathlib import Path

if __name__ == "__main__" and str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QPoint, QThread, QTimer, Qt, QObject, Signal
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QFont, QPainter, QPolygon
from PySide6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QInputDialog,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedLayout,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.config import (
    DEFAULT_MAX_WORKERS,
    DEFAULT_MEDIA_EXTENSIONS,
    PipelineConfig,
)
from src.main import run_one_video
from src.monitor import GpuMonitor
from src.translator import (
    TranslationError,
    create_translator,
    iso_to_player_suffix,
    translate_srt_with_outputs,
)
from src.utils import is_srt_valid


APP_STYLE = """
QWidget {
    background: #f4f7fb;
    color: #1f2d43;
    font-family: "Microsoft YaHei UI", "Segoe UI";
    font-size: 10pt;
}
QMainWindow { background: #f4f7fb; }
QLabel, QCheckBox { background: transparent; }
QFrame#card {
    background: #ffffff;
    border: 1px solid #e3eaf3;
    border-radius: 8px;
}
QFrame#header {
    background: #ffffff;
    border: 1px solid #e3eaf3;
    border-radius: 8px;
}
QFrame#dropZone {
    background: #f8fbff;
    border: 1px dashed #9db8d8;
    border-radius: 8px;
}
QFrame#dropZone:hover { background: #f0f7ff; border-color: #2f76c7; }
QLabel#title { color: #183153; font-size: 22pt; font-weight: 700; }
QLabel#subtitle { color: #607590; font-size: 10pt; }
QLabel#sectionTitle { color: #263b59; font-size: 11pt; font-weight: 700; }
QLabel#muted { color: #52677f; }
QLabel#dropTitle { color: #28476e; font-size: 12pt; font-weight: 600; }
QLabel#dropHint { color: #7b8da5; }
QLabel#badge {
    color: #2464a4;
    background: #eaf3ff;
    border: 1px solid #cfe3fa;
    border-radius: 10px;
    padding: 5px 10px;
    font-weight: 600;
}
QLabel#gpuValue { color: #2b6cb0; font-weight: 600; }
QLabel#status { color: #60738d; }
QLineEdit, QComboBox, QSpinBox {
    background: #ffffff;
    border: 1px solid #d5dfeb;
    border-radius: 7px;
    padding: 7px 9px;
    min-height: 18px;
}
QSpinBox { padding-right: 30px; }
QSpinBox::up-button, QSpinBox::down-button {
    subcontrol-origin: border;
    width: 22px;
    background: #f5f8fc;
    border-left: 1px solid #d5dfeb;
}
QSpinBox::up-button {
    subcontrol-position: top right;
    border-top-right-radius: 6px;
    border-bottom: 1px solid #d5dfeb;
}
QSpinBox::down-button {
    subcontrol-position: bottom right;
    border-bottom-right-radius: 6px;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover { background: #e7f1fc; }
QSpinBox::up-arrow, QSpinBox::down-arrow { width: 9px; height: 7px; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
    border: 1px solid #4f91d3;
}
QComboBox::drop-down { border: 0; width: 24px; }
QPushButton {
    background: #ffffff;
    border: 1px solid #d3deeb;
    border-radius: 7px;
    padding: 8px 13px;
    color: #314967;
}
QPushButton:hover { background: #edf5fe; border-color: #9fc4e8; }
QPushButton:pressed { background: #e0eefc; }
QPushButton:disabled { color: #a7b3c2; background: #f2f5f8; }
QPushButton#primary {
    background: #2f76c7;
    border-color: #2f76c7;
    color: #ffffff;
    font-weight: 700;
    padding: 9px 18px;
}
QPushButton#primary:hover { background: #2567b1; border-color: #2567b1; }
QPushButton#danger { color: #b14d4d; }
QPushButton#link { border: 0; color: #2f76c7; padding: 3px 6px; }
QTableWidget {
    background: #ffffff;
    alternate-background-color: #f8fafc;
    border: 0;
    gridline-color: #edf1f6;
    selection-background-color: #e6f1ff;
    selection-color: #1f2d43;
}
QHeaderView::section {
    background: #f5f8fc;
    color: #72839a;
    border: 0;
    border-bottom: 1px solid #e5ebf2;
    padding: 9px 8px;
    font-weight: 600;
}
QProgressBar {
    background: #e8eef5;
    border: 0;
    border-radius: 5px;
    text-align: center;
    color: #ffffff;
    min-height: 10px;
    max-height: 10px;
}
QProgressBar::chunk { background: #3b82c4; border-radius: 5px; }
QTextEdit {
    background: #172235;
    border: 0;
    border-radius: 8px;
    color: #d9e4f1;
    selection-background-color: #2b527d;
    padding: 8px;
}
QCheckBox { spacing: 7px; color: #425873; }
QCheckBox::indicator { width: 15px; height: 15px; }
QCheckBox::indicator:unchecked {
    background: #ffffff;
    border: 1px solid #c5d2e0;
    border-radius: 4px;
}
QCheckBox::indicator:checked {
    background: #2f76c7;
    border: 1px solid #2f76c7;
    border-radius: 4px;
}
QScrollArea { border: 0; background: transparent; }
"""


LANGUAGE_MAP = {
    "自动检测": "auto",
    "日语 (ja)": "ja",
    "中文 (zh)": "zh",
    "英语 (en)": "en",
    "韩语 (ko)": "ko",
    "法语 (fr)": "fr",
    "德语 (de)": "de",
    "西班牙语 (es)": "es",
    "葡萄牙语 (pt)": "pt",
    "意大利语 (it)": "it",
    "俄语 (ru)": "ru",
    "阿拉伯语 (ar)": "ar",
    "泰语 (th)": "th",
    "越南语 (vi)": "vi",
}
TRANSLATE_MAP = {
    "中文 (zh)": "zh",
    "英语 (en)": "en",
    "韩语 (ko)": "ko",
    "日语 (ja)": "ja",
    "法语 (fr)": "fr",
    "德语 (de)": "de",
    "俄语 (ru)": "ru",
    "西班牙语 (es)": "es",
    "葡萄牙语 (pt)": "pt",
}
TRANSLATOR_MAP = {
    "Microsoft Edge Translator (免费)": "bing",
    "Legacy GTX (免费)": "gtx",
}
MEDIA_SUFFIXES = {*DEFAULT_MEDIA_EXTENSIONS, "srt"}


def _card(title: str, subtitle: str = "") -> tuple[QFrame, QVBoxLayout]:
    card = QFrame()
    card.setObjectName("card")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(10)
    heading = QLabel(title)
    heading.setObjectName("sectionTitle")
    layout.addWidget(heading)
    if subtitle:
        hint = QLabel(subtitle)
        hint.setObjectName("muted")
        layout.addWidget(hint)
    return card, layout


class DropZone(QFrame):
    files_dropped = Signal(list)
    choose_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(144)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 14)
        layout.setSpacing(4)
        icon = QLabel("↓")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet("color:#3b82c4; font-size:18pt; font-weight:700;")
        icon.setFixedHeight(24)
        title = QLabel("拖放音视频或 SRT 文件")
        title.setObjectName("dropTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFixedHeight(24)
        hint = QLabel("支持 MP4、M4A、MP3、WAV、FLAC、OGG 等格式")
        hint.setObjectName("dropHint")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setFixedHeight(22)
        choose = QPushButton("选择文件")
        choose.setFixedHeight(28)
        choose.clicked.connect(self.choose_requested.emit)
        layout.addWidget(icon)
        layout.addWidget(title)
        layout.addWidget(choose, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()


class ArrowSpinBox(QSpinBox):
    """A spin box with arrows that remain visible under the application stylesheet."""

    _button_width = 22

    def paintEvent(self, event) -> None:
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#547493") if self.isEnabled() else QColor("#a7b3c2"))
        center_x = self.width() - self._button_width // 2 - 1
        self._draw_arrow(painter, center_x, self.height() // 4, pointing_up=True)
        self._draw_arrow(painter, center_x, self.height() * 3 // 4, pointing_up=False)
        painter.end()

    @staticmethod
    def _draw_arrow(painter: QPainter, x: int, y: int, *, pointing_up: bool) -> None:
        if pointing_up:
            points = [QPoint(x - 4, y + 2), QPoint(x + 4, y + 2), QPoint(x, y - 3)]
        else:
            points = [QPoint(x - 4, y - 2), QPoint(x + 4, y - 2), QPoint(x, y + 3)]
        painter.drawPolygon(QPolygon(points))


class LogBridge(QObject):
    message = Signal(str)


class QtLogHandler(logging.Handler):
    def __init__(self, bridge: LogBridge):
        super().__init__()
        self.bridge = bridge
        self.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.bridge.message.emit(self.format(record))
        except Exception:
            pass


class PipelineWorker(QObject):
    log = Signal(str)
    progress = Signal(float, int, int)
    file_status = Signal(str, str)
    finished = Signal(int, int, int, float, bool)

    def __init__(
        self,
        paths: list[Path],
        statuses: dict[str, str],
        config: PipelineConfig,
        cancel_event: threading.Event,
    ):
        super().__init__()
        self.paths = paths
        self.statuses = statuses
        self.config = config
        self.cancel_event = cancel_event

    @staticmethod
    def check_existing_subs(
        media_path: Path,
        target_lang: str = "",
        swap_subtitles: bool = True,
        source_lang: str = "auto",
        output_dir: Path | None = None,
    ) -> str:
        source_base = media_path.with_suffix("")
        base = (Path(output_dir) / media_path.stem) if output_dir else source_base
        if target_lang:
            target_path = Path(f"{base}.{iso_to_player_suffix(target_lang)}.srt")
            if swap_subtitles:
                source_code = source_lang if source_lang != "auto" else "ja"
                source_suffix = iso_to_player_suffix(source_code)
                required = (
                    Path(f"{base}.srt"),
                    Path(f"{base}.bilingual.srt"),
                    Path(f"{base}.{source_suffix}.srt"),
                )
                if all(is_srt_valid(path) for path in required):
                    return "done"
            elif is_srt_valid(target_path):
                return "done"
        if is_srt_valid(Path(f"{base}.srt")):
            if target_lang and base != source_base:
                return "translate_output"
            return "translate" if target_lang else "done"
        if base != source_base and is_srt_valid(Path(f"{source_base}.srt")):
            return "translate" if target_lang else "done"
        return "asr"

    def _translate_subtitles(self, media_path: Path, srt_path: Path) -> None:
        """Translate one completed SRT without blocking the ASR producer."""
        translate_srt_with_outputs(
            srt_path,
            self.config.translate_to,
            provider=create_translator(
                self.config.translation_provider,
                proxy=self.config.translation_proxy,
            ),
            source_lang=self.config.language,
            swap_subtitles=self.config.swap_subtitles,
        )

    def _run_translation_queue(
        self,
        translation_queue: Queue[tuple[Path, Path] | None],
        results: Queue[tuple[Path, str, str]],
    ) -> None:
        """Consume subtitle jobs serially while ASR continues in the Qt worker."""
        while True:
            task = translation_queue.get()
            if task is None:
                return

            media_path, srt_path = task
            if self.cancel_event.is_set():
                self.file_status.emit(str(media_path), "等待翻译")
                results.put((media_path, "deferred", ""))
                continue

            self.file_status.emit(str(media_path), "翻译中")
            try:
                self._translate_subtitles(media_path, srt_path)
            except TranslationError as exc:
                self.file_status.emit(str(media_path), "翻译失败")
                self.log.emit(f"翻译失败: {media_path.name} — {exc}")
                results.put((media_path, "failed", str(exc)))
            except Exception as exc:
                self.file_status.emit(str(media_path), "翻译失败")
                logging.getLogger(__name__).exception(
                    "GUI translation failed for %s", media_path
                )
                self.log.emit(f"翻译失败: {media_path.name} — {exc}")
                results.put((media_path, "failed", str(exc)))
            else:
                self.file_status.emit(str(media_path), "完成 · 已翻译")
                results.put((media_path, "done", ""))

    @staticmethod
    def _collect_translation_results(
        results: Queue[tuple[Path, str, str]],
    ) -> tuple[int, int]:
        done_count = 0
        failed_count = 0
        while not results.empty():
            _media_path, outcome, _error = results.get()
            if outcome == "done":
                done_count += 1
            elif outcome == "failed":
                failed_count += 1
        return done_count, failed_count

    def run(self) -> None:
        started = time.time()
        done_count = 0
        failed_count = 0
        total = len(self.paths)
        translation_queue: Queue[tuple[Path, Path] | None] = Queue()
        translation_results: Queue[tuple[Path, str, str]] = Queue()
        translation_thread: threading.Thread | None = None

        if self.config.translate_to:
            translation_thread = threading.Thread(
                target=self._run_translation_queue,
                args=(translation_queue, translation_results),
                name="subtitle-translation",
            )
            translation_thread.start()

        for index, media_path in enumerate(self.paths):
            if self.cancel_event.is_set():
                break
            try:
                status = self.statuses.get(str(media_path), "asr")
                if media_path.suffix.lower() != ".srt":
                    status = self.check_existing_subs(
                        media_path,
                        self.config.translate_to,
                        self.config.swap_subtitles,
                        self.config.language,
                        self.config.output_dir,
                    )
                self.log.emit(f"检测: {media_path.name} → {status}")
                self.progress.emit(0.0, index, total)

                if status == "done":
                    self.file_status.emit(str(media_path), "已有字幕")
                    done_count += 1
                    self.progress.emit(100.0, index, total)
                    continue

                if status == "direct_srt" and not self.config.translate_to:
                    self.file_status.emit(str(media_path), "已有字幕")
                    done_count += 1
                    self.progress.emit(100.0, index, total)
                    continue

                if status in ("translate", "translate_output", "direct_srt"):
                    if status == "direct_srt":
                        srt_path = media_path
                    elif status == "translate_output":
                        srt_path = self.config.output_dir / f"{media_path.stem}.srt"
                    else:
                        srt_path = media_path.with_suffix(".srt")
                    self.file_status.emit(str(media_path), "等待翻译")
                    translation_queue.put((media_path, srt_path))
                    self.progress.emit(100.0, index, total)
                    continue

                self.file_status.emit(str(media_path), "处理中")
                asr_config = copy.copy(self.config)
                asr_config.translate_to = ""
                ok, segment_count, error = run_one_video(
                    asr_config,
                    media_path,
                    progress_callback=lambda _stage, current, target, i=index, t=total: self.progress.emit(
                        (current / max(target, 1)) * 100, i, t
                    ),
                    cancel_event=self.cancel_event,
                )
                if ok:
                    if self.config.translate_to:
                        srt_path = self.config.output_dir / f"{media_path.stem}.srt"
                        self.file_status.emit(str(media_path), "等待翻译")
                        translation_queue.put((media_path, srt_path))
                    else:
                        self.file_status.emit(str(media_path), f"完成 · {segment_count} 段")
                        done_count += 1
                else:
                    if self.cancel_event.is_set():
                        self.file_status.emit(str(media_path), "已停止")
                    else:
                        failed_count += 1
                        self.file_status.emit(str(media_path), "失败")
                        self.log.emit(f"处理失败: {media_path.name} — {error}")
                self.progress.emit(100.0, index, total)
            except Exception as exc:
                failed_count += 1
                self.file_status.emit(str(media_path), "失败")
                logging.getLogger(__name__).exception("GUI pipeline failed for %s", media_path)
                self.log.emit(f"处理失败: {media_path.name} — {exc}")

            completed, failures = self._collect_translation_results(translation_results)
            done_count += completed
            failed_count += failures

        if translation_thread:
            translation_queue.put(None)
            translation_thread.join()
            completed, failures = self._collect_translation_results(translation_results)
            done_count += completed
            failed_count += failures

        self.finished.emit(
            done_count,
            total,
            failed_count,
            time.time() - started,
            self.cancel_event.is_set(),
        )


class AsrWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASR Pipeline")
        self.setMinimumSize(1000, 820)
        self.resize(1180, 860)

        self.paths: list[Path] = []
        self.file_status: dict[str, str] = {}
        self._row_by_path: dict[str, int] = {}
        self._cancel = threading.Event()
        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None
        self._gpu_monitor: GpuMonitor | None = None
        self._is_running = False
        self._log_bridge = LogBridge()
        self._log_handler = QtLogHandler(self._log_bridge)
        self._log_bridge.message.connect(self._append_log)
        self._setup_logging()
        self._load_config()
        self._build_ui()
        self._start_gpu_monitor()
        self._append_log("ASR Pipeline 已启动。拖入音视频或点击“选择文件”开始。")

    def _setup_logging(self) -> None:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(self._log_handler)

    def _load_config(self) -> None:
        self.config: PipelineConfig | None = None
        self.config_error = ""
        try:
            self.config = PipelineConfig.build({"config": "./config.yaml"})
        except Exception as exc:
            self.config_error = str(exc)
        self.default_output_dir = str(self.config.output_dir if self.config else (Path.cwd() / "output").resolve())

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(22, 20, 22, 16)
        outer.setSpacing(14)
        outer.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setSpacing(14)
        body.addWidget(self._build_left_panel(), 3)
        body.addWidget(self._build_settings_panel(), 2)
        outer.addLayout(body, 1)
        outer.addWidget(self._build_progress_footer())
        self.setCentralWidget(root)
        self._refresh_queue_view()

    def _build_header(self) -> QWidget:
        card = QFrame()
        card.setObjectName("header")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(20, 15, 20, 15)
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("ASR Pipeline")
        title.setObjectName("title")
        subtitle = QLabel("音视频转写 · 本地 GPU 加速 · SRT / TXT / Markdown")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        layout.addLayout(title_box)
        layout.addStretch()
        status_box = QVBoxLayout()
        status_box.setSpacing(5)
        badge = QLabel("LOCAL  ·  GPU")
        badge.setObjectName("badge")
        status_box.addWidget(badge, alignment=Qt.AlignmentFlag.AlignRight)
        self.gpu_label = QLabel("GPU 检测中...")
        self.gpu_label.setObjectName("gpuValue")
        self.gpu_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        status_box.addWidget(self.gpu_label)
        layout.addLayout(status_box)
        return card

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        drop_card, drop_layout = _card("输入文件", "支持批量处理，自动检测同名字幕并支持断点续跑")
        drop_card.setMaximumHeight(220)
        self.drop_zone = DropZone()
        self.drop_zone.files_dropped.connect(self._add_paths_from_strings)
        self.drop_zone.choose_requested.connect(self._choose_files)
        drop_layout.addWidget(self.drop_zone)
        layout.addWidget(drop_card)

        list_card, list_layout = _card("处理队列")
        self.queue_summary = QLabel("尚未添加文件")
        self.queue_summary.setObjectName("muted")
        list_layout.addWidget(self.queue_summary)
        self.file_table = QTableWidget(0, 3)
        self.file_table.setHorizontalHeaderLabels(["文件", "时长", "状态"])
        self.file_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.file_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.file_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.file_table.setAlternatingRowColors(True)
        self.file_table.verticalHeader().setVisible(False)
        self.file_table.horizontalHeader().setStretchLastSection(False)
        self.file_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.file_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.file_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.file_table.doubleClicked.connect(self._open_selected_file)
        queue_host = QWidget()
        self.queue_stack = QStackedLayout(queue_host)
        self.queue_empty = QLabel("尚未添加文件")
        self.queue_empty.setObjectName("muted")
        self.queue_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.queue_stack.addWidget(self.queue_empty)
        self.queue_stack.addWidget(self.file_table)
        list_layout.addWidget(queue_host, 1)

        actions = QHBoxLayout()
        add_button = QPushButton("添加文件")
        add_button.clicked.connect(self._choose_files)
        remove_button = QPushButton("移除选中")
        remove_button.clicked.connect(self._remove_selected)
        clear_button = QPushButton("清空")
        clear_button.setObjectName("link")
        clear_button.clicked.connect(self._clear_files)
        self.add_file_button = add_button
        self.remove_button = remove_button
        self.clear_button = clear_button
        self._file_buttons = [add_button, remove_button, clear_button]
        self.file_table.itemSelectionChanged.connect(self._refresh_file_action_states)
        actions.addWidget(add_button)
        actions.addWidget(remove_button)
        actions.addStretch()
        actions.addWidget(clear_button)
        list_layout.addLayout(actions)
        layout.addWidget(list_card, 2)

        log_card, log_layout = _card("实时日志")
        self.log_card = log_card
        log_card.setMaximumHeight(190)
        log_actions = QHBoxLayout()
        self.expand_log_check = QCheckBox("展开日志")
        self.expand_log_check.toggled.connect(self._set_log_expanded)
        log_actions.addStretch()
        log_actions.addWidget(self.expand_log_check)
        log_layout.addLayout(log_actions)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(112)
        self.log_edit.setMaximumHeight(130)
        self.log_edit.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_edit)
        layout.addWidget(log_card)
        return panel

    def _build_settings_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(6)

        output_card, output_layout = _card("输出")
        output_row = QHBoxLayout()
        self.output_edit = QLineEdit(self.default_output_dir)
        self.output_edit.editingFinished.connect(self._on_output_directory_changed)
        browse = QPushButton("浏览")
        browse.clicked.connect(self._browse_output)
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(browse)
        output_layout.addWidget(QLabel("输出目录"))
        output_layout.addLayout(output_row)
        format_label = QLabel("输出格式")
        output_layout.addWidget(format_label)
        formats = QHBoxLayout()
        self.srt_check = QCheckBox("SRT")
        self.srt_check.setChecked(True)
        self.txt_check = QCheckBox("TXT")
        self.md_check = QCheckBox("MD")
        formats.addWidget(self.srt_check)
        formats.addWidget(self.txt_check)
        formats.addWidget(self.md_check)
        formats.addStretch()
        output_layout.addLayout(formats)
        self.translate_check = QCheckBox("翻译字幕")
        self.translate_check.setChecked(True)
        output_layout.addWidget(self.translate_check)
        layout.addWidget(output_card)

        transcribe_card, transcribe_layout = _card("转写设置")
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(7)
        grid.setColumnMinimumWidth(0, 64)
        grid.setColumnStretch(1, 1)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["large-v3-turbo", "large-v3", "medium"])
        self.model_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.model_combo.setCurrentText(self.config.model_size if self.config else "large-v3-turbo")
        self.workers_spin = ArrowSpinBox()
        self.workers_spin.setRange(1, DEFAULT_MAX_WORKERS)
        self.workers_spin.setValue(self.config.max_workers if self.config else DEFAULT_MAX_WORKERS)
        self.workers_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        self.workers_spin.setToolTip("启动时逐个尝试，直到模型加载失败；本次使用失败前的并发数")
        self.language_combo = QComboBox()
        self.language_combo.addItems(list(LANGUAGE_MAP))
        self.language_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.language_combo.setCurrentText("自动检测")
        self.beam_spin = ArrowSpinBox()
        self.beam_spin.setRange(1, 10)
        self.beam_spin.setValue(self.config.beam_size if self.config else 5)
        self.beam_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        self.chunk_combo = QComboBox()
        self.chunk_combo.setEditable(True)
        self.chunk_combo.addItems(["自动", "300", "600", "900", "1200", "1800", "3600"])
        self.chunk_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.vad_check = QCheckBox("启用 VAD 语音检测")
        self.vad_check.setChecked(self.config.vad_filter if self.config else True)
        self._add_form_row(grid, 0, "模型", self.model_combo)
        self._add_form_row(grid, 1, "最大并发", self.workers_spin)
        self._add_form_row(grid, 2, "识别语言", self.language_combo)
        transcribe_layout.addLayout(grid)
        transcribe_layout.addWidget(self.vad_check)
        self.advanced_toggle = QCheckBox("显示高级参数")
        self.advanced_toggle.setChecked(False)
        transcribe_layout.addWidget(self.advanced_toggle)
        self.advanced_settings = QWidget()
        advanced_grid = QGridLayout(self.advanced_settings)
        advanced_grid.setContentsMargins(0, 0, 0, 0)
        advanced_grid.setHorizontalSpacing(10)
        advanced_grid.setVerticalSpacing(7)
        self._add_form_row(advanced_grid, 0, "束搜索宽度", self.beam_spin)
        self._add_form_row(advanced_grid, 1, "切割时长", self.chunk_combo)
        self.advanced_settings.setVisible(False)
        self.advanced_toggle.toggled.connect(self.advanced_settings.setVisible)
        transcribe_layout.addWidget(self.advanced_settings)
        layout.addWidget(transcribe_card)

        translate_card, translate_layout = _card("翻译设置")
        translate_layout.setSpacing(7)
        translate_grid = QGridLayout()
        translate_grid.setHorizontalSpacing(10)
        translate_grid.setVerticalSpacing(7)
        translate_grid.setColumnMinimumWidth(0, 64)
        translate_grid.setColumnStretch(1, 1)
        self.translator_combo = QComboBox()
        self.translator_combo.addItems(list(TRANSLATOR_MAP))
        self.translator_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        configured_provider = self.config.translation_provider if self.config else "bing"
        self.translator_combo.setCurrentText(
            next(
                (label for label, value in TRANSLATOR_MAP.items() if value == configured_provider),
                "Microsoft Edge Translator (免费)",
            )
        )
        self.translate_combo = QComboBox()
        self.translate_combo.addItems(list(TRANSLATE_MAP))
        self.translate_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        configured_target = self.config.translate_to if self.config else "zh"
        self.translate_combo.setCurrentText(
            next(
                (label for label, value in TRANSLATE_MAP.items() if value == configured_target),
                "中文 (zh)",
            )
        )
        self.swap_check = QCheckBox("生成单语译文、双语字幕和原始字幕")
        self.swap_check.setChecked(True)
        self.proxy_edit = QLineEdit(self.config.translation_proxy if self.config else "")
        self.proxy_edit.setPlaceholderText("127.0.0.1:7897")
        self.proxy_edit.setToolTip("Legacy GTX 必须使用代理，例如 127.0.0.1:7897")
        self._add_form_row(translate_grid, 0, "后端", self.translator_combo)
        self._add_form_row(translate_grid, 1, "目标语言", self.translate_combo)
        self.proxy_label = self._add_form_row(translate_grid, 2, "GTX 代理", self.proxy_edit)
        translate_layout.addLayout(translate_grid)
        translate_layout.addWidget(self.swap_check)
        self.translator_combo.currentTextChanged.connect(self._on_translator_changed)
        self._update_gtx_controls()
        self.translate_card = translate_card
        self.translate_check.toggled.connect(self._set_translation_enabled)
        self.language_combo.currentTextChanged.connect(self._on_source_language_changed)
        self._set_translation_enabled(self.translate_check.isChecked())
        layout.addWidget(translate_card)

        layout.addStretch()
        scroll.setWidget(content)
        return scroll

    @staticmethod
    def _add_form_row(grid: QGridLayout, row: int, label: str, widget: QWidget) -> QLabel:
        label_widget = QLabel(label)
        label_widget.setObjectName("muted")
        label_widget.setMinimumWidth(64)
        grid.addWidget(label_widget, row, 0)
        grid.addWidget(widget, row, 1)
        return label_widget

    def _build_progress_footer(self) -> QWidget:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        top = QHBoxLayout()
        self.progress_label = QLabel("准备就绪")
        self.progress_label.setObjectName("status")
        self.run_summary = QLabel("")
        self.run_summary.setObjectName("muted")
        self.file_counter = QLabel("")
        self.file_counter.setObjectName("muted")
        top.addWidget(self.progress_label)
        top.addWidget(self.run_summary)
        top.addStretch()
        top.addWidget(self.file_counter)
        layout.addLayout(top)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)
        buttons = QHBoxLayout()
        buttons.addStretch()
        self.stop_button = QPushButton("停止")
        self.stop_button.setObjectName("danger")
        self.stop_button.setEnabled(False)
        self.stop_button.setVisible(False)
        self.stop_button.clicked.connect(self._stop)
        self.start_button = QPushButton("开始转写")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self._start)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.start_button)
        layout.addLayout(buttons)
        self._settings_widgets = [
            self.output_edit, self.model_combo, self.workers_spin, self.language_combo,
            self.beam_spin, self.chunk_combo, self.vad_check, self.srt_check,
            self.txt_check, self.md_check, self.translate_check, self.translator_combo, self.translate_combo,
            self.proxy_edit, self.swap_check, self.advanced_toggle,
        ]
        for widget in (
            self.srt_check,
            self.txt_check,
            self.md_check,
            self.translate_check,
            self.translate_combo,
        ):
            widget.toggled.connect(self._refresh_run_summary) if isinstance(widget, QCheckBox) else widget.currentTextChanged.connect(self._refresh_run_summary)
        return card

    @staticmethod
    def _queue_summary_text(statuses: list[str]) -> str:
        if not statuses:
            return "尚未添加文件"

        asr_active = sum(status == "处理中" for status in statuses)
        translating = sum(status == "翻译中" for status in statuses)
        waiting_translation = sum(
            status in {"待翻译", "等待翻译", "translate", "translate_output", "direct_srt"}
            for status in statuses
        )
        pending_asr = sum(status in {"asr", "等待中"} for status in statuses)
        completed = sum(
            status.startswith("完成") or status in {"已有字幕", "done"}
            for status in statuses
        )
        parts = [f"{len(statuses)} 个文件"]
        if pending_asr:
            parts.append(f"ASR 待处理 {pending_asr}")
        if asr_active:
            parts.append(f"ASR 中 {asr_active}")
        if translating:
            parts.append(f"翻译中 {translating}")
        if waiting_translation:
            parts.append(f"待翻译 {waiting_translation}")
        if completed:
            parts.append(f"完成 {completed}")
        return "  ·  ".join(parts)

    def _refresh_queue_view(self) -> None:
        if not hasattr(self, "queue_stack"):
            return
        self.queue_stack.setCurrentIndex(1 if self.paths else 0)
        self.queue_summary.setVisible(bool(self.paths))
        self.queue_summary.setText(self._queue_summary_text(list(self.file_status.values())))
        self._refresh_file_action_states()
        self._refresh_run_summary()

    def _refresh_file_action_states(self) -> None:
        if not hasattr(self, "add_file_button"):
            return
        can_edit = not self._is_running
        self.add_file_button.setEnabled(can_edit)
        self.remove_button.setEnabled(
            can_edit and bool(self.file_table.selectionModel().selectedRows())
        )
        self.clear_button.setEnabled(can_edit and bool(self.paths))

    def _set_log_expanded(self, expanded: bool) -> None:
        self.log_card.setMaximumHeight(16777215 if expanded else 190)
        self.log_edit.setMaximumHeight(16777215 if expanded else 130)
        self.log_edit.setMinimumHeight(280 if expanded else 112)

    def _on_output_directory_changed(self) -> None:
        self._refresh_file_statuses(self.translate_check.isChecked())

    def _refresh_run_summary(self, _value: object = None) -> None:
        if not hasattr(self, "run_summary"):
            return
        if not self.paths:
            self.run_summary.setText("等待添加文件")
            return
        formats = [
            label
            for label, checked in (
                ("SRT", self.srt_check.isChecked()),
                ("TXT", self.txt_check.isChecked()),
                ("MD", self.md_check.isChecked()),
            )
            if checked
        ]
        summary = f"{len(self.paths)} 个文件 · {' / '.join(formats)}"
        if self._translation_requested():
            summary += f" · 翻译至 {self.translate_combo.currentText()}"
        self.run_summary.setText(summary)

    def _start_gpu_monitor(self) -> None:
        self._gpu_monitor = GpuMonitor(gpu_index=0, interval=1.0)
        self._gpu_monitor.start()
        self._gpu_timer = QTimer(self)
        self._gpu_timer.timeout.connect(self._update_gpu)
        self._gpu_timer.start(1000)

    def _update_gpu(self) -> None:
        if self._gpu_monitor and self._gpu_monitor.available:
            snap = self._gpu_monitor.latest()
            if snap:
                self.gpu_label.setText(
                    f"利用率 {snap.utilization_pct}%  ·  显存 "
                    f"{snap.memory_used_mb / 1024:.1f}/{snap.memory_total_mb / 1024:.1f} GB  ·  "
                    f"温度 {snap.temperature_c}°C"
                )
                return
        self.gpu_label.setText("未检测到 NVIDIA GPU")

    def _update_gtx_controls(self) -> None:
        is_gtx = (
            self.translate_check.isChecked()
            and TRANSLATOR_MAP.get(self.translator_combo.currentText()) == "gtx"
        )
        self.proxy_edit.setEnabled(is_gtx)
        self.proxy_edit.setVisible(is_gtx)
        self.proxy_label.setVisible(is_gtx)

    def _set_translation_enabled(self, enabled: bool) -> None:
        self.translate_card.setVisible(enabled)
        self._update_gtx_controls()
        self._refresh_file_statuses(enabled)
        self._refresh_run_summary()

    def _on_source_language_changed(self, _label: str) -> None:
        self._refresh_file_statuses(self.translate_check.isChecked())

    def _translation_requested(self) -> bool:
        source_lang = LANGUAGE_MAP[self.language_combo.currentText()]
        target_lang = TRANSLATE_MAP[self.translate_combo.currentText()]
        return self.translate_check.isChecked() and source_lang != target_lang

    def _refresh_file_statuses(self, translate_enabled: bool) -> None:
        if not hasattr(self, "file_table"):
            return
        source_lang = LANGUAGE_MAP[self.language_combo.currentText()]
        target_lang = TRANSLATE_MAP[self.translate_combo.currentText()]
        translation_requested = translate_enabled and source_lang != target_lang
        status_labels = {
            "done": "已有字幕",
            "translate": "待翻译",
            "translate_output": "待翻译",
            "direct_srt": "等待翻译",
            "asr": "等待中",
        }
        for path in self.paths:
            if path.suffix.lower() == ".srt":
                status = "direct_srt" if translation_requested else "done"
            else:
                status = PipelineWorker.check_existing_subs(
                    path,
                    target_lang if translation_requested else "",
                    self.swap_check.isChecked(),
                    source_lang,
                    Path(self.output_edit.text()),
                )
            self.file_status[str(path)] = status
            row = self._row_by_path.get(str(path))
            if row is not None:
                self.file_table.item(row, 2).setText(status_labels[status])
        self._refresh_queue_view()

    def _on_translator_changed(self, label: str) -> None:
        if TRANSLATOR_MAP.get(label) == "gtx" and not self.proxy_edit.text().strip():
            QMessageBox.information(
                self,
                "Legacy GTX 需要代理",
                "该翻译接口需要通过代理访问。请输入代理地址，例如 127.0.0.1:7897。",
            )
            proxy, accepted = QInputDialog.getText(
                self,
                "添加 Legacy GTX 代理",
                "代理地址:",
                text="127.0.0.1:7897",
            )
            if accepted and proxy.strip():
                self.proxy_edit.setText(proxy.strip())
            else:
                self.translator_combo.setCurrentText("Microsoft Edge Translator (免费)")
        self._update_gtx_controls()

    def _choose_files(self) -> None:
        patterns = " ".join(f"*.{ext}" for ext in (*DEFAULT_MEDIA_EXTENSIONS, "srt"))
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择音视频文件",
            str(Path.cwd()),
            f"音视频 / 字幕 ({patterns});;所有文件 (*.*)",
        )
        self._add_paths_from_strings(files)

    def _browse_output(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_edit.text())
        if directory:
            self.output_edit.setText(directory)
            self._on_output_directory_changed()

    def _add_paths_from_strings(self, raw_paths: list[str]) -> None:
        for raw_path in raw_paths:
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file() or path.suffix.lower().lstrip(".") not in MEDIA_SUFFIXES:
                continue
            self._add_file(path)

    def _add_file(self, path: Path) -> None:
        if path in self.paths:
            return
        if any(existing.stem == path.stem for existing in self.paths):
            self._append_log(f"警告: 重名文件未添加，避免覆盖输出: {path.name}")
            return
        if not self.paths:
            self.output_edit.setText(str(path.parent))
        self.paths.append(path)
        if path.suffix.lower() == ".srt":
            status = "direct_srt" if self._translation_requested() else "done"
            status_text = "等待翻译" if self._translation_requested() else "已有字幕"
        else:
            status = PipelineWorker.check_existing_subs(
                path,
                TRANSLATE_MAP[self.translate_combo.currentText()] if self._translation_requested() else "",
                self.swap_check.isChecked(),
                LANGUAGE_MAP[self.language_combo.currentText()],
                Path(self.output_edit.text()),
            )
            status_text = {
                "done": "已有字幕",
                "translate": "待翻译",
                "translate_output": "待翻译",
                "asr": "等待中",
            }[status]
        self.file_status[str(path)] = status
        row = self.file_table.rowCount()
        self.file_table.insertRow(row)
        name_item = QTableWidgetItem(path.name)
        name_item.setToolTip(str(path))
        name_item.setData(Qt.ItemDataRole.UserRole, str(path))
        self.file_table.setItem(row, 0, name_item)
        duration = "—" if path.suffix.lower() == ".srt" else self._get_duration(path)
        self.file_table.setItem(row, 1, QTableWidgetItem(duration))
        self.file_table.setItem(row, 2, QTableWidgetItem(status_text))
        self._row_by_path[str(path)] = row
        self._append_log(f"已添加: {path.name} → {status_text}")
        self._refresh_queue_view()

    @staticmethod
    def _get_duration(path: Path) -> str:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                capture_output=True,
                text=True,
                timeout=15,
            )
            seconds = float(result.stdout.strip())
            hours, minutes = divmod(int(seconds), 3600)
            minutes, seconds = divmod(minutes, 60)
            return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{seconds:02d}s"
        except Exception:
            return "?"

    def _remove_selected(self) -> None:
        rows = sorted({index.row() for index in self.file_table.selectionModel().selectedRows()}, reverse=True)
        for row in rows:
            path = Path(self.file_table.item(row, 0).data(Qt.ItemDataRole.UserRole))
            self.paths.remove(path)
            self.file_status.pop(str(path), None)
            self.file_table.removeRow(row)
        self._rebuild_row_index()
        self._refresh_queue_view()

    def _clear_files(self) -> None:
        self.paths.clear()
        self.file_status.clear()
        self._row_by_path.clear()
        self.file_table.setRowCount(0)
        self._refresh_queue_view()

    def _rebuild_row_index(self) -> None:
        self._row_by_path.clear()
        for row in range(self.file_table.rowCount()):
            path = self.file_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            self._row_by_path[path] = row

    def _open_selected_file(self) -> None:
        row = self.file_table.currentRow()
        if row < 0:
            return
        path = Path(self.file_table.item(row, 0).data(Qt.ItemDataRole.UserRole))
        if path.exists() and hasattr(os, "startfile"):
            os.startfile(str(path))

    def _start(self) -> None:
        if not self.paths:
            QMessageBox.warning(self, "没有输入文件", "请先添加音视频或 SRT 文件。")
            return
        formats = [
            fmt for fmt, checked in (
                ("srt", self.srt_check.isChecked()),
                ("txt", self.txt_check.isChecked()),
                ("md", self.md_check.isChecked()),
            ) if checked
        ]
        if not formats:
            QMessageBox.warning(self, "没有输出格式", "请至少选择一种输出格式。")
            return
        translate_to = TRANSLATE_MAP[self.translate_combo.currentText()] if self._translation_requested() else ""
        translation_provider = TRANSLATOR_MAP[self.translator_combo.currentText()]
        translation_proxy = self.proxy_edit.text().strip()
        source_language = LANGUAGE_MAP[self.language_combo.currentText()]
        if translate_to and translation_provider == "bing" and source_language == "auto":
            QMessageBox.warning(
                self,
                "需要选择源语言",
                "Microsoft Edge 翻译接口不支持自动检测，请在“识别语言”中选择实际源语言。",
            )
            return
        if translate_to and translation_provider == "gtx" and not translation_proxy:
            QMessageBox.warning(
                self,
                "缺少代理",
                "Legacy GTX 需要代理，请填写例如 127.0.0.1:7897。",
            )
            return
        if translate_to and "srt" not in formats:
            QMessageBox.warning(self, "翻译需要 SRT", "启用翻译时必须勾选 SRT 输出。")
            return
        chunk_value = self.chunk_combo.currentText().strip()
        try:
            config = PipelineConfig.build({
                "config": "./config.yaml",
                "input_dir": str(self.paths[0].parent),
                "output_dir": self.output_edit.text().strip(),
                "model": self.model_combo.currentText(),
                "workers": self.workers_spin.value(),
                "language": source_language,
                "beam_size": self.beam_spin.value(),
                "vad_filter": self.vad_check.isChecked(),
                "chunk_duration": 0 if chunk_value == "自动" else int(chunk_value),
                "output_formats": formats,
                "translate_to": translate_to,
                "translation_provider": translation_provider,
                "translation_proxy": translation_proxy,
                "swap_subtitles": self.swap_check.isChecked(),
            })
        except Exception as exc:
            QMessageBox.critical(self, "配置错误", str(exc))
            return

        self._cancel.clear()
        self._set_running(True)
        self.progress.setValue(0)
        self.progress_label.setText("处理中...")
        self.file_counter.setText(f"0/{len(self.paths)}")
        self._refresh_queue_view()
        self._worker = PipelineWorker(self.paths.copy(), self.file_status.copy(), config, self._cancel)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_log)
        self._worker.progress.connect(self._update_progress)
        self._worker.file_status.connect(self._update_file_status)
        self._worker.finished.connect(self._worker_finished)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._thread_finished)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _stop(self) -> None:
        self._cancel.set()
        self.stop_button.setEnabled(False)
        self.progress_label.setText("正在停止，等待当前任务结束...")
        self._append_log("警告: 用户请求停止，等待当前任务完成...")

    def _set_running(self, running: bool) -> None:
        self._is_running = running
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.stop_button.setVisible(running)
        for widget in self._settings_widgets:
            widget.setEnabled(not running)
        self.translate_card.setEnabled(not running)
        self._refresh_file_action_states()
        if not running:
            self._update_gtx_controls()

    def _update_progress(self, percent: float, file_index: int, total: int) -> None:
        self.progress.setValue(max(0, min(100, int(percent))))
        self.progress_label.setText(f"处理中 · {percent:.1f}%")
        self.file_counter.setText(f"{file_index + 1}/{total}")

    def _update_file_status(self, raw_path: str, status: str) -> None:
        self.file_status[raw_path] = status
        row = self._row_by_path.get(raw_path)
        if row is not None and row < self.file_table.rowCount():
            self.file_table.item(row, 2).setText(status)
        self._refresh_queue_view()

    def _worker_finished(self, done: int, total: int, failed: int, elapsed: float, cancelled: bool) -> None:
        self._set_running(False)
        self.progress.setValue(100 if not cancelled else self.progress.value())
        if cancelled:
            summary = f"已停止 · 完成 {done}/{total}"
        elif failed:
            summary = f"完成 · {done}/{total} · {failed} 个失败"
        else:
            summary = f"完成 · {done}/{total} · 用时 {elapsed:.0f}s"
        self.progress_label.setText(summary)
        self.file_counter.setText("")
        self._refresh_queue_view()
        self._append_log(summary)
        output_dir = Path(self.output_edit.text())
        if output_dir.exists():
            self._append_log(f"输出目录: {output_dir}")

    def _thread_finished(self) -> None:
        self._thread = None
        self._worker = None

    def _append_log(self, message: str) -> None:
        if not hasattr(self, "log_edit"):
            return
        lowered = message.lower()
        color = "#d9e4f1"
        if "失败" in message or "error" in lowered or "failed" in lowered:
            color = "#ff9b9b"
        elif "完成" in message or "success" in lowered or "已翻译" in message:
            color = "#9be1b0"
        elif "警告" in message or "warn" in lowered:
            color = "#ffd18a"
        self.log_edit.append(f'<span style="color:{color}">{html.escape(message)}</span>')
        self.log_edit.ensureCursorVisible()

    def closeEvent(self, event) -> None:
        if self._thread and self._thread.isRunning():
            self._cancel.set()
            QMessageBox.information(self, "正在停止", "当前任务正在停止，请等待处理线程退出后再关闭窗口。")
            event.ignore()
            return
        if self._gpu_monitor:
            self._gpu_monitor.stop()
        logging.getLogger().removeHandler(self._log_handler)
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("ASR Pipeline")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)
    window = AsrWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
