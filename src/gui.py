#!/usr/bin/env python
"""
Subtitle Maker desktop UI built with PySide6.

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
import re
from queue import Queue
import sys
import subprocess
import threading
import time
from pathlib import Path

if __name__ == "__main__" and str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QByteArray, QPoint, QSettings, QThread, QTimer, Qt, QObject, QUrl, Signal
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QPainter,
    QPolygon,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
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
    translate_srt,
    translate_srt_with_outputs,
    parse_srt,
    write_bilingual_srt,
    write_translation_srt,
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
QLabel#title { color: #183153; font-size: 14pt; font-weight: 600; }
QLabel#subtitle { color: #607590; font-size: 9pt; }
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
    padding: 4px 8px;
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
QProgressBar#thin { min-height: 6px; max-height: 6px; }
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
    "本地 Hy-MT2 模型 (离线)": "llm",
}
MEDIA_SUFFIXES = {*DEFAULT_MEDIA_EXTENSIONS, "srt"}
# CTranslate2 repos backing the model combo. Verified reachable Sep 2026.
HF_BASE_URL = "https://huggingface.co"
HF_MIRROR_BASE_URL = "https://hf-mirror.com"
MODEL_REPOS = {
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "medium": "Systran/faster-whisper-medium",
}
DEFAULT_MODEL_REPO = MODEL_REPOS["large-v3-turbo"]
# Every entry needs a matching repo above, otherwise there is nothing to offer
# users when the model is missing (see tests/test_gui_model_guard.py).
MODEL_SIZES = ("large-v3-turbo", "large-v3", "medium")
# Without these the model directory exists but every transcription fails.
MODEL_REQUIRED_FILES = ("config.json", "model.bin")
# Raised by PipelineConfig.validate() when the model directory is missing.
MODEL_PATH_ERROR_MARKER = "Model path does not exist"
# Queue table column indices — keep every row/column access in sync with these.
COL_FILE = 0
COL_PROGRESS = 1
COL_DURATION = 2
COL_STATUS = 3
# Height budget for the log strip when collapsed (card + inner log view).
LOG_CARD_COLLAPSED = 104
LOG_VIEW_COLLAPSED = 58
LOG_VIEW_EXPANDED = 280
# Window geometry persistence.
SETTINGS_ORG = "Subtitle Maker"
SETTINGS_APP = "Subtitle Maker"


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
        hint = QLabel("支持 MP4、M4A、MP3、WAV、FLAC、OGG、SRT，也可拖入整个文件夹")
        hint.setObjectName("dropHint")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setFixedHeight(22)
        choose = QPushButton("选择文件")
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


class QueueTable(QTableWidget):
    """Row table that doubles as a drop target — files can land anywhere on it."""

    files_dropped = Signal(list)

    _ACTIVE_STYLE = "QTableWidget { border: 2px dashed #2f76c7; background: #f0f7ff; }"

    def __init__(self, column_count: int, parent: QWidget | None = None):
        super().__init__(0, column_count, parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setDragDropOverwriteMode(False)
        self.setDefaultDropAction(Qt.DropAction.CopyAction)
        self._drag_active = False

    @staticmethod
    def _local_paths(mime_data) -> list[str]:
        return [url.toLocalFile() for url in mime_data.urls() if url.isLocalFile()]

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            self._set_drag_active(True)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self._set_drag_active(False)
        event.accept()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = self._local_paths(event.mimeData())
        self._set_drag_active(False)
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    def _set_drag_active(self, active: bool) -> None:
        if self._drag_active == active:
            return
        self._drag_active = active
        self.setStyleSheet(self._ACTIVE_STYLE if active else "")
        self.style().unpolish(self)
        self.style().polish(self)


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
                required = (
                    Path(f"{base}.srt"),
                    Path(f"{base}.bilingual.srt"),
                )
                if source_lang != "auto":
                    required += (
                        Path(f"{base}.{iso_to_player_suffix(source_lang)}.srt"),
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

    def _translate_subtitles(
        self, media_path: Path, srt_path: Path, source_lang: str
    ) -> None:
        """Translate one completed SRT without blocking the ASR producer."""
        translate_srt_with_outputs(
            srt_path,
            self.config.translate_to,
            provider=create_translator(
                self.config.translation_provider,
                proxy=self.config.translation_proxy,
            ),
            source_lang=source_lang,
            swap_subtitles=self.config.swap_subtitles,
        )

    def _run_translation_queue(
        self,
        translation_queue: Queue[tuple[Path, Path, str] | None],
        results: Queue[tuple[Path, str, str]],
    ) -> None:
        """Consume subtitle jobs serially while ASR continues in the Qt worker."""
        while True:
            task = translation_queue.get()
            if task is None:
                return

            media_path, srt_path, source_lang = task
            if self.cancel_event.is_set():
                self.file_status.emit(str(media_path), "等待翻译")
                results.put((media_path, "deferred", ""))
                continue

            self.file_status.emit(str(media_path), "翻译中")
            try:
                self._translate_subtitles(media_path, srt_path, source_lang)
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
        # The local LLM backend translates only after ALL ASR work is done so
        # the translation model gets exclusive VRAM (no competition with
        # Whisper workers). Web backends keep the concurrent queue.
        llm_mode = (
            self.config.translation_provider == "llm" and bool(self.config.translate_to)
        )
        llm_pending: list[tuple[Path, Path, str]] = []  # deferred LLM jobs
        self._llm_pending = llm_pending
        translation_queue: Queue[tuple[Path, Path, str] | None] = Queue()
        translation_results: Queue[tuple[Path, str, str]] = Queue()
        translation_thread: threading.Thread | None = None

        if self.config.translate_to and not llm_mode:
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
                    if llm_mode:
                        llm_pending.append((media_path, srt_path, self.config.language))
                    else:
                        translation_queue.put((media_path, srt_path, self.config.language))
                    self.progress.emit(100.0, index, total)
                    continue

                self.file_status.emit(str(media_path), "处理中")
                asr_config = copy.copy(self.config)
                asr_config.translate_to = ""
                detected_language: str | None = None

                def capture_detected_language(stage, value, _total):
                    nonlocal detected_language
                    if stage == "detected_language":
                        detected_language = value

                ok, segment_count, error = run_one_video(
                    asr_config,
                    media_path,
                    progress_callback=lambda stage, current, target, i=index, t=total: (
                        capture_detected_language(stage, current, target)
                        if stage == "detected_language"
                        else self.progress.emit(
                            (current / max(target, 1)) * 100, i, t
                        )
                    ),
                    cancel_event=self.cancel_event,
                )
                if ok:
                    if self.config.translate_to:
                        srt_path = self.config.output_dir / f"{media_path.stem}.srt"
                        self.file_status.emit(str(media_path), "等待翻译")
                        if llm_mode:
                            llm_pending.append((
                                media_path,
                                srt_path,
                                detected_language or self.config.language,
                            ))
                        else:
                            translation_queue.put((
                                media_path,
                                srt_path,
                                detected_language or self.config.language,
                            ))
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

        # --- Deferred local-LLM translation pass (after all ASR) ---
        if llm_mode and not self.cancel_event.is_set() and self._llm_pending:
            self._log.emit(
                "转写完成，开始本地模型翻译（独占显存，"
                f"{len(self._llm_pending)} 个文件）"
            )
            provider = None
            try:
                from src.translator.llm import (
                    LlmTranslator,
                    wait_for_vram_release,
                )

                wait_for_vram_release(LlmTranslator.minimum_free_vram_bytes())
                provider = LlmTranslator()
                provider.translate("预热", source_lang="auto", target_lang="zh")
            except Exception as exc:
                self._log.emit(f"本地翻译模型加载失败: {exc}")
                for media_path, _srt, _lang in self._llm_pending:
                    self.file_status.emit(str(media_path), "翻译失败")
                    failed_count += 1
                self._llm_pending.clear()
            for media_path, srt_path, source_lang in self._llm_pending:
                if self.cancel_event.is_set():
                    break
                self.file_status.emit(str(media_path), "翻译中")
                try:
                    translate_srt_with_outputs(
                        srt_path,
                        self.config.translate_to,
                        provider=provider,
                        source_lang=source_lang,
                        swap_subtitles=self.config.swap_subtitles,
                    )
                except Exception as exc:
                    self.file_status.emit(str(media_path), "翻译失败")
                    self._log.emit(f"翻译失败: {media_path.name} — {exc}")
                    failed_count += 1
                else:
                    self.file_status.emit(str(media_path), "完成 · 已翻译")
                    done_count += 1
            self._llm_pending.clear()
            # Return VRAM to the system once the translation pass ends.
            if provider is not None:
                provider.release()

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
        self.setWindowTitle("Subtitle Maker")
        self.setMinimumSize(960, 620)
        self.setAcceptDrops(True)
        self._restore_geometry()

        self.paths: list[Path] = []
        self.file_status: dict[str, str] = {}
        self._row_by_path: dict[str, int] = {}
        self._progress_bars: dict[str, QProgressBar] = {}
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
        self._append_log("Subtitle Maker 已启动。拖入音视频或点击“选择文件”开始。")

    def _restore_geometry(self) -> None:
        saved = QSettings(SETTINGS_ORG, SETTINGS_APP).value("geometry")
        if isinstance(saved, QByteArray) and not saved.isEmpty():
            self.restoreGeometry(saved)
        else:
            self.resize(1180, 860)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API naming
        QSettings(SETTINGS_ORG, SETTINGS_APP).setValue("geometry", self.saveGeometry())
        super().closeEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt API naming
        """Window-wide fallback so files can land anywhere outside the table."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt API naming
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self._add_paths_from_strings(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

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
        layout.setContentsMargins(20, 10, 20, 10)
        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title = QLabel("Subtitle Maker")
        title.setObjectName("title")
        subtitle = QLabel("音视频转写 · 本地 GPU 加速 · SRT / TXT / Markdown")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        layout.addLayout(title_box)
        layout.addStretch()
        badge = QLabel("LOCAL  ·  GPU")
        badge.setObjectName("badge")
        layout.addWidget(badge, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return card

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # The big drop zone is only shown as an empty state. Once files are in
        # the queue the table itself takes over as the drop target, which hands
        # ~200px back to the queue instead.
        self.drop_card, drop_layout = _card("输入文件", "支持批量处理，自动检测同名字幕并支持断点续跑")
        self.drop_card.setMaximumHeight(200)
        self.drop_zone = DropZone()
        self.drop_zone.files_dropped.connect(self._add_paths_from_strings)
        self.drop_zone.choose_requested.connect(self._choose_files)
        drop_layout.addWidget(self.drop_zone)
        layout.addWidget(self.drop_card)

        list_card, list_layout = _card("处理队列")
        self.queue_summary = QLabel("尚未添加文件")
        self.queue_summary.setObjectName("muted")
        list_layout.addWidget(self.queue_summary)
        self.file_table = QueueTable(4)
        self.file_table.setHorizontalHeaderLabels(["文件", "进度", "时长", "状态"])
        self.file_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.file_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.file_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.file_table.setAlternatingRowColors(True)
        self.file_table.verticalHeader().setVisible(False)
        self.file_table.verticalHeader().setDefaultSectionSize(24)
        self.file_table.verticalHeader().setMinimumSectionSize(24)
        header = self.file_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(COL_FILE, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(COL_PROGRESS, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(COL_DURATION, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(COL_STATUS, QHeaderView.ResizeMode.Fixed)
        self.file_table.setColumnWidth(COL_PROGRESS, 72)
        self.file_table.setColumnWidth(COL_DURATION, 66)
        self.file_table.setColumnWidth(COL_STATUS, 96)
        self.file_table.files_dropped.connect(self._add_paths_from_strings)
        self.file_table.doubleClicked.connect(self._open_selected_file)
        queue_host = QWidget()
        self.queue_stack = QStackedLayout(queue_host)
        self.queue_empty = QLabel("拖入音视频 / SRT 文件（可拖整个文件夹），或点击“添加文件”")
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
        layout.addWidget(list_card, 1)

        # Compact log drawer: collapsed by default so the queue keeps its space.
        log_card = QFrame()
        log_card.setObjectName("card")
        self.log_card = log_card
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(14, 8, 14, 8)
        log_layout.setSpacing(6)
        log_header = QHBoxLayout()
        log_title = QLabel("实时日志")
        log_title.setObjectName("sectionTitle")
        self.log_hint = QLabel("")
        self.log_hint.setObjectName("muted")
        log_header.addWidget(log_title)
        log_header.addWidget(self.log_hint, 1)
        self.expand_log_check = QCheckBox("展开")
        self.expand_log_check.toggled.connect(self._set_log_expanded)
        log_header.addWidget(self.expand_log_check)
        log_layout.addLayout(log_header)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setAcceptDrops(False)
        self.log_edit.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_edit)
        log_card.setMaximumHeight(LOG_CARD_COLLAPSED)
        self._set_log_expanded(False)
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
        self.output_edit.setAcceptDrops(False)
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
        self.model_combo.addItems(list(MODEL_SIZES))
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

        # The GPU card takes a stretch factor so it grows into whatever space
        # the three settings cards leave behind — no empty gap at the bottom.
        layout.addWidget(self._build_gpu_card(), 1)
        scroll.setWidget(content)
        return scroll

    @staticmethod
    def _add_gpu_metric(grid: QGridLayout, row: int, name: str) -> tuple[QLabel, QProgressBar]:
        name_label = QLabel(name)
        name_label.setObjectName("muted")
        value_label = QLabel("—")
        value_label.setObjectName("gpuValue")
        value_label.setMinimumWidth(88)
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bar = QProgressBar()
        bar.setObjectName("thin")
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(False)
        bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        grid.addWidget(name_label, row, 0)
        grid.addWidget(value_label, row, 1)
        grid.addWidget(bar, row, 2)
        return value_label, bar

    def _build_gpu_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        card.setMinimumHeight(150)
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        heading = QLabel("GPU 状态")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)

        self.gpu_state_label = QLabel("GPU 检测中...")
        self.gpu_state_label.setObjectName("muted")
        layout.addWidget(self.gpu_state_label)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setColumnMinimumWidth(0, 48)
        grid.setColumnStretch(2, 1)
        self.gpu_util_value, self.gpu_util_bar = self._add_gpu_metric(grid, 0, "利用率")
        self.gpu_memory_value, self.gpu_memory_bar = self._add_gpu_metric(grid, 1, "显存")
        self.gpu_temp_value, self.gpu_temp_bar = self._add_gpu_metric(grid, 2, "温度")
        self.gpu_temp_bar.setMaximum(110)
        layout.addLayout(grid)
        layout.addStretch(1)
        return card

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
        if hasattr(self, "drop_card"):
            self.drop_card.setVisible(not self.paths)
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
        if hasattr(self, "start_button"):
            self.start_button.setEnabled(can_edit and bool(self.paths))

    def _set_log_expanded(self, expanded: bool) -> None:
        height = LOG_VIEW_EXPANDED if expanded else LOG_VIEW_COLLAPSED
        self.log_card.setMaximumHeight(16777215 if expanded else LOG_CARD_COLLAPSED)
        self.log_edit.setMinimumHeight(height)
        self.log_edit.setMaximumHeight(height)
        self.log_edit.setVisible(True)
        self.expand_log_check.setText("收起" if expanded else "展开")

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
        monitor = self._gpu_monitor
        if not (monitor and monitor.available):
            self.gpu_state_label.setText("未检测到 NVIDIA GPU")
            self._set_gpu_values(utilization=None, used_gb=None, total_gb=None, temperature=None)
            return

        snapshot = monitor.latest()
        if not snapshot:
            self.gpu_state_label.setText("GPU 初始化中...")
            self._set_gpu_values(utilization=None, used_gb=None, total_gb=None, temperature=None)
            return

        used_gb = snapshot.memory_used_mb / 1024
        total_gb = snapshot.memory_total_mb / 1024
        self.gpu_state_label.setText("")
        self._set_gpu_values(
            utilization=snapshot.utilization_pct,
            used_gb=used_gb,
            total_gb=total_gb,
            temperature=snapshot.temperature_c,
        )

    def _set_gpu_values(
        self,
        *,
        utilization: int | None,
        used_gb: float | None,
        total_gb: float | None,
        temperature: int | None,
    ) -> None:
        self.gpu_util_value.setText(f"{utilization}%" if utilization is not None else "—")
        self.gpu_util_bar.setValue(int(utilization or 0))
        if used_gb is not None and total_gb:
            self.gpu_memory_value.setText(f"{used_gb:.1f} / {total_gb:.1f} GB")
            self.gpu_memory_bar.setValue(int(used_gb / max(total_gb, 0.001) * 100))
        else:
            self.gpu_memory_value.setText("—")
            self.gpu_memory_bar.setValue(0)
        self.gpu_temp_value.setText(f"{temperature}°C" if temperature is not None else "—")
        self.gpu_temp_bar.setValue(int(temperature or 0))

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
                item = self.file_table.item(row, COL_STATUS)
                item.setText(status_labels[status])
                self._style_status_item(item)
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

    def _model_download_urls(self) -> tuple[str, str, str]:
        """(repo, Hugging Face url, hf-mirror url) for the selected model."""
        repo = MODEL_REPOS.get(self.model_combo.currentText(), DEFAULT_MODEL_REPO)
        return repo, f"{HF_BASE_URL}/{repo}", f"{HF_MIRROR_BASE_URL}/{repo}"

    @staticmethod
    def _model_missing_files(model_path: Path) -> list[str]:
        return [name for name in MODEL_REQUIRED_FILES if not (model_path / name).is_file()]

    def _report_missing_model(self, model_path: Path) -> None:
        """Model is unusable — explain where to get it instead of crashing later."""
        repo, hf_url, mirror_url = self._model_download_urls()
        missing = self._model_missing_files(model_path)
        if not model_path.exists():
            reason = "目录不存在"
        else:
            reason = "不完整，缺少 " + "、".join(missing)
        self._append_log(f"缺少转写模型: {model_path}（{reason}）")

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("缺少转写模型")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(
            f"没有找到可用的转写模型（{html.escape(reason)}）。<br><br>"
            f"当前模型：<b>{html.escape(self.model_combo.currentText())}</b><br>"
            f"期望目录：<code>{html.escape(str(model_path))}</code>"
        )
        box.setInformativeText(
            "下载后请把整个仓库放进上面的目录（目录名必须带 <b>-ct2</b> 后缀），"
            "并确保里面有 <code>config.json</code> 和 <code>model.bin</code>。<br><br>"
            f"Hugging Face 打不开时，用国内镜像 <code>hf-mirror.com</code>，路径完全相同："
            f"<br><a href=\"{html.escape(hf_url)}\">{html.escape(hf_url)}</a>"
            f"<br><a href=\"{html.escape(mirror_url)}\">{html.escape(mirror_url)}</a>"
        )
        for label in box.findChildren(QLabel):
            label.setOpenExternalLinks(True)
        download_button = box.addButton("打开下载页", QMessageBox.ButtonRole.AcceptRole)
        mirror_button = box.addButton("打开镜像下载页", QMessageBox.ButtonRole.ActionRole)
        box.addButton("关闭", QMessageBox.ButtonRole.RejectRole)
        box.exec()

        clicked = box.clickedButton()
        if clicked is download_button:
            QDesktopServices.openUrl(QUrl(hf_url))
        elif clicked is mirror_button:
            QDesktopServices.openUrl(QUrl(mirror_url))

    @staticmethod
    def _is_supported_file(path: Path) -> bool:
        return path.is_file() and path.suffix.lower().lstrip(".") in MEDIA_SUFFIXES

    @staticmethod
    def _collect_directory(directory: Path) -> list[Path]:
        found = [child for child in directory.rglob("*") if child.is_file()]
        return sorted(child for child in found if child.suffix.lower().lstrip(".") in MEDIA_SUFFIXES)

    def _expand_dropped_paths(self, raw_paths: list[str]) -> list[Path]:
        resolved: list[Path] = []
        for raw_path in raw_paths:
            path = Path(raw_path).expanduser().resolve()
            if path.is_dir():
                found = self._collect_directory(path)
                if found:
                    resolved.extend(found)
                    self._append_log(f"已展开目录: {path.name} → {len(found)} 个文件")
                else:
                    self._append_log(f"目录内没有可用的音视频 / SRT 文件: {path}")
            elif self._is_supported_file(path):
                resolved.append(path)
            else:
                self._append_log(f"跳过不支持的路径: {path.name or path}")
        return resolved

    def _add_paths_from_strings(self, raw_paths: list[str]) -> None:
        if self._is_running:
            self._append_log("警告: 任务运行中，先停止后再添加文件。")
            return
        for path in self._expand_dropped_paths(raw_paths):
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
        self.file_table.setItem(row, COL_FILE, name_item)
        progress_bar = QProgressBar()
        progress_bar.setObjectName("thin")
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        progress_bar.setTextVisible(False)
        self.file_table.setCellWidget(row, COL_PROGRESS, progress_bar)
        self._progress_bars[str(path)] = progress_bar
        duration_item = QTableWidgetItem("—" if path.suffix.lower() == ".srt" else self._get_duration(path))
        duration_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_table.setItem(row, COL_DURATION, duration_item)
        status_item = QTableWidgetItem(status_text)
        status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._style_status_item(status_item)
        self.file_table.setItem(row, COL_STATUS, status_item)
        self._row_by_path[str(path)] = row
        self._append_log(f"已添加: {path.name} → {status_text}")
        self._refresh_queue_view()

    @staticmethod
    def _style_status_item(item: QTableWidgetItem) -> None:
        """Colour-code the status cell so scanning the queue needs no reading."""
        text = item.text()
        if text.startswith("完成") or text == "已有字幕":
            background, foreground = "#e6f7ee", "#1c7a4b"
        elif text in {"处理中", "翻译中"}:
            background, foreground = "#e6f1ff", "#1d5fa8"
        elif "失败" in text:
            background, foreground = "#fdeaea", "#b23c3c"
        elif text == "已停止":
            background, foreground = "#fdf1e0", "#9a6410"
        else:
            background, foreground = "#f1f4f8", "#4f6280"
        item.setBackground(QColor(background))
        item.setForeground(QColor(foreground))

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
            path = Path(self.file_table.item(row, COL_FILE).data(Qt.ItemDataRole.UserRole))
            if path not in self.paths:
                continue
            self.file_table.removeRow(row)
            self.paths.remove(path)
            self.file_status.pop(str(path), None)
            self._progress_bars.pop(str(path), None)
        self._rebuild_row_index()
        self._refresh_queue_view()

    def _clear_files(self) -> None:
        self.paths.clear()
        self.file_status.clear()
        self._row_by_path.clear()
        self._progress_bars.clear()
        self.file_table.setRowCount(0)
        self._refresh_queue_view()

    def _rebuild_row_index(self) -> None:
        self._row_by_path.clear()
        self._progress_bars.clear()
        for row in range(self.file_table.rowCount()):
            path = self.file_table.item(row, COL_FILE).data(Qt.ItemDataRole.UserRole)
            self._row_by_path[path] = row
            widget = self.file_table.cellWidget(row, COL_PROGRESS)
            if isinstance(widget, QProgressBar):
                self._progress_bars[path] = widget

    def _open_selected_file(self) -> None:
        row = self.file_table.currentRow()
        if row < 0:
            return
        path = Path(self.file_table.item(row, COL_FILE).data(Qt.ItemDataRole.UserRole))
        target = self._output_path_for(path)
        if target.exists() and hasattr(os, "startfile"):
            os.startfile(str(target))
        elif path.exists() and hasattr(os, "startfile"):
            os.startfile(str(path))

    def _output_path_for(self, path: Path) -> Path:
        """Prefer the generated subtitle over the source media on double-click."""
        try:
            output_dir = Path(self.output_edit.text().strip() or path.parent)
        except Exception:
            output_dir = path.parent
        if path.suffix.lower() == ".srt":
            return path
        return output_dir / f"{path.stem}.srt"

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
        if (
            translate_to
            and translation_provider == "bing"
            and source_language == "auto"
            and any(path.suffix.lower() == ".srt" for path in self.paths)
        ):
            QMessageBox.warning(
                self,
                "直接翻译 SRT 需要源语言",
                "视频转写时可由 Whisper 自动识别；直接翻译 SRT 时，请在“识别语言”中选择实际源语言。",
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
            message = str(exc)
            match = re.search(
                rf"{MODEL_PATH_ERROR_MARKER}:\s*(.+)", message
            )
            if match:
                self._report_missing_model(Path(match.group(1).strip()))
                return
            QMessageBox.critical(self, "配置错误", message)
            return
        # The directory may exist while still being incomplete — catch it here
        # rather than failing deep inside the worker.
        if self._model_missing_files(config.model_path):
            self._report_missing_model(config.model_path)
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
        self.stop_button.setEnabled(running)
        self.stop_button.setVisible(running)
        for widget in self._settings_widgets:
            widget.setEnabled(not running)
        self.translate_card.setEnabled(not running)
        self._refresh_file_action_states()
        if not running:
            self._update_gtx_controls()

    def _update_progress(self, percent: float, file_index: int, total: int) -> None:
        value = max(0, min(100, int(percent)))
        self.progress.setValue(value)
        self.progress_label.setText(f"处理中 · {percent:.1f}%")
        self.file_counter.setText(f"{file_index + 1}/{total}")
        if 0 <= file_index < len(self.paths):
            bar = self._progress_bars.get(str(self.paths[file_index]))
            if bar is not None and bar.value() != value:
                bar.setValue(value)

    def _update_file_status(self, raw_path: str, status: str) -> None:
        self.file_status[raw_path] = status
        row = self._row_by_path.get(raw_path)
        if row is not None and row < self.file_table.rowCount():
            item = self.file_table.item(row, COL_STATUS)
            item.setText(status)
            self._style_status_item(item)
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
        color = self._log_color(message)
        self.log_edit.append(f'<span style="color:{color}">{html.escape(message)}</span>')
        self.log_edit.ensureCursorVisible()

    @staticmethod
    def _log_color(message: str) -> str:
        lowered = message.lower()
        failed_count_match = re.search(
            r"\bfailed\s*[:=]\s*(\d+)\b|\b(\d+)\s+failed\b",
            lowered,
        )
        has_failure = (
            "失败" in message
            or "error" in lowered
            or (
                "failed" in lowered
                and (
                    failed_count_match is None
                    or int(next(group for group in failed_count_match.groups() if group)) > 0
                )
            )
        )
        if has_failure:
            return "#ff9b9b"
        if "完成" in message or "success" in lowered or "已翻译" in message:
            return "#9be1b0"
        if "警告" in message or "warn" in lowered:
            return "#ffd18a"
        return "#d9e4f1"

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
    app.setApplicationName("Subtitle Maker")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)
    window = AsrWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
