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

from PySide6.QtCore import (
    QByteArray,
    QEvent,
    QObject,
    QSettings,
    QThread,
    QTimer,
    Qt,
    QUrl,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QFontMetrics,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QFileDialog,
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
    DEFAULT_TRANSLATION_MODEL_PATH,
    PipelineConfig,
)
from src.main import run_one_video
from src.monitor import GpuMonitor
from src.translator import (
    TranslateConfig,
    TranslationError,
    create_translator,
    iso_to_player_suffix,
    translate_srt,
    translate_srt_with_outputs,
    parse_srt,
    write_bilingual_srt,
    write_translation_srt,
)
from src.ui_theme import (
    APP_STYLE,
    ArrowSpinBox,
    CheckBox,
    ChevronComboBox,
    DropZone,
    field_label,
    hairline,
    inline,
    log_level_color,
    panel,
    section,
    section_label,
    status_colors,
)
from src.utils import find_executable, is_srt_valid


logger = logging.getLogger(__name__)


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
    "Index-Translate 官方 API (免费·35B)": "index_api",
    "本地 Index-Translate 模型 (离线)": "llm",
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
# Longest strings the 状态 column has to show. The column is sized from the
# font against these instead of a hard-coded pixel count, which clipped
# "完成 · 已翻译" at 92 px and made every finished row read "完成 · 已…".
STATUS_SAMPLES = (
    "完成 · 已翻译",
    "完成 · 1234 段",
    "等待翻译",
    "已有字幕",
    "翻译失败",
)
# Padding for Qt's own cell margins inside a table cell.
STATUS_CELL_PADDING = 26
# Height budget for the log strip when collapsed / expanded.
LOG_VIEW_COLLAPSED = 62
LOG_VIEW_EXPANDED = 260
# Height used when the log has nothing to show yet — the strip then costs a
# title row instead of a dead terminal box.
LOG_VIEW_EMPTY = 0
# One visible line plus the widget's own frame, so a single startup message
# does not reserve a three-line terminal.
LOG_LINE_HEIGHT = 20
LOG_COLLAPSED_LINES = 3


def _collapsed_log_height() -> int:
    """Height for the collapsed strip: a peek at the tail, not a fixed box."""
    return LOG_LINE_HEIGHT * LOG_COLLAPSED_LINES


def status_column_width(font: QFont) -> int:
    """Pixel width that fits every status label at the current font and DPI."""
    metrics = QFontMetrics(font)
    widest = max(metrics.horizontalAdvance(text) for text in STATUS_SAMPLES)
    return widest + STATUS_CELL_PADDING

# Window geometry persistence.
SETTINGS_ORG = "Subtitle Maker"
SETTINGS_APP = "Subtitle Maker"


class ProgressCell(QWidget):
    """A per-file progress bar, optically centred inside its queue cell.

    ``setCellWidget`` hands the widget the full cell rect, so the bare bar ran
    edge to edge and sat flush against the top separator — in a 28 px row that
    left 22 px of dead space below it and made the bar read as a divider rather
    than a gauge. Wrapping it puts the centring in the geometry instead of
    leaving it to the item delegate's padding, and leaves
    :data:`COL_PROGRESS` at a stable width regardless of row height.
    """

    # Inset on each side, so the bar reads as a centred slot with equal
    # breathing room rather than a rule pinned to the column edges. The cell
    # widget itself already arrives inset ~8px, so this lands the bar at 51px
    # inside an 84px column with 16/17 gutters.
    SIDE_INSET = 8

    def __init__(self, bar: QProgressBar, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        # Equal side insets centre the bar horizontally; AlignVCenter centres it
        # vertically inside the row. Deliberately no addStretch(): a stretch and
        # an Expanding widget split the leftover space evenly, which left the
        # bar noticeably shorter than the available track.
        layout.setContentsMargins(self.SIDE_INSET, 0, self.SIDE_INSET, 0)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(bar)


class QueueTable(QTableWidget):
    """Row table that doubles as a drop target — files can land anywhere on it."""

    files_dropped = Signal(list)

    _ACTIVE_STYLE = (
        "QTableWidget { border: 1px dashed #a86f10; background: #fdf4e3; }"
    )

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


class DurationProbe(QObject):
    """Reads media durations off the UI thread.

    ``ffprobe`` costs ~145 ms per file and blocks for up to 15 s on a damaged
    file, so probing it inline froze the window while a folder was added.
    """

    done = Signal(str, str)  # (path, formatted duration or "?")

    def __init__(self, paths: list[Path], *, parent=None, ffprobe_bin: str | None = None) -> None:
        super().__init__(parent)
        self._paths = paths
        self._ffprobe_bin = ffprobe_bin
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, name="duration-probe", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until probing finishes. Used on window close only."""
        return self._done.wait(timeout)

    def _format_duration(self, path: Path) -> str:
        try:
            result = subprocess.run(
                [find_executable("ffprobe", self._ffprobe_bin), "-v", "error",
                 "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(path)],
                capture_output=True,
                text=True,
                timeout=15,
            )
            seconds = float(result.stdout.strip())
            hours, minutes = divmod(int(seconds), 3600)
            minutes, seconds = divmod(minutes, 60)
            return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{seconds:02d}s"
        except FileNotFoundError as exc:
            logger.error(f"ffprobe unavailable: {exc}")
            return "?"
        except Exception:
            return "?"

    def _run(self) -> None:
        try:
            for path in self._paths:
                if self._done.is_set():
                    break
                self.done.emit(str(path), self._format_duration(path))
        finally:
            self._done.set()


STAGE_LABELS = {
    "extracting": "提取音频",
    "splitting": "切分音频",
    "loading_model": "加载模型",
    "transcribing": "语音识别",
    "writing": "写出字幕",
    "translating": "翻译字幕",
    "skipped": "已跳过",
}


class PipelineWorker(QObject):
    log = Signal(str)
    # (overall_percent, file_percent, row_index, label_index, stage_label)
    #
    # row_index is the queue row whose bar just moved; label_index is the file
    # the status line names. They differ while the translation thread reports
    # on an earlier file than the one ASR is on.
    progress = Signal(float, float, int, int, str)
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
        # Built on first translation and reused, so the HTTP session and its
        # connection pool survive across files in the same run.
        self._translator = None
        # Per-file 0..1 completion. The overall bar is the mean of these, which
        # is the only model that stays monotonic when ASR and translation run
        # concurrently on different files.
        self._file_progress: dict[int, float] = {}
        self._progress_lock = threading.Lock()
        self._index_by_path: dict[str, int] = {
            str(path): index for index, path in enumerate(paths)
        }
        # How much of a file's bar ASR fills before translation takes over.
        # Translation is a real second half of the work and used to be free.
        self._asr_ceiling = 1.0
        # Index of the file the ASR loop is on, plus the stage it is in, so a
        # translation report on an earlier file cannot steal the status line.
        self._active_index = -1
        self._active_stage = "transcribing"

    # ------------------------------------------------------------------
    # Progress bookkeeping
    # ------------------------------------------------------------------

    def _set_file_progress(
        self, file_index: int, fraction: float, stage: str, *, speak: bool = True
    ) -> None:
        """Record a file's completion and emit the recomputed overall bar.

        ``speak`` is False for background work (a translation thread on an
        earlier file): its row still advances, but the status line keeps
        describing what the ASR loop is doing.
        """
        if not 0 <= file_index < len(self.paths):
            return
        total = len(self.paths)
        with self._progress_lock:
            previous = self._file_progress.get(file_index, 0.0)
            self._file_progress[file_index] = max(previous, min(1.0, fraction))
            overall = sum(self._file_progress.values()) / total * 100
            file_percent = self._file_progress[file_index] * 100

        if speak or self._active_index < 0:
            label_index, label_stage = file_index, stage
        else:
            label_index, label_stage = self._active_index, self._active_stage
        self.progress.emit(overall, file_percent, file_index, label_index, label_stage)

    def _asr_progress(self, file_index: int, fraction: float, stage: str) -> None:
        """ASR covers the first slice of a file, translation the rest."""
        self._active_stage = stage
        self._set_file_progress(file_index, fraction * self._asr_ceiling, stage)

    def _translate_progress(self, media_path: Path, fraction: float) -> None:
        """Translation fills whatever ASR left behind."""
        index = self._index_by_path.get(str(media_path))
        if index is None:
            return
        if self._needs_asr(media_path):
            start, span = self._asr_ceiling, 1.0 - self._asr_ceiling
        else:
            # Nothing to transcribe — translation is this file's whole job.
            start, span = 0.0, 1.0
        self._set_file_progress(
            index, start + span * fraction, "translating", speak=False
        )

    def _needs_asr(self, media_path: Path) -> bool:
        return self.statuses.get(str(media_path)) == "asr"

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
        # A dropped .srt IS the source text, never a produced output. Judge it
        # on its own path; checking the output dir made a fresh drop look for
        # translations that do not exist yet.
        if media_path.suffix.lower() == ".srt":
            if not target_lang:
                return "done"
            return "translate" if is_srt_valid(media_path) else "asr"
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
        provider, error = self._provider()
        if provider is None:
            raise TranslationError(error)
        translate_srt_with_outputs(
            srt_path,
            self.config.translate_to,
            provider=provider,
            source_lang=source_lang,
            swap_subtitles=self.config.swap_subtitles,
            progress_callback=(
                lambda fraction: self._translate_progress(media_path, fraction)
            ),
        )

    def _provider(self):
        """Return (provider, error). Built once so sessions/pools are reused."""
        if self._translator is not None:
            return self._translator, None
        try:
            self._translator = create_translator(
                self.config.translation_provider,
                proxy=self.config.translation_proxy,
            )
        except Exception as exc:
            return None, f"翻译后端初始化失败: {exc}"
        return self._translator, None

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
        self._file_progress = {}
        # The local LLM backend translates only after ALL ASR work is done so
        # the translation model gets exclusive VRAM (no competition with
        # Whisper workers). Web backends keep the concurrent queue.
        llm_mode = (
            self.config.translation_provider == "llm" and bool(self.config.translate_to)
        )
        # Translation owns the tail of each file's bar, so ASR stops short of
        # 100% instead of flashing full and then sitting there.
        if llm_mode:
            self._asr_ceiling = 0.85
        elif self.config.translate_to:
            self._asr_ceiling = 0.75
        else:
            self._asr_ceiling = 1.0
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
            self._active_index = index
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
                # The ASR/translation split depends on this, and it is decided
                # here rather than by the (possibly stale) snapshot the window
                # handed over when the queue was built.
                self.statuses[str(media_path)] = status
                self.log.emit(f"检测: {media_path.name} → {status}")

                if status == "done":
                    self.file_status.emit(str(media_path), "已有字幕")
                    done_count += 1
                    self._set_file_progress(index, 1.0, "skipped")
                    continue

                if status == "direct_srt" and not self.config.translate_to:
                    self.file_status.emit(str(media_path), "已有字幕")
                    done_count += 1
                    self._set_file_progress(index, 1.0, "skipped")
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
                    # The file is queued, not finished — the translation
                    # thread owns the rest of its bar.
                    continue

                # A subtitle file must never reach the ASR stage: there is no
                # audio in it, and ffmpeg fails with the confusing
                # "Output file does not contain any stream". check_existing_subs
                # inspects the *output* directory, so a freshly dropped .srt can
                # fall through here when the expected outputs do not exist yet.
                if media_path.suffix.lower() == ".srt":
                    self.file_status.emit(str(media_path), "失败")
                    self.log.emit(
                        f"跳过 {media_path.name}: 字幕文件缺少可翻译内容或状态判定异常"
                    )
                    failed_count += 1
                    self._set_file_progress(index, 1.0, "skipped")
                    continue

                self.file_status.emit(str(media_path), "处理中")
                asr_config = copy.copy(self.config)
                asr_config.translate_to = ""
                detected_language: str | None = None

                def capture_detected_language(stage, value, _total, _index=index):
                    nonlocal detected_language
                    if stage == "detected_language":
                        detected_language = value

                def report(stage, current, target, _index=index):
                    capture_detected_language(stage, current, target)
                    if stage == "detected_language":
                        return
                    try:
                        fraction = float(current) / max(float(target), 1e-9)
                    except (TypeError, ValueError, ZeroDivisionError):
                        return
                    self._asr_progress(_index, fraction, stage)

                ok, segment_count, error = run_one_video(
                    asr_config,
                    media_path,
                    progress_callback=report,
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
                        # Land exactly on the ASR/translation boundary.
                        self._set_file_progress(
                            index, self._asr_ceiling, "transcribing"
                        )
                    else:
                        self.file_status.emit(str(media_path), f"完成 · {segment_count} 段")
                        done_count += 1
                        self._set_file_progress(index, 1.0, "transcribing")
                else:
                    if self.cancel_event.is_set():
                        self.file_status.emit(str(media_path), "已停止")
                    else:
                        failed_count += 1
                        self.file_status.emit(str(media_path), "失败")
                        self.log.emit(f"处理失败: {media_path.name} — {error}")
                    # A failed file must not hold the bar below full forever.
                    self._set_file_progress(index, 1.0, "transcribing")
            except Exception as exc:
                failed_count += 1
                self.file_status.emit(str(media_path), "失败")
                logging.getLogger(__name__).exception("GUI pipeline failed for %s", media_path)
                self.log.emit(f"处理失败: {media_path.name} — {exc}")
                self._set_file_progress(index, 1.0, "transcribing")

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
            self.log.emit(
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
                # Empty translation_model_path = the bundled default weights.
                custom = (self.config.translation_model_path or "").strip()
                if custom:
                    self.log.emit(f"使用自定义翻译模型: {custom}")
                provider = LlmTranslator(custom) if custom else LlmTranslator()
                provider.translate("预热", source_lang="auto", target_lang="zh")
            except Exception as exc:
                self.log.emit(f"本地翻译模型加载失败: {exc}")
                for media_path, _srt, _lang in self._llm_pending:
                    self.file_status.emit(str(media_path), "翻译失败")
                    self._translate_progress(media_path, 1.0)
                    failed_count += 1
                self._llm_pending.clear()
            for media_path, srt_path, source_lang in self._llm_pending:
                if self.cancel_event.is_set():
                    break
                self.file_status.emit(str(media_path), "翻译中")
                self._active_index = self._index_by_path.get(str(media_path), -1)
                self._active_stage = "translating"
                try:
                    translate_srt_with_outputs(
                        srt_path,
                        self.config.translate_to,
                        provider=provider,
                        source_lang=source_lang,
                        swap_subtitles=self.config.swap_subtitles,
                        config=TranslateConfig.for_local_llm(),
                        progress_callback=(
                            lambda fraction, _path=media_path: (
                                self._translate_progress(_path, fraction)
                            )
                        ),
                    )
                except Exception as exc:
                    self.file_status.emit(str(media_path), "翻译失败")
                    self.log.emit(f"翻译失败: {media_path.name} — {exc}")
                    failed_count += 1
                else:
                    self.file_status.emit(str(media_path), "完成 · 已翻译")
                    done_count += 1
                self._translate_progress(media_path, 1.0)
            self._llm_pending.clear()
            # Return VRAM to the system once the translation pass ends.
            if provider is not None:
                provider.release()
        self._active_index = -1

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
        self._duration_probes: set[DurationProbe] = set()
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
        outer.setContentsMargins(20, 16, 20, 12)
        outer.setSpacing(12)
        outer.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._build_left_panel(), 3)
        body.addWidget(hairline(vertical=True))
        body.addWidget(self._build_settings_panel(), 2)
        outer.addLayout(body, 1)
        outer.addWidget(self._build_action_bar())
        self.setCentralWidget(root)
        self._refresh_queue_view()

    def _build_header(self) -> QWidget:
        """Flat masthead — no box. Identity on the left, provenance on the right."""
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(12)

        text_box = QVBoxLayout()
        text_box.setSpacing(1)
        title = QLabel("Subtitle Maker")
        title.setObjectName("appTitle")
        subtitle = QLabel("音视频转写 · 本地 GPU 加速 · SRT / TXT / Markdown")
        subtitle.setObjectName("appSubtitle")
        text_box.addWidget(title)
        text_box.addWidget(subtitle)

        badge = QLabel("LOCAL · GPU")
        badge.setObjectName("badge")

        layout.addLayout(text_box)
        layout.addStretch()
        layout.addWidget(badge, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return header

    def _build_left_panel(self) -> QWidget:
        """The one raised surface: intake, queue, and the log strip."""
        card, layout = panel()
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        # Section head: label left, live counts right — one line, no card title.
        head = QHBoxLayout()
        head.setSpacing(12)
        head.addWidget(section_label("处理队列"))
        self.queue_summary = QLabel("尚未添加文件")
        self.queue_summary.setObjectName("faint")
        head.addWidget(self.queue_summary)
        head.addStretch()
        layout.addLayout(head)

        self.file_table = QueueTable(4)
        self.file_table.setHorizontalHeaderLabels(["文件", "进度", "时长", "状态"])
        self.file_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.file_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.file_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.file_table.setAlternatingRowColors(True)
        self.file_table.verticalHeader().setVisible(False)
        self.file_table.verticalHeader().setDefaultSectionSize(28)
        self.file_table.verticalHeader().setMinimumSectionSize(28)
        header_view = self.file_table.horizontalHeader()
        header_view.setStretchLastSection(False)
        header_view.setSectionResizeMode(COL_FILE, QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(COL_PROGRESS, QHeaderView.ResizeMode.Fixed)
        header_view.setSectionResizeMode(COL_DURATION, QHeaderView.ResizeMode.Fixed)
        header_view.setSectionResizeMode(COL_STATUS, QHeaderView.ResizeMode.Fixed)
        self.file_table.setColumnWidth(COL_PROGRESS, 84)
        self.file_table.setColumnWidth(COL_DURATION, 76)
        # 状态 is the widest fixed column: a clipped status ("完成 · 已…") is a
        # dead end for the user, while a clipped file name still has a tooltip.
        self._sync_status_column_width()
        self.file_table.files_dropped.connect(self._add_paths_from_strings)
        self.file_table.doubleClicked.connect(self._open_selected_file)

        # Empty state and populated queue share one slot: dropping a file onto
        # either works, and the intake copy disappears once it is not needed.
        self.drop_zone = DropZone()
        self.drop_zone.files_dropped.connect(self._add_paths_from_strings)
        self.drop_zone.choose_requested.connect(self._choose_files)
        self.drop_card = self.drop_zone  # kept for the smoke test / compat
        # The intake page centres inside the stack instead of stretching, so an
        # empty queue reads as an invitation rather than a tall empty box.
        empty_host = QWidget()
        empty_layout = QVBoxLayout(empty_host)
        empty_layout.setContentsMargins(0, 0, 0, 0)
        empty_layout.addStretch()
        empty_layout.addWidget(self.drop_zone)
        empty_layout.addStretch()
        queue_host = QWidget()
        self.queue_stack = QStackedLayout(queue_host)
        self.queue_stack.setContentsMargins(0, 0, 0, 0)
        self.queue_empty = self.drop_zone
        self.queue_stack.addWidget(empty_host)
        self.queue_stack.addWidget(self.file_table)
        layout.addWidget(queue_host, 1)

        actions = QHBoxLayout()
        actions.setSpacing(8)
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
        layout.addLayout(actions)

        layout.addWidget(self._build_log_strip())
        return card

    def _build_log_strip(self) -> QWidget:
        """Collapsed terminal strip. Quiet by default, expandable on demand."""
        holder = QWidget()
        holder_layout = QVBoxLayout(holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        holder_layout.setSpacing(6)

        log_head = QHBoxLayout()
        log_head.setSpacing(8)
        log_title = QLabel("实时日志")
        log_title.setObjectName("logTitle")
        self.log_hint = QLabel("")
        self.log_hint.setObjectName("faint")
        self.expand_log_check = CheckBox("展开")
        self.expand_log_check.toggled.connect(self._set_log_expanded)
        log_head.addWidget(log_title)
        log_head.addWidget(self.log_hint, 1)
        log_head.addWidget(self.expand_log_check)
        holder_layout.addLayout(log_head)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setAcceptDrops(False)
        self.log_edit.setFont(QFont("Cascadia Mono", 9))
        self.log_edit.setMinimumHeight(_collapsed_log_height())
        self.log_edit.setMaximumHeight(_collapsed_log_height())
        holder_layout.addWidget(self.log_edit)

        self.log_card = holder
        self._log_lines = 0
        self._set_log_expanded(False)
        return holder

    def _build_settings_panel(self) -> QWidget:
        """One continuous list of borderless sections; only the scroll clips."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 0, 8, 0)
        layout.setSpacing(16)

        output_holder, output_layout = section("输出")
        output_row = QHBoxLayout()
        output_row.setSpacing(8)
        self.output_edit = QLineEdit(self.default_output_dir)
        self.output_edit.setAcceptDrops(False)
        self.output_edit.editingFinished.connect(self._on_output_directory_changed)
        browse = QPushButton("浏览")
        browse.clicked.connect(self._browse_output)
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(browse)
        output_layout.addLayout(output_row)

        formats = QHBoxLayout()
        formats.setSpacing(14)
        self.srt_check = CheckBox("SRT")
        self.srt_check.setChecked(True)
        self.txt_check = CheckBox("TXT")
        self.md_check = CheckBox("MD")
        for widget in (self.srt_check, self.txt_check, self.md_check):
            formats.addWidget(widget)
        formats.addStretch()
        output_layout.addLayout(formats)

        self.translate_check = CheckBox("翻译字幕")
        # Follow config.yaml (translate_to: "" disables translation) so the UI
        # does not advertise work the run will not do.
        self.translate_check.setChecked(
            bool(self.config.translate_to) if self.config else True
        )
        output_layout.addWidget(inline(self.translate_check), 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(output_holder)

        transcribe_holder, transcribe_layout = section("转写")
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(7)
        grid.setColumnMinimumWidth(0, 58)
        grid.setColumnStretch(1, 1)
        self.model_combo = ChevronComboBox()
        self.model_combo.addItems(list(MODEL_SIZES))
        self.model_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.model_combo.setCurrentText(self.config.model_size if self.config else "large-v3-turbo")
        self.workers_spin = ArrowSpinBox()
        self.workers_spin.setRange(1, DEFAULT_MAX_WORKERS)
        self.workers_spin.setValue(self.config.max_workers if self.config else DEFAULT_MAX_WORKERS)
        self.workers_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        self.workers_spin.setToolTip("启动时逐个尝试，直到模型加载失败；本次使用失败前的并发数")
        self.language_combo = ChevronComboBox()
        self.language_combo.addItems(list(LANGUAGE_MAP))
        self.language_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.language_combo.setCurrentText("自动检测")
        self.beam_spin = ArrowSpinBox()
        self.beam_spin.setRange(1, 10)
        self.beam_spin.setValue(self.config.beam_size if self.config else 5)
        self.beam_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        self.chunk_combo = ChevronComboBox()
        self.chunk_combo.setEditable(True)
        self.chunk_combo.addItems(["自动", "300", "600", "900", "1200", "1800", "3600"])
        self.chunk_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.vad_check = CheckBox("启用 VAD 语音检测")
        self.vad_check.setChecked(self.config.vad_filter if self.config else True)
        self._add_form_row(grid, 0, "模型", self.model_combo)
        self._add_form_row(grid, 1, "最大并发", self.workers_spin)
        self._add_form_row(grid, 2, "识别语言", self.language_combo)
        transcribe_layout.addLayout(grid)
        transcribe_layout.addWidget(inline(self.vad_check), 0, Qt.AlignmentFlag.AlignLeft)
        self.advanced_toggle = CheckBox("显示高级参数")
        self.advanced_toggle.setChecked(False)
        transcribe_layout.addWidget(inline(self.advanced_toggle), 0, Qt.AlignmentFlag.AlignLeft)
        self.advanced_settings = QWidget()
        advanced_grid = QGridLayout(self.advanced_settings)
        advanced_grid.setContentsMargins(0, 0, 0, 0)
        advanced_grid.setHorizontalSpacing(10)
        advanced_grid.setVerticalSpacing(7)
        advanced_grid.setColumnMinimumWidth(0, 58)
        advanced_grid.setColumnStretch(1, 1)
        self._add_form_row(advanced_grid, 0, "束搜索宽度", self.beam_spin)
        self._add_form_row(advanced_grid, 1, "切割时长", self.chunk_combo)
        self.advanced_settings.setVisible(False)
        self.advanced_toggle.toggled.connect(self.advanced_settings.setVisible)
        transcribe_layout.addWidget(self.advanced_settings)
        layout.addWidget(transcribe_holder)

        translate_holder, translate_layout = section("翻译")
        translate_grid = QGridLayout()
        translate_grid.setHorizontalSpacing(10)
        translate_grid.setVerticalSpacing(7)
        translate_grid.setColumnMinimumWidth(0, 58)
        translate_grid.setColumnStretch(1, 1)
        self.translator_combo = ChevronComboBox()
        self.translator_combo.addItems(list(TRANSLATOR_MAP))
        self.translator_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        configured_provider = self.config.translation_provider if self.config else "bing"
        self.translator_combo.setCurrentText(
            next(
                (label for label, value in TRANSLATOR_MAP.items() if value == configured_provider),
                "Microsoft Edge Translator (免费)",
            )
        )
        self.translate_combo = ChevronComboBox()
        self.translate_combo.addItems(list(TRANSLATE_MAP))
        self.translate_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        configured_target = self.config.translate_to if self.config else "zh"
        self.translate_combo.setCurrentText(
            next(
                (label for label, value in TRANSLATE_MAP.items() if value == configured_target),
                "中文 (zh)",
            )
        )
        self.swap_check = CheckBox("生成单语译文、双语字幕和原始字幕")
        self.swap_check.setChecked(
            self.config.swap_subtitles if self.config else True
        )
        self.proxy_edit = QLineEdit(self.config.translation_proxy if self.config else "")
        self.proxy_edit.setPlaceholderText("127.0.0.1:7897")
        self.proxy_edit.setToolTip("Legacy GTX 必须使用代理，例如 127.0.0.1:7897")
        self.llm_model_edit = QLineEdit(
            (self.config.translation_model_path if self.config else "") or ""
        )
        self.llm_model_edit.setPlaceholderText(DEFAULT_TRANSLATION_MODEL_PATH)
        self.llm_model_edit.setToolTip(
            "本地翻译模型权重：填 .gguf 文件路径，或含 .gguf 的目录。留空使用内置默认。\n"
            "例如 models/index-translate-9b 或 D:\\models\\Index-Translate-9B.Q4_K_M.gguf"
        )
        self._add_form_row(translate_grid, 0, "后端", self.translator_combo)
        self._add_form_row(translate_grid, 1, "目标语言", self.translate_combo)
        self.proxy_label = self._add_form_row(translate_grid, 2, "GTX 代理", self.proxy_edit)
        llm_model_btn = QPushButton("浏览…")
        llm_model_btn.setFixedWidth(66)
        llm_model_btn.setToolTip("选择 .gguf 文件或模型目录")
        llm_model_btn.clicked.connect(self._browse_llm_model)
        self.llm_model_btn = llm_model_btn
        # Input + browse button share one row; inline() is checkbox-specific
        # (it pins the widget to its natural width) and would break the edit.
        llm_model_row = QWidget()
        llm_model_layout = QHBoxLayout(llm_model_row)
        llm_model_layout.setContentsMargins(0, 0, 0, 0)
        llm_model_layout.setSpacing(6)
        llm_model_layout.addWidget(self.llm_model_edit, 1)
        llm_model_layout.addWidget(llm_model_btn, 0)
        self.llm_model_label = self._add_form_row(
            translate_grid, 3, "本地模型", llm_model_row
        )
        translate_layout.addLayout(translate_grid)
        translate_layout.addWidget(inline(self.swap_check), 0, Qt.AlignmentFlag.AlignLeft)
        self.translator_combo.currentTextChanged.connect(self._on_translator_changed)
        self._update_gtx_controls()
        self.translate_card = translate_holder
        self.translate_check.toggled.connect(self._set_translation_enabled)
        self.language_combo.currentTextChanged.connect(self._on_source_language_changed)
        self._set_translation_enabled(self.translate_check.isChecked())
        layout.addWidget(translate_holder)

        layout.addWidget(self._build_gpu_section())
        layout.addStretch(1)
        scroll.setWidget(content)
        return scroll

    def _build_gpu_section(self) -> QWidget:
        """Telemetry, not a feature: a compact three-row block, no card."""
        holder, layout = section("GPU 状态")
        self.gpu_state_label = QLabel("检测中…")
        self.gpu_state_label.setObjectName("faint")
        layout.addWidget(self.gpu_state_label)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        grid.setColumnMinimumWidth(0, 44)
        grid.setColumnMinimumWidth(1, 96)
        grid.setColumnStretch(2, 1)
        self.gpu_util_value, self.gpu_util_bar = self._add_gpu_metric(grid, 0, "利用率")
        self.gpu_memory_value, self.gpu_memory_bar = self._add_gpu_metric(grid, 1, "显存")
        self.gpu_temp_value, self.gpu_temp_bar = self._add_gpu_metric(grid, 2, "温度")
        self.gpu_temp_bar.setMaximum(110)
        layout.addLayout(grid)
        return holder

    @staticmethod
    def _add_gpu_metric(grid: QGridLayout, row: int, name: str) -> tuple[QLabel, QProgressBar]:
        name_label = QLabel(name)
        name_label.setObjectName("fieldLabel")
        value_label = QLabel("—")
        value_label.setObjectName("meterValue")
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

    @staticmethod
    def _add_form_row(grid: QGridLayout, row: int, label: str, widget: QWidget) -> QLabel:
        label_widget = field_label(label)
        label_widget.setMinimumWidth(58)
        grid.addWidget(label_widget, row, 0)
        grid.addWidget(widget, row, 1)
        return label_widget

    def _build_action_bar(self) -> QWidget:
        """One flat bar: state on the left, commitment on the right."""
        bar = QWidget()
        layout = QVBoxLayout(bar)
        layout.setContentsMargins(4, 10, 4, 0)
        layout.setSpacing(8)

        self.progress = QProgressBar()
        self.progress.setObjectName("edge")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(3)
        layout.addWidget(self.progress)

        row = QHBoxLayout()
        row.setSpacing(10)
        state_box = QVBoxLayout()
        state_box.setSpacing(1)
        self.progress_label = QLabel("准备就绪")
        self.progress_label.setObjectName("status")
        self.run_summary = QLabel("")
        self.run_summary.setObjectName("faint")
        state_box.addWidget(self.progress_label)
        state_box.addWidget(self.run_summary)
        self.file_counter = QLabel("")
        self.file_counter.setObjectName("meterValue")
        self.stop_button = QPushButton("停止")
        self.stop_button.setObjectName("danger")
        self.stop_button.setEnabled(False)
        self.stop_button.setVisible(False)
        self.stop_button.clicked.connect(self._stop)
        self.start_button = QPushButton("开始转写")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self._start)

        row.addLayout(state_box)
        row.addStretch()
        row.addWidget(self.file_counter)
        row.addWidget(self.stop_button)
        row.addWidget(self.start_button)
        layout.addLayout(row)

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
        ):
            widget.toggled.connect(self._refresh_run_summary)
        self.translate_combo.currentTextChanged.connect(self._refresh_run_summary)
        return bar

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
        # Page 0 is the intake empty state, page 1 the queue itself; the stack
        # owns visibility, so the drop zone is never toggled by hand.
        self.queue_stack.setCurrentIndex(1 if self.paths else 0)
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
        if hasattr(self, "start_button"):
            self.start_button.setEnabled(can_edit and bool(self.paths))

    def _set_log_expanded(self, expanded: bool) -> None:
        """Collapsed height follows the content: one line of log stays one line."""
        if expanded:
            height = LOG_VIEW_EXPANDED
        else:
            height = LOG_VIEW_EMPTY if not self._log_lines else _collapsed_log_height()
        self.log_edit.setMinimumHeight(height)
        self.log_edit.setMaximumHeight(height)
        self.expand_log_check.setText("收起" if expanded else "展开")
        self.expand_log_check.setEnabled(self._log_lines > 1 or expanded)

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
        provider = TRANSLATOR_MAP.get(self.translator_combo.currentText())
        enabled = self.translate_check.isChecked()
        # Both web backends need a proxy on networks that cannot reach them
        # directly, so the row follows whichever is selected.
        needs_proxy = enabled and provider in ("gtx", "index_api")
        self.proxy_edit.setEnabled(needs_proxy)
        self.proxy_edit.setVisible(needs_proxy)
        self.proxy_label.setVisible(needs_proxy)
        self.proxy_label.setText(
            "网络代理" if provider == "index_api" else "GTX 代理"
        )
        # The custom-weights row only makes sense for the local GGUF backend.
        is_llm = enabled and provider == "llm"
        self.llm_model_edit.setVisible(is_llm)
        self.llm_model_btn.setVisible(is_llm)
        self.llm_model_label.setVisible(is_llm)

    def _browse_llm_model(self) -> None:
        """Pick a .gguf file or a directory holding one, for the local backend."""
        start = self.llm_model_edit.text().strip() or DEFAULT_TRANSLATION_MODEL_PATH
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择翻译模型（.gguf 文件）",
            start,
            "GGUF 模型 (*.gguf);;所有文件 (*)",
        )
        if not path:
            # User cancelled: fall back to picking the enclosing directory.
            path = QFileDialog.getExistingDirectory(
                self, "选择包含 .gguf 的目录", start
            )
            if not path:
                return
        self.llm_model_edit.setText(path)
        self.llm_model_btn.setToolTip(f"已选择: {path}")

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
        provider = TRANSLATOR_MAP.get(label)
        if provider in ("gtx", "index_api") and not self.proxy_edit.text().strip():
            who = "Index-Translate 官方 API" if provider == "index_api" else "Legacy GTX"
            QMessageBox.information(
                self,
                f"{who} 需要代理",
                "该翻译接口需要通过代理访问。请输入代理地址，例如 127.0.0.1:7897。",
            )
            proxy, accepted = QInputDialog.getText(
                self,
                f"添加 {who} 代理",
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
        added: list[Path] = []
        for path in self._expand_dropped_paths(raw_paths):
            before = len(self.paths)
            self._add_file(path)
            if len(self.paths) > before:
                added.append(path)
        # Probe durations in one background pass rather than per file inline.
        self._probe_durations(added)

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
        progress_cell, progress_bar = self._make_progress_cell()
        self.file_table.setCellWidget(row, COL_PROGRESS, progress_cell)
        self._progress_bars[str(path)] = progress_bar
        duration_item = QTableWidgetItem("—" if path.suffix.lower() == ".srt" else "读取中…")
        duration_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        duration_item.setFont(QFont("Cascadia Mono", 9))
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
        background, foreground = status_colors(item.text())
        item.setBackground(QColor(background))
        item.setForeground(QColor(foreground))
        # Narrow windows can still elide the text; the tooltip keeps the full
        # reason reachable instead of hiding it.
        item.setToolTip(item.text())

    def _sync_status_column_width(self) -> None:
        """Size 状态 from the current font metrics, not a frozen pixel count."""
        self.file_table.setColumnWidth(
            COL_STATUS, status_column_width(self.file_table.font())
        )

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802 (Qt naming)
        # Moving the window to a monitor with a different scale factor, or a
        # system font change, invalidates the measurement taken at build time.
        if event.type() in (
            QEvent.Type.FontChange,
            QEvent.Type.ApplicationFontChange,
        ):
            self._sync_status_column_width()
        super().changeEvent(event)

    def _probe_durations(self, paths: list[Path]) -> None:
        """Fill in media durations without blocking the UI thread."""
        pending = [p for p in paths if p.suffix.lower() != ".srt"]
        if not pending:
            return

        ffprobe_bin = getattr(getattr(self, "config", None), "ffprobe_path", "") or None
        probe = DurationProbe(pending, parent=self, ffprobe_bin=ffprobe_bin)
        self._duration_probes.add(probe)

        def on_done(raw_path: str, text: str) -> None:
            self._set_duration_cell(raw_path, text)
            if probe._done.is_set():
                self._duration_probes.discard(probe)

        probe.done.connect(on_done)
        probe.start()

    def _set_duration_cell(self, raw_path: str, text: str) -> None:
        row = self._row_by_path.get(raw_path)
        if row is None or row >= self.file_table.rowCount():
            return
        item = self.file_table.item(row, COL_DURATION)
        if item is not None:
            item.setText(text)
            item.setFont(QFont("Cascadia Mono", 9))

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

    @staticmethod
    def _make_progress_cell() -> tuple[QWidget, QProgressBar]:
        """Build one queue progress cell; returns the cell and the bare bar.

        The bar is returned separately because ``_progress_bars`` maps paths to
        it for :meth:`_update_progress`, and the widget index rebuild has to
        dig it back out of whatever wrapper the cell holds.
        """
        bar = QProgressBar()
        bar.setObjectName("thin")
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(False)
        # Fixed vertically so the cell centres it rather than stretching it.
        bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return ProgressCell(bar), bar

    def _rebuild_row_index(self) -> None:
        self._row_by_path.clear()
        self._progress_bars.clear()
        for row in range(self.file_table.rowCount()):
            path = self.file_table.item(row, COL_FILE).data(Qt.ItemDataRole.UserRole)
            self._row_by_path[path] = row
            self._progress_bars[path] = self._cell_progress_bar(
                self.file_table.cellWidget(row, COL_PROGRESS)
            )

    @staticmethod
    def _cell_progress_bar(widget: QWidget | None) -> QProgressBar | None:
        """Pull the bar out of a progress cell, whatever wraps it."""
        if isinstance(widget, QProgressBar):
            return widget
        if isinstance(widget, QWidget):
            return widget.findChild(QProgressBar)
        return None

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

    def _translated_without_asr(self, translate_to: str) -> list[str]:
        """Names that will be translated from an SRT instead of from audio.

        These never reach Whisper, so no detected language is available and a
        backend that refuses "auto" would only fail once the run is underway.
        """
        if not translate_to:
            return []
        output_text = self.output_edit.text().strip()
        output_dir = Path(output_text) if output_text else (
            self.paths[0].parent if self.paths else Path(".")
        )
        source_language = LANGUAGE_MAP[self.language_combo.currentText()]
        affected: list[str] = []
        for path in self.paths:
            if path.suffix.lower() == ".srt":
                affected.append(path.name)
                continue
            status = PipelineWorker.check_existing_subs(
                path,
                translate_to,
                self.swap_check.isChecked(),
                source_language,
                output_dir,
            )
            if status in ("translate", "translate_output"):
                affected.append(path.name)
        return affected

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
            # Edge rejects "auto" as a source language. ASR supplies the real
            # one, but a file that skips ASR has nothing to fall back on.
            no_asr = self._translated_without_asr(translate_to)
            if no_asr:
                preview = "、".join(no_asr[:5])
                if len(no_asr) > 5:
                    preview += f" 等 {len(no_asr)} 个文件"
                QMessageBox.warning(
                    self,
                    "直接翻译 SRT 需要源语言",
                    f"Microsoft Edge Translator 不支持自动识别源语言，以下文件不会经过语音识别：{preview}。\n"
                    "请在“识别语言”中选择实际源语言。",
                )
                return
        if translate_to and translation_provider in ("gtx", "index_api") and not translation_proxy:
            QMessageBox.warning(
                self,
                "缺少代理",
                "Legacy GTX 与 Index-Translate 官方 API 需要代理，请填写例如 127.0.0.1:7897。",
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
                "translation_model_path": self.llm_model_edit.text().strip(),
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
        self.progress_label.setText("准备中…")
        self.file_counter.setText("0%")
        for bar in self._progress_bars.values():
            bar.setValue(0)
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

    def _update_progress(
        self,
        overall: float,
        file_percent: float,
        row_index: int,
        label_index: int,
        stage: str,
    ) -> None:
        """Paint the overall bar, one row's bar, and the status line.

        ``overall`` is the mean of every file's completion, so the bar only ever
        climbs across the whole run. Deriving it from the current file's percent
        (what this used to do) made it reset to zero on every new file.
        """
        value = max(0, min(100, int(round(overall))))
        if value > self.progress.value():
            self.progress.setValue(value)

        label = STAGE_LABELS.get(stage, "处理中")
        name = (
            self.paths[label_index].name
            if 0 <= label_index < len(self.paths)
            else ""
        )
        self.progress_label.setText(f"{label} · {name}" if name else label)
        self.file_counter.setText(f"{self.progress.value()}%")

        row_value = max(0, min(100, int(round(file_percent))))
        if 0 <= row_index < len(self.paths):
            bar = self._progress_bars.get(str(self.paths[row_index]))
            # A translation thread can finish an older row after the ASR loop
            # moved on, so never walk a row backwards either.
            if bar is not None and row_value > bar.value():
                bar.setValue(row_value)

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
        self.file_counter.setText(f"{self.progress.value()}%")
        self._refresh_queue_view()
        self._append_log(summary)
        output_dir = Path(self.output_edit.text())
        if output_dir.exists():
            self._append_log(f"输出目录: {output_dir}")

    def _thread_finished(self) -> None:
        self._thread = None
        self._worker = None

    @staticmethod
    def _log_color(message: str) -> str:
        """Kept as the window-level entry point; the palette lives in ui_theme."""
        return log_level_color(message)

    def _append_log(self, message: str) -> None:
        if not hasattr(self, "log_edit"):
            return
        color = log_level_color(message)
        self.log_edit.append(f'<span style="color:{color}">{html.escape(message)}</span>')
        self.log_edit.ensureCursorVisible()
        self._log_lines += 1
        # Re-fit the collapsed strip on the first line (it grows from nothing
        # into a peek at the tail) and enable the toggle once there is more
        # than one line to reveal.
        if self._log_lines == 1 and not self.expand_log_check.isChecked():
            self._set_log_expanded(False)
        elif self._log_lines == 2:
            self.expand_log_check.setEnabled(True)

    def closeEvent(self, event) -> None:
        if self._thread and self._thread.isRunning():
            self._cancel.set()
            QMessageBox.information(self, "正在停止", "当前任务正在停止，请等待处理线程退出后再关闭窗口。")
            event.ignore()
            return
        if self._gpu_monitor:
            self._gpu_monitor.stop()
        # Duration probes are daemon threads reading ffprobe; give them a
        # moment so they are not killed mid-write while the window tears down.
        for probe in list(self._duration_probes):
            probe.wait(timeout=1.0)
        self._duration_probes.clear()
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
