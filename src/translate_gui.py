#!/usr/bin/env python
"""
Standalone subtitle translation UI built with PySide6.

Drag SRT files in, pick a backend (Microsoft Edge / Legacy GTX / local
Index-Translate GGUF model) and target language, then translate without running
the Subtitle Maker pipeline.

Usage:
    uv run python -m src.translate_gui
"""
from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

if __name__ == "__main__" and str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QStackedLayout,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.translator import (
    TranslationError,
    create_translator,
    iso_to_player_suffix,
    parse_srt,
    translate_srt_with_outputs,
)
from src.ui_theme import (
    APP_STYLE,
    CheckBox,
    ChevronComboBox,
    DropZone,
    inline,
    panel,
    section,
    status_colors,
)



SOURCE_LANGUAGE_MAP = {
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
TARGET_LANGUAGE_MAP = {
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
BACKEND_MAP = {
    "Microsoft Edge Translator (免费·联网)": "bing",
    "Legacy GTX (免费·联网·需代理)": "gtx",
    "本地 Index-Translate 模型 (离线·GGUF)": "llm",
}


class TranslationWorker(QObject):
    """Translate a list of SRT files sequentially in a background thread."""

    log = Signal(str)
    progress = Signal(int, int)          # completed, total
    file_status = Signal(str, str)       # path, status text
    finished = Signal(int, int, bool)    # done, failed, cancelled

    def __init__(
        self,
        paths: list[Path],
        backend: str,
        source_lang: str,
        target_lang: str,
        swap_subtitles: bool,
        proxy: str,
        cancel_event: threading.Event,
    ):
        super().__init__()
        self.paths = paths
        self.backend = backend
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.swap_subtitles = swap_subtitles
        self.proxy = proxy
        self.cancel_event = cancel_event

    def run(self) -> None:
        done = 0
        failed = 0
        total = len(self.paths)
        try:
            provider = create_translator(self.backend, proxy=self.proxy)
        except Exception as exc:
            self.log.emit(f"后端初始化失败: {exc}")
            self.finished.emit(0, total, False)
            return

        for index, path in enumerate(self.paths):
            if self.cancel_event.is_set():
                break
            self.file_status.emit(str(path), "翻译中")
            try:
                outputs = translate_srt_with_outputs(
                    path,
                    self.target_lang,
                    provider=provider,
                    source_lang=self.source_lang,
                    swap_subtitles=self.swap_subtitles,
                )
            except TranslationError as exc:
                failed += 1
                self.file_status.emit(str(path), "翻译失败")
                self.log.emit(f"翻译失败: {path.name} — {exc}")
            except Exception as exc:
                failed += 1
                self.file_status.emit(str(path), "翻译失败")
                logging.getLogger(__name__).exception("Translation failed for %s", path)
                self.log.emit(f"翻译失败: {path.name} — {exc}")
            else:
                done += 1
                translated, bilingual, original = outputs
                self.file_status.emit(str(path), "完成")
                self.log.emit(f"已翻译: {path.name} → {translated.name}")
                for extra in (bilingual, original):
                    if extra:
                        self.log.emit(f"  输出: {extra.name}")
            self.progress.emit(index + 1, total)

        self.finished.emit(done, failed, self.cancel_event.is_set())


class TranslateWindow(QMainWindow):
    """Drag-and-drop SRT translation window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("字幕翻译工具")
        self.setMinimumSize(720, 520)
        self.resize(860, 640)
        self.setAcceptDrops(True)

        self.paths: list[Path] = []
        self._row_by_path: dict[str, str] = {}
        self._cancel = threading.Event()
        self._thread: QThread | None = None
        self._worker: TranslationWorker | None = None
        self._is_running = False
        self._build_ui()
        self._refresh_queue_view()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt API naming
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt API naming
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self._add_paths(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(20, 16, 20, 12)
        layout.setSpacing(12)

        # Flat masthead, same shape as the main window.
        header = QWidget()
        head = QHBoxLayout(header)
        head.setContentsMargins(4, 0, 4, 0)
        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        title = QLabel("字幕翻译工具")
        title.setObjectName("appTitle")
        subtitle = QLabel("拖入 SRT 直接翻译 · Edge / GTX / 本地 Index-Translate 模型")
        subtitle.setObjectName("appSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        head.addLayout(title_box)
        head.addStretch()
        layout.addWidget(header)

        layout.addWidget(self._build_settings_section())
        layout.addWidget(self._build_queue_panel(), 1)
        layout.addWidget(self._build_action_bar())
        self.setCentralWidget(root)

    def _build_settings_section(self) -> QWidget:
        """One borderless block: three choices, all of which are required."""
        holder, layout = section("翻译设置")
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(7)
        grid.setColumnMinimumWidth(0, 58)
        grid.setColumnStretch(1, 1)

        self.backend_combo = ChevronComboBox()
        self.backend_combo.addItems(list(BACKEND_MAP))
        self.source_combo = ChevronComboBox()
        self.source_combo.addItems(list(SOURCE_LANGUAGE_MAP))
        self.source_combo.setCurrentText("自动检测")
        self.target_combo = ChevronComboBox()
        self.target_combo.addItems(list(TARGET_LANGUAGE_MAP))
        self.swap_check = CheckBox("生成单语译文、双语字幕和原始字幕")
        self.swap_check.setChecked(True)

        for row, (label, widget) in enumerate((
            ("后端", self.backend_combo),
            ("源语言", self.source_combo),
            ("目标语言", self.target_combo),
        )):
            caption = QLabel(label)
            caption.setObjectName("fieldLabel")
            caption.setMinimumWidth(58)
            grid.addWidget(caption, row, 0)
            grid.addWidget(widget, row, 1)
        layout.addLayout(grid)
        # Without this the checkbox stretches to the full panel width, which
        # reads as a selectable row instead of a checkbox.
        layout.addWidget(inline(self.swap_check), 0, Qt.AlignmentFlag.AlignLeft)
        self.backend_combo.currentTextChanged.connect(self._on_backend_changed)
        return holder

    def _build_queue_panel(self) -> QWidget:
        """The one raised surface: intake empty state plus the queue table."""
        card, layout = panel()

        head = QHBoxLayout()
        head.setSpacing(12)
        caption = QLabel("翻译队列")
        caption.setObjectName("sectionLabel")
        self.queue_hint = QLabel("尚未添加文件")
        self.queue_hint.setObjectName("faint")
        head.addWidget(caption)
        head.addWidget(self.queue_hint)
        head.addStretch()
        layout.addLayout(head)

        self.drop_zone = DropZone(
            title="拖放 SRT 文件到此处",
            hint="也可点击下方“添加文件”按钮",
            show_button=False,
        )
        self.drop_zone.files_dropped.connect(self._add_paths)
        self.file_table = QTableWidget(0, 2)
        self.file_table.setHorizontalHeaderLabels(["文件", "状态"])
        self.file_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.file_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.file_table.setAlternatingRowColors(True)
        self.file_table.verticalHeader().setVisible(False)
        self.file_table.verticalHeader().setDefaultSectionSize(28)
        self.file_table.verticalHeader().setMinimumSectionSize(28)
        self.file_table.horizontalHeader().setStretchLastSection(False)
        self.file_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.file_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.file_table.setColumnWidth(1, 84)

        empty_host = QWidget()
        empty_layout = QVBoxLayout(empty_host)
        empty_layout.setContentsMargins(0, 0, 0, 0)
        empty_layout.addStretch()
        empty_layout.addWidget(self.drop_zone)
        empty_layout.addStretch()
        stack_host = QWidget()
        self.queue_stack = QStackedLayout(stack_host)
        self.queue_stack.setContentsMargins(0, 0, 0, 0)
        self.queue_stack.addWidget(empty_host)
        self.queue_stack.addWidget(self.file_table)
        layout.addWidget(stack_host, 1)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        add_button = QPushButton("添加文件")
        add_button.clicked.connect(self._choose_files)
        remove_button = QPushButton("移除选中")
        remove_button.clicked.connect(self._remove_selected)
        clear_button = QPushButton("清空")
        clear_button.setObjectName("link")
        clear_button.clicked.connect(self._clear_files)
        actions.addWidget(add_button)
        actions.addWidget(remove_button)
        actions.addStretch()
        actions.addWidget(clear_button)
        layout.addLayout(actions)
        self.add_button = add_button
        self.remove_button = remove_button
        self.clear_button = clear_button
        return card

    def _build_action_bar(self) -> QWidget:
        """Flat bar matching the main window: state left, commitment right."""
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
        self.status_label = QLabel("准备就绪")
        self.status_label.setObjectName("status")
        self.counter_label = QLabel("")
        self.counter_label.setObjectName("meterValue")
        self.stop_button = QPushButton("停止")
        self.stop_button.setObjectName("danger")
        self.stop_button.setEnabled(False)
        self.stop_button.setVisible(False)
        self.stop_button.clicked.connect(self._stop)
        self.start_button = QPushButton("开始翻译")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self._start)
        row.addWidget(self.status_label)
        row.addStretch()
        row.addWidget(self.counter_label)
        row.addWidget(self.stop_button)
        row.addWidget(self.start_button)
        layout.addLayout(row)
        self._settings_widgets = [
            self.backend_combo, self.source_combo, self.target_combo,
            self.swap_check, self.add_button, self.remove_button, self.clear_button,
        ]
        return bar

    # ------------------------------------------------------------------
    # Drag & drop / file list
    # ------------------------------------------------------------------

    def _choose_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择 SRT 字幕文件", str(Path.cwd()), "字幕文件 (*.srt);;所有文件 (*.*)"
        )
        self._add_paths(files)

    def _refresh_queue_view(self) -> None:
        self.queue_stack.setCurrentIndex(1 if self.paths else 0)
        self.queue_hint.setText(f"{len(self.paths)} 个字幕文件" if self.paths else "尚未添加文件")
        self.remove_button.setEnabled(not self._is_running and bool(self.paths))
        self.clear_button.setEnabled(not self._is_running and bool(self.paths))
        self.start_button.setEnabled(not self._is_running and bool(self.paths))

    def _style_status_item(self, item: QTableWidgetItem) -> None:
        background, foreground = status_colors(item.text())
        item.setBackground(QColor(background))
        item.setForeground(QColor(foreground))

    def _add_paths(self, raw_paths: list[str]) -> None:
        # The worker iterates a snapshot of self.paths, so files dropped in
        # mid-run would sit in the table as "等待翻译" forever.
        if self._is_running:
            self._set_status("警告: 任务运行中，先停止后再添加文件。")
            return
        for raw_path in raw_paths:
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file() or path.suffix.lower() != ".srt":
                continue
            if path in self.paths:
                continue
            self.paths.append(path)
            row = self.file_table.rowCount()
            self.file_table.insertRow(row)
            name_item = QTableWidgetItem(path.name)
            name_item.setToolTip(str(path))
            name_item.setData(Qt.ItemDataRole.UserRole, str(path))
            self.file_table.setItem(row, 0, name_item)
            status_item = QTableWidgetItem("等待翻译")
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._style_status_item(status_item)
            self.file_table.setItem(row, 1, status_item)
            self._row_by_path[str(path)] = row
        self._refresh_queue_view()

    def _remove_selected(self) -> None:
        if self._is_running:
            self._set_status("警告: 任务运行中，先停止后再移除文件。")
            return
        rows = sorted(
            {index.row() for index in self.file_table.selectionModel().selectedRows()},
            reverse=True,
        )
        for row in rows:
            path = Path(self.file_table.item(row, 0).data(Qt.ItemDataRole.UserRole))
            self.paths.remove(path)
            self._row_by_path.pop(str(path), None)
            self.file_table.removeRow(row)
        self._rebuild_row_index()
        self._refresh_queue_view()

    def _rebuild_row_index(self) -> None:
        """Row numbers shift after a removal; re-derive them from the table."""
        self._row_by_path.clear()
        for row in range(self.file_table.rowCount()):
            key = self.file_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            self._row_by_path[key] = row

    def _clear_files(self) -> None:
        if self._is_running:
            self._set_status("警告: 任务运行中，先停止后再清空列表。")
            return
        self.paths.clear()
        self._row_by_path.clear()
        self.file_table.setRowCount(0)
        self._refresh_queue_view()

    # ------------------------------------------------------------------
    # Run control
    # ------------------------------------------------------------------

    def _on_backend_changed(self, label: str) -> None:
        backend = BACKEND_MAP.get(label, "bing")
        if backend == "bing" and SOURCE_LANGUAGE_MAP[self.source_combo.currentText()] == "auto":
            self._set_status("Edge 后端不支持自动检测源语言，请选择实际源语言。")

    def _set_status(self, message: str) -> None:
        """The status line is the only channel: no separate log pane.

        Per-file progress already lives in the 状态 column, so a second log
        box would repeat it in prose.
        """
        self.status_label.setText(message)

    def _start(self) -> None:
        if not self.paths:
            QMessageBox.warning(self, "没有输入文件", "请先添加 SRT 字幕文件。")
            return
        backend = BACKEND_MAP[self.backend_combo.currentText()]
        source_lang = SOURCE_LANGUAGE_MAP[self.source_combo.currentText()]
        target_lang = TARGET_LANGUAGE_MAP[self.target_combo.currentText()]
        if source_lang == target_lang:
            QMessageBox.warning(self, "语言相同", "源语言与目标语言相同，无需翻译。")
            return
        if backend == "bing" and source_lang == "auto":
            QMessageBox.warning(
                self, "需要源语言",
                "Microsoft Edge 后端不支持自动检测源语言，请在“源语言”中选择实际语言。",
            )
            return
        if backend == "llm":
            from src.translator.llm import _resolve_gguf, DEFAULT_MODEL_PATH
            try:
                gguf = _resolve_gguf(Path(DEFAULT_MODEL_PATH))
            except Exception as exc:
                QMessageBox.warning(
                    self, "缺少本地模型",
                    f"未找到本地 GGUF 模型：\n{DEFAULT_MODEL_PATH}\n\n{exc}",
                )
                return
            self._append_log(f"使用本地模型: {gguf.name}（首次翻译需加载模型，请耐心等待）")

        self._cancel.clear()
        self._set_running(True)
        self.progress.setValue(0)
        self.status_label.setText("翻译中...")
        self.counter_label.setText(f"0/{len(self.paths)}")

        self._worker = TranslationWorker(
            self.paths.copy(),
            backend,
            source_lang,
            target_lang,
            self.swap_check.isChecked(),
            "",  # proxy reserved for GTX extension
            self._cancel,
        )
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
        self._thread.start()

    def _stop(self) -> None:
        self._cancel.set()
        self.stop_button.setEnabled(False)
        self.status_label.setText("正在停止，等待当前任务结束...")

    def _set_running(self, running: bool) -> None:
        self._is_running = running
        self.stop_button.setEnabled(running)
        self.stop_button.setVisible(running)
        self.start_button.setVisible(not running)
        for widget in self._settings_widgets:
            widget.setEnabled(not running)
        self._refresh_queue_view()

    def _update_progress(self, completed: int, total: int) -> None:
        self.progress.setValue(int(completed / max(total, 1) * 100))
        self.counter_label.setText(f"{completed}/{total}")
        current = self.paths[completed - 1].name if 0 < completed <= len(self.paths) else ""
        self.status_label.setText(f"翻译中 · {current}" if current else "翻译中…")

    def _update_file_status(self, raw_path: str, status: str) -> None:
        row = self._row_by_path.get(raw_path)
        if row is not None and row < self.file_table.rowCount():
            item = self.file_table.item(row, 1)
            item.setText(status)
            self._style_status_item(item)

    def _worker_finished(self, done: int, failed: int, cancelled: bool) -> None:
        self._set_running(False)
        if cancelled:
            summary = f"已停止 · 完成 {done}"
        elif failed:
            summary = f"完成 · 成功 {done} · 失败 {failed}"
        else:
            summary = f"完成 · 成功 {done}"
        self.status_label.setText(summary)
        self.counter_label.setText("")

    def _thread_finished(self) -> None:
        self._thread = None
        self._worker = None

    def _append_log(self, message: str) -> None:
        """Worker log lines collapse into the status line."""
        self.status_label.setText(message)

    def closeEvent(self, event) -> None:
        # The translation worker finishes the batch in flight before emitting
        # finished, so waiting here would freeze the UI for the whole run.
        # Ask the user to retry instead — the window closes normally once the
        # worker signals completion and clears self._thread.
        if self._thread and self._thread.isRunning():
            self._cancel.set()
            QMessageBox.information(
                self,
                "正在停止",
                "当前任务正在停止，请等待进度条走完后关闭窗口。",
            )
            event.ignore()
            return
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Subtitle Translator")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)
    window = TranslateWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
