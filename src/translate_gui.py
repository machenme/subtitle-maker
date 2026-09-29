#!/usr/bin/env python
"""
Standalone subtitle translation UI built with PySide6.

Drag SRT files in, pick a backend (Microsoft Edge / Legacy GTX / local
Hy-MT2 GGUF model) and target language, then translate without running
the Subtitle Maker pipeline.

Usage:
    uv run python -m src.translate_gui
"""
from __future__ import annotations

import html
import logging
import sys
import threading
from pathlib import Path

if __name__ == "__main__" and str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
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
QFrame#dropZone {
    background: #f8fbff;
    border: 1px dashed #9db8d8;
    border-radius: 8px;
}
QFrame#dropZone:hover { background: #f0f7ff; border-color: #2f76c7; }
QLabel#title { color: #183153; font-size: 20pt; font-weight: 700; }
QLabel#subtitle { color: #607590; font-size: 10pt; }
QLabel#sectionTitle { color: #263b59; font-size: 11pt; font-weight: 700; }
QLabel#muted { color: #52677f; }
QLabel#dropTitle { color: #28476e; font-size: 12pt; font-weight: 600; }
QLabel#dropHint { color: #7b8da5; }
QLineEdit, QComboBox {
    background: #ffffff;
    border: 1px solid #d5dfeb;
    border-radius: 7px;
    padding: 7px 9px;
    min-height: 18px;
}
QLineEdit:focus, QComboBox:focus { border: 1px solid #4f91d3; }
QPushButton {
    background: #ffffff;
    border: 1px solid #d3deeb;
    border-radius: 7px;
    padding: 8px 13px;
    color: #314967;
}
QPushButton:hover { background: #edf5fe; border-color: #9fc4e8; }
QPushButton:disabled { color: #a7b3c2; background: #f2f5f8; }
QPushButton#primary {
    background: #2f76c7;
    border-color: #2f76c7;
    color: #ffffff;
    font-weight: 700;
    padding: 9px 18px;
}
QPushButton#primary:hover { background: #2567b1; }
QPushButton#danger { color: #b14d4d; }
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
    padding: 8px;
}
QCheckBox { spacing: 7px; color: #425873; }
"""

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
    "本地 Hy-MT2 模型 (离线·GGUF)": "llm",
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
        self.setMinimumSize(760, 620)
        self.resize(860, 700)

        self.paths: list[Path] = []
        self._row_by_path: dict[str, int] = {}
        self._cancel = threading.Event()
        self._thread: QThread | None = None
        self._worker: TranslationWorker | None = None
        self._is_running = False
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(12)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("字幕翻译工具")
        title.setObjectName("title")
        subtitle = QLabel("拖入 SRT 字幕直接翻译 · 支持 Edge / GTX / 本地 Hy-MT2 模型")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        layout.addLayout(title_box)

        layout.addWidget(self._build_drop_card())
        layout.addWidget(self._build_settings_card(), 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self._build_queue_card(), 1)
        layout.addWidget(self._build_footer())
        self.setCentralWidget(root)

    def _card(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        heading = QLabel(title)
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        return card, layout

    def _build_drop_card(self) -> QFrame:
        card, layout = self._card("输入文件")
        zone = QFrame()
        zone.setObjectName("dropZone")
        zone.setAcceptDrops(True)
        zone.setMinimumHeight(110)
        zone.dropEvent = self._drop_event  # type: ignore[method-assign]
        zone.dragEnterEvent = self._drag_enter_event  # type: ignore[method-assign]
        zone_layout = QVBoxLayout(zone)
        zone_layout.setContentsMargins(12, 10, 12, 12)
        zone_layout.setSpacing(4)
        drop_title = QLabel("拖放 SRT 文件到此处")
        drop_title.setObjectName("dropTitle")
        drop_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_hint = QLabel("也可点击下方“选择文件”按钮")
        drop_hint.setObjectName("dropHint")
        drop_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        zone_layout.addWidget(drop_title)
        zone_layout.addWidget(drop_hint)
        layout.addWidget(zone)

        actions = QHBoxLayout()
        add_button = QPushButton("选择文件")
        add_button.clicked.connect(self._choose_files)
        remove_button = QPushButton("移除选中")
        remove_button.clicked.connect(self._remove_selected)
        clear_button = QPushButton("清空")
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

    def _build_settings_card(self) -> QFrame:
        card, layout = self._card("翻译设置")
        row = QHBoxLayout()
        row.setSpacing(10)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(list(BACKEND_MAP))
        self.source_combo = QComboBox()
        self.source_combo.addItems(list(SOURCE_LANGUAGE_MAP))
        self.source_combo.setCurrentText("自动检测")
        self.target_combo = QComboBox()
        self.target_combo.addItems(list(TARGET_LANGUAGE_MAP))
        self.swap_check = QCheckBox("生成单语译文、双语字幕和原始字幕")
        self.swap_check.setChecked(True)

        backend_label = QLabel("后端")
        backend_label.setObjectName("muted")
        source_label = QLabel("源语言")
        source_label.setObjectName("muted")
        target_label = QLabel("目标语言")
        target_label.setObjectName("muted")
        for widget in (self.backend_combo, self.source_combo, self.target_combo):
            widget.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        row.addWidget(backend_label)
        row.addWidget(self.backend_combo)
        row.addSpacing(10)
        row.addWidget(source_label)
        row.addWidget(self.source_combo)
        row.addSpacing(10)
        row.addWidget(target_label)
        row.addWidget(self.target_combo)
        row.addStretch()
        layout.addLayout(row)
        layout.addWidget(self.swap_check)
        self.backend_combo.currentTextChanged.connect(self._on_backend_changed)
        return card

    def _build_queue_card(self) -> QFrame:
        card, layout = self._card("翻译队列")
        self.file_table = QTableWidget(0, 2)
        self.file_table.setHorizontalHeaderLabels(["文件", "状态"])
        self.file_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.file_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.file_table.setAlternatingRowColors(True)
        self.file_table.verticalHeader().setVisible(False)
        self.file_table.horizontalHeader().setStretchLastSection(False)
        self.file_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.file_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.file_table, 1)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Consolas", 9))
        self.log_edit.setMinimumHeight(110)
        self.log_edit.setMaximumHeight(150)
        layout.addWidget(self.log_edit)
        return card

    def _build_footer(self) -> QWidget:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        top = QHBoxLayout()
        self.status_label = QLabel("准备就绪")
        self.status_label.setObjectName("muted")
        self.counter_label = QLabel("")
        self.counter_label.setObjectName("muted")
        top.addWidget(self.status_label)
        top.addStretch()
        top.addWidget(self.counter_label)
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
        self.start_button = QPushButton("开始翻译")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self._start)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.start_button)
        layout.addLayout(buttons)
        self._settings_widgets = [
            self.backend_combo, self.source_combo, self.target_combo,
            self.swap_check, self.add_button, self.remove_button, self.clear_button,
        ]
        return card

    # ------------------------------------------------------------------
    # Drag & drop / file list
    # ------------------------------------------------------------------

    def _drag_enter_event(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def _drop_event(self, event: QDropEvent) -> None:
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self._add_paths(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    def _choose_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择 SRT 字幕文件", str(Path.cwd()), "字幕文件 (*.srt);;所有文件 (*.*)"
        )
        self._add_paths(files)

    def _add_paths(self, raw_paths: list[str]) -> None:
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
            self.file_table.setItem(row, 1, QTableWidgetItem("等待翻译"))
            self._row_by_path[str(path)] = row
            self._append_log(f"已添加: {path.name}")

    def _remove_selected(self) -> None:
        rows = sorted(
            {index.row() for index in self.file_table.selectionModel().selectedRows()},
            reverse=True,
        )
        for row in rows:
            path = Path(self.file_table.item(row, 0).data(Qt.ItemDataRole.UserRole))
            self.paths.remove(path)
            self._row_by_path.pop(str(path), None)
            self.file_table.removeRow(row)

    def _clear_files(self) -> None:
        self.paths.clear()
        self._row_by_path.clear()
        self.file_table.setRowCount(0)

    # ------------------------------------------------------------------
    # Run control
    # ------------------------------------------------------------------

    def _on_backend_changed(self, label: str) -> None:
        backend = BACKEND_MAP.get(label, "bing")
        if backend == "bing" and SOURCE_LANGUAGE_MAP[self.source_combo.currentText()] == "auto":
            self._append_log("提示: Edge 后端不支持自动检测源语言，请选择实际源语言。")

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

    def _update_progress(self, completed: int, total: int) -> None:
        self.progress.setValue(int(completed / max(total, 1) * 100))
        self.counter_label.setText(f"{completed}/{total}")

    def _update_file_status(self, raw_path: str, status: str) -> None:
        row = self._row_by_path.get(raw_path)
        if row is not None and row < self.file_table.rowCount():
            self.file_table.item(row, 1).setText(status)

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
        self._append_log(summary)

    def _thread_finished(self) -> None:
        self._thread = None
        self._worker = None

    def _append_log(self, message: str) -> None:
        self.log_edit.append(html.escape(message))
        self.log_edit.ensureCursorVisible()

    def closeEvent(self, event) -> None:
        if self._thread and self._thread.isRunning():
            self._cancel.set()
            self._thread.quit()
            self._thread.wait(5000)
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
