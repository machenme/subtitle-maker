#!/usr/bin/env python
"""
Standalone SRT → TXT / Markdown window built with PySide6.

Drag SRT files in, pick the output formats and folder, and export — no
transcription, no translation, no model load.  The conversion itself lives in
:mod:`src.srt_convert`, so the CLI and this window cannot drift apart.

Usage:
    uv run python -m src.srt_convert_gui
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

if __name__ == "__main__" and str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QColor, QDesktopServices, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
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
from PySide6.QtCore import QUrl

from src.srt_convert import convert_file
from src.ui_theme import (
    APP_STYLE,
    CheckBox,
    DropZone,
    inline,
    panel,
    section,
    status_colors,
)

logger = logging.getLogger(__name__)


class ConvertWorker(QObject):
    """Convert a list of SRT files one by one off the UI thread."""

    log = Signal(str)
    progress = Signal(int, int)          # completed, total
    file_status = Signal(str, str)       # path, status text
    finished = Signal(int, list)         # succeeded, errors

    def __init__(
        self,
        paths: list[Path],
        formats: list[str],
        output_dir: Path | None,
    ):
        super().__init__()
        self.paths = paths
        self.formats = formats
        self.output_dir = output_dir

    def run(self) -> None:
        succeeded = 0
        errors: list[str] = []
        total = len(self.paths)
        for index, path in enumerate(self.paths):
            self.file_status.emit(str(path), "转换中")
            try:
                written = convert_file(path, self.output_dir, self.formats)
            except Exception as exc:
                logger.exception("Conversion failed for %s", path)
                errors.append(f"{path.name}: {exc}")
                self.file_status.emit(str(path), "转换失败")
                self.log.emit(f"转换失败: {path.name} — {exc}")
            else:
                succeeded += 1
                self.file_status.emit(str(path), "完成")
                self.log.emit(f"已导出: {', '.join(p.name for p in written)}")
            self.progress.emit(index + 1, total)
        self.finished.emit(succeeded, errors)


class ConvertWindow(QMainWindow):
    """Drag-and-drop SRT → TXT/Markdown export window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("字幕转文本")
        self.setMinimumSize(720, 520)
        self.resize(860, 640)
        self.setAcceptDrops(True)

        self.paths: list[Path] = []
        self._row_by_path: dict[str, int] = {}
        self._is_running = False
        self._thread: QThread | None = None
        self._worker: ConvertWorker | None = None
        self._build_ui()
        self._refresh_queue_view()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt API naming
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt API naming
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        # A dropped folder is a natural thing to do here, so expand it instead
        # of silently ignoring it the way a plain *.srt filter would.
        self._add_paths(paths)
        if paths:
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

        header = QWidget()
        head = QHBoxLayout(header)
        head.setContentsMargins(4, 0, 4, 0)
        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        title = QLabel("字幕转文本")
        title.setObjectName("appTitle")
        subtitle = QLabel("SRT 直接导出纯文本或 Markdown · 离线，无需转写")
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
        """Formats and destination — both always editable, both optional."""
        holder, layout = section("导出设置")
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(7)
        grid.setColumnMinimumWidth(0, 58)
        grid.setColumnStretch(1, 1)

        self.txt_check = CheckBox("纯文本 .txt")
        self.txt_check.setChecked(True)
        self.md_check = CheckBox("Markdown .md")
        self.md_check.setChecked(True)
        self.same_dir_check = CheckBox("输出到字幕所在文件夹")
        self.same_dir_check.setChecked(True)
        self.same_dir_check.toggled.connect(self._on_output_mode_changed)

        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("选择输出文件夹（取消上面的勾选后可指定）")
        self.output_edit.setEnabled(False)
        browse = QPushButton("浏览")
        browse.setEnabled(False)
        browse.clicked.connect(self._choose_output_dir)

        caption = QLabel("格式")
        caption.setObjectName("fieldLabel")
        caption.setMinimumWidth(58)
        grid.addWidget(caption, 0, 0)
        grid.addWidget(self._build_format_row(), 0, 1)

        dest_caption = QLabel("输出")
        dest_caption.setObjectName("fieldLabel")
        dest_caption.setMinimumWidth(58)
        grid.addWidget(dest_caption, 1, 0)
        grid.addWidget(self.output_edit, 1, 1)
        layout.addLayout(grid)

        self._browse_button = browse
        self.output_edit.textChanged.connect(self._refresh_run_summary)
        self.txt_check.toggled.connect(self._refresh_run_summary)
        self.md_check.toggled.connect(self._refresh_run_summary)
        self.same_dir_check.toggled.connect(self._refresh_run_summary)
        return holder

    def _build_format_row(self) -> QWidget:
        row = QWidget()
        box = QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(14)
        box.addWidget(inline(self.txt_check))
        box.addWidget(inline(self.md_check))
        box.addWidget(inline(self.same_dir_check))
        box.addStretch()
        return row

    def _build_queue_panel(self) -> QWidget:
        """The one raised surface: intake empty state plus the queue table."""
        card, layout = panel()

        head = QHBoxLayout()
        head.setSpacing(12)
        caption = QLabel("字幕文件")
        caption.setObjectName("sectionLabel")
        self.queue_hint = QLabel("尚未添加文件")
        self.queue_hint.setObjectName("faint")
        head.addWidget(caption)
        head.addWidget(self.queue_hint)
        head.addStretch()
        layout.addLayout(head)

        self.drop_zone = DropZone(
            title="拖放 SRT 文件或文件夹到此处",
            hint="也可点击下方“添加文件”按钮 · 支持整个文件夹批量转换",
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
        self.file_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.file_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Fixed
        )
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
        add_folder_button = QPushButton("添加文件夹")
        add_folder_button.clicked.connect(self._choose_folder)
        remove_button = QPushButton("移除选中")
        remove_button.clicked.connect(self._remove_selected)
        clear_button = QPushButton("清空")
        clear_button.setObjectName("link")
        clear_button.clicked.connect(self._clear_files)
        actions.addWidget(add_button)
        actions.addWidget(add_folder_button)
        actions.addWidget(remove_button)
        actions.addStretch()
        actions.addWidget(clear_button)
        layout.addLayout(actions)
        self.add_button = add_button
        self.add_folder_button = add_folder_button
        self.remove_button = remove_button
        self.clear_button = clear_button
        return card

    def _build_action_bar(self) -> QWidget:
        """Flat bar: state on the left, commitment on the right."""
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
        self.open_button = QPushButton("打开输出文件夹")
        self.open_button.setObjectName("ghost")
        self.open_button.setVisible(False)
        self.open_button.clicked.connect(self._open_output_dir)
        self.start_button = QPushButton("开始转换")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self._start)

        row.addWidget(self.status_label)
        row.addStretch()
        row.addWidget(self.counter_label)
        row.addWidget(self.open_button)
        row.addWidget(self.start_button)
        layout.addLayout(row)
        self._settings_widgets = [
            self.txt_check, self.md_check, self.same_dir_check,
            self.output_edit, self._browse_button,
            self.add_button, self.add_folder_button,
            self.remove_button, self.clear_button,
        ]
        return bar

    # ------------------------------------------------------------------
    # File list
    # ------------------------------------------------------------------

    def _choose_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择 SRT 字幕文件", str(Path.cwd()),
            "字幕文件 (*.srt);;所有文件 (*.*)",
        )
        self._add_paths(files)

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择包含字幕的文件夹")
        if folder:
            self._add_paths([folder])

    def _choose_output_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if folder:
            self.output_edit.setText(folder)

    def _on_output_mode_changed(self, same_dir: bool) -> None:
        self.output_edit.setEnabled(not same_dir)
        self._browse_button.setEnabled(not same_dir)

    def _style_status_item(self, item: QTableWidgetItem) -> None:
        background, foreground = status_colors(item.text())
        item.setBackground(QColor(background))
        item.setForeground(QColor(foreground))

    def _add_paths(self, raw_paths: list[str]) -> None:
        if self._is_running:
            self.status_label.setText("警告: 任务运行中，先停止后再添加文件。")
            return

        from src.srt_convert import collect_srt_files

        candidates = [Path(p).expanduser() for p in raw_paths]
        for srt_path in collect_srt_files(candidates):
            if srt_path in self.paths:
                continue
            self.paths.append(srt_path)
            row = self.file_table.rowCount()
            self.file_table.insertRow(row)
            name_item = QTableWidgetItem(srt_path.name)
            name_item.setToolTip(str(srt_path))
            name_item.setData(Qt.ItemDataRole.UserRole, str(srt_path))
            self.file_table.setItem(row, 0, name_item)
            status_item = QTableWidgetItem("等待转换")
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._style_status_item(status_item)
            self.file_table.setItem(row, 1, status_item)
            self._row_by_path[str(srt_path)] = row
        self._refresh_queue_view()

    def _remove_selected(self) -> None:
        if self._is_running:
            self.status_label.setText("警告: 任务运行中，先停止后再移除文件。")
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
            self.status_label.setText("警告: 任务运行中，先停止后再清空列表。")
            return
        self.paths.clear()
        self._row_by_path.clear()
        self.file_table.setRowCount(0)
        self._refresh_queue_view()

    # ------------------------------------------------------------------
    # Run control
    # ------------------------------------------------------------------

    def _selected_formats(self) -> list[str]:
        formats = []
        if self.txt_check.isChecked():
            formats.append("txt")
        if self.md_check.isChecked():
            formats.append("md")
        return formats

    def _target_output_dir(self) -> Path | None:
        """``None`` means "next to each SRT" — handled inside convert_file."""
        if self.same_dir_check.isChecked():
            return None
        text = self.output_edit.text().strip()
        return Path(text) if text else None

    def _refresh_run_summary(self) -> None:
        if self._is_running:
            return
        if not self.paths:
            self.status_label.setText("准备就绪")
            return
        formats = self._selected_formats()
        if not formats:
            # Not an error yet: the Start button is what enforces the choice.
            self.status_label.setText("请至少选择一种输出格式")
            return
        destination = "字幕所在文件夹" if self.same_dir_check.isChecked() else (
            self.output_edit.text().strip() or "字幕所在文件夹"
        )
        labels = {"txt": "TXT", "md": "Markdown"}
        self.status_label.setText(
            f"{len(self.paths)} 个文件 → {' + '.join(labels[f] for f in formats)}"
            f" · 输出到 {destination}"
        )

    def _refresh_queue_view(self) -> None:
        self.queue_stack.setCurrentIndex(1 if self.paths else 0)
        self.queue_hint.setText(
            f"{len(self.paths)} 个字幕文件" if self.paths else "尚未添加文件"
        )
        editable = not self._is_running
        self.remove_button.setEnabled(editable and bool(self.paths))
        self.clear_button.setEnabled(editable and bool(self.paths))
        self.start_button.setEnabled(editable and bool(self.paths))
        self._refresh_run_summary()

    def _start(self) -> None:
        if not self.paths:
            QMessageBox.warning(self, "没有输入文件", "请先添加 SRT 字幕文件。")
            return
        formats = self._selected_formats()
        if not formats:
            QMessageBox.warning(
                self, "没有选择格式", "请至少勾选一种输出格式（TXT 或 Markdown）。"
            )
            return
        # A folder that does not exist yet is created on write, same as the
        # CLI does — only an empty field is wrong, because that would silently
        # fall back to writing next to each SRT.
        if not self.same_dir_check.isChecked() and not self.output_edit.text().strip():
            QMessageBox.warning(
                self, "没有选择输出文件夹",
                "请取消勾选“输出到字幕所在文件夹”后指定一个文件夹，或重新勾选该项。",
            )
            return
        output_dir = self._target_output_dir()

        self._is_running = True
        self.progress.setValue(0)
        self.open_button.setVisible(False)
        self.counter_label.setText(f"0/{len(self.paths)}")
        for path in self.paths:
            self._set_file_status(str(path), "等待转换")

        self._worker = ConvertWorker(self.paths.copy(), formats, output_dir)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self.status_label.setText)
        self._worker.progress.connect(self._update_progress)
        self._worker.file_status.connect(self._set_file_status)
        self._worker.finished.connect(self._worker_finished)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()
        self._set_running(True)

    def _set_running(self, running: bool) -> None:
        self._is_running = running
        for widget in self._settings_widgets:
            widget.setEnabled(not running)
        self._refresh_queue_view()

    def _update_progress(self, completed: int, total: int) -> None:
        self.progress.setValue(int(completed / max(total, 1) * 100))
        self.counter_label.setText(f"{completed}/{total}")
        current = self.paths[completed - 1].name if 0 < completed <= len(self.paths) else ""
        self.status_label.setText(f"转换中 · {current}" if current else "转换中…")

    def _set_file_status(self, raw_path: str, status: str) -> None:
        row = self._row_by_path.get(raw_path)
        if row is not None and row < self.file_table.rowCount():
            item = self.file_table.item(row, 1)
            item.setText(status)
            self._style_status_item(item)

    def _worker_finished(self, succeeded: int, errors: list) -> None:
        self._set_running(False)
        failed = len(errors)
        parts = [f"完成 · 成功 {succeeded}"]
        if failed:
            parts.append(f"失败 {failed}")
        self.status_label.setText(" · ".join(parts))
        self.counter_label.setText("")
        self.open_button.setVisible(bool(self.output_edit.text().strip()))
        if errors:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("部分文件转换失败")
            box.setText(f"{succeeded} 个成功，{failed} 个失败。")
            box.setDetailedText("\n".join(errors))
            box.exec()

    # No stop button on purpose: one file converts in milliseconds, so there is
    # no batch long enough to be worth interrupting, and a dead control would
    # be worse than none.

    def _open_output_dir(self) -> None:
        target = Path(self.output_edit.text().strip())
        if target.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API naming
        # Converting is fast, but a large queue may still be in flight; closing
        # the window underneath the worker would leave files half-written.
        if self._is_running:
            QMessageBox.information(
                self, "正在转换", "当前正在转换，请等待完成后再关闭窗口。"
            )
            event.ignore()
            return
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Subtitle Converter")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)
    window = ConvertWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
