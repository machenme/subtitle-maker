"""Render both windows to PNG so the redesign can be eyeballed.

Run on the real platform (not ``offscreen``): the offscreen plugin ships no
font database, so every glyph comes out as a tofu box and the screenshots are
useless for judging type and rhythm.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QProgressBar, QTableWidgetItem

from src.gui import APP_STYLE, COL_FILE, COL_STATUS, AsrWindow
from src.translate_gui import TranslateWindow

OUT = ROOT / "docs" / "ui-preview"
OUT.mkdir(parents=True, exist_ok=True)


def pump(app: QApplication, times: int = 6) -> None:
    for _ in range(times):
        app.processEvents()


def shoot(widget, app: QApplication, name: str) -> None:
    pump(app)
    path = OUT / f"{name}.png"
    widget.grab().save(str(path))
    print(f"{name:22s} {widget.width()}x{widget.height()}  -> {path}")


def main() -> int:
    app = QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)

    window = AsrWindow()
    window.resize(1240, 820)
    window.show()
    pump(app)
    shoot(window, app, "asr-empty")

    samples = [
        ("lecture-01-lecture-on-distributed-systems.mp4", "34m12s", "完成"),
        ("interview-with-engineer.wav", "1h02m05s", "处理中"),
        ("keynote-opening.m4a", "12m48s", "翻译中"),
        ("podcast-ep-12.mp3", "58m20s", "等待中"),
        ("teaser-trailer.mp4", "2m31s", "已有字幕"),
    ]
    window.file_table.setRowCount(len(samples))
    for row, (name, duration, status) in enumerate(samples):
        item = QTableWidgetItem(name)
        item.setData(256, name)
        window.file_table.setItem(row, COL_FILE, item)
        bar = QProgressBar()
        bar.setObjectName("thin")
        bar.setRange(0, 100)
        bar.setValue([100, 46, 72, 0, 100][row])
        bar.setTextVisible(False)
        window.file_table.setCellWidget(row, 1, bar)
        duration_item = QTableWidgetItem(duration)
        duration_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        window.file_table.setItem(row, 2, duration_item)
        status_item = QTableWidgetItem(status)
        status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        window._style_status_item(status_item)
        window.file_table.setItem(row, COL_STATUS, status_item)
    window.paths = [None] * len(samples)
    window.file_status = {name: "done" for name, _, _ in samples}
    window._refresh_queue_view()
    window.progress.setValue(58)
    window.progress_label.setText("处理中 · 58.2%")
    window.run_summary.setText("5 个文件 · SRT / TXT · 翻译至 中文 (zh)")
    window.file_counter.setText("3/5")
    window._append_log("03:12:44  转写完成: keynote-opening.m4a")
    window._append_log("03:13:02  警告: 模型未找到，已回退到上一档并发")
    pump(app)
    shoot(window, app, "asr-queue")

    window.expand_log_check.setChecked(True)
    window._append_log("03:13:40  开始翻译: interview-with-engineer.wav")
    window._append_log("03:14:12  翻译完成 128/128 段")
    pump(app)
    shoot(window, app, "asr-log-expanded")

    window.expand_log_check.setChecked(False)

    translate = TranslateWindow()
    translate.resize(900, 680)
    translate.show()
    pump(app)
    shoot(translate, app, "translate-empty")

    translate.file_table.setRowCount(3)
    for row, (name, status) in enumerate((
        ("lecture-01.zh.srt", "完成"),
        ("interview-with-engineer.srt", "翻译中"),
        ("keynote-opening.srt", "等待翻译"),
    )):
        item = QTableWidgetItem(name)
        translate.file_table.setItem(row, 0, item)
        status_item = QTableWidgetItem(status)
        status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        translate._style_status_item(status_item)
        translate.file_table.setItem(row, 1, status_item)
    translate.paths = [None] * 3
    translate._refresh_queue_view()
    translate.progress.setValue(66)
    translate.status_label.setText("翻译中 · interview-with-engineer.srt")
    translate.counter_label.setText("1/3")
    pump(app)
    shoot(translate, app, "translate-queue")

    window.close()
    translate.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())