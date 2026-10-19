"""Offscreen smoke test for the SRT → TXT/MD window.

Builds the real widget and drives a conversion through the worker thread, so a
broken signal connection or a missing import fails here instead of in front of
the user. Runs under the offscreen Qt platform — no display needed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from src.srt_convert_gui import ConvertWindow  # noqa: E402
from src.ui_theme import APP_STYLE  # noqa: E402

SAMPLE = (
    "1\n00:00:01,000 --> 00:00:03,000\nこんにちは\n\n"
    "2\n00:00:04,000 --> 00:00:06,000\n世界\n\n"
)


@pytest.fixture(scope="module")
def app():
    existing = QApplication.instance()
    if existing is None:
        instance = QApplication(sys.argv)
        instance.setStyle("Fusion")
        instance.setStyleSheet(APP_STYLE)
        return instance
    return existing


def _pump_until_idle(window: ConvertWindow, timeout_ms: int = 15000) -> None:
    """Spin the event loop until the worker clears the running flag."""
    loop = QEventLoop()
    elapsed = {"ms": 0}
    step = {"tick": 20}
    timer = QTimer()
    timer.setInterval(step["tick"])
    timer.timeout.connect(lambda: None)

    def poll() -> None:
        elapsed["ms"] += step["tick"]
        if not window._is_running or elapsed["ms"] >= timeout_ms:
            timer.stop()
            loop.quit()

    timer.timeout.connect(poll)
    timer.start()
    loop.exec()
    assert not window._is_running, "conversion did not finish in time"


def test_window_builds_and_summarizes(app, tmp_path):
    source = tmp_path / "demo.srt"
    source.write_text(SAMPLE, encoding="utf-8")

    window = ConvertWindow()
    window._add_paths([str(source)])
    assert len(window.paths) == 1
    assert "TXT" in window.status_label.text()
    assert window.start_button.isEnabled()

    window.txt_check.setChecked(False)
    window.md_check.setChecked(False)
    assert "至少选择一种" in window.status_label.text()

    window.close()


def test_conversion_through_the_worker(app, tmp_path):
    source = tmp_path / "demo.srt"
    source.write_text(SAMPLE, encoding="utf-8")
    destination = tmp_path / "out"

    window = ConvertWindow()
    window._add_paths([str(source)])
    window.same_dir_check.setChecked(False)
    window.output_edit.setText(str(destination))
    window._start()
    _pump_until_idle(window)

    assert (destination / "demo.txt").is_file()
    assert (destination / "demo.md").is_file()
    assert "成功 1" in window.status_label.text()
    assert window.file_table.item(0, 1).text() == "完成"
    window.close()


def test_folder_intake_expands_to_srt_files(app, tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.srt").write_text(SAMPLE, encoding="utf-8")
    (tmp_path / "nested" / "b.srt").write_text(SAMPLE, encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")

    window = ConvertWindow()
    window._add_paths([str(tmp_path)])
    assert [p.name for p in window.paths] == ["a.srt", "b.srt"]
    window.close()


if __name__ == "__main__":
    subprocess_run = __import__("subprocess").run
    subprocess_run([sys.executable, "-m", "pytest", __file__, "-v"])
