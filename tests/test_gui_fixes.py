"""Offscreen checks for the GUI fixes: signal names, async durations, config."""
import os
import sys
import threading
import wave
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pytest
from PySide6.QtWidgets import QApplication

from src.config import PipelineConfig
from src.gui import COL_DURATION, AsrWindow, DurationProbe, PipelineWorker


def _write_wav(path: Path, seconds: int = 3) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000 * seconds)
    return path


def _write_config(root: Path, **overrides) -> Path:
    model_dir = root / "model"
    model_dir.mkdir(exist_ok=True)
    lines = [
        f'input_dir: "{root.as_posix()}"',
        f'output_dir: "{(root / "out").as_posix()}"',
        f'model_path: "{model_dir.as_posix()}"',
    ]
    lines += [f"{key}: {json_value}" for key, json_value in overrides.items()]
    config_path = root / "config.yaml"
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


@pytest.fixture(scope="module")
def app():
    existing = QApplication.instance()
    return existing or QApplication([])


@pytest.fixture
def window(app, tmp_path: Path, monkeypatch):
    """A real AsrWindow reading a throwaway config.yaml from the cwd."""
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    win = AsrWindow()
    yield win
    for probe in list(win._duration_probes):
        probe.wait(timeout=2.0)
    win._duration_probes.clear()
    win.close()


def _window_with_config(app, tmp_path: Path, monkeypatch, **overrides) -> AsrWindow:
    """Build a real AsrWindow against a config.yaml written into tmp_path."""
    _write_config(tmp_path, **overrides)
    monkeypatch.chdir(tmp_path)
    win = AsrWindow()
    assert win.config is not None, f"config failed to load: {win.config_error}"
    return win


# --------------------------------------------------------------------------
# Regression: the LLM translation pass used a signal that does not exist
# --------------------------------------------------------------------------

def test_pipeline_worker_exposes_log_signal_not_private_log():
    """self._log.emit() raised AttributeError and left the window unclosable."""
    assert hasattr(PipelineWorker, "log")
    assert not hasattr(PipelineWorker, "_log")


def test_gui_module_has_no_dangling_self_log_calls():
    import src.gui as gui_module

    source = Path(gui_module.__file__).read_text(encoding="utf-8")
    assert "self._log.emit" not in source, "self._log.emit is back in gui.py"


# --------------------------------------------------------------------------
# Regression: ffprobe used to run inline and blocked the UI thread
# --------------------------------------------------------------------------

def test_duration_probe_start_returns_immediately(tmp_path: Path):
    """start() must hand work to a thread rather than probing inline."""
    paths = [_write_wav(tmp_path / f"v{i}.wav") for i in range(3)]

    probe = DurationProbe(paths)
    assert probe.wait(timeout=0) is False, "probe finished before start()"

    probe.start()
    assert probe.wait(timeout=30) is True, "probe did not finish"


def test_duration_probe_reports_every_file(app, tmp_path: Path):
    paths = [_write_wav(tmp_path / f"s{i}.wav", seconds=i + 1) for i in range(3)]

    probe = DurationProbe(paths)
    results: list[tuple[str, str]] = []
    probe.done.connect(lambda p, t: results.append((p, t)))
    probe.start()
    assert probe.wait(timeout=30) is True

    # Cross-thread signal delivery needs the event loop; drain it.
    for _ in range(300):
        app.processEvents()
        if len(results) == len(paths):
            break

    reported = {Path(p).name: text for p, text in results}
    assert set(reported) == {p.name for p in paths}
    assert reported["s0.wav"] == "0m01s"
    assert reported["s1.wav"] == "0m02s"
    assert reported["s2.wav"] == "0m03s"


def test_duration_probe_tolerates_unreadable_file(tmp_path: Path):
    """A corrupt file must yield "?" instead of killing the probe thread."""
    good = _write_wav(tmp_path / "ok.wav")
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a media file")

    probe = DurationProbe([broken, good])
    probe.start()
    assert probe.wait(timeout=30) is True


def test_add_file_shows_placeholder_instead_of_blocking(window, app, tmp_path: Path):
    """_add_file must not call ffprobe inline any more."""
    media = _write_wav(tmp_path / "clip.wav")

    window._add_file(media)
    app.processEvents()

    cell = window.file_table.item(0, COL_DURATION)
    assert cell is not None
    assert cell.text() in {"读取中…", "0m03s"}


# --------------------------------------------------------------------------
# Regression: checkbox defaults ignored config.yaml
# --------------------------------------------------------------------------

def test_translate_checkbox_follows_config(app, tmp_path: Path, monkeypatch):
    """translate_to: "" must not show the translation box as enabled."""
    win = _window_with_config(
        app, tmp_path, monkeypatch,
        **{"translate_to": '""', "swap_subtitles": "false"},
    )
    try:
        assert win.config.translate_to == ""
        assert win.translate_check.isChecked() is False
        assert win.swap_check.isChecked() is False
    finally:
        win._duration_probes.clear()
        win.close()


def test_translate_checkbox_enabled_when_config_translates(app, tmp_path: Path, monkeypatch):
    win = _window_with_config(
        app, tmp_path, monkeypatch,
        **{"translate_to": '"zh"', "swap_subtitles": "true"},
    )
    try:
        assert win.translate_check.isChecked() is True
        assert win.swap_check.isChecked() is True
    finally:
        win._duration_probes.clear()
        win.close()


# --------------------------------------------------------------------------
# Regression: translator was rebuilt per file, discarding connection pools
# --------------------------------------------------------------------------

def test_translator_is_created_once_and_reused(window):
    """The provider lives on PipelineWorker, which processes the queue."""
    created: list[str] = []

    class FakeProvider:
        def translate(self, text, source_lang="auto", target_lang="zh"):
            return text

        def translate_batch(self, texts, source_lang="auto", target_lang="zh"):
            return list(texts)

    def fake_create(provider: str = "bing", *, proxy: str = ""):
        created.append(provider)
        return FakeProvider()

    import src.gui as gui_module

    real = gui_module.create_translator
    gui_module.create_translator = fake_create
    worker = PipelineWorker([], {}, window.config, threading.Event())
    try:
        provider = None
        for _ in range(3):
            provider, error = worker._provider()
            assert provider is not None, error
        assert len(created) == 1, f"provider rebuilt {len(created)} times"

        again, error = worker._provider()
        assert again is provider
        assert len(created) == 1, "second call rebuilt the provider"
    finally:
        gui_module.create_translator = real


def test_worker_starts_without_a_translator(window):
    """A fresh worker has no provider until one is actually needed."""
    worker = PipelineWorker([], {}, window.config, threading.Event())
    assert worker._translator is None
