from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication

from src.gui import AsrWindow, PipelineWorker
from src.task_manager import TaskManager
from src.translator import TranslationError
from src.ui_theme import FAILURE_LOG_HEX, SUCCESS_LOG_HEX, log_level_color


_qt_app: QCoreApplication | None = None


def _flush_qt_events() -> None:
    global _qt_app
    _qt_app = QCoreApplication.instance() or QCoreApplication([])
    _qt_app.processEvents()


def test_existing_srt_is_not_queued_for_translation_when_disabled(tmp_path: Path):
    media_path = tmp_path / "sample.mp4"
    media_path.touch()
    media_path.with_suffix(".srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n"
        "A valid subtitle line for this test.\n",
        encoding="utf-8",
    )

    assert PipelineWorker.check_existing_subs(media_path, target_lang="") == "done"
    assert PipelineWorker.check_existing_subs(media_path, target_lang="zh") == "translate"


def test_direct_srt_skips_translation_when_target_language_is_empty(tmp_path: Path, monkeypatch):
    srt_path = tmp_path / "sample.srt"
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
    worker = PipelineWorker(
        [srt_path],
        {str(srt_path): "direct_srt"},
        SimpleNamespace(translate_to="", translation_provider="bing"),
        threading.Event(),
    )
    statuses: list[str] = []
    worker.file_status.connect(lambda _path, status: statuses.append(status))
    monkeypatch.setattr(
        "src.gui.translate_srt_with_outputs",
        lambda *_args, **_kwargs: pytest.fail("should not translate"),
    )

    worker.run()

    assert statuses == ["已有字幕"]


def _translation_config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        translate_to="zh",
        output_dir=tmp_path,
        language="ja",
        swap_subtitles=True,
        translation_provider="bing",
        translation_proxy="",
    )


def test_existing_subtitles_are_detected_in_custom_output_directory(tmp_path: Path):
    media_path = tmp_path / "input" / "sample.mp4"
    output_dir = tmp_path / "output"
    media_path.parent.mkdir()
    output_dir.mkdir()
    media_path.touch()
    content = "1\n00:00:00,000 --> 00:00:01,000\nA valid subtitle line for this test.\n"
    for suffix in (".srt", ".bilingual.srt", ".jpn.srt"):
        (output_dir / f"sample{suffix}").write_text(content, encoding="utf-8")

    assert PipelineWorker.check_existing_subs(
        media_path,
        target_lang="zh",
        source_lang="ja",
        output_dir=output_dir,
    ) == "done"


def test_untranslated_srt_in_custom_output_directory_is_queued_for_translation(tmp_path: Path):
    media_path = tmp_path / "input" / "sample.mp4"
    output_dir = tmp_path / "output"
    media_path.parent.mkdir()
    output_dir.mkdir()
    media_path.touch()
    (output_dir / "sample.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nA valid subtitle line for this test.\n",
        encoding="utf-8",
    )

    assert PipelineWorker.check_existing_subs(
        media_path,
        target_lang="zh",
        output_dir=output_dir,
    ) == "translate_output"


def test_queue_summary_reports_asr_and_translation_stages():
    summary = AsrWindow._queue_summary_text(
        ["asr", "处理中", "翻译中", "等待翻译", "完成 · 已翻译"]
    )

    assert summary == "5 个文件  ·  ASR 待处理 1  ·  ASR 中 1  ·  翻译中 1  ·  待翻译 1  ·  完成 1"


def test_log_color_treats_zero_failures_as_success():
    # The assertion is about semantics — "0 failed" must not read as a failure —
    # so compare against the palette rather than hard-coded hex values.
    clean = AsrWindow._log_color("Transcription complete: 17 success, 0 failed (out of 17)")
    failed = AsrWindow._log_color("Transcription complete: 16 success, 1 failed (out of 17)")
    assert clean == log_level_color("Transcription complete: 17 success, 0 failed (out of 17)")
    assert clean != failed
    assert clean == SUCCESS_LOG_HEX
    assert failed == FAILURE_LOG_HEX


def test_asr_continues_while_previous_subtitles_are_translating(tmp_path: Path, monkeypatch):
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.touch()
    second.touch()
    translation_started = threading.Event()
    second_asr_started = threading.Event()
    asr_calls: list[str] = []
    translated_source_languages: list[str] = []
    statuses: list[tuple[str, str]] = []
    summaries: list[tuple[int, int, int, float, bool]] = []

    def fake_asr(config, media_path, **_kwargs):
        assert config.translate_to == ""
        asr_calls.append(media_path.name)
        _kwargs["progress_callback"]("detected_language", "ko", "ko")
        if media_path == second:
            assert translation_started.wait(timeout=1)
            second_asr_started.set()
        return True, 12, ""

    def fake_translate(_srt_path, *_args, **kwargs):
        translated_source_languages.append(kwargs["source_lang"])
        translation_started.set()
        assert second_asr_started.wait(timeout=1)
        return tmp_path / "translated.srt"

    monkeypatch.setattr("src.gui.run_one_video", fake_asr)
    monkeypatch.setattr("src.gui.translate_srt_with_outputs", fake_translate)
    monkeypatch.setattr("src.gui.create_translator", lambda *_args, **_kwargs: object())
    worker = PipelineWorker(
        [first, second],
        {},
        _translation_config(tmp_path),
        threading.Event(),
    )
    worker.file_status.connect(lambda path, status: statuses.append((path, status)))
    worker.finished.connect(lambda *args: summaries.append(args))

    worker.run()
    _flush_qt_events()

    assert asr_calls == ["first.mp4", "second.mp4"]
    assert translated_source_languages == ["ko", "ko"]
    assert (str(first), "完成 · 已翻译") in statuses
    assert (str(second), "完成 · 已翻译") in statuses
    assert summaries[0][0:3] == (2, 2, 0)


def test_auto_source_language_does_not_require_original_suffix_for_resume(tmp_path: Path):
    manager = TaskManager(
        tmp_path / "output",
        output_formats=["srt"],
        translate_to="zh",
        swap_subtitles=True,
        source_lang="auto",
    )

    assert manager._expected_output_names("movie") == [
        "movie.bilingual.srt",
        "movie.srt",
    ]


def test_translation_failure_does_not_block_later_asr_or_translation(tmp_path: Path, monkeypatch):
    existing_srt = tmp_path / "existing.srt"
    existing_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\ntext\n", encoding="utf-8")
    media_path = tmp_path / "fresh.mp4"
    media_path.touch()
    asr_calls: list[str] = []
    statuses: list[tuple[str, str]] = []
    summaries: list[tuple[int, int, int, float, bool]] = []

    def fake_asr(config, path, **_kwargs):
        assert config.translate_to == ""
        asr_calls.append(path.name)
        return True, 8, ""

    def fake_translate(srt_path, *_args, **_kwargs):
        if Path(srt_path) == existing_srt:
            raise TranslationError("provider unavailable")
        return tmp_path / "fresh.bilingual.srt"

    monkeypatch.setattr("src.gui.run_one_video", fake_asr)
    monkeypatch.setattr("src.gui.translate_srt_with_outputs", fake_translate)
    monkeypatch.setattr("src.gui.create_translator", lambda *_args, **_kwargs: object())
    worker = PipelineWorker(
        [existing_srt, media_path],
        {str(existing_srt): "direct_srt"},
        _translation_config(tmp_path),
        threading.Event(),
    )
    worker.file_status.connect(lambda path, status: statuses.append((path, status)))
    worker.finished.connect(lambda *args: summaries.append(args))

    worker.run()
    _flush_qt_events()

    assert asr_calls == ["fresh.mp4"]
    assert (str(existing_srt), "翻译失败") in statuses
    assert (str(media_path), "完成 · 已翻译") in statuses
    assert summaries[0][0:3] == (1, 2, 1)


def test_stop_finishes_active_translation_and_leaves_pending_subtitles_queued(tmp_path: Path, monkeypatch):
    first = tmp_path / "first.srt"
    second = tmp_path / "second.srt"
    for path in (first, second):
        path.write_text("1\n00:00:00,000 --> 00:00:01,000\ntext\n", encoding="utf-8")
    translation_started = threading.Event()
    release_translation = threading.Event()
    cancel_event = threading.Event()
    calls: list[Path] = []
    statuses: list[tuple[str, str]] = []
    summaries: list[tuple[int, int, int, float, bool]] = []

    def fake_translate(srt_path, *_args, **_kwargs):
        path = Path(srt_path)
        calls.append(path)
        if path == first:
            translation_started.set()
            assert release_translation.wait(timeout=1)
        return tmp_path / f"{path.stem}.bilingual.srt"

    monkeypatch.setattr("src.gui.translate_srt_with_outputs", fake_translate)
    monkeypatch.setattr("src.gui.create_translator", lambda *_args, **_kwargs: object())
    worker = PipelineWorker(
        [first, second],
        {str(first): "direct_srt", str(second): "direct_srt"},
        _translation_config(tmp_path),
        cancel_event,
    )
    worker.file_status.connect(lambda path, status: statuses.append((path, status)))
    worker.finished.connect(lambda *args: summaries.append(args))
    runner = threading.Thread(target=worker.run)
    runner.start()

    assert translation_started.wait(timeout=1)
    cancel_event.set()
    release_translation.set()
    runner.join(timeout=2)
    _flush_qt_events()

    assert not runner.is_alive()
    assert calls == [first]
    assert (str(first), "完成 · 已翻译") in statuses
    assert (str(second), "等待翻译") in statuses
    assert summaries[0][0:3] == (1, 2, 0)
    assert summaries[0][4] is True
