"""Resume: subtitles exist, translation missing → translate, never re-transcribe.

A failed translation used to leave the media file "not done", so the next run
repeated the whole ASR pass to reproduce an SRT that was already on disk.
"""
from pathlib import Path
from types import SimpleNamespace

from src.task_manager import TaskManager, existing_source_srt
from src import main as main_module


def _srt(tmp_path: Path, name: str = "clip.srt") -> Path:
    path = tmp_path / name
    path.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\n第一条原文\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\n第二条原文\n",
        encoding="utf-8",
    )
    return path


def _video(tmp_path: Path, name: str = "clip.mp4") -> Path:
    path = tmp_path / name
    path.write_bytes(b"\x00" * 2048)
    return path


class StubProvider:
    def translate_batch(self, texts, source_lang, target_lang):
        return [f"译文:{text}" for text in texts]


def test_existing_source_srt_prefers_the_output_directory(tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    _srt(out, "clip.srt")
    video = _video(tmp_path)

    assert existing_source_srt(video, out) == out / "clip.srt"


def test_existing_source_srt_finds_the_per_video_subdirectory(tmp_path: Path):
    out = tmp_path / "out"
    nested = out / "clip"
    nested.mkdir(parents=True)
    video = _video(tmp_path)

    assert existing_source_srt(video, out) is None
    nested_srt = _srt(nested, "clip.srt")
    assert existing_source_srt(video, out) == nested_srt


def test_existing_source_srt_ignores_foreign_subtitles_next_to_the_media(
    tmp_path: Path,
):
    """An SRT beside the video may be someone else's translation."""
    out = tmp_path / "out"
    out.mkdir()
    video = _video(tmp_path)
    _srt(tmp_path, "clip.srt")

    assert existing_source_srt(video, out) is None


def test_build_queue_translates_instead_of_transcribing_again(tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    srt_path = _srt(out)
    video = _video(tmp_path)

    manager = TaskManager(
        out,
        output_formats=["srt"],
        translate_to="zh",
    )
    tasks = manager.build_queue([video])

    assert len(tasks) == 1
    assert tasks[0].mode == "translate_only"
    assert tasks[0].source_srt == srt_path


def test_failed_translation_resumes_as_translation_only(
    tmp_path: Path, monkeypatch
):
    """The full cycle: translation failed, next run only translates."""
    out = tmp_path / "out"
    out.mkdir()
    _srt(out)
    video = _video(tmp_path)
    monkeypatch.setattr(
        main_module, "create_translator", lambda *a, **k: StubProvider()
    )
    config = SimpleNamespace(
        translate_to="zh",
        translation_provider="index_api",
        translation_proxy="",
        language="ja",
    )

    manager = TaskManager(
        out,
        output_formats=["srt"],
        translate_to="zh",
    )
    tasks = manager.build_queue([video])
    assert tasks[0].mode == "translate_only"

    ok, _segments, _error = main_module.run_translation_only(
        config, video, tasks[0].source_srt, source_lang="ja"
    )
    assert ok
    manager.mark_done(video)
    manager.save_progress()

    # Third act: everything is on disk, so the file is not queued at all.
    assert TaskManager(
        out,
        output_formats=["srt"],
        translate_to="zh",
    ).build_queue([video]) == []


def test_build_queue_transcribes_when_there_are_no_subtitles(tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    video = _video(tmp_path)

    manager = TaskManager(
        out,
        output_formats=["srt"],
        translate_to="zh",
    )
    tasks = manager.build_queue([video])

    assert len(tasks) == 1
    assert tasks[0].mode == "full"
    assert tasks[0].source_srt is None


def test_run_translation_only_writes_outputs_without_asr(
    tmp_path: Path, monkeypatch
):
    out = tmp_path / "out"
    out.mkdir()
    srt_path = _srt(out)
    video = _video(tmp_path)

    monkeypatch.setattr(
        main_module, "create_translator", lambda *a, **k: StubProvider()
    )
    config = SimpleNamespace(
        translate_to="zh",
        translation_provider="index_api",
        translation_proxy="",
        language="ja",
    )

    ok, segments, error = main_module.run_translation_only(
        config, video, srt_path, source_lang="ja"
    )

    assert (ok, segments, error) == (True, 0, "")
    # PotPlayer layout: the .srt stays the source, translations are tagged.
    assert "第一条原文" in srt_path.read_text(encoding="utf-8")
    assert (out / "clip.chs.srt").exists()
    assert "译文:第一条原文" in (out / "clip.chs.srt").read_text(encoding="utf-8")
    assert (out / "clip.bilingual.chs.srt").exists()


def test_detected_language_is_remembered_for_later_runs(tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()

    manager = TaskManager(out, output_formats=["srt"], translate_to="zh")
    assert manager.language_for("clip") == ""

    manager.set_language("clip", "ja")
    manager.save_progress()

    reloaded = TaskManager(out, output_formats=["srt"], translate_to="zh")
    reloaded.build_queue([])
    assert reloaded.language_for("clip") == "ja"
