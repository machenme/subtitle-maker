import os
from pathlib import Path

import pytest

from src.config import PipelineConfig
from src.main import _exit_code
from src.task_manager import TaskManager


def _write_config(path: Path, model_path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                'input_dir: "."',
                'output_dir: "./yaml-output"',
                f'model_path: "{model_path.as_posix()}"',
                'output_formats: [srt]',
            ]
        ),
        encoding="utf-8",
    )


def _write_valid_srt(path: Path) -> None:
    path.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\n"
        "This is a valid subtitle line for testing.\n",
        encoding="utf-8",
    )


def test_cli_paths_override_yaml(tmp_path: Path):
    input_dir = tmp_path / "cli-input"
    output_dir = tmp_path / "cli-output"
    input_dir.mkdir()
    model_path = tmp_path / "model"
    model_path.mkdir()
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, model_path)

    config = PipelineConfig.build(
        {
            "config": str(config_path),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
        }
    )

    assert config.input_dir == input_dir.resolve()
    assert config.output_dir == output_dir.resolve()


def test_translation_requires_srt_output(tmp_path: Path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, model_path)

    with pytest.raises(ValueError, match="requires 'srt'"):
        PipelineConfig.build(
            {
                "config": str(config_path),
                "output_formats": ["txt"],
                "translate_to": "zh",
            }
        )


def test_changed_input_is_requeued_and_srt_only_output_is_complete(tmp_path: Path):
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"source-v1")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_valid_srt(output_dir / "sample.srt")

    first = TaskManager(output_dir, output_formats=["srt"])
    assert first.build_queue([source]) == [first._tasks["sample"]]
    first.mark_done(source)
    first.save_progress()

    unchanged = TaskManager(output_dir, output_formats=["srt"])
    assert unchanged.build_queue([source]) == []

    source.write_bytes(b"source-v2")
    changed = TaskManager(output_dir, output_formats=["srt"])
    pending = changed.build_queue([source])
    assert len(pending) == 1
    assert pending[0].video_path == source


def test_translation_defaults_to_chinese_and_requires_three_subtitles(tmp_path: Path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, model_path)

    config = PipelineConfig.build({"config": str(config_path)})
    assert config.translate_to == "zh"
    assert PipelineConfig.build({"config": str(config_path), "translate_to": ""}).translate_to == ""

    manager = TaskManager(
        tmp_path / "output",
        output_formats=["srt"],
        translate_to="zh",
        swap_subtitles=True,
        source_lang="en",
    )
    assert manager._expected_output_names("movie") == [
        "movie.bilingual.srt",
        "movie.eng.srt",
        "movie.srt",
    ]


def test_zero_chunk_duration_from_cli_overrides_yaml(tmp_path: Path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, model_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "\nchunk_duration: 900\n",
        encoding="utf-8",
    )

    config = PipelineConfig.build({"config": str(config_path), "chunk_duration": 0})

    assert config.chunk_duration == 0


def test_interrupted_run_has_nonzero_exit_code():
    assert _exit_code(failed_count=0, done_count=1, interrupted=True) == 130


def test_fingerprint_cache_returns_equal_result_without_rereading(tmp_path: Path):
    """A repeated fingerprint call must reuse the cached value, not redo the IO."""
    from src import task_manager as tm

    media = tmp_path / "movie.mp4"
    media.write_bytes(os.urandom(1024))

    tm._FINGERPRINT_CACHE.clear()
    calls: list[str] = []
    real_compute = tm._compute_fingerprint

    def counting_compute(path, size, mtime):
        calls.append(path.name)
        return real_compute(path, size, mtime)

    tm._compute_fingerprint = counting_compute
    try:
        first = tm._file_fingerprint(media)
        second = tm._file_fingerprint(media)
    finally:
        tm._compute_fingerprint = real_compute
        tm._FINGERPRINT_CACHE.clear()

    assert first == second
    assert calls == ["movie.mp4"], "second call must be served from the memo"


def test_fingerprint_cache_returns_a_copy(tmp_path: Path):
    """Callers must not be able to corrupt the memo through the return value."""
    from src import task_manager as tm

    media = tmp_path / "movie.mp4"
    media.write_bytes(os.urandom(1024))

    tm._FINGERPRINT_CACHE.clear()
    try:
        first = tm._file_fingerprint(media)
        first["head_tail_hash"] = "TAMPERED"
        second = tm._file_fingerprint(media)
    finally:
        tm._FINGERPRINT_CACHE.clear()

    assert second["head_tail_hash"] != "TAMPERED"


def test_fingerprint_recomputes_when_file_changes(tmp_path: Path):
    """A modified file must not be answered from the memo."""
    from src import task_manager as tm

    media = tmp_path / "movie.mp4"
    media.write_bytes(os.urandom(2048))

    tm._FINGERPRINT_CACHE.clear()
    try:
        before = tm._file_fingerprint(media)
        stat = media.stat()
        media.write_bytes(os.urandom(4096))
        # Ensure a distinct mtime even on coarse-grained filesystems.
        os.utime(media, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        after = tm._file_fingerprint(media)
    finally:
        tm._FINGERPRINT_CACHE.clear()

    assert before["head_tail_hash"] != after["head_tail_hash"]
    assert after["size"] == 4096
