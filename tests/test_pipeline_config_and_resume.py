from pathlib import Path

import pytest

from src.config import PipelineConfig
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
