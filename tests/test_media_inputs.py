from pathlib import Path

from src.config import DEFAULT_MAX_WORKERS, DEFAULT_MEDIA_EXTENSIONS, PipelineConfig
from src.utils import scan_video_files


def test_scan_includes_common_audio_extensions(tmp_path: Path):
    for extension in ("m4a", "mp3", "wav", "flac", "ogg", "opus", "aac", "wma"):
        (tmp_path / f"sample.{extension}").write_bytes(b"audio")
    (tmp_path / "ignored.txt").write_text("ignore", encoding="utf-8")

    found = scan_video_files(tmp_path, DEFAULT_MEDIA_EXTENSIONS)

    assert {path.suffix.lower() for path in found} == {
        ".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma",
    }


def test_default_media_extensions_are_used_when_config_omits_them(tmp_path: Path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f'input_dir: "{tmp_path.as_posix()}"\n'
        f'model_path: "{model_path.as_posix()}"\n',
        encoding="utf-8",
    )

    config = PipelineConfig.build({"config": str(config_path)})

    assert config.video_extensions == DEFAULT_MEDIA_EXTENSIONS
    assert config.max_workers == DEFAULT_MAX_WORKERS == 16
