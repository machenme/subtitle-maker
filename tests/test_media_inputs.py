from pathlib import Path

from src.config import DEFAULT_MAX_WORKERS, DEFAULT_MEDIA_EXTENSIONS, PipelineConfig
from src.utils import scan_video_files


def test_scan_walks_the_tree_once_regardless_of_extension_count(
    tmp_path: Path, monkeypatch
):
    """Scanning must not cost one full directory walk per extension."""
    for extension in ("mp4", "m4a", "mp3"):
        (tmp_path / f"sample.{extension}").write_bytes(b"audio")

    walks: list[str] = []
    real_walk = Path.rglob

    def counting_walk(pattern):
        walks.append(str(pattern))
        return real_walk(pattern)

    monkeypatch.setattr(Path, "rglob", counting_walk)

    found = scan_video_files(tmp_path, DEFAULT_MEDIA_EXTENSIONS)

    assert len(found) == 3
    assert walks == [], f"scan_video_files must not use per-extension globs: {walks}"


def test_scan_handles_uppercase_extensions_and_nested_dirs(tmp_path: Path):
    (tmp_path / "a.M4A").write_bytes(b"audio")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "deep.MP3").write_bytes(b"audio")
    (tmp_path / "skip.txt").write_text("skip", encoding="utf-8")

    found = scan_video_files(tmp_path, DEFAULT_MEDIA_EXTENSIONS)

    assert {path.name for path in found} == {"a.M4A", "deep.MP3"}


def test_scan_with_no_extensions_returns_empty(tmp_path: Path):
    (tmp_path / "a.mp4").write_bytes(b"audio")

    assert scan_video_files(tmp_path, []) == []


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
