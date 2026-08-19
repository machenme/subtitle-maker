from pathlib import Path

from src.utils import atomic_copy_file, atomic_write_text


def test_atomic_write_text_replaces_existing_file_without_temp_artifacts(tmp_path: Path):
    output = tmp_path / "subtitle.srt"
    output.write_text("old", encoding="utf-8")

    atomic_write_text(output, "new subtitle")

    assert output.read_text(encoding="utf-8") == "new subtitle"
    assert list(tmp_path.glob(".subtitle.srt.*.tmp")) == []


def test_atomic_copy_file_replaces_destination(tmp_path: Path):
    source = tmp_path / "source.srt"
    destination = tmp_path / "destination.srt"
    source.write_text("source subtitle", encoding="utf-8")
    destination.write_text("old subtitle", encoding="utf-8")

    atomic_copy_file(source, destination)

    assert destination.read_text(encoding="utf-8") == "source subtitle"
