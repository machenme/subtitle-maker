from pathlib import Path
from types import SimpleNamespace

from src.audio_extractor import AudioExtractor


def test_audio_cache_is_invalidated_when_source_file_changes(tmp_path: Path, monkeypatch):
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"first version")
    extractor = AudioExtractor(tmp_path / "temp")
    generated: list[Path] = []

    def fake_run(command, **_kwargs):
        output = Path(command[-1])
        output.write_bytes(b"x" * 2048)
        generated.append(output)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("src.audio_extractor.subprocess.run", fake_run)

    first = extractor.extract(source)
    source.write_bytes(b"second version with a different length")
    second = extractor.extract(source)

    assert first != second
    assert generated == [first, second]
