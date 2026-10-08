from pathlib import Path
import threading
from types import SimpleNamespace

from src.audio_extractor import AudioExtractor
from src.main import run_one_video


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


def test_cancelled_run_respects_temp_audio_cleanup_setting(tmp_path: Path, monkeypatch):
    source = tmp_path / "sample.mp4"
    source.touch()
    wav_path = tmp_path / "sample.wav"

    def fake_extract(self, _source, **_kwargs):
        wav_path.write_bytes(b"audio cache")
        return wav_path

    monkeypatch.setattr(AudioExtractor, "extract", fake_extract)

    for cleanup_temp, should_exist in ((False, True), (True, False)):
        cancel_event = threading.Event()
        config = SimpleNamespace(
            effective_temp_dir=tmp_path,
            cleanup_temp=cleanup_temp,
            chunk_duration=30,
        )

        result = run_one_video(
            config,
            source,
            progress_callback=lambda stage, *_args: (
                cancel_event.set() if stage == "extracting" else None
            ),
            cancel_event=cancel_event,
        )

        assert result[0] is False
        assert wav_path.exists() is should_exist
