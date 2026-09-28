import sys
from types import SimpleNamespace

from src.transcribe_worker import transcribe_worker


def test_transcribe_worker_returns_whisper_detected_language(monkeypatch, tmp_path):
    class FakeModel:
        def transcribe(self, *_args, **_kwargs):
            segment = SimpleNamespace(
                start=0.0,
                end=1.0,
                text="Hello",
                avg_logprob=-0.1,
            )
            info = SimpleNamespace(language="en", language_probability=0.99, duration=1.0)
            return iter([segment]), info

    class FakeWhisperModel:
        def __new__(cls, *_args, **_kwargs):
            return FakeModel()

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=FakeWhisperModel),
    )

    import src.transcribe_worker as worker_module

    worker_module._model_cache.clear()
    segments, language = transcribe_worker("model", tmp_path / "sample.wav")

    assert language == "en"
    assert [segment.text for segment in segments] == ["Hello"]
