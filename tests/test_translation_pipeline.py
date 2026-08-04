from pathlib import Path

from src.translator.pipeline import translate_srt
from src.translator.types import TranslateConfig


class RecordingProvider:
    def __init__(self):
        self.requests: list[list[str]] = []

    def translate_batch(
        self, texts: list[str], source_lang: str, target_lang: str
    ) -> list[str]:
        self.requests.append(texts)
        return ["第一条", "第二条"]


def test_multiline_cues_do_not_shift_batch_alignment(tmp_path: Path):
    source = tmp_path / "sample.srt"
    source.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\n原文第一行\n原文第二行\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\n第二条字幕\n",
        encoding="utf-8",
    )
    provider = RecordingProvider()

    output = translate_srt(
        source,
        "zh",
        provider=provider,
        config=TranslateConfig(batch_size=50, max_workers=1, request_delay=0),
    )

    assert provider.requests == [["原文第一行 原文第二行", "第二条字幕"]]
    content = output.read_text(encoding="utf-8")
    assert "原文第一行\n原文第二行\n第一条" in content
    assert "第二条字幕\n第二条" in content


class SimpleProvider:
    def __init__(self):
        self.requests: list[str] = []

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        self.requests.append(text)
        return f"译文:{text}"


def test_provider_without_batch_api_cannot_shift_results(tmp_path: Path):
    source = tmp_path / "sample.srt"
    source.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\n第一条\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\n第二条\n\n"
        "3\n00:00:07,000 --> 00:00:09,000\n第三条\n",
        encoding="utf-8",
    )
    provider = SimpleProvider()

    output = translate_srt(
        source,
        "zh",
        provider=provider,
        config=TranslateConfig(batch_size=2, max_workers=1, request_delay=0),
    )

    assert provider.requests == ["第一条", "第二条", "第三条"]
    content = output.read_text(encoding="utf-8")
    assert "第二条\n译文:第二条" in content
    assert "第一条 第二条" not in content
