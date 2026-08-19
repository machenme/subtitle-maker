from pathlib import Path

from src.translator.pipeline import translate_srt, translate_srt_with_outputs
from src.translator.types import TranslateConfig, TranslationError
from src.translator.edge import EdgeTranslator


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

    bilingual = tmp_path / "sample.bilingual.srt"
    translated = tmp_path / "sample.srt"
    original = tmp_path / "sample.jpn.srt"
    output = translate_srt(
        source,
        "zh",
        provider=provider,
        config=TranslateConfig(batch_size=50, max_workers=1, request_delay=0),
        output_path=bilingual,
        monolingual_output_path=translated,
        original_output_path=original,
    )

    assert provider.requests == [["原文第一行 原文第二行", "第二条字幕"]]
    content = output.read_text(encoding="utf-8")
    assert "原文第一行\n原文第二行\n第一条" in content
    assert "第二条字幕\n第二条" in content
    assert translated.read_text(encoding="utf-8") == (
        "1\n00:00:01,000 --> 00:00:03,000\n第一条\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\n第二条\n"
    )
    assert original.read_text(encoding="utf-8") == (
        "1\n00:00:01,000 --> 00:00:03,000\n原文第一行\n原文第二行\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\n第二条字幕\n"
    )


def test_standard_output_layout_writes_translated_bilingual_and_original(tmp_path: Path):
    source = tmp_path / "sample.srt"
    source.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\n原文第一条\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\n原文第二条\n",
        encoding="utf-8",
    )

    translated, bilingual, original = translate_srt_with_outputs(
        source,
        "zh",
        provider=RecordingProvider(),
        source_lang="ja",
    )

    assert translated == tmp_path / "sample.bilingual.srt"
    assert bilingual == translated
    assert original == tmp_path / "sample.jpn.srt"
    assert source.read_text(encoding="utf-8") == (
        "1\n00:00:01,000 --> 00:00:03,000\n第一条\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\n第二条\n"
    )
    assert original.read_text(encoding="utf-8") == (
        "1\n00:00:01,000 --> 00:00:03,000\n原文第一条\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\n原文第二条\n"
    )


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


class FailingBatchProvider:
    def __init__(self):
        self.calls: list[list[str]] = []

    def translate_batch(
        self, texts: list[str], source_lang: str, target_lang: str
    ) -> list[str]:
        self.calls.append(texts)
        raise TranslationError("batch separators were changed")


def test_translation_stops_submitting_batches_after_failure(tmp_path: Path):
    source = tmp_path / "sample.srt"
    source.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\n第一条\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\n第二条\n\n"
        "3\n00:00:07,000 --> 00:00:09,000\n第三条\n",
        encoding="utf-8",
    )
    provider = FailingBatchProvider()

    try:
        translate_srt(
            source,
            "zh",
            provider=provider,
            config=TranslateConfig(batch_size=1, max_workers=2, request_delay=0),
        )
    except TranslationError:
        pass
    else:
        raise AssertionError("Expected translation to fail")

    assert len(provider.calls) == 1


def test_edge_batch_uses_one_json_request_and_preserves_order(monkeypatch):
    provider = EdgeTranslator()
    captured: list[str] = []

    def fake_request_batch(texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        captured.extend(texts)
        return [f"{text}译文" for text in texts]

    monkeypatch.setattr(provider, "_request_batch", fake_request_batch)

    assert provider.translate_batch(
        ["第一条", "第二条", "第三条"], "ja", "zh"
    ) == ["第一条译文", "第二条译文", "第三条译文"]
    assert captured == ["第一条", "第二条", "第三条"]
