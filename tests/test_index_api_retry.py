"""Transient-failure handling for the Index-Translate API backend.

The public endpoint returns HTTP 504 (openresty gateway timeout) on a small
slice of requests. Before these tests, one 504 aborted the batch, which
aborted the whole subtitle file.
"""
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.translator.index_api import IndexApiTranslator
from src.translator.pipeline import translate_srt
from src.translator.types import (
    TranslateConfig,
    TranslationError,
    TransientTranslationError,
)


def _ok_response(*lines: str) -> Mock:
    response = Mock(ok=True, status_code=200)
    response.json.return_value = {
        "choices": [{"message": {"content": "\n".join(lines)}}]
    }
    return response


def _error_response(status: int, body: str = "gateway blown") -> Mock:
    response = Mock(ok=False, status_code=status)
    response.text = body
    return response


def _translator(**kwargs) -> IndexApiTranslator:
    kwargs.setdefault("retries", 3)
    kwargs.setdefault("retry_backoff", 0.0)
    return IndexApiTranslator(**kwargs)


def test_transient_http_504_is_retried_then_succeeds():
    translator = _translator()
    translator._session.post = Mock(
        side_effect=[_error_response(504), _ok_response("你好")]
    )

    assert translator.translate("Hello", "en", "zh") == "你好"
    assert translator._session.post.call_count == 2
    # The retry gets a shorter budget so a hung server cannot stall the file
    # for the full timeout once per attempt.
    first, second = translator._session.post.call_args_list
    assert first.kwargs["timeout"] == 120
    assert second.kwargs["timeout"] == 60


def test_transient_error_exhausts_retries_before_raising():
    translator = _translator(retries=2)
    translator._session.post = Mock(
        side_effect=[_error_response(504), _error_response(504)]
    )

    with pytest.raises(TranslationError, match="HTTP 504"):
        translator.translate("Hello", "en", "zh")
    assert translator._session.post.call_count == 2


def test_non_transient_http_412_is_not_retried():
    translator = _translator()
    translator._session.post = Mock(return_value=_error_response(412))

    with pytest.raises(TranslationError, match="HTTP 412"):
        translator.translate("Hello", "en", "zh")
    assert translator._session.post.call_count == 1


def test_connection_error_is_treated_as_transient():
    import requests

    translator = _translator(retries=2)
    translator._session.post = Mock(
        side_effect=[requests.ConnectionError("reset"), _ok_response("你好")]
    )

    assert translator.translate("Hello", "en", "zh") == "你好"
    assert translator._session.post.call_count == 2


def test_packed_batch_splits_after_a_persistent_timeout():
    """A group that keeps timing out is cut in half instead of failing."""
    translator = _translator(retries=1, batch_size=20)
    translator._session.post = Mock(
        side_effect=[
            _error_response(504),                       # all 4 lines: times out
            _ok_response("1. 译文一", "2. 译文二"),      # first half
            _ok_response("1. 译文三", "2. 译文四"),      # second half
        ]
    )

    assert translator.translate_batch(
        ["第一条", "第二条", "第三条", "第四条"], "ja", "zh"
    ) == ["译文一", "译文二", "译文三", "译文四"]
    assert translator._session.post.call_count == 3


def test_pipeline_retries_a_failed_batch_instead_of_losing_the_file(tmp_path: Path):
    source = tmp_path / "sample.srt"
    source.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\n第一条\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\n第二条\n",
        encoding="utf-8",
    )

    calls = {"n": 0}

    def flaky(texts, source_lang, target_lang):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TranslationError("Index API HTTP 504")
        return ["译文一", "译文二"]

    provider = Mock()
    provider.translate_batch = Mock(side_effect=flaky)

    out = translate_srt(
        source,
        "zh",
        provider=provider,
        config=TranslateConfig(
            batch_size=50, max_workers=1, request_delay=0, retry_delay=0
        ),
        output_path=tmp_path / "sample.bilingual.srt",
    )

    assert calls["n"] == 2
    assert "译文一" in out.read_text(encoding="utf-8")


def test_pipeline_still_fails_when_every_retry_fails(tmp_path: Path):
    source = tmp_path / "sample.srt"
    source.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\n第一条\n",
        encoding="utf-8",
    )
    provider = Mock()
    provider.translate_batch = Mock(
        side_effect=TranslationError("Index API HTTP 504")
    )

    with pytest.raises(TranslationError):
        translate_srt(
            source,
            "zh",
            provider=provider,
            config=TranslateConfig(
                batch_size=50, max_workers=1, request_delay=0, retry_delay=0,
                batch_retries=1,
            ),
            output_path=tmp_path / "sample.bilingual.srt",
        )

    assert provider.translate_batch.call_count == 2


def test_pipeline_does_not_retry_what_the_provider_already_retried(tmp_path: Path):
    """A transient error means the provider burned its attempts already."""
    source = tmp_path / "sample.srt"
    source.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\n第一条\n",
        encoding="utf-8",
    )
    provider = Mock()
    provider.translate_batch = Mock(
        side_effect=TransientTranslationError("Index API HTTP 504")
    )
    errors: list[BaseException] = []

    with pytest.raises(TransientTranslationError):
        translate_srt(
            source,
            "zh",
            provider=provider,
            config=TranslateConfig(
                batch_size=50, max_workers=1, request_delay=0, retry_delay=0,
                batch_retries=2,
            ),
            output_path=tmp_path / "sample.bilingual.srt",
            on_batch_error=errors.append,
        )

    assert provider.translate_batch.call_count == 1
    assert [str(e) for e in errors] == ["Index API HTTP 504"]
