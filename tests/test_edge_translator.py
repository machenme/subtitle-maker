from unittest.mock import Mock

import pytest

from src.translator.edge import EdgeTranslator
from src.translator.types import TranslationError


def _translation_response(*texts: str) -> Mock:
    response = Mock(ok=True, status_code=200)
    response.json.return_value = [
        {"translations": [{"text": text, "to": "zh-Hans"}]} for text in texts
    ]
    return response


def test_translate_uses_edge_json_endpoint_and_maps_languages():
    translator = EdgeTranslator()
    translator._session.post = Mock(return_value=_translation_response("你好"))

    assert translator.translate("Hello", "en", "zh") == "你好"

    post = translator._session.post.call_args
    assert post.args[0] == "https://edge.microsoft.com/translate/translatetext"
    assert post.kwargs["params"] == {
        "from": "en",
        "to": "zh-Hans",
        "isEnterpriseClient": "false",
    }
    assert post.kwargs["json"] == ["Hello"]
    assert post.kwargs["headers"] == {"Accept": "application/json"}


def test_translate_batch_preserves_subtitle_boundaries_and_empty_rows():
    translator = EdgeTranslator()
    translator._session.post = Mock(return_value=_translation_response("第一条译文", "第三条译文"))

    assert translator.translate_batch(["第一条", "", "第三条"], "ja", "zh") == [
        "第一条译文",
        "",
        "第三条译文",
    ]
    assert translator._session.post.call_args.kwargs["json"] == ["第一条", "第三条"]


def test_translate_batch_rejects_response_with_missing_rows():
    translator = EdgeTranslator()
    translator._session.post = Mock(return_value=_translation_response("第一条译文"))

    with pytest.raises(TranslationError, match="expected 2 result\\(s\\), received 1"):
        translator.translate_batch(["第一条", "第二条"], "ja", "zh")


def test_translate_batch_rejects_automatic_source_language():
    translator = EdgeTranslator()
    translator._session.post = Mock()

    with pytest.raises(TranslationError, match="explicit source language"):
        translator.translate_batch(["Hello"], "auto", "zh")

    translator._session.post.assert_not_called()
