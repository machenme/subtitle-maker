from unittest.mock import Mock

import pytest

from src.translator.gtx import GtxTranslator
from src.translator.types import TranslationError


def test_gtx_batch_uses_one_newline_separated_request(monkeypatch):
    response = Mock(ok=True, status_code=200)
    response.json.return_value = ["第一条译文\n第二条译文\n第三条译文"]
    request = Mock(return_value=response)
    monkeypatch.setattr("src.translator.gtx.requests.get", request)

    translator = GtxTranslator("127.0.0.1:7897")
    assert translator.translate_batch(["第一条", "第二条", "第三条"], "en", "zh") == [
        "第一条译文",
        "第二条译文",
        "第三条译文",
    ]

    assert request.call_count == 1
    assert request.call_args.kwargs["proxies"] == {
        "http": "http://127.0.0.1:7897",
        "https": "http://127.0.0.1:7897",
    }
    assert request.call_args.kwargs["params"]["tl"] == "zh-CN"
    assert request.call_args.kwargs["params"]["q"] == "第一条\n第二条\n第三条"


def test_gtx_requires_proxy():
    with pytest.raises(TranslationError, match="requires a proxy"):
        GtxTranslator()


def test_gtx_rejects_changed_line_boundaries(monkeypatch):
    response = Mock(ok=True, status_code=200)
    response.json.return_value = ["只有一行"]
    monkeypatch.setattr("src.translator.gtx.requests.get", Mock(return_value=response))

    with pytest.raises(TranslationError, match="line boundaries"):
        GtxTranslator("127.0.0.1:7897").translate_batch(["第一条", "第二条"], "en", "zh")


def test_gtx_accepts_auto_detect_response_shape(monkeypatch):
    response = Mock(ok=True, status_code=200)
    response.json.return_value = [["第一条译文\n第二条译文", "en"]]
    monkeypatch.setattr("src.translator.gtx.requests.get", Mock(return_value=response))

    assert GtxTranslator("127.0.0.1:7897").translate_batch(
        ["第一条", "第二条"], "auto", "zh"
    ) == ["第一条译文", "第二条译文"]
