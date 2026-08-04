import time
from unittest.mock import Mock

from dotenv import dotenv_values

from src.translator.edge import EdgeTranslator


def _auth_response() -> Mock:
    html = (
        'var params_AbusePreventionHelper = '
        '[1785800000000,"token-value",3600000];'
        'var x={IG:"ig-value"};'
        '<div id="tta_outGDCont" data-iid="translator.5023"></div>'
    )
    response = Mock()
    response.ok = True
    response.status_code = 200
    response.text = html
    response.url = "https://cn.bing.com/Translator"
    return response


def test_translate_uses_bing_token_and_maps_chinese_language(tmp_path):
    env_path = tmp_path / ".env"
    translator = EdgeTranslator(env_path=env_path)
    translator._session.get = Mock(return_value=_auth_response())

    translated = Mock()
    translated.ok = True
    translated.status_code = 200
    translated.json.return_value = [
        {"translations": [{"text": "你好", "to": "zh-Hans"}]}
    ]
    translator._session.post = Mock(return_value=translated)

    assert translator.translate("Hello", "auto", "zh") == "你好"

    post = translator._session.post.call_args
    assert post.kwargs["data"]["fromLang"] == "auto-detect"
    assert post.kwargs["data"]["to"] == "zh-Hans"
    assert post.kwargs["data"]["key"] == "1785800000000"
    assert post.kwargs["data"]["token"] == "token-value"
    assert post.kwargs["params"] == {
        "isVertical": "1",
        "IG": "ig-value",
        "IID": "translator.5023",
    }
    env_values = dotenv_values(env_path)
    assert env_values["TRANSLATE_KEY"] == "1785800000000"
    assert env_values["TRANSLATE_TOKEN"] == "token-value"
    assert env_values["TRANSLATE_KEY_TIMESTAMP"]


def test_cached_token_is_reused_until_timestamp_expires(tmp_path):
    env_path = tmp_path / ".env"
    first = EdgeTranslator(env_path=env_path)
    first._session.get = Mock(return_value=_auth_response())
    first_response = Mock(ok=True, status_code=200)
    first_response.json.return_value = [
        {"translations": [{"text": "你好", "to": "zh-Hans"}]}
    ]
    first._session.post = Mock(return_value=first_response)
    assert first.translate("Hello", "auto", "zh") == "你好"

    cached = EdgeTranslator(env_path=env_path)
    cached._session.get = Mock(side_effect=AssertionError("cache should be used"))
    cached_response = Mock(ok=True, status_code=200)
    cached_response.json.return_value = [
        {"translations": [{"text": "你好", "to": "zh-Hans"}]}
    ]
    cached._session.post = Mock(return_value=cached_response)
    assert cached.translate("Hello", "auto", "zh") == "你好"
    cached._session.get.assert_not_called()

    expired_timestamp = int(time.time()) - 481
    expired_env = [
        line if not line.startswith("TRANSLATE_KEY_TIMESTAMP=")
        else f"TRANSLATE_KEY_TIMESTAMP={expired_timestamp}"
        for line in env_path.read_text(encoding="utf-8").splitlines()
    ]
    env_path.write_text("\n".join(expired_env) + "\n", encoding="utf-8")
    expired = EdgeTranslator(env_path=env_path)
    expired._session.get = Mock(return_value=_auth_response())
    expired_response = Mock(ok=True, status_code=200)
    expired_response.json.return_value = [
        {"translations": [{"text": "你好", "to": "zh-Hans"}]}
    ]
    expired._session.post = Mock(return_value=expired_response)
    assert expired.translate("Hello", "auto", "zh") == "你好"
    expired._session.get.assert_called_once()


def test_translate_refreshes_token_once_after_auth_failure(tmp_path):
    translator = EdgeTranslator(env_path=tmp_path / ".env")
    first_auth = _auth_response()
    second_auth = _auth_response()
    second_auth.text = (
        'var params_AbusePreventionHelper = '
        '[1785800000001,"new-token",3600000];'
        'var x={IG:"new-ig"};'
        '<div id="tta_outGDCont" data-iid="translator.5024"></div>'
    )
    translator._session.get = Mock(side_effect=[first_auth, second_auth])

    unauthorized = Mock(ok=False, status_code=401, text='{"ShowCaptcha":false}')
    success = Mock(ok=True, status_code=200)
    success.json.return_value = [
        {"translations": [{"text": "你好", "to": "zh-Hans"}]}
    ]
    translator._session.post = Mock(side_effect=[unauthorized, success])

    assert translator.translate("Hello", "auto", "zh") == "你好"
    assert translator._session.get.call_count == 2
    assert translator._session.post.call_count == 2
