"""Legacy Google Translate GTX backend."""
from __future__ import annotations

import logging
from typing import Any

import requests

from src.translator.types import TranslationError

logger = logging.getLogger(__name__)

_GTX_URL = "https://translate.googleapis.com/translate_a/t"
_DEFAULT_TIMEOUT = 60

_TARGET_LANGUAGE_MAP = {
    "zh": "zh-CN",
    "zh-cn": "zh-CN",
    "zh-hans": "zh-CN",
    "zh-tw": "zh-TW",
    "zh-hant": "zh-TW",
}


class GtxTranslator:
    """Translate batches through the legacy anonymous GTX endpoint."""

    def __init__(
        self,
        proxy: str | None = None,
        *,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        self.proxy = _normalize_proxy(proxy)
        self.timeout = timeout
        if not self.proxy:
            raise TranslationError(
                "Legacy GTX requires a proxy, for example 127.0.0.1:7897"
            )

    def translate(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "zh",
    ) -> str:
        """Translate one text through the GTX endpoint."""
        return self._request(text, source_lang, target_lang)

    def translate_batch(
        self,
        texts: list[str],
        source_lang: str = "auto",
        target_lang: str = "zh",
    ) -> list[str]:
        """Send a whole batch in one request, using newlines as boundaries."""
        if not texts:
            return []
        merged = "\n".join(" ".join(text.splitlines()) for text in texts)
        translated = self._request(merged, source_lang, target_lang)
        lines = translated.splitlines()
        if len(lines) != len(texts):
            raise TranslationError(
                "Legacy GTX changed the subtitle line boundaries; "
                f"expected {len(texts)} lines, got {len(lines)}"
            )
        return lines

    def _request(self, text: str, source_lang: str, target_lang: str) -> str:
        params = {
            "client": "gtx",
            "sl": _map_source_language(source_lang),
            "tl": _map_target_language(target_lang),
            "dt": "t",
            "q": text,
        }
        proxies = {"http": self.proxy, "https": self.proxy}
        try:
            response = requests.get(
                _GTX_URL,
                params=params,
                proxies=proxies,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise TranslationError(f"Legacy GTX request failed: {exc}") from exc

        if not response.ok:
            raise TranslationError(
                f"Legacy GTX HTTP {response.status_code}: {response.text[:300]}"
            )
        try:
            payload: Any = response.json()
            translated = payload[0]
            if isinstance(translated, list):
                translated = translated[0] if translated else None
        except (ValueError, IndexError, KeyError, TypeError) as exc:
            raise TranslationError("Legacy GTX returned an invalid response") from exc
        if not isinstance(translated, str):
            raise TranslationError("Legacy GTX returned non-text translation")
        return translated


def _normalize_proxy(proxy: str | None) -> str:
    value = (proxy or "").strip()
    if value and "://" not in value:
        value = f"http://{value}"
    return value


def _map_source_language(language: str) -> str:
    return (language or "auto").strip().lower() or "auto"


def _map_target_language(language: str) -> str:
    normalized = (language or "zh").strip().lower()
    return _TARGET_LANGUAGE_MAP.get(normalized, language)
