"""Microsoft Edge web translation backend."""
from __future__ import annotations

import logging
import threading
import time

import requests

from src.translator.types import TranslationError

logger = logging.getLogger(__name__)

_TRANSLATE_URL = "https://edge.microsoft.com/translate/translatetext"
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 1.0  # seconds
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0.0.0 Safari/537.36"
)

_LANGUAGE_MAP = {
    "auto": "auto-detect",
    "zh": "zh-Hans",
    "zh-cn": "zh-Hans",
    "zh-hans": "zh-Hans",
    "zh-tw": "zh-Hant",
    "zh-hant": "zh-Hant",
}


class EdgeTranslator:
    """Translate subtitle text through Edge's JSON translation endpoint."""

    _rate_lock = threading.Lock()
    _rate_limit_until: float = 0.0

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})

    def translate(
        self, text: str, source_lang: str = "auto", target_lang: str = "zh"
    ) -> str:
        """Translate one text while retaining the provider interface."""
        if not text:
            return ""
        return self.translate_batch([text], source_lang, target_lang)[0]

    def translate_batch(
        self, texts: list[str], source_lang: str = "auto", target_lang: str = "zh"
    ) -> list[str]:
        """Translate a subtitle batch in one request without merging its rows."""
        if not texts:
            return []

        results = ["" for _ in texts]
        non_empty = [(index, text) for index, text in enumerate(texts) if text]
        if not non_empty:
            return results

        translated = self._request_batch(
            [text for _, text in non_empty], source_lang, target_lang
        )
        for (index, _), translation in zip(non_empty, translated, strict=True):
            results[index] = translation

        logger.info("Translation batch progress: %d/%d subtitle(s)", len(texts), len(texts))
        return results

    def _request_batch(
        self, texts: list[str], source_lang: str, target_lang: str
    ) -> list[str]:
        mapped_source = _map_language(source_lang)
        if mapped_source == "auto-detect":
            raise TranslationError(
                "Edge Translator requires an explicit source language; "
                "automatic language detection is not supported"
            )
        params = {
            "from": mapped_source,
            "to": _map_language(target_lang),
            "isEnterpriseClient": "false",
        }
        headers = {"Accept": "application/json"}
        last_exc: TranslationError | None = None

        for attempt in range(_MAX_RETRIES + 1):
            with EdgeTranslator._rate_lock:
                wait = EdgeTranslator._rate_limit_until - time.time()
            if wait > 0:
                logger.debug("Rate-limit cooldown: waiting %.1fs", wait)
                time.sleep(wait)

            try:
                response = self._session.post(
                    _TRANSLATE_URL,
                    params=params,
                    json=texts,
                    headers=headers,
                    timeout=30,
                )
            except requests.RequestException as exc:
                last_exc = TranslationError(f"Edge translate request failed: {exc}")
            else:
                if response.status_code == 429:
                    backoff = _RETRY_BACKOFF_BASE * (2**attempt)
                    logger.warning(
                        "Edge translate rate-limited, backoff %.1fs (attempt %d/%d)",
                        backoff,
                        attempt + 1,
                        _MAX_RETRIES,
                    )
                    with EdgeTranslator._rate_lock:
                        EdgeTranslator._rate_limit_until = time.time() + backoff
                    last_exc = TranslationError("Edge translate rate-limited (429)")
                elif not response.ok:
                    raise TranslationError(
                        f"Edge translate HTTP {response.status_code}: {response.text[:200]}"
                    )
                else:
                    return _parse_response(response, len(texts))

            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_BACKOFF_BASE * (2**attempt))

        raise last_exc or TranslationError("Edge translate failed after retries")


def _parse_response(response: requests.Response, expected_count: int) -> list[str]:
    """Extract exactly one translated string for each submitted input string."""
    try:
        data = response.json()
        if not isinstance(data, list) or len(data) != expected_count:
            raise ValueError(
                f"expected {expected_count} result(s), received "
                f"{len(data) if isinstance(data, list) else 'a non-list response'}"
            )

        translations = [item["translations"][0]["text"] for item in data]
        if not all(isinstance(text, str) for text in translations):
            raise ValueError("translation text is not a string")
        return translations
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise TranslationError(f"Invalid response format from Edge Translator: {exc}") from exc


def _map_language(language: str) -> str:
    """Map the project's ISO code to Edge Translator's language code."""
    normalized = (language or "auto").strip().lower()
    return _LANGUAGE_MAP.get(normalized, normalized)
