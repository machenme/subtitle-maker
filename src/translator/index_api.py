"""Index-Translate official API backend (free public endpoint).

Calls ``https://index-translate.bilibili.com/v1/chat/completions`` — the
OpenAI-compatible server behind bilibili/Index-Translate, which serves
Index-Translate-35B-A3B. The API key is a placeholder: the public endpoint
accepts any value ("EMPTY" in the official docs).

Why this backend exists: the local GGUF path (:mod:`src.translator.llm`) is
fully offline but runs a 9B model on the user's GPU. This endpoint is a 35B
model, measurably better on both quality axes and ~1.4x faster end-to-end —
at the cost of sending subtitle text to a third-party server and depending on
a public service with no SLA.

Packing strategy deliberately mirrors :mod:`src.translator.llm`: numbered
lines in, numbered lines out, with a binary-split retry when the model drops
numbers. The upstream model does honour the numbered-line format; the same
failure mode appears at larger batch sizes, so the retry is not optional.
"""
from __future__ import annotations

import logging
import re
from typing import Any

import requests

from src.translator.types import TranslationError

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://index-translate.bilibili.com/v1"
_DEFAULT_MODEL = "Index-Translate-35B-A3B"
_DEFAULT_API_KEY = "EMPTY"
_DEFAULT_TIMEOUT = 120

# The public endpoint sits behind a WAF that rejects requests without a
# browser-like User-Agent (HTTP 412). Sending the stdlib default is enough to
# get blocked, so always present one.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Full language names required by the model card's prompt template. Names are
# written in Chinese because the prompts themselves are Chinese.
_LANGUAGE_NAMES: dict[str, str] = {
    "zh": "中文",
    "zh-cn": "中文",
    "zh-hans": "中文",
    "zh-hant": "繁体中文",
    "zh-tw": "繁体中文",
    "en": "英语",
    "ja": "日语",
    "ko": "韩语",
    "fr": "法语",
    "de": "德语",
    "es": "西班牙语",
    "pt": "葡萄牙语",
    "it": "意大利语",
    "ru": "俄语",
    "ar": "阿拉伯语",
    "th": "泰语",
    "vi": "越南语",
}

_PROMPT_TEMPLATE = (
    "将以下{source_part}文本翻译为{target_name}，"
    "注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
)

_MULTI_PROMPT_TEMPLATE = (
    "请将下面每一条编号文本分别翻译为{target_name}，"
    "输出时保留相同的编号，每条一行，"
    "只输出编号和翻译结果，不要输出任何解释：\n\n{text}"
)

_LINE_TEMPLATE = "{number}. {text}"
_RESPONSE_LINE_RE = re.compile(r"^\s*(\d+)\s*[.、．)）]\s*(.*)$")

# Upper bound on lines per packed request. Measured on the 35B endpoint:
# 20 lines parse reliably; beyond ~40 it starts dropping trailing numbers.
_MAX_PACKED_LINES = 20

# Matches the local backend's offline defaults (greedy, thinking off) so the
# same batch yields comparable output whichever backend is selected.
_SAMPLING: dict[str, Any] = {
    "temperature": 0.0,
    "max_tokens": 1024,
    # Officially documented default; without it the model narrates its
    # reasoning instead of translating, which costs ~13x output tokens.
    "chat_template_kwargs": {"enable_thinking": False},
}


def _language_name(language: str) -> str:
    normalized = (language or "").strip().lower()
    return _LANGUAGE_NAMES.get(normalized, normalized)


def _parse_numbered_lines(response: str, expected: int) -> list[str] | None:
    """Extract "N. translation" lines, or None when any number is missing."""
    translations: dict[int, str] = {}
    for raw_line in response.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _RESPONSE_LINE_RE.match(line)
        if not match:
            continue  # ignore instruction echo / stray prose
        number = int(match.group(1))
        text = match.group(2).strip()
        if not 1 <= number <= expected or not text:
            continue
        translations.setdefault(number, text)
    if len(translations) != expected:
        return None
    return [translations[n] for n in range(1, expected + 1)]


class IndexApiTranslator:
    """Translate subtitles through bilibili's public Index-Translate API."""

    def __init__(
        self,
        proxy: str | None = None,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        model: str = _DEFAULT_MODEL,
        api_key: str = _DEFAULT_API_KEY,
        timeout: int = _DEFAULT_TIMEOUT,
        batch_size: int = _MAX_PACKED_LINES,
    ) -> None:
        self.proxy = _normalize_proxy(proxy)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or _DEFAULT_API_KEY
        self.timeout = timeout
        self.batch_size = max(1, batch_size)
        self._session = requests.Session()
        # Reuse connections across batches; the local Bing backend already
        # follows this pattern (a session per request cannot pool sockets).
        self._session.headers.update({"User-Agent": _USER_AGENT})
        if self.proxy:
            self._session.proxies.update(
                {"http": self.proxy, "https": self.proxy}
            )

    # ------------------------------------------------------------------
    # TranslationProvider interface
    # ------------------------------------------------------------------

    def translate(
        self, text: str, source_lang: str = "auto", target_lang: str = "zh"
    ) -> str:
        if not text:
            return ""
        return self.translate_batch([text], source_lang, target_lang)[0]

    def translate_batch(
        self,
        texts: list[str],
        source_lang: str = "auto",
        target_lang: str = "zh",
    ) -> list[str]:
        """Translate a subtitle batch, packing lines into few requests."""
        if not texts:
            return []
        target_name = _language_name(target_lang)
        use_packing = target_name in _LANGUAGE_NAMES.values()
        if not use_packing:
            logger.info(
                "Target language %r lacks a localized name; falling back to "
                "per-line translation", target_lang,
            )

        results: list[str] = [""] * len(texts)
        groups: list[list[int]] = []
        current: list[int] = []
        for index, text in enumerate(texts):
            if not text:
                continue
            # Tokenizer-free heuristic: subtitle lines are short and uniform,
            # and the endpoint caps by output budget anyway.
            if len(current) >= self.batch_size:
                groups.append(current)
                current = []
            current.append(index)
        if current:
            groups.append(current)

        for group in groups:
            if use_packing and len(group) > 1:
                self._translate_packed(group, texts, results, target_name)
            else:
                for index in group:
                    results[index] = self._complete(
                        _PROMPT_TEMPLATE.format(
                            source_part="",
                            target_name=target_name,
                            text=texts[index],
                        )
                    )

        logger.info(
            "Index API translation batch complete: %d subtitle(s) in %d "
            "request(s)", len(texts), len(groups),
        )
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _translate_packed(
        self,
        group: list[int],
        texts: list[str],
        results: list[str],
        target_name: str,
    ) -> None:
        """One packed request, splitting in half when numbers go missing."""
        if len(group) == 1:
            index = group[0]
            results[index] = self._complete(
                _PROMPT_TEMPLATE.format(
                    source_part="", target_name=target_name, text=texts[index]
                )
            )
            return

        packed = "\n".join(
            _LINE_TEMPLATE.format(number=n + 1, text=texts[i])
            for n, i in enumerate(group)
        )
        translated = self._complete(
            _MULTI_PROMPT_TEMPLATE.format(target_name=target_name, text=packed)
        )
        parsed = _parse_numbered_lines(translated, len(group))
        if parsed is not None:
            for n, index in enumerate(group):
                results[index] = parsed[n]
            return

        middle = len(group) // 2
        logger.info(
            "Packed translation of %d line(s) missing/malformed; splitting "
            "into %d + %d", len(group), middle, len(group) - middle,
        )
        self._translate_packed(group[:middle], texts, results, target_name)
        self._translate_packed(group[middle:], texts, results, target_name)

    def _complete(self, prompt: str) -> str:
        """Send one chat completion and return the cleaned translation."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            **_SAMPLING,
        }
        url = f"{self.base_url}/chat/completions"
        try:
            response = self._session.post(
                url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise TranslationError(f"Index API request failed: {exc}") from exc

        if not response.ok:
            hint = ""
            if response.status_code == 412:
                hint = (
                    "（HTTP 412 通常是网络/WAF 拦截：确认已配置代理，"
                    "例如 translation_proxy: 127.0.0.1:7897）"
                )
            raise TranslationError(
                f"Index API HTTP {response.status_code}{hint}: "
                f"{response.text[:300]}"
            )
        try:
            data: Any = response.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, IndexError, KeyError, TypeError) as exc:
            raise TranslationError(
                f"Index API returned an invalid response: {data!r}"[:300]
            ) from exc
        if not isinstance(content, str):
            raise TranslationError("Index API returned non-text content")
        return _clean(_strip_think(content))


def _strip_think(text: str) -> str:
    """Drop any chain-of-thought block, mirroring the local backend."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip().removeprefix("<think>").strip()


def _clean(text: str) -> str:
    """Strip stray whitespace/quotes the model may add around the answer."""
    cleaned = text.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'`“”‘’":
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _normalize_proxy(proxy: str | None) -> str:
    value = (proxy or "").strip()
    if value and "://" not in value:
        value = f"http://{value}"
    return value
