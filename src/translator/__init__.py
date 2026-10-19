"""
SRT subtitle translation module.

Public API
----------
- :func:`translate_srt` — one-shot SRT translation (parse → translate → write).
- :func:`translate_srt_with_outputs` — translation using the standard output layout.
- :func:`parse_srt` — parse an SRT file into :class:`SrtCue` list.
- :func:`write_bilingual_srt` — write bilingual SRT from cue list.
- :class:`EdgeTranslator` — Bing Translator web backend (legacy class name).
- :class:`GtxTranslator` — Legacy Google GTX web backend.
- :class:`IndexApiTranslator` — Index-Translate official API (35B-A3B, free).
- :class:`LlmTranslator` — Local GGUF backend (llama.cpp; Index-Translate-9B).
- :class:`TranslateConfig` — batch-tuning configuration.
- :class:`SrtCue` — parsed subtitle entry.
- :class:`TranslationError` — unrecoverable translation failure.
- :class:`ParseError` — malformed SRT input.
"""
from src.translator.types import (
    ParseError,
    SrtCue,
    TranslateConfig,
    TranslationError,
    bilingual_player_suffix,
    iso_to_player_suffix,
)
from src.translator.parser import parse_srt
from src.translator.writer import write_bilingual_srt, write_translation_srt
from src.translator.pipeline import translate_srt, translate_srt_with_outputs

# The backends pull in requests / llama_cpp. Importing them eagerly made every
# consumer of the parser pay for them, including the offline SRT→TXT converter
# (src.srt_convert), which needs no network at all. PEP 562 keeps
# `from src.translator import EdgeTranslator` working while loading each backend
# only when it is actually referenced.
_LAZY_BACKENDS = {
    "EdgeTranslator": "src.translator.edge",
    "GtxTranslator": "src.translator.gtx",
    "IndexApiTranslator": "src.translator.index_api",
    "LlmTranslator": "src.translator.llm",
}


def __getattr__(name: str):
    module_path = _LAZY_BACKENDS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_path), name)
    globals()[name] = value  # cache: only the first access pays the import
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


def create_translator(provider: str = "bing", *, proxy: str = ""):
    """Create the configured translation backend."""
    if provider == "bing":
        from src.translator.edge import EdgeTranslator
        return EdgeTranslator()
    if provider == "gtx":
        from src.translator.gtx import GtxTranslator
        return GtxTranslator(proxy=proxy)
    if provider == "index_api":
        # The public endpoint rejects requests without a proxy on networks
        # that cannot reach it directly; reuse the gtx proxy setting.
        from src.translator.index_api import IndexApiTranslator
        return IndexApiTranslator(proxy=proxy)
    if provider == "llm":
        from src.translator.llm import LlmTranslator
        return LlmTranslator()
    raise ValueError(f"Unknown translation provider: {provider}")

__all__ = [
    "EdgeTranslator",
    "GtxTranslator",
    "IndexApiTranslator",
    "LlmTranslator",
    "ParseError",
    "SrtCue",
    "TranslateConfig",
    "TranslationError",
    "bilingual_player_suffix",
    "iso_to_player_suffix",
    "parse_srt",
    "translate_srt",
    "translate_srt_with_outputs",
    "create_translator",
    "write_bilingual_srt",
    "write_translation_srt",
]
