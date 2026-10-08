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
    iso_to_player_suffix,
)
from src.translator.parser import parse_srt
from src.translator.writer import write_bilingual_srt, write_translation_srt
from src.translator.edge import EdgeTranslator
from src.translator.gtx import GtxTranslator
from src.translator.index_api import IndexApiTranslator
from src.translator.llm import LlmTranslator
from src.translator.pipeline import translate_srt, translate_srt_with_outputs


def create_translator(provider: str = "bing", *, proxy: str = ""):
    """Create the configured translation backend."""
    if provider == "bing":
        return EdgeTranslator()
    if provider == "gtx":
        return GtxTranslator(proxy=proxy)
    if provider == "index_api":
        # The public endpoint rejects requests without a proxy on networks
        # that cannot reach it directly; reuse the gtx proxy setting.
        return IndexApiTranslator(proxy=proxy)
    if provider == "llm":
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
    "iso_to_player_suffix",
    "parse_srt",
    "translate_srt",
    "translate_srt_with_outputs",
    "create_translator",
    "write_bilingual_srt",
    "write_translation_srt",
]
