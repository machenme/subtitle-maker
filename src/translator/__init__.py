"""
SRT subtitle translation module.

Public API
----------
- :func:`translate_srt` — one-shot SRT translation (parse → translate → write).
- :func:`parse_srt` — parse an SRT file into :class:`SrtCue` list.
- :func:`write_bilingual_srt` — write bilingual SRT from cue list.
- :class:`EdgeTranslator` — Bing Translator web backend (legacy class name).
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
from src.translator.writer import write_bilingual_srt
from src.translator.edge import EdgeTranslator
from src.translator.pipeline import translate_srt

__all__ = [
    "EdgeTranslator",
    "ParseError",
    "SrtCue",
    "TranslateConfig",
    "TranslationError",
    "iso_to_player_suffix",
    "parse_srt",
    "translate_srt",
    "write_bilingual_srt",
]
