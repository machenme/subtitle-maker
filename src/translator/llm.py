"""
LLM translator placeholder — future integration point for local models.

When a suitable LLM backend is available, implement :class:`LlmTranslator`
following the :class:`TranslationProvider` protocol in :mod:`src.translator.types`.

Example planned architecture::

    class LlmTranslator:
        def translate(self, text, source_lang, target_lang) -> str: ...
        def translate_batch(self, texts, source_lang, target_lang) -> list[str]: ...
"""
from __future__ import annotations
