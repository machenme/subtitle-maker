"""
Translation pipeline — parse SRT → batch → translate → merge → write.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from src.translator.types import (
    SrtCue,
    TranslateConfig,
    TranslationError,
    TranslationProvider,
    iso_to_player_suffix,
)
from src.translator.parser import parse_srt
from src.translator.writer import write_bilingual_srt, write_translation_srt
from src.utils import atomic_copy_file

logger = logging.getLogger(__name__)


def translate_srt(
    srt_path: str | Path,
    target_lang: str,
    *,
    provider: TranslationProvider,
    source_lang: str = "auto",
    config: TranslateConfig | None = None,
    output_path: str | Path | None = None,
    monolingual_output_path: str | Path | None = None,
    original_output_path: str | Path | None = None,
    progress_callback: callable | None = None,
) -> Path:
    """
    Translate an SRT file end-to-end.

    1. Parse *srt_path* into :class:`SrtCue` list.
    2. Split cues into batches of *config.batch_size*.
    3. Concurrently translate each batch via *provider*, preserving one
       result per cue.
    4. Merge results back in original order.
    5. Write bilingual SRT and optional translated-only/source copies.

    Args:
        srt_path: Path to the source SRT.
        target_lang: ISO 639-1 target language code.
        provider: A :class:`TranslationProvider` instance.
        source_lang: ISO 639-1 source language (default ``"auto"``).
        config: Optional tuning overrides.
        output_path: Explicit output path.  Defaults to
            ``{srt_stem}.{target_lang}.srt`` alongside the source. This is the
            bilingual output.
        monolingual_output_path: Optional path for the translated-only SRT.
        original_output_path: Optional path for a copy of the source SRT.
        progress_callback: Optional ``(fraction: float) -> None`` invoked as
            batches complete, so a long subtitle file is not a silent wait.

    Returns:
        Path to the written bilingual SRT file.

    Raises:
        TranslationError: All batches failed or provider is unreachable.
    """
    cfg = config or TranslateConfig()
    file_path = Path(srt_path)

    # --- 1. Parse ---
    cues = parse_srt(file_path)
    total = len(cues)
    logger.info("Parsed %d cues from %s", total, file_path.name)

    # --- 2. Batch ---
    texts = [cue.text for cue in cues]
    batches: list[tuple[int, list[int]]] = []  # [(batch_idx, [cue_indices])]
    for start in range(0, total, cfg.batch_size):
        end = min(start + cfg.batch_size, total)
        batches.append((start // cfg.batch_size, list(range(start, end))))

    logger.info(
        "%d batches (batch_size=%d, workers=%d)",
        len(batches), cfg.batch_size, cfg.max_workers,
    )

    # --- 3. Translate concurrently ---
    def _translate_batch(batch_idx: int, indices: list[int]) -> tuple[int, list[str]]:
        """Returns (batch_idx, [translated_lines])."""
        logger.info(
            "Translation batch %d/%d queued (%d subtitle(s))",
            batch_idx + 1,
            len(batches),
            len(indices),
        )
        # Flatten internal cue line breaks before sending them. Boundaries
        # between cues are preserved by translate_batch, not by response
        # newlines, which translation services may merge or rewrite.
        request_texts = [" ".join(texts[i].splitlines()) for i in indices]
        try:
            translate_batch = getattr(provider, "translate_batch", None)
            if callable(translate_batch):
                lines = list(translate_batch(request_texts, source_lang, target_lang))
            else:
                # Keep compatibility with simple providers that only expose
                # translate(), while still guaranteeing one request/result per cue.
                lines = [
                    provider.translate(text, source_lang, target_lang)
                    for text in request_texts
                ]

            if len(lines) != len(indices):
                raise TranslationError(
                    f"Batch {batch_idx} returned {len(lines)} translations for "
                    f"{len(indices)} cues; refusing to write misaligned subtitles"
                )
            logger.info(
                "Translation batch %d/%d complete",
                batch_idx + 1,
                len(batches),
            )
            return (batch_idx, [str(line) for line in lines])
        except TranslationError:
            logger.exception(
                "Translation batch %d/%d failed",
                batch_idx + 1,
                len(batches),
            )
            raise

    start_time = time.time()
    results: dict[int, list[str]] = {}

    executor = ThreadPoolExecutor(max_workers=cfg.max_workers)
    futures: dict[object, int] = {}
    next_batch = 0
    failed = False
    active_limit = 1  # Validate the provider boundary protocol before ramping up.
    try:
        # Keep only a bounded number of requests in flight. New work is
        # submitted only after an existing batch completes successfully, so
        # an invalid response cannot cause the rest of the file to be sent.
        while next_batch < len(batches) or futures:
            while next_batch < len(batches) and len(futures) < active_limit:
                if next_batch:
                    time.sleep(cfg.request_delay)
                idx, indices = batches[next_batch]
                futures[executor.submit(_translate_batch, idx, indices)] = idx
                next_batch += 1

            if not futures:
                break

            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            completed: list[tuple[int, list[str]]] = []
            failure: BaseException | None = None
            for future in sorted(done, key=lambda item: futures[item]):
                batch_idx = futures.pop(future)
                try:
                    completed.append(future.result())
                except BaseException as exc:
                    failure = exc
                    break

            if failure is not None:
                failed = True
                for future in futures:
                    future.cancel()
                raise failure

            for idx, lines in completed:
                results[idx] = lines
            if completed:
                active_limit = cfg.max_workers
                if progress_callback:
                    try:
                        progress_callback(len(results) / len(batches) if batches else 1.0)
                    except Exception:
                        logger.debug("Translation progress callback failed", exc_info=True)
    finally:
        # Finish already-started requests before reporting failure, so the GUI
        # cannot finish while worker threads continue logging/API activity.
        executor.shutdown(wait=True, cancel_futures=failed)

    elapsed = time.time() - start_time
    logger.info(
        "Translation done in %.1fs (%d/%d batches ok)",
        elapsed,
        len(results),
        len(batches),
    )
    if progress_callback:
        try:
            progress_callback(1.0)
        except Exception:
            logger.debug("Translation progress callback failed", exc_info=True)

    # --- 4. Merge ---
    for batch_idx, (_, indices) in enumerate(batches):
        lines = results.get(batch_idx, [texts[i] for i in indices])
        for j, cue_idx in enumerate(indices):
            cues[cue_idx].translation = lines[j] if j < len(lines) else cues[cue_idx].text

    # --- 5. Write ---
    suffix = iso_to_player_suffix(target_lang)
    out = Path(output_path) if output_path else file_path.with_stem(
        f"{file_path.stem}.{suffix}"
    )
    if original_output_path:
        atomic_copy_file(file_path, Path(original_output_path))
    write_bilingual_srt(cues, out)
    if monolingual_output_path:
        write_translation_srt(cues, monolingual_output_path)
    logger.info("Bilingual SRT written: %s", out)
    if monolingual_output_path:
        logger.info("Translated-only SRT written: %s", monolingual_output_path)

    return out


def translate_srt_with_outputs(
    srt_path: str | Path,
    target_lang: str,
    *,
    provider: TranslationProvider,
    source_lang: str = "auto",
    swap_subtitles: bool = True,
    config: TranslateConfig | None = None,
    progress_callback: callable | None = None,
) -> tuple[Path, Path | None, Path | None]:
    """Translate an SRT using the project's standard output file layout.

    Returns ``(translated_or_bilingual, bilingual, original)``. The final two
    paths are ``None`` when subtitle swapping is disabled.

    ``config`` lets a caller override pipeline tunables; local-LLM callers
    should pass :meth:`TranslateConfig.for_local_llm` so batch size and rate
    limiting match the in-process model instead of the Bing defaults.
    """
    file_path = Path(srt_path)
    if not swap_subtitles:
        translated_path = translate_srt(
            file_path,
            target_lang,
            provider=provider,
            source_lang=source_lang,
            config=config,
            progress_callback=progress_callback,
        )
        return translated_path, None, None

    source_code = source_lang if source_lang != "auto" else "ja"
    bilingual_path = file_path.with_stem(f"{file_path.stem}.bilingual")
    original_path = file_path.with_stem(
        f"{file_path.stem}.{iso_to_player_suffix(source_code)}"
    )
    translated_path = translate_srt(
        file_path,
        target_lang,
        provider=provider,
        source_lang=source_lang,
        config=config,
        output_path=bilingual_path,
        monolingual_output_path=file_path,
        original_output_path=original_path,
        progress_callback=progress_callback,
    )
    return translated_path, bilingual_path, original_path
