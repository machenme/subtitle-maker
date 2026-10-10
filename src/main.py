#!/usr/bin/env python
"""
Subtitle Maker CLI — end-to-end speech-to-text for audio and video files.

Usage:
    uv run python -m src.main --input ./videos --output ./subtitles
    uv run src/main.py --input . --output ./output --verbose
"""
from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging
import shutil
import signal
import threading
import time

from src.config import PipelineConfig
from src.translator import create_translator, translate_srt_with_outputs
from src.utils import scan_video_files
from src.audio_extractor import (
    AudioExtractionCancelled,
    AudioExtractionError,
    AudioExtractor,
)
from src.gpu_scheduler import GpuScheduler
from src.text_formatter import Segment, TextFormatter, set_cps_language
from src.task_manager import TaskManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reusable pipeline function (used by both CLI and GUI)
# ---------------------------------------------------------------------------

# Each pipeline stage owns a slice of one file's 0..1 progress. Reporting the
# raw per-stage fraction made a long ASR look stalled and then jump, because
# nothing ever crossed the stage boundaries.
_STAGE_SPANS: dict[str, tuple[float, float]] = {
    "extracting": (0.00, 0.12),
    "splitting": (0.12, 0.16),
    "loading_model": (0.16, 0.26),
    "transcribing": (0.26, 0.96),
    "writing": (0.96, 1.00),
}


def _emit_stage(progress_callback, stage: str, fraction: float) -> None:
    """Map a stage-local 0..1 fraction onto this file's overall 0..1 span."""
    if not progress_callback:
        return
    low, high = _STAGE_SPANS.get(stage, (0.0, 1.0))
    overall = low + (high - low) * max(0.0, min(1.0, float(fraction)))
    progress_callback(stage, overall, 1.0)


def _cleanup_temp_audio(wav_path: Path) -> None:
    """Remove WAV and its chunk directory.  Errors are silently ignored."""
    import shutil as _shutil
    try:
        wav_path.unlink(missing_ok=True)
        chunk_dir = wav_path.parent / f"{wav_path.stem}_chunks"
        if chunk_dir.exists():
            _shutil.rmtree(chunk_dir)
    except OSError:
        pass


def run_one_video(
    config: PipelineConfig,
    video_path: Path,
    *,
    progress_callback: callable = None,
    cancel_event: threading.Event = None,
    translation_targets: list[tuple[Path, str]] | None = None,
) -> tuple[bool, int, str]:
    """
    Process a single media file through the full pipeline.

    Args:
        config: Validated PipelineConfig.
        video_path: Path to the source media file.
        progress_callback: Optional callable(stage, current, total) for progress.
        cancel_event: Optional threading.Event; set to request graceful stop.
        translation_targets: Optional list that receives
            ``(srt_path, detected_language)`` for every subtitle written by
            this run. Callers use it to translate exactly the files produced
            here instead of rescanning the output directory.

    Returns:
        (success, segment_count, error_message)
    """
    _check_cancelled = lambda: cancel_event and cancel_event.is_set()
    temp_dir = config.effective_temp_dir
    extractor = AudioExtractor(
        temp_dir,
        # getattr: tests and older callers pass lightweight config stand-ins.
        ffmpeg_bin=getattr(config, "ffmpeg_path", "") or None,
        ffprobe_bin=getattr(config, "ffprobe_path", "") or None,
    )

    # --- Stage 1: Audio extraction + chunking ---
    if _check_cancelled():
        return (False, 0, "Cancelled before extraction")

    logger.info(f"Extracting audio: {video_path.name}")
    if progress_callback:
        _emit_stage(progress_callback, "extracting", 0.0)

    extract_kwargs = {}
    if progress_callback:
        extract_kwargs["progress_callback"] = (
            lambda fraction: _emit_stage(progress_callback, "extracting", fraction)
        )
    if cancel_event is not None:
        # Handing the event down lets a stop request kill ffmpeg instead of
        # waiting for a long (or hung) extraction to finish on its own.
        extract_kwargs["cancel_event"] = cancel_event
    try:
        wav_path = extractor.extract(video_path, **extract_kwargs)
    except AudioExtractionCancelled:
        # A half-written WAV would poison the cache on the next run.
        if config.cleanup_temp:
            try:
                _cleanup_temp_audio(extractor.wav_path_for(video_path))
            except AudioExtractionError:
                pass
        return (False, 0, "Cancelled during extraction")
    except Exception as exc:
        return (False, 0, str(exc))

    chunk_sec = config.chunk_duration
    if chunk_sec == 0:
        # Auto: split evenly by worker count so all workers finish simultaneously
        dur = extractor.get_duration(wav_path)
        chunk_sec = max(30, int(dur / config.max_workers))
        logger.info(f"Auto chunk: duration={dur:.0f}s, workers={config.max_workers} → {chunk_sec}s/chunk")
    if chunk_sec > 0:
        split_kwargs = {}
        if progress_callback:
            _emit_stage(progress_callback, "splitting", 0.0)
            split_kwargs["progress_callback"] = (
                lambda fraction: _emit_stage(progress_callback, "splitting", fraction)
            )
        if cancel_event is not None:
            split_kwargs["cancel_event"] = cancel_event
        try:
            chunks = extractor.split_wav(wav_path, chunk_sec, **split_kwargs)
        except AudioExtractionCancelled:
            if config.cleanup_temp:
                _cleanup_temp_audio(wav_path)
            return (False, 0, "Cancelled during splitting")
        except Exception as exc:
            if config.cleanup_temp:
                _cleanup_temp_audio(wav_path)
            return (False, 0, str(exc))
    else:
        chunks = [(0.0, wav_path)]

    logger.info(f"Audio ready: {len(chunks)} chunk(s)")

    if _check_cancelled():
        if config.cleanup_temp:
            _cleanup_temp_audio(wav_path)
        return (False, 0, "Cancelled after extraction")

    # --- Stage 2: GPU ASR ---
    scheduler_tasks = [(cp, video_path) for _, cp in chunks]
    scheduler = GpuScheduler(config)
    start_time = time.time()
    # Filled with one reason per failed chunk, including chunks a dead worker
    # never answered for — otherwise "some chunks failed" is all we can say.
    chunk_errors: dict[Path, str] = {}

    try:
        raw_results = scheduler.process(
            scheduler_tasks,
            error_sink=chunk_errors,
            progress_callback=(
                lambda fraction: _emit_stage(
                    progress_callback, "transcribing", fraction
                )
                if progress_callback
                else None
            ),
            status_callback=(
                lambda stage: _emit_stage(progress_callback, stage, 0.0)
                if progress_callback
                else None
            ),
        )
    except Exception:
        if config.cleanup_temp:
            _cleanup_temp_audio(wav_path)
        raise
    elapsed = time.time() - start_time

    if _check_cancelled():
        if config.cleanup_temp:
            _cleanup_temp_audio(wav_path)
        return (False, 0, "Cancelled after transcription")

    # --- Stage 3: Merge & write ---
    formatter = TextFormatter()
    output_dir = config.output_dir

    # Collect chunk results
    chunk_results: list[tuple[float, list[Segment]]] = []
    detected_languages: list[str] = []
    all_done = True
    for offset, chunk_path in chunks:
        if chunk_path in raw_results:
            chunk_segments, detected_language = raw_results[chunk_path]
            chunk_results.append((offset, chunk_segments))
            detected_languages.append(detected_language)
            continue
        all_done = False
        reason = chunk_errors.get(chunk_path, "no result returned by the GPU worker")
        logger.error(
            "Transcription failed for %s (offset %ss): %s",
            chunk_path.name, offset, reason,
        )

    if not all_done or not chunk_results:
        if config.cleanup_temp:
            _cleanup_temp_audio(wav_path)
        failed_chunks = [c for c in chunks if c[1] not in raw_results]
        first_reason = chunk_errors.get(
            failed_chunks[0][1], "no result returned by the GPU worker"
        ) if failed_chunks else ""
        return (
            False,
            0,
            f"{len(failed_chunks)}/{len(chunks)} chunk(s) failed transcription"
            + (f": {first_reason}" if first_reason else ""),
        )

    if len(chunk_results) > 1:
        segments = formatter.combine_chunk_segments(chunk_results)
        logger.info(f"Combined {len(chunk_results)} chunks → {len(segments)} segments")
    else:
        segments = chunk_results[0][1]

    detected_language = config.language
    if config.language == "auto" and detected_languages:
        detected_language = max(
            dict.fromkeys(detected_languages), key=detected_languages.count
        )
    if detected_language == "auto":
        raise RuntimeError("Whisper did not return a detected source language")
    if progress_callback:
        progress_callback("detected_language", detected_language, detected_language)

    video_out_dir = output_dir
    video_out_dir.mkdir(parents=True, exist_ok=True)
    set_cps_language(detected_language)
    try:
        _emit_stage(progress_callback, "writing", 0.0)
        written = formatter.write_all(
            segments,
            base_path=video_out_dir / video_path.stem,
            formats=config.output_formats,
        )

        # Hand the caller the exact file this run produced, together with the
        # language Whisper actually detected. A later translation pass uses it
        # instead of rescanning the output directory, which would also pick up
        # old subtitles, previous translations and source copies.
        if (
            translation_targets is not None
            and config.translate_to
            and config.translation_provider == "llm"
        ):
            srt_written = next(
                (p for p in written if p.suffix.lower() == ".srt"), None
            )
            if srt_written is not None:
                translation_targets.append((srt_written, detected_language))

        # --- Stage 3b: Translation (optional) ---
        # The local LLM backend runs as a separate pass after all ASR work
        # (see run_batch), so it never competes with Whisper for VRAM.
        translated_path: Path | None = None
        if config.translate_to and config.translation_provider != "llm":
            srt_path = video_out_dir / f"{video_path.stem}.srt"
            if not srt_path.exists():
                raise ValueError("Translation requires SRT output")
            logger.info(
                f"Translating SRT: {srt_path.name} → {config.translate_to}"
            )
            provider = create_translator(
                config.translation_provider,
                proxy=config.translation_proxy,
            )
            translated_path, bilingual_path, original_path = translate_srt_with_outputs(
                srt_path,
                config.translate_to,
                provider=provider,
                source_lang=detected_language,
                swap_subtitles=config.swap_subtitles,
            )
            written.extend(
                [translated_path, original_path]
                if config.swap_subtitles and original_path
                else [translated_path]
            )
            if config.swap_subtitles:
                logger.info(
                    "Subtitle outputs: %s (translated), %s (bilingual), %s (original)",
                    srt_path.name,
                    bilingual_path.name if bilingual_path else "",
                    original_path.name if original_path else "",
                )
    except Exception:
        if config.cleanup_temp:
            _cleanup_temp_audio(wav_path)
        raise

    # --- Cleanup temp audio for this video ---
    if config.cleanup_temp:
        _cleanup_temp_audio(wav_path)

    logger.info(
        f"✓ {video_path.stem}: {len(segments)} segments → "
        f"{', '.join(p.suffix for p in written)}  ({elapsed:.1f}s)"
    )

    if progress_callback:
        _emit_stage(progress_callback, "writing", 1.0)
        progress_callback("done", len(segments), len(segments))

    return (True, len(segments), "")


# ---------------------------------------------------------------------------
# Batch runner (CLI)
# ---------------------------------------------------------------------------

# Global flag for graceful shutdown
_shutdown_requested = False


def _on_sigint(signum, frame):
    global _shutdown_requested
    print("\n⚠ Ctrl+C received. Finishing current tasks and exiting...")
    _shutdown_requested = True


def _run_llm_translation_pass(
    config: PipelineConfig,
    targets: list[tuple[Path, str]],
) -> int:
    """Translate this run's SRTs with the local LLM, exclusively on the GPU.

    Called after the ASR loop so Whisper workers have already released their
    VRAM; the translator probes free VRAM at load time and sizes its context
    and batch-token budget accordingly (1GB safety margin included).

    *targets* is the list ASR reported as newly written, as
    ``(srt_path, detected_language)`` pairs. Nothing is inferred by scanning
    the output directory, so pre-existing subtitles, earlier translations and
    source copies are never re-translated.
    Returns the number of failures.
    """
    from src.translator.llm import LlmTranslator

    # De-duplicate while preserving order: the same file can only reach the
    # list once, but a defensive pass costs nothing.
    pending: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for srt_path, language in targets:
        if srt_path in seen:
            continue
        seen.add(srt_path)
        pending.append((srt_path, language))

    if not pending:
        logger.info("LLM translation pass: nothing to translate")
        return 0

    # Skip files whose translation outputs are already on disk, so a rerun
    # after a partial failure only redoes the missing part.
    todo: list[tuple[Path, str]] = []
    for srt_path, language in pending:
        if _translation_outputs_present(config, srt_path):
            logger.info("LLM translation already present, skipping: %s", srt_path.name)
            continue
        todo.append((srt_path, language))
    if not todo:
        logger.info("LLM translation pass: every subtitle is already translated")
        return 0

    logger.info(
        "LLM translation pass: %d subtitle file(s); ASR finished, "
        "loading local model with exclusive VRAM access", len(todo),
    )
    failed = 0
    provider = None
    try:
        # ASR workers exited, but Windows releases their VRAM asynchronously —
        # wait until the model can actually fit before loading.
        from src.translator.llm import LlmTranslator, wait_for_vram_release

        wait_for_vram_release(LlmTranslator.minimum_free_vram_bytes())
        provider = LlmTranslator()
    except Exception as exc:
        logger.error("Failed to initialize local translation model: %s", exc)
        return len(todo)

    try:
        for srt_path, language in todo:
            if _shutdown_requested:
                break
            logger.info("LLM translating: %s → %s", srt_path.name, config.translate_to)
            try:
                translate_srt_with_outputs(
                    srt_path,
                    config.translate_to,
                    provider=provider,
                    source_lang=language or config.language,
                    swap_subtitles=config.swap_subtitles,
                )
            except Exception as exc:
                failed += 1
                logger.error("LLM translation failed for %s: %s", srt_path.name, exc)
    finally:
        # Give the VRAM back so the desktop / other tools get a clean GPU even
        # when a translation raised.
        try:
            provider.release()
        except Exception:
            logger.debug("Releasing the translation model failed", exc_info=True)
    logger.info("LLM translation pass complete (%d failure(s))", failed)
    return failed


def _translation_outputs_present(config: PipelineConfig, srt_path: Path) -> bool:
    """True when this SRT already has the translation outputs enabled by config."""
    from src.utils import is_srt_valid

    if not is_srt_valid(srt_path):
        return False
    if not config.swap_subtitles:
        from src.translator.types import iso_to_player_suffix

        target = srt_path.with_stem(
            f"{srt_path.stem}.{iso_to_player_suffix(config.translate_to)}"
        )
        return is_srt_valid(target)
    return is_srt_valid(srt_path.with_stem(f"{srt_path.stem}.bilingual"))


def _exit_code(*, failed_count: int, done_count: int, interrupted: bool) -> int:
    if interrupted:
        return 130
    if failed_count > 0 and done_count == 0:
        return 3
    if failed_count > 0:
        return 2
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="subtitle-maker",
        description="Offline batch Japanese audio/video speech-to-text with GPU parallel inference.",
    )
    # Required
    p.add_argument("--input", required=True, help="Input audio/video directory")
    p.add_argument("--output", required=True, help="Output text/subtitle directory")

    # Optional overrides
    p.add_argument("--config", default="./config.yaml", help="Path to config.yaml")
    p.add_argument("--model", default=None, help="Model size: large-v3-turbo | large-v3 | medium")
    p.add_argument("--workers", type=int, default=None, help="Max parallel GPU workers")
    p.add_argument("--temp-dir", default=None, help="Temp audio directory")
    p.add_argument("--language", default=None, help="Target language (ISO 639-1)")
    p.add_argument("--beam-size", type=int, default=None, help="Beam search width (1-10)")
    p.add_argument("--no-vad", action="store_true", help="Disable VAD filter")
    p.add_argument("--compute-type", default=None, help="float16 | int8_float16 | int8")
    p.add_argument("--no-cleanup", action="store_true", help="Keep temp audio files")
    p.add_argument("--chunk-duration", type=int, default=None,
                   help="Split audio into N-second chunks (0=automatic chunking)")
    p.add_argument("--translate", default=None, metavar="LANG",
                   help="Auto-translate SRT to target language (ISO 639-1, e.g. zh)")
    p.add_argument("--translator", default=None, choices=("bing", "gtx", "llm"),
                   help="Translation backend (default: bing; llm = local Index-Translate GGUF)")
    p.add_argument("--proxy", default=None, metavar="URL",
                   help="Proxy for Legacy GTX, e.g. 127.0.0.1:7897")
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    p.add_argument("--force", action="store_true", help="Re-process all files (ignore checkpoint)")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Parse CLI ---
    parser = build_argparser()
    cli = parser.parse_args()

    # --- Setup logging ---
    log_level = logging.DEBUG if cli.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Build config ---
    cli_dict = {
        "config": cli.config,
        "input_dir": cli.input,
        "output_dir": cli.output,
        "model": cli.model,
        "workers": cli.workers,
        "temp_dir": cli.temp_dir,
        "language": cli.language,
        "beam_size": cli.beam_size,
        "compute_type": cli.compute_type,
        "chunk_duration": cli.chunk_duration,
        "translate_to": cli.translate,
        "translation_provider": cli.translator,
        "translation_proxy": cli.proxy,
    }
    if cli.no_vad:
        cli_dict["vad_filter"] = False
    if cli.no_cleanup:
        cli_dict["cleanup_temp"] = False
    if cli.verbose:
        cli_dict["verbose"] = True
    # Clean None values so they don't override YAML defaults
    cli_dict = {k: v for k, v in cli_dict.items() if v is not None}

    try:
        config = PipelineConfig.build(cli_dict)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info(f"Config loaded: model={config.model_size}, workers={config.max_workers}, "
                f"language={config.language}, beam={config.beam_size}, "
                f"compute={config.compute_type}, vad={config.vad_filter}, "
                f"chunk_duration={config.chunk_duration}s")

    # --- Signal handling ---
    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigint)

    # --- Scan input ---
    logger.info(f"Scanning input directory: {config.input_dir}")
    video_paths = scan_video_files(config.input_dir, config.video_extensions)
    if not video_paths:
        logger.warning(f"No audio/video files found in {config.input_dir}")
        sys.exit(0)

    logger.info(f"Found {len(video_paths)} media file(s)")

    stems = [path.stem for path in video_paths]
    duplicates = sorted({stem for stem in stems if stems.count(stem) > 1})
    if duplicates:
        logger.error(
            "Duplicate video stems would overwrite outputs: "
            + ", ".join(duplicates)
        )
        sys.exit(1)

    # --- Task manager (resume checkpoint) ---
    task_mgr = TaskManager(
        config.output_dir,
        output_formats=config.output_formats,
        translate_to=config.translate_to,
        swap_subtitles=config.swap_subtitles,
        source_lang=config.language,
    )
    tasks = task_mgr.build_queue(video_paths, force=cli.force)
    if not tasks:
        logger.info("All videos already processed. Nothing to do.")
        sys.exit(0)

    logger.info(f"Pending tasks: {len(tasks)} / {len(video_paths)} total")

    total_segments = 0
    failed_count = 0
    start_time = time.time()
    # Filled by run_one_video: exactly the subtitles this run produced, each
    # paired with the language ASR detected for it.
    translation_targets: list[tuple[Path, str]] = []

    for task in tasks:
        if _shutdown_requested:
            break
        task_mgr.mark_started(task.video_path)
        try:
            ok, seg_count, err = run_one_video(
                config, task.video_path, translation_targets=translation_targets
            )
        except Exception as exc:
            logger.exception("Unhandled pipeline error for %s", task.video_path.name)
            ok, seg_count, err = False, 0, str(exc)
        if ok:
            task_mgr.mark_done(task.video_path)
            total_segments += seg_count
        elif err.startswith("Cancelled"):
            # A user stop is not a media failure: keep the task retryable.
            logger.warning("%s: %s", task.video_path.name, err)
        else:
            task_mgr.mark_failed(task.video_path, err)
            failed_count += 1

    # --- Stage 4: local-LLM translation pass (after ASR) ---
    # Run only when every video is transcribed, so the translation model can
    # claim the free VRAM exclusively instead of competing with Whisper.
    if (
        config.translate_to
        and config.translation_provider == "llm"
        and not _shutdown_requested
        and task_mgr.done_count == task_mgr.total_count
    ):
        failed_count += _run_llm_translation_pass(config, translation_targets)

    elapsed = time.time() - start_time

    # --- Cleanup ---
    if config.cleanup_temp and config.effective_temp_dir.exists():
        try:
            shutil.rmtree(config.effective_temp_dir)
        except Exception as exc:
            logger.warning(f"Failed to clean temp dir: {exc}")

    # --- Summary ---
    logger.info(
        f"Pipeline complete in {elapsed:.1f}s | "
        f"Tasks: {task_mgr.done_count}/{task_mgr.total_count} done"
        + (f", {failed_count} failed" if failed_count else "")
        + f" | Total segments: {total_segments}"
    )

    # --- Save progress ---
    task_mgr.save_progress()

    # --- Exit code ---
    sys.exit(
        _exit_code(
            failed_count=failed_count,
            done_count=task_mgr.done_count,
            interrupted=_shutdown_requested,
        )
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Windows multiprocessing guard
    import multiprocessing
    multiprocessing.freeze_support()
    main()
