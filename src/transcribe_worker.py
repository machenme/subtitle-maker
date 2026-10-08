"""
Stage 2 sub-module: Single GPU worker process.
Loads WhisperModel once at startup, loops consuming audio paths,
produces segment lists.
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.text_formatter import Segment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker entry point (runs in child process)
# ---------------------------------------------------------------------------

# Progress granularity. Reporting every decoded segment floods the queue on
# long files without making the bar any smoother.
_PROGRESS_STEP = 0.02


def _report_progress(progress_queue, audio_path: Path, fraction: float) -> None:
    """Best-effort progress report; telemetry must never break transcription."""
    if progress_queue is None:
        return
    try:
        progress_queue.put((str(audio_path), fraction))
    except Exception:  # pragma: no cover - queue torn down mid-shutdown
        logger.debug("Dropped progress report for %s", audio_path, exc_info=True)


def transcribe_worker(
    model_path: str,
    audio_path: Path,
    language: str = "auto",
    beam_size: int = 5,
    vad_filter: bool = True,
    compute_type: str = "float16",
    progress_queue=None,
) -> tuple[list[Segment], str]:
    """
    Transcribe a single audio file using faster-whisper.

    This function is called inside the child process. The model is loaded
    ONCE per process and reused across calls via the worker loop in gpu_scheduler.

    Args:
        model_path: Path to CTranslate2 model directory.
        audio_path: Path to 16kHz mono WAV file.
        language: ISO 639-1 language code.
        beam_size: Beam search width.
        vad_filter: Enable Silero VAD.
        compute_type: "float16", "int8_float16", etc.
        progress_queue: Optional queue that receives ``(audio_path, fraction)``
            as segments stream out of the decoder, so the UI can show progress
            inside a single long chunk instead of only between chunks.

    Returns:
        Transcribed segments and Whisper's detected ISO 639-1 language code.
    """
    # Deferred import: only the child process imports faster_whisper
    from faster_whisper import WhisperModel

    # --- load model with fallback ---
    model = _load_model(model_path, compute_type)

    # "auto" → None so faster-whisper auto-detects the language
    lang = None if language == "auto" else language

    logger.info(f"Transcribing: {audio_path.name}")

    segments_raw, info = model.transcribe(
        str(audio_path),
        language=lang,
        beam_size=beam_size,
        vad_filter=vad_filter,
    )

    logger.info(
        f"[{audio_path.stem}] Detected language: {info.language} "
        f"(p={info.language_probability:.2f}), "
        f"duration={info.duration:.1f}s"
    )

    segments: list[Segment] = []
    # Segments carry absolute timestamps into the chunk, so seg.end / duration
    # is a true 0..1 progress figure for this chunk — the only signal available
    # while Whisper is still decoding.
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    last_reported = -1.0
    _report_progress(progress_queue, audio_path, 0.0)
    for seg in segments_raw:
        segments.append(Segment(
            start=seg.start,
            end=seg.end,
            text=seg.text,
            avg_logprob=seg.avg_logprob,
        ))
        if duration > 0:
            fraction = min(1.0, max(0.0, seg.end / duration))
            if fraction - last_reported >= _PROGRESS_STEP:
                last_reported = fraction
                _report_progress(progress_queue, audio_path, fraction)
    # A silent or segment-free chunk still has to close at 100%, otherwise the
    # aggregate would stall below full for the rest of the run.
    _report_progress(progress_queue, audio_path, 1.0)

    logger.info(f"[{audio_path.stem}] → {len(segments)} segments")
    return segments, info.language


# ---------------------------------------------------------------------------
# Model loader (per-process singleton)
# ---------------------------------------------------------------------------

_model_cache: dict[str, object] = {}


def _load_model(model_path: str, compute_type: str):
    """Load WhisperModel with fallback on compute_type."""
    from faster_whisper import WhisperModel

    global _model_cache
    cache_key = f"{model_path}:{compute_type}"

    if cache_key in _model_cache:
        return _model_cache[cache_key]

    # Try primary compute_type, fallback to int8_float16
    for ct in (compute_type, "int8_float16"):
        try:
            logger.info(f"Loading WhisperModel from {model_path} (compute_type={ct})")
            model = WhisperModel(
                model_path,
                device="cuda",
                compute_type=ct,
            )
            _model_cache[cache_key] = model
            return model
        except Exception as exc:
            logger.warning(f"Failed with compute_type={ct}: {exc}")
            if ct == compute_type:
                continue
            raise

    raise RuntimeError(f"Failed to load model from {model_path}")
