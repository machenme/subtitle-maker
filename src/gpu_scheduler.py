"""
Stage 2: GPU scheduler with semaphore-based concurrency control.
Uses multiprocessing (spawn) to isolate CUDA contexts per worker.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
from pathlib import Path
from typing import Any

from src.config import PipelineConfig
from src.text_formatter import Segment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sentinel sent through the task queue to signal worker shutdown
_SHUTDOWN = "__SHUTDOWN__"
_WORKER_STARTUP_TIMEOUT = 120  # seconds for model loading
_MIN_FREE_VRAM_GB = 1.0  # emergency stop threshold while transcribing
_STARTUP_MIN_FREE_VRAM_GB = 2.5  # reserve for inference-time allocations
_STARTUP_MEMORY_MARGIN = 1.25
_RESULT_POLL_TIMEOUT = 1


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

TaskRequest = tuple[Path, Path]          # (audio_path, video_path)
TaskResult = tuple[Path, tuple[list[Segment], str]]  # (audio_path, (segments, language))
TaskError = tuple[Path, str]             # (video_path, error_message)


def _free_vram_gb(gpu_index: int = 0) -> float | None:
    """Return current free VRAM, or None when NVML is unavailable."""
    initialized = False
    try:
        import pynvml

        pynvml.nvmlInit()
        initialized = True
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).free / (1024 ** 3)
    except Exception:
        return None
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def _can_start_worker(
    free_vram_gb: float | None,
    previous_worker_vram_gb: float | None,
) -> bool:
    """Avoid starting a model when the next load would likely trigger swap."""
    if free_vram_gb is None:
        return True
    required = _STARTUP_MIN_FREE_VRAM_GB
    if previous_worker_vram_gb:
        required = max(required, previous_worker_vram_gb * _STARTUP_MEMORY_MARGIN)
    return free_vram_gb >= required


# ---------------------------------------------------------------------------
# Worker process body
# ---------------------------------------------------------------------------

def _worker_process(
    model_path: str,
    language: str,
    beam_size: int,
    vad_filter: bool,
    compute_type: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    startup_queue: mp.Queue,
    progress_queue: mp.Queue,
    semaphore: mp.Semaphore,
    worker_id: int,
) -> None:
    """
    Child process entry point.

    - Loads WhisperModel before announcing startup readiness.
    - Acquires semaphore slot before processing (controls GPU concurrency).
    - Reuses the loaded model while looping on task_queue.
    - Sends (audio_path, segments) or (audio_path, error_string) to result_queue.
    - Streams (audio_path, fraction) to progress_queue while decoding.
    """
    # Ensure nvidia CUDA DLLs are on the DLL search path BEFORE any import
    # of faster_whisper / ctranslate2. Must happen here in the child process
    # since spawn mode starts a fresh interpreter.
    import os as _os
    import sys as _sys

    dll_dirs: set[str] = set()
    for p in _sys.path:
        nvidia_root = _os.path.join(p, "nvidia")
        if not _os.path.isdir(nvidia_root):
            continue
        for pkg in _os.listdir(nvidia_root):
            for sub in ("bin", "lib"):
                d = _os.path.join(nvidia_root, pkg, sub)
                if _os.path.isdir(d):
                    _os.add_dll_directory(d)
                    dll_dirs.add(d)

    # Prepend nvidia bin dirs to PATH — ctranslate2's native loader
    # uses LoadLibrary which relies on PATH, not LOAD_LIBRARY_SEARCH.
    _path_parts = _os.environ.get("PATH", "").split(_os.pathsep)
    _os.environ["PATH"] = _os.pathsep.join(sorted(dll_dirs) + _path_parts)

    # Safe to import now that DLL search paths are configured
    from src.transcribe_worker import transcribe_worker

    logger.info(f"Worker-{worker_id} started, loading model...")

    # Pre-load the model so the parent can reduce concurrency before tasks run
    # if this worker cannot fit in available VRAM.
    try:
        from src.transcribe_worker import _load_model
        _load_model(str(model_path), compute_type)
        startup_queue.put(("ready", worker_id))
    except Exception as exc:
        startup_queue.put(("failed", worker_id, f"{type(exc).__name__}: {exc}"))
        logger.error(f"Worker-{worker_id} failed during startup: {exc}")
        return

    while True:
        # Wait for GPU slot
        semaphore.acquire()
        logger.debug(f"Worker-{worker_id} acquired semaphore")

        try:
            item = task_queue.get()
            if item == _SHUTDOWN:
                logger.info(f"Worker-{worker_id} received shutdown")
                break

            audio_path, video_path = item
            logger.info(f"Worker-{worker_id} processing: {video_path.name}")

            try:
                segments, detected_language = transcribe_worker(
                    model_path=str(model_path),
                    audio_path=audio_path,
                    language=language,
                    beam_size=beam_size,
                    vad_filter=vad_filter,
                    compute_type=compute_type,
                    progress_queue=progress_queue,
                )
                result_queue.put((audio_path, (segments, detected_language)))
            except Exception as exc:
                logger.error(f"Worker-{worker_id} error on {audio_path.name}: {exc}")
                result_queue.put((audio_path, str(exc)))

        except Exception as exc:
            logger.error(f"Worker-{worker_id} queue error: {exc}")
        finally:
            semaphore.release()

    logger.info(f"Worker-{worker_id} exiting")


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class GpuScheduler:
    """
    GPU-memory-aware task scheduler.

    Spawns N worker processes, feeds them audio paths via a task queue,
    and collects results via a result queue.

    Semaphore ensures at most max_workers GPUs are active simultaneously.
    """

    def __init__(self, config: PipelineConfig):
        self._cfg = config
        self._task_queue: mp.Queue | None = None
        self._result_queue: mp.Queue | None = None
        self._startup_queue: mp.Queue | None = None
        self._progress_queue: mp.Queue | None = None
        self._semaphore: mp.Semaphore | None = None
        self._workers: list[mp.Process] = []
        self._ctx = mp.get_context("spawn")
        self._abort_workers = False
        # audio_path -> 0..1 decode progress, fed by the worker processes.
        self._chunk_progress: dict[str, float] = {}
        self._scheduled_tasks: list[tuple[Path, Path]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        tasks: list[tuple[Path, Path]],  # [(audio_path, video_path), ...]
        *,
        progress_callback: callable = None,  # (fraction: float) -> None
        status_callback: callable = None,   # (stage: str) -> None
    ) -> dict[Path, tuple[list[Segment], str]]:
        """
        Run ASR transcription on all queued audio files.

        Args:
            tasks: List of (audio_path, video_path) tuples to process.
            progress_callback: Optional callback receiving a 0..1 fraction of
                the whole batch, updated continuously while chunks decode.
            status_callback: Optional callback receiving a coarse stage name so
                the UI can explain a pause (model loading, for instance).

        Returns:
            Dict mapping audio_path → (segments, detected language).
            Failed tasks are excluded from the dict (errors are logged).
        """
        if not tasks:
            logger.info("No tasks to process")
            if progress_callback:
                progress_callback(1.0)
            return {}

        self._abort_workers = False
        self._chunk_progress = {}
        self._setup_queues()
        try:
            self._start_workers(task_count=len(tasks), status_callback=status_callback)
            self._enqueue_tasks(tasks)
            return self._collect_results(
                expected=len(tasks),
                progress_callback=progress_callback,
                status_callback=status_callback,
            )
        finally:
            self._shutdown()

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _setup_queues(self) -> None:
        self._task_queue = self._ctx.Queue()
        self._result_queue = self._ctx.Queue()
        self._startup_queue = self._ctx.Queue()
        self._progress_queue = self._ctx.Queue()
        self._semaphore = self._ctx.Semaphore(self._cfg.max_workers)

    def _start_workers(self, task_count: int, *, status_callback=None) -> None:
        # max_workers is an upper bound. Startup proceeds one worker at a time
        # until the first model load fails, leaving the previous workers alive.
        num_workers = max(1, min(self._cfg.max_workers, task_count))
        logger.info("Trying up to %d GPU worker(s) via startup probing", num_workers)
        if status_callback:
            status_callback("loading_model")
        previous_worker_vram_gb: float | None = None
        for i in range(num_workers):
            before_free_vram_gb = _free_vram_gb()
            if not _can_start_worker(before_free_vram_gb, previous_worker_vram_gb):
                logger.warning(
                    "Stopping GPU worker probing before Worker-%d: %.2f GiB VRAM "
                    "free is below the startup safety margin",
                    i,
                    before_free_vram_gb,
                )
                break

            p = self._ctx.Process(
                target=_worker_process,
                    args=(
                        str(self._cfg.model_path),
                        self._cfg.language,
                        self._cfg.beam_size,
                        self._cfg.vad_filter,
                        self._cfg.compute_type,
                        self._task_queue,
                        self._result_queue,
                        self._startup_queue,
                        self._progress_queue,
                        self._semaphore,
                        i,
                    ),
                    name=f"asr-worker-{i}",
                    daemon=True,
                )
            try:
                p.start()
            except Exception as exc:
                logger.error("Failed to spawn Worker-%d: %s", i, exc)
                break
            self._workers.append(p)
            deadline = time.monotonic() + _WORKER_STARTUP_TIMEOUT
            while True:
                try:
                    startup = self._startup_queue.get(timeout=1)
                    break
                except queue.Empty:
                    if not p.is_alive():
                        startup = ("failed", i, "process exited during startup")
                        break
                    free_vram_gb = _free_vram_gb()
                    if not _can_start_worker(free_vram_gb, None):
                        startup = (
                            "failed",
                            i,
                            "startup stopped: free VRAM dropped below safety threshold",
                        )
                        break
                    if time.monotonic() >= deadline:
                        startup = ("failed", i, "startup timed out")
                        break

            if startup[0] == "ready":
                after_free_vram_gb = _free_vram_gb()
                if (
                    after_free_vram_gb is not None
                    and after_free_vram_gb < _STARTUP_MIN_FREE_VRAM_GB
                ):
                    startup = (
                        "failed",
                        i,
                        "startup stopped: worker left insufficient free VRAM",
                    )
                elif (
                    before_free_vram_gb is not None
                    and after_free_vram_gb is not None
                    and before_free_vram_gb > after_free_vram_gb
                ):
                    previous_worker_vram_gb = (
                        before_free_vram_gb - after_free_vram_gb
                    )

            if startup[0] == "ready":
                logger.info("Worker-%d ready (pid=%s)", i, p.pid)
                continue

            reason = startup[2] if len(startup) > 2 else "unknown startup error"
            logger.warning(
                "Worker-%d cannot load the model: %s; using %d worker(s) for this run",
                i,
                reason,
                len(self._workers) - 1,
            )
            if p.is_alive():
                p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                logger.error("Force killing stalled Worker-%d", i)
                p.kill()
                p.join(timeout=3)
            self._workers.pop()
            break

        if not self._workers:
            raise RuntimeError(
                "No GPU worker could load the Whisper model. "
                "Check the model path, CUDA installation, and available VRAM."
            )

    def _enqueue_tasks(self, tasks: list[tuple[Path, Path]]) -> None:
        """Push tasks + sentinel values (one per worker) into the queue."""
        self._scheduled_tasks = list(tasks)
        for task in tasks:
            self._task_queue.put(task)

        # One sentinel per worker so each exits cleanly
        for _ in self._workers:
            self._task_queue.put(_SHUTDOWN)

    def _collect_results(
        self,
        expected: int,
        *,
        progress_callback: callable = None,
        status_callback=None,
    ) -> dict[Path, tuple[list[Segment], str]]:
        """Drain result queue until expected count reached."""
        results: dict[Path, tuple[list[Segment], str]] = {}
        received = 0
        errors = 0
        reported = -1.0
        if status_callback:
            status_callback("transcribing")
        # Every task counts as one unit regardless of its length, so the
        # fraction is the mean of the per-chunk decode progress. Counting
        # finished chunks instead left a single-chunk video pinned at 0% for
        # its entire run.
        task_keys = [str(audio_path) for audio_path, _ in self._scheduled_tasks]

        def report_progress() -> None:
            nonlocal reported
            if not progress_callback:
                return
            fraction = self._aggregate_progress(task_keys)
            # Throttle to whole percent: the queue drains fast enough that
            # every report would otherwise be a wasted repaint.
            if fraction - reported < 0.005 and fraction < 1.0:
                return
            reported = fraction
            progress_callback(fraction)

        self._drain_progress_queue()
        report_progress()

        while received < expected:
            free_vram_gb = _free_vram_gb()
            if (
                free_vram_gb is not None
                and free_vram_gb < _MIN_FREE_VRAM_GB
            ):
                self._abort_workers = True
                raise RuntimeError(
                    "GPU memory safety stop: free VRAM dropped below "
                    f"{_MIN_FREE_VRAM_GB:.1f} GiB"
                )
            try:
                item = self._result_queue.get(timeout=_RESULT_POLL_TIMEOUT)
                audio_path, payload = item

                if isinstance(payload, str):
                    # Error string
                    logger.error(f"FAILED: {audio_path.name} — {payload}")
                    errors += 1
                else:
                    results[audio_path] = payload

                received += 1
                # A finished chunk counts as fully decoded even if its worker
                # never managed to stream intermediate updates.
                self._chunk_progress[str(audio_path)] = 1.0
            except queue.Empty:
                # A worker killed by CUDA/OOM can leave tasks without a result.
                # Fail immediately instead of reporting endless progress while
                # another process remains alive but cannot make progress.
                abnormal = [
                    w for w in self._workers
                    if not w.is_alive() and w.exitcode not in (None, 0)
                ]
                if abnormal:
                    names = ", ".join(w.name for w in abnormal)
                    self._abort_workers = True
                    raise RuntimeError(
                        f"GPU worker exited unexpectedly ({names}); "
                        "transcription was stopped to protect system resources"
                    )

                free_vram_gb = _free_vram_gb()
                if (
                    free_vram_gb is not None
                    and free_vram_gb < _MIN_FREE_VRAM_GB
                ):
                    self._abort_workers = True
                    raise RuntimeError(
                        "GPU memory safety stop: free VRAM dropped below "
                        f"{_MIN_FREE_VRAM_GB:.1f} GiB"
                    )

                alive = sum(1 for w in self._workers if w.is_alive())
                if alive == 0:
                    self._abort_workers = True
                    raise RuntimeError(
                        "All GPU workers exited before transcription completed "
                        f"({received}/{expected} results received)"
                    )
                logger.info(
                    f"Transcribing... ({received}/{expected} done, {alive} worker(s) active)"
                )

            # Drain outside both branches so intra-chunk updates land on the
            # poll iterations where no result arrived.
            self._drain_progress_queue()
            report_progress()

        report_progress()
        logger.info(
            f"Transcription complete: {len(results)} success, {errors} failed "
            f"(out of {expected})"
        )
        return results

    def _drain_progress_queue(self) -> None:
        """Absorb every pending (audio_path, fraction) report from the workers."""
        if self._progress_queue is None:
            return
        while True:
            try:
                audio_path, fraction = self._progress_queue.get_nowait()
            except queue.Empty:
                return
            except (OSError, ValueError):  # pragma: no cover - closed queue
                return
            try:
                value = max(0.0, min(1.0, float(fraction)))
            except (TypeError, ValueError):
                continue
            key = str(audio_path)
            # Never let a late report walk a chunk backwards.
            if value > self._chunk_progress.get(key, 0.0):
                self._chunk_progress[key] = value

    def _aggregate_progress(self, task_keys: list[str]) -> float:
        if not task_keys:
            return 1.0
        total = sum(self._chunk_progress.get(key, 0.0) for key in task_keys)
        return min(1.0, total / len(task_keys))

    def _shutdown(self) -> None:
        """Join all worker processes."""
        for w in self._workers:
            if self._abort_workers and w.is_alive():
                w.terminate()
            w.join(timeout=2)
            if w.is_alive():
                logger.warning(f"Force terminating {w.name}")
                w.terminate()
                w.join(timeout=3)
            if w.is_alive():
                logger.error(f"Force killing {w.name}")
                w.kill()
                w.join(timeout=3)
        self._workers.clear()
        logger.info("All workers shut down")
