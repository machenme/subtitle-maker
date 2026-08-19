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
TaskResult = tuple[Path, list[Segment]]  # (video_path, segments)
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
    semaphore: mp.Semaphore,
    worker_id: int,
) -> None:
    """
    Child process entry point.

    - Loads WhisperModel before announcing startup readiness.
    - Acquires semaphore slot before processing (controls GPU concurrency).
    - Reuses the loaded model while looping on task_queue.
    - Sends (audio_path, segments) or (audio_path, error_string) to result_queue.
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
                segments = transcribe_worker(
                    model_path=str(model_path),
                    audio_path=audio_path,
                    language=language,
                    beam_size=beam_size,
                    vad_filter=vad_filter,
                    compute_type=compute_type,
                )
                result_queue.put((audio_path, segments))
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
        self._semaphore: mp.Semaphore | None = None
        self._workers: list[mp.Process] = []
        self._ctx = mp.get_context("spawn")
        self._abort_workers = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        tasks: list[tuple[Path, Path]],  # [(audio_path, video_path), ...]
        *,
        progress_callback: callable = None,  # (received: int, total: int) -> None
    ) -> dict[Path, list[Segment]]:
        """
        Run ASR transcription on all queued audio files.

        Args:
            tasks: List of (audio_path, video_path) tuples to process.
            progress_callback: Optional callback for progress updates.

        Returns:
            Dict mapping audio_path → list of Segments.
            Failed tasks are excluded from the dict (errors are logged).
        """
        if not tasks:
            logger.info("No tasks to process")
            return {}

        self._abort_workers = False
        self._setup_queues()
        try:
            self._start_workers(task_count=len(tasks))
            self._enqueue_tasks(tasks)
            return self._collect_results(
                expected=len(tasks), progress_callback=progress_callback
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
        self._semaphore = self._ctx.Semaphore(self._cfg.max_workers)

    def _start_workers(self, task_count: int) -> None:
        # max_workers is an upper bound. Startup proceeds one worker at a time
        # until the first model load fails, leaving the previous workers alive.
        num_workers = max(1, min(self._cfg.max_workers, task_count))
        logger.info("Trying up to %d GPU worker(s) via startup probing", num_workers)
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
        for task in tasks:
            self._task_queue.put(task)

        # One sentinel per worker so each exits cleanly
        for _ in self._workers:
            self._task_queue.put(_SHUTDOWN)

    def _collect_results(
        self, expected: int, *, progress_callback: callable = None
    ) -> dict[Path, list[Segment]]:
        """Drain result queue until expected count reached."""
        results: dict[Path, list[Segment]] = {}
        received = 0
        errors = 0

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
                if progress_callback:
                    progress_callback(received, expected)
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
                if progress_callback:
                    progress_callback(received, expected)

        logger.info(
            f"Transcription complete: {len(results)} success, {errors} failed "
            f"(out of {expected})"
        )
        return results

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
