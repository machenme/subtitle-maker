"""
Configuration loader and validator.
Loads YAML config file, merges with CLI arguments, validates required fields.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = "./models/faster-whisper-large-v3-turbo-ct2"
VALID_MODEL_SIZES = {"large-v3-turbo", "large-v3", "medium"}


# ---------------------------------------------------------------------------
# GPU-aware worker count
# ---------------------------------------------------------------------------

def detect_optimal_workers(
    gpu_index: int = 0,
    model_memory_gb: float = 2.8,
    model_path: str | None = None,
) -> int:
    """
    Query GPU **free** VRAM and compute recommended worker count.

    If *model_path* is provided, probes actual model memory cost by loading
    the model once and measuring VRAM delta.  Otherwise uses *model_memory_gb*
    estimate (~2.8 GB for large-v3-turbo FP16).

    Returns 1 if GPU detection fails.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()

        free_gb = mem.free / (1024 ** 3)

        # Optional: probe / cache actual model cost
        cost_gb = model_memory_gb
        if model_path:
            from pathlib import Path as _Path
            p = _Path(model_path)
            if p.exists():
                cached = _read_gpu_cache()
                cache_key = f"{p.resolve()}:float16"
                if cache_key in cached:
                    cost_gb = cached[cache_key]
                else:
                    probed = _probe_model_vram_cost(str(model_path), gpu_index)
                    if probed:
                        cost_gb = probed
                        cached[cache_key] = round(probed, 2)
                        _write_gpu_cache(cached)

        # 1.2x buffer to avoid OOM from transient allocations
        workers = int(free_gb / (cost_gb * 1.2))
        return max(1, min(8, workers))
    except Exception:
        return 1


def _probe_model_vram_cost(model_path: str, gpu_index: int = 0) -> float | None:
    """Load the model once and return VRAM delta in GB (or None on failure)."""
    try:
        import logging as _logging
        import threading as _threading
        import multiprocessing as _mp
        import queue as _queue

        _log = _logging.getLogger(__name__)

        ctx = _mp.get_context("spawn")
        result_q: _mp.Queue = ctx.Queue()
        stop = _threading.Event()

        def _probe_worker(q, model_path, compute_type):
            import os as _os, sys as _sys
            # CUDA DLL setup (same as gpu_scheduler)
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
            _path_parts = _os.environ.get("PATH", "").split(_os.pathsep)
            _os.environ["PATH"] = _os.pathsep.join(sorted(dll_dirs) + _path_parts)

            import pynvml as _pn
            _pn.nvmlInit()
            h = _pn.nvmlDeviceGetHandleByIndex(0)
            before = _pn.nvmlDeviceGetMemoryInfo(h).used
            _pn.nvmlShutdown()

            try:
                from faster_whisper import WhisperModel
                WhisperModel(model_path, device="cuda", compute_type=compute_type)
            except Exception as exc:
                q.put(None)
                return

            _pn.nvmlInit()
            h = _pn.nvmlDeviceGetHandleByIndex(0)
            after = _pn.nvmlDeviceGetMemoryInfo(h).used
            _pn.nvmlShutdown()
            q.put((after - before) / (1024 ** 3))

        p = ctx.Process(target=_probe_worker, args=(result_q, model_path, "float16"))
        p.start()
        p.join(timeout=120)
        if p.is_alive():
            p.terminate()
            p.join()
            return None
        try:
            return result_q.get(timeout=5)
        except _queue.Empty:
            return None
    except Exception:
        return None


def _read_gpu_cache() -> dict[str, float]:
    """Read ``.gpu_cache.json`` from cwd, return ``{model_path: vram_gb}``."""
    import json as _json
    try:
        cache = _json.loads(Path(".gpu_cache.json").read_text(encoding="utf-8"))
        return cache if isinstance(cache, dict) else {}
    except Exception:
        return {}


def _write_gpu_cache(cache: dict[str, float]) -> None:
    """Write ``.gpu_cache.json``."""
    import json as _json
    try:
        Path(".gpu_cache.json").write_text(
            _json.dumps(cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


VALID_COMPUTE_TYPES = {"float16", "int8_float16", "int8"}
DEFAULT_VIDEO_EXTENSIONS = ["mp4", "mkv", "mov", "avi", "flv", "wmv"]


@dataclass
class PipelineConfig:
    """Immutable-ish config object built from YAML + CLI overrides."""

    input_dir: Path
    output_dir: Path
    temp_dir: Path | None = None
    model_path: Path = Path(DEFAULT_MODEL_PATH)
    model_size: str = "large-v3-turbo"
    language: str = "auto"
    beam_size: int = 5
    vad_filter: bool = True
    compute_type: str = "float16"
    max_workers: int = 4
    chunk_duration: int = 0  # seconds; 0 = auto (split evenly by worker count)
    video_extensions: list[str] = field(default_factory=lambda: DEFAULT_VIDEO_EXTENSIONS.copy())
    output_formats: list[str] = field(default_factory=lambda: ["srt"])
    translate_to: str = ""  # "" = skip translation; non-empty = ISO 639-1 target
    swap_subtitles: bool = True  # rename translated → .srt, original → .{lang}.srt
    cleanup_temp: bool = True
    verbose: bool = False

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> dict[str, Any]:
        """Load raw config dict from a YAML file."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return raw

    @classmethod
    def build(cls, cli_args: dict[str, Any] | None = None) -> "PipelineConfig":
        """
        Primary entry point.

        1. Load config.yaml (or CLI --config path).
        2. Override with any CLI-supplied values.
        3. Validate and return a PipelineConfig instance.
        """
        cli = cli_args or {}

        # --- locate config file ---
        config_path = cli.get("config", "./config.yaml")
        raw = cls.from_yaml(config_path)

        # --- resolve paths relative to config file location ---
        config_dir = Path(config_path).parent.resolve()

        def _resolve_path(key: str) -> Path | None:
            val = cli.get(key) or raw.get(key)
            if val is None or val == "":
                return None
            p = Path(val)
            if p.is_absolute():
                return p.resolve()
            return (config_dir / p).resolve()

        # --- merge: CLI > YAML > defaults ---
        input_dir = _resolve_path("input_dir") or Path.cwd()
        output_dir = _resolve_path("output_dir") or (config_dir / "output")
        temp_dir = _resolve_path("temp_dir")  # may be None

        model_path = cli.get("model_path") or raw.get("model_path") or DEFAULT_MODEL_PATH
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = (config_dir / model_path).resolve()

        model_size = cli.get("model") or raw.get("model_size") or "large-v3-turbo"
        language = cli.get("language") or raw.get("language") or "auto"
        beam_size = int(cli.get("beam_size") or raw.get("beam_size") or 5)
        vad_filter = cli.get("vad_filter", raw.get("vad_filter", True))
        compute_type = cli.get("compute_type") or raw.get("compute_type") or "float16"
        max_workers = int(cli.get("workers") or raw.get("max_workers") or detect_optimal_workers())
        chunk_duration = int(cli.get("chunk_duration") or raw.get("chunk_duration") or 0)
        video_extensions = cli.get("video_extensions") or raw.get("video_extensions") or DEFAULT_VIDEO_EXTENSIONS
        output_formats = cli.get("output_formats") or raw.get("output_formats") or ["srt"]
        translate_to = cli.get("translate_to") or raw.get("translate_to") or ""
        swap_subtitles = cli.get("swap_subtitles", raw.get("swap_subtitles", True))
        cleanup_temp = cli.get("cleanup_temp", raw.get("cleanup_temp", True))
        verbose = cli.get("verbose", raw.get("verbose", False))

        cfg = cls(
            input_dir=Path(input_dir) if not isinstance(input_dir, Path) else input_dir,
            output_dir=Path(output_dir) if not isinstance(output_dir, Path) else output_dir,
            temp_dir=Path(temp_dir) if temp_dir and not isinstance(temp_dir, Path) else temp_dir,
            model_path=Path(model_path) if not isinstance(model_path, Path) else model_path,
            model_size=model_size,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            compute_type=compute_type,
            max_workers=max_workers,
            chunk_duration=chunk_duration,
            video_extensions=list(video_extensions),
            output_formats=list(output_formats),
            translate_to=translate_to,
            swap_subtitles=swap_subtitles,
            cleanup_temp=cleanup_temp,
            verbose=verbose,
        )
        cfg.validate()
        return cfg

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise ValueError on invalid configuration."""
        errors: list[str] = []

        if not self.input_dir.exists():
            errors.append(f"Input directory does not exist: {self.input_dir}")
        if self.beam_size < 1 or self.beam_size > 10:
            errors.append(f"beam_size must be 1-10, got {self.beam_size}")
        if self.max_workers < 1 or self.max_workers > 8:
            errors.append(f"max_workers must be 1-8, got {self.max_workers}")
        if self.chunk_duration < 0:
            errors.append(f"chunk_duration must be >= 0, got {self.chunk_duration}")  # 0 = auto, >0 = manual seconds
        if self.model_size not in VALID_MODEL_SIZES:
            errors.append(f"model_size must be one of {VALID_MODEL_SIZES}, got {self.model_size}")
        if self.compute_type not in VALID_COMPUTE_TYPES:
            errors.append(f"compute_type must be one of {VALID_COMPUTE_TYPES}, got {self.compute_type}")
        if not self.model_path.exists():
            errors.append(f"Model path does not exist: {self.model_path}")
        valid_formats = {"srt", "txt", "md"}
        unknown = set(self.output_formats) - valid_formats
        if unknown:
            errors.append(f"Invalid output_formats: {unknown}. Valid: {valid_formats}")

        if errors:
            raise ValueError("Configuration errors:\n  - " + "\n  - ".join(errors))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def effective_temp_dir(self) -> Path:
        """Return temp_dir or OS default temp directory."""
        if self.temp_dir:
            return self.temp_dir
        import tempfile
        return Path(tempfile.gettempdir()) / "asr-pipeline-temp"

    def __repr__(self) -> str:
        lines = ["PipelineConfig:"]
        for f in fields(self):
            lines.append(f"  {f.name}: {getattr(self, f.name)}")
        return "\n".join(lines)
