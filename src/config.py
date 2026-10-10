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
# Default GGUF location for translation_provider="llm". Kept in sync with
# src.translator.llm.DEFAULT_MODEL_PATH (duplicated deliberately: importing
# the translator here would create a cycle, since it depends on this module).
DEFAULT_TRANSLATION_MODEL_PATH = "./models/index-translate-9b"
VALID_MODEL_SIZES = {"large-v3-turbo", "large-v3", "medium"}
DEFAULT_MAX_WORKERS = 16
VALID_TRANSLATION_PROVIDERS = {"bing", "gtx", "index_api", "llm"}


def _default_model_path(model_size: str) -> str:
    """Return the conventional local model directory for a model size."""
    return f"./models/faster-whisper-{model_size}-ct2"


_TRUE_STRINGS = {"true", "1", "yes", "on"}
_FALSE_STRINGS = {"false", "0", "no", "off"}


def _as_int(value: Any, default: int, key: str) -> int:
    """Coerce a config value to int, or explain which key is wrong."""
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer, got {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer, got {value!r}") from exc


def _as_bool(value: Any, default: bool, key: str) -> bool:
    """Coerce a config value to bool, or explain which key is wrong.

    YAML already yields real booleans; a quoted ``"false"`` is the trap this
    catches, since ``bool("false")`` is ``True``.
    """
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    raise ValueError(f"{key} must be a boolean, got {value!r}")


# ---------------------------------------------------------------------------
# Legacy worker-count helper
# ---------------------------------------------------------------------------

def detect_optimal_workers(
    gpu_index: int = 0,
    model_memory_gb: float = 2.8,
    model_path: str | None = None,
) -> int:
    """
    Return the maximum startup-probe ceiling for backward compatibility.

    Actual capacity is detected by ``GpuScheduler`` while loading workers.
    """
    return DEFAULT_MAX_WORKERS


VALID_COMPUTE_TYPES = {"float16", "int8_float16", "int8"}
# Common video and audio inputs supported by ffmpeg.
DEFAULT_MEDIA_EXTENSIONS = [
    "mp4", "mkv", "mov", "avi", "flv", "wmv",
    "m4a", "mp3", "wav", "flac", "ogg", "opus", "aac", "wma",
]
# Backward-compatible alias for callers using the old constant name.
DEFAULT_VIDEO_EXTENSIONS = DEFAULT_MEDIA_EXTENSIONS


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
    max_workers: int = DEFAULT_MAX_WORKERS  # startup-probe upper bound
    chunk_duration: int = 0  # seconds; 0 = auto (split evenly by worker count)
    video_extensions: list[str] = field(default_factory=lambda: DEFAULT_VIDEO_EXTENSIONS.copy())
    output_formats: list[str] = field(default_factory=lambda: ["srt"])
    translate_to: str = "zh"  # "" = skip translation; non-empty = ISO 639-1 target
    translation_provider: str = "bing"
    translation_proxy: str = ""
    # GGUF weights for translation_provider="llm". Accepts a directory holding
    # one .gguf or the .gguf file itself; "" = use DEFAULT_TRANSLATION_MODEL_PATH.
    # Lets users point at their own quantized Index-Translate / Hy-MT2 build.
    translation_model_path: str = ""
    swap_subtitles: bool = True  # write translated .srt, bilingual, and original subtitles
    cleanup_temp: bool = True
    verbose: bool = False
    # ffmpeg/ffprobe override: a file path or the folder holding the .exe.
    # "" = auto-detect (PATH → persisted PATH → winget/scoop/choco install dirs).
    ffmpeg_path: str = ""
    ffprobe_path: str = ""

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

        def _cli_or_yaml(cli_key: str, yaml_key: str | None = None, default: Any = None) -> Any:
            if cli_key in cli and cli[cli_key] is not None:
                return cli[cli_key]
            return raw.get(yaml_key or cli_key, default)

        # --- resolve paths relative to config file location ---
        config_dir = Path(config_path).parent.resolve()

        def _resolve_path(key: str) -> Path | None:
            val = _cli_or_yaml(key)
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

        model_size = _cli_or_yaml("model", "model_size", "large-v3-turbo")
        explicit_model_path = _cli_or_yaml("model_path")
        # A model selected from CLI/GUI should select its conventional model
        # directory unless the caller supplied a custom path explicitly.
        if cli.get("model") and not cli.get("model_path"):
            raw_default = raw.get("model_path")
            if raw_default in (None, DEFAULT_MODEL_PATH, _default_model_path("large-v3-turbo")):
                explicit_model_path = _default_model_path(model_size)
        model_path = explicit_model_path or DEFAULT_MODEL_PATH
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = (config_dir / model_path).resolve()

        language = _cli_or_yaml("language", default="auto")
        beam_size = _as_int(_cli_or_yaml("beam_size", default=5), 5, "beam_size")
        vad_filter = _as_bool(
            cli.get("vad_filter", raw.get("vad_filter", True)), True, "vad_filter"
        )
        compute_type = _cli_or_yaml("compute_type", default="float16")
        # A YAML ``max_workers: null`` means "use the default", not "zero".
        max_workers = _as_int(
            _cli_or_yaml("workers", "max_workers", DEFAULT_MAX_WORKERS),
            DEFAULT_MAX_WORKERS,
            "max_workers",
        )
        chunk_duration = _as_int(_cli_or_yaml("chunk_duration", default=0), 0, "chunk_duration")
        video_extensions = _cli_or_yaml("video_extensions", default=DEFAULT_VIDEO_EXTENSIONS)
        output_formats = _cli_or_yaml("output_formats", default=["srt"])
        # A bare `output_formats: srt` would otherwise become ['s', 'r', 't'].
        for key, value in (
            ("video_extensions", video_extensions),
            ("output_formats", output_formats),
        ):
            if isinstance(value, str):
                raise ValueError(f"{key} must be a list, got the string {value!r}")
        if cli.get("translate_to") is not None:
            translate_to = cli["translate_to"]
        else:
            translate_to = raw.get("translate_to", "zh")
        translation_provider = (
            cli.get("translation_provider")
            or raw.get("translation_provider")
            or "bing"
        )
        translation_proxy = (
            cli.get("translation_proxy")
            if cli.get("translation_proxy") is not None
            else raw.get("translation_proxy", "")
        )
        translation_model_path = (
            cli.get("translation_model_path")
            if cli.get("translation_model_path") is not None
            else raw.get("translation_model_path", "")
        )
        swap_subtitles = _as_bool(
            cli.get("swap_subtitles", raw.get("swap_subtitles", True)), True, "swap_subtitles"
        )
        cleanup_temp = _as_bool(
            cli.get("cleanup_temp", raw.get("cleanup_temp", True)), True, "cleanup_temp"
        )
        verbose = _as_bool(cli.get("verbose", raw.get("verbose", False)), False, "verbose")
        ffmpeg_path = (
            cli.get("ffmpeg_path")
            if cli.get("ffmpeg_path") is not None
            else raw.get("ffmpeg_path", "")
        ) or ""
        ffprobe_path = (
            cli.get("ffprobe_path")
            if cli.get("ffprobe_path") is not None
            else raw.get("ffprobe_path", "")
        ) or ""

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
            translation_provider=translation_provider,
            translation_proxy=translation_proxy,
            translation_model_path=translation_model_path,
            swap_subtitles=swap_subtitles,
            cleanup_temp=cleanup_temp,
            verbose=verbose,
            ffmpeg_path=str(ffmpeg_path),
            ffprobe_path=str(ffprobe_path),
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
        elif not self.input_dir.is_dir():
            # A file path scans to zero media files and the run then reports
            # "nothing to do" instead of pointing at the mistyped setting.
            errors.append(f"input_dir must be a directory, not a file: {self.input_dir}")
        if self.beam_size < 1 or self.beam_size > 10:
            errors.append(f"beam_size must be 1-10, got {self.beam_size}")
        if self.max_workers < 1 or self.max_workers > DEFAULT_MAX_WORKERS:
            errors.append(
                f"max_workers must be 1-{DEFAULT_MAX_WORKERS}, got {self.max_workers}"
            )
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
        if self.translate_to and "srt" not in self.output_formats:
            errors.append("translate_to requires 'srt' in output_formats")
        if self.translation_provider not in VALID_TRANSLATION_PROVIDERS:
            errors.append(
                "translation_provider must be one of "
                f"{sorted(VALID_TRANSLATION_PROVIDERS)}, got {self.translation_provider}"
            )
        # A custom translation model is only meaningful for the local backend,
        # and it must actually point at a readable .gguf.
        if self.translation_model_path and self.translation_provider != "llm":
            errors.append(
                "translation_model_path only applies to translation_provider='llm', "
                f"got {self.translation_provider!r}"
            )
        if self.translation_provider == "llm":
            errors.extend(self._validate_translation_model_path())
        if self.translation_provider == "index_api" and not self.translation_proxy:
            errors.append(
                "translation_provider='index_api' needs translation_proxy "
                "(e.g. 127.0.0.1:7897) to reach the public endpoint"
            )
        errors.extend(self._validate_value_types())

        if errors:
            raise ValueError("Configuration errors:\n  - " + "\n  - ".join(errors))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_value_types(self) -> list[str]:
        """Catch list/string/bool settings that only fail deep in the pipeline."""
        errors: list[str] = []

        def _string_list(key: str, value: Any) -> None:
            if isinstance(value, str) or not isinstance(value, (list, tuple, set)):
                errors.append(
                    f"{key} must be a list of extensions, got {type(value).__name__}: {value!r}"
                )
                return
            bad = [item for item in value if not isinstance(item, str) or not item.strip()]
            if bad:
                errors.append(f"{key} must contain non-empty strings, got {bad!r}")

        _string_list("video_extensions", self.video_extensions)
        _string_list("output_formats", self.output_formats)

        for key in ("vad_filter", "swap_subtitles", "cleanup_temp", "verbose"):
            if not isinstance(getattr(self, key), bool):
                errors.append(
                    f"{key} must be a boolean, got {getattr(self, key)!r}"
                )
        for key in ("language", "compute_type", "model_size", "translate_to",
                    "translation_provider", "translation_proxy",
                    "translation_model_path"):
            if not isinstance(getattr(self, key), str):
                errors.append(
                    f"{key} must be a string, got {getattr(self, key)!r}"
                )
        if not str(self.language).strip():
            errors.append("language must be an ISO 639-1 code or 'auto'")
        return errors

    def _validate_translation_model_path(self) -> list[str]:
        """Check the custom translation GGUF path.

        Mirrors :func:`src.translator.llm._resolve_gguf`: a directory must
        contain at least one ``*.gguf``. Reported early (at config time) so a
        typo surfaces in the GUI instead of minutes later during model load.
        """
        raw = (self.translation_model_path or "").strip()
        if not raw:
            # Empty = the default bundled location; only validate when present.
            default = Path(DEFAULT_TRANSLATION_MODEL_PATH)
            if not (default.is_dir() or default.is_file()):
                return [
                    f"Default translation model not found: {default}. "
                    "Set translation_model_path to your own .gguf file."
                ]
            if default.is_dir() and not any(default.glob("*.gguf")):
                return [f"No .gguf file in default translation model dir: {default}"]
            return []

        path = Path(raw)
        if path.is_file():
            if path.suffix.lower() != ".gguf":
                return [f"translation_model_path must be a .gguf file: {path}"]
            return []
        if path.is_dir():
            if not any(path.glob("*.gguf")):
                return [f"No .gguf file in translation_model_path: {path}"]
            return []
        return [
            f"translation_model_path does not exist: {path} "
            "(expected a .gguf file or a directory containing one)"
        ]

    @property
    def effective_temp_dir(self) -> Path:
        """Return an explicit temp_dir or a process-isolated default directory."""
        if self.temp_dir:
            return self.temp_dir
        import tempfile
        return Path(tempfile.gettempdir()) / "subtitle-maker-temp" / str(os.getpid())

    def __repr__(self) -> str:
        lines = ["PipelineConfig:"]
        for f in fields(self):
            lines.append(f"  {f.name}: {getattr(self, f.name)}")
        return "\n".join(lines)
