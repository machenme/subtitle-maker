"""
Utility functions: media file scanning, file integrity checks, SRT validation.
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path


def scan_video_files(
    directory: Path,
    extensions: list[str],
    recursive: bool = True,
) -> list[Path]:
    """
    Recursively scan a directory for media files matching the given extensions.

    Returns a sorted list of absolute paths.
    """
    ext_set = {e.lower().lstrip(".") for e in extensions}
    if not ext_set:
        return []

    # One walk instead of one glob per extension: the previous version walked
    # the whole tree once per extension (14x for the default set).
    results: list[Path] = []
    if recursive:
        for dir_path, dir_names, file_names in os.walk(directory):
            # Do not descend into symlinked dirs — they can form cycles.
            dir_names[:] = [d for d in dir_names if not (Path(dir_path) / d).is_symlink()]
            for file_name in file_names:
                if file_name.rpartition(".")[2].lower() in ext_set:
                    results.append(Path(dir_path) / file_name)
    else:
        pattern = "[!.]*"
        for ext in ext_set:
            results.extend(directory.glob(f"{pattern}.{ext}"))

    # Deduplicate and sort
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in sorted(results, key=lambda x: x.name):
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def is_file_valid(path: Path, min_bytes: int = 100) -> bool:
    """Check if a file exists, is > min_bytes, and is readable."""
    try:
        return path.exists() and path.is_file() and path.stat().st_size > min_bytes
    except OSError:
        return False


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write text through a same-directory temporary file and atomically replace it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def atomic_copy_file(source: Path, destination: Path) -> None:
    """Copy a file without exposing a partially written destination."""
    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.", suffix=".tmp", dir=destination_path.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle, source_path.open("rb") as input_file:
            shutil.copyfileobj(input_file, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(destination_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def is_srt_valid(srt_path: Path) -> bool:
    """
    Basic SRT structural validation.
    Checks that the file is non-empty and has at least one properly formed entry.
    """
    if not is_file_valid(srt_path, min_bytes=50):
        return False

    try:
        content = srt_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False

    # SRT entry pattern: number, timestamp line, at least one text line, blank line
    # Broad check: must have at least one timestamp line
    timestamp_pattern = re.compile(
        r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}"
    )
    return bool(timestamp_pattern.search(content))


def output_exists_and_valid(video_name: str, output_dir: Path) -> bool:
    """
    Check if all expected output files for a video exist and are valid.
    Returns True only if SRT (required) exists and passes validation.
    """
    srt_path = output_dir / f"{video_name}.srt"
    return is_srt_valid(srt_path)


def format_timestamp(seconds: float, fmt: str = "srt") -> str:
    """
    Convert seconds to timestamp string.

    fmt='srt':  HH:MM:SS,mmm
    fmt='md':   [HH:MM:SS]
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)

    if fmt == "srt":
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    else:
        return f"[{h:02d}:{m:02d}:{s:02d}]"
