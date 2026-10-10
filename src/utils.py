"""
Utility functions: media file scanning, file integrity checks, SRT validation.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
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


# ---------------------------------------------------------------------------
# External tool resolution (ffmpeg / ffprobe)
# ---------------------------------------------------------------------------

# Resolved once per process: a fallback probe costs one subprocess spawn.
_TOOL_CACHE: dict[str, str] = {}

# Tools verified by actually running them — a dangling WinGet symlink or a
# half-installed package otherwise looks like a working binary.
_VERIFIABLE_TOOLS = frozenset({"ffmpeg", "ffprobe"})


def _persisted_path_dirs() -> list[str]:
    """PATH entries stored in the Windows registry.

    A process started before ffmpeg was installed (or before the installer
    broadcast the PATH change) inherits a PATH without it and never sees the
    new entry, even though ``where.exe ffmpeg`` works in every new shell.
    Reading the persisted value lets a long-running app recover anyway.
    """
    if os.name != "nt":
        return []
    try:
        import winreg
    except ImportError:  # pragma: no cover - non-Windows
        return []

    collected: list[str] = []
    sources = (
        (winreg.HKEY_CURRENT_USER, "Environment"),
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ),
    )
    for root, subkey in sources:
        try:
            with winreg.OpenKey(root, subkey) as key:
                raw, _ = winreg.QueryValueEx(key, "Path")
        except OSError:
            continue
        collected.extend(
            os.path.expandvars(part) for part in str(raw).split(os.pathsep)
        )
    return [part for part in collected if part]


def _known_tool_dirs() -> list[Path]:
    """Install locations that are on PATH in some setups but not in others."""
    env = os.environ
    localappdata = env.get("LOCALAPPDATA", "")
    profile = env.get("USERPROFILE", "")
    dirs: list[Path] = []

    if localappdata:
        winget = Path(localappdata) / "Microsoft" / "WinGet"
        dirs.append(winget / "Links")  # winget shim directory
        packages = winget / "Packages"
        if packages.is_dir():
            for package in sorted(packages.glob("*[Ff][Ff]mpeg*")):
                dirs.append(package / "bin")
                dirs.extend(sorted(package.glob("*/bin"))[:4])
    if profile:
        dirs.append(Path(profile) / "scoop" / "shims")
        dirs.append(Path(profile) / "scoop" / "apps" / "ffmpeg" / "current" / "bin")

    dirs.append(Path(r"C:\ProgramData\chocolatey\bin"))
    dirs.append(Path(r"C:\ffmpeg\bin"))
    dirs.append(Path(__file__).resolve().parent.parent / "bin")

    # PyInstaller onefile: binaries bundled next to the extracted payload.
    bundle_dir = getattr(sys, "_MEIPASS", None)
    if bundle_dir:
        dirs.append(Path(bundle_dir) / "bin")
        dirs.append(Path(bundle_dir))
    return dirs


def _join_tool(directory: Path, name: str) -> Path:
    """Directory + tool name. ``which`` applies PATHEXT, so .exe/.bat/.cmd
    and a suffix-less Unix binary are all handled."""
    found = shutil.which(name, path=str(directory))
    return Path(found) if found else directory / name


def _expand_tool_path(value: str, name: str) -> Path:
    """Normalise a user-supplied path that may point at a file or a folder."""
    path = Path(os.path.expandvars(value).strip().strip('"')).expanduser()
    if path.is_dir():
        return _join_tool(path, name)
    if os.name == "nt" and not path.suffix:
        exe = path.with_suffix(".exe")
        if exe.is_file():
            return exe
    return path


def _is_runnable(path: Path, verify: bool) -> bool:
    try:
        if not path.is_file():  # also False for a dangling symlink
            return False
    except OSError:
        return False
    if not verify:
        return True
    try:
        probe = subprocess.run(
            [str(path), "-version"], capture_output=True, timeout=15
        )
        return probe.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def find_executable(name: str, configured: str | None = None) -> str:
    """Resolve an external tool to a runnable absolute path.

    Order: explicit argument → ``$<NAME>_PATH`` env var → current ``PATH`` →
    persisted (registry) PATH → well-known install locations.

    Raises:
        FileNotFoundError: with every location searched, so the message can be
            shown to the user instead of a bare "not found".
    """
    cache_key = f"{name}|{configured or ''}"
    cached = _TOOL_CACHE.get(cache_key)
    if cached:
        return cached

    search_dirs = [Path(entry) for entry in _persisted_path_dirs()]
    search_dirs.extend(_known_tool_dirs())

    # (candidate, must_verify): a PATH hit is already the platform's own
    # lookup, so it is trusted; everything else is a guess and gets probed.
    candidates: list[tuple[Path, bool]] = []
    explicit = configured or os.environ.get(f"{name.upper()}_PATH") or ""
    if explicit:
        candidates.append((_expand_tool_path(explicit, name), True))

    on_path = shutil.which(name)
    if on_path:
        candidates.append((Path(on_path), False))
    candidates.extend(
        (_join_tool(directory, name), True) for directory in search_dirs
    )

    supports_probe = name in _VERIFIABLE_TOOLS
    for candidate, verify in candidates:
        if _is_runnable(candidate, verify and supports_probe):
            resolved = str(candidate)
            _TOOL_CACHE[cache_key] = resolved
            return resolved

    raise FileNotFoundError(_tool_not_found_message(name, search_dirs))


def _tool_not_found_message(name: str, search_dirs: list[Path]) -> str:
    tried = ", ".join(str(d) for d in dict.fromkeys(search_dirs))
    if len(tried) > 400:
        tried = tried[:400] + " …"
    return (
        f"{name} not found. Install it (winget install Gyan.FFmpeg), restart the "
        f"app so it picks up a newly added PATH entry, or set {name}_path in "
        f"config.yaml to the .exe. Searched PATH plus: {tried}"
    )


def reset_tool_cache() -> None:
    """Drop cached resolutions. Tests install/move tools between cases."""
    _TOOL_CACHE.clear()


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
