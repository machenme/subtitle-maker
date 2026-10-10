"""ffmpeg/ffprobe resolution must survive a process with a stale PATH.

The real-world failure this guards: ffmpeg is installed (winget → WinGet
Links) and `where ffmpeg` works in every new shell, but a process started
before the install inherited a PATH without that directory, so a bare
`subprocess.run(["ffmpeg", ...])` raised FileNotFoundError.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.utils import find_executable, reset_tool_cache


@pytest.fixture(autouse=True)
def _clean_cache():
    reset_tool_cache()
    yield
    reset_tool_cache()


def _fake_tool(directory: Path, name: str) -> Path:
    """An executable that answers `-version` with exit code 0."""
    directory.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        path = directory / f"{name}.bat"
        path.write_text("@echo off\r\nexit /b 0\r\n", encoding="utf-8")
    else:
        path = directory / name
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
    return path


def test_finds_tool_on_path(monkeypatch, tmp_path):
    tool = _fake_tool(tmp_path, "ffmpeg")
    monkeypatch.setenv("PATH", str(tmp_path))

    assert Path(find_executable("ffmpeg")).name.lower() == tool.name.lower()


def test_env_var_wins_over_missing_path(monkeypatch, tmp_path):
    tool = _fake_tool(tmp_path, "ffmpeg")
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("FFMPEG_PATH", str(tool))

    assert Path(find_executable("ffmpeg")).name.lower() == tool.name.lower()


def test_configured_directory_is_expanded(monkeypatch, tmp_path):
    _fake_tool(tmp_path, "ffprobe")
    monkeypatch.setenv("PATH", "")

    assert Path(find_executable("ffprobe", str(tmp_path))).parent == tmp_path


def test_known_install_dir_is_searched(monkeypatch, tmp_path):
    """A tool outside PATH, in a directory the resolver knows about."""
    install_dir = tmp_path / "install"
    _fake_tool(install_dir, "ffmpeg")
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(
        "src.utils._known_tool_dirs", lambda: [install_dir]
    )

    assert Path(find_executable("ffmpeg")).name.lower().startswith("ffmpeg")


def test_persisted_path_dirs_are_searched(monkeypatch, tmp_path):
    """The stale-PATH case: entry only in the persisted (registry) PATH."""
    install_dir = tmp_path / "winget-links"
    _fake_tool(install_dir, "ffmpeg")
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(
        "src.utils._persisted_path_dirs", lambda: [str(install_dir)]
    )

    assert Path(find_executable("ffmpeg")).parent == install_dir


def test_missing_tool_error_lists_where_it_looked(monkeypatch):
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr("src.utils._persisted_path_dirs", lambda: [])
    monkeypatch.setattr("src.utils._known_tool_dirs", lambda: [Path("/no/such/dir")])

    with pytest.raises(FileNotFoundError) as excinfo:
        find_executable("ffmpeg")

    message = str(excinfo.value)
    assert "ffmpeg" in message
    assert "config.yaml" in message


def test_broken_symlink_is_skipped(monkeypatch, tmp_path):
    """WinGet Links entries are symlinks; a dangling one must not be used."""
    broken = tmp_path / "ffmpeg.exe"
    if os.name == "nt":
        monkeypatch.setattr("src.utils._known_tool_dirs", lambda: [tmp_path])
        monkeypatch.setenv("PATH", "")
        monkeypatch.setattr("src.utils._persisted_path_dirs", lambda: [])
        with pytest.raises(FileNotFoundError):
            find_executable("ffmpeg")
    else:
        broken.symlink_to(tmp_path / "missing")
        monkeypatch.setattr("src.utils._known_tool_dirs", lambda: [tmp_path])
        with pytest.raises(FileNotFoundError):
            find_executable("ffmpeg")


@pytest.mark.needs_ffmpeg
def test_extractor_uses_resolved_absolute_path(tmp_path):
    """Regression: cmd[0] must be runnable even when PATH lookup is odd."""
    from src.audio_extractor import AudioExtractor

    extractor = AudioExtractor(tmp_path / "temp")
    resolved = Path(extractor.ffmpeg)
    assert resolved.is_absolute()
    subprocess.run([str(resolved), "-version"], capture_output=True, check=True)
