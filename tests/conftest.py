"""Shared pytest setup.

Several tests drive ffmpeg/ffprobe for real. On a machine without them those
tests cannot say anything about the code under test, so they skip instead of
failing — otherwise a missing optional binary looks like a regression.
"""
from __future__ import annotations

import shutil

import pytest


def _has_tool(tool: str) -> bool:
    return shutil.which(tool) is not None


HAS_FFMPEG = _has_tool("ffmpeg")
HAS_FFPROBE = _has_tool("ffprobe")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "needs_ffmpeg: skipped when ffmpeg is not on PATH"
    )
    config.addinivalue_line(
        "markers", "needs_ffprobe: skipped when ffprobe is not on PATH"
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    if "needs_ffmpeg" in item.keywords and not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")
    if "needs_ffprobe" in item.keywords and not HAS_FFPROBE:
        pytest.skip("ffprobe not installed")
