"""
SRT file parser — reads a standard SRT file into a list of :class:`SrtCue`.
"""
from __future__ import annotations

import re
from pathlib import Path

from src.translator.types import ParseError, SrtCue

# "HH:MM:SS,mmm --> HH:MM:SS,mmm"  (comma or dot separator)
_TIME_LINE = re.compile(
    r"^(\d{2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{1,3})"
)


def parse_srt(path: str | Path) -> list[SrtCue]:
    """
    Parse an SRT file.

    Returns:
        List of :class:`SrtCue` ordered by appearance (1-based index).

    Raises:
        ParseError: The file is empty or structurally invalid.
        FileNotFoundError: *path* does not exist.
    """
    file_path = Path(path)
    # utf-8-sig: a BOM from a Windows editor would otherwise glue itself to the
    # first cue number and fail the int() below.
    raw = file_path.read_text(encoding="utf-8-sig")
    lines = raw.splitlines()

    cues: list[SrtCue] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # skip blank lines between cues
        if not line:
            i += 1
            continue

        # expect cue number
        try:
            index = int(line)
        except ValueError:
            raise ParseError(
                f"Expected cue number at line {i + 1}, got: {line!r}"
            )

        i += 1
        if i >= len(lines):
            raise ParseError(f"Cue {index}: missing timestamp line")

        # timestamp line
        m = _TIME_LINE.match(lines[i].strip())
        if not m:
            raise ParseError(
                f"Cue {index}: invalid timestamp at line {i + 1}: "
                f"{lines[i].strip()!r}"
            )
        start, end = m.group(1), m.group(2)

        i += 1
        # collect text lines until blank or EOF
        text_parts: list[str] = []
        while i < len(lines) and lines[i].strip():
            text_parts.append(lines[i].strip())
            i += 1

        if not text_parts:
            raise ParseError(f"Cue {index}: no text at line {i + 1}")

        cues.append(SrtCue(
            index=index,
            start=start,
            end=end,
            text="\n".join(text_parts),
        ))

    if not cues:
        raise ParseError(f"SRT file is empty or contains no valid cues: {file_path}")

    return cues


def parse_srt_lenient(path: str | Path) -> tuple[list[SrtCue], int]:
    """
    Parse an SRT file, skipping malformed blocks instead of raising.

    Hand-edited and machine-mixed subtitle files regularly contain a cue whose
    index line is missing or whose timestamps are damaged.  The strict parser
    rejects the whole file for one such block, which is the wrong trade-off for
    a text export: every readable cue still has value.

    Args:
        path: SRT file to read.

    Returns:
        ``(cues, skipped)`` — the cues that parsed, and how many blocks were
        dropped.  Raises only when the file cannot be read at all.
    """
    file_path = Path(path)
    raw = file_path.read_text(encoding="utf-8-sig", errors="replace")
    lines = raw.splitlines()

    cues: list[SrtCue] = []
    skipped = 0
    i = 0
    while i < len(lines):
        # A timestamp line is the anchor of a block.  Everything before it
        # (index number, blank lines) is decoration we can drop safely.
        match = _TIME_LINE.match(lines[i].strip())
        if match is None:
            i += 1
            continue

        start, end = match.group(1), match.group(2)
        i += 1
        text_parts: list[str] = []
        while i < len(lines) and lines[i].strip():
            text_parts.append(lines[i].strip())
            i += 1
        if not text_parts:
            skipped += 1
            continue

        cues.append(SrtCue(
            index=len(cues) + 1,
            start=start,
            end=end,
            text="\n".join(text_parts),
        ))

    return cues, skipped
