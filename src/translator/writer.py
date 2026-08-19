"""SRT writers for translated, bilingual, and source subtitle files."""
from __future__ import annotations

from pathlib import Path

from src.translator.types import SrtCue
from src.utils import atomic_write_text


def write_bilingual_srt(cues: list[SrtCue], output_path: str | Path) -> None:
    """
    Write bilingual SRT to *output_path*.

    Format per cue:

        1
        00:00:01,234 --> 00:00:05,678
        原文
        译文

    Un-translated cues repeat the original text in the translation slot.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for cue in cues:
        translation = cue.translation or cue.text  # fallback to original
        lines.append(str(cue.index))
        lines.append(f"{cue.start} --> {cue.end}")
        lines.append(cue.text)
        lines.append(translation)
        lines.append("")  # blank separator

    atomic_write_text(out_path, "\n".join(lines))


def write_translation_srt(cues: list[SrtCue], output_path: str | Path) -> None:
    """Write one translated subtitle line per cue."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for cue in cues:
        lines.append(str(cue.index))
        lines.append(f"{cue.start} --> {cue.end}")
        lines.append(cue.translation or cue.text)
        lines.append("")

    atomic_write_text(out_path, "\n".join(lines))
