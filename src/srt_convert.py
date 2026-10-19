#!/usr/bin/env python
"""
SRT → TXT / Markdown converter.

Reuses the pieces the transcription pipeline already has — the SRT parser in
:mod:`src.translator.parser` and the writers in :class:`src.text_formatter.
TextFormatter` — so an exported file is byte-identical to what the pipeline
would have produced for the same subtitles.

Usage:
    uv run python -m src.srt_convert demo.srt --formats txt md
    uv run python -m src.srt_convert ./subs --output ./output
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

if __name__ == "__main__" and str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from src.text_formatter import (
    Segment,
    TextFormatter,
    get_cps_language,
    join_separator,
    set_cps_language,
)
from src.translator.parser import parse_srt, parse_srt_lenient
from src.utils import atomic_write_text, scan_video_files

logger = logging.getLogger(__name__)

VALID_FORMATS = ("txt", "md")
# Extension → writer name on TextFormatter.
_WRITERS = {"txt": "to_plaintext", "md": "to_markdown"}


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

def _ts_to_seconds(value: str) -> float:
    """``"HH:MM:SS,mmm"`` → seconds. The parser already validated the shape."""
    normalised = value.replace(",", ".")
    hours, minutes, seconds = normalised.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


# ---------------------------------------------------------------------------
# SRT → Segment
# ---------------------------------------------------------------------------

def cues_to_segments(cues) -> list[Segment]:
    """Turn parsed SRT cues into :class:`Segment` objects for the formatter.

    Multi-line cues (bilingual subtitles, wrapped English lines) are folded
    into a single line per cue: the subtitle writer's own wrapping is a
    display concern, and leaving the raw newlines in would leak into TXT and
    Markdown output.
    """
    segments: list[Segment] = []
    for cue in cues:
        text = " ".join(
            line.strip() for line in cue.text.splitlines() if line.strip()
        )
        if not text:
            continue
        segments.append(Segment(_ts_to_seconds(cue.start), _ts_to_seconds(cue.end), text))
    return segments


def load_segments(path: Path) -> tuple[list[Segment], int]:
    """
    Read an SRT file into segments.

    Falls back to the lenient parser when the file is not strictly valid, so a
    single damaged cue does not cost the user every other line.

    Returns:
        ``(segments, skipped)`` — ``skipped`` counts cues the lenient parser
        had to drop (0 when the strict parser handled the file).
    """
    try:
        return cues_to_segments(parse_srt(path)), 0
    except Exception as exc:  # ParseError, or a decoding/IO failure
        logger.debug("Strict SRT parse failed for %s: %s", path.name, exc)

    cues, skipped = parse_srt_lenient(path)
    if not cues:
        raise ValueError(f"未找到可用的字幕条目: {path}") from None
    if skipped:
        logger.warning("%s: 跳过 %d 条无法解析的字幕块", path.name, skipped)
    return cues_to_segments(cues), skipped


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_file(
    srt_path: Path,
    output_dir: Path | None = None,
    formats: list[str] | None = None,
    *,
    stem: str | None = None,
) -> list[Path]:
    """
    Convert one SRT file to the requested text formats.

    Args:
        srt_path: Source subtitle file.
        output_dir: Destination folder; defaults to the SRT's own folder.
        formats: Any of ``("txt", "md")``; defaults to both.
        stem: Output filename without extension; defaults to the SRT's name.

    Returns:
        The paths written, in the order requested.
    """
    formats = [f for f in (formats or VALID_FORMATS) if f in VALID_FORMATS]
    if not formats:
        raise ValueError(f"没有可用的输出格式，请从 {VALID_FORMATS} 中选择")

    segments, _ = load_segments(srt_path)
    if not segments:
        raise ValueError(f"字幕文件没有文本内容: {srt_path}")

    # to_plaintext joins cues with a language-dependent separator. Detect from
    # the text itself and restore the pipeline's language afterwards, so this
    # never leaks state into an in-process transcription run.
    previous_language = get_cps_language()
    set_cps_language("auto")
    try:
        formatter = TextFormatter()
        contents = {
            fmt: getattr(formatter, _WRITERS[fmt])(segments)
            for fmt in formats
        }
    finally:
        set_cps_language(previous_language)

    destination = Path(output_dir) if output_dir else srt_path.parent
    destination.mkdir(parents=True, exist_ok=True)
    out_stem = stem or srt_path.stem

    written: list[Path] = []
    for fmt in formats:
        out_path = destination / f"{out_stem}.{fmt}"
        atomic_write_text(out_path, contents[fmt])
        written.append(out_path)
    return written


def collect_srt_files(inputs: list[Path]) -> list[Path]:
    """Expand files and directories into a sorted, de-duplicated SRT list."""
    found: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            found.extend(scan_video_files(path, ["srt"]))
        elif path.suffix.lower() == ".srt":
            found.append(path)
    unique = list(dict.fromkeys(p.resolve() for p in found))
    return sorted(unique, key=lambda p: str(p))


def convert_paths(
    inputs: list[Path],
    output_dir: Path | None = None,
    formats: list[str] | None = None,
) -> tuple[int, list[str]]:
    """
    Convert every SRT under *inputs*.

    Returns:
        ``(succeeded, errors)``.
    """
    targets = collect_srt_files(inputs)
    if not targets:
        return 0, ["未找到 .srt 文件"]

    succeeded = 0
    errors: list[str] = []
    for srt_path in targets:
        try:
            written = convert_file(srt_path, output_dir, formats)
        except Exception as exc:
            logger.error("转换失败 %s: %s", srt_path.name, exc)
            errors.append(f"{srt_path.name}: {exc}")
            continue
        succeeded += 1
        logger.info("✓ %s → %s", srt_path.name, ", ".join(p.name for p in written))
    return succeeded, errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="srt-convert",
        description="把 SRT 字幕转成纯文本 (.txt) 或 Markdown (.md)。",
    )
    parser.add_argument(
        "input", nargs="+", type=Path,
        help="SRT 文件或包含 SRT 的目录（目录会递归查找）",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="输出目录（默认与源 SRT 同目录）",
    )
    parser.add_argument(
        "-f", "--formats", nargs="+", default=list(VALID_FORMATS),
        choices=list(VALID_FORMATS), metavar="FMT",
        help=f"输出格式，可选 {' / '.join(VALID_FORMATS)}（默认两者都出）",
    )
    parser.add_argument("--verbose", action="store_true", help="输出调试日志")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    succeeded, errors = convert_paths(args.input, args.output, args.formats)
    if errors:
        for message in errors:
            logger.error(message)
        sys.exit(2 if succeeded else 1)
    logger.info("完成：%d 个文件", succeeded)
    sys.exit(0)


if __name__ == "__main__":
    main()
