"""Unit tests for the SRT → TXT / Markdown converter."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

from src.srt_convert import (
    collect_srt_files,
    convert_file,
    convert_paths,
    cues_to_segments,
)
from src.text_formatter import get_cps_language
from src.translator.parser import parse_srt_lenient

# The CLI tests run `python -m src.srt_convert`, which needs the project root on
# the path exactly as a user launching it from the repo would have it.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SAMPLE = (
    "1\n00:00:01,000 --> 00:00:03,000\nこんにちは\n\n"
    "2\n00:00:04,500 --> 00:00:06,000\n世界\n\n"
)

ENGLISH_SAMPLE = (
    "1\n00:00:01,000 --> 00:00:03,000\nHello there\n\n"
    "2\n00:00:04,000 --> 00:00:06,000\nGeneral Kenobi\n\n"
)

MULTILINE_SAMPLE = (
    "1\n00:00:01,000 --> 00:00:03,000\n这是第一行\n这是第二行\n\n"
    "2\n00:00:04,000 --> 00:00:06,000\n下一条\n\n"
)

# A cue with no index line and one with no text — the strict parser rejects the
# whole file for either, the lenient one keeps the readable cues.
BROKEN_SAMPLE = (
    "1\n00:00:01,000 --> 00:00:03,000\n第一条\n\n"
    "00:00:04,000 --> 00:00:06,000\n缺索引号但文字正常\n\n"
    "3\n00:00:07,000 --> 00:00:09,000\n\n"
    "4\n00:00:10,000 --> 00:00:12,000\n最后一条\n"
)


def _write(content: str, suffix: str = ".srt") -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    )
    handle.write(content)
    handle.close()
    return Path(handle.name)


@pytest.fixture
def srt_file():
    path = _write(SAMPLE)
    yield path
    path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Segment construction
# ---------------------------------------------------------------------------

def test_timestamp_conversion():
    segments = cues_to_segments(parse_srt_lenient(srt_file := _write(SAMPLE))[0])
    srt_file.unlink(missing_ok=True)
    assert segments[0].start == pytest.approx(1.0)
    assert segments[0].end == pytest.approx(3.0)
    assert segments[1].start == pytest.approx(4.5)


def test_multiline_cue_is_folded_into_one_line():
    path = _write(MULTILINE_SAMPLE)
    try:
        segments = cues_to_segments(parse_srt_lenient(path)[0])
        assert len(segments) == 2
        # A raw newline here would leak into the TXT/MD output.
        assert "\n" not in segments[0].text
        assert segments[0].text == "这是第一行 这是第二行"
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def test_convert_writes_both_formats(srt_file, tmp_path):
    written = convert_file(srt_file, tmp_path)
    assert [p.name for p in written] == [
        f"{srt_file.stem}.txt",
        f"{srt_file.stem}.md",
    ]
    assert all(p.is_file() for p in written)


def test_txt_has_no_timestamps_and_keeps_text(srt_file, tmp_path):
    out = convert_file(srt_file, tmp_path, ["txt"])[0]
    content = out.read_text(encoding="utf-8")
    assert "こんにちは" in content and "世界" in content
    assert "-->" not in content and "00:00:01" not in content


def test_md_carries_timestamps(srt_file, tmp_path):
    out = convert_file(srt_file, tmp_path, ["md"])[0]
    content = out.read_text(encoding="utf-8")
    assert "[00:00:01]" in content
    assert "こんにちは" in content


def test_english_text_is_space_joined(tmp_path):
    path = _write(ENGLISH_SAMPLE)
    try:
        out = convert_file(path, tmp_path, ["txt"])[0]
        # Without a separator this would read "Hello thereGeneral Kenobi".
        assert "Hello there General Kenobi" in out.read_text(encoding="utf-8")
    finally:
        path.unlink(missing_ok=True)


def test_default_output_dir_is_the_source_folder(srt_file):
    written = convert_file(srt_file, None, ["txt"])
    try:
        assert written[0].parent == srt_file.parent
    finally:
        written[0].unlink(missing_ok=True)


def test_conversion_restores_pipeline_language(srt_file, tmp_path):
    """A converter run must not change the language the pipeline is using."""
    before = get_cps_language()
    convert_file(srt_file, tmp_path)
    assert get_cps_language() == before


def test_custom_stem(srt_file, tmp_path):
    written = convert_file(srt_file, tmp_path, ["txt"], stem="renamed")
    assert written[0].name == "renamed.txt"


def test_rejects_unknown_format(srt_file, tmp_path):
    with pytest.raises(ValueError):
        convert_file(srt_file, tmp_path, ["docx"])


def test_empty_file_is_an_error(tmp_path):
    path = _write("\n\n")
    try:
        with pytest.raises(ValueError):
            convert_file(path, tmp_path)
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Damaged input
# ---------------------------------------------------------------------------

def test_bom_prefixed_file_parses(srt_file, tmp_path):
    path = tmp_path / "bom.srt"
    path.write_bytes(b"\xef\xbb\xbf" + SAMPLE.encode("utf-8"))
    written = convert_file(path, tmp_path, ["txt"])
    assert "こんにちは" in written[0].read_text(encoding="utf-8")


def test_lenient_parser_keeps_good_cues(tmp_path):
    path = tmp_path / "broken.srt"
    path.write_text(BROKEN_SAMPLE, encoding="utf-8")
    cues, skipped = parse_srt_lenient(path)
    texts = [c.text for c in cues]
    assert "第一条" in texts
    assert "缺索引号但文字正常" in texts
    assert "最后一条" in texts
    assert skipped == 1  # only the empty-text cue is unrecoverable


def test_damaged_file_still_converts(tmp_path):
    """One broken cue must not cost the user every readable line."""
    path = tmp_path / "broken.srt"
    path.write_text(BROKEN_SAMPLE, encoding="utf-8")
    out = convert_file(path, tmp_path, ["txt"])[0]
    content = out.read_text(encoding="utf-8")
    assert "第一条" in content and "最后一条" in content


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def test_collect_expands_directories(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.srt").write_text(SAMPLE, encoding="utf-8")
    (tmp_path / "nested" / "b.srt").write_text(SAMPLE, encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("nope", encoding="utf-8")
    found = collect_srt_files([tmp_path])
    assert [p.name for p in found] == ["a.srt", "b.srt"]


def test_convert_paths_reports_per_file_failures(tmp_path):
    good = tmp_path / "good.srt"
    good.write_text(SAMPLE, encoding="utf-8")
    bad = tmp_path / "bad.srt"
    bad.write_text("not a subtitle at all", encoding="utf-8")

    succeeded, errors = convert_paths([tmp_path], tmp_path, ["txt"])
    assert succeeded == 1
    assert len(errors) == 1
    assert "bad.srt" in errors[0]


def test_convert_paths_without_input_is_an_error(tmp_path):
    succeeded, errors = convert_paths([tmp_path], tmp_path, ["txt"])
    assert succeeded == 0
    assert errors == ["未找到 .srt 文件"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_writes_formats_to_output_dir(tmp_path):
    """End-to-end through `python -m src.srt_convert`, the way a user runs it."""
    import subprocess

    source = tmp_path / "src"
    source.mkdir()
    (source / "demo.srt").write_text(SAMPLE, encoding="utf-8")
    destination = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable, "-m", "src.srt_convert",
            str(source), "-o", str(destination), "-f", "txt", "md",
        ],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert (destination / "demo.txt").is_file()
    assert (destination / "demo.md").is_file()


def test_cli_single_format_only(tmp_path):
    import subprocess

    source = tmp_path / "src"
    source.mkdir()
    (source / "demo.srt").write_text(SAMPLE, encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable, "-m", "src.srt_convert",
            str(source / "demo.srt"), "-o", str(tmp_path / "out"), "-f", "md",
        ],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out" / "demo.md").is_file()
    assert not (tmp_path / "out" / "demo.txt").exists()


def test_cli_reports_missing_input(tmp_path):
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "src.srt_convert", str(tmp_path), "-o", str(tmp_path)],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=120,
    )
    assert result.returncode == 1
    assert "未找到" in result.stderr


def test_round_trip_matches_pipeline_output(tmp_path):
    """Converting a pipeline-written SRT must reproduce its .txt / .md exactly.

    This is the promise the feature makes: the converter is not a second,
    subtly different formatter — it is the same formatter fed from a file.
    """
    from src.text_formatter import Segment, TextFormatter

    segments = [
        Segment(1.0, 3.0, "おはようございます"),
        Segment(4.0, 6.0, "今日はよろしくお願いします"),
        Segment(90.0, 92.0, "Hello there"),
    ]
    formatter = TextFormatter()
    formatter.write_all(
        segments, base_path=tmp_path / "clip", formats=["srt", "txt", "md"]
    )

    written = convert_file(tmp_path / "clip.srt", tmp_path, ["txt", "md"])
    assert (tmp_path / "clip.txt").read_text(encoding="utf-8") == formatter.to_plaintext(segments)
    assert (tmp_path / "clip.md").read_text(encoding="utf-8") == formatter.to_markdown(segments)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
