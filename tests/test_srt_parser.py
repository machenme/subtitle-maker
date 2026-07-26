"""Unit tests for the SRT parser."""
import tempfile
from pathlib import Path
from src.translator.parser import parse_srt
from src.translator.types import ParseError


def _write_srt(content: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False, encoding="utf-8")
    f.write(content)
    f.close()
    return f.name


def test_parse_simple():
    srt = "1\n00:00:01,000 --> 00:00:03,000\nHello\n\n2\n00:00:04,000 --> 00:00:06,000\nWorld\n"
    path = _write_srt(srt)
    cues = parse_srt(path)
    assert len(cues) == 2, f"Expected 2 cues, got {len(cues)}"
    assert cues[0].index == 1
    assert cues[0].start == "00:00:01,000"
    assert cues[0].text == "Hello"
    assert cues[1].text == "World"


def test_parse_multiline_text():
    srt = "1\n00:00:01,000 --> 00:00:03,000\nLine 1\nLine 2\n\n"
    path = _write_srt(srt)
    cues = parse_srt(path)
    assert len(cues) == 1
    assert cues[0].text == "Line 1\nLine 2"


def test_parse_empty_raises():
    path = _write_srt("\n\n")
    try:
        parse_srt(path)
        assert False, "Should have raised ParseError"
    except ParseError:
        pass


if __name__ == "__main__":
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"])
