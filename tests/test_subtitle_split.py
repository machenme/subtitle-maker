"""Unit tests for CPS semantic subtitle splitting."""
from src.text_formatter import _split_semantic, set_cps_language


def test_semantic_strong_punctuation():
    # Use punctuation that matches _SEMANTIC_SPLIT regex: [。！？!?\n]
    result = _split_semantic("AAAAA!AAAAA?BBBBB!CCC", 10)
    assert len(result) > 1, f"Expected split, got {result}"


def test_semantic_no_punctuation_fallback():
    result = _split_semantic("ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "abcdefghijklmnopqrstuvwxyz", 20)
    # No punctuation → returns [] (caller falls back to char-based)
    assert result == [], f"Expected empty fallback, got {result}"


def test_cps_language_switch():
    set_cps_language("zh")
    set_cps_language("ja")
    set_cps_language("en")
    # Should not crash
    assert True


if __name__ == "__main__":
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"])
