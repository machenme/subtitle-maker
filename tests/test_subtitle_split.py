"""Unit tests for CPS semantic subtitle splitting."""
from src.text_formatter import (
    Segment,
    TextFormatter,
    _split_by_char_count,
    _split_semantic,
    set_cps_language,
)


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


def test_english_split_does_not_break_words():
    chunks = _split_by_char_count(
        "update to the luminary and i got a lot of tips and tricks", 40
    )

    assert chunks == [
        "update to the luminary and i got",
        "a lot of tips and tricks",
    ]


def test_english_breaks_protect_function_words_and_phrases():
    set_cps_language("en")
    chunks = _split_by_char_count(
        "we are talking about the day one update for everyone", 40
    )

    assert chunks == [
        "we are talking about the day one update",
        "for everyone",
    ]
    assert not chunks[0].endswith(("a", "an", "the", "to", "of", "for"))


def test_short_parent_segment_never_creates_zero_duration_subtitles():
    set_cps_language("auto")
    segments = TextFormatter.split_for_srt([
        Segment(
            4.0,
            5.639,
            "update to the luminary and i got a lot of tips and tricks",
        )
    ])

    assert len(segments) == 2
    assert all(segment.end > segment.start for segment in segments)
    assert segments[-1].end <= 5.639


def test_short_adjacent_english_cues_are_merged():
    set_cps_language("en")
    segments = TextFormatter.split_for_srt([
        Segment(0.0, 0.45, "uh"),
        Segment(0.45, 1.20, "welcome everyone"),
    ])

    assert len(segments) == 1
    assert segments[0].text == "uh welcome everyone"
    assert segments[0].end == 1.2


def test_plaintext_uses_language_specific_spacing():
    set_cps_language("en")
    english = TextFormatter.to_plaintext([
        Segment(0, 1, "hello"),
        Segment(1, 2, "world"),
    ])
    assert english == "hello world\n"

    set_cps_language("ja")
    japanese = TextFormatter.to_plaintext([
        Segment(0, 1, "こんにちは"),
        Segment(1, 2, "世界"),
    ])
    assert japanese == "こんにちは世界\n"


def test_english_sentences_split_on_period_and_question_mark():
    set_cps_language("en")
    segments = TextFormatter.split_for_srt([
        Segment(0, 4, "Hello. How are you?"),
    ])

    assert [segment.text for segment in segments] == [
        "Hello.",
        "How are you?",
    ]


def test_japanese_sentences_split_on_japanese_punctuation():
    set_cps_language("ja")
    segments = TextFormatter.split_for_srt([
        Segment(0, 4, "こんにちは。元気ですか？"),
    ])

    assert [segment.text for segment in segments] == [
        "こんにちは。",
        "元気ですか？",
    ]


def test_multilingual_subtitle_quality_constraints():
    samples = [
        ("zh", "这是第一句中文字幕。这里是第二句中文内容！", 15),
        ("ja", "これは最初の字幕です。次の文も読みやすく分けます！", 12),
        ("en", "This is the first English subtitle. Here is the next sentence!", 17),
    ]

    for language, text, max_cps in samples:
        set_cps_language(language)
        original = Segment(0.0, 30.0, text)
        subtitles = TextFormatter.split_for_srt([original])

        assert len(subtitles) >= 2
        assert all(subtitle.text for subtitle in subtitles)
        assert all(subtitle.end > subtitle.start for subtitle in subtitles)
        assert all(len(subtitle.text) <= 40 for subtitle in subtitles)
        assert all(subtitle.end <= original.end for subtitle in subtitles)
        assert all(subtitle.end - subtitle.start <= 7.0 for subtitle in subtitles)
        assert all(
            len(subtitle.text) / (subtitle.end - subtitle.start) <= max_cps
            for subtitle in subtitles
        )


if __name__ == "__main__":
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"])
