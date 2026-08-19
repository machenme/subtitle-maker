"""
Stage 3: Text formatter.
Converts segment lists to SRT, plain text, and Markdown with timestamps.
"""
from __future__ import annotations

import re
from pathlib import Path

from src.utils import atomic_write_text, format_timestamp


# ---------------------------------------------------------------------------
# Subtitle split constants
# ---------------------------------------------------------------------------

# Sentence-ending punctuation (splits here)
_SENTENCE_END_EN = re.compile(r"([.!?\n])")
_SENTENCE_END_CJK = re.compile(r"([。！？!?…\n])")
# Max characters per subtitle line
_MAX_CHARS_PER_SUB = 40
# Max seconds a single subtitle should stay on screen
_MAX_SUB_DURATION = 7.0
_MIN_SUB_DURATION = 0.8
_MAX_MERGE_GAP = 0.35
_MAX_ENGLISH_PHRASE_OVERFLOW = 12
# Characters-per-second by language (reading speed model)
# zh: Chinese   ja: Japanese   en: English
_CPS_BY_LANG: dict[str, int] = {"zh": 15, "ja": 10, "en": 17, "ko": 12}
_DEFAULT_CPS = 18

_ENGLISH_FUNCTION_WORDS = frozenset({
    "a", "an", "the", "this", "that", "these", "those",
    "to", "of", "in", "on", "at", "by", "for", "from", "with",
    "into", "onto", "over", "under", "after", "before", "during",
    "without", "about", "as", "than", "via", "and", "or", "but",
})
_ENGLISH_PROTECTED_PHRASES = frozenset({
    "a lot of",
    "day one update",
    "first things first",
    "for example",
    "in order to",
    "one of the",
    "right now",
    "up and running",
})


_cps_language = "ja"  # set by pipeline config


def set_cps_language(lang: str) -> None:
    """Set language code for CPS calculation (called before formatting)."""
    global _cps_language
    _cps_language = lang if lang in _CPS_BY_LANG or lang == "auto" else "ja"


def _cps_for_lang(lang: str) -> int:
    """Return recommended CPS for *lang* (ISO 639-1)."""
    return _CPS_BY_LANG.get(lang, _DEFAULT_CPS)


def _cps_for_text(text: str) -> int:
    """Choose a readable speed when the source language is auto-detected."""
    if _cps_language != "auto":
        return _cps_for_lang(_cps_language)
    if re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", text):
        return _cps_for_lang("ja")
    if re.search(r"[A-Za-z]", text):
        return _cps_for_lang("en")
    return _DEFAULT_CPS


def _is_english_text(text: str) -> bool:
    """Return whether *text* should use English word-spacing rules."""
    if _cps_language == "en":
        return True
    if _cps_language != "auto":
        return False
    has_latin = bool(re.search(r"[A-Za-z]", text))
    has_cjk = bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", text))
    return has_latin and not has_cjk


def _sentence_end_pattern(text: str) -> re.Pattern[str]:
    return _SENTENCE_END_EN if _is_english_text(text) else _SENTENCE_END_CJK


def _join_text(left: str, right: str) -> str:
    """Join adjacent subtitle text according to the active language."""
    separator = " " if _is_english_text(f"{left} {right}") else ""
    return f"{left.rstrip()}{separator}{right.lstrip()}".strip()


def _english_break_is_safe(
    left: str, right: str, *, allow_right_function_word: bool = False
) -> bool:
    """Avoid orphaned function words and known tightly bound phrases."""
    left_words = re.findall(r"[A-Za-z']+", left.lower())
    right_words = re.findall(r"[A-Za-z']+", right.lower())
    if not left_words or not right_words:
        return True

    if left_words[-1] in _ENGLISH_FUNCTION_WORDS:
        return False
    if not allow_right_function_word and right_words[0] in _ENGLISH_FUNCTION_WORDS:
        return False

    left_tail = left_words[-3:]
    right_head = right_words[:3]
    for phrase in _ENGLISH_PROTECTED_PHRASES:
        phrase_words = phrase.split()
        for split_at in range(1, len(phrase_words)):
            if (
                tuple(left_tail[-split_at:]) == tuple(phrase_words[:split_at])
                and tuple(right_head[:len(phrase_words) - split_at])
                == tuple(phrase_words[split_at:])
            ):
                return False
    return True


def _find_english_boundary(text: str, max_chars: int) -> int | None:
    """Find the latest conventional English subtitle break."""
    boundaries = [m.start() for m in re.finditer(r"\s+", text)]
    in_range = [p for p in boundaries if 0 < p <= max_chars]
    safe_in_range: int | None = None
    for position in reversed(in_range):
        if _english_break_is_safe(text[:position], text[position:]):
            safe_in_range = position
            break

    # Preserve a protected phrase even when it slightly exceeds the target.
    overflow = max_chars + _MAX_ENGLISH_PHRASE_OVERFLOW
    for position in boundaries:
        if max_chars < position <= overflow and _english_break_is_safe(
            text[:position], text[position:]
        ):
            return position

    if safe_in_range is not None and safe_in_range >= max_chars * 0.75:
        return safe_in_range

    # If grammar leaves no strict break, avoid orphaning a word at the end of
    # the line while allowing a function word to begin the next line.
    for position in reversed(in_range):
        if _english_break_is_safe(
            text[:position], text[position:], allow_right_function_word=True
        ):
            return position
    if safe_in_range is not None:
        return safe_in_range
    for position in boundaries:
        if position > max_chars and _english_break_is_safe(
            text[:position],
            text[position:],
            allow_right_function_word=True,
        ):
            return position
    # Never split an English word just to satisfy the nominal width.
    return len(text)


def _split_by_length(text: str, t_start: float, t_end: float) -> list[Segment]:
    """Split text into subtitle chunks by char count, distributing time evenly."""
    return _split_by_char_count_with_time(text, t_start, t_end, _MAX_CHARS_PER_SUB)


def _split_by_char_count(text: str, max_chars: int) -> list[str]:
    """Split text at semantic boundaries, falling back to char-based cuts."""
    # Try semantic split first; fall back to fixed-length
    chunks = _split_semantic(text, max_chars)
    if not chunks:
        return _split_at_word_boundaries(text, max_chars)
    return chunks


def _split_at_word_boundaries(text: str, max_chars: int) -> list[str]:
    """Split Latin text at whitespace, with a hard fallback for CJK/URLs."""
    remaining = text.strip()
    chunks: list[str] = []
    while len(remaining) > max_chars:
        if _is_english_text(remaining):
            boundary = _find_english_boundary(remaining, max_chars)
        else:
            boundary = remaining.rfind(" ", 1, max_chars + 1)
            if boundary <= 0:
                boundary = max_chars
        if boundary is None or boundary <= 0:
            boundary = max_chars
        chunks.append(remaining[:boundary].rstrip())
        remaining = remaining[boundary:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


# Punctuation priority for semantic splitting (strong → weak)
_SEMANTIC_SPLIT_MEDIUM_EN = re.compile(r"([;:])")
_SEMANTIC_SPLIT_WEAK_EN = re.compile(r"([,])")
_SEMANTIC_SPLIT_MEDIUM_CJK = re.compile(r"([；;])")
_SEMANTIC_SPLIT_WEAK_CJK = re.compile(r"([，,、])")


def _split_semantic(text: str, max_chars: int) -> list[str]:
    """Greedy semantic split: accumulate chars until punctuation, split when
    approaching *max_chars*.  Tries strong → medium → weak separators."""
    if len(text) <= max_chars:
        return [text]

    # Find best split point within max_chars from the end
    candidate = text[:max_chars]

    strong = _sentence_end_pattern(text)
    if _is_english_text(text):
        medium = _SEMANTIC_SPLIT_MEDIUM_EN
        weak = _SEMANTIC_SPLIT_WEAK_EN
    else:
        medium = _SEMANTIC_SPLIT_MEDIUM_CJK
        weak = _SEMANTIC_SPLIT_WEAK_CJK

    # Try strong punctuation
    for m in reversed(list(strong.finditer(candidate))):
        return [candidate[:m.end()]] + _split_semantic(text[m.end():], max_chars)

    # Try medium punctuation (；)
    for m in reversed(list(medium.finditer(candidate))):
        return [candidate[:m.end()]] + _split_semantic(text[m.end():], max_chars)

    # Try weak punctuation (，)
    for m in reversed(list(weak.finditer(candidate))):
        return [candidate[:m.end()]] + _split_semantic(text[m.end():], max_chars)

    # No punctuation found — fall back to char-based
    return []


def _split_by_char_count_with_time(
    text: str, t_start: float, t_end: float, max_chars: int
) -> list[Segment]:
    """Split text into subtitle-sized chunks and distribute timing.

    Uses CPS (characters-per-second) to guarantee minimum readable display
    time: each chunk gets at least ``len(chunk) / _MIN_CPS`` seconds.
    """
    chunks = _split_by_char_count(text, max_chars)
    if not chunks:
        return []
    dur = t_end - t_start
    total = len(text)
    if total == 0:
        return [Segment(t_start, t_end, text)]
    min_total = sum(len(part) / _cps_for_text(part) for part in chunks)
    results: list[Segment] = []
    t = t_start
    for chunk in chunks:
        # Proportional duration from original timing
        chunk_dur = (len(chunk) / total) * dur
        # Enforce minimum readable time (CPS, language-adaptive)
        min_dur = len(chunk) / _cps_for_text(chunk)
        # If the parent ASR segment is too short for all minimum durations,
        # retain proportional timing instead of producing zero-duration
        # chunks after the first chunk consumes the whole interval.
        if min_total <= dur:
            chunk_dur = max(chunk_dur, min_dur)
        # Cap
        chunk_dur = min(chunk_dur, _MAX_SUB_DURATION)
        chunk_end = min(t + chunk_dur, t_end)
        results.append(Segment(round(t, 3), round(chunk_end, 3), chunk.strip()))
        t = chunk_end
    return results


def _merge_short_subtitles(segments: list[Segment]) -> list[Segment]:
    """Merge short adjacent cues without crossing pauses or line limits."""
    if not segments:
        return []

    merged: list[Segment] = []
    for segment in segments:
        if not merged:
            merged.append(segment)
            continue

        previous = merged[-1]
        gap = segment.start - previous.end
        combined_text = _join_text(previous.text, segment.text)
        previous_tail = previous.text.rstrip()[-1:]
        can_merge = (
            gap <= _MAX_MERGE_GAP
            and gap >= -0.05
            and len(combined_text) <= _MAX_CHARS_PER_SUB
            and (
                segment.end - segment.start < _MIN_SUB_DURATION
                or previous.end - previous.start < _MIN_SUB_DURATION
            )
        )
        if can_merge:
            # Do not merge after a complete sentence even when the cue is short.
            if previous_tail and _sentence_end_pattern(previous.text).match(
                previous_tail
            ):
                can_merge = False
        if can_merge:
            previous.end = segment.end
            previous.text = combined_text
        else:
            merged.append(segment)
    return merged


# ---------------------------------------------------------------------------
# Segment type (lightweight dict-like, avoids heavy deps)
# ---------------------------------------------------------------------------

class Segment:
    """A single transcribed segment with timestamp and text."""

    __slots__ = ("start", "end", "text", "avg_logprob")

    def __init__(self, start: float, end: float, text: str, avg_logprob: float = 0.0):
        self.start = start
        self.end = end
        self.text = text.strip()
        self.avg_logprob = avg_logprob

    def __repr__(self) -> str:
        return f"Segment({self.start:.1f}-{self.end:.1f}: {self.text[:40]}...)"


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

class TextFormatter:
    """Formats a list of Segments into various output formats."""

    # ------------------------------------------------------------------
    # Subtitle-friendly splitting
    # ------------------------------------------------------------------

    @staticmethod
    def split_for_srt(segments: list[Segment]) -> list[Segment]:
        """
        Split long VAD segments into subtitle-friendly short lines.

        Rules:
        - Split on sentence-ending punctuation (。！？)
        - Max ~40 chars per subtitle
        - Max ~7 seconds display duration
        - Timestamps distributed proportionally within the parent segment
        """
        result: list[Segment] = []
        normalized: list[Segment] = []
        for seg in segments:
            if seg.end <= seg.start:
                if normalized and seg.text:
                    normalized[-1].text = (
                        f"{normalized[-1].text} {seg.text}"
                    ).strip()
                continue
            normalized.append(seg)

        for seg in normalized:
            dur = seg.end - seg.start
            sentence_matches = list(_sentence_end_pattern(seg.text).finditer(seg.text))
            if (
                dur <= _MAX_SUB_DURATION
                and len(seg.text) <= _MAX_CHARS_PER_SUB
                and len(sentence_matches) <= 1
            ):
                result.append(seg)
                continue

            # Simpler: find all positions of sentence-ending punctuation
            boundaries: list[int] = []
            text = seg.text
            for m in _sentence_end_pattern(text).finditer(text):
                boundaries.append(m.end())

            if not boundaries:
                # No sentence boundaries — split by char count
                subs = _split_by_length(text, seg.start, seg.end)
                result.extend(subs)
                continue

            # Keep complete sentences separate; short adjacent cues are merged
            # later only when there is no sentence-ending punctuation.
            chunks: list[str] = []
            start = 0
            for b in boundaries:
                sentence = text[start:b]
                chunks.append(sentence)
                start = b
            # Last piece (no trailing punctuation)
            if start < len(text):
                sentence = text[start:]
                chunks.append(sentence)

            # Further split any chunk that's too long
            final_chunks: list[str] = []
            for chunk in chunks:
                if len(chunk) > _MAX_CHARS_PER_SUB:
                    final_chunks.extend(_split_by_char_count(chunk, _MAX_CHARS_PER_SUB))
                else:
                    final_chunks.append(chunk)

            # Distribute timestamps proportionally
            total_chars = sum(len(c) for c in final_chunks)
            if total_chars == 0:
                result.append(seg)
                continue

            t = seg.start
            for chunk in final_chunks:
                chunk_dur = (len(chunk) / total_chars) * dur
                # Clamp duration
                chunk_dur = min(chunk_dur, _MAX_SUB_DURATION)
                chunk_end = min(t + chunk_dur, seg.end)

                result.append(Segment(
                    start=round(t, 3),
                    end=round(chunk_end, 3),
                    text=chunk.strip(),
                    avg_logprob=seg.avg_logprob,
                ))
                t = chunk_end

        return _merge_short_subtitles(result)

    # ------------------------------------------------------------------
    # SRT
    # ------------------------------------------------------------------

    @staticmethod
    def to_srt(segments: list[Segment]) -> str:
        """
        Generate standard SRT subtitle content.

        Format:
            1
            00:00:01,234 --> 00:00:05,678
            Text line

            2
            ...
        """
        if not segments:
            return ""

        lines: list[str] = []
        for i, seg in enumerate(segments, start=1):
            start_ts = format_timestamp(seg.start, fmt="srt")
            end_ts = format_timestamp(seg.end, fmt="srt")
            lines.append(str(i))
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(seg.text)
            lines.append("")  # blank separator

        return "\n".join(lines).rstrip("\n") + "\n"

    # ------------------------------------------------------------------
    # Plain text
    # ------------------------------------------------------------------

    @staticmethod
    def to_plaintext(segments: list[Segment]) -> str:
        """
        Generate clean continuous text without timestamps.
        English uses spaces; Japanese and Chinese do not.
        """
        if not segments:
            return "\n"
        separator = " " if _is_english_text(" ".join(seg.text for seg in segments[:5])) else ""
        return separator.join(seg.text.strip() for seg in segments).strip() + "\n"

    # ------------------------------------------------------------------
    # Markdown with timestamps
    # ------------------------------------------------------------------

    @staticmethod
    def to_markdown(segments: list[Segment]) -> str:
        """
        Generate Markdown with [HH:MM:SS] timestamp markers.
        Useful for human review / correction.
        """
        if not segments:
            return ""

        lines: list[str] = []
        for seg in segments:
            ts = format_timestamp(seg.start, fmt="md")
            lines.append(f"{ts} {seg.text}")

        return "\n".join(lines).strip() + "\n"

    # ------------------------------------------------------------------
    # Segment combining (for chunked parallel transcription)
    # ------------------------------------------------------------------

    @staticmethod
    def combine_chunk_segments(
        chunk_results: list[tuple[float, list[Segment]]]
    ) -> list[Segment]:
        """
        Combine segments from multiple audio chunks with offset correction.

        Only merges segments that *actually overlap in time* (happens at chunk
        boundaries where VAD detects the same speech from both sides).
        Adjacent segments with a gap are kept separate — their original
        faster-whisper timestamps are precise and should be preserved.

        Returns:
            Single sorted list of Segments with corrected absolute timestamps.
        """
        all_segments: list[Segment] = []
        for offset, segments in chunk_results:
            for seg in segments:
                seg.start += offset
                seg.end += offset
                all_segments.append(seg)

        all_segments.sort(key=lambda s: s.start)

        # Only merge segments whose time ranges actually overlap
        # (cross-chunk boundary duplication).
        merged: list[Segment] = []
        for seg in all_segments:
            if merged and seg.start < merged[-1].end:
                # Actual time overlap — merge into previous
                previous_duration = merged[-1].end - merged[-1].start
                current_duration = seg.end - seg.start
                merged[-1].end = max(merged[-1].end, seg.end)
                # If the overlap is substantial, the text is likely duplicate;
                # keep the longer version to avoid double text.
                if current_duration > previous_duration:
                    merged[-1].text = seg.text
                merged[-1].avg_logprob = max(merged[-1].avg_logprob, seg.avg_logprob)
            else:
                merged.append(seg)

        return merged

    # ------------------------------------------------------------------
    # Write all formats
    # ------------------------------------------------------------------

    def write_all(self, segments: list[Segment], base_path: Path, formats: list[str] | None = None) -> list[Path]:
        """
        Write all requested formats to disk.

        Args:
            segments: Transcribed segments.
            base_path: Output base path (e.g. output_dir/video_name). Extensions are appended.
            formats: List of formats to output ("srt", "txt", "md"). Default: all three.

        Returns:
            List of written file paths.
        """
        if formats is None:
            formats = ["srt", "txt", "md"]

        writers = {
            "srt": (".srt", self.to_srt),
            "txt": (".txt", self.to_plaintext),
            "md": (".md", self.to_markdown),
        }

        written: list[Path] = []
        base = Path(base_path)
        # Use the last path component as-is — caller is responsible for passing
        # a stem (no extension).  Don't re-strip via .stem because filenames
        # like "123.com.xxx" contain dots that are NOT an output-format suffix.
        stem = base.name
        parent = base.parent

        for fmt_key in formats:
            if fmt_key not in writers:
                continue
            ext, writer_fn = writers[fmt_key]
            out_path = parent / f"{stem}{ext}"

            # SRT gets subtitle-friendly short segments; TXT/MD get raw text
            if fmt_key == "srt":
                sub_segments = self.split_for_srt(segments)
                content = writer_fn(sub_segments)
            else:
                content = writer_fn(segments)

            atomic_write_text(out_path, content)
            written.append(out_path)

        return written
