"""Regression tests for the issues listed in POTENTIAL_ISSUES.md.

Each test pins one behaviour that used to be wrong. They are deliberately
cheap: real ffmpeg only where the bug lived inside an ffmpeg interaction.
"""
from __future__ import annotations

import subprocess
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.audio_extractor import (
    AudioExtractionCancelled,
    AudioExtractor,
)
from src.config import DEFAULT_MAX_WORKERS, PipelineConfig
from src.gpu_scheduler import GpuScheduler
from src.main import _run_llm_translation_pass
from src.translator.pipeline import TranslationError


def _silent_wav(path: Path, seconds: int) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000 * seconds)
    return path


def _long_ffmpeg_cmd(tmp_path: Path) -> list[str]:
    """An ffmpeg invocation that cannot finish before the deadline.

    ``-re`` makes ffmpeg read the input at 1x, so a 20 s clip really takes
    20 s to encode. Generating silence instead would finish in milliseconds
    and the deadline would never be reached.
    """
    source = _silent_wav(tmp_path / "long.wav", 20)
    return [
        "ffmpeg", "-y", "-loglevel", "error",
        "-re", "-i", str(source),
        "-c", "copy", str(tmp_path / "out.wav"),
    ]


# ---------------------------------------------------------------------------
# Issue 2 — a hung ffmpeg must still hit its timeout
# ---------------------------------------------------------------------------

@pytest.mark.needs_ffmpeg
def test_ffmpeg_timeout_fires_when_no_progress_arrives(tmp_path: Path):
    """The old loop blocked inside stdout iteration and never checked the clock."""
    extractor = AudioExtractor(tmp_path / "temp")
    started = time.monotonic()

    with pytest.raises(subprocess.TimeoutExpired):
        extractor._run_ffmpeg(
            _long_ffmpeg_cmd(tmp_path),
            timeout=2,
            total_seconds=60.0,
            progress_callback=lambda _fraction: None,
        )

    elapsed = time.monotonic() - started
    assert elapsed < 20, f"timeout took {elapsed:.1f}s to fire"


@pytest.mark.needs_ffmpeg
def test_ffmpeg_timeout_leaves_no_child_process(tmp_path: Path):
    """A killed ffmpeg must not keep running behind the caller's back."""
    extractor = AudioExtractor(tmp_path / "temp")

    with pytest.raises(subprocess.TimeoutExpired):
        extractor._run_ffmpeg(
            _long_ffmpeg_cmd(tmp_path),
            timeout=2,
            total_seconds=60.0,
            progress_callback=lambda _fraction: None,
        )

    assert not _ffmpeg_processes_alive()


def _ffmpeg_processes_alive() -> bool:
    """True when an ffmpeg.exe is still running (tasklist output is not UTF-8)."""
    time.sleep(0.5)
    listed = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq ffmpeg.exe"],
        capture_output=True, timeout=30,
    )
    output = (listed.stdout or b"").decode("utf-8", errors="replace")
    return "ffmpeg.exe" in output


# ---------------------------------------------------------------------------
# Issue 3 — a raising progress callback must not orphan ffmpeg
# ---------------------------------------------------------------------------

@pytest.mark.needs_ffmpeg
def test_raising_progress_callback_kills_ffmpeg(tmp_path: Path):
    """Previously the finally block only closed pipes; ffmpeg kept running."""
    extractor = AudioExtractor(tmp_path / "temp")

    def exploding_callback(_fraction):
        raise RuntimeError("progress UI died")

    with pytest.raises(RuntimeError, match="progress UI died"):
        extractor._run_ffmpeg(
            _long_ffmpeg_cmd(tmp_path),
            timeout=60,
            total_seconds=60.0,
            progress_callback=exploding_callback,
        )

    assert not _ffmpeg_processes_alive()


def _ffmpeg_processes_alive() -> bool:
    """True when an ffmpeg.exe is still running (tasklist output is not UTF-8)."""
    time.sleep(0.5)
    listed = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq ffmpeg.exe"],
        capture_output=True, timeout=30,
    )
    output = (listed.stdout or b"").decode("utf-8", errors="replace")
    return "ffmpeg.exe" in output


# ---------------------------------------------------------------------------
# Issue 6 — cancellation reaches a running ffmpeg
# ---------------------------------------------------------------------------

@pytest.mark.needs_ffmpeg
def test_cancel_event_stops_a_running_ffmpeg(tmp_path: Path):
    """A stop request used to wait for ffmpeg to finish on its own."""
    extractor = AudioExtractor(tmp_path / "temp")
    cancel_event = threading.Event()

    threading.Timer(0.5, cancel_event.set).start()
    started = time.monotonic()

    with pytest.raises(AudioExtractionCancelled):
        extractor._run_ffmpeg(
            _long_ffmpeg_cmd(tmp_path),
            timeout=60,
            total_seconds=20.0,
            progress_callback=lambda _f: None,
            cancel_event=cancel_event,
        )

    assert time.monotonic() - started < 15
    assert not _ffmpeg_processes_alive()


def test_cancel_before_extraction_short_circuits(tmp_path: Path):
    source = _silent_wav(tmp_path / "long.wav", 2)
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(AudioExtractionCancelled):
        AudioExtractor(tmp_path / "temp").extract(
            source, cancel_event=cancel_event
        )


# ---------------------------------------------------------------------------
# Issue 4 — stale chunks from another split must not be reused
# ---------------------------------------------------------------------------

def test_split_wav_clears_stale_chunks_even_when_no_split_is_needed(tmp_path: Path):
    """Short audio still has to drop leftovers from an earlier split.

    Needs no media tooling: the duration is injected, and an audio short
    enough to skip splitting never reaches ffmpeg at all.
    """
    wav_path = tmp_path / "source.wav"
    wav_path.write_bytes(b"RIFF" + b"\x00" * 4096)
    chunk_dir = tmp_path / "source_chunks"
    chunk_dir.mkdir()
    stale = chunk_dir / "source_30s_000.wav"
    stale.write_bytes(b"stale shard from an earlier run")

    extractor = AudioExtractor(tmp_path / "temp")
    extractor.get_duration = lambda _path: 2.0  # shorter than chunk_duration

    chunks = extractor.split_wav(wav_path, 30)

    assert chunks == [(0.0, wav_path)]
    assert not stale.exists(), "leftovers survive a run that does not split"


def test_split_wav_clears_stale_chunks_without_a_chunk_dir(tmp_path: Path):
    """No chunk directory at all must not break the short-audio path."""
    wav_path = tmp_path / "fresh.wav"
    wav_path.write_bytes(b"RIFF" + b"\x00" * 4096)
    extractor = AudioExtractor(tmp_path / "temp")
    extractor.get_duration = lambda _path: 1.0

    assert extractor.split_wav(wav_path, 30) == [(0.0, wav_path)]


@pytest.mark.needs_ffmpeg
@pytest.mark.needs_ffprobe
def test_split_wav_ignores_chunks_from_a_previous_split(tmp_path: Path):
    """A re-split at a different chunk_duration used to inherit old files."""
    wav_path = _silent_wav(tmp_path / "source.wav", 8)
    extractor = AudioExtractor(tmp_path / "temp")
    chunk_dir = tmp_path / "source_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    stale = chunk_dir / "source_000.wav"
    stale.write_bytes(b"stale chunk from an earlier run")

    chunks = extractor.split_wav(wav_path, 3)

    assert not stale.exists(), "a stale shard was left to be picked up later"
    assert chunks, "expected the audio to be split"
    assert all(path.name.startswith("source_3s_") for _, path in chunks), chunks
    assert all(path.exists() for _, path in chunks)


# ---------------------------------------------------------------------------
# Issues 1 & 5 — the deferred LLM pass translates only this run's subtitles
# ---------------------------------------------------------------------------

def _valid_srt(path: Path) -> Path:
    path.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nThis is a valid subtitle line for testing.\n",
        encoding="utf-8",
    )
    return path


def test_llm_pass_only_translates_reported_targets(tmp_path: Path, monkeypatch):
    """Scanning the output directory swept in old subtitles and copies."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    fresh = _valid_srt(output_dir / "fresh.srt")
    old = _valid_srt(output_dir / "old.srt")
    _valid_srt(output_dir / "old.bilingual.srt")

    translated: list[Path] = []

    def fake_translate(srt_path, target_lang, **kwargs):
        translated.append(Path(srt_path))
        assert kwargs["source_lang"] == "en", (
            "the detected language must reach the translator, not the config"
        )
        return (Path(srt_path), None, None)

    import src.main as main_module

    monkeypatch.setattr(main_module, "translate_srt_with_outputs", fake_translate)

    import src.translator.llm as llm_module

    class FakeLlm:
        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def minimum_free_vram_bytes():
            return 0

        def release(self):
            pass

    monkeypatch.setattr(llm_module, "LlmTranslator", FakeLlm)
    monkeypatch.setattr(llm_module, "wait_for_vram_release", lambda *_a, **_k: None)

    config = SimpleNamespace(
        output_dir=output_dir,
        translate_to="zh",
        swap_subtitles=True,
        language="auto",
        translation_provider="llm",
    )

    failures = _run_llm_translation_pass(config, [(fresh, "en")])

    assert failures == 0
    assert translated == [fresh], f"translated the wrong files: {translated}"
    assert old.exists()


def test_llm_pass_skips_already_translated_subtitles(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    done = _valid_srt(output_dir / "done.srt")
    _valid_srt(output_dir / "done.bilingual.srt")

    translated: list[Path] = []
    import src.main as main_module

    monkeypatch.setattr(
        main_module,
        "translate_srt_with_outputs",
        lambda srt_path, _lang, **_kw: translated.append(Path(srt_path)),
    )
    import src.translator.llm as llm_module

    monkeypatch.setattr(
        llm_module,
        "LlmTranslator",
        type("FakeLlm", (), {
            "__init__": lambda self, *_a, **_k: None,
            "minimum_free_vram_bytes": staticmethod(lambda: 0),
            "release": lambda self: None,
        }),
    )
    monkeypatch.setattr(llm_module, "wait_for_vram_release", lambda *_a, **_k: None)

    config = SimpleNamespace(
        output_dir=output_dir,
        translate_to="zh",
        swap_subtitles=True,
        language="auto",
        translation_provider="llm",
    )

    assert _run_llm_translation_pass(config, [(done, "ja")]) == 0
    assert translated == []


# ---------------------------------------------------------------------------
# Issue 8 — a dead worker fails each untouched task, not the whole batch
# ---------------------------------------------------------------------------

def test_dead_workers_fail_each_unfinished_task(tmp_path: Path):
    scheduler = GpuScheduler(SimpleNamespace(max_workers=2))
    tasks = [(tmp_path / f"chunk{i}.wav", tmp_path / "movie.mp4") for i in range(3)]

    class DeadWorker:
        name = "asr-worker-0"
        exitcode = -9

        def is_alive(self):
            return False

        def terminate(self):
            pass

        def kill(self):
            pass

        def join(self, *_args):
            pass

    scheduler._workers = [DeadWorker()]
    scheduler._scheduled_tasks = tasks
    scheduler._task_queue = None
    scheduler._progress_queue = None
    scheduler._result_queue = type("EmptyQueue", (), {
        "get": lambda self, timeout=None: (_ for _ in ()).throw(__import__("queue").Empty),
        "get_nowait": lambda self: (_ for _ in ()).throw(__import__("queue").Empty),
    })()

    results = scheduler._collect_results(expected=3)

    assert results == {}
    assert len(scheduler._task_errors) == 3
    assert all("exited unexpectedly" in reason for reason in scheduler._task_errors.values())


def test_process_reports_failures_through_the_error_sink(tmp_path: Path):
    """Per-file reasons must reach the caller, not stay inside the scheduler."""
    scheduler = GpuScheduler(SimpleNamespace(max_workers=1))
    audio = tmp_path / "chunk0.wav"
    video = tmp_path / "movie.mp4"
    scheduler.process = lambda tasks, **kwargs: (
        kwargs["error_sink"].update({audio: "CUDA out of memory"}) or {}
    )

    sink: dict = {}
    result = scheduler.process(
        [(audio, video)], error_sink=sink
    )

    assert result == {}
    assert sink == {audio: "CUDA out of memory"}


def test_run_one_video_names_the_failed_chunk_reason(monkeypatch, tmp_path: Path):
    """"Some chunks failed" hid which chunk broke and why."""
    import src.main as main_module

    source = tmp_path / "movie.mp4"
    source.write_bytes(b"media")
    chunk_a = tmp_path / "movie_0.wav"
    chunk_b = tmp_path / "movie_1.wav"
    for chunk in (chunk_a, chunk_b):
        chunk.write_bytes(b"audio")

    class FakeExtractor:
        def __init__(self, *_args, **_kwargs):
            pass

        def extract(self, *_args, **_kwargs):
            return chunk_a

        def get_duration(self, *_args, **_kwargs):
            return 600.0

        def split_wav(self, *_args, **_kwargs):
            return [(0.0, chunk_a), (300.0, chunk_b)]

    class FakeScheduler:
        def __init__(self, *_args, **_kwargs):
            pass

        def process(self, tasks, *, error_sink=None, **_kwargs):
            if error_sink is not None:
                error_sink[chunk_b] = "CUDA out of memory"
            return {chunk_a: ([], "ja")}

    monkeypatch.setattr(main_module, "AudioExtractor", FakeExtractor)
    monkeypatch.setattr(main_module, "GpuScheduler", FakeScheduler)

    ok, _segments, error = main_module.run_one_video(
        SimpleNamespace(
            effective_temp_dir=tmp_path,
            cleanup_temp=False,
            chunk_duration=300,
            max_workers=2,
            output_dir=tmp_path,
            output_formats=["srt"],
            translate_to="",
            translation_provider="bing",
            language="ja",
        ),
        source,
    )

    assert ok is False
    assert "1/2 chunk(s) failed" in error, error
    assert "CUDA out of memory" in error, error


# ---------------------------------------------------------------------------
# Issue 9 — configuration is validated at the boundary
# ---------------------------------------------------------------------------

def _write_config(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_input_dir_must_be_a_directory(tmp_path: Path):
    not_a_dir = tmp_path / "video.mp4"
    not_a_dir.write_bytes(b"media")
    config_path = _write_config(
        tmp_path / "config.yaml",
        f'input_dir: "{not_a_dir.as_posix()}"\nmodel_path: "{tmp_path.as_posix()}"\n',
    )

    with pytest.raises(ValueError, match="must be a directory"):
        PipelineConfig.build({"config": str(config_path)})


def test_null_max_workers_falls_back_to_the_default(tmp_path: Path):
    """`max_workers: null` used to crash with a TypeError from int(None)."""
    config_path = _write_config(
        tmp_path / "config.yaml",
        'input_dir: "."\nmax_workers: null\n'
        f'model_path: "{tmp_path.as_posix()}"\n',
    )

    config = PipelineConfig.build({"config": str(config_path)})

    assert config.max_workers == DEFAULT_MAX_WORKERS


def test_quoted_boolean_is_coerced_not_truthy(tmp_path: Path):
    config_path = _write_config(
        tmp_path / "config.yaml",
        'input_dir: "."\nvad_filter: "false"\nswap_subtitles: "no"\n'
        f'model_path: "{tmp_path.as_posix()}"\n',
    )

    config = PipelineConfig.build({"config": str(config_path)})

    assert config.vad_filter is False
    assert config.swap_subtitles is False


def test_non_boolean_flag_is_reported_with_its_key(tmp_path: Path):
    config_path = _write_config(
        tmp_path / "config.yaml",
        'input_dir: "."\ncleanup_temp: sometimes\n'
        f'model_path: "{tmp_path.as_posix()}"\n',
    )

    with pytest.raises(ValueError, match="cleanup_temp must be a boolean"):
        PipelineConfig.build({"config": str(config_path)})


def test_output_formats_must_be_a_list(tmp_path: Path):
    config_path = _write_config(
        tmp_path / "config.yaml",
        'input_dir: "."\noutput_formats: srt\n'
        f'model_path: "{tmp_path.as_posix()}"\n',
    )

    with pytest.raises(ValueError, match="output_formats must be a list"):
        PipelineConfig.build({"config": str(config_path)})


# ---------------------------------------------------------------------------
# Issue 11 — a failing translation batch returns instead of blocking
# ---------------------------------------------------------------------------

def test_failed_translation_does_not_wait_forever(tmp_path: Path, monkeypatch):
    """A request already sent cannot be cancelled, but it must not block us."""
    from src.translator import pipeline as pipeline_module
    from src.translator.pipeline import translate_srt
    from src.translator.types import TranslateConfig

    monkeypatch.setattr(pipeline_module, "_FAILURE_DRAIN_TIMEOUT", 0.1)
    cfg = TranslateConfig(max_workers=4, batch_size=1, request_delay=0)

    class PartiallyHangingProvider:
        """First batch succeeds, second fails fast, third never answers."""

        def __init__(self):
            self.calls = 0

        def translate_batch(self, texts, source_lang, target_lang):
            self.calls += 1
            if self.calls == 1:
                return [f"translated: {text}" for text in texts]
            if self.calls == 2:
                raise TranslationError("provider is unreachable")
            time.sleep(2)
            raise TranslationError("too late to matter")

    srt_path = tmp_path / "clip.srt"
    srt_path.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nOne line of dialogue for testing.\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\nA second line of dialogue here.\n\n"
        "3\n00:00:07,000 --> 00:00:09,000\nA third line of dialogue here.\n",
        encoding="utf-8",
    )
    provider = PartiallyHangingProvider()
    started = time.monotonic()

    with pytest.raises(TranslationError, match="provider is unreachable"):
        translate_srt(srt_path, "zh", provider=provider, config=cfg)

    assert time.monotonic() - started < 3, "a hung request blocked the caller"


def test_in_flight_request_finishes_before_return(tmp_path: Path, monkeypatch):
    """In-flight work is normally waited for, so the provider is not orphaned."""
    from src.translator import pipeline as pipeline_module
    from src.translator.pipeline import translate_srt
    from src.translator.types import TranslateConfig

    monkeypatch.setattr(pipeline_module, "_FAILURE_DRAIN_TIMEOUT", 5.0)
    cfg = TranslateConfig(max_workers=4, batch_size=1, request_delay=0)
    finished: list[int] = []

    class SlowThenFailingProvider:
        def __init__(self):
            self.calls = 0

        def translate_batch(self, texts, source_lang, target_lang):
            self.calls += 1
            if self.calls == 1:
                return [f"translated: {text}" for text in texts]
            if self.calls == 2:
                raise TranslationError("second batch failed")
            time.sleep(0.4)
            finished.append(self.calls)
            raise TranslationError("third batch failed too late")

    srt_path = tmp_path / "clip.srt"
    srt_path.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nOne line of dialogue for testing.\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\nA second line of dialogue here.\n\n"
        "3\n00:00:07,000 --> 00:00:09,000\nA third line of dialogue here.\n",
        encoding="utf-8",
    )

    with pytest.raises(TranslationError, match="second batch failed"):
        translate_srt(srt_path, "zh", provider=SlowThenFailingProvider(), config=cfg)

    # Nothing is still using the provider once we are back: the pending call
    # was waited for rather than abandoned mid-request.
    assert finished == [3], "the caller returned while a request was still running"


# ---------------------------------------------------------------------------
# Issue 10 — the fingerprint memo expires
# ---------------------------------------------------------------------------

def test_fingerprint_cache_expires(tmp_path: Path, monkeypatch):
    """An in-place rewrite can restore size+mtime; a permanent memo would lie."""
    from src import task_manager as tm

    media = tmp_path / "movie.mp4"
    media.write_bytes(b"v1" * 512)

    tm._FINGERPRINT_CACHE.clear()
    monkeypatch.setattr(tm, "_FINGERPRINT_CACHE_TTL_SECONDS", -1.0)
    try:
        calls: list[str] = []
        real_compute = tm._compute_fingerprint
        monkeypatch.setattr(
            tm,
            "_compute_fingerprint",
            lambda path, size, mtime: (
                calls.append(path.name), real_compute(path, size, mtime)
            )[-1],
        )
        tm._file_fingerprint(media)
        tm._file_fingerprint(media)
    finally:
        tm._FINGERPRINT_CACHE.clear()

    assert calls == ["movie.mp4", "movie.mp4"], "expired entry was reused"


def test_fingerprint_cache_detects_an_in_place_rewrite(tmp_path: Path):
    """Same path, same size, same mtime, same inode — different content.

    This is the case the memo used to get wrong: it answered from the cache
    and the rewritten file was skipped as already processed.
    """
    import os

    from src import task_manager as tm

    media = tmp_path / "movie.mp4"
    media.write_bytes(b"A" * 8192)
    stat = media.stat()

    tm._FINGERPRINT_CACHE.clear()
    real_compute = tm._compute_fingerprint
    calls: list[str] = []
    tm._compute_fingerprint = lambda path, size, mtime: (
        calls.append(path.name), real_compute(path, size, mtime)
    )[-1]
    try:
        tm._file_fingerprint(media)
        media.write_bytes(b"B" * 8192)
        os.utime(media, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        rewritten = tm._file_fingerprint(media)
    finally:
        tm._FINGERPRINT_CACHE.clear()

    assert calls == ["movie.mp4", "movie.mp4"], "the rewrite was served from the memo"
    assert rewritten["head_tail_hash"] != ""
