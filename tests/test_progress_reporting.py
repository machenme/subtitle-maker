"""The progress bar must actually move — and only forward.

These are end-to-end checks on the plumbing: real ffmpeg, a real ffmpeg
progress stream, and the real ``PipelineWorker`` signal flow. They are what
would have caught a frozen bar before shipping.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import wave
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pytest
from PySide6.QtWidgets import QApplication

from src.audio_extractor import AudioExtractor
from src.gpu_scheduler import GpuScheduler
from src.gui import STAGE_LABELS, AsrWindow, PipelineWorker
from src.transcribe_worker import transcribe_worker


def _make_media(path: Path, seconds: float) -> Path:
    """A real mp4 with a real audio track, built by ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-c:a", "aac", "-b:a", "64k", str(path),
        ],
        check=True,
        timeout=120,
    )
    assert path.is_file()
    return path


# ---------------------------------------------------------------------------
# 1. ffmpeg extraction streams real progress
# ---------------------------------------------------------------------------

def test_audio_extraction_reports_progress_bookends(tmp_path: Path):
    """The contract every caller sees: a 0% start and a 100% finish."""
    source = _make_media(tmp_path / "tone.m4a", 6)
    extractor = AudioExtractor(tmp_path / "temp")
    samples: list[float] = []

    wav_path = extractor.extract(source, progress_callback=samples.append)

    assert wav_path.is_file()
    assert samples[0] == 0.0, f"no starting point: {samples}"
    assert samples[-1] == 1.0, f"never closed at 100%: {samples}"
    assert samples == sorted(samples), f"progress went backwards: {samples}"


def test_ffmpeg_runner_streams_intermediate_progress(tmp_path: Path):
    """The regression this fixes: extraction was one opaque blocking call.

    ``-re`` makes ffmpeg consume the input at 1x, so the encode lasts long
    enough for ffmpeg's periodic progress blocks to be read as they arrive —
    which is exactly what happens on a real video.
    """
    source_wav = tmp_path / "long.wav"
    with wave.open(str(source_wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000 * 8)

    extractor = AudioExtractor(tmp_path / "temp")
    samples: list[float] = []
    result = extractor._run_ffmpeg(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-re", "-i", str(source_wav),
            "-c", "copy", str(tmp_path / "out.wav"),
        ],
        timeout=60,
        total_seconds=8.0,
        progress_callback=samples.append,
    )

    assert result.returncode == 0, result.stderr
    assert len(samples) >= 3, f"expected a stream of updates, got {samples}"
    assert samples[0] == 0.0
    assert any(0.0 < value < 1.0 for value in samples), (
        f"no intermediate progress at all: {samples}"
    )
    assert samples == sorted(samples), f"progress went backwards: {samples}"
    assert max(samples) <= 1.0


def test_audio_extraction_without_callback_still_works(tmp_path: Path):
    """The CLI path must not regress: no callback, no extra probing."""
    source = _make_media(tmp_path / "plain.m4a", 2)
    wav_path = AudioExtractor(tmp_path / "temp").extract(source)
    assert wav_path.is_file() and wav_path.stat().st_size > 1024


def test_cached_audio_short_circuits_to_full(tmp_path: Path):
    source = _make_media(tmp_path / "cached.m4a", 1)
    extractor = AudioExtractor(tmp_path / "temp")
    extractor.extract(source)
    samples: list[float] = []
    extractor.extract(source, progress_callback=samples.append)
    assert samples == [1.0]


# ---------------------------------------------------------------------------
# 2. The worker process reports intra-chunk decode progress
# ---------------------------------------------------------------------------

def test_transcribe_worker_streams_decode_progress(monkeypatch, tmp_path: Path):
    """A single chunk used to be one indivisible step; it now reports 0..1."""
    from types import SimpleNamespace

    def make_segment(end: float):
        return SimpleNamespace(
            start=max(0.0, end - 1.0), end=end, text=f"seg{end}", avg_logprob=-0.1
        )

    class FakeModel:
        def transcribe(self, *_args, **_kwargs):
            segments = [make_segment(float(i)) for i in range(1, 101)]
            info = SimpleNamespace(
                language="en", language_probability=0.99, duration=100.0
            )
            return iter(segments), info

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=lambda *_a, **_k: FakeModel()),
    )

    import src.transcribe_worker as worker_module

    worker_module._model_cache.clear()
    reported: list[tuple[str, float]] = []
    transcribe_worker(
        "model",
        tmp_path / "sample.wav",
        progress_queue=_Queue(reported),
    )

    fractions = [fraction for _key, fraction in reported]
    assert len(fractions) >= 5, f"too few reports to move a bar: {fractions}"
    assert fractions[0] == 0.0
    assert fractions[-1] == 1.0, "the chunk never closed at 100%"
    assert fractions == sorted(fractions), f"progress went backwards: {fractions}"


class _Queue:
    """Stand-in for the mp.Queue handed to the worker process."""

    def __init__(self, items: list | None = None):
        # Aliased, not copied: several tests assert on the caller's list.
        self.items: list[tuple[str, float]] = items if items is not None else []

    def put(self, item):
        self.items.append(item)

    def get_nowait(self):
        from queue import Empty

        if not self.items:
            raise Empty
        return self.items.pop(0)


# ---------------------------------------------------------------------------
# 3. The scheduler aggregates chunk progress instead of counting chunks
# ---------------------------------------------------------------------------

def test_single_chunk_run_reports_intermediate_progress(tmp_path: Path):
    """The regression: one chunk == one step, so the bar sat at 0% for ages.

    No real GPU here — the workers are simulated and only the parent-side
    aggregation is exercised, which is where the bug lived.
    """
    tasks = [(tmp_path / "only.wav", tmp_path / "only.mp4")]
    scheduler = GpuScheduler.__new__(GpuScheduler)
    scheduler._scheduled_tasks = list(tasks)
    scheduler._chunk_progress = {}
    scheduler._progress_queue = _Queue()
    scheduler._chunk_progress["only.wav"] = 0.0
    assert scheduler._aggregate_progress(["only.wav"]) == 0.0
    scheduler._chunk_progress["only.wav"] = 0.42
    assert scheduler._aggregate_progress(["only.wav"]) == pytest.approx(0.42)
    scheduler._chunk_progress["only.wav"] = 1.0
    assert scheduler._aggregate_progress(["only.wav"]) == pytest.approx(1.0)


def test_scheduler_drain_never_walks_a_chunk_backwards(tmp_path: Path):
    tasks = [(tmp_path / "a.wav", tmp_path / "a.mp4")]
    scheduler = GpuScheduler.__new__(GpuScheduler)
    scheduler._scheduled_tasks = list(tasks)
    scheduler._chunk_progress = {"a.wav": 0.8}
    queue = _Queue()
    queue.items.append(("a.wav", 0.2))  # stale / out-of-order report
    queue.items.append(("a.wav", 0.9))
    scheduler._progress_queue = queue

    scheduler._drain_progress_queue()

    assert scheduler._chunk_progress["a.wav"] == pytest.approx(0.9)


def test_scheduler_means_over_tasks_not_finished_count(tmp_path: Path):
    """Two chunks at 50% each must read as 50%, not 0%."""
    tasks = [
        (tmp_path / "a.wav", tmp_path / "a.mp4"),
        (tmp_path / "b.wav", tmp_path / "b.mp4"),
    ]
    scheduler = GpuScheduler.__new__(GpuScheduler)
    scheduler._scheduled_tasks = list(tasks)
    scheduler._chunk_progress = {"a.wav": 0.5, "b.wav": 0.5}
    assert scheduler._aggregate_progress(["a.wav", "b.wav"]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 4. The GUI worker keeps the overall bar monotonic across the whole run
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


class _Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _drain(app):
    for _ in range(200):
        app.processEvents()


def _collect_worker(qt_app, worker):
    events: list[tuple[float, float, int, int, str]] = []
    worker.progress.connect(lambda *args: events.append(args))
    worker.run()
    _drain(qt_app)
    return events


def _fake_asr_monotonic(monkeypatch, tmp_path: Path, steps=40):
    """Stand in for the pipeline: sweep 0..1 in `steps` visible increments."""
    import src.gui as gui_module

    def fake_run_one_video(config, media_path, *, progress_callback=None, **_kwargs):
        for step in range(steps + 1):
            progress_callback("transcribing", step / steps, 1.0)
        return True, 3, ""

    monkeypatch.setattr(gui_module, "run_one_video", fake_run_one_video)


def test_overall_bar_never_resets_when_the_next_file_starts(qt_app, tmp_path, monkeypatch):
    """The headline bug: the bar was the *current* file's percent.

    With two files the second file restarting at 0% drove the bar backwards,
    so a batch never looked like it was advancing.
    """
    _fake_asr_monotonic(monkeypatch, tmp_path)
    first = tmp_path / "one.mp4"
    second = tmp_path / "two.mp4"
    for path in (first, second):
        path.touch()

    worker = PipelineWorker(
        [first, second],
        {"one.mp4": "asr", "two.mp4": "asr"},
        _Config(
            translate_to="", swap_subtitles=True, language="ja",
            translation_provider="bing", translation_proxy="",
            translation_model_path="", output_dir=tmp_path,
        ),
        threading.Event(),
    )
    events = _collect_worker(qt_app, worker)
    overall = [event[0] for event in events]

    assert overall == sorted(overall), f"overall bar went backwards: {overall}"
    assert overall[-1] == pytest.approx(100.0)
    assert len(set(overall)) >= 20, f"bar only had {len(set(overall))} distinct values"


def test_bar_reaches_full_and_stays_there(qt_app, tmp_path, monkeypatch):
    _fake_asr_monotonic(monkeypatch, tmp_path)
    only = tmp_path / "one.mp4"
    only.touch()
    worker = PipelineWorker(
        [only],
        {"one.mp4": "asr"},
        _Config(
            translate_to="", swap_subtitles=True, language="ja",
            translation_provider="bing", translation_proxy="",
            translation_model_path="", output_dir=tmp_path,
        ),
        threading.Event(),
    )
    events = _collect_worker(qt_app, worker)
    assert events[-1][0] == pytest.approx(100.0)
    assert events[-1][1] == pytest.approx(100.0)
    assert events[-1][4] in STAGE_LABELS


def test_translation_does_not_jump_the_bar_to_full_before_it_starts(
    qt_app, tmp_path, monkeypatch
):
    """A queued translation used to report 100% the instant it was queued."""
    import src.gui as gui_module

    media = tmp_path / "movie.m4a"
    media.touch()
    srt = tmp_path / "movie.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")

    def fake_run_one_video(config, media_path, *, progress_callback=None, **_kwargs):
        for step in range(11):
            progress_callback("transcribing", step / 10, 1.0)
        return True, 1, ""

    monkeypatch.setattr(gui_module, "run_one_video", fake_run_one_video)
    monkeypatch.setattr(
        gui_module,
        "create_translator",
        lambda *_a, **_k: SimpleNamespaceObject(),
    )

    started = threading.Event()
    release = threading.Event()

    def slow_translate(_path, *_args, progress_callback=None, **_kwargs):
        started.set()
        release.wait(timeout=5)
        if progress_callback:
            progress_callback(1.0)
        return tmp_path / "movie.bilingual.srt"

    monkeypatch.setattr(gui_module, "translate_srt_with_outputs", slow_translate)

    worker = PipelineWorker(
        [media],
        {"movie.m4a": "asr"},
        _Config(
            translate_to="zh", swap_subtitles=True, language="ja",
            translation_provider="bing", translation_proxy="",
            translation_model_path="", output_dir=tmp_path,
        ),
        threading.Event(),
    )
    events: list[tuple[float, float, int, int, str]] = []
    worker.progress.connect(lambda *args: events.append(args))

    runner = threading.Thread(target=worker.run)
    runner.start()
    for _ in range(300):
        qt_app.processEvents()
        if started.is_set():
            break
    mid_run = [event[0] for event in events]
    release.set()
    runner.join(timeout=10)
    _drain(qt_app)

    overall = [event[0] for event in events]
    assert overall == sorted(overall), f"overall bar went backwards: {overall}"
    # ASR owns the first slice only, so translation was still outstanding.
    assert max(mid_run or [0]) < 100.0, "bar was already full while translating"
    assert overall[-1] == pytest.approx(100.0)


class SimpleNamespaceObject:
    def translate(self, *_a, **_k):
        return "x"

    def translate_batch(self, texts, *_a, **_k):
        return list(texts)


def test_window_bar_climbs_monotonically(qt_app, tmp_path, monkeypatch):
    """The painted widget must follow the worker's aggregate, not reset."""
    config_path = tmp_path / "config.yaml"
    (tmp_path / "model").mkdir()
    config_path.write_text(
        "\n".join(
            [
                f'input_dir: "{tmp_path.as_posix()}"',
                f'output_dir: "{(tmp_path / "out").as_posix()}"',
                f'model_path: "{(tmp_path / "model").as_posix()}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    window = AsrWindow()
    try:
        observed: list[int] = []
        for overall, file_percent, row, label, stage in [
            (0.0, 0.0, 0, 0, "extracting"),
            (18.0, 36.0, 0, 0, "transcribing"),
            (18.5, 37.0, 0, 0, "transcribing"),
            (55.0, 100.0, 0, 0, "transcribing"),
            (72.0, 50.0, 1, 1, "extracting"),
            (100.0, 100.0, 1, 1, "writing"),
        ]:
            window._update_progress(overall, file_percent, row, label, stage)
            observed.append(window.progress.value())

        assert observed == sorted(observed), f"widget bar went backwards: {observed}"
        assert observed[-1] == 100
    finally:
        window._duration_probes.clear()
        window.close()


def test_status_line_keeps_naming_the_asr_file_while_translation_reports(
    qt_app, tmp_path, monkeypatch
):
    """A background translation must not make the label flap between files."""
    import src.gui as gui_module

    first = tmp_path / "first.m4a"
    second = tmp_path / "second.m4a"
    for path in (first, second):
        path.touch()

    release = threading.Event()
    translated = threading.Event()

    def fake_run_one_video(config, media_path, *, progress_callback=None, **_kwargs):
        for step in range(11):
            progress_callback("transcribing", step / 20, 1.0)
        if media_path == second:
            # Pause mid-file so the translation thread reports while ASR is
            # still the thing the user is waiting on.
            assert translated.wait(timeout=5)
        for step in range(11, 21):
            progress_callback("transcribing", step / 20, 1.0)
        return True, 1, ""

    def slow_translate(_path, *_args, progress_callback=None, **_kwargs):
        translated.set()
        if progress_callback:
            progress_callback(0.5)
        release.wait(timeout=5)
        if progress_callback:
            progress_callback(1.0)
        return tmp_path / "out.srt"

    monkeypatch.setattr(gui_module, "run_one_video", fake_run_one_video)
    monkeypatch.setattr(gui_module, "translate_srt_with_outputs", slow_translate)
    monkeypatch.setattr(
        gui_module, "create_translator", lambda *_a, **_k: SimpleNamespaceObject()
    )

    worker = PipelineWorker(
        [first, second],
        {},
        _Config(
            translate_to="zh", swap_subtitles=True, language="ja",
            translation_provider="bing", translation_proxy="",
            translation_model_path="", output_dir=tmp_path,
        ),
        threading.Event(),
    )
    # (overall, file_percent, row_index, label_index, stage)
    events: list[tuple[float, float, int, int, str]] = []
    worker.progress.connect(lambda *args: events.append(args))

    runner = threading.Thread(target=worker.run)
    runner.start()
    # Wait until the ASR loop is demonstrably on the second file, so the
    # translation report that follows is genuinely "background".
    for _ in range(600):
        qt_app.processEvents()
        if translated.is_set() and any(event[3] == 1 for event in events):
            break
    release.set()
    runner.join(timeout=10)
    _drain(qt_app)

    during_asr = [event for event in events if event[3] == 1]
    assert during_asr, "never saw the ASR file's own progress"
    # Row 0 is the file being translated in the background; the status line
    # must stay on file 1 rather than flapping over to it.
    background = [event for event in events if event[2] == 0 and event[3] == 1]
    assert background, "the translation thread never reported"
    assert all(event[4] != "translating" for event in background), (
        f"status line flapped to the background file: {background}"
    )


def test_every_stage_reported_by_the_worker_has_a_label():
    """An unlabelled stage falls back to a generic string; keep the map tight."""
    from src.main import _STAGE_SPANS

    missing = set(_STAGE_SPANS) - set(STAGE_LABELS)
    assert not missing, f"stages without a UI label: {sorted(missing)}"


# ---------------------------------------------------------------------------
# 5. The per-file bar sits centred in its queue cell
# ---------------------------------------------------------------------------

@pytest.fixture
def queued_window(qt_app, tmp_path, monkeypatch):
    """An AsrWindow with three rows queued and their bars set to known values."""
    from src.gui import AsrWindow

    (tmp_path / "model").mkdir()
    (tmp_path / "config.yaml").write_text(
        "\n".join(
            [
                f'input_dir: "{tmp_path.as_posix()}"',
                f'output_dir: "{(tmp_path / "out").as_posix()}"',
                f'model_path: "{(tmp_path / "model").as_posix()}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    window = AsrWindow()
    # The cell geometry only settles once the table has been laid out; without
    # this the cell widget keeps an unassigned rect and every assertion below
    # measures nonsense.
    window.show()
    for index in range(3):
        media = tmp_path / f"clip-{index}.m4a"
        media.write_bytes(b"x")
        window._add_file(media)
    qt_app.processEvents()
    for bar, value in zip(window._progress_bars.values(), (100, 62, 18)):
        bar.setValue(value)
    qt_app.processEvents()
    try:
        yield window
    finally:
        window._duration_probes.clear()
        window.close()


def test_queue_progress_bar_is_vertically_centred_in_its_cell(queued_window, qt_app):
    """The bar used to sit flush on the row's top separator, 22px above the floor."""
    from src.gui import COL_PROGRESS

    table = queued_window.file_table
    for row in range(table.rowCount()):
        cell = table.cellWidget(row, COL_PROGRESS)
        bar = queued_window._cell_progress_bar(cell)
        cell_geom = cell.geometry()
        bar_geom = bar.geometry()

        top = bar_geom.y()
        bottom = cell_geom.height() - (bar_geom.y() + bar_geom.height())
        assert abs(top - bottom) <= 1, (
            f"row {row}: not vertically centred — {top}px above, {bottom}px below"
        )
        # And it must actually be inside the row, not pinned to an edge.
        assert top > 0, f"row {row}: bar is flush against the top edge ({top}px)"


def test_queue_progress_bar_has_symmetric_horizontal_gutters(queued_window):
    """A bar spanning the full cell reads as a divider, not a gauge."""
    from src.gui import COL_PROGRESS

    table = queued_window.file_table
    for row in range(table.rowCount()):
        cell = table.cellWidget(row, COL_PROGRESS)
        bar = queued_window._cell_progress_bar(cell)
        width = cell.geometry().width()
        left = bar.geometry().x()
        right = width - (bar.geometry().x() + bar.geometry().width())

        assert left > 0, f"row {row}: no left inset ({left}px)"
        assert right > 0, f"row {row}: no right inset ({right}px)"
        assert abs(left - right) <= 1, (
            f"row {row}: lopsided gutters — {left}px left, {right}px right"
        )


def test_progress_cell_survives_the_row_index_rebuild(queued_window):
    """The cell now wraps the bar, so the rebuild has to unwrap it.

    It used to ``isinstance(widget, QProgressBar)`` and would have silently
    dropped every bar from the cache after a row removal.
    """
    from src.gui import COL_PROGRESS, ProgressCell

    table = queued_window.file_table
    assert isinstance(table.cellWidget(0, COL_PROGRESS), ProgressCell)

    table.selectRow(0)
    queued_window._remove_selected()
    queued_window._rebuild_row_index()
    queued_window._refresh_queue_view()

    assert table.rowCount() == 2
    assert len(queued_window._progress_bars) == 2, (
        "row index rebuild lost the wrapped progress bars"
    )
    for path in queued_window.paths:
        assert queued_window._progress_bars[str(path)] is not None