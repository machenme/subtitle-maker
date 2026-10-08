"""End-to-end progress trace: real ffmpeg, real pipeline, simulated GPU.

Run directly to see the numbers the UI would have received:
    .venv/Scripts/python.exe tests/progress_trace.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.gpu_scheduler as scheduler_module
import src.main as main_module
from src.audio_extractor import AudioExtractor
from src.config import PipelineConfig
from src.text_formatter import Segment

# The fake scheduler probes chunk durations, which needs the run's temp dir.
config_temp_dir = "."


class _FakeScheduler:
    """Stands in for the GPU pool: emits intra-chunk decode progress."""

    def __init__(self, _config):
        pass

    def process(self, tasks, *, progress_callback=None, status_callback=None):
        if status_callback:
            status_callback("loading_model")
        time.sleep(0.4)  # model load
        if progress_callback:
            progress_callback(0.0)
        chunks = len(tasks)
        for done in range(1, chunks + 1):
            for step in range(1, 21):
                progress_callback((done - 1 + step / 20) / chunks)
                time.sleep(0.02)
        if progress_callback:
            progress_callback(1.0)
        # One cue per 2s so the merged output is a plausible subtitle.
        out = {}
        for audio_path, _video_path in tasks:
            duration = AudioExtractor(Path(config_temp_dir)).get_duration(audio_path)
            cues = [
                Segment(start=float(i), end=float(i) + 2.0, text=f"台詞{i}")
                for i in range(max(1, int(duration // 2)))
            ]
            out[audio_path] = (cues, "ja")
        return out


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="progress-trace-"))
    media = tmp / "demo.m4a"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=40", "-c:a", "aac", str(media)],
        check=True, timeout=120,
    )

    main_module.GpuScheduler = _FakeScheduler
    scheduler_module.GpuScheduler = _FakeScheduler

    model_dir = tmp / "model-ct2"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.bin").write_bytes(b"\0")
    config_path = tmp / "config.yaml"
    config_path.write_text(
        "\n".join([
            f'input_dir: "{tmp.as_posix()}"',
            f'output_dir: "{(tmp / "out").as_posix()}"',
            f'model_path: "{model_dir.as_posix()}"',
            "language: ja",
            "output_formats: [srt]",
            "chunk_duration: 0",
        ]) + "\n",
        encoding="utf-8",
    )
    config = PipelineConfig.build({"config": str(config_path)})
    global config_temp_dir
    config_temp_dir = str(config.effective_temp_dir)

    timeline: list[tuple[float, str, float]] = []
    lock = threading.Lock()

    def on_progress(stage, current, target):
        # "detected_language" is a pre-existing overload: it carries the
        # language code, not a fraction.
        if stage == "detected_language":
            with lock:
                timeline.append((time.monotonic(), f"lang={current}", 1.0))
            return
        with lock:
            timeline.append((time.monotonic(), stage, float(current) / max(float(target), 1e-9)))

    ok, count, error = main_module.run_one_video(
        config, media, progress_callback=on_progress
    )
    assert ok, error

    start = timeline[0][0]
    print(f"\nresult: ok={ok} segments={count}")
    print(f"{len(timeline)} progress reports\n")
    print(f"{'t(s)':>6}  {'stage':<14} {'file %':>7}")
    print("-" * 32)
    step = max(1, len(timeline) // 28)
    for stamp, stage, value in timeline[::step]:
        print(f"{stamp - start:6.2f}  {stage:<14} {value * 100:6.1f}%")
    stamp, stage, value = timeline[-1]
    print(f"{stamp - start:6.2f}  {stage:<14} {value * 100:6.1f}%   <- final")

    values = [value for _s, stage, value in timeline if not stage.startswith("lang=")]
    assert values == sorted(values), "file progress went backwards"
    assert values[-1] == 1.0, f"ended at {values[-1]}"
    print("\nOK: monotonic 0 -> 100%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())