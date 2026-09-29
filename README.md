# Subtitle Maker — Parallel Speech-to-Text for Video and Audio

[English](README.md) | [简体中文](README_CN.md)

An offline batch speech-to-text pipeline built on `faster-whisper` (CTranslate2 backend). It runs local GPU inference across multiple parallel workers, automatically splits long media, transcribes the chunks concurrently, merges the segments back together, and writes SRT subtitles, plain text and Markdown in one pass.

Input covers common video containers plus `m4a`, `mp3`, `wav`, `flac`, `ogg`, `opus`, `aac` and `wma`. Everything is normalised through `ffmpeg` before transcription.

## Features

- **PySide6 desktop UI** — drag in audio/video/SRT, pick a target language, watch the live log and GPU monitor, no command line needed
- **Console-less launcher** — double-click `Subtitle-Maker.exe` and the UI opens without any terminal window flashing up
- **The queue is a drop target** — the drop zone hides itself once you have files; drag audio/video, SRT files or a whole folder onto any part of the queue table to enqueue them
- **Language auto-detection** — detects the spoken language by default, or pin any of 14 languages manually
- **Multi-stream GPU parallelism** — spawns independent processes, each holding a resident `WhisperModel`, probing the safe concurrency at startup
- **Automatic long-video splitting** — audio longer than N minutes is cut into pieces, processed in parallel, then merged back with corrected timestamps
- **Resumable** — a re-run skips already completed videos; progress persists in `.progress.json`
- **Readable subtitle segmentation** — SRT lines are split on punctuation and duration into short lines (2–7 s / line, ≤ 40 characters)
- **Multiple outputs** — SRT (default) + TXT plain text + MD timeline, each optional
- **Free translation** — Microsoft Edge Translator web endpoint, 14 languages, batched into PotPlayer-compatible subtitles
- **Smart skipping** — detects existing `.srt` / `.bilingual.srt` / source-language subtitles next to the video; already translated files are skipped, files with subtitles are translated only
- **Three subtitle outputs** — by default it writes `name.srt` (target language), `name.bilingual.srt` (dual line) and `name.<source-lang>.srt` (original)
- **Direct SRT translation** — drop an existing `.srt` in and skip transcription entirely; results in seconds
- **Per-video cleanup** — temporary audio is deleted right after each video, so GB-sized temp files never pile up
- **Local ASR** — models load locally; when translation is enabled only subtitle text is sent out, batched, to the Microsoft Edge Translator endpoint

## Hardware requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA 8 GB VRAM | RTX 5070 Ti 16 GB |
| NVIDIA driver | ≥ 535 | ≥ 545 |
| RAM | 16 GB | 32 GB |
| Storage | SSD | NVMe SSD (temp audio I/O) |
| OS | Windows 11 / Ubuntu 22.04+ | |

## Setup

### 1. Install ffmpeg

```bash
# Windows (scoop)
scoop install ffmpeg

# Verify
ffmpeg -version
```

### 2. Python environment

`mise` → `uv` → Python 3.11:

```bash
cd video-to-text
uv python pin 3.11          # pin Python 3.11
uv venv                     # create the virtualenv
uv sync
```

### 3. Download a model

Clone `faster-whisper-large-v3-turbo-ct2` (CT2 format, ready to use) from Hugging Face:

```bash
mkdir models
cd models
git lfs install
git clone https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2
```

> One FP16 instance costs ~2.5 GB of VRAM. At startup workers are loaded one at a time until a load fails, then the pipeline falls back to the last working concurrency.

If huggingface.com is unreachable from your network, swap in the `hf-mirror.com` mirror — the paths are identical:

```bash
git clone https://hf-mirror.com/deepdml/faster-whisper-large-v3-turbo-ct2
```

See [models/README.md](models/README.md) for other model sizes and exact directory names.

## Quick start

### Desktop launcher (no terminal)

A prebuilt `Subtitle-Maker.exe` can be dropped in the project root: double-click it and the GUI opens with **no console window**. Build it yourself (after changing `launcher/` sources) with:

```bash
# Requires PyInstaller: uv sync --group dev
launcher\build_exe.bat
```

The launcher (~8 MB) only starts the GUI; the application still runs from the source tree, so upgrading dependencies does **not** require repackaging. Startup failures pop up an error box and everything is appended to `logs\gui.log`.

### GUI (recommended)

```bash
uv run python -m src.gui
```

1. Drag in audio/video files (or an existing `.srt`, or a whole folder); the drop zone collapses once you have files, and you can keep dropping onto any part of the queue table
2. Pick a target language in the "Translate to" dropdown (e.g. Chinese zh)
3. Click **Start transcription** → transcription and translation run in one go, emitting the single-language, bilingual and original subtitles by default

> **Smart detection:** existing subtitles next to the video are checked on import; when all three exist the video is skipped, when only the original exists it is translated without re-transcribing.
>
> **Three subtitles:** by default the target language uses `video.srt`, the bilingual version `video.bilingual.srt`, and the original `video.<source-lang>.srt` (e.g. `video.jpn.srt`).
>
> **SRT-only translation:** drag an `.srt` file straight onto the window, pick a language and start — transcription is skipped.

### Translation backends

The Microsoft Edge Translator endpoint needs no API key; it sends a batch of 50 subtitle lines at a time and writes results back in order. It cannot auto-detect the source language, so always set the actual source language when translating.

The legacy GTX endpoint uses `translate.googleapis.com/translate_a/t`, merging each batch into a single request separated by newlines. It requires a proxy; selecting "Legacy GTX (free)" in the GUI prompts for one, e.g. `127.0.0.1:7897`.

### Command line — transcribe and translate

```bash
# Transcription only
uv run python -m src.main --input ./videos --output ./output

# Transcribe and translate into Chinese
uv run python -m src.main --input ./videos --output ./output --translate zh

# Output layout:
#   output/
#   ├── demo1.srt             ← Chinese subtitles (primary track)
#   ├── demo1.jpn.srt         ← original Japanese subtitles
#   └── ...
```

### Command line — translate an existing SRT only

```bash
uv run python -c "
from src.translator import EdgeTranslator, translate_srt
translate_srt('demo.srt', 'zh', provider=EdgeTranslator())
"
```

## CLI reference

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--input` | PATH | **required** | Input video directory |
| `--output` | PATH | **required** | Output root directory |
| `--config` | PATH | `./config.yaml` | Config file path |
| `--model` | str | `large-v3-turbo` | Model: large-v3-turbo / large-v3 / medium |
| `--workers` | int | 16 | Concurrency probe ceiling; falls back to the last successful count when a model load fails |
| `--chunk-duration` | int | 0 | Manual chunk length in seconds (0 = auto even split) |
| `--temp-dir` | PATH | system temp | Where temporary audio is stored |
| `--language` | str | `auto` | Target language (ISO 639-1, auto = detect) |
| `--beam-size` | int | 5 | Beam search width (1–10) |
| `--compute-type` | str | `float16` | Precision: float16 / int8_float16 |
| `--no-vad` | flag | false | Disable VAD voice detection |
| `--no-cleanup` | flag | false | Keep temporary audio files |
| `--translate` | str | — | Translation target language (e.g. zh / en / ko); omit to skip translation |
| `--translator` | str | `bing` | Backend: `bing` / `gtx` |
| `--proxy` | URL | — | Proxy for legacy GTX, e.g. `127.0.0.1:7897` |
| `--verbose` | flag | false | Enable DEBUG level logging |
| `--force` | flag | false | Ignore the resume checkpoint and redo everything |

CLI arguments take priority over the config file.

## Configuration file

`config.yaml` in the project root, overridable with `--config`:

```yaml
input_dir: "."                     # input audio/video directory
output_dir: "./output"             # output root directory
temp_dir: null                     # temp audio (null = system temp)

model_path: "./models/faster-whisper-large-v3-turbo-ct2"
model_size: "large-v3-turbo"
language: "auto"                  # auto-detect, or pin ja/zh/en/ko/...
beam_size: 5
vad_filter: true
compute_type: "float16"

max_workers: null                   # concurrency probe ceiling (null = 16; downgrades automatically)
chunk_duration: 0                   # 0 = auto split by concurrency, >0 = manual seconds

video_extensions:                  # scanned media extensions
  - mp4
  - mkv
  - mov
  - avi
  - flv
  - wmv
  - m4a
  - mp3
  - wav
  - flac
  - ogg
  - opus
  - aac
  - wma

output_formats:                    # output formats (SRT only by default, add txt / md)
  - srt
translate_to: "zh"                 # translation target language ("" = disabled)
swap_subtitles: true               # emit single-language, bilingual and original subtitles

cleanup_temp: true                 # clean temp files when done
```

## Examples

### Batch processing

```bash
uv run python -m src.main --input ./videos --output ./subtitles
```

### Tuning concurrency and splitting

```bash
# Auto split (default): duration / concurrency = per-chunk length, so all workers finish together
uv run python -m src.main --input ./videos --output ./out

# Manual 10-minute chunks
uv run python -m src.main --input ./videos --output ./out --chunk-duration 600
```

### Resuming

```bash
# Re-run the same command after an interruption; finished videos are skipped
uv run python -m src.main --input ./videos --output ./subtitles

# Force a full re-run
uv run python -m src.main --input ./videos --output ./subtitles --force
```

### Switching language

```bash
# Pin the ASR language explicitly (default is auto-detect)
uv run python -m src.main --input ./videos --output ./out --language en
```

### Translating subtitles

```bash
# Transcribe then translate into Chinese
uv run python -m src.main --input ./videos --output ./out --translate zh

# In the GUI: drag in a video or .srt, choose "Translate to → Chinese (zh)", click Start
uv run python -m src.gui
```

> Translation uses the Microsoft Edge Translator web endpoint — free and key-less. It produces `video.srt` (target language), `video.bilingual.srt` (bilingual) and `video.jpn.srt` / `video.eng.srt` (original).

## Pipeline architecture

```
[media scan] → [ffmpeg → 16 kHz mono WAV] → [optional audio split]
                                                    │
                                          ┌─────────┼─────────┐
                                          ▼         ▼         ▼
                                     [Worker 0] [Worker 1] [Worker 2] ...
                                     (WhisperModel resident in VRAM, parallel GPU inference)
                                          │         │         │
                                          └─────────┼─────────┘
                                                    ▼
                                        [merge segments + time offsets]
                                                    │
                                    ┌───────────────┼───────────────┐
                                    ▼               ▼               ▼
                                output/           output/         output/
                              video_a.srt      video_b.srt     video_c.srt
```

### GPU concurrency probing and resource protection

`max_workers` is only the ceiling for this run's probe (default 16). The scheduler preloads Whisper models one by one: it only starts another worker while the model loads successfully **and** at least 2.5 GiB of VRAM headroom remains; when the previous worker's footprint is measurable it also reserves 25% of it for the next load. If free VRAM drops below 1 GiB during a run, or any worker fails to load or exits abnormally, the run stops immediately and child processes are cleaned up rather than thrashing system memory and swap.

### Long-video splitting

**Auto mode (default, `chunk_duration: 0`)**: duration ÷ concurrency = chunk length, giving every worker an even workload so they finish together instead of leaving one last stub of a few seconds.

**Manual mode**: fixed chunk length via the ffmpeg segment muxer, for special cases. Segments are fed to the GPU scheduler in parallel, and `combine_chunk_segments()` restores absolute timestamps from the offsets and merges duplicates.

## Output formats

### SRT (standard subtitles)

```
1
00:00:18,000 --> 00:00:20,000
時間ないです。今日はありがとうございます。

2
00:00:20,000 --> 00:00:24,000
フラエティーに参加していただいて、料金ゲットっていうのをやってるんですけど、
```

Each line is 2–7 seconds and ≤ 40 characters, split on sentence-final punctuation, ready to mux straight into a video.

### TXT (plain text)

Continuous Japanese text without spaces — suitable for full-text search and NLP downstream.

### MD (Markdown with timeline)

```markdown
[00:00:18] 時間ないです。今日はありがとうございます。
[00:00:20] フラエティーに参加していただいて、料金ゲットっていうのをやってるんですけど、
```

Timestamps are second-accurate, which makes manual proofreading easy to locate.

## Measured performance (RTX 5070 Ti 16 GB)

| Video length | Chunk | Chunks | Workers | Wall time | Realtime factor |
|--------------|-------|--------|---------|-----------|-----------------|
| 29 min | 15 min | 2 | 4 | ~27 s | ~63x |
| 131 min | 15 min | 9 | 4 | ~95 s | ~**82x** |
| 270 min (est.) | 15 min | 18 | 4 | ~3 min | ~90x |

> Realtime factor = video duration / wall time. It climbs with longer videos because splitting keeps the GPU parallelism saturated.

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | All tasks succeeded |
| 1 | Invalid arguments or configuration |
| 2 | Some tasks failed |
| 3 | Fatal error (GPU unavailable / OOM) |

## Project structure

```
subtitle-maker/
├── src/
│   ├── gui.py                # PySide6 desktop UI entry point
│   ├── main.py               # CLI entry point, orchestration, signal handling
│   ├── config.py             # YAML config loading / CLI overrides / validation
│   ├── audio_extractor.py    # Stage 1: ffmpeg extract + split + fallback
│   ├── gpu_scheduler.py      # Stage 2: multi-process spawn + GPU semaphore
│   ├── transcribe_worker.py  # Stage 2: faster-whisper inference + model cache
│   ├── text_formatter.py     # Stage 3: merge segments + split subtitles + format
│   ├── task_manager.py       # Task state tracking + resume
│   ├── monitor.py            # GPU VRAM / utilisation monitor
│   └── utils.py              # File scanning, SRT validation, timestamp formatting
├── models/                   # Models (download yourself, see models/README.md)
│   └── faster-whisper-large-v3-turbo-ct2/
├── launcher/                 # Console-less desktop launcher (sources + build script)
├── output/                   # Output directory (defaults to the video's own folder)
│   ├── {video_name}.srt
│   └── ...
├── Subtitle-Maker.exe        # Double-click desktop launcher
├── config.yaml               # Default configuration
├── pyproject.toml            # uv project metadata
├── uv.lock                   # Locked dependency versions (reproducible builds)
├── .gitignore
├── README.md                 # English
└── README_CN.md              # Simplified Chinese
```

## FAQ

**Q: `cublas64_12.dll is not found`?**

```bash
uv pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12
```

Both are already in `requirements.txt`. If it still fails, verify the CUDA driver is ≥ 535.

**Q: Out of VRAM / OOM?**

Workers are loaded one at a time at startup; if the Nth worker runs out of memory the run continues with N-1 concurrent workers. You can also lower the probe ceiling with `--workers`.

**Q: Audio and video out of sync?**

A corrupted AAC stream triggers an automatic raw-AAC fallback during extraction (two passes: raw stream copy, then standalone decode). If that still fails, the source audio track is probably badly damaged.

**Q: SRT lines too long / too short?**

Edit `_MAX_CHARS_PER_SUB` (default 40) and `_MAX_SUB_DURATION` (default 7.0 s) in `text_formatter.py`. Auto mode is recommended for splitting granularity.

**Q: Subtitle timing does not match the video?**

This should not happen while the child CUDA contexts stay isolated. If it does, check the source video's frame rate for anomalies.

**Q: The UI says the model is missing?**

See [models/README.md](models/README.md): download the CT2 repository into `models/faster-whisper-<size>-ct2` and make sure `config.json` and `model.bin` are inside. Use `hf-mirror.com` when Hugging Face is blocked.

## Related projects

For turning generated subtitles into bilingual ones (e.g. Japanese → Chinese):

👉 [rockbenben/subtitle-translator](https://github.com/rockbenben/subtitle-translator) — an LLM-based subtitle translator supporting multiple engines; it can import the SRT files produced by this tool directly.
