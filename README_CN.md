# Subtitle Maker — 视频语音转文字并行处理管线

[English](README.md) | [简体中文](README_CN.md)

基于 `faster-whisper`（CTranslate2 后端）的离线批量音视频语音转文字工具。利用本地 GPU 多路并行推理，支持长音视频自动切割、并行转写、分段合并，一键输出 SRT 字幕、纯文本和 Markdown 文件。

输入支持常见视频格式以及 `m4a`、`mp3`、`wav`、`flac`、`ogg`、`opus`、`aac`、`wma` 等音频格式，均由 `ffmpeg` 统一转换后转写。

## 特性

- **PySide6 图形界面** — 拖拽音视频/SRT、选翻译语言、实时日志、GPU 监控，零命令行操作
- **可选双击启动器** — 不想开命令行的话，可自行打包一个不带控制台窗口的 `Subtitle-Maker.exe`
- **队列即拖放区** — 有文件时上方的拖放区自动隐藏，音视频 / SRT / 整个文件夹直接拖到队列任意位置即可入列
- **语言自动检测** — 默认自动识别视频语言，支持手动指定 14 种语言
- **GPU 多路并行** — spawn 独立进程，每进程常驻一个 WhisperModel，启动时逐个试探可用并发
- **长视频自动切割** — 超过 N 分钟的音频自动切成片段，多 Worker 并行处理，最后合并时间戳
- **断点续跑** — 中断后重跑自动跳过已完成的视频，进度持久化到 `.progress.json`
- **字幕级切分** — SRT 输出按标点 + 时长切分为可读短句（2-7 秒 / 条，≤40 字）
- **多种输出** — SRT 字幕（默认）+ TXT 纯文本 + MD 带时间轴，可选勾选
- **免费翻译** — 基于 Microsoft Edge Translator Web 接口，14 种语言，按批次输出 PotPlayer 兼容字幕
- **智能跳过** — 导入视频自动检测同名 `.srt` / `.bilingual.srt` / 源语言字幕，已翻译的直接跳过，有字幕的只翻译不转写
- **三份字幕输出** — 默认生成同名单语译文 `.srt`、`.bilingual.srt` 双语字幕和带源语言后缀的原始字幕
- **SRT 直翻** — 已有 SRT 文件拖入即翻，跳过转写，秒级出结果
- **每视频清理** — 处理完立即删除临时音频，不堆积 GB 级 temp 文件
- **本地 ASR** — ASR 模型本地加载；启用翻译时，字幕文本会按批次发送到 Microsoft Edge Translator Web 接口

## 硬件要求

| 组件 | 最低 | 推荐 |
|------|------|------|
| GPU | NVIDIA 8 GB VRAM | RTX 5070 Ti 16 GB |
| NVIDIA 驱动 | ≥535 | ≥545 |
| 内存 | 16 GB | 32 GB |
| 存储 | SSD | NVMe SSD（临时音频 I/O） |
| OS | Windows 11 / Ubuntu 22.04+ | |

## 环境准备

### 1. 安装 ffmpeg

```bash
# Windows (scoop)
scoop install ffmpeg

# 验证
ffmpeg -version
```

### 2. Python 环境

使用 `mise` → `uv` → Python 3.11 工具链：

```bash
cd video-to-text
uv python pin 3.11          # 固定 Python 3.11
uv venv                      # 创建虚拟环境
uv sync
```

### 3. 下载模型

从 HuggingFace 克隆 `faster-whisper-large-v3-turbo-ct2`（CT2 格式，开箱即用）：

```bash
mkdir models
cd models
git lfs install
git clone https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2
```

> 单实例 FP16 约 2.5 GB 显存。程序启动时会逐个加载 Worker，直到模型加载失败，再自动回退到失败前的并发数。

## 快速开始

### 桌面启动器（可选，不用开命令行）

仓库里**不附带** `Subtitle-Maker.exe`——它是可重新生成的构建产物，不入库。
需要双击即开的启动器时自行打包（需先 `uv sync --group dev` 装好 PyInstaller）：

```bash
launcher\build_exe.bat
```

脚本会在项目根目录生成约 8 MB 的 `Subtitle-Maker.exe`，双击即可打开界面，**不会弹出任何控制台窗口**。
它只负责拉起 GUI，程序本体仍在源码树里运行，因此升级依赖**不需要重新打包**。
启动失败会弹窗提示，详细输出追加到 `logs\gui.log`；想跟踪日志就盯着这个文件。

### 图形界面（推荐）

```bash
uv run python -m src.gui
```

1. 拖入音视频文件（或直接拖入已有的 `.srt` 字幕、整个文件夹）；有文件时拖放区会自动隐藏，之后把文件拖到队列表格任意位置即可继续添加
2. 在"翻译为"下拉选择目标语言（如中文 chs）
3. 点"开始转写" → 自动完成转写 + 翻译，默认输出单语译文、双语字幕和原始字幕

> **智能检测**：导入视频时自动检查同目录是否已有三份字幕；三份齐全时直接跳过，有原文只翻译不转写。
>
> **三份字幕**：默认单语译文使用视频同名 `.srt`，双语字幕使用 `.bilingual.srt`，原始字幕使用源语言后缀（例如 `.jpn.srt`）。
>
> **字幕直翻**：直接把 `.srt` 文件拖进窗口，选择翻译语言，点开始即可跳过转写、只做翻译。

### 翻译接口

Microsoft Edge Translator 接口无需 API Key；每批默认发送 50 条字幕，并按返回顺序写回。Edge 接口不支持自动检测源语言，使用 GUI 或 CLI 翻译时请指定实际源语言。

Legacy GTX 翻译接口使用 `translate.googleapis.com/translate_a/t`，每批字幕通过换行合并为一次请求。该接口需要代理，GUI 选择“Legacy GTX (免费)”时会提示输入代理，例如 `127.0.0.1:7897`。

**Index-Translate 官方 API**（`index_api`）免费提供 `Index-Translate-35B-A3B` 模型，强度高于本地 9B；Key 任意填写即可。**字幕文本会发往 `index-translate.bilibili.com`**，且该接口需代理才能连通。

**本地模型**（`llm`）完全离线，用 GGUF 权重在本机翻译。可在界面的「本地模型」输入框指定自己的量化版本（目录或 `.gguf` 文件均可），留空则用 `models/index-translate-9b/` 里的默认权重。量化选型实测见 `models/README.md`。

### 命令行 — 转写 + 翻译

```bash
# 仅转写
uv run python -m src.main --input ./videos --output ./output

# 转写并翻译为中文
uv run python -m src.main --input ./videos --output ./output --translate zh

# 输出结构：
#   output/
#   ├── demo1.srt             ← 中文双语字幕（译为主字幕）
#   ├── demo1.jpn.srt         ← 原始日文字幕
#   └── ...
```

### 命令行 — 仅翻译已有 SRT

```bash
# 翻译单个 SRT 文件
uv run python -c "
from src.translator import EdgeTranslator, translate_srt
translate_srt('demo.srt', 'zh', provider=EdgeTranslator())
"
```

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | PATH | **必填** | 输入视频目录 |
| `--output` | PATH | **必填** | 输出根目录 |
| `--config` | PATH | `./config.yaml` | 配置文件路径 |
| `--model` | str | `large-v3-turbo` | 模型：large-v3-turbo / large-v3 / medium |
| `--workers` | int | 16 | 最大尝试并发数；模型加载失败时自动降为失败前的并发数 |
| `--chunk-duration` | int | 0 | 手动指定切割时长（秒），0 = 自动均分 |
| `--temp-dir` | PATH | 系统临时目录 | 临时音频存放路径 |
| `--language` | str | `auto` | 目标语言（ISO 639-1，auto = 自动检测） |
| `--beam-size` | int | 5 | Beam Search 宽度 (1-10) |
| `--compute-type` | str | `float16` | 推理精度：float16 / int8_float16 |
| `--no-vad` | flag | false | 禁用 VAD 语音检测 |
| `--no-cleanup` | flag | false | 保留临时音频文件 |
| `--translate` | str | — | 翻译目标语言（如 zh / en / ko），不指定则不翻译 |
| `--translator` | str | `bing` | 翻译后端：`bing` / `gtx` / `index_api` / `llm` |
| `--proxy` | URL | — | GTX / index_api 代理，例如 `127.0.0.1:7897` |
| `--verbose` | flag | false | 输出 DEBUG 级日志 |
| `--force` | flag | false | 忽略断点续跑，强制全部重跑 |

CLI 参数优先级高于配置文件。

## 配置文件

项目根目录的 `config.yaml`，可通过 `--config` 指定自定义路径：

```yaml
input_dir: "."                     # 输入音视频目录
output_dir: "./output"             # 输出根目录
temp_dir: null                     # 临时音频（null = 系统临时目录）

model_path: "./models/faster-whisper-large-v3-turbo-ct2"
model_size: "large-v3-turbo"
language: "auto"                  # 自动检测，可手动指定 ja/zh/en/ko/...
beam_size: 5
vad_filter: true
compute_type: "float16"

max_workers: null                   # 最大尝试并发数（null = 16；模型加载失败时自动降级）
chunk_duration: 0                   # 0 = 自动均分（按并发数），>0 = 手动秒数

video_extensions:                  # 扫描的音视频扩展名（兼容旧配置键名）
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

output_formats:                    # 输出格式（默认仅 SRT，可追加 txt / md）
  - srt
translate_to: "zh"                 # 自动翻译目标语言（"" = 不翻译）
swap_subtitles: true               # 生成单语译文、双语字幕和原始字幕

cleanup_temp: true                 # 完成后清理临时文件
```

## 使用示例

### 批量处理

```bash
uv run python -m src.main --input ./videos --output ./subtitles
```

### 调整并发和切割策略

```bash
# 自动均分（默认）：视频时长 / 并发数 = 每块时长，确保同时结束
uv run python -m src.main --input ./videos --output ./out

# 手动指定 10 分钟切割
uv run python -m src.main --input ./videos --output ./out --chunk-duration 600
```

### 断点续跑

```bash
# 中断后重新执行相同命令，自动跳过已完成视频
uv run python -m src.main --input ./videos --output ./subtitles

# 强制重跑全部
uv run python -m src.main --input ./videos --output ./subtitles --force
```

### 切换语言

```bash
# 手动指定 ASR 识别语言（默认自动检测）
uv run python -m src.main --input ./videos --output ./out --language en
```

### 翻译字幕

```bash
# 转写后自动翻译为中文
uv run python -m src.main --input ./videos --output ./out --translate zh

# GUI 方式：拖入视频或 .srt 文件，选择"翻译为 → 中文(zh)"，点开始
uv run python -m src.gui
```

> 翻译基于 Microsoft Edge Translator Web 接口，免费、无需 API Key。默认生成：`视频名.srt`（单语译文）、`视频名.bilingual.srt`（双语字幕）和 `视频名.jpn.srt` / `视频名.eng.srt`（原始字幕）。

## 管线架构

```
[音视频扫描] → [ffmpeg 提取 16kHz Mono WAV] → [音频切割 (可选)]
                                                    │
                                          ┌─────────┼─────────┐
                                          ▼         ▼         ▼
                                     [Worker 0] [Worker 1] [Worker 2] ...
                                     (WhisperModel 常驻显存，GPU 并行推理)
                                          │         │         │
                                          └─────────┼─────────┘
                                                    ▼
                                          [分段合并 + 时间偏移]
                                                    │
                                    ┌───────────────┼───────────────┐
                                    ▼               ▼               ▼
                                output/           output/         output/
                              video_a.srt      video_b.srt     video_c.srt
```

### GPU 并发探测与资源保护

`max_workers` 只是本次启动探测的上限，默认上限为 16。调度器逐个预加载
Whisper 模型：只有模型成功加载且显存仍至少保留 2.5 GiB 推理余量时，才会
继续启动下一个 Worker；如果上一 Worker 的显存占用可测量，还会额外保留其
占用量的 25% 作为下一次加载余量。运行中显存低于 1 GiB，或某个 Worker
加载失败、异常退出时，立即停止本次转写并清理进程，避免继续消耗系统内存和
磁盘交换空间。

### 长视频切割机制

**自动模式（默认 `chunk_duration: 0`）**：取视频时长 ÷ 并发数 = 每块时长，确保所有 Worker 分到均匀负载、同时结束，避免"最后一块只有几十秒"的浪费。

**手动模式**：指定固定秒数，ffmpeg segment muxer 切分，适合特殊场景。切出的片段并行送入 GPU 调度器，完成后 `combine_chunk_segments()` 根据偏移量还原绝对时间戳并去重合并。

## 输出格式

### SRT（标准字幕）

```
1
00:00:18,000 --> 00:00:20,000
時間ないです。今日はありがとうございます。

2
00:00:20,000 --> 00:00:24,000
フラエティーに参加していただいて、料金ゲットっていうのをやってるんですけど、
```

每条 2-7 秒、≤40 字符，按句尾标点切分，适合直接嵌入视频。

### TXT（纯文本）

日文无空格连续拼接，适合全文搜索和 NLP 下游处理。

### MD（Markdown 带时间轴）

```markdown
[00:00:18] 時間ないです。今日はありがとうございます。
[00:00:20] フラエティーに参加していただいて、料金ゲットっていうのをやってるんですけど、
```

时间戳精确到秒，方便人工校对定位。

## 实测性能（RTX 5070 Ti 16 GB）

| 视频时长 | 切割 | Chunk 数 | Worker | 处理耗时 | 实时倍数 |
|----------|------|----------|--------|----------|----------|
| 29 分钟 | 15 min | 2 | 4 | ~27 秒 | ~63x |
| 131 分钟 | 15 min | 9 | 4 | ~95 秒 | ~**82x** |
| 270 分钟（估） | 15 min | 18 | 4 | ~3 分钟 | ~90x |

> 实时倍数 = 视频时长 / 处理耗时。倍数随视频增长而上升，因为 GPU 并行度被长视频切割填满。

## 退出码

| Code | 含义 |
|------|------|
| 0 | 全部任务成功 |
| 1 | 参数错误或配置无效 |
| 2 | 部分任务失败 |
| 3 | 致命错误（GPU 不可用 / OOM） |

## 项目结构

```
subtitle-maker/
├── src/
│   ├── gui.py                # PySide6 图形界面入口
│   ├── main.py               # CLI 入口，流程编排，信号处理
│   ├── config.py            # YAML 配置加载 / CLI 覆盖 / 校验
│   ├── audio_extractor.py   # Stage 1: ffmpeg 提取 + 切割 + 容错 fallback
│   ├── gpu_scheduler.py     # Stage 2: spawn 多进程 + 信号量 GPU 调度
│   ├── transcribe_worker.py # Stage 2: faster-whisper 推理 + 模型缓存
│   ├── text_formatter.py    # Stage 3: 分段合并 + 字幕切分 + 格式化
│   ├── task_manager.py      # 任务状态跟踪 + 断点续跑
│   ├── monitor.py           # GPU 显存 / 利用率监控
│   └── utils.py             # 文件扫描、SRT 校验、时间戳格式化
├── models/                  # 模型文件（需自行下载，见 models/README.md）
│   └── faster-whisper-large-v3-turbo-ct2/
├── launcher/                # 可选：无控制台启动器（源码 + 构建脚本，不入库产物）
├── output/                  # 输出目录（默认即视频所在目录）
│   ├── {video_name}.srt
│   └── ...
├── config.yaml              # 默认配置
├── pyproject.toml           # uv 项目配置
├── uv.lock                  # 锁定的依赖版本（可复现构建）
├── .gitignore
├── README.md                # English
└── README_CN.md             # 简体中文
```

## 常见问题

**Q: 报错 `cublas64_12.dll is not found`？**

```bash
uv pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12
```

已包含在 `requirements.txt` 中。若仍报错，确认 CUDA 驱动版本 ≥535。

**Q: 显存不足 OOM？**

程序启动 Worker 时会逐个加载模型；如果第 N 个 Worker 因显存不足失败，本次自动使用 N-1 个并发。仍可通过 `--workers` 手动降低尝试上限。

**Q: 音视频不同步？**

ffmpeg 提取时若遇损坏 AAC 流会自动触发 raw-AAC fallback（两步法：先裸流拷贝，再独立解码）。若仍失败，源文件音频轨道可能严重损坏。

**Q: SRT 字幕太长 / 太短？**

调整编辑 `text_formatter.py` 中的 `_MAX_CHARS_PER_SUB`（默认 40）和 `_MAX_SUB_DURATION`（默认 7.0 秒）。切割粒度建议用自动模式。

**Q: 字幕时间轴对不上视频？**

子进程 CUDA context 隔离正常时不应出现。若发生，检查源视频帧率是否异常。

## 相关项目

生成字幕后如需翻译为双语字幕（如日→中），推荐：

👉 [rockbenben/subtitle-translator](https://github.com/rockbenben/subtitle-translator) — 基于 LLM 的字幕翻译工具，支持多种翻译引擎，可直接导入本工具生成的 SRT 文件进行翻译。
