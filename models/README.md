# models/ — model storage directory (not tracked by git)

English docs: [../README.md](../README.md) · 中文说明：[../README_CN.md](../README_CN.md)

这个目录**整体被 git 忽略**，仓库里只留这个占位说明。模型请自行下载，体积较大。

| 用途 | 目录名 | 大小 | 来源 |
|------|--------|------|------|
| 语音转写（默认） | `faster-whisper-large-v3-turbo-ct2` | ~1.6 GB | [deepdml/faster-whisper-large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2) |
| 语音转写（更准） | `faster-whisper-large-v3-ct2` | ~3.1 GB | [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) |
| 语音转写（更快） | `faster-whisper-medium-ct2` | ~1.5 GB | [Systran/faster-whisper-medium](https://huggingface.co/Systran/faster-whisper-medium) |
| 本地离线翻译（可选） | `index-translate-9b` | ~5.0 GB | [IndexTeam/Index-Translate-9B-GGUF](https://huggingface.co/IndexTeam/Index-Translate-9B-GGUF) |

**目录名必须带 `-ct2` 后缀**：GUI 的模型下拉框按 `models/faster-whisper-{模型名}-ct2` 这个约定去找目录，差一个字符都找不到。

## 本地翻译模型（GGUF）

把 `config.yaml` 的 `translation_provider` 设为 `llm` 即可启用离线翻译。

**默认位置**：`models/index-translate-9b/`，目录里放 `.gguf` 文件即可。
目录内有多个 `.gguf` 时优先选文件名带 `q8` 的，其次按文件名排序取第一个。

**用自己的模型**：不用挪文件，在 `translation_model_path` 里填路径即可，
支持两种写法 —— 具体的 `.gguf` 文件，或含 `.gguf` 的目录：

```yaml
translation_provider: "llm"
translation_model_path: "models/index-translate-9b"                      # 目录
# translation_model_path: "D:\\models\\Index-Translate-9B.Q4_K_M.gguf"   # 单个文件
```

GUI 里选「本地 Index-Translate 模型 (离线)」后，「翻译」区会多出一行「本地模型」，
可以直接粘贴路径或点「浏览…」选择。留空 = 使用默认位置。

### 量化版本怎么选

实测（RTX 5070 Ti，834 条英译中，日译中术语准确性）：

| 量化 | 体积 | 速度 | 术语准确性 |
|------|------|------|-----------|
| **IQ4_XS + imat** | ~5.0 GB | **79–109 ms/条** | **最好** |
| Q6_K | ~7.0 GB | 109–143 ms/条 | 偶发误译（游戏术语「抽」→「射」） |
| Q4_K_M | ~5.4 GB | 143–191 ms/条 | 偶发误译 |

**优先选带 imat（重要性矩阵校准）的量化版本**：它按权重重要性分配比特，
低比特下精度损失最小，字幕场景的专有名词更不容易译错。

下载：

```bash
mkdir -p models/index-translate-9b && cd models/index-translate-9b

# 官方全量化（社区 imat 版本质量更好，需自行获取）
git clone https://hf-mirror.com/IndexTeam/Index-Translate-9B-GGUF
```

`translate.py` 之类的推理脚本不在此列——本项目只读 GGUF 权重。

### 换模型前要知道的事

不同翻译模型的「指令遵循」能力差别很大，直接影响能不能正确输出「一条字幕对一条译文」：

| 模型 | 实测表现 |
|------|---------|
| **Index-Translate-9B** | 编号输出稳定，20 条一批可靠 |
| Index-Translate-2B | 会把英文原文照抄回来，且编号齐全 → **静默漏译** |
| Hy-MT2 1.8B / 7B | 会把多条字幕合并成一条（20 条输入 → 12 条输出） |

Hy-MT2 的 instTrans IFscore 只有 0.61~0.64，而 Index-Translate 是 0.82~0.83，
这是「整段翻译」与「逐条对齐」两种设计目标的差异，换模型前请先用几行字幕试一下。

## 最小可用文件

一个 CT2 模型目录里至少要这几个文件，缺任何一个都会转写失败：

```
faster-whisper-large-v3-turbo-ct2/
├── config.json
├── model.bin
├── preprocessor_config.json
├── tokenizer.json
└── vocabulary.json
```

## 下载方法

先装 Git LFS（`git lfs install`），然后：

```bash
mkdir -p models && cd models

git clone https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2
```

下载完确认目录里有 `config.json` 和 `model.bin` 即可。

### Hugging Face 连不上时用 hf-mirror

国内访问 `huggingface.co` 经常被墙，把域名换成 `hf-mirror.com` 即可，路径完全一致：

```bash
git clone https://hf-mirror.com/deepdml/faster-whisper-large-v3-turbo-ct2

# 也可以临时走环境变量
HF_ENDPOINT=https://hf-mirror.com git clone https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2
```

`large-v3` / `medium` 同理：

```bash
git clone https://hf-mirror.com/Systran/faster-whisper-large-v3  faster-whisper-large-v3-ct2
git clone https://hf-mirror.com/Systran/faster-whisper-medium    faster-whisper-medium-ct2
```

> 注意：Systran 的仓库 clone 下来目录名不带 `-ct2`，上面已经用第二个参数重命名了。

## 没有模型会怎样

点「开始转写」时会弹窗提示缺少模型，并提供 Hugging Face 原文页和 hf-mirror 镜像页两个跳转按钮，不会直接崩掉。
