# models/ — 模型存放目录（不进 git）

这个目录**整体被 git 忽略**，仓库里只留这个占位说明。模型请自行下载，体积较大。

| 用途 | 目录名 | 大小 | 来源 |
|------|--------|------|------|
| 语音转写（默认） | `faster-whisper-large-v3-turbo-ct2` | ~1.6 GB | [deepdml/faster-whisper-large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2) |
| 语音转写（更准） | `faster-whisper-large-v3-ct2` | ~3.1 GB | [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) |
| 语音转写（更快） | `faster-whisper-medium-ct2` | ~1.5 GB | [Systran/faster-whisper-medium](https://huggingface.co/Systran/faster-whisper-medium) |
| 本地离线翻译（可选） | `hy-mt2-1.8b-guff` | ~1.9 GB | 见 README 的翻译章节 |

**目录名必须带 `-ct2` 后缀**：GUI 的模型下拉框按 `models/faster-whisper-{模型名}-ct2` 这个约定去找目录，差一个字符都找不到。

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
