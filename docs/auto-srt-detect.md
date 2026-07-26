# PRD: 同名字幕自动检测

---

## 1. PM 备注

> 此功能本质是预处理阶段的一次文件系统扫描，逻辑简单：检查同名 `.srt` / `.chs.srt` 是否存在，按规则分流。不涉及 ASR 或翻译模块内部改动，仅在导入入口和任务队列构建阶段插入判断。
>
> 风险点：文件名匹配仅靠 stem 比对（如 `myvideo.mp4` ↔ `myvideo.srt`），不会比对内容。用户修改过 SRT 文件名或内容不会被感知。

---

## 2. 项目概述

> 导入视频时自动检测同目录下是否已有同名 `.srt` 或 `.chs.srt` 文件，智能跳过已完成或仅需翻译的任务。

**核心目标**：减少重复劳动——视频 ASR 已完成（有 `.srt`）就只做翻译；翻译也完成了（有 `.chs.srt`）就直接跳过。

**明确不做**：
- 不做跨目录搜索
- 不做内容校验（不会打开 SRT 检查是否有效）
- 不改变 CLI 的 `--force` 行为

---

## 3. 功能矩阵

| 优先级 | 功能 | 描述 |
|--------|------|------|
| P0 | 检测 `.chs.srt` 跳过 | 同名 `.chs.srt` 存在 → 视频标记为"已完成"，不处理 |
| P0 | 检测 `.srt` 直翻 | 同名 `.srt` 存在（无 `.chs.srt`）→ 跳过 ASR，直接翻译为中文 `.chs.srt` |
| P1 | GUI 状态显示 | 列表中显示跳过原因："已有字幕" / "翻译中" / "已翻译" |

---

## 4. 用户故事

```
Story 1: 智能跳过 (P0)
> As a 用户, I want 导入视频时自动跳过已经有中文字幕的, so that 不用手动检查哪些处理过.

验收标准：
- Given 目录有 demo.mp4 和 demo.chs.srt,
  When 导入 demo.mp4,
  Then 列表显示状态"已有字幕"，不执行任何处理

Story 2: 只翻译不转写 (P0)
> As a 用户, I want 有日文字幕的视频直接翻译, so that 不重复跑 ASR.

验收标准：
- Given 目录有 demo.mp4 和 demo.srt（无 demo.chs.srt）,
  When 导入 demo.mp4,
  Then 跳过 ASR，直接翻译 demo.srt → demo.chs.srt
```

---

## 5. SPEC

### 5.1 检测时机

在 `_add_video` 时触发检测（GUI），或在 `scan_video_files` 后构建任务队列时（CLI）。

### 5.2 检测逻辑

```python
def check_existing_subs(video_path: Path) -> str:
    """
    Returns:
        "done"       — .chs.srt exists, skip entirely
        "translate"  — .srt exists (no .chs.srt), translate only
        "asr"        — neither exists, full pipeline
    """
    base = video_path.with_suffix("")
    if Path(f"{base}.chs.srt").exists():
        return "done"
    if Path(f"{base}.srt").exists():
        return "translate"
    return "asr"
```

### 5.3 分流处理

在 `_run_all` 中已有 SRT 直翻分支，新增 `"done"` 状态跳过：

```python
status = check_existing_subs(file_path)

if status == "done":
    self._set_video_status(file_path, "已有字幕")
    done_count += 1
    continue
elif status == "translate":
    # 使用已有的 .srt 而非视频文件
    srt_path = file_path.with_suffix(".srt")
    translate_srt(srt_path, "zh", ...)
    self._set_video_status(file_path, "✅ 已翻译")
elif is_srt:
    # 用户直接拖入 .srt 的现有分支
    ...
else:
    # 完整 ASR 管线
    ...
```

### 5.4 不改的文件

- `translator/` 子模块（零改动）
- `config.py` / `config.yaml`（无新增字段）
- `main.py`（CLI 通过 `--force` 覆盖）

---

## 📊 Final State: DONE
   ✅ PM备注  ✅ 项目概述  ✅ 功能矩阵  ✅ 用户故事  ✅ SPEC
