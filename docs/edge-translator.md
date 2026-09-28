# Edge 翻译模块调用说明

本文说明本项目如何通过 Microsoft Edge Translator Web 接口翻译 SRT 字幕，包括调用入口、配置、请求协议、批处理规则、输出文件和异常处理。

## 1. 模块定位

Edge 翻译实现位于 `src/translator/edge.py`，公开类为 `EdgeTranslator`。项目内将该后端标识为 `bing`，原因是历史配置兼容；它实际访问的是 Edge Web 翻译接口，而不是 Azure Translator，也不需要 Azure Key。

```text
CLI / GUI / Python API
          |
          v
create_translator("bing")
          |
          v
EdgeTranslator
          |
          v
translate_srt_with_outputs() / translate_srt()
          |
          v
parse SRT -> 分批翻译 -> 合并结果 -> 写入字幕文件
          |
          v
https://edge.microsoft.com/translate/translatetext
```

需要网络连接。原始字幕文本会发送到 Microsoft Edge 的在线服务，因此该翻译方式不是离线处理，也不适合包含不应外发的内容。

## 2. 代码职责

| 文件 | 职责 |
| --- | --- |
| `src/translator/edge.py` | 构造 HTTP 请求、映射语言代码、校验响应、处理 429 限流和网络重试。 |
| `src/translator/__init__.py` | 通过 `create_translator("bing")` 创建 `EdgeTranslator`。 |
| `src/translator/pipeline.py` | 解析 SRT、每批调用 `translate_batch()`、保持字幕顺序并写文件。 |
| `src/translator/parser.py` | 解析 SRT 的序号、时间轴和原文。 |
| `src/translator/writer.py` | 写入单语译文和双语字幕。 |
| `src/main.py` | CLI 转写完成后触发翻译。 |
| `src/gui.py` | GUI 中选择 Edge 后端并翻译已存在或刚生成的 SRT。 |

`EdgeTranslator` 同时实现以下翻译提供方接口：

```python
def translate(text: str, source_lang: str, target_lang: str) -> str: ...

def translate_batch(
    texts: list[str], source_lang: str, target_lang: str
) -> list[str]: ...
```

`translate()` 是单条调用的兼容入口，内部转发给 `translate_batch([text], ...)`。实际翻译字幕时应优先使用批量入口。

## 3. HTTP 请求协议

`EdgeTranslator._request_batch()` 向下列地址发送一个 JSON 请求：

```text
POST https://edge.microsoft.com/translate/translatetext
```

请求由 `requests.Session` 发起，超时为 30 秒，带有浏览器 `User-Agent` 和 `Accept: application/json`。

以日语翻译为简体中文为例：

```http
POST /translate/translatetext?from=ja&to=zh-Hans&isEnterpriseClient=false HTTP/1.1
Host: edge.microsoft.com
Accept: application/json
Content-Type: application/json

["こんにちは", "次の字幕です"]
```

接口成功时，代码期望响应为与请求数组等长的数组，并从每一项读取 `translations[0].text`：

```json
[
  {"translations": [{"text": "你好", "to": "zh-Hans"}]},
  {"translations": [{"text": "下一条字幕", "to": "zh-Hans"}]}
]
```

结果数与请求数不一致、响应不是数组或缺少 `translations[0].text` 时，模块会抛出 `TranslationError`，不会写出可能错位的字幕。

### 语言代码

项目传入常用 ISO 639-1 代码；发送请求前仅对以下代码做显式转换：

| 项目代码 | Edge 请求代码 |
| --- | --- |
| `zh`、`zh-cn`、`zh-hans` | `zh-Hans` |
| `zh-tw`、`zh-hant` | `zh-Hant` |
| `auto` | `auto-detect`，随后被拒绝 |
| 其他代码，如 `ja`、`en`、`ko` | 原样发送 |

**Edge 后端不支持 `auto` 源语言。** `source_lang` 必须是实际的源语言，例如 `ja`、`en` 或 `ko`。若传入 `auto`，模块在发起 HTTP 请求前抛出：

```text
Edge Translator requires an explicit source language; automatic language detection is not supported
```

目标语言不能与源语言相同；GUI 会在界面层跳过这种翻译请求。

## 4. 字幕批处理与顺序保证

`translate_srt()` 的处理步骤如下：

1. 使用 `parse_srt()` 读取全部字幕条目，保留时间轴和原文。
2. 按 `TranslateConfig.batch_size` 分批，默认每批 50 条字幕。
3. 将单条字幕内部的换行替换为空格后，调用一次 `provider.translate_batch()`；不同字幕条目不会通过换行拼成一个字符串。
4. 确认翻译数组长度与本批条目数完全相同后，按索引合并回原字幕。
5. 通过 `writer.py` 写入目标文件。

`EdgeTranslator.translate_batch()` 会跳过空字符串，向接口发送非空条目；收到结果后再放回原索引。因此输入 `['第一条', '', '第三条']` 会只发送两条文本，但返回结果仍是三项，空条目保持为空。

翻译管线开始时只提交一个批次，用于先验证提供方的边界协议。该批次成功后，最多按 `TranslateConfig.max_workers` 并发翻译；默认值为 2。两批提交之间默认间隔 3 秒（`request_delay`），用于降低被限流的概率。

## 5. 限流、重试与失败行为

`src/translator/edge.py` 对每次接口调用采取以下策略：

| 场景 | 行为 |
| --- | --- |
| 网络异常（`requests.RequestException`） | 最多重试 3 次，即最多尝试 4 次。退避间隔为 1、2、4 秒。 |
| HTTP 429 | 最多重试 3 次；按 1、2、4 秒指数退避，并通过类级锁让同进程所有 `EdgeTranslator` 实例共用冷却时间。 |
| 其他非 2xx 状态 | 立即抛出 `TranslationError`，消息最多包含响应文本的前 200 个字符。 |
| 响应结构或条目数量异常 | 立即抛出 `TranslationError`。 |
| 一个批次失败 | 管线停止提交后续批次，取消尚未开始的任务，等待已开始任务结束后再把异常返回给调用方。 |

接口地址属于 Web 端接口，并非本项目持有的稳定 API 契约。若服务端调整请求或响应格式，翻译可能失败；请优先查看日志中的 HTTP 状态码或 `Invalid response format from Edge Translator` 信息。

## 6. 调用方式

### 6.1 Python：翻译一个 SRT

```python
from src.translator import EdgeTranslator, translate_srt

output_path = translate_srt(
    "demo.srt",
    "zh",
    provider=EdgeTranslator(),
    source_lang="ja",
)
print(output_path)
```

默认输出为同目录的双语字幕：`demo.chs.srt`。每条字幕的格式是原文在前、译文在后。

若需要项目默认的三文件布局，使用 `translate_srt_with_outputs()`：

```python
from src.translator import EdgeTranslator, translate_srt_with_outputs

bilingual_path, bilingual_output, original_path = translate_srt_with_outputs(
    "demo.srt",
    "zh",
    provider=EdgeTranslator(),
    source_lang="ja",
    swap_subtitles=True,
)

print(bilingual_path, bilingual_output, original_path)
```

此函数返回 `(translated_or_bilingual, bilingual, original)`。在 `swap_subtitles=True` 时：

| 文件 | 内容 |
| --- | --- |
| `demo.srt` | 单语译文；原文件会被覆盖。 |
| `demo.bilingual.srt` | 双语字幕，原文后紧跟译文。 |
| `demo.jpn.srt` | 原始日语字幕备份。 |

当 `swap_subtitles=False` 时，只生成 `demo.chs.srt` 双语字幕，不覆盖 `demo.srt`，并且返回元组的后两个路径为 `None`。

### 6.2 CLI：转写后自动翻译

```powershell
uv run python -m src.main `
  --input .\videos `
  --output .\output `
  --language ja `
  --translate zh `
  --translator bing
```

`--translator bing` 是 Edge 后端的选择值，也是默认值。`--translate` 为空或未指定时不调用翻译模块。

CLI 配置优先级为：命令行参数 > `config.yaml` > 代码默认值。对应的 YAML 配置为：

```yaml
output_formats:
  - srt
translate_to: "zh"
translation_provider: "bing"
translation_proxy: ""
swap_subtitles: true
```

翻译必须启用 `srt` 输出。CLI 转写流程在 `config.language == "auto"` 时会将翻译源语言回退为 `ja`，以满足 Edge 的显式源语言要求。因此，非日语视频应显式传入实际源语言，例如 `--language en`；否则该回退会导致翻译语言不准确。

### 6.3 GUI：转写或直接翻译 SRT

在 GUI 中执行以下操作：

1. 勾选翻译。
2. 在“后端”选择 `Microsoft Edge Translator (免费)`。
3. 在“识别语言”选择实际源语言，不能选择“自动”。
4. 在“目标语言”选择目标语言，并保证与源语言不同。
5. 保持 SRT 输出已勾选，然后开始任务。

GUI 将该后端映射为 `bing`，并在开始前校验：Edge 后端且源语言为 `auto` 时会显示警告并阻止任务启动。直接拖入 `.srt` 文件时，工作线程调用 `translate_srt_with_outputs()`，不经过 ASR 转写阶段。

### 6.4 在应用代码中按配置创建后端

业务代码不应自行判断 `bing` 后端的实现细节，使用工厂函数即可：

```python
from src.translator import create_translator, translate_srt_with_outputs

provider = create_translator("bing")
translated_path, bilingual_path, original_path = translate_srt_with_outputs(
    srt_path="demo.srt",
    target_lang="zh",
    provider=provider,
    source_lang="ja",
    swap_subtitles=True,
)
```

`proxy` 参数只用于 `gtx` 后端，Edge 翻译模块不会读取 `translation_proxy`。

## 7. 常见问题排查

| 现象 | 原因与处理 |
| --- | --- |
| 提示需要显式源语言 | 将 `source_lang`、`--language` 或 GUI 的“识别语言”设置为实际语言，不要使用 `auto`。 |
| HTTP 429 或日志出现 `rate-limited` | 接口限流。模块会自动退避重试；持续发生时降低任务频率，稍后重试。 |
| `Invalid response format` | Edge Web 接口的返回格式不符合当前实现预期。保留日志并检查 `src/translator/edge.py` 中的解析逻辑是否仍匹配服务端响应。 |
| 翻译后字幕错位 | 正常实现会拒绝条目数不一致的响应。若确实出现错位，应检查是否有绕过 `translate_srt()` 的自定义调用。 |
| 原始 `.srt` 被覆盖 | `swap_subtitles=true` 时这是预期行为；原文副本保存为源语言播放器后缀，例如 `.jpn.srt`。如不希望覆盖，传入 `swap_subtitles=False`。 |

## 8. 自动化测试覆盖

`tests/test_edge_translator.py` 覆盖以下契约：

- 请求地址、查询参数、JSON 请求体和简体中文语言映射；
- 批量请求保留每条字幕的边界与空条目的原始位置；
- 返回条目数量不匹配时拒绝写入；
- `auto` 源语言在发请求前被拒绝。

`tests/test_translation_pipeline.py` 进一步验证多行字幕不会导致批次错位、标准三文件输出规则，以及批次失败后不再继续提交后续字幕。

运行相关测试：

```powershell
uv run pytest tests/test_edge_translator.py tests/test_translation_pipeline.py
```
