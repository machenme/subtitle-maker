"""Local Hy-MT2 GGUF translation backend (llama.cpp via llama-cpp-python).

Model: tencent/Hy-MT2-1.8B-GGUF (HunyuanMT2 1.8B translator, quantized GGUF).

Usage notes (from the model card):
- The model has **no default system prompt** — send a single user message.
- Recommended sampling: temperature=0.7, top_p=0.6, top_k=20,
  repetition_penalty=1.05, max_tokens=4096.
- Prompt template: translate instruction + source text, output only the
  translation.
"""
from __future__ import annotations

import gc
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path

from src.translator.types import TranslationError

logger = logging.getLogger(__name__)

# Directory (or file) that holds the GGUF weights.
DEFAULT_MODEL_PATH = "./models/hy-mt2-1.8b-guff"

# Recommended generation parameters for the 1.8B model (see model card).
RECOMMENDED_SAMPLING: dict[str, float] = {
    "temperature": 0.7,
    "top_p": 0.6,
    "top_k": 20,
    "repeat_penalty": 1.05,
    "max_tokens": 4096,
}

# Full language names required by the Hy-MT2 prompt template.
_LANGUAGE_NAMES: dict[str, str] = {
    "zh": "中文",
    "zh-cn": "中文",
    "zh-hans": "中文",
    "zh-hant": "繁体中文",
    "zh-tw": "繁体中文",
    "en": "英语",
    "ja": "日语",
    "ko": "韩语",
    "fr": "法语",
    "de": "德语",
    "es": "西班牙语",
    "pt": "葡萄牙语",
    "it": "意大利语",
    "ru": "俄语",
    "ar": "阿拉伯语",
    "th": "泰语",
    "vi": "越南语",
}

_PROMPT_TEMPLATE = (
    "将以下{source_part}文本翻译为{target_name}，"
    "注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
)

# Multi-line batch prompt. Each line is numbered ("1. text") and the model
# must echo the numbers, which makes the response parseable line-by-line and
# prevents instruction echo from leaking into the first translation.
_MULTI_PROMPT_TEMPLATE = (
    "请将下面每一条编号文本分别翻译为{target_name}，"
    "输出时保留相同的编号，每条一行，"
    "只输出编号和翻译结果，不要输出任何解释：\n\n{text}"
)

# Numbered-line format used to pack a group into one completion.
_LINE_TEMPLATE = "{number}. {text}"
# Parses "12. translated text" (tolerating CJK punctuation after the dot).
_RESPONSE_LINE_RE = re.compile(r"^\s*(\d+)\s*[.、．)）]\s*(.*)$")
# Cap on prompt size per completion, in tokens (measured with the model's
# own tokenizer, not estimated). ~20000 input tokens + 4096 output tokens
# fits a 32768 context with headroom for the instruction template.
_DEFAULT_MAX_BATCH_TOKENS = 20000
# Per-line overhead in tokens: the "N. " number prefix, the newline, and
# tokenizer boundary effects between packed lines.
_LINE_TOKEN_OVERHEAD = 10
_DEFAULT_N_CTX = 32768
# Tokens reserved inside the context for the generated translation plus the
# instruction template (model card recommends max_tokens=4096).
_OUTPUT_RESERVE_TOKENS = 4352
# n_ctx candidates tried in descending order; smaller ones are used when the
# GPU cannot fit the KV cache of the full context.
_N_CTX_CANDIDATES = (32768, 16384, 8192, 4096)
# Fallback KV-cache estimate (bytes/token, f16 K+V) when model metadata is
# unavailable — conservative value for a 1.8B-class model.
_FALLBACK_KV_BYTES_PER_TOKEN = 32 * 1024
# VRAM headroom kept for the CUDA context, compute buffers and activations,
# on top of weights + KV cache. Includes ~1GB extra safety margin so small
# overruns (spikes in compute-buffer usage, desktop compositor, etc.) never
# spill into other processes or trigger driver-side eviction.
_VRAM_OVERHEAD_BYTES = 1700 * 1024 * 1024
# How long to wait for ASR worker VRAM to be released after their processes
# exit (Windows driver release is asynchronous).
_VRAM_RELEASE_TIMEOUT_S = 60.0
_VRAM_RELEASE_POLL_S = 1.0


def _language_name(language: str) -> str:
    normalized = (language or "").strip().lower()
    return _LANGUAGE_NAMES.get(normalized, normalized)


def _parse_numbered_lines(response: str, expected: int) -> list[str] | None:
    """Extract "N. translation" lines from a packed model response.

    Returns a list of *expected* translations (index 0 = line "1"), or
    ``None`` when any number is missing, duplicated, or empty — the caller
    then falls back to per-line translation. Unnumbered prose (e.g. echoed
    instructions) is ignored, which is exactly what we want when the model
    echoes part of the prompt.
    """
    translations: dict[int, str] = {}
    for raw_line in response.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _RESPONSE_LINE_RE.match(line)
        if not match:
            continue  # ignore instruction echo / stray prose
        number = int(match.group(1))
        text = match.group(2).strip()
        if not 1 <= number <= expected or not text:
            continue
        translations.setdefault(number, text)
    if len(translations) != expected:
        return None
    return [translations[n] for n in range(1, expected + 1)]


class LlmTranslator:
    """Translate subtitle text with a local Hy-MT2 GGUF model via llama-cpp-python.

    The model weights are loaded lazily on first use and kept resident for the
    process lifetime. Generation is serialized with a lock because a single
    llama.cpp context is not thread-safe.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        n_gpu_layers: int = -1,
        n_ctx: int | None = None,
        max_batch_tokens: int = _DEFAULT_MAX_BATCH_TOKENS,
        verbose: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.n_gpu_layers = n_gpu_layers
        # None = pick the largest context the free VRAM can hold (falls back
        # to a safe default when VRAM cannot be queried).
        self.n_ctx = n_ctx
        self.requested_n_ctx = n_ctx
        self.max_batch_tokens = max_batch_tokens
        self.verbose = verbose
        self._llm = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # TranslationProvider interface
    # ------------------------------------------------------------------

    def translate(
        self, text: str, source_lang: str = "auto", target_lang: str = "zh"
    ) -> str:
        """Translate one text while retaining the provider interface."""
        if not text:
            return ""
        return self.translate_batch([text], source_lang, target_lang)[0]

    def translate_batch(
        self, texts: list[str], source_lang: str = "auto", target_lang: str = "zh"
    ) -> list[str]:
        """Translate a subtitle batch with the local model.

        Non-empty lines are packed into as few completions as possible
        (delimiter-preserving multi-line prompt) and split back afterwards;
        a line-count mismatch triggers a per-line fallback for that group.
        """
        if not texts:
            return []
        llm = self._ensure_model()
        target_name = _language_name(target_lang)
        # The multi-line prompt relies on the delimiter staying intact, so it
        # is only used when the target language is known to the model card.
        use_packing = target_name in _LANGUAGE_NAMES.values()
        if not use_packing:
            logger.info(
                "Target language %r lacks a localized name; falling back to "
                "per-line translation", target_lang,
            )

        results: list[str] = [""] * len(texts)
        groups: list[list[int]] = []  # groups of non-empty indices
        current: list[int] = []
        current_tokens = 0
        for index, text in enumerate(texts):
            if not text:
                continue
            line_tokens = self._count_tokens(text) + _LINE_TOKEN_OVERHEAD
            if current and current_tokens + line_tokens > self.max_batch_tokens:
                groups.append(current)
                current = []
                current_tokens = 0
            current.append(index)
            current_tokens += line_tokens
        if current:
            groups.append(current)

        with self._lock:
            for group in groups:
                if not use_packing or len(group) == 1:
                    for index in group:
                        results[index] = self._complete(
                            llm,
                            _PROMPT_TEMPLATE.format(
                                source_part="",
                                target_name=target_name,
                                text=texts[index],
                            ),
                        )
                    continue
                packed = "\n".join(
                    _LINE_TEMPLATE.format(number=n + 1, text=texts[i])
                    for n, i in enumerate(group)
                )
                translated = self._complete(
                    llm,
                    _MULTI_PROMPT_TEMPLATE.format(
                        target_name=target_name, text=packed
                    ),
                )
                parsed = _parse_numbered_lines(translated, len(group))
                if parsed is not None:
                    for n, index in enumerate(group):
                        results[index] = parsed[n]
                else:
                    logger.info(
                        "Packed translation missing/malformed line(s); retrying "
                        "per line",
                    )
                    for index in group:
                        results[index] = self._complete(
                            llm,
                            _PROMPT_TEMPLATE.format(
                                source_part="",
                                target_name=target_name,
                                text=texts[index],
                            ),
                        )
        logger.info(
            "Local model translation batch complete: %d subtitle(s) in %d "
            "request(s)", len(texts), len(groups),
        )
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _complete(self, llm, prompt: str) -> str:
        """Run one chat completion and return the cleaned translation."""
        try:
            output = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                **RECOMMENDED_SAMPLING,
            )
        except Exception as exc:  # llama_cpp raises bare exceptions
            raise TranslationError(f"Local model inference failed: {exc}") from exc
        try:
            content = output["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise TranslationError(
                f"Unexpected local model output structure: {output!r}"
            ) from exc
        if not isinstance(content, str):
            raise TranslationError("Local model returned non-text content")
        return self._clean(content)

    @staticmethod
    def _clean(text: str) -> str:
        """Strip stray whitespace/quotes the model may add around the answer."""
        cleaned = text.strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'`“”‘’":
            cleaned = cleaned[1:-1].strip()
        return cleaned

    def _ensure_model(self):
        """Load the GGUF model on first use (lazy, thread-safe enough via lock)."""
        if self._llm is not None:
            return self._llm
        _enable_cuda_dll_search()
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise TranslationError(
                "llama-cpp-python is not installed. "
                "Install it with: uv pip install llama-cpp-python"
            ) from exc
        except OSError as exc:
            raise TranslationError(
                f"llama-cpp-python failed to load its native library: {exc}"
            ) from exc

        gguf_path = _resolve_gguf(self.model_path)
        n_ctx = self._pick_n_ctx(gguf_path)
        # Derive the batch-token budget from the chosen context: input tokens
        # + generation reserve must always fit inside n_ctx. An explicitly
        # requested budget (anything other than the default) is honoured as-is.
        if self.max_batch_tokens == _DEFAULT_MAX_BATCH_TOKENS:
            self.max_batch_tokens = max(1024, n_ctx - _OUTPUT_RESERVE_TOKENS)
        logger.info(
            "Loading local translation model: %s (n_gpu_layers=%d, n_ctx=%d, "
            "max_batch_tokens=%d)",
            gguf_path, self.n_gpu_layers, n_ctx, self.max_batch_tokens,
        )
        try:
            self._llm = Llama(
                model_path=str(gguf_path),
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=n_ctx,
                verbose=self.verbose,
            )
        except Exception as exc:
            raise TranslationError(
                f"Failed to load local model {gguf_path}: {exc}"
            ) from exc
        logger.info("Local translation model loaded: %s", gguf_path)
        return self._llm

    def _pick_n_ctx(self, gguf_path: Path) -> int:
        """Choose the largest context size that fits into the free VRAM.

        An explicit ``n_ctx`` (constructor / max_batch_tokens caller) wins.
        Otherwise the KV-cache bytes-per-token is read from the GGUF metadata
        and compared against current free VRAM (weights + overhead deducted);
        when VRAM cannot be queried, the safest candidate is used.
        """
        if self.requested_n_ctx is not None:
            return self.requested_n_ctx

        free_bytes = _free_vram_bytes()
        if free_bytes is None:
            logger.info(
                "VRAM query unavailable; using conservative n_ctx=%d",
                _N_CTX_CANDIDATES[-1],
            )
            return _N_CTX_CANDIDATES[-1]

        weights_bytes = gguf_path.stat().st_size
        kv_bytes_per_token = _kv_bytes_per_token(gguf_path)
        usable = free_bytes - weights_bytes - _VRAM_OVERHEAD_BYTES
        for candidate in _N_CTX_CANDIDATES:
            if usable >= candidate * kv_bytes_per_token:
                logger.info(
                    "VRAM budget: free=%.1fGB weights=%.1fGB usable_for_kv=%.1fGB "
                    "→ n_ctx=%d",
                    free_bytes / 2**30, weights_bytes / 2**30,
                    usable / 2**30, candidate,
                )
                return candidate
        logger.info(
            "Free VRAM too small for GPU KV cache; using n_ctx=%d",
            _N_CTX_CANDIDATES[-1],
        )
        return _N_CTX_CANDIDATES[-1]

    def _count_tokens(self, text: str) -> int:
        """Count tokens with the loaded model's tokenizer (exact, no estimate)."""
        llm = self._ensure_model()
        try:
            return len(llm.tokenize(text.encode("utf-8"), special=False))
        except Exception:
            # Tokenizer hiccup: fall back to a conservative character estimate.
            return max(1, len(text) // 2)

    # ------------------------------------------------------------------
    # VRAM lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def minimum_free_vram_bytes() -> int:
        """VRAM needed to load the model with the smallest useful context.

        Used by callers to wait for ASR workers to release their memory
        before starting the local-model translation pass.
        """
        weights = _resolve_gguf(Path(DEFAULT_MODEL_PATH)).stat().st_size
        return weights + _VRAM_OVERHEAD_BYTES + _N_CTX_CANDIDATES[-1] * 1024

    def release(self) -> None:
        """Free the model and its VRAM/CPU memory immediately.

        llama.cpp keeps weights, KV cache and the CUDA context alive until
        the Python object is destroyed; drop it explicitly and force a GC so
        the translation pass does not hold VRAM after finishing.
        """
        with self._lock:
            llm, self._llm = self._llm, None
        if llm is not None:
            try:
                # Close the CUDA handle before dropping the last reference.
                close = getattr(llm, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
            del llm
        gc.collect()
        logger.info("Local translation model released (VRAM returned)")


def _enable_cuda_dll_search() -> None:
    """Make CUDA 12 runtime DLLs (from the pip nvidia-* wheels) importable.

    The CUDA build of llama-cpp-python links against cudart/cublas, which are
    not on PATH by default. The project already ships them via
    ``nvidia-cuda-runtime-cu12`` / ``nvidia-cublas-cu12`` for faster-whisper,
    so register those directories with os.add_dll_directory.
    """
    if sys.platform != "win32":
        return
    for bin_dir in (
        Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / "cuda_runtime" / "bin",
        Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin",
    ):
        if bin_dir.is_dir() and str(bin_dir) not in _registered_dll_dirs:
            try:
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = f"{bin_dir}{os.pathsep}" + os.environ.get("PATH", "")
                _registered_dll_dirs.add(str(bin_dir))
            except OSError:
                pass


_registered_dll_dirs: set[str] = set()


def _free_vram_bytes() -> int | None:
    """Return free VRAM on GPU 0 via nvidia-ml-py, or None when unavailable.

    The project already depends on ``nvidia-ml-py`` (same handle as
    :class:`src.monitor.GpuMonitor`); query it on demand instead of polling.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.free
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def wait_for_vram_release(
    required_free_bytes: int,
    *,
    timeout: float = _VRAM_RELEASE_TIMEOUT_S,
    poll_interval: float = _VRAM_RELEASE_POLL_S,
) -> bool:
    """Block until free VRAM reaches *required_free_bytes* (or timeout).

    ASR worker processes hold Whisper weights/CUDA contexts; after they exit,
    Windows releases their VRAM asynchronously. The local-model translation
    pass calls this before loading, so it never starts on stale numbers.
    Returns True when the threshold was observed, False on timeout.
    """
    deadline = time.monotonic() + timeout
    last_free: int | None = None
    while time.monotonic() < deadline:
        free = _free_vram_bytes()
        if free is None:
            return False
        if free >= required_free_bytes:
            return True
        last_free = free
        time.sleep(poll_interval)
    logger.warning(
        "VRAM release wait timed out after %.0fs (free=%.1fGB, wanted %.1fGB); "
        "proceeding anyway — context size will be reduced to fit",
        timeout, (last_free or 0) / 2**30, required_free_bytes / 2**30,
    )
    return False


def _kv_bytes_per_token(gguf_path: Path) -> int:
    """Estimate KV-cache bytes per token from GGUF metadata.

    Reads the actual head count / head size / layer count from the model
    file when possible; otherwise falls back to a conservative default for
    a 1.8B-class model.
    """
    default = _FALLBACK_KV_BYTES_PER_TOKEN
    try:
        import gguf  # shipped with llama-cpp-python

        reader = gguf.GGUFReader(str(gguf_path))
        fields = {k: v for k, v in reader.fields.items()}

        def _u32(name: str) -> int | None:
            field = fields.get(name)
            if field is None:
                return None
            value = field.parts[-1][0]
            return int(value)

        heads = _u32("{0}.attention.head_count".format(gguf_path.stem.upper()))
        head_dim = _u32("{0}.attention.key_length".format(gguf_path.stem.upper()))
        layers = _u32("{0}.block_count".format(gguf_path.stem.upper()))
        if not (heads and head_dim and layers):
            return default
        # f16 K and V: 2 (K+V) * 2 bytes * heads * head_dim * layers
        return 2 * 2 * heads * head_dim * layers
    except Exception:
        return default


def _resolve_gguf(model_path: Path) -> Path:
    """Accept either a .gguf file or a directory containing exactly one GGUF."""
    if model_path.is_file():
        return model_path
    if model_path.is_dir():
        candidates = sorted(model_path.glob("*.gguf"))
        if not candidates:
            raise TranslationError(
                f"No .gguf file found in model directory: {model_path}"
            )
        if len(candidates) > 1:
            # Prefer the highest quant (lexicographic Q8 first in most dumps).
            preferred = [p for p in candidates if "q8" in p.name.lower()]
            chosen = preferred[0] if preferred else candidates[0]
            logger.info(
                "Multiple GGUF files in %s, using %s", model_path, chosen.name
            )
            return chosen
        return candidates[0]
    raise TranslationError(f"Model path does not exist: {model_path}")


# Backwards-friendly alias used by the GUI labels and docs.
HyMT2Translator = LlmTranslator
