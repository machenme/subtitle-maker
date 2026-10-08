import threading
import time

from src.translator.llm import LlmTranslator


def _bare_translator() -> LlmTranslator:
    """Build a translator without touching the filesystem or the GPU."""
    translator = LlmTranslator.__new__(LlmTranslator)
    translator._llm = None
    translator._lock = threading.Lock()
    translator._load_lock = threading.Lock()
    translator.model_path = "."
    translator.n_gpu_layers = -1
    translator.n_ctx = None
    translator.requested_n_ctx = None
    translator.max_batch_tokens = 20000
    translator.verbose = False
    return translator


def test_concurrent_ensure_model_loads_weights_only_once(monkeypatch):
    """Several threads racing into _ensure_model must not each load the model.

    Loading a GGUF pulls multiple GB into VRAM; a duplicate load would waste
    it and could OOM the GPU.
    """
    translator = _bare_translator()
    loads: list[str] = []

    def fake_load(self):
        loads.append(threading.current_thread().name)
        time.sleep(0.05)
        self._llm = object()
        return self._llm

    monkeypatch.setattr(LlmTranslator, "_load_model_locked", fake_load)

    barrier = threading.Barrier(4)

    def worker():
        barrier.wait()
        translator._ensure_model()

    threads = [threading.Thread(target=worker, name=f"T{i}") for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert len(loads) == 1, f"model loaded {len(loads)} times: {loads}"
    assert translator._llm is not None


def test_ensure_model_reuses_loaded_model(monkeypatch):
    """After the first load the fast path must not call the loader again."""
    translator = _bare_translator()
    loads: list[str] = []

    def fake_load(self):
        loads.append("load")
        self._llm = object()
        return self._llm

    monkeypatch.setattr(LlmTranslator, "_load_model_locked", fake_load)

    first = translator._ensure_model()
    second = translator._ensure_model()

    assert first is second
    assert loads == ["load"]


def test_generation_lock_is_not_held_while_loading(monkeypatch):
    """Loading must not require self._lock, which is not reentrant.

    translate_batch() tokenizes (and therefore loads) before it takes the
    generation lock; a shared lock there would deadlock.
    """
    translator = _bare_translator()

    def fake_load(self):
        self._llm = object()
        return self._llm

    monkeypatch.setattr(LlmTranslator, "_load_model_locked", fake_load)

    assert not translator._lock.locked()
    translator._ensure_model()
    assert not translator._lock.locked(), "loading must not touch the generation lock"
