from src.gpu_scheduler import _STARTUP_MIN_FREE_VRAM_GB, _can_start_worker


def test_startup_probe_allows_unknown_vram():
    assert _can_start_worker(None, None)


def test_startup_probe_requires_minimum_free_vram():
    assert not _can_start_worker(_STARTUP_MIN_FREE_VRAM_GB - 0.01, None)
    assert _can_start_worker(_STARTUP_MIN_FREE_VRAM_GB, None)


def test_startup_probe_keeps_margin_for_previous_worker():
    previous_worker_vram_gb = 2.0

    assert not _can_start_worker(2.49, previous_worker_vram_gb)
    assert _can_start_worker(2.5, previous_worker_vram_gb)
