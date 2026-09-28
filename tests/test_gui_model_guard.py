from pathlib import Path
import re

import pytest

from src.config import PipelineConfig
from src.gui import (
    HF_BASE_URL,
    HF_MIRROR_BASE_URL,
    MODEL_PATH_ERROR_MARKER,
    MODEL_REPOS,
    MODEL_REQUIRED_FILES,
    MODEL_SIZES,
    AsrWindow,
)


def test_config_error_names_the_missing_model_path(tmp_path: Path):
    """The GUI parses this message to tell users where to put the download."""
    missing = tmp_path / "does-not-exist"
    with pytest.raises(ValueError) as caught:
        # workers stays mandatory: config.yaml ships it as null and build()
        # falls over on None before validation unless the caller supplies it.
        PipelineConfig.build(
            {"config": "./config.yaml", "model_path": str(missing), "workers": 1}
        )

    match = re.search(rf"{MODEL_PATH_ERROR_MARKER}:\s*(.+)", str(caught.value))
    assert match is not None, "config.py no longer names the missing model path"
    assert Path(match.group(1).strip()) == missing


def test_every_model_size_offers_a_download_page():
    """Adding a model size without mapping its repo breaks the missing-model flow."""
    for size in MODEL_SIZES:
        assert size in MODEL_REPOS, f"{size} has no download repository"
        repo = MODEL_REPOS[size]
        assert f"{HF_BASE_URL}/{repo}"
        assert f"{HF_MIRROR_BASE_URL}/{repo}"


def test_missing_files_detects_empty_then_partial_then_complete(tmp_path: Path):
    model_dir = tmp_path / "faster-whisper-medium-ct2"
    assert AsrWindow._model_missing_files(model_dir) == list(MODEL_REQUIRED_FILES)

    model_dir.mkdir()
    assert AsrWindow._model_missing_files(model_dir) == list(MODEL_REQUIRED_FILES)

    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    assert AsrWindow._model_missing_files(model_dir) == ["model.bin"]

    (model_dir / "model.bin").write_bytes(b"\x00")
    assert AsrWindow._model_missing_files(model_dir) == []


def test_models_folder_placeholder_is_documented():
    readme = Path(__file__).resolve().parents[1] / "models" / "README.md"
    content = readme.read_text(encoding="utf-8")
    assert "hf-mirror" in content, "placeholder must mention the mirror for CN users"
    for size in MODEL_SIZES:
        assert f"faster-whisper-{size}-ct2" in content, f"missing dir name for {size}"
