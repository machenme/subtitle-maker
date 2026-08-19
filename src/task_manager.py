"""
Task manager: job state tracking, checkpoint/resume logic,
and progress persistence to .progress.json.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone

from src.translator.types import iso_to_player_suffix
from src.utils import is_file_valid, is_srt_valid

logger = logging.getLogger(__name__)

# Read first + last N bytes for file fingerprint
_HEAD_TAIL_BYTES = 10 * 1024 * 1024  # 10 MB


def _file_fingerprint(path: Path) -> dict | None:
    """Return ``{size, mtime, head_tail_hash, duration}`` or None."""
    try:
        stat = path.stat()
        size = stat.st_size
        mtime = int(stat.st_mtime)

        dur = 0.0
        try:
            import subprocess as _sp
            r = _sp.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(path)],
                capture_output=True, text=True, timeout=15,
            )
            dur = float(r.stdout.strip())
        except Exception:
            pass

        if size == 0:
            return {"size": 0, "mtime": mtime, "head_tail_hash": "", "duration": 0}

        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read(_HEAD_TAIL_BYTES))
            if size > _HEAD_TAIL_BYTES * 2:
                f.seek(-_HEAD_TAIL_BYTES, 2)
                h.update(f.read(_HEAD_TAIL_BYTES))
        return {"size": size, "mtime": mtime, "head_tail_hash": h.hexdigest(), "duration": round(dur, 1)}
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A single video-to-text processing task."""
    video_path: Path
    audio_path: Path | None = None
    status: str = "pending"  # pending | extracting | transcribing | done | failed
    error_message: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    fingerprint: dict | None = None  # {size, mtime, head_hash} at completion


@dataclass
class ProgressSnapshot:
    """Serialisable progress state for resume."""
    total: int
    completed: list[str]   # list of video names (stems)
    failed: list[str]
    updated_at: str


# ---------------------------------------------------------------------------
# TaskManager
# ---------------------------------------------------------------------------

class TaskManager:
    """
    Manages task lifecycle, checkpointing, and resume logic.

    On startup, scans output_dir to build the DoneSet and filters
    already-processed videos from the work queue.
    """

    def __init__(
        self,
        output_dir: Path,
        progress_file: Path | None = None,
        *,
        output_formats: list[str] | None = None,
        translate_to: str = "",
        swap_subtitles: bool = True,
        source_lang: str = "auto",
    ):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._progress_file = progress_file or (self._output_dir / ".progress.json")
        self._tasks: dict[str, Task] = {}       # keyed by video stem
        self._done_set: set[str] = set()
        self._fingerprints: dict[str, dict] = {}  # stem → {size, mtime, head_hash}
        self._output_formats = set(output_formats or ["srt"])
        self._translate_to = translate_to
        self._swap_subtitles = swap_subtitles
        self._source_lang = source_lang

    # ------------------------------------------------------------------
    # Build work queue
    # ------------------------------------------------------------------

    def build_queue(self, video_paths: list[Path], force: bool = False) -> list[Task]:
        """
        Build the pending task list, filtering already-completed videos.

        Args:
            video_paths: All discovered video files.
            force: If True, re-process even if output exists.

        Returns:
            List of Task objects with status='pending' (needs processing).
        """
        # Load fingerprints, then inspect only outputs required by this run.
        self._load_progress()
        self._done_set = self._scan_completed(video_paths)

        tasks: list[Task] = []
        for vp in video_paths:
            stem = vp.stem
            if stem in self._tasks:
                raise ValueError(
                    f"Duplicate video stem '{stem}' is not supported in one batch: {vp}"
                )
            task = Task(video_path=vp)
            self._tasks[stem] = task

            if not force and stem in self._done_set:
                # Verify fingerprint: skip only if file hasn't changed
                fp = _file_fingerprint(vp)
                if fp and self._verify_fingerprint(stem, fp):
                    task.status = "done"
                    logger.info(f"Skip (already done): {stem}")
                else:
                    logger.info(f"File changed, re-processing: {stem}")
                    tasks.append(task)
            else:
                tasks.append(task)

        return tasks

    def _scan_completed(self, video_paths: list[Path]) -> set[str]:
        """Return stems whose outputs satisfy the current run configuration."""
        done: set[str] = set()
        for video_path in video_paths:
            stem = video_path.stem
            if self._outputs_complete(stem, self._output_dir):
                done.add(stem)
                continue
            # Preserve support for the documented per-video layout.
            nested_dir = self._output_dir / stem
            if self._outputs_complete(stem, nested_dir):
                done.add(stem)

        return done

    def _expected_output_names(self, stem: str) -> list[str]:
        names = [f"{stem}.{fmt}" for fmt in self._output_formats]
        if self._translate_to and "srt" in self._output_formats:
            target_suffix = iso_to_player_suffix(self._translate_to)
            source_lang = self._source_lang if self._source_lang != "auto" else "ja"
            source_suffix = iso_to_player_suffix(source_lang)
            if self._swap_subtitles:
                names.extend([
                    f"{stem}.srt",
                    f"{stem}.bilingual.srt",
                    f"{stem}.{source_suffix}.srt",
                ])
            else:
                names.append(f"{stem}.{target_suffix}.srt")
        return sorted(set(names))

    def _outputs_complete(self, stem: str, directory: Path) -> bool:
        for name in self._expected_output_names(stem):
            path = directory / name
            if path.suffix.lower() == ".srt":
                if not is_srt_valid(path):
                    return False
            elif not is_file_valid(path, min_bytes=0):
                return False
        return True

    # ------------------------------------------------------------------
    # Progress persistence
    # ------------------------------------------------------------------

    def save_progress(self) -> None:
        """Write current progress to .progress.json."""
        completed = [
            stem for stem, t in self._tasks.items() if t.status == "done"
        ]
        failed = [
            stem for stem, t in self._tasks.items() if t.status == "failed"
        ]
        fingerprints = {
            stem: t.fingerprint
            for stem, t in self._tasks.items()
            if t.fingerprint
        }
        snap = {
            "total": len(self._tasks),
            "completed": completed,
            "failed": failed,
            "fingerprints": fingerprints,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            temp_path = self._progress_file.with_suffix(".json.tmp")
            temp_path.write_text(
                json.dumps(snap, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temp_path.replace(self._progress_file)
        except OSError as exc:
            logger.warning(f"Failed to save progress: {exc}")

    def _load_progress(self) -> ProgressSnapshot | None:
        """Load previous progress file if it exists."""
        if not self._progress_file.exists():
            return None
        try:
            data = json.loads(self._progress_file.read_text(encoding="utf-8"))
            # Load fingerprints (new field)
            fps = data.pop("fingerprints", {})
            if isinstance(fps, dict):
                self._fingerprints.update(fps)
            return ProgressSnapshot(**data)
        except (json.JSONDecodeError, TypeError, KeyError, AttributeError) as exc:
            logger.warning(f"Progress file corrupted, ignoring: {exc}")
            return None

    def _verify_fingerprint(self, stem: str, current: dict) -> bool:
        """True if *current* fingerprint matches the stored one for *stem*."""
        stored = self._fingerprints.get(stem)
        if not stored:
            return False
        return (
            stored.get("size") == current.get("size")
            and stored.get("mtime") == current.get("mtime")
            and stored.get("head_tail_hash") == current.get("head_tail_hash")
            and stored.get("duration") == current.get("duration")
        )

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def mark_started(self, video_path: Path, status: str = "transcribing") -> None:
        stem = video_path.stem
        if stem in self._tasks:
            self._tasks[stem].status = status
            self._tasks[stem].started_at = datetime.now(timezone.utc).isoformat()

    def mark_done(self, video_path: Path) -> None:
        stem = video_path.stem
        if stem in self._tasks:
            self._tasks[stem].status = "done"
            self._tasks[stem].finished_at = datetime.now(timezone.utc).isoformat()
            self._tasks[stem].fingerprint = _file_fingerprint(video_path)

    def mark_failed(self, video_path: Path, error: str) -> None:
        stem = video_path.stem
        if stem in self._tasks:
            self._tasks[stem].status = "failed"
            self._tasks[stem].error_message = error
            self._tasks[stem].finished_at = datetime.now(timezone.utc).isoformat()
        logger.error(f"Task failed [{stem}]: {error}")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def done_count(self) -> int:
        return sum(1 for t in self._tasks.values() if t.status == "done")

    @property
    def failed_count(self) -> int:
        return sum(1 for t in self._tasks.values() if t.status == "failed")

    @property
    def total_count(self) -> int:
        return len(self._tasks)

    def summary(self) -> str:
        return (
            f"Tasks: {self.done_count}/{self.total_count} done"
            + (f", {self.failed_count} failed" if self.failed_count else "")
        )
