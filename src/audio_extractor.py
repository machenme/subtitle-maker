"""
Stage 1: Audio extraction via ffmpeg subprocess.
Extracts audio streams from media files → 16kHz Mono 16-bit PCM WAV.
"""
from __future__ import annotations

import queue
import re
import subprocess
import logging
import hashlib
import threading
import time
from pathlib import Path

from src.utils import find_executable, is_file_valid

logger = logging.getLogger(__name__)

# ffmpeg -progress emits key=value pairs; out_time_us is the encoded position.
_OUT_TIME_RE = re.compile(r"^out_time_us=(\d+)", re.MULTILINE)
# Emit at most this often while streaming ffmpeg progress.
_PROGRESS_INTERVAL = 0.2
# How often the supervisor loop wakes up to check the deadline / cancel event.
_POLL_INTERVAL = 0.1
# Grace period for ffmpeg to exit on its own once its output stream closes.
_EXIT_GRACE_SECONDS = 5


class AudioExtractionError(Exception):
    """Raised when ffmpeg fails to extract audio from a media file."""


class AudioExtractionCancelled(AudioExtractionError):
    """Raised when a cancel event stops an in-flight ffmpeg run.

    Subclasses :class:`AudioExtractionError` so existing handlers keep working,
    but callers can distinguish "the user stopped this" from a media failure
    instead of recording it as a processing error.
    """


def _as_explicit_path(value: str | None) -> str | None:
    """Keep only a path-like value; a bare "ffmpeg" means "look it up"."""
    if not value:
        return None
    return value if ("/" in value or "\\" in value) else None


def _drain_text(stream) -> str:
    """Read a pipe to EOF, tolerating a pipe closed concurrently by cleanup."""
    try:
        return stream.read() or ""
    except (OSError, ValueError):
        return ""


class _FFmpegResult:
    """Minimal stand-in for ``CompletedProcess`` returned by :meth:`_run_ffmpeg`."""

    __slots__ = ("returncode", "stdout", "stderr")

    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class AudioExtractor:
    """
    Extracts audio from media files using ffmpeg (subprocess).

    Output format: 16 kHz, mono, 16-bit PCM WAV (Whisper native format).
    """

    def __init__(
        self,
        temp_dir: Path,
        ffmpeg_bin: str | None = None,
        ffprobe_bin: str | None = None,
    ):
        self._temp_dir = Path(temp_dir)
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        # A bare "ffmpeg" means "look it up"; only a path-like value is used
        # verbatim. Resolution is deferred to first use so FileNotFoundError
        # surfaces inside extract(), where callers already handle it.
        self._configured_ffmpeg = _as_explicit_path(ffmpeg_bin)
        self._configured_ffprobe = _as_explicit_path(ffprobe_bin)
        self._ffmpeg: str | None = None
        self._ffprobe: str | None = None

    @property
    def ffmpeg(self) -> str:
        """Runnable ffmpeg path (PATH → registry PATH → known install dirs)."""
        if self._ffmpeg is None:
            self._ffmpeg = find_executable("ffmpeg", self._configured_ffmpeg)
            logger.info(f"Using ffmpeg: {self._ffmpeg}")
        return self._ffmpeg

    @property
    def ffprobe(self) -> str:
        """Runnable ffprobe path, resolved the same way as :attr:`ffmpeg`."""
        if self._ffprobe is None:
            self._ffprobe = find_executable("ffprobe", self._configured_ffprobe)
        return self._ffprobe

    # ------------------------------------------------------------------
    # Cache path
    # ------------------------------------------------------------------

    def wav_path_for(self, video_path: Path, output_dir: Path | None = None) -> Path:
        """Return the cache path this source maps to, without running ffmpeg.

        The name is content-keyed (path + size + mtime), so re-encoding a
        source yields a different file and never reuses stale audio. Callers
        use it to clean up a WAV that extraction abandoned mid-write.
        """
        dest_dir = output_dir or self._temp_dir
        try:
            stat = Path(video_path).stat()
        except OSError as exc:
            raise AudioExtractionError(
                f"Cannot read source media: {video_path}"
            ) from exc
        source_key = (
            f"{Path(video_path).resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}"
        ).encode("utf-8")
        source_hash = hashlib.sha1(source_key).hexdigest()[:10]
        return dest_dir / f"{Path(video_path).stem}.{source_hash}.wav"

    # ------------------------------------------------------------------
    # ffmpeg invocation
    # ------------------------------------------------------------------

    def _run_ffmpeg(
        self,
        cmd: list[str],
        *,
        timeout: int,
        total_seconds: float = 0.0,
        progress_callback=None,
        cancel_event=None,
    ) -> _FFmpegResult:
        """Run ffmpeg, optionally streaming 0..1 progress to a callback.

        Without a callback and without a cancel event this stays a plain
        ``subprocess.run``. Otherwise ``-progress pipe:1`` is added and the
        stream is supervised from this thread: extraction is often the longest
        single step, and a frozen bar there reads as a hang.

        Both pipes are drained by dedicated reader threads and handed to this
        loop through queues. Reading them inline can block forever, which
        previously made the configured ``timeout`` unenforceable whenever
        ffmpeg stopped emitting progress. Whatever happens here — timeout,
        cancellation, or an exception from ``progress_callback`` — the child
        process is terminated and its pipes reclaimed before returning.
        """
        if progress_callback is None and cancel_event is None:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return _FFmpegResult(
                result.returncode,
                getattr(result, "stdout", "") or "",
                getattr(result, "stderr", "") or "",
            )

        proc = subprocess.Popen(
            [*cmd[:-1], "-progress", "pipe:1", "-nostats", cmd[-1]],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        stderr_lines: list[str] = []
        # ffmpeg blocks on a full stderr pipe, so it must be drained while we
        # read progress off stdout.
        stderr_reader = threading.Thread(
            target=lambda: stderr_lines.extend(_drain_text(proc.stderr).splitlines()),
            daemon=True,
        )
        stderr_reader.start()

        # Progress must also come off a queue: iterating proc.stdout directly
        # parks this thread inside a blocking read with no way to notice that
        # the deadline passed or that cancellation was requested.
        progress_lines: queue.Queue[str | None] = queue.Queue()

        def _read_stdout() -> None:
            try:
                for line in proc.stdout:
                    progress_lines.put(line)
            except (OSError, ValueError):
                pass
            finally:
                progress_lines.put(None)

        stdout_reader = threading.Thread(target=_read_stdout, daemon=True)
        stdout_reader.start()

        deadline = time.monotonic() + timeout
        seen: list[str] = []
        last_emit = 0.0
        timed_out = False
        cancelled = False
        callback_error: BaseException | None = None
        try:
            # ffmpeg's first status block can already be near the end on a
            # short clip, so pin an explicit 0% to guarantee a visible start.
            if progress_callback:
                progress_callback(0.0)

            while True:
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break
                now = time.monotonic()
                if now >= deadline:
                    timed_out = True
                    break
                try:
                    line = progress_lines.get(timeout=_POLL_INTERVAL)
                except queue.Empty:
                    continue
                if line is None:  # stdout closed → ffmpeg is done writing
                    break
                seen.append(line)
                if "=" not in line:
                    continue
                now = time.monotonic()
                if now - last_emit < _PROGRESS_INTERVAL:
                    continue
                match = _OUT_TIME_RE.search("".join(seen[-40:]))
                if match:
                    last_emit = now
                    encoded = int(match.group(1)) / 1_000_000
                    if progress_callback:
                        try:
                            progress_callback(
                                min(1.0, encoded / total_seconds)
                                if total_seconds > 0
                                else 0.0
                            )
                        except Exception as exc:
                            # Never leave ffmpeg running behind a broken UI.
                            callback_error = exc
                            break

            if not (timed_out or cancelled or callback_error is not None):
                try:
                    proc.wait(timeout=_EXIT_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    timed_out = True
        finally:
            # Single cleanup path: a live process is always killed and waited
            # for, then the pipes are closed and the readers collected.
            try:
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=_EXIT_GRACE_SECONDS)
            except (subprocess.TimeoutExpired, OSError):
                logger.warning("ffmpeg process did not exit cleanly after kill")
            stdout_reader.join(timeout=2)
            stderr_reader.join(timeout=2)
            for pipe in (proc.stdout, proc.stderr):
                try:
                    pipe.close()
                except Exception:
                    pass

        if callback_error is not None:
            raise callback_error
        if cancelled:
            raise AudioExtractionCancelled("ffmpeg run cancelled by request")
        if timed_out:
            raise subprocess.TimeoutExpired(cmd, timeout)
        return _FFmpegResult(proc.returncode, "", "\n".join(stderr_lines))

    # ------------------------------------------------------------------
    # Single file
    # ------------------------------------------------------------------

    def extract(
        self,
        video_path: Path,
        output_dir: Path | None = None,
        *,
        progress_callback=None,
        cancel_event=None,
    ) -> Path:
        """
        Extract audio from a single video file.

        Args:
            video_path: Source video file path.
            output_dir: Directory for the output WAV (defaults to self._temp_dir).
            progress_callback: Optional ``(fraction: float) -> None`` invoked
                while ffmpeg runs. The source duration is probed only when a
                callback is supplied, so the plain path pays nothing for it.
            cancel_event: Optional ``threading.Event``; when set, the running
                ffmpeg process is terminated and
                :class:`AudioExtractionCancelled` is raised.

        Returns:
            Path to the extracted WAV file.

        Raises:
            AudioExtractionError: ffmpeg call failed.
            AudioExtractionCancelled: the caller requested cancellation.
        """
        dest_dir = output_dir or self._temp_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        if cancel_event is not None and cancel_event.is_set():
            raise AudioExtractionCancelled("Cancelled before extraction")

        wav_path = self.wav_path_for(video_path, dest_dir)

        # Skip if already extracted and valid
        if is_file_valid(wav_path, min_bytes=1024):
            logger.info(f"Audio already extracted: {wav_path}")
            if progress_callback:
                progress_callback(1.0)
            return wav_path

        logger.info(f"Extracting audio: {video_path.name} → {wav_path.name}")

        try:
            cmd = [
                self.ffmpeg,                   # resolved lazily (raises if missing)
                "-y",                          # overwrite
                "-threads", "1",               # single-threaded: better error resilience
                "-err_detect", "ignore_err",   # tolerate corrupt packets
                "-fflags", "+genpts+discardcorrupt",  # timestamps regen, skip broken frames
                "-i", str(video_path),
                "-vn",                         # no video
                "-c:a", "pcm_s16le",           # 16-bit PCM (re-encode, not stream copy)
                "-ar", "16000",                # 16 kHz
                "-ac", "1",                    # mono
                "-af", "aresample=async=1",    # fill gaps from corrupt frames with silence
                "-loglevel", "error",          # suppress ffmpeg output except errors
                str(wav_path),
            ]

            total_seconds = (
                self.get_video_duration(video_path) if progress_callback else 0.0
            )
            result = self._run_ffmpeg(
                cmd,
                timeout=600,
                total_seconds=total_seconds,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
            if result.returncode != 0:
                # ffmpeg exits non-zero on decode errors (corrupt source).
                # If output file exists and is usable, treat as partial success.
                if is_file_valid(wav_path, min_bytes=1024):
                    logger.warning(
                        f"Audio extracted with decode errors: {video_path.name} "
                        f"(output {wav_path.stat().st_size} bytes)"
                    )
                    # If the output duration is suspiciously short, try raw-AAC fallback.
                    actual_dur = self.get_duration(wav_path)
                    expected_dur = self.get_video_duration(video_path)
                    if expected_dur > 0 and actual_dur < expected_dur * 0.5:
                        logger.warning(
                            f"Extracted audio too short ({actual_dur:.0f}s vs {expected_dur:.0f}s). "
                            f"Trying raw-AAC fallback..."
                        )
                        return self._extract_via_raw_aac(video_path, wav_path)
                else:
                    # First attempt failed entirely; try raw-AAC fallback before giving up.
                    try:
                        return self._extract_via_raw_aac(video_path, wav_path)
                    except AudioExtractionError:
                        stderr = result.stderr.strip()
                        raise AudioExtractionError(
                            f"ffmpeg failed for {video_path.name}: {stderr}"
                        )
        except subprocess.TimeoutExpired:
            raise AudioExtractionError(f"ffmpeg timed out for {video_path.name}")
        except FileNotFoundError as exc:
            # Windows keeps the PATH a process started with: an ffmpeg installed
            # later is invisible to a running app even though `where ffmpeg`
            # works in every new shell. find_executable() also checks the
            # persisted PATH and the usual install locations, so this only
            # fires when the binary genuinely is not there.
            raise AudioExtractionError(str(exc))

        if not is_file_valid(wav_path, min_bytes=1024):
            raise AudioExtractionError(f"Output WAV is missing or too small: {wav_path}")

        if progress_callback:
            progress_callback(1.0)
        return wav_path

    # ------------------------------------------------------------------
    # Audio duration
    # ------------------------------------------------------------------

    def get_duration(self, path: Path) -> float:
        """Return media duration in seconds via ffprobe."""
        cmd = [
            self.ffprobe, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return float(result.stdout.strip())
        except (FileNotFoundError, OSError, ValueError, subprocess.TimeoutExpired):
            logger.warning(f"Could not determine duration for {path.name}, assuming 0")
            return 0.0

    def get_video_duration(self, video_path: Path) -> float:
        """Return the *container-level* video duration (may differ from decoded audio)."""
        return self.get_duration(video_path)

    # ------------------------------------------------------------------
    # Raw AAC fallback (for corrupt audio streams)
    # ------------------------------------------------------------------

    def _extract_via_raw_aac(self, video_path: Path, wav_path: Path) -> Path:
        """
        Two-step extraction for corrupt AAC streams:
        1. Stream-copy raw AAC out of the container (avoids decoder).
        2. Decode the raw AAC with error tolerance.

        Stripping the MP4/MKV container often lets the AAC decoder
        recover from errors that were fatal inside the container.
        """
        aac_path = wav_path.with_suffix(".aac")
        logger.info(f"Step 1/2: extracting raw AAC stream: {aac_path.name}")

        # Step 1: extract raw AAC
        cmd1 = [
            self.ffmpeg, "-y",
            "-err_detect", "ignore_err",
            "-i", str(video_path),
            "-vn", "-c:a", "copy",
            "-f", "adts",
            "-loglevel", "error",
            str(aac_path),
        ]
        result1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=300)
        if not is_file_valid(aac_path, min_bytes=1024):
            raise AudioExtractionError(
                f"Raw AAC extraction produced no usable output for {video_path.name}"
            )

        # Step 2: decode raw AAC → WAV with error tolerance
        logger.info(f"Step 2/2: decoding raw AAC → WAV")
        cmd2 = [
            self.ffmpeg, "-y",
            "-threads", "1",
            "-err_detect", "ignore_err",
            "-fflags", "+genpts",
            "-i", str(aac_path),
            "-c:a", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-af", "aresample=async=1",
            "-loglevel", "error",
            str(wav_path),
        ]
        result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=600)

        # Clean up intermediate AAC
        try:
            aac_path.unlink()
        except OSError:
            pass

        if is_file_valid(wav_path, min_bytes=1024):
            dur = self.get_duration(wav_path)
            logger.info(f"Raw-AAC fallback produced {wav_path.stat().st_size} bytes ({dur:.0f}s)")
            return wav_path

        stderr = result2.stderr.strip()
        raise AudioExtractionError(
            f"Raw-AAC decode also failed for {video_path.name}: {stderr}"
        )

    # ------------------------------------------------------------------
    # Audio chunking (for parallel transcription of long videos)
    # ------------------------------------------------------------------

    def split_wav(
        self, wav_path: Path, chunk_duration: int, *, progress_callback=None,
        cancel_event=None,
    ) -> list[tuple[float, Path]]:
        """
        Split a WAV file into fixed-duration chunks using ffmpeg segment muxer.

        Args:
            wav_path: Path to the full 16kHz mono WAV.
            chunk_duration: Max seconds per chunk.
            progress_callback: Optional ``(fraction: float) -> None`` callback.
            cancel_event: Optional ``threading.Event`` checked while ffmpeg runs.

        Returns:
            List of (offset_seconds, chunk_wav_path) sorted by offset.
            Returns [(0.0, wav_path)] if the audio is shorter than chunk_duration.

        Raises:
            AudioExtractionCancelled: the caller requested cancellation.
        """
        duration = self.get_duration(wav_path)
        if cancel_event is not None and cancel_event.is_set():
            raise AudioExtractionCancelled("Cancelled before splitting")

        chunk_dir = wav_path.parent / f"{wav_path.stem}_chunks"
        chunk_prefix = f"{wav_path.stem}_{chunk_duration}s_"
        # Clear the namespace first, before any early return: this directory is
        # owned by this WAV alone, and anything in it comes from an earlier
        # split (another chunk_duration, or a run cut short). Left behind, those
        # files would be globbed up as if they were this run's output — and a
        # "too short to split" result still has to drop them, or the next call
        # finds them again.
        self._clear_stale_chunks(chunk_dir, wav_path.stem)

        if duration <= chunk_duration:
            logger.info(f"Audio {wav_path.name} ({duration:.0f}s) within chunk limit, no split")
            if progress_callback:
                progress_callback(1.0)
            return [(0.0, wav_path)]

        chunk_dir.mkdir(parents=True, exist_ok=True)
        pattern = str(chunk_dir / f"{chunk_prefix}%03d.wav")

        logger.info(f"Splitting {wav_path.name} ({duration:.0f}s) into {chunk_duration}s chunks")
        cmd = [
            self.ffmpeg, "-y",
            "-i", str(wav_path),
            "-f", "segment",
            "-segment_time", str(chunk_duration),
            "-c", "copy",
            "-loglevel", "error",
            pattern,
        ]
        result = self._run_ffmpeg(
            cmd,
            timeout=120,
            total_seconds=duration,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        if result.returncode != 0:
            raise AudioExtractionError(
                f"ffmpeg segment failed for {wav_path.name}: {result.stderr.strip()}"
            )

        # Collect chunks sorted by name (which encodes order). The prefix pins
        # the chunk length, so a stale split at another duration cannot leak in.
        chunk_files = sorted(chunk_dir.glob(f"{chunk_prefix}*.wav"))
        chunks: list[tuple[float, Path]] = []
        for i, cf in enumerate(chunk_files):
            offset = i * chunk_duration
            chunks.append((offset, cf))
            logger.debug(f"  Chunk {i}: offset={offset}s, file={cf.name}")

        logger.info(f"Split into {len(chunks)} chunk(s)")
        if progress_callback:
            progress_callback(1.0)
        return chunks

    @staticmethod
    def _clear_stale_chunks(chunk_dir: Path, stem: str) -> int:
        """Delete leftover shards for *stem*. Returns how many were removed."""
        if not chunk_dir.is_dir():
            return 0
        removed = 0
        for stale in chunk_dir.glob(f"{stem}_*.wav"):
            try:
                stale.unlink()
                removed += 1
            except OSError as exc:
                logger.warning("Could not remove stale chunk %s: %s", stale.name, exc)
        if removed:
            logger.info("Cleared %d stale chunk(s) from %s", removed, chunk_dir.name)
        return removed

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def extract_batch(
        self,
        video_paths: list[Path],
        output_dir: Path | None = None,
    ) -> list[Path]:
        """
        Batch extract audio from multiple video files.
        Individual failures are logged but do not stop the batch.

        Returns:
            List of successfully extracted WAV paths (in the same order as input,
            with failed entries omitted).
        """
        wav_paths: list[Path] = []
        for vp in video_paths:
            try:
                wav = self.extract(vp, output_dir)
                wav_paths.append(wav)
            except AudioExtractionError as exc:
                logger.error(str(exc))
        return wav_paths
