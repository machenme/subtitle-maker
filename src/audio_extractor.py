"""
Stage 1: Audio extraction via ffmpeg subprocess.
Extracts audio streams from media files → 16kHz Mono 16-bit PCM WAV.
"""
from __future__ import annotations

import re
import subprocess
import logging
import hashlib
import threading
import time
from pathlib import Path

from src.utils import is_file_valid

logger = logging.getLogger(__name__)

# ffmpeg -progress emits key=value pairs; out_time_us is the encoded position.
_OUT_TIME_RE = re.compile(r"^out_time_us=(\d+)", re.MULTILINE)
# Emit at most this often while streaming ffmpeg progress.
_PROGRESS_INTERVAL = 0.2


class AudioExtractionError(Exception):
    """Raised when ffmpeg fails to extract audio from a media file."""


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

    def __init__(self, temp_dir: Path, ffmpeg_bin: str = "ffmpeg"):
        self._temp_dir = Path(temp_dir)
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._ffmpeg = ffmpeg_bin

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
    ) -> _FFmpegResult:
        """Run ffmpeg, optionally streaming 0..1 progress to a callback.

        Without a callback this stays a plain ``subprocess.run``. With one,
        ``-progress pipe:1`` is added and the stream is parsed as it arrives:
        extraction is often the longest single step, and a frozen bar there
        reads as a hang.
        """
        if progress_callback is None:
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
        # ffmpeg's first status block can already be near the end on a short
        # clip, so pin an explicit 0% to guarantee a visible starting point.
        progress_callback(0.0)
        stderr_lines: list[str] = []
        # ffmpeg blocks on a full stderr pipe, so it must be drained while we
        # read progress off stdout.
        reader = threading.Thread(
            target=lambda: stderr_lines.extend(proc.stderr.read().splitlines()),
            daemon=True,
        )
        reader.start()

        deadline = time.monotonic() + timeout
        seen: list[str] = []
        last_emit = 0.0
        timed_out = False
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
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
                    progress_callback(
                        min(1.0, encoded / total_seconds) if total_seconds > 0 else 0.0
                    )
                if now > deadline:
                    timed_out = True
                    proc.kill()
                    break
            proc.wait(timeout=5)
        finally:
            proc.stdout.close()
            reader.join(timeout=2)
            proc.stderr.close()

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
    ) -> Path:
        """
        Extract audio from a single video file.

        Args:
            video_path: Source video file path.
            output_dir: Directory for the output WAV (defaults to self._temp_dir).
            progress_callback: Optional ``(fraction: float) -> None`` invoked
                while ffmpeg runs. The source duration is probed only when a
                callback is supplied, so the plain path pays nothing for it.

        Returns:
            Path to the extracted WAV file.

        Raises:
            AudioExtractionError: ffmpeg call failed.
        """
        dest_dir = output_dir or self._temp_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        try:
            stat = video_path.stat()
        except OSError as exc:
            raise AudioExtractionError(f"Cannot read source media: {video_path}") from exc
        source_key = (
            f"{video_path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}"
        ).encode("utf-8")
        source_hash = hashlib.sha1(source_key).hexdigest()[:10]
        wav_name = f"{video_path.stem}.{source_hash}.wav"
        wav_path = dest_dir / wav_name

        # Skip if already extracted and valid
        if is_file_valid(wav_path, min_bytes=1024):
            logger.info(f"Audio already extracted: {wav_path}")
            if progress_callback:
                progress_callback(1.0)
            return wav_path

        logger.info(f"Extracting audio: {video_path.name} → {wav_name}")

        cmd = [
            self._ffmpeg,
            "-y",                          # overwrite
            "-threads", "1",               # single-threaded: better error resilience
            "-err_detect", "ignore_err",   # tolerate corrupt packets
            "-fflags", "+genpts+discardcorrupt",  # regenerate timestamps, skip broken frames
            "-i", str(video_path),
            "-vn",                         # no video
            "-c:a", "pcm_s16le",           # 16-bit PCM (re-encode, not stream copy)
            "-ar", "16000",                # 16 kHz
            "-ac", "1",                    # mono
            "-af", "aresample=async=1",    # fill gaps from corrupt frames with silence
            "-loglevel", "error",          # suppress ffmpeg output except errors
            str(wav_path),
        ]

        total_seconds = self.get_video_duration(video_path) if progress_callback else 0.0

        try:
            result = self._run_ffmpeg(
                cmd,
                timeout=600,
                total_seconds=total_seconds,
                progress_callback=progress_callback,
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
        except FileNotFoundError:
            raise AudioExtractionError(
                "ffmpeg not found. Please install ffmpeg and ensure it's on PATH."
            )

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
            "ffprobe", "-v", "error",
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
            self._ffmpeg, "-y",
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
            self._ffmpeg, "-y",
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
        self, wav_path: Path, chunk_duration: int, *, progress_callback=None
    ) -> list[tuple[float, Path]]:
        """
        Split a WAV file into fixed-duration chunks using ffmpeg segment muxer.

        Args:
            wav_path: Path to the full 16kHz mono WAV.
            chunk_duration: Max seconds per chunk.
            progress_callback: Optional ``(fraction: float) -> None`` callback.

        Returns:
            List of (offset_seconds, chunk_wav_path) sorted by offset.
            Returns [(0.0, wav_path)] if the audio is shorter than chunk_duration.
        """
        duration = self.get_duration(wav_path)
        if duration <= chunk_duration:
            logger.info(f"Audio {wav_path.name} ({duration:.0f}s) within chunk limit, no split")
            if progress_callback:
                progress_callback(1.0)
            return [(0.0, wav_path)]

        chunk_dir = wav_path.parent / f"{wav_path.stem}_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        pattern = str(chunk_dir / f"{wav_path.stem}_%03d.wav")

        logger.info(f"Splitting {wav_path.name} ({duration:.0f}s) into {chunk_duration}s chunks")
        cmd = [
            self._ffmpeg, "-y",
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
        )
        if result.returncode != 0:
            raise AudioExtractionError(
                f"ffmpeg segment failed for {wav_path.name}: {result.stderr.strip()}"
            )

        # Collect chunks sorted by name (which encodes order)
        chunk_files = sorted(chunk_dir.glob(f"{wav_path.stem}_*.wav"))
        chunks: list[tuple[float, Path]] = []
        for i, cf in enumerate(chunk_files):
            offset = i * chunk_duration
            chunks.append((offset, cf))
            logger.debug(f"  Chunk {i}: offset={offset}s, file={cf.name}")

        logger.info(f"Split into {len(chunks)} chunk(s)")
        if progress_callback:
            progress_callback(1.0)
        return chunks

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
