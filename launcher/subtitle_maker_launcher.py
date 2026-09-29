"""Headless launcher for the Subtitle Maker GUI.

Double-clickable replacement for 一键启动.bat: it starts the PySide6 GUI out
of the project virtual environment and never flashes a console window.

Everything stays in the source tree (no bundled interpreter), so a dependency
upgrade needs no rebuild — only this launcher would need rebuilding if the
interpreter lookup below ever changes.

Debug: stdout/stderr of the GUI are appended to <project>/logs/gui.log. Set
SUBTITLE_MAKER_NOWAIT=1 to skip the 2s startup sanity check.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CREATE_NO_WINDOW = 0x08000000
LOG_NAME = Path("logs") / "gui.log"
STARTUP_CHECK_SECONDS = 2.0
TAIL_LINES = 30


def project_root() -> Path:
    """Directory holding src/gui.py — the folder the .exe sits in."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def find_interpreter(root: Path) -> list[str] | None:
    """pythonw first: it is a GUI-subsystem binary, so no console can appear."""
    candidates = [
        [str(root / ".venv" / "Scripts" / "pythonw.exe")],
        [str(root / ".venv" / "Scripts" / "python.exe")],
    ]
    pythonw = shutil.which("pythonw")
    if pythonw:
        candidates.append([pythonw])
    if shutil.which("uv"):
        candidates.append(["uv", "run", "pythonw"])
    for candidate in candidates:
        if candidate[0].startswith("uv") or Path(candidate[0]).is_file():
            return candidate
    return None


def show_error(message: str) -> None:
    ctypes.windll.user32.MessageBoxW(0, message, "Subtitle Maker 启动失败", 0x10)


def read_tail(path: Path, lines: int) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines(True)
    except OSError:
        return ""
    return "".join(content[-lines:]).strip() or "(日志为空)"


def nowait_requested() -> bool:
    # os.environ rather than sys.environ: sys.environ is unavailable inside the
    # frozen bundle produced by PyInstaller --windowed.
    return os.environ.get("SUBTITLE_MAKER_NOWAIT") == "1"


def main() -> int:
    root = project_root()
    interpreter = find_interpreter(root)
    if interpreter is None:
        show_error(
            f"没有找到 Python 解释器，也没有找到 uv。\n\n"
            f"请先在 {root} 里执行一次：uv sync"
        )
        return 1

    log_path = root / LOG_NAME
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 启动 GUI =====\n")
        log.write(f"cwd: {root}\ninterpreter: {' '.join(interpreter)}\n")
        log.flush()
        try:
            process = subprocess.Popen(
                [*interpreter, "-m", "src.gui"],
                cwd=str(root),
                stdout=log,
                stderr=log,
                stdin=subprocess.DEVNULL,
                creationflags=CREATE_NO_WINDOW,
            )
        except OSError as exc:
            show_error(f"启动失败：{exc}\n\n日志：{log_path}")
            return 1

    if not nowait_requested():
        deadline = time.monotonic() + STARTUP_CHECK_SECONDS
        while time.monotonic() < deadline and process.poll() is None:
            time.sleep(0.1)
        code = process.poll()
        if code is not None and code != 0:
            show_error(
                f"GUI 启动后立即退出（退出码 {code}）。\n\n"
                f"最近日志：\n{read_tail(log_path, TAIL_LINES)}\n\n"
                f"完整日志：{log_path}"
            )
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
