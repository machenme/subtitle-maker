@echo off
REM SRT -> TXT / Markdown 转换窗口（双击运行）
cd /d "%~dp0"
uv run python -m src.srt_convert_gui
if errorlevel 1 pause
