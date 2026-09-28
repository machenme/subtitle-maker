@echo off
rem Build the double-clickable ASR-Pipeline.exe launcher (no console window).
rem Requires: .venv\Scripts\python.exe with pyinstaller installed (uv sync --group dev)
pushd "%~dp0.."

.venv\Scripts\python.exe -m PyInstaller ^
    --noconfirm --clean --onefile --windowed --noupx ^
    --name ASR-Pipeline ^
    --distpath dist ^
    --workpath build ^
    --specpath launcher ^
    launcher\asr_launcher.py

if errorlevel 1 goto :fail

if not exist dist\ASR-Pipeline.exe goto :fail
copy /Y dist\ASR-Pipeline.exe ASR-Pipeline.exe >nul
echo.
echo Built: %CD%\ASR-Pipeline.exe
popd
exit /b 0

:fail
echo.
echo Build failed. Check the output above.
popd
exit /b 1
