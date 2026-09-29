@echo off
rem Build the double-clickable Subtitle-Maker.exe launcher (no console window).
rem Requires: .venv\Scripts\python.exe with pyinstaller installed (uv sync --group dev)
pushd "%~dp0.."

.venv\Scripts\python.exe -m PyInstaller ^
    --noconfirm --clean --onefile --windowed --noupx ^
    --name Subtitle-Maker ^
    --distpath dist ^
    --workpath build ^
    --specpath launcher ^
    launcher\subtitle_maker_launcher.py

if errorlevel 1 goto :fail

if not exist dist\Subtitle-Maker.exe goto :fail
copy /Y dist\Subtitle-Maker.exe Subtitle-Maker.exe >nul
echo.
echo Built: %CD%\Subtitle-Maker.exe
popd
exit /b 0

:fail
echo.
echo Build failed. Check the output above.
popd
exit /b 1
