@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\spyder.exe" (
    echo Spyder is missing. Run setup.ps1 first.
    pause
    exit /b 1
)
set "QT_API=pyqt5"
set "QT_PLUGIN_PATH="
set "QT_QPA_PLATFORM_PLUGIN_PATH="
"%~dp0.venv\Scripts\spyder.exe" --new-instance --conf-dir "%~dp0.spyder-cv" --workdir "%~dp0."
if errorlevel 1 pause
