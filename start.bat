@echo off
setlocal
cd /d "%~dp0"

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Transformer Learning Studio failed to start. Exit code: %EXIT_CODE%
    echo See "%~dp0startup.log" for details.
    pause
)

exit /b %EXIT_CODE%
